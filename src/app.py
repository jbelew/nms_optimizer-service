# --- Gevent monkey-patching ---
# IMPORTANT: This must be the very first import and execution in the app
from gevent import monkey

monkey.patch_all()
# --- End Gevent monkey-patching ---

import os
import time
import uuid
from typing import cast

import gevent

from flask import Flask, jsonify, request
from flask_compress import Compress
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Filter,
    FilterExpression,
    Metric,
    OrderBy,
    RunReportRequest,
)
from google.oauth2 import service_account

from .data_definitions.grids import grids
from .data_definitions.recommended_builds import recommended_builds
from .data_loader import get_all_module_data, get_module_data
from .grid_utils import Grid
from .modules_utils import get_tech_tree_json
from .optimization import optimize_placement
from .analytics import send_analytics_event
from dotenv import load_dotenv

load_dotenv()


# app.py
# Configure logging as the first step after monkey-patching


# Define a custom Request type for Flask-SocketIO to include 'sid'
class SocketIORequest(request.__class__):
    sid: str


app = Flask(__name__)
# Define allowed origins
# Default to production and common local dev ports
DEFAULT_ORIGINS = [
    "https://nms-optimizer.app",
    "http://localhost:5173",  # Vite dev
    "http://localhost:4173",  # Vite preview
    "http://localhost:3000",
]
allowed_origins = os.environ.get("ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",")

CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=True)
Compress(app)  # Initialize Flask-Compress
socketio = SocketIO(app, cors_allowed_origins="*")


@app.after_request
def add_cache_headers(response):
    if (
        request.path.startswith("/tech_tree/")
        or request.path == "/platforms"
        or request.path == "/analytics/popular_data"
    ):
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response


# --- Google Analytics 4 (GA4) Configuration ---
# IMPORTANT: For production, store this path securely (e.g., environment variable)
# and ensure the JSON key file is NOT committed to version control.
GA_PROPERTY_ID = "484727815"  # Your GA4 Property ID


def initialize_ga4_client():
    """Initializes the Google Analytics Data API V1Beta client.

    It attempts to load credentials first from the environment variable
    `GOOGLE_APPLICATION_CREDENTIALS_JSON` (for services like Heroku) and
    falls back to a local JSON key file for development.

    Returns:
        BetaAnalyticsDataClient: An initialized GA4 client instance, or None
        if initialization fails.
    """
    try:
        # Try to get credentials from environment variable first (Heroku)
        gcp_key_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if gcp_key_json:
            import json

            credentials_info = json.loads(gcp_key_json)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=["https://www.googleapis.com/auth/analytics.readonly"],
            )
        else:
            # Fallback to file-based credentials (local development)
            GA_KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), "cosmic-inkwell-467922-v5-85707f3bcc80.json")
            credentials = service_account.Credentials.from_service_account_file(
                GA_KEY_FILE_PATH,
                scopes=["https://www.googleapis.com/auth/analytics.readonly"],
            )
        client = BetaAnalyticsDataClient(credentials=credentials)
        return client
    except Exception as e:
        app.logger.error(f"Error initializing Google Analytics Data API client: {e}")
        return None


# Initialize the client globally or on first request
ga4_client = initialize_ga4_client()
if not ga4_client:
    app.logger.error("Failed to initialize Google Analytics Data API client. Analytics endpoints may not function.")

# --- End GA4 Configuration ---


def run_optimization(data, progress_callback=None, run_id=None):
    """Executes the core optimization logic based on the provided data.

    This function serves as a central dispatcher for both the REST and
    WebSocket endpoints. It parses the incoming data, loads the necessary
    module and grid information, and calls the `optimize_placement` function.

    Args:
        data (dict): The input data from the request, containing ship, tech,
            grid, and other optimization parameters.
        progress_callback (callable, optional): A function to call with
            progress updates, used by the WebSocket handler.
        run_id (str, optional): A unique identifier for the optimization run.

    Returns:
        tuple: A tuple containing a dictionary with the optimization result
               and an HTTP status code.
    """
    ship = data.get("ship")
    tech = data.get("tech")
    available_modules = data.get("available_modules")
    forced_solve = data.get("forced", False)
    send_grid_updates = data.get("send_grid_updates", False)

    if tech is None:
        return {"error": "No tech specified"}, 400

    grid_data = data.get("grid")
    if grid_data is None:
        return {"error": "No grid specified"}, 400

    grid = Grid.from_dict(grid_data)

    # --- Load module data on-demand ---
    module_data = get_module_data(ship)
    if not module_data:
        return {"error": f"Invalid or unsupported ship type: {ship}"}, 400
    # ---

    try:
        optimized_grid, percentage, solved_bonus, solve_method = optimize_placement(
            grid,
            ship,
            module_data,
            tech,
            forced=forced_solve,
            progress_callback=progress_callback,
            run_id=run_id,
            send_grid_updates=send_grid_updates,
            available_modules=available_modules,
        )

        result = {
            "grid": optimized_grid.to_dict() if optimized_grid else None,
            "max_bonus": percentage,
            "solved_bonus": solved_bonus,
            "solve_method": solve_method,
            "run_id": run_id,
        }

        if solve_method == "Pattern No Fit":
            result["message"] = (
                "Official solve map exists, but no pattern variation fits the current grid. User can choose to force a Simulated Annealing solve."
            )
            return result, 200

        return result, 200

    except ValueError as e:
        app.logger.error(f"ValueError during optimization: {str(e)}")
        return {"error": str(e), "run_id": run_id}, 500


@app.route("/optimize", methods=["POST"])
def optimize_grid():
    """Handles HTTP POST requests for grid optimization.

    This is the main REST endpoint for running an optimization. It expects a
    JSON payload with the grid configuration and other parameters.

    Request JSON Body:
        ship (str): The ship type (e.g., "hauler").
        tech (str): The technology to optimize (e.g., "pulse").
        grid (dict): The grid object from the client.
        ... and other parameters for `run_optimization`.

    Returns:
        JSON: A JSON response containing the optimized grid and bonus stats,
              or an error message.
    """
    data = request.get_json()
    result, status_code = run_optimization(data)
    return jsonify(result), status_code


@socketio.on("optimize")
def handle_optimize_socket(data):
    """Handles WebSocket connections for real-time grid optimization.

    This endpoint allows for a long-running optimization process that can
    send progress updates back to the client.

    Args:
        data (dict): The data sent with the socket event, containing the
                     same parameters as the REST endpoint.

    Emits:
        'progress': Sent periodically with updates on the optimization process.
        'optimization_result': Sent when the optimization is complete.
    """
    req = cast(SocketIORequest, request)
    sid = req.sid
    run_id = str(uuid.uuid4())
    last_emit_time = 0
    THROTTLE_INTERVAL = 0.10  # seconds

    def progress_callback(progress_data):
        """Callback to emit progress over the socket, with throttling."""
        nonlocal last_emit_time
        current_time = time.time()

        # An important event is one with a grid update, or any status
        # update that is NOT the generic "in_progress".
        is_important_event = "best_grid" in progress_data or (
            "status" in progress_data and progress_data["status"] != "in_progress"
        )

        if is_important_event:
            emit("progress", {**progress_data, "run_id": run_id}, room=sid)  # type: ignore
            last_emit_time = current_time  # Reset throttle for subsequent messages
            gevent.sleep(0)  # Yield to allow other greenlets to run
        elif current_time - last_emit_time > THROTTLE_INTERVAL:
            emit("progress", {**progress_data, "run_id": run_id}, room=sid)  # type: ignore
            last_emit_time = current_time
            gevent.sleep(0)  # Yield to allow other greenlets to run

    result, status_code = run_optimization(data, progress_callback=progress_callback, run_id=run_id)
    emit("optimization_result", {**result, "run_id": run_id}, room=sid)  # type: ignore


@app.route("/tech_tree/<ship_name>")
def get_technology_tree(ship_name):
    """Retrieves the technology tree for a given ship.

    This endpoint returns a structured JSON object containing all technologies
    available for a specific ship, along with recommended builds and the
    grid definition.

    Args:
        ship_name (str): The name of the ship (e.g., "hauler") from the URL.

    Returns:
        JSON: A JSON response with the technology tree, recommended builds,
              and grid layout. Returns a 404 error if the ship is not found.
    """
    try:
        # --- Load module data on-demand ---
        module_data = get_module_data(ship_name)
        if not module_data:
            return jsonify({"error": f"Invalid or unsupported ship type: {ship_name}"}), 404
        # ---
        tree_data = get_tech_tree_json(ship_name, module_data)

        # Check if a recommended build exists for the current ship_name
        recommended_builds_list = recommended_builds.get(ship_name)

        # If tree_data is a JSON string, parse it to a dict, add recommended_build, then re-serialize
        # Otherwise, assume it's already a dict and add directly
        import json

        if isinstance(tree_data, str):
            tree_dict = json.loads(tree_data)
        else:
            tree_dict = tree_data

        if recommended_builds_list:
            # Now expecting a list of builds, so name the key accordingly
            tree_dict["recommended_builds"] = recommended_builds_list

        # Check if a grid definition exists for the current ship_name
        grid_definition = grids.get(ship_name)
        if grid_definition:
            tree_dict["grid_definition"] = grid_definition

        return jsonify(tree_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/platforms", methods=["GET"])
def get_ship_types():
    """Retrieves a list of all available ship types (platforms).

    The returned data includes the key, label, and type for each ship,
    which is used by the frontend to populate selection menus.

    Returns:
        JSON: A dictionary where keys are ship identifiers and values are
              objects containing the ship's label and type.
    """
    # --- Load all module data for this endpoint ---
    all_modules = get_all_module_data()
    # ---
    ship_types = {}
    for ship_key, ship_data in all_modules.items():
        # Create a dictionary containing both label and type
        ship_info = {
            "label": ship_data.get("label"),
            "type": ship_data.get("type"),
        }  # Get the 'type' field
        ship_types[ship_key] = ship_info
    return jsonify(ship_types)


@app.route("/analytics/popular_data", methods=["GET"])
def get_popular_analytics_data():
    """Fetches and returns popular optimization data from Google Analytics.

    This endpoint requires the GA4 Data API client to be initialized.
    It queries for the most frequent "optimize_tech" events and returns
    the count for different combinations of ship, technology, and
    supercharged status.

    Query Parameters:
        start_date (str, optional): The start date for the report, in
            "YYYY-MM-DD" or "NdaysAgo" format. Defaults to "30daysAgo".
        end_date (str, optional): The end date for the report. Defaults to
            "today".

    Returns:
        JSON: A list of dictionaries, each containing the ship type,
              technology, supercharged status, and the total event count.
    """
    if not ga4_client:
        return jsonify({"error": "Google Analytics Data API client not initialized."}), 500

    try:
        # Define the date range for the report
        start_date = request.args.get("start_date", "30daysAgo")
        end_date = request.args.get("end_date", "today")

        # Define the report request for GA4
        # This example assumes you are tracking 'optimize' events and have custom dimensions
        # for 'ship_type' and 'technology'. Adjust dimension names as per your GA4 setup.
        request_body = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            dimensions=[
                Dimension(name="eventName"),
                Dimension(name="customEvent:platform"),  # Your actual custom dimension name for Platform
                Dimension(name="customEvent:tech"),  # Your actual custom dimension name for Tech
                Dimension(name="customEvent:supercharged"),  # New custom dimension for supercharged
            ],
            dimension_filter=FilterExpression(
                filter=Filter(
                    field_name="eventName",
                    string_filter=Filter.StringFilter(value="optimize_tech"),
                )
            ),
            metrics=[Metric(name="eventCount")],
            order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="eventCount"), desc=True)],
        )

        response = ga4_client.run_report(request_body)

        popular_data = []
        for row in response.rows:
            popular_data.append(
                {
                    "event_name": row.dimension_values[0].value,
                    "ship_type": row.dimension_values[1].value,
                    "technology": row.dimension_values[2].value,
                    "supercharged": row.dimension_values[3].value,
                    "total_events": int(row.metric_values[0].value),
                }
            )

        return jsonify(popular_data)

    except Exception as e:
        app.logger.error(f"Error fetching Google Analytics data: {e}")
        return jsonify({"error": f"Failed to fetch analytics data: {e}"}), 500


# --- Analytics Event Tracking Endpoint ---
@app.route("/api/events", methods=["POST"])
def track_event():
    """Receive and relay analytics events to GA4.

    Expected JSON body:
    {
            "clientId": "string",
            "userId": "string (optional)",
            "eventName": "string",
            "params": { "key": "value", ... }
    }

    Returns:
            200 OK on success
            400 Bad Request if required fields missing
            500 Server Error on GA4 send failure
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        client_id = data.get("clientId")
        event_name = data.get("eventName")
        params = data.get("params", {})
        user_id = data.get("userId")

        # Validate required fields
        if not client_id or not event_name:
            return jsonify({"error": "Missing required fields: clientId, eventName"}), 400

        # Send event to GA4
        success = send_analytics_event(
            event_name=event_name,
            client_id=client_id,
            params=params,
            user_id=user_id,
        )

        if success:
            return jsonify({"status": "ok"}), 200
        else:
            return jsonify({"error": "Failed to send event to GA4"}), 500

    except Exception as e:
        app.logger.error(f"Error in analytics endpoint: {e}")
        return jsonify({"error": str(e)}), 500


# Start the message sending thread
if __name__ == "__main__":
    socketio.run(app, debug=True)
