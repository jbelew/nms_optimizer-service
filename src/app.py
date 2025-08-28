# --- Gevent monkey-patching ---
# IMPORTANT: This must be the very first import and execution in the app
from gevent import monkey
monkey.patch_all()
# --- End Gevent monkey-patching ---

# app.py
# Configure logging as the first step after monkey-patching
import logger
import logging

from flask import Flask, jsonify, request
from flask_compress import Compress
from flask_cors import CORS
from flask_socketio import SocketIO, emit


from optimization_algorithms import optimize_placement
from optimizer import get_tech_tree_json, Grid
from data_loader import get_module_data, get_all_module_data
from data_definitions.recommended_builds import recommended_builds
from data_definitions.grids import grids
import os
import uuid
import time
from google.oauth2 import service_account
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
    OrderBy,
    FilterExpression,
    Filter,
)
from typing import cast


# Define a custom Request type for Flask-SocketIO to include 'sid'
class SocketIORequest(request.__class__):
    sid: str


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
Compress(app)  # Initialize Flask-Compress
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Google Analytics 4 (GA4) Configuration ---
# IMPORTANT: For production, store this path securely (e.g., environment variable)
# and ensure the JSON key file is NOT committed to version control.
GA_PROPERTY_ID = "484727815"  # Your GA4 Property ID


def initialize_ga4_client():
    """Initializes the Google Analytics Data API V1Beta client."""
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
            GA_KEY_FILE_PATH = os.path.join(
                os.path.dirname(__file__), "cosmic-inkwell-467922-v5-85707f3bcc80.json"
            )
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
    app.logger.error(
        "Failed to initialize Google Analytics Data API client. Analytics endpoints may not function."
    )

# --- End GA4 Configuration ---


def run_optimization(data, progress_callback=None, run_id=None):
    """Central function to run the optimization logic."""
    ship = data.get("ship")
    tech = data.get("tech")
    player_owned_rewards = data.get("player_owned_rewards")
    forced_solve = data.get("forced", False)
    experimental_window_sizing_req = data.get("experimental_window_sizing", True)
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
            player_owned_rewards,
            forced=forced_solve,
            experimental_window_sizing=experimental_window_sizing_req,
            progress_callback=progress_callback,
            run_id=run_id,
            send_grid_updates=send_grid_updates,
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
    """Endpoint to optimize the grid."""
    data = request.get_json()
    result, status_code = run_optimization(data)
    return jsonify(result), status_code


@socketio.on("optimize")
def handle_optimize_socket(data):
    """WebSocket endpoint to optimize the grid."""
    req = cast(SocketIORequest, request)
    sid = req.sid
    run_id = str(uuid.uuid4())
    last_emit_time = 0
    THROTTLE_INTERVAL = 0.1  # seconds

    def progress_callback(progress_data):
        """Callback to emit progress over the socket, with throttling."""
        nonlocal last_emit_time
        current_time = time.time()
        if "best_grid" in progress_data:  # Guarantee send_grid_update messages
            emit("progress", {**progress_data, "run_id": run_id}, room=sid)  # type: ignore
            last_emit_time = current_time  # Reset throttle for subsequent messages
        elif current_time - last_emit_time > THROTTLE_INTERVAL:
            emit("progress", {**progress_data, "run_id": run_id}, room=sid)  # type: ignore
            last_emit_time = current_time

    result, status_code = run_optimization(
        data, progress_callback=progress_callback, run_id=run_id
    )
    emit("optimization_result", {**result, "run_id": run_id}, room=sid)  # type: ignore


@app.route("/tech_tree/<ship_name>")
def get_technology_tree(ship_name):
    """Endpoint to get the technology tree for a given ship."""
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
    """Endpoint to get the available ship types, their labels, and their types."""
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
    """Endpoint to pull Google Analytics data for popular technologies and ship types.
    Requires GA_PROPERTY_ID to be configured.
    """
    if not ga4_client:
        return jsonify(
            {"error": "Google Analytics Data API client not initialized."}
        ), 500

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
                Dimension(
                    name="customEvent:platform"
                ),  # Your actual custom dimension name for Platform
                Dimension(
                    name="customEvent:tech"
                ),  # Your actual custom dimension name for Tech
                Dimension(
                    name="customEvent:supercharged"
                ),  # New custom dimension for supercharged
            ],
            dimension_filter=FilterExpression(
                filter=Filter(
                    field_name="eventName",
                    string_filter=Filter.StringFilter(value="optimize_tech"),
                )
            ),
            metrics=[Metric(name="eventCount")],
            order_bys=[
                OrderBy(
                    metric=OrderBy.MetricOrderBy(metric_name="eventCount"), desc=True
                )
            ],
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


# Start the message sending thread
if __name__ == "__main__":
    socketio.run(app, debug=True)
