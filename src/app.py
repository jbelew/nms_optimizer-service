# --- Gevent monkey-patching ---
# IMPORTANT: This must be the very first import and execution in the app
from gevent import monkey

monkey.patch_all()
# --- End Gevent monkey-patching ---

import os
import re
import time
import uuid
from typing import cast

from dotenv import load_dotenv

# Load environment variables before importing other local modules
load_dotenv()

import gevent

from flask import Flask, jsonify, request
from flask_compress import Compress
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from .data_definitions.grids import grids
from .data_definitions.recommended_builds import recommended_builds
from .data_loader import get_all_module_data, get_module_data
from .grid_utils import Grid
from .modules_utils import get_tech_tree_json
from .optimization import optimize_placement
from .routes.analytics import analytics_bp


# app.py
# Configure logging as the first step after monkey-patching


# Define a custom Request type for Flask-SocketIO to include 'sid'
class SocketIORequest(request.__class__):
    sid: str


app = Flask(__name__)

# --- CORS Configuration ---
# In development/testing, we want to be permissive but still support credentials.
# In production, we restrict to known origins.
DEFAULT_ORIGINS = [
    "https://nms-optimizer.app",
    re.compile(r"https?://localhost:\d+"),  # Matches any localhost port
    re.compile(r"https?://127\.0\.0\.1:\d+"),  # Matches any 127.0.0.1 port
]

# Allow overriding via environment variable
env_origins = os.environ.get("ALLOWED_ORIGINS")
if env_origins:
    allowed_origins = env_origins.split(",")
else:
    allowed_origins = DEFAULT_ORIGINS

# --- CORS Configuration ---
# We split the CORS policy:
# 1. /api/events requires credentials and a strict origin list.
# 2. Everything else is permissive to support Storybook, local dev, and tests.
CORS(
    app,
    resources={r"/api/events": {"origins": allowed_origins, "supports_credentials": True}, r"/*": {"origins": "*"}},
)
# --- End CORS Configuration ---
Compress(app)  # Initialize Flask-Compress
socketio = SocketIO(app, cors_allowed_origins="*")

# Register Blueprints
app.register_blueprint(analytics_bp)


@app.after_request
def add_cache_headers(response):
    if (
        request.path.startswith("/tech_tree/")
        or request.path == "/platforms"
        or request.path == "/analytics/popular_data"
        or request.path == "/analytics/performance_data"
    ):
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response


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


# Start the message sending thread
if __name__ == "__main__":
    socketio.run(app, debug=False)
