# app.py
from flask import Flask, jsonify, request
from flask_compress import Compress
from flask_cors import CORS
from optimization_algorithms import optimize_placement
from optimizer import get_tech_tree_json, Grid
from data_definitions.modules import modules  # Keep modules as it's used directly
from data_definitions.recommended_builds import recommended_builds
from data_definitions.grids import grids
import logging
import os
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

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
Compress(app)  # Initialize Flask-Compress

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
    app.logger.error(
        "Failed to initialize Google Analytics Data API client. Analytics endpoints may not function."
    )

# --- End GA4 Configuration ---


@app.route("/optimize", methods=["POST"])
def optimize_grid():
    """Endpoint to optimize the grid."""
    data = request.get_json()
    ship = data.get("ship")
    tech = data.get("tech")
    player_owned_rewards = data.get("player_owned_rewards")
    forced_solve = data.get("forced", False)
    experimental_window_sizing_req = data.get("experimental_window_sizing", True)

    if tech is None:
        return jsonify({"error": "No tech specified"}), 400

    grid_data = data.get("grid")
    if grid_data is None:
        return jsonify({"error": "No grid specified"}), 400

    grid = Grid.from_dict(grid_data)

    try:
        # Pass the forced_solve flag to optimize_placement
        optimized_grid, percentage, solved_bonus, solve_method = optimize_placement(
            grid,
            ship,
            modules,
            tech,
            player_owned_rewards,
            forced=forced_solve,
            experimental_window_sizing=experimental_window_sizing_req,
        )

        if solve_method == "Pattern No Fit":
            return (
                jsonify(
                    {
                        "grid": None,  # No grid to return in this specific case
                        "max_bonus": 0.0,
                        "solved_bonus": 0.0,
                        "solve_method": "Pattern No Fit",
                        "message": "Official solve map exists, but no pattern variation fits the current grid. User can choose to force a Simulated Annealing solve.",
                    }
                ),
                200,
            )  # 200 OK, but with a specific message for the UI
        return jsonify(
            {
                "grid": optimized_grid.to_dict(),
                "max_bonus": percentage,
                "solved_bonus": solved_bonus,
                "solve_method": solve_method,
            }
        )
    except ValueError as e:
        app.logger.error(f"ValueError during optimization: {str(e)}")
        # Consider if printing the grid here is necessary or too verbose for production logs
        return jsonify({"error": str(e)}), 500


@app.route("/tech_tree/<ship_name>")
def get_technology_tree(ship_name):
    """Endpoint to get the technology tree for a given ship."""
    try:
        tree_data = get_tech_tree_json(ship_name)

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
    ship_types = {}
    for ship_key, ship_data in modules.items():
        # Create a dictionary containing both label and type
        ship_info = {
            "label": ship_data.get("label"),
            "type": ship_data.get("type"),
        }  # Get the 'type' field
        ship_types[ship_key] = ship_info
    return jsonify(ship_types)


@app.route("/analytics/popular_data", methods=["GET"])
def get_popular_analytics_data():
    """
    Endpoint to pull Google Analytics data for popular technologies and ship types.
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
    app.run(debug=True)
