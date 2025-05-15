# app.py
from flask import Flask, jsonify, request
from flask_compress import Compress  # Import Compress
from flask_cors import CORS
from optimization_algorithms import optimize_placement  # Import directly from optimization_algorithms
from optimizer import get_tech_tree_json, Grid  # Keep these imports from optimizer
from modules import modules
from grid_display import print_grid_compact
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
Compress(app)  # Initialize Flask-Compress


@app.route("/optimize", methods=["POST"])
def optimize_grid():
    """Endpoint to optimize the grid."""
    data = request.get_json()
    ship = data.get("ship")
    tech = data.get("tech")
    player_owned_rewards = data.get("player_owned_rewards")
    forced_solve = data.get("forced", False)  # Get the 'forced' flag, default to False
    # print(f"Received request for ship: {ship}, tech: {tech}, player_owned_rewards: {player_owned_rewards}")

    if tech is None:
        return jsonify({"error": "No tech specified"}), 400

    grid_data = data.get("grid")
    if grid_data is None:
        return jsonify({"error": "No grid specified"}), 400

    grid = Grid.from_dict(grid_data)

    try:
        # Pass the forced_solve flag to optimize_placement
        optimized_grid, percentage, solved_bonus, solve_method = optimize_placement(
            grid, ship, modules, tech, player_owned_rewards, True, forced=forced_solve
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
        print(f"ERROR -- {str(e)}")
        print_grid_compact(grid)
        return jsonify({"error": str(e)}), 500


@app.route("/tech_tree/<ship_name>")
def get_technology_tree(ship_name):
    """Endpoint to get the technology tree for a given ship."""
    try:
        tree_data = get_tech_tree_json(ship_name)
        return tree_data
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/platforms", methods=["GET"])
def get_ship_types():
    """Endpoint to get the available ship types, their labels, and their types."""
    ship_types = {}
    for ship_key, ship_data in modules.items():
        # Create a dictionary containing both label and type
        ship_info = {"label": ship_data.get("label"), "type": ship_data.get("type")}  # Get the 'type' field
        ship_types[ship_key] = ship_info

    # print(f"Ship types: {ship_types}")
    return jsonify(ship_types)


# Start the message sending thread
if __name__ == "__main__":
    app.run(debug=True)
