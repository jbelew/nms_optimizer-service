# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from optimization_algorithms import optimize_placement # Import directly from optimization_algorithms
from optimizer import get_tech_tree_json, Grid # Keep these imports from optimizer
from modules import modules
from grid_display import print_grid_compact
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/optimize", methods=["POST"])
def optimize_grid():
    """Endpoint to optimize the grid."""
    data = request.get_json()
    ship = data.get("ship")
    tech = data.get("tech")
    player_owned_rewards = data.get("player_owned_rewards")
    print(f"Received request for ship: {ship}, tech: {tech}, player_owned_rewards: {player_owned_rewards}")

    if tech is None:
        return jsonify({"error": "No tech specified"}), 400

    grid_data = data.get("grid")
    if grid_data is None:
        return jsonify({"error": "No grid specified"}), 400

    grid = Grid.from_dict(grid_data)

    try:
        grid, percentage, solved_bonus = optimize_placement(grid, ship, modules, tech, player_owned_rewards, True)
        return jsonify({"grid": grid.to_dict(), "max_bonus": percentage, "solved_bonus": solved_bonus})
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


@app.route("/ship_types", methods=["GET"])
def get_ship_types():
    """Endpoint to get the available ship types and their labels."""
    ship_types = {}
    for ship_key, ship_data in modules.items():
        ship_types[ship_key] = ship_data.get("label")

    print(f"Ship types: {ship_types}")
    return jsonify(ship_types)


# Start the message sending thread
if __name__ == "__main__":
    app.run(debug=True)
