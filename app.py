# app.py
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from optimizer import optimize_placement, get_tech_tree_json, Grid
from modules import modules
from grid_display import print_grid_compact, print_grid
import logging
import json
import time
import queue

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global queue for SSE messages
message_queue = queue.Queue()

def send_messages():
    """Sends messages from the queue to connected clients."""
    while True:
        message = message_queue.get()
        if message is None:
            break  # Signal to stop the thread
        yield message

@app.route('/stream')
def stream():
    """SSE endpoint for streaming messages."""
    return Response(send_messages(), mimetype='text/event-stream')

@app.route('/optimize', methods=['POST'])
def optimize_grid():
    """Endpoint to optimize the grid and send status updates via SSE."""
    data = request.get_json()
    ship = data.get("ship")
    tech = data.get('tech')
    player_owned_rewards = data.get('player_owned_rewards')
    print(f"Received request for ship: {ship}, tech: {tech}, player_owned_rewards: {player_owned_rewards}")

    if tech is None:
        return jsonify({'error': 'No tech specified'}), 400

    grid_data = data.get('grid')
    if grid_data is None:
        return jsonify({'error': 'No grid specified'}), 400

    grid = Grid.from_dict(grid_data)

    try:
        from sse_events import sse_message
        # Send a start message
        message_queue.put(sse_message({"message": "Optimization started"}, event='status'))

        grid, max_bonus = optimize_placement(grid, ship, modules, tech, player_owned_rewards, message_queue)

        # Send a completion message
        message_queue.put(sse_message({"message": "Optimization completed"}, event='status'))

        return jsonify({'grid': grid.to_dict(), 'max_bonus': max_bonus})
    except ValueError as e:
        print(f"ERROR -- {str(e)}")
        print_grid_compact(grid)
        message_queue.put(sse_message({"message": f"Error: {str(e)}"}, event='status'))
        return jsonify({'error': str(e)}), 500

@app.route('/tech_tree/<ship_name>')
def get_technology_tree(ship_name):
    """Endpoint to get the technology tree for a given ship."""
    try:
        tree_data = get_tech_tree_json(ship_name)
        return tree_data
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ship_types', methods=['GET'])
def get_ship_types():
    """Endpoint to get the available ship types and their labels."""
    ship_types = {}
    for ship_key, ship_data in modules.items():
        ship_types[ship_key] = ship_data.get("label")

    print(f"Ship types: {ship_types}")
    return jsonify(ship_types)

@app.route('/test_sse')
def test_sse():
    """Endpoint to test SSE messages."""
    from sse_events import sse_message
    message_queue.put(sse_message({"message": "Test message from server"}, event='test'))
    return "Message sent!"

# Start the message sending thread
if __name__ == '__main__':
    app.run(debug=True)
