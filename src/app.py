# app.py
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from optimizer import optimize_placement, get_tech_tree_json, Grid
from modules import modules
import logging
import json
import queue
import time

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Single message queue for all clients
message_queue = queue.Queue()

def send_messages(client_id):
    """Generator function for SSE to send messages from the queue, filtered by client_id."""
    while True:
        try:
            message = message_queue.get(timeout=1)
            if message is None:
                break  # Signal to stop the thread
            message_data = json.loads(message)
            if message_data.get("clientId") == client_id:
                print(f"Sending message to client {client_id}: {message}")
                yield f"data: {message}\n\n"
        except queue.Empty:
            time.sleep(0.1)
            continue
        except Exception as e:
            print(f"An error occurred in send_messages: {e}")
            break

@app.route('/stream')
def stream():
    """SSE endpoint to stream messages to the client."""
    client_id = request.args.get('clientId')
    if not client_id:
        return "Client ID is required", 400

    return Response(send_messages(client_id), mimetype='text/event-stream')

@app.route('/optimize', methods=['POST'])
def optimize_grid():
    """Endpoint to optimize the grid and send status updates via SSE."""
    data = request.get_json()
    client_id = data.get("clientId")

    if not client_id:
        return jsonify({'error': 'No client id specified'}), 400

    ship = data.get("ship")
    tech = data.get('tech')
    if tech is None:
        return jsonify({'error': 'No tech specified'}), 400

    grid_data = data.get('grid')
    if grid_data is None:
        return jsonify({'error': 'No grid specified'}), 400

    grid = Grid.from_dict(grid_data)

    message_queue.put(json.dumps({"clientId": client_id, "status": "info", "message": "Starting optimization..."}))
    grid, max_bonus = optimize_placement(grid, ship, modules, tech, client_id=client_id, message_queue=message_queue)
    message_queue.put(json.dumps({"clientId": client_id, "status": "success", "message": "Optimization complete!"}))
    return jsonify({'grid': grid.to_dict(), 'max_bonus': max_bonus})

@app.route('/tech_tree/<ship_name>')
def get_technology_tree(ship_name):
    """Endpoint to get the technology tree for a given ship."""
    try:
        tree_data = get_tech_tree_json(ship_name)
        return tree_data
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
