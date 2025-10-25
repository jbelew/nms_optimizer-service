# Technical Design Document

This document details the implementation plan for the NMS Optimizer Service.

## 1. Backend

The backend is a Python Flask application that provides a REST API and a WebSocket interface for the optimization service.

### 1.1. Technologies

*   **Python 3.14**: The programming language used for the backend.
*   **Flask**: A lightweight web framework for Python.
*   **Flask-SocketIO**: A Flask extension that provides WebSocket support.
*   **Gunicorn**: A Python WSGI HTTP Server for UNIX.
*   **Gevent**: A coroutine-based Python networking library that is used to handle concurrent WebSocket connections.
*   **PyTorch**: A machine learning framework used for the module placement prediction.
*   **NumPy**: A library for numerical computing in Python.

### 1.2. API Endpoints

*   `POST /optimize`: The main endpoint for running an optimization. It accepts a JSON payload with the grid configuration and other parameters, and returns the optimized grid and bonus stats.
*   `GET /tech_tree/<ship_name>`: Retrieves the technology tree for a given ship, including recommended builds and the grid definition.
*   `GET /platforms`: Retrieves a list of all available ship types (platforms).
*   `GET /analytics/popular_data`: Fetches and returns popular optimization data from Google Analytics.

### 1.3. WebSocket Events

*   `optimize`: A WebSocket event that triggers a real-time optimization. The server sends progress updates to the client during the optimization process.

### 1.4. Optimization Logic

The optimization logic is implemented in the `src/optimization/core.py` module. The `optimize_placement` function uses a multi-step approach to find the optimal layout:

1.  **Solve Map Lookup**: It first checks for a pre-calculated optimal solution (a "solve map") in the `src/data_definitions/solves/` directory.
2.  **Pattern Matching**: If a solve map is found, it uses pattern matching to find the best placement of the pattern on the user's grid.
3.  **Opportunity Refinement**: It then identifies "opportunity windows" (areas with supercharged slots) and uses either a Machine Learning model or a Simulated Annealing algorithm to refine the placement within that window.
4.  **Simulated Annealing**: If no solve map is found, or if the pattern matching fails, it falls back to using a Simulated Annealing algorithm to find a good placement.

### 1.5. Machine Learning Model

The machine learning model is a convolutional neural network (CNN) implemented in PyTorch. The model is trained to predict the optimal placement of modules based on the grid's supercharged slot configuration. The trained models are stored in the `src/trained_models/` directory.

## 2. Frontend

The frontend is a web-based client that interacts with the backend to provide a user-friendly interface for the optimization service. The frontend is not part of this project, but it is expected to be a single-page application (SPA) built with a modern JavaScript framework like React or Vue.js.

## 3. Data

The application's data is stored in a collection of JSON and Python files in the `src/data_definitions` directory. This includes:

*   **`grids.py`**: Defines the grid layouts for different ship types.
*   **`modules_data/*.json`**: Contains the data for each module, including its bonus, adjacency requirements, etc.
*   **`solves/*.json`**: Contains the pre-calculated optimal layouts (solve maps).
*   **`recommended_builds.py`**: Contains recommended builds for different ships.
