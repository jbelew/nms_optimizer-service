# WASM Optimizer Proof-of-Concept

This directory contains a proof-of-concept (POC) to run the simulated annealing optimization algorithm in the browser using WebAssembly (via Pyodide).

## How to Run This POC

### 1. Build the Python Package

First, you need to build the `sa_wasm_optimizer` Python package into a distribution file (a `.whl` file) that Pyodide can install.

Navigate to the `wasm` directory in your terminal and run the following command:

```bash
pyodide-build
```

This command will build the package and place it in the `dist/` directory inside `wasm/`, for example: `dist/sa_wasm_optimizer-0.1.0-py3-none-any.whl`.

*(Note: If you don't have `pyodide-build` installed, you can install it with `pip install pyodide-build`)*

### 2. Start a Local HTTP Server

To serve the `index.html` and the built package, you need to run a local HTTP server from within the `wasm` directory. A simple way to do this is with Python's built-in HTTP server.

Make sure you are in the `wasm` directory, then run:

```bash
python3 -m http.server 8000
```

This will start a server on port 8000.

### 3. Open the Test Harness in Your Browser

Open your web browser and navigate to the following URL:

[http://localhost:8000](http://localhost:8000)

### 4. Run the Optimization

You should see a page titled "Simulated Annealing WASM Test". Click the **"Run Optimization"** button.

- You will see status updates as the Pyodide runtime and our custom package are loaded.
- Once ready, the optimization will run. The status display will show real-time progress from the simulated annealing algorithm.
- When complete, the final optimized grid and score will be displayed in the "Result" section.

This setup demonstrates that the core Python logic can be successfully executed on the client-side, offloading the computation from the server.