import asyncio
import logging
from pyodide.ffi import to_js, create_proxy

from .refinement import simulated_annealing
from .grid_utils import Grid

# Set up basic logging to see output in the browser console
logging.basicConfig(level=logging.INFO)

# Helper to convert Python dicts to JS Objects, including nested ones
def js_object_converter(data):
    if isinstance(data, dict):
        return {k: js_object_converter(v) for k, v in data.items()}
    if isinstance(data, list):
        return [js_object_converter(i) for i in data]
    return data

# This is the main entrypoint that will be called from JavaScript
async def run_optimization(grid_data, modules_data, ship, tech, player_owned_rewards, progress_callback_js):
    """
    Runs the simulated annealing optimization in a browser environment.

    Args:
        grid_data (JsProxy): A JavaScript object representing the grid.
        modules_data (JsProxy): A JavaScript object with module definitions.
        ship (str): The ship type key.
        tech (str): The technology key.
        player_owned_rewards (JsProxy): A JavaScript array of reward module IDs.
        progress_callback_js (JsProxy): A JavaScript function to be called for progress updates.
    """
    try:
        logging.info("Starting optimization in WASM...")
        # Convert JavaScript data to Python objects
        grid_dict = grid_data.to_py()
        modules = modules_data.to_py()
        rewards = player_owned_rewards.to_py()

        # Create a Grid instance from the dictionary
        grid = Grid.from_dict(grid_dict)

        # Create a proxy for the JavaScript progress callback.
        # This allows the Python code to invoke the JS function.
        # The proxy also ensures that the call is handled correctly in the browser's event loop.
        async def progress_callback_wrapper(progress_data):
            # The call to the JS function is proxied. We add an `asyncio.sleep(0)`
            # to explicitly yield control back to the browser's event loop,
            # allowing the UI to repaint with the progress update.
            progress_callback_js(to_js(progress_data, default_converter=js_object_converter))
            await asyncio.sleep(0)

        # We are calling simulated_annealing directly. In a full implementation,
        # we would replicate the logic from `optimization/core.py` here.
        optimized_grid, best_score = await simulated_annealing(
            grid=grid,
            ship=ship,
            modules=modules,
            tech=tech,
            full_grid=grid.copy(), # Pass a copy for the full grid context
            player_owned_rewards=rewards,
            progress_callback=progress_callback_wrapper,
            # Using slightly reduced parameters for faster browser execution
            initial_temperature=3000,
            iterations_per_temp=25,
            max_processing_time=30.0, # 30-second timeout
        )

        if optimized_grid:
            logging.info(f"Optimization successful. Final score: {best_score}")
            result = {
                "success": True,
                "grid": optimized_grid.to_dict(),
                "score": best_score,
            }
        else:
            logging.error("Optimization failed to return a valid grid.")
            result = {"success": False, "error": "Optimization failed"}

        return to_js(result, default_converter=js_object_converter)

    except Exception as e:
        logging.error(f"An error occurred during optimization: {e}", exc_info=True)
        # Ensure errors are also returned in a JS-friendly format
        return to_js({"success": False, "error": str(e)}, default_converter=js_object_converter)