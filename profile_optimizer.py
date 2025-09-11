import cProfile
import pstats
import io
import os
import sys

# Ensure the source directory is in the Python path
# This allows the script to be run from the root of the project
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.app import run_optimization
from src.grid_utils import Grid
from src.data_definitions.grids import grids

def profile_optimization_run():
    """
    Sets up and runs a profiling session for the optimization function.
    """
    # 1. Construct the input data
    ship_type = "standard"
    tech_type = "pulse"

    # Get the grid definition for the standard ship
    grid_def = grids.get(ship_type)
    if not grid_def:
        print(f"Error: Could not find grid definition for ship type '{ship_type}'")
        return

    # Create a Grid object from the definition
    grid = Grid(grid_def['grid'][0].__len__(), grid_def['grid'].__len__())
    for y, row in enumerate(grid_def['grid']):
        for x, cell_def in enumerate(row):
            grid.set_active(x, y, cell_def.get('active', False))
            grid.set_supercharged(x, y, cell_def.get('supercharged', False))

    # The 'player_owned_rewards' can be an empty list for this test
    player_owned_rewards = []

    # Assemble the data dictionary
    data = {
        "ship": ship_type,
        "tech": tech_type,
        "grid": grid.to_dict(),
        "player_owned_rewards": player_owned_rewards,
        "forced": True  # Force SA to run
    }

    # 2. Run the profiler
    pr = cProfile.Profile()
    pr.enable()

    print(f"Starting optimization for {ship_type}/{tech_type}...")
    # We don't need the result for profiling, just the execution time
    run_optimization(data)

    pr.disable()
    print("Optimization finished.")

    # 3. Print the stats
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    print("\n--- Profiling Results ---")
    # Print stats for the key functions we are interested in
    print(f"Stats for simulated_annealing and related functions:\n")
    ps.print_stats('simulated_annealing', 'calculate_grid_score', 'ml_placement')

if __name__ == "__main__":
    profile_optimization_run()
