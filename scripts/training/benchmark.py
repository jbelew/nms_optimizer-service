import sys
import os
import numpy as np
from collections import Counter

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.grid_utils import Grid
from src.data_loader import get_all_module_data
from src.optimization.refinement import simulated_annealing
from src.modules_utils import get_tech_modules

def run_benchmark(num_runs=10):
    """
    Runs a benchmark on the simulated annealing algorithm.
    """
    print("Starting benchmark...")

    # Define a static grid
    grid = Grid(4, 4)
    grid.set_supercharged(1, 1, True)
    grid.set_supercharged(1, 2, True)
    grid.set_supercharged(2, 1, True)
    grid.set_supercharged(2, 2, True)

    # Load modules
    all_modules = get_all_module_data()
    ship = "corvette"
    modules = all_modules.get(ship)
    if not modules:
        print(f"Could not find module data for ship: {ship}")
        return
    tech = "hyper"
    player_owned_rewards = []
    solve_type = None

    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards, solve_type)

    scores = []
    for i in range(num_runs):
        print(f"--- Running iteration {i+1}/{num_runs} ---")
        # We need to pass a full_grid argument to simulated_annealing
        full_grid = grid.copy()

        best_grid, best_score = simulated_annealing(
            grid.copy(),
            ship,
            modules,
            tech,
            full_grid,
            player_owned_rewards,
            solve_type=solve_type,
            tech_modules=tech_modules
        )
        scores.append(best_score)
        print(f"Run {i+1}: Score = {best_score}")

    print("\n--- Benchmark Results ---")
    print(f"Scores: {scores}")
    print(f"Mean: {np.mean(scores)}")
    print(f"Median: {np.median(scores)}")
    print(f"Best Score: {np.max(scores)}")
    print(f"Standard Deviation: {np.std(scores)}")

if __name__ == "__main__":
    run_benchmark()