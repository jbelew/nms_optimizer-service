# Add the project root to the Python path
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import numpy as np
import time
from src.grid_utils import Grid  # noqa: E402
from src.data_loader import get_all_module_data  # noqa: E402
from src.optimization.refinement import simulated_annealing  # noqa: E402
from src.modules_utils import get_tech_modules  # noqa: E402


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
    run_times = []
    for i in range(num_runs):
        start_time = time.time()
        print(f"--- Running iteration {i+1}/{num_runs} ---")
        full_grid = grid.copy()

        best_grid, best_score = simulated_annealing(
            grid.copy(),
            ship,
            modules,
            tech,
            full_grid,
            player_owned_rewards,
            solve_type=solve_type,
            tech_modules=tech_modules,
            max_processing_time=25.0,
        )
        end_time = time.time()
        run_time = end_time - start_time
        run_times.append(run_time)
        scores.append(best_score)
        print(f"Run {i+1}: Score = {best_score:.4f}, Time = {run_time:.2f}s")

    print("\n--- Benchmark Results ---")
    print(f"Scores: {[f'{s:.4f}' for s in scores]}")
    print(f"Mean Score: {np.mean(scores):.4f}")
    print(f"Median Score: {np.median(scores):.4f}")
    print(f"Best Score: {np.max(scores):.4f}")
    print(f"Standard Deviation: {np.std(scores):.4f}")
    print(f"Average Run Time: {np.mean(run_times):.2f}s")


if __name__ == "__main__":
    run_benchmark()
