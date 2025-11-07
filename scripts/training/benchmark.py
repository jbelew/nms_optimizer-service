# Add the project root to the Python path
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import numpy as np
import time
import argparse
from src.grid_utils import Grid  # noqa: E402
from src.data_loader import get_all_module_data  # noqa: E402
from src.optimization.refinement import simulated_annealing  # noqa: E402
from src.modules_utils import get_tech_modules  # noqa: E402


def run_benchmark(
    num_runs=10,
    initial_temperature=3000,
    cooling_rate=0.999,
    stopping_temperature=0.1,
    iterations_per_temp=35,
    initial_swap_probability=0.55,
    final_swap_probability=0.25,
    max_steps_without_improvement=150,
    reheat_factor=0.6,
    max_reheats=10,
    num_sa_runs=2,
    max_processing_time=25.0,
):
    """
    Runs a benchmark on the simulated annealing algorithm.
    """
    print("Starting benchmark with the following parameters:")
    print(f"  num_runs: {num_runs}")
    print(f"  initial_temperature: {initial_temperature}")
    print(f"  cooling_rate: {cooling_rate}")
    print(f"  stopping_temperature: {stopping_temperature}")
    print(f"  iterations_per_temp: {iterations_per_temp}")
    print(f"  initial_swap_probability: {initial_swap_probability}")
    print(f"  final_swap_probability: {final_swap_probability}")
    print(f"  max_steps_without_improvement: {max_steps_without_improvement}")
    print(f"  reheat_factor: {reheat_factor}")
    print(f"  max_reheats: {max_reheats}")
    print(f"  num_sa_runs: {num_sa_runs}")
    print(f"  max_processing_time: {max_processing_time}")
    print("-" * 20)

    # Define a static grid
    grid = Grid(3, 4)
    grid.set_supercharged(1, 1, True)
    grid.set_supercharged(2, 3, True)
    # grid.set_supercharged(2, 1, True)
    # grid.set_supercharged(2, 2, True)

    # Load modules
    all_modules = get_all_module_data()
    ship = "corvette"
    modules = all_modules.get(ship)
    if not modules:
        print(f"Could not find module data for ship: {ship}")
        return
    tech = "pulse"

    tech_modules = get_tech_modules(modules, ship, tech)

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
            tech_modules=tech_modules,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            stopping_temperature=stopping_temperature,
            iterations_per_temp=iterations_per_temp,
            initial_swap_probability=initial_swap_probability,
            final_swap_probability=final_swap_probability,
            max_steps_without_improvement=max_steps_without_improvement,
            reheat_factor=reheat_factor,
            max_reheats=max_reheats,
            num_sa_runs=num_sa_runs,
            max_processing_time=max_processing_time,
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
    parser = argparse.ArgumentParser(description="Benchmark for simulated annealing.")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of benchmark runs.")
    parser.add_argument("--initial_temperature", type=float, default=3000, help="Initial temperature.")
    parser.add_argument("--cooling_rate", type=float, default=0.999, help="Cooling rate.")
    parser.add_argument("--stopping_temperature", type=float, default=0.1, help="Stopping temperature.")
    parser.add_argument("--iterations_per_temp", type=int, default=35, help="Iterations per temperature.")
    parser.add_argument("--initial_swap_probability", type=float, default=0.55, help="Initial swap probability.")
    parser.add_argument("--final_swap_probability", type=float, default=0.25, help="Final swap probability.")
    parser.add_argument("--max_steps_without_improvement", type=int, default=150, help="Max steps without improvement.")
    parser.add_argument("--reheat_factor", type=float, default=0.6, help="Reheat factor.")
    parser.add_argument("--max_reheats", type=int, default=10, help="Maximum number of reheats.")
    parser.add_argument("--num_sa_runs", type=int, default=2, help="Number of SA runs.")
    parser.add_argument("--max_processing_time", type=float, default=25.0, help="Max processing time.")
    args = parser.parse_args()

    run_benchmark(
        num_runs=args.num_runs,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        stopping_temperature=args.stopping_temperature,
        iterations_per_temp=args.iterations_per_temp,
        initial_swap_probability=args.initial_swap_probability,
        final_swap_probability=args.final_swap_probability,
        max_steps_without_improvement=args.max_steps_without_improvement,
        reheat_factor=args.reheat_factor,
        max_reheats=args.max_reheats,
        num_sa_runs=args.num_sa_runs,
        max_processing_time=args.max_processing_time,
    )
