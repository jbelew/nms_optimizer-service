# test_simulated_annealing.py
import sys
import os
import time
import statistics
import random
from copy import deepcopy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from optimizer import (
    refine_placement,
    Grid,
    calculate_grid_score
)
from simulated_annealing import simulated_annealing
from modules import modules
from module_placement import place_module, clear_all_modules_of_tech
from grid_display import print_grid_compact

def create_random_test_grid(grid):
    """Creates a 4x3 test grid with 4 randomly placed supercharged slots and random active slots."""
    # Reset all cells to inactive and not supercharged
    for y in range(grid.height):
        for x in range(grid.width):
            grid.cells[y][x]["active"] = False
            grid.cells[y][x]["supercharged"] = False

    # Randomly set active slots
    for y in range(grid.height):
        for x in range(grid.width):
            if random.random() < 0.99:  # 99% chance to be active
                grid.cells[y][x]["active"] = True

    # Randomly select 4 positions for supercharged slots
    all_positions = [(x, y) for y in range(grid.height) for x in range(grid.width)]
    supercharged_positions = random.sample(all_positions, 4)

    # Set the selected positions as supercharged
    for x, y in supercharged_positions:
        grid.cells[y][x]["supercharged"] = True

    return grid

def run_simulated_annealing(base_grid, ship, tech, num_runs=10, **kwargs):
    """Runs simulated annealing multiple times on the same base grid and returns the results."""
    scores = []
    times = []
    best_grid = None
    best_score = -float('inf')
    final_grid = None

    for _ in range(num_runs):
        # Reset the grid to the base grid for each run
        temp_grid = base_grid.copy()
        clear_all_modules_of_tech(temp_grid, tech)
        start_time = time.time()
        result_grid, score = simulated_annealing(temp_grid, ship, modules, tech, **kwargs)
        end_time = time.time()
        scores.append(score)
        times.append(end_time - start_time)
        if score > best_score:
            best_score = score
            best_grid = result_grid
        final_grid = result_grid # Store the last grid

    return scores, times, best_grid, best_score

def run_refine_placement(grid, ship, tech):
    """Runs refine_placement and returns the results."""
    start_time = time.time()
    temp_grid = grid.copy()
    clear_all_modules_of_tech(temp_grid, tech)
    result_grid, score = refine_placement(temp_grid, ship, modules, tech)
    end_time = time.time()
    return score, end_time - start_time, result_grid, temp_grid

def test_algorithm_comparison(ship, tech, sa_params):
    """Compares simulated annealing and refine_placement on a given grid."""
    # Create a fresh, empty grid for each test case
    base_grid = Grid(4, 3)
    base_grid = create_random_test_grid(base_grid)

    print(f"\n--- Testing with Random Grid ---")

    # Run simulated annealing
    sa_scores, sa_times, sa_best_grid, sa_best_score = run_simulated_annealing(base_grid, ship, tech, **sa_params)
    sa_avg_score = statistics.mean(sa_scores)
    sa_avg_time = statistics.mean(sa_times)

    print(f"Simulated Annealing - Average Score: {sa_avg_score:.2f}, Best Score: {sa_best_score:.2f}, Average Time: {sa_avg_time:.4f}s")
    if sa_avg_time > 15:
        print(f"WARNING -- Simulated annealing average time is over 10 seconds: {sa_avg_time:.4f}s")

    # Run refine_placement
    rp_score, rp_time, rp_grid, rp_start_grid = run_refine_placement(base_grid.copy(), ship, tech)
    print(f"Refine Placement - Score: {rp_score:.2f}, Time: {rp_time:.4f}s")

    # Comparison
    print("\n--- Comparison ---")
    if sa_avg_score > rp_score:
        print(f"Simulated Annealing (Average) is BETTER than Refine Placement by {sa_avg_score - rp_score:.2f}")
    else:
        print(f"Refine Placement is BETTER than Simulated Annealing (Average) by {rp_score - sa_avg_score:.2f}")

    if sa_best_score > rp_score:
        print(f"Simulated Annealing (Best) is BETTER than Refine Placement by {sa_best_score - rp_score:.2f}")
    else:
        print(f"Refine Placement is BETTER than Simulated Annealing (Best) by {rp_score - sa_best_score:.2f}")

    if sa_avg_time < rp_time:
        print(f"Simulated Annealing is FASTER than Refine Placement by {rp_time - sa_avg_time:.4f}s")
    else:
        print(f"Refine Placement is FASTER than Simulated Annealing by {sa_avg_time - rp_time:.4f}s")

    print("-" * 30)
    print("\n--- Simulated Annealing Best Grid ---")
    print_grid_compact(sa_best_grid)
    print("\n--- Refine Placement Grid ---")
    print_grid_compact(rp_grid)
    print("\n--- Refine Placement Start Grid ---")
    print_grid_compact(rp_start_grid)
    print("-" * 30)

if __name__ == "__main__":
    ship = "standard"
    tech = "pulse"

    # Simulated Annealing Parameters
    sa_params = {
        "initial_temperature": 4000,  # Small bump to ensure early movement
        "cooling_rate": 0.995,  # Slower cooling to avoid premature freezing
        "stopping_temperature": 1.0,  # Ensures enough late-stage refinement
        "iterations_per_temp": 30,  # Keep runtime in check
        "initial_swap_probability": 0.55,  # Lower early randomness slightly
        "final_swap_probability": 0.4,  # Allow some late movement without instability
    }
    
    for _ in range(5):
        test_algorithm_comparison(ship, tech, sa_params)
