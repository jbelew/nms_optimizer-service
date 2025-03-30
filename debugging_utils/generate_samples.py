import sys
import os
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from optimizer import optimize_placement, Grid, print_grid_compact
from modules import modules  # Import modules from modules_refactored.py

# --- Data Generation ---
def generate_debug_data(num_samples, grid_width, grid_height, max_supercharged, ship, tech_filter):
    """
    Generates data for debugging purposes, focusing on visual output.

    Args:
        num_samples: The number of samples to generate.
        grid_width: The width of the grid.
        grid_height: The height of the grid.
        max_supercharged: The maximum number of supercharged slots.
        ship: The type of ship.
        tech_filter: The technology filter to apply.

    Returns:
        A list of optimized grids (Grid objects) and their corresponding bonus scores.
    """
    results = []

    for i in range(num_samples):
        grid = Grid(grid_width, grid_height)
        num_supercharged = 4  # Always have 4 supercharged slots

        # Determine the number of inactive cells (25% of total grid cells)
        total_cells = grid_width * grid_height
        num_inactive_cells = int(total_cells * 0.25)

        # Randomly select 25% of the grid to be inactive
        inactive_positions = random.sample(
            [(x, y) for y in range(grid_height) for x in range(grid_width)],
            num_inactive_cells,
        )

        for x, y in inactive_positions:
            grid.set_active(x, y, False)  # Assuming the Grid class has this method

        # Constrain supercharged slots to the top three rows and only active cells
        top_three_rows_active_positions = [
            (x, y) for y in range(min(3, grid_height)) for x in range(grid_width)
            if grid.get_cell(x, y)["active"]
        ]

        supercharged_positions = random.sample(top_three_rows_active_positions, min(num_supercharged, len(top_three_rows_active_positions)))

        for x, y in supercharged_positions:
            grid.set_supercharged(x, y, True)

        # Use the specified technology
        tech = tech_filter

        # try:
        optimized_grid, best_bonus = optimize_placement(grid, ship, modules, tech, ["PC"])
        results.append((optimized_grid, best_bonus))
        # except Exception as e:
        #     print(f"Error during optimization for sample {i + 1}: {e}")
        #     continue  # Skip this sample if optimization fails

    return results


# Example usage for debugging:
num_samples = 32  # Generate 5 samples for debugging
grid_width = 8
grid_height = 8
max_supercharged = 4
ship = "standard"
tech_filter = "infra"

debug_data = generate_debug_data(num_samples, grid_width, grid_height, max_supercharged, ship, tech_filter)
