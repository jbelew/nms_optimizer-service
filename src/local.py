# local.py
from optimizer import (
    optimize_placement,
    print_grid,
    print_grid_compact,
    Grid,
)
from modules import modules

# Define the grid dimensions
grid_width = 4
grid_height = 3

# Create a new grid
grid = Grid(width=grid_width, height=grid_height)

# Define the initial grid configuration (optional)
initial_grid_config = {
    "cells": [
        [
            {
                "adjacency": False,
                "adjacency_bonus": 0.0,
                "bonus": 0.0,
                "image": None,
                "module": None,
                "sc_eligible": False,
                "supercharged": False,
                "tech": None,
                "total": 0.0,
                "type": "",
                "value": 0,
                "active": True,
                "label": "",
            }
            for _ in range(grid_width)
        ]
        for _ in range(grid_height)
    ],
    "height": grid_height,
    "width": grid_width,
}

# Set supercharged slots in the initial grid (optional)
supercharged_positions = [(0, 1), (1, 0), (1,1)]  # Example: (x, y) coordinates
for x, y in supercharged_positions:
    initial_grid_config["cells"][y][x]["supercharged"] = True
    
inactive_positions = [(2, 0)]  # Example: (x, y) coordinates
for x, y in inactive_positions:
    initial_grid_config["cells"][y][x]["active"] = False

# Load the initial grid configuration
grid = Grid.from_dict(initial_grid_config)

# Define the optimization parameters
ship = "Exotic"
tech = "infra"

# Run the simulated annealing optimization
grid, max_bonus = optimize_placement(
    grid,
    ship,
    modules,
    tech
)

# Print the results
print(f"Optimized layout for {ship} ({tech}) -- Max Bonus: {max_bonus}")
print_grid(grid)

# Alternative: Run the brute-force optimization (uncomment to use)
# grid, max_bonus = optimize_placement(grid, ship, modules, tech)
# print(f"Optimized layout (brute-force) for {ship} ({tech}) -- Max Bonus: {max_bonus}")
# print_grid(grid)
