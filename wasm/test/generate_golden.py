import json
import os
import sys
import random

from src.data_loader import load_ship_data, load_module_data
from src.grid import Grid
from src.optimization.refinement import simulated_annealing

def main():
    """
    This script generates a 'golden' file for testing the WASM version of the
    simulated annealing algorithm. It runs the Python implementation with a
    fixed seed and saves the input and output to `wasm/test/golden.json`.
    """
    # Set a fixed seed for reproducibility
    random.seed(42)

    # --- Test Case Configuration ---
    ship_name = "Sentinel"
    tech_name = "infra"
    grid_width = 3
    grid_height = 2
    # ---

    # Load data using the correct functions
    ships = load_ship_data()
    modules = load_module_data()

    # Initialize the grid and modules
    ship = next((s for s in ships if s['name'] == ship_name), None)
    if not ship:
        raise ValueError(f"Ship '{ship_name}' not found.")

    grid_layout = ship['grid']
    grid = Grid(grid_width, grid_height)
    for y, row in enumerate(grid_layout):
        for x, cell_info in enumerate(row):
            if cell_info:
                grid.get_cell(x, y).active = True
                grid.get_cell(x, y).supercharged = cell_info.get('sc', False)

    all_modules = modules
    tech_modules = [m for m in all_modules if m['tech'] == tech_name]

    # These parameters are chosen to be simple and fast for a test case.
    params = {
        'initial_temperature': 1.0,
        'cooling_rate': 0.95,
        'stopping_temperature': 1e-3,
        'iterations_per_temp': 5,
        'initial_swap_probability': 0.6,
        'final_swap_probability': 0.1,
        'start_from_current_grid': False,
        'max_processing_time': 5,
        'max_steps_without_improvement': 50,
        'reheat_factor': 0.2
    }

    # Run the Python implementation
    final_grid, final_score = simulated_annealing(
        grid, ship_name, all_modules, tech_name, tech_modules, **params
    )

    # Prepare data for JSON serialization
    def grid_to_dict(g):
        return {
            "width": g.width,
            "height": g.height,
            "cells": [
                [
                    {
                        "active": cell.active,
                        "supercharged": cell.supercharged,
                        "module_id": cell.module_id,
                        "tech": cell.tech,
                        "x": cell.x,
                        "y": cell.y,
                        "adjacency": cell.adjacency,
                    }
                    for cell in row
                ]
                for row in g.cells
            ],
        }

    def modules_to_list(mods):
        return [
            {
                "id": m["id"],
                "label": m["label"],
                "tech": m["tech"],
                "type": m["type"],
                "bonus": m["bonus"],
                "adjacency": m["adjacency"],
                "sc_eligible": m["sc_eligible"],
                "image": m["image"]
            } for m in mods
        ]


    golden_data = {
        "inputs": {
            "grid": grid_to_dict(grid),
            "ship": ship_name,
            "modules": modules_to_list(all_modules),
            "tech": tech_name,
            "tech_modules": modules_to_list(tech_modules),
            "params": params
        },
        "output": {
            "grid": grid_to_dict(final_grid),
            "score": final_score,
        },
    }

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), 'golden.json')
    with open(output_path, 'w') as f:
        json.dump(golden_data, f, indent=2)

    print(f"Golden file generated at: {output_path}")
    print(f"Test Score: {final_score}")


if __name__ == "__main__":
    main()