"""
This module provides an iterative solver for finding the best module layout.
"""
import glob
import logging
import os
from typing import Optional, Tuple

import numpy as np

from .bonus_calculations import calculate_grid_score
from .data_loader import get_all_module_data
from .grid_utils import Grid
from .module_placement import place_module

modules = get_all_module_data()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def rotate_np_array(arr):
    """Rotates a 2D numpy array 90 degrees clockwise."""
    return np.rot90(arr, k=-1) # k=-1 for clockwise

def iterative_placement(
    grid: Grid,
    ship: str,
    tech: str,
    solve_type: Optional[str] = None,
) -> Tuple[Optional[Grid], float]:
    """
    Finds the best layout for a given number of supercharged slots in a directory of generated data.
    """
    num_supercharged = grid.count_supercharged()
    directory = os.path.join(os.path.dirname(__file__), "../scripts/training/generated_batches", ship, tech)
    if solve_type:
        directory = os.path.join(directory, solve_type)

    best_score = -1.0
    best_grid = None

    ship_data = modules.get(ship)
    if not ship_data:
        print(f"Error: Ship '{ship}' not found in module data.")
        return None, 0.0

    all_modules_for_ship = []
    for tech_list in ship_data.get("types", {}).values():
        all_modules_for_ship.extend(tech_list)

    candidates_for_tech = []
    for tech_info_candidate in all_modules_for_ship:
        if tech_info_candidate.get("key") == tech:
            candidates_for_tech.append(tech_info_candidate)

    selected_tech_info = None
    for candidate in candidates_for_tech:
        if candidate.get("type") == solve_type:
            selected_tech_info = candidate
            break

    if selected_tech_info is None and solve_type is None:
        for candidate in candidates_for_tech:
            if candidate.get("type") is None:
                selected_tech_info = candidate
                break

    if not selected_tech_info:
        print(f"Error: No tech modules found for ship='{ship}', tech='{tech}'.")
        return None, 0.0

    tech_modules = selected_tech_info.get("modules", [])

    tech_modules.sort(key=lambda m: m["id"])
    module_id_mapping = {i + 1: module["id"] for i, module in enumerate(tech_modules)}
    module_defs_map = {m["id"]: m for m in tech_modules}

    filepaths = glob.glob(os.path.join(directory, "*.npz"))
    if not filepaths:
        print(f"No .npz files found in directory: {directory}")
        return None, 0.0

    for filepath in filepaths:
        try:
            data = np.load(filepath)
            X_supercharge_npz = data["X_supercharge"]
            y_npz = data["y"]

            grid_height_npz, grid_width_npz = X_supercharge_npz[0].shape
            print(f"DEBUG: grid.width: {grid.width}, grid.height: {grid.height}")
            print(f"DEBUG: grid_width_npz: {grid_width_npz}, grid_height_npz: {grid_height_npz}")

            # Check for direct match
            if grid_width_npz == grid.width and grid_height_npz == grid.height:
                X_supercharge_processed = X_supercharge_npz
                y_processed = y_npz
            # Check for rotated match
            elif grid_width_npz == grid.height and grid_height_npz == grid.width:
                X_supercharge_processed = np.array([rotate_np_array(arr) for arr in X_supercharge_npz])
                y_processed = np.array([rotate_np_array(arr) for arr in y_npz])
                print("DEBUG: Rotated NPZ data to match grid dimensions.")
            else:
                continue # Skip if dimensions don't match directly or rotated

            for i in range(len(X_supercharge_processed)):
                if np.sum(X_supercharge_processed[i]) == num_supercharged:
                    current_grid = Grid(grid.width, grid.height)

                    # Use the dimensions of the processed NPZ data for iteration
                    processed_height, processed_width = X_supercharge_processed[i].shape

                    for r in range(processed_height):
                        for c in range(processed_width):
                            if X_supercharge_processed[i][r, c] == 1:
                                current_grid.set_supercharged(c, r, True)

                    for r in range(processed_height):
                        for c in range(processed_width):
                            module_class = y_processed[i][r, c]
                            if module_class > 0:
                                module_id = module_id_mapping.get(module_class)
                                if module_id:
                                    module_data = module_defs_map.get(module_id)
                                    if module_data:
                                        # Transform coordinates if the input grid was rotated
                                        target_c, target_r = c, r
                                        if rotated_this_npz:
                                            # This means the NPZ data was originally rotated relative to the input grid
                                            # So, the (c, r) from the NPZ data needs to be rotated back to match the input grid's orientation
                                            # The rotation is 90 degrees clockwise, so to get original, rotate 90 degrees counter-clockwise
                                            target_c = r
                                            target_r = processed_width - 1 - c

                                        place_module(
                                            current_grid,
                                            target_c,
                                            target_r,
                                            module_id=module_data.get("id"),
                                            label=module_data.get("label"),
                                            tech=tech,
                                            module_type=module_data.get("type"),
                                            bonus=module_data.get("bonus"),
                                            adjacency=module_data.get("adjacency"),
                                            sc_eligible=module_data.get("sc_eligible"),
                                            image=module_data.get("image"),
                                        )

                    score = calculate_grid_score(current_grid, tech)

                    if score > best_score:
                        best_score = score
                        best_grid = current_grid

            if best_grid and rotated_this_npz:
                best_grid = best_grid.rotate_grid() # Rotate back if input was rotated

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    return best_grid, best_score
