"""
This module provides a solver for finding the best module layout from .npz files.
"""

import glob
import os
from typing import Optional, Tuple, List

from src.grid_display import print_grid
import numpy as np

from src.bonus_calculations import calculate_grid_score
from src.data_loader import get_all_module_data
from src.grid_utils import Grid
from src.module_placement import place_module
from src.data_definitions.npz_mapping import get_npz_keys

modules = get_all_module_data()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def find_best_layout_from_npz(
    grid: Grid,
    ship: str,
    tech: str,
    solve_type: Optional[str] = None,
    player_owned_rewards: Optional[List[str]] = None,
) -> Tuple[Optional[Grid], float]:
    """
    Finds the best layout for a given number of supercharged slots in a directory of generated data.
    """
    keys = get_npz_keys(ship, tech, solve_type, player_owned_rewards=player_owned_rewards)
    npz_ship_key = keys["npz_ship_key"]
    npz_tech_key = keys["npz_tech_key"]

    num_supercharged = grid.count_supercharged()
    directory = os.path.join(
        "/home/jbelew/projects/nms_optimizer-service/scripts/training/generated_batches",
        npz_ship_key,
        npz_tech_key,
    )
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
    print(f"DEBUG: Searching for NPZ files in directory: {directory}. Found {len(filepaths)} files.")
    if not filepaths:
        print(f"No .npz files found in directory: {directory}")
        return None, 0.0

    # Determine NPZ grid dimensions from the first file
    first_filepath = filepaths[0]
    try:
        data = np.load(first_filepath)
        X_supercharge_sample = data["X_supercharge"]
        npz_grid_height, npz_grid_width = X_supercharge_sample[0].shape
    except Exception as e:
        print(f"Error reading dimensions from file {first_filepath}: {e}")
        return None, 0.0

    original_grid = grid
    rotated_input_grid = False

    # Check if the input grid needs to be rotated to match NPZ dimensions
    if grid.width == npz_grid_height and grid.height == npz_grid_width:
        grid = grid.rotate_grid()
        rotated_input_grid = True
        num_supercharged = grid.count_supercharged() # Recalculate for rotated grid

    for filepath in filepaths:
        try:
            data = np.load(filepath)
            X_supercharge = data["X_supercharge"]
            y = data["y"]

            grid_height_npz, grid_width_npz = X_supercharge[0].shape

            if grid_width_npz != grid.width or grid_height_npz != grid.height:
                continue

            for i in range(len(X_supercharge)):
                if np.sum(X_supercharge[i]) == num_supercharged:
                    grid_height, grid_width = X_supercharge[i].shape
                    current_grid = Grid(grid_width, grid_height)

                    for r in range(grid_height):
                        for c in range(grid_width):
                            if X_supercharge[i][r, c] == 1:
                                current_grid.set_supercharged(c, r, True)

                    # Check if the supercharged slots match the input grid
                    sc_match = True
                    if grid.count_supercharged() > 0:
                        for r in range(grid.height):
                            for c in range(grid.width):
                                if grid.is_supercharged(c, r) != current_grid.is_supercharged(c, r):
                                    sc_match = False
                                    break
                            if not sc_match:
                                break

                    if not sc_match:
                        continue

                    for r in range(grid_height):
                        for c in range(grid_width):
                            module_class = y[i][r, c]
                            if module_class > 0:
                                module_id = module_id_mapping.get(module_class)
                                if module_id:
                                    module_data = module_defs_map.get(module_id)
                                    if module_data:
                                        place_module(
                                            current_grid,
                                            c,
                                            r,
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
                        print_grid(best_grid)

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    if best_grid and rotated_input_grid:
        best_grid = best_grid.rotate_grid()

    return best_grid, best_score
