import numpy as np
import os
import argparse
import glob
from src.grid_utils import Grid
from src.bonus_calculations import calculate_grid_score
from src.data_loader import get_all_module_data
from src.grid_display import print_grid_compact
from src.module_placement import place_module
from src.modules_utils import get_tech_modules_for_training
from src.data_definitions.model_mapping import get_model_keys
from src.optimization.helpers import determine_window_dimensions

modules = get_all_module_data()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def find_best_layout(directory, num_supercharged, ship, tech):
    """
    Finds the best layout for a given number of supercharged slots in a directory of generated data.
    """
    best_score = -1.0
    best_grid = None
    best_file = None

    tech_modules = get_tech_modules_for_training(modules, ship, tech)
    if not tech_modules:
        print(f"Error: No tech modules found for ship='{ship}', tech='{tech}'.")
        return

    tech_modules.sort(key=lambda m: m["id"])
    module_id_mapping = {i + 1: module["id"] for i, module in enumerate(tech_modules)}
    module_defs_map = {m["id"]: m for m in tech_modules}

    filepaths = glob.glob(os.path.join(directory, '*.npz'))
    if not filepaths:
        print(f"No .npz files found in directory: {directory}")
        return

    for filepath in filepaths:
        try:
            data = np.load(filepath)
            X_supercharge = data['X_supercharge']
            y = data['y']

            for i in range(len(X_supercharge)):
                if np.sum(X_supercharge[i]) == num_supercharged:
                    grid_height, grid_width = X_supercharge[i].shape
                    grid = Grid(grid_width, grid_height)

                    for r in range(grid_height):
                        for c in range(grid_width):
                            if X_supercharge[i][r, c] == 1:
                                grid.set_supercharged(c, r, True)

                    for r in range(grid_height):
                        for c in range(grid_width):
                            module_class = y[i][r, c]
                            if module_class > 0:
                                module_id = module_id_mapping.get(module_class)
                                if module_id:
                                    module_data = module_defs_map.get(module_id)
                                    if module_data:
                                        place_module(
                                            grid,
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

                    score = calculate_grid_score(grid, tech)

                    if score > best_score:
                        best_score = score
                        best_grid = grid
                        best_file = os.path.basename(filepath)
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    if best_grid:
        print(f"Best layout found in file: {best_file}")
        print(f"Best score: {best_score}")
        print_grid_compact(best_grid)
    else:
        print(f"No layout found with {num_supercharged} supercharged slots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best layout from generated data.")
    parser.add_argument("num_supercharged", type=int, help="Number of supercharged slots.")
    parser.add_argument("--ship", type=str, required=True, help="Ship type (e.g., sentinel, standard).")
    parser.add_argument("--tech", type=str, required=True, help="Tech type (e.g., infra).")
    parser.add_argument("--rewards", nargs='*', help="List of player-owned reward module IDs.")
    args = parser.parse_args()

    # Determine grid dimensions dynamically based on UI-facing keys
    tech_modules_for_dims = get_tech_modules_for_training(modules, args.ship, args.tech)
    if not tech_modules_for_dims:
        print(f"Error: No tech modules found for UI keys ship='{args.ship}', tech='{args.tech}' to determine grid size.")
        exit()
    module_count = len(tech_modules_for_dims)
    grid_w, grid_h = determine_window_dimensions(module_count, args.tech)

    # Get the correct internal keys for locating data, which may be different from UI keys
    keys = get_model_keys(
        ui_ship_key=args.ship,
        ui_tech_key=args.tech,
        grid_width=grid_w,
        grid_height=grid_h,
        player_owned_rewards=args.rewards
    )
    data_ship_key = keys["module_def_ship_key"]
    data_tech_key = keys["module_def_tech_key"]

    # Construct the directory path from the mapped keys
    directory = os.path.join(project_root, "src", "training", "generated_batches", data_ship_key, data_tech_key)
    print(f"Searching for data in: {directory}")

    # Call the function with the mapped keys for data processing
    find_best_layout(directory, args.num_supercharged, data_ship_key, data_tech_key)
