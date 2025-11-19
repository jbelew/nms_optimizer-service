import numpy as np
import os
import argparse
import glob
import logging
from src.grid_utils import Grid
from src.bonus_calculations import calculate_grid_score
from src.data_loader import get_all_module_data
from src.grid_display import print_grid_compact
from src.module_placement import place_module

logging.basicConfig(level=logging.DEBUG)

modules = get_all_module_data()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def find_best_layout(directory, num_supercharged, ship, tech, solve_type=None):
    """
    Finds the best layout for a given number of supercharged slots in a directory of generated data.
    """
    best_score = -1.0
    best_grid = None
    best_file = None

    ship_data = modules.get(ship)
    if not ship_data:
        print(f"Error: Ship '{ship}' not found in module data.")
        return

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
        return

    tech_modules = selected_tech_info.get("modules", [])

    tech_modules.sort(key=lambda m: m["id"])
    module_id_mapping = {i + 1: module["id"] for i, module in enumerate(tech_modules)}
    module_defs_map = {m["id"]: m for m in tech_modules}

    filepaths = glob.glob(os.path.join(directory, "*.npz"))
    if not filepaths:
        print(f"No .npz files found in directory: {directory}")
        return

    for filepath in filepaths:
        try:
            data = np.load(filepath)
            X_supercharge = data["X_supercharge"]
            y = data["y"]

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
        print('            "map": {')

        map_items = []
        for r in range(best_grid.height):
            for c in range(best_grid.width):
                key = f"{c},{r}"
                cell = best_grid.cells[r][c]
                module_id = cell.get("module")
                if module_id:
                    map_items.append(f'                "{key}": "{module_id}"')
                else:
                    map_items.append(f'                "{key}": "None"')

        print(",\n".join(map_items))
        print("            },")
        print(f'            "score": {best_score:.4f}')
    else:
        print(f"No layout found with {num_supercharged} supercharged slots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best layout from generated data.")
    parser.add_argument("num_supercharged", type=int, help="Number of supercharged slots.")
    parser.add_argument("--ship", type=str, required=True, help="Ship type (e.g., sentinel, standard).")
    parser.add_argument("--tech", type=str, required=True, help="Tech type (e.g., infra).")
    parser.add_argument("--rewards", nargs="*", help="List of player-owned reward module IDs.")
    parser.add_argument("--solve_type", type=str, default=None, help="Specific solve type to use (optional).")
    args = parser.parse_args()

    # Construct the directory path from the arguments
    directory = os.path.join(os.path.dirname(__file__), "generated_batches", args.ship, args.tech)
    if args.solve_type:
        directory = os.path.join(directory, args.solve_type)
    print(f"Searching for data in: {directory}")

    # Call the function with the arguments
    find_best_layout(directory, args.num_supercharged, args.ship, args.tech, args.solve_type)
