import argparse
import sys
import os
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from grid_utils import Grid
from modules import modules
from optimization_algorithms import refine_placement
from grid_display import print_grid, print_grid_compact


def generate_solve_map(ship_type, tech, grid_width=4, grid_height=3, player_owned_rewards=None, supercharged_positions=None):
    """
    Generates a single solve map for a given technology.

    Args:
        ship_type (str): The ship type key.
        tech (str): The technology key.
        grid_width (int, optional): The width of the grid. Defaults to 3.
        grid_height (int, optional): The height of the grid. Defaults to 2.
        player_owned_rewards (list, optional): List of player-owned reward module IDs. Defaults to ["PC", "SB", "SP", "TT"].
        supercharged_positions (list, optional): List of (x, y) tuples for supercharged cells. Defaults to None.
    """
    if player_owned_rewards is None:
        player_owned_rewards = ["PC", "SB", "SP", "TT"]

    grid = Grid(width=grid_width, height=grid_height)

    # Set the specified positions as supercharged
    if supercharged_positions:
        for x, y in supercharged_positions:
            if 0 <= x < grid.width and 0 <= y < grid.height:
                grid.set_supercharged(x, y, True)

    try:
        optimized_grid, optimized_score = refine_placement(grid, ship_type, modules, tech, player_owned_rewards)
        return optimized_grid, optimized_score
    except Exception as e:
        print(f"Error generating solve map for {tech}: {e}")
        return None, None


def generate_solve_map_template(grid):
    """Generates a solve map template from a Grid object."""
    template = {}
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            module_id = cell["module"]
            template[f"({x}, {y})"] = module_id if module_id is not None else "None"
    return template


def generate_all_solves(modules, tech_to_generate=None, weapon_to_generate=None):
    """Generates the solves object for all technologies and types."""
    all_solves = {}
    techs_to_process = [tech_to_generate] if tech_to_generate else modules.keys()
    for tech_key in techs_to_process:
        if tech_key not in modules:
            print(f"Warning: Technology '{tech_key}' not found in modules. Skipping.")
            continue
        tech_data = modules[tech_key]
        all_solves[tech_key] = {}

        for type_key, type_data in tech_data["types"].items():
            if type_key == "Utilities":  # Skip the "Utilities" type
                continue
            for weapon in type_data:
                weapon_key = weapon["key"]
                if weapon_to_generate and weapon_key != weapon_to_generate:
                    continue

                module_count = len(weapon["modules"])

                # Determine grid dimensions based on module count
                if module_count < 4:
                    grid_width, grid_height = 1, 3
                elif module_count < 7:
                    grid_width, grid_height = 2, 3
                elif module_count < 10:
                    grid_width, grid_height = 3, 3
                else:
                    grid_width, grid_height = 4, 3

                print(f"Generating solve map for {tech_key} - {weapon_key} ({module_count} modules) - Grid: {grid_width}x{grid_height}")
                solve_map, solve_score = generate_solve_map(tech_key, weapon_key, grid_width, grid_height)
                if solve_map:
                    solve_map_template = generate_solve_map_template(solve_map)
                    all_solves[tech_key][weapon_key] = {
                        "map": solve_map_template,
                        "score": solve_score,
                    }
        if tech_key in ["standard", "sentinel", "living"]:
            all_solves[tech_key]["trails"] = {
                "map": {
                    "(0, 0)": "None",
                    "(1, 0)": "RT",
                    "(2, 0)": "CT",
                    "(3, 0)": "TT",
                    "(0, 1)": "SB",
                    "(1, 1)": "AB",
                    "(2, 1)": "PB",
                    "(3, 1)": "ET",
                    "(0, 2)": "None",
                    "(1, 2)": "GT",
                    "(2, 2)": "ST",
                    "(3, 2)": "SP",
                },
                "score": 0.162,
            }
    return all_solves


def load_trails_stub(filepath):
    """Loads the trails stub data from a JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            return data[0]["trails"]  # Extract the "trails" data
    except FileNotFoundError:
        print(f"Warning: Trails stub file not found at {filepath}. Skipping trails insertion.")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON format in trails stub file at {filepath}. Skipping trails insertion.")
        return None
    except (KeyError, IndexError):
        print(f"Warning: Invalid format in trails stub file at {filepath}. Skipping trails insertion.")
        return None

def save_solves_to_file(solves_data, filename="new_solves.json", trails_stub_filepath=None):
    """Saves the solves data to a JSON file."""
    output = "solves = " + json.dumps(solves_data, indent=4, sort_keys=False)

    # Replace the quoted keys with unquoted keys
    output = output.replace('"("', "(").replace('")"', ")")

    with open(filename, "w") as f:
        f.write(output)
    print(f"Solves data saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate solve maps for all technologies and types.")
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Generate solves for all technologies and types",
    )
    parser.add_argument(
        "--tech",
        type=str,
        help="Generate solves for a specific technology",
    )
    parser.add_argument(
        "--weapon",
        type=str,
        help="Generate solves for a specific weapon within a technology",
    )
    parser.add_argument(
        "--trails-stub",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "trails_stub.json"),
        help="Path to the trails stub JSON file",
    )
    args = parser.parse_args()

    if args.generate_all:
        all_solves = generate_all_solves(modules)
        save_solves_to_file(all_solves, trails_stub_filepath=args.trails_stub)
    elif args.tech and args.weapon:
        all_solves = generate_all_solves(modules, args.tech, args.weapon)
        save_solves_to_file(all_solves, trails_stub_filepath=args.trails_stub)
    elif args.tech:
        all_solves = generate_all_solves(modules, args.tech)
        save_solves_to_file(all_solves, trails_stub_filepath=args.trails_stub)
    else:
        print("Please use the --generate-all flag to generate all solves, the --tech flag to generate a specific tech, or the --tech and --weapon flags to generate a specific weapon within a tech.")
