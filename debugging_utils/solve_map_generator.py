import argparse
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from grid_utils import Grid
from modules import modules
from optimization_algorithms import refine_placement, refine_placement_for_training  # Import refine_placement
from grid_display import print_grid, print_grid_compact


def generate_solve_map(tech, grid_width=3, grid_height=3, player_owned_rewards=None, supercharged_positions=None):
    """
    Generates a single solve map for a given technology.

    Args:
        tech (str): The technology key.
        grid_width (int, optional): The width of the grid. Defaults to 3.
        grid_height (int, optional): The height of the grid. Defaults to 2.
        player_owned_rewards (list, optional): List of player-owned reward module IDs. Defaults to ["PC", "SB", "SP", "TT"].
        supercharged_positions (list, optional): List of (x, y) tuples for supercharged cells. Defaults to None.
    """
    if player_owned_rewards is None:
        player_owned_rewards = ["PC", "SB", "SP", "TT", "RL", "PR"]

    grid = Grid(width=grid_width, height=grid_height)

    # Set the specified positions as supercharged
    if supercharged_positions:
        for x, y in supercharged_positions:
            if 0 <= x < grid.width and 0 <= y < grid.height:
                grid.set_supercharged(x, y, True)

    try:
        optimized_grid, optimized_score = refine_placement(grid, "atlantid-mt", modules, tech, player_owned_rewards)
        print_grid(optimized_grid)
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
            template[(x, y)] = module_id
    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a solve map for a given technology.")
    parser.add_argument("--tech", type=str, default="infra", help="Technology key (e.g., 'pulse', 'infra')")
    parser.add_argument("--width", type=int, default=3, help="Grid width")
    parser.add_argument("--height", type=int, default=3, help="Grid height")
    parser.add_argument(
        "--rewards",
        type=str,
        nargs="*",
        default=["PC", "SB", "SP", "TT", "PR", "RL", "SH", "RA"],
        help="List of player-owned reward module IDs",
    )
    parser.add_argument(
        "--supercharged",
        type=int,
        nargs="*",
        help="List of supercharged positions as x y pairs (e.g., --supercharged 0 0 1 1)",
    )
    args = parser.parse_args()

    tech = args.tech  # Get technology from command line or use default
    grid_width = args.width
    grid_height = args.height
    player_owned_rewards = args.rewards

    # Parse supercharged positions from command line arguments
    supercharged_positions = []
    if args.supercharged:
        if len(args.supercharged) % 2 != 0:
            print("Error: Supercharged positions must be provided in x y pairs.")
            exit()
        for i in range(0, len(args.supercharged), 2):
            supercharged_positions.append((args.supercharged[i], args.supercharged[i + 1]))

    solve_map, solve_score = generate_solve_map(tech, grid_width, grid_height, player_owned_rewards, supercharged_positions=supercharged_positions)

    if solve_map:
        print(f"\nSolve map for {tech}: {solve_score:.2f}")  # Corrected formatting
        print_grid(solve_map)  # Directly print the Grid object

        # Generate the solve map template from the Grid object
        solve_map_template = generate_solve_map_template(solve_map)
        print("\nSolve Map Template:")
        print(f'    "{tech}": {{')
        print('        "map": {')
        for (x, y), module_id in solve_map_template.items():
            print(f"            ({x}, {y}): \"{module_id}\",")
        print("        },")
        print(f'        "score": {solve_score:.4f}')
        print("    },")
