# /home/jbelew/projects/nms_optimizer/nms_optimizer-service/debugging_utils/solve_map_generator.py
import argparse
import sys
import os


from src.grid_utils import Grid
from src.data_loader import get_all_module_data

modules = get_all_module_data()

# Import both solver options
from src.optimization.training import refine_placement_for_training
from src.optimization.refinement import simulated_annealing
from src.grid_display import print_grid


# <<< Update function signature to accept ship_type >>>
def generate_solve_map(
    ship_type,
    tech,
    grid_width=3,
    grid_height=3,
    player_owned_rewards=None,
    supercharged_positions=None,
    solver_choice="sa",
    solve_type=None,
):
    """
    Generates a single solve map for a given technology and ship type.

    Args:
        ship_type (str): The ship type key (e.g., 'standard', 'sentinel').
        tech (str): The technology key.
        grid_width (int, optional): The width of the grid. Defaults to 3.
        grid_height (int, optional): The height of the grid. Defaults to 3.
        player_owned_rewards (list, optional): List of player-owned reward module IDs. Defaults to ["PC", "SB", "SP", "TT"].
        supercharged_positions (list, optional): List of (x, y) tuples for supercharged cells. Defaults to None.
        solver_choice (str, optional): The solver to use ('sa' or 'refine_training'). Defaults to 'sa'.
    """
    if player_owned_rewards is None:
        # <<< Simplified default rewards list >>>
        player_owned_rewards = ["SB", "SP", "TT"]

    ship_modules = modules.get(ship_type)
    if not ship_modules:
        print(f"Error: no modules for ship {ship_type}")
        return None, None

    grid = Grid(width=grid_width, height=grid_height)

    # Set the specified positions as supercharged
    if supercharged_positions:
        for x, y in supercharged_positions:
            if 0 <= x < grid.width and 0 <= y < grid.height:
                grid.set_supercharged(x, y, True)

    try:
        if solver_choice == "sa":
            sa_params = {
                "initial_temperature": 5000,
                "cooling_rate": 0.999,
                "stopping_temperature": 0.1,
                "iterations_per_temp": 35,
                "initial_swap_probability": 0.55,
                "final_swap_probability": 0.25,
                "start_from_current_grid": False,
                "max_processing_time": 600.0,
            }
            print(f"INFO -- Using Simulated Annealing for {ship_type}/{tech} with params: {sa_params}")
            optimized_grid, optimized_score = simulated_annealing(
                grid, ship_type, ship_modules, tech, player_owned_rewards, solve_type=solve_type or "normal", **sa_params
            )
        elif solver_choice == "refine_training":
            print(f"INFO -- Using refine_placement_for_training for {ship_type}/{tech}")
            optimized_grid, optimized_score = refine_placement_for_training(grid, ship_type, ship_modules, tech)
        else:
            print(f"Error: Unknown solver_choice '{solver_choice}'. Use 'sa' or 'refine_training'.")
            return None, None
        print_grid(optimized_grid)
        return optimized_grid, optimized_score
    except Exception as e:
        print(f"Error generating solve map for {ship_type}/{tech}: {e}")
        return None, None


def generate_solve_map_template(grid):
    """Generates a solve map template from a Grid object."""
    template = {}
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            module_id = cell["module"]
            # <<< Ensure None is represented as the STRING "None" in the template dict >>>
            template[(x, y)] = module_id if module_id is not None else "None"
    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a solve map for a given technology and ship type.")
    # <<< Add ship argument >>>
    parser.add_argument("--ship", type=str, default="standard", help="Ship type key (e.g., 'standard', 'sentinel')")
    parser.add_argument("--tech", type=str, default="infra", help="Technology key (e.g., 'pulse', 'infra')")
    parser.add_argument("--width", type=int, default=3, help="Grid width")
    parser.add_argument("--height", type=int, default=3, help="Grid height")
    parser.add_argument(
        "--rewards",
        type=str,
        nargs="*",
        # <<< Updated default rewards list >>>
        default=["SB", "SP", "TT"],
        help="List of player-owned reward module IDs",
    )
    parser.add_argument(
        "--supercharged",
        type=int,
        nargs="*",
        help="List of supercharged positions as x y pairs (e.g., --supercharged 0 0 1 1)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="sa",
        choices=["sa", "refine_training"],
        help="Solver to use: 'sa' for Simulated Annealing, 'refine_training' for refine_placement_for_training",
    )
    parser.add_argument("--solve_type", type=str, default=None, help="Solve type (e.g., 'max')")
    args = parser.parse_args()

    # <<< Get ship type from args >>>
    ship_type = args.ship
    tech = args.tech
    grid_width = args.width
    grid_height = args.height
    player_owned_rewards = args.rewards
    solver = args.solver

    # Parse supercharged positions from command line arguments
    supercharged_positions = []
    if args.supercharged:
        if len(args.supercharged) % 2 != 0:
            print("Error: Supercharged positions must be provided in x y pairs.")
            exit()
        for i in range(0, len(args.supercharged), 2):
            supercharged_positions.append((args.supercharged[i], args.supercharged[i + 1]))

    # <<< Pass ship_type and solve_type to generate_solve_map >>>
    solve_map, solve_score = generate_solve_map(
        ship_type,
        tech,
        grid_width,
        grid_height,
        player_owned_rewards,
        supercharged_positions=supercharged_positions,
        solver_choice=solver,
        solve_type=args.solve_type,
    )

    if solve_map:
        # <<< Use ship_type in output >>>
        print(f"\nSolve map for {ship_type}/{tech} (solve_type: {args.solve_type}): {solve_score:.2f}")
        print_grid(solve_map)

        # Generate the solve map template from the Grid object
        solve_map_template = generate_solve_map_template(solve_map)
        print("\nSolve Map Template:")
        # <<< Use ship_type in template output >>>
        print(f'    "{ship_type}": {{')
        print(f'        "{tech}": {{')
        print('            "map": {')
        # <<< Ensure correct formatting for "None" string in template output >>>
        for (x, y), module_id in solve_map_template.items():
            # Always wrap the module_id (which is either a real ID or the string "None") in quotes
            module_str = f'"{module_id}"'
            print(f'                "{x},{y}": {module_str},')
        print("            },")
        print(f'            "score": {solve_score:.4f}')
        print("        },")
        print("    },")
