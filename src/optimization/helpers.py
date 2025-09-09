# optimization/helpers.py
import logging
from typing import Optional
from src.modules_utils import get_tech_modules
from src.module_placement import place_module


def determine_window_dimensions(module_count: int, tech: str, solve_type: str | None = None) -> tuple[int, int]:
    """
    Determines the window width and height based on the number of modules.

    This defines the size of the grid window used for placement calculations.

    Args:
        module_count: The total number of modules for a given technology.

    Returns:
        A tuple containing the calculated window_width and window_height.
    """
    window_width, window_height = 3, 3

    if module_count < 1:
        # Handle cases with zero or negative modules (optional, but good practice)
        logging.warning(f"Module count is {module_count}. Returning default 1x1 window.")
        return 1, 1
    elif module_count < 3:
        window_width, window_height = 1, 2
    elif module_count < 4:
        window_width, window_height = 1, 3
    elif module_count < 7:
        window_width, window_height = 2, 3
    elif module_count < 8 and (tech == "pulse-spitter" or tech == "jetpack"):
        window_width, window_height = 3, 3
    elif module_count < 8:
        window_width, window_height = 4, 2
    elif module_count < 9:
        window_width, window_height = 4, 2
    elif module_count < 10 and tech == "bolt-caster":
        window_width, window_height = 4, 3
    elif module_count < 10:
        window_width, window_height = 3, 3
    elif module_count >= 10 and tech == "hyper":
        window_width, window_height = 4, 3
    elif module_count >= 10:
        window_width, window_height = 4, 3

    return window_width, window_height


def place_all_modules_in_empty_slots(
    grid,
    modules,
    ship,
    tech,
    player_owned_rewards=None,
    solve_type: Optional[str] = None,
    tech_modules=None,
):
    """Places all modules of a given tech in any remaining empty slots, going column by column."""
    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards, solve_type=solve_type)
    if tech_modules is None:
        logging.error(f"No modules found for ship: '{ship}' -- tech: '{tech}'")
        return grid

    module_index = 0  # Keep track of the current module to place

    for x in range(grid.width):  # Iterate through columns first
        for y in range(grid.height):  # Then iterate through rows
            if module_index >= len(tech_modules):
                return grid  # All modules placed, exit early

            if grid.get_cell(x, y)["module"] is None and grid.get_cell(x, y)["active"]:
                module = tech_modules[module_index]
                place_module(
                    grid,
                    x,
                    y,
                    module["id"],
                    module["label"],
                    tech,
                    module["type"],
                    module["bonus"],
                    module["adjacency"],
                    module["sc_eligible"],
                    module["image"],
                )
                module_index += 1  # Move to the next module

    if module_index < len(tech_modules) and len(tech_modules) > 0:
        logging.warning(f"Not enough space to place all modules for ship: '{ship}' -- tech: '{tech}'")

    return grid


def count_empty_in_localized(localized_grid):
    """Counts the number of truly empty slots in a localized grid."""
    count = 0
    for y in range(localized_grid.height):
        for x in range(localized_grid.width):
            cell = localized_grid.get_cell(x, y)
            if cell["module"] is None:  # Only count if the module slot is empty
                count += 1
    return count


def check_all_modules_placed(
    grid,
    modules,
    ship,
    tech,
    player_owned_rewards=None,
    solve_type: Optional[str] = None,
    tech_modules=None,
):
    """
    Checks if all expected modules for a given tech have been placed in the grid.

    Args:
        grid (Grid): The grid layout.
        modules (dict): The module data.
        ship (str): The ship type.
        tech (str): The technology type.
        player_owned_rewards (list, optional): Rewards owned by the player. Defaults to None.
        solve_type (str, optional): The type of solve, e.g., "normal" or "max".
        tech_modules (list, optional): A pre-fetched list of modules to check against.
                                       If provided, this list is used directly.

    Returns:
        bool: True if all modules are placed, False otherwise.
    """
    # If a specific list of modules to check against isn't provided, fetch it.
    if tech_modules is None:
        if player_owned_rewards is None:
            player_owned_rewards = []  # Ensure it's an empty list if None

        tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards, solve_type=solve_type)

    if tech_modules is None:
        logging.warning(
            f"check_all_modules_placed (opt_alg) - Could not get expected modules for {ship}/{tech}. Assuming not all modules are placed."
        )
        return False  # If modules for the tech couldn't be retrieved, assume they aren't all placed.

    if not tech_modules:  # Handles empty list case (no modules defined for this tech)
        return True  # All zero modules are considered placed.

    placed_module_ids = set()
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech and cell["module"]:
                placed_module_ids.add(cell["module"])

    expected_module_ids = {module["id"] for module in tech_modules}

    return placed_module_ids == expected_module_ids
