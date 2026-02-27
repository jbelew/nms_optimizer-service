"""
Helper utilities for the optimization module.

Provides helper functions for:
- Window dimension calculation based on module count and technology
- Module placement in empty grid slots
- Grid state queries (empty slot counting, placement verification)

Key Functions:
- determine_window_dimensions(): Calculates optimal window sizes
- place_all_modules_in_empty_slots(): Fills remaining grid spaces
- count_empty_in_localized(): Counts empty slots in a subgrid
- check_all_modules_placed(): Verifies complete module placement
"""

import logging
import json
from typing import Optional, Tuple
from src.grid_utils import Grid
from src.modules_utils import get_tech_modules
from src.module_placement import place_module
from src.modules_utils import get_tech_window_rules, _get_window_profiles  # Moved from inside function


def determine_window_dimensions(
    module_count: int, tech: str, ship: str, modules: Optional[dict] = None
) -> Tuple[int, int]:
    rules = {}
    if module_count < 1:
        logging.warning(f"Module count is {module_count}. Returning default 1x1 window.")
        return 1, 1

    if modules is not None:
        rules = get_tech_window_rules(modules, ship, tech)

    if not rules:
        # Fallback for when modules is not passed or rules not found in modules
        profiles = _get_window_profiles()
        rules = json.loads(json.dumps(profiles.get("standard", {})))

    # The flat map structure maps "count" to "dimensions".
    # e.g. "6": [3, 2], "7": [4, 2].
    count_str = str(module_count)
    if count_str in rules and rules[count_str] is not None:
        return rules[count_str][0], rules[count_str][1]

    int_keys = [int(k) for k in rules.keys() if k.isdigit() and rules[k] is not None]
    larger_keys = [k for k in int_keys if k > module_count]

    if larger_keys:
        best_key = str(min(larger_keys))
        return rules[best_key][0], rules[best_key][1]

    if "default" in rules:
        return rules["default"][0], rules["default"][1]

    # Final safety fallback (should only hit if JSON misses default)
    return 1, 1


def place_all_modules_in_empty_slots(
    grid: Grid,
    modules: dict,
    ship: str,
    tech: str,
    tech_modules: Optional[list] = None,
) -> Grid:
    """
    Places all modules of a given tech into the remaining empty slots.

    Iterates through the grid column-by-column (left-to-right, top-to-bottom),
    placing modules one at a time in the first available empty, active cell.
    Used as a final fallback to ensure all modules are placed.

    Args:
        grid (Grid): The grid to place modules in (modified in-place).
        modules (dict): Module definitions indexed by ship and tech.
        ship (str): Ship identifier (e.g., "corvette", "freighter").
        tech (str): Technology identifier (e.g., "trails", "photon", "pulse").
        tech_modules (Optional[list], optional): Pre-fetched list of module definitions.
                                                If None, fetched from modules dict.
                                                Defaults to None.

    Returns:
        Grid: The modified grid with modules placed. Same reference as input grid.

    Iteration Order:
        - Columns first (x): left to right
        - Rows within each column (y): top to bottom
        - Ensures left-biased, top-biased placement pattern

    Behavior:
        - Skips occupied cells (module is not None)
        - Skips inactive cells (active is False)
        - Places first module with first available cell, second with next, etc.
        - Returns early once all modules are placed
        - Logs warning if insufficient space for all modules

    Side Effects:
        - Modifies grid in-place
        - Logs error if modules cannot be retrieved for tech
        - Logs warning if not all modules can be placed

    Notes:
        - Simple greedy approach (not optimized for score)
        - Used as fallback when better placement strategies aren't available
        - Placement order matters less than ensuring all modules are placed
        - Returns input grid unchanged if no modules found for tech
    """
    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech)
    if tech_modules is None:
        logging.error(f"No modules found for ship: '{ship}' -- tech: '{tech}'")
        return grid

    # Two-pass approach to respect sc_eligible constraint
    # Pass 1: Place non-sc_eligible modules in non-supercharged slots only
    # Pass 2: Place sc_eligible modules in remaining slots (preferring supercharged)

    placed_modules = set()

    # Pass 1: Place non-sc_eligible modules in non-supercharged slots
    for module in tech_modules:
        if module.get("sc_eligible", False):
            continue  # Skip eligible modules in pass 1

        placed = False
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get_cell(x, y)
                if cell["module"] is None and cell["active"] and not cell["supercharged"]:
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
                    placed_modules.add(module["id"])
                    placed = True
                    break
            if placed:
                break

        # If non-sc_eligible module couldn't be placed in non-supercharged slot,
        # fall back to any available slot (including supercharged) as last resort
        if not placed:
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get_cell(x, y)
                    if cell["module"] is None and cell["active"]:
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
                        placed_modules.add(module["id"])
                        placed = True
                        logging.warning(
                            f"Non-sc_eligible module {module['id']} placed in supercharged slot "
                            f"- no non-supercharged slots available (fallback behavior)"
                        )
                        break
                if placed:
                    break

    # Pass 2: Place sc_eligible modules in remaining slots (can use supercharged)
    for module in tech_modules:
        if module["id"] in placed_modules:
            continue  # Already placed

        # sc_eligible modules can use any active slot
        placed = False
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get_cell(x, y)
                if cell["module"] is None and cell["active"]:
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
                    placed_modules.add(module["id"])
                    placed = True
                    break
            if placed:
                break

    if len(placed_modules) < len(tech_modules):
        logging.warning(f"Not enough space to place all modules for ship: '{ship}' -- tech: '{tech}'")

    return grid


def count_empty_in_localized(localized_grid: Grid) -> int:
    """
    Counts unoccupied module slots in a localized grid.

    Iterates through all cells and counts those where no module is placed,
    regardless of active/inactive status. Used to verify grid capacity
    and check space availability for module placement.

    Args:
        localized_grid (Grid): The localized (sub)grid to analyze.

    Returns:
        int: The number of cells with no module (module is None).
             Counts all cells including inactive ones.

    Behavior:
        - Counts every cell where cell["module"] is None
        - Includes both active and inactive empty cells
        - Includes both unoccupied and supercharged slots
        - Does not verify cell accessibility (use with care for inactive cells)

    Notes:
        - Simple cell-by-cell scan (O(width × height))
        - Empty does not necessarily mean available (check active status separately)
        - Used to verify capacity before attempting module placement
        - Returns 0 if all cells are occupied
    """
    count = 0
    for y in range(localized_grid.height):
        for x in range(localized_grid.width):
            cell = localized_grid.get_cell(x, y)
            if cell["module"] is None:  # Only count if the module slot is empty
                count += 1
    return count


def check_all_modules_placed(
    grid: Grid,
    modules: dict,
    ship: str,
    tech: str,
    tech_modules: Optional[list] = None,
) -> bool:
    """
    Verifies that all expected modules for a tech are placed in the grid.

    Compares the set of module IDs found in the grid for a specific tech against
    the set of expected module IDs for that tech. Returns True only if they match exactly.

    Args:
        grid (Grid): The grid layout to check.
        modules (dict): Module definitions indexed by ship and tech.
        ship (str): Ship identifier (e.g., "corvette", "freighter").
        tech (str): Technology identifier (e.g., "trails", "photon", "pulse").
        tech_modules (Optional[list], optional): Pre-fetched list of module definitions.
                                                If None, fetched from modules dict.
                                                Defaults to None.

    Returns:
        bool: True if all expected modules are placed and no duplicates exist.
              False if:
              - Some expected modules are missing from grid
              - Extra modules are placed (different set)
              - Modules could not be retrieved for tech

    Algorithm:
        1. Fetch expected module IDs for (ship, tech) pair
        2. Scan grid and collect all placed module IDs for matching tech
        3. Compare sets for exact equality

    Edge Cases:
        - Returns False if modules dict is empty or tech has no modules (logs warning)
        - Returns True if tech has zero expected modules (vacuous truth)
        - Only counts modules with matching tech (ignores other techs)

    Side Effects:
        - Logs warning if modules cannot be retrieved
        - Does not modify grid

    Notes:
        - Requires exact set match (not subset)
        - Does not verify placement quality or position
        - Only checks presence/absence, not count per cell
        - Useful for validation before/after optimization
        - Used in testing and finalization steps
    """
    # If a specific list of modules to check against isn't provided, fetch it.
    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech)

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
