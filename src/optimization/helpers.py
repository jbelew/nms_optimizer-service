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
from typing import Optional, Tuple
from src.grid_utils import Grid
from src.modules_utils import get_tech_modules
from src.module_placement import place_module


def determine_window_dimensions(module_count: int, tech: str, ship: str) -> Tuple[int, int]:
    """
    Calculates optimal window dimensions for module placement optimization.

    Uses a hierarchical strategy to determine the window size for localized grid
    operations, prioritizing ship/tech-specific overrides, then tech-specific rules,
    then generic fallbacks based on module count.

    Args:
        module_count (int): Total number of modules for the technology.
                           Used to determine window capacity.
        tech (str): Technology identifier (e.g., "trails", "photon", "pulse", "hyper").
                   Used for tech-specific sizing rules.
        ship (str): Ship identifier (e.g., "corvette", "freighter", "sentinel").
                   Used for ship-specific overrides.

    Returns:
        Tuple[int, int]: A tuple (window_width, window_height) representing
                        the optimal dimensions for localized windows.
                        Typical values range from 1×1 to 4×4.

    Priority Order:
        1. Ship + tech + module_count specific overrides (highest priority)
        2. Ship + module_count overrides
        3. Tech-specific rules (varies by tech)
        4. Generic fallbacks based on module_count only (lowest priority)

    Special Cases:
        - **sentinel + photonix**: Always 4×3
        - **corvette + pulse + 7 modules**: 4×2
        - **corvette + 7-8 modules**: 3×3
        - **hyper tech**: Scaled by module_count (1×1 to 4×4)
        - **bolt-caster**: Always 4×3
        - **pulse-spitter, jetpack**: 3×3 or 4×2 depending on count
        - **pulse**: Varies 3×2 to 4×3 depending on count

    Fallback Rules:
        - module_count < 1: 1×1 (warning logged)
        - module_count < 3: 2×1
        - module_count 3-9: Scales 3×1 to 3×3
        - module_count ≥10: 4×3

    Notes:
        - Windows are typically square or near-square for balanced coverage
        - Larger windows accommodate more modules for parallel evaluation
        - Width generally ≥ height for better module distribution
        - Default is 3×3 for most generic cases
    """
    # --- Ship- and tech-specific overrides ---
    if ship == "sentinel" and tech == "photonix":
        return 4, 3

    if ship == "corvette" and tech == "pulse" and module_count == 7:
        return 4, 2

    if ship == "corvette" and module_count in (7, 8):
        return 3, 3

    # Set 8 modules to 3x3 unless tech has specific sizing rules
    if module_count == 8 and tech not in ("pulse", "photonix", "hyper", "pulse-spitter"):
        return 3, 3

    # Default window size if no other conditions are met
    window_width, window_height = 3, 3

    # --- Technology-specific rules ---
    if tech == "hyper":
        if module_count >= 12:
            window_width, window_height = 4, 4
        elif module_count >= 10:
            window_width, window_height = 4, 3
        elif module_count >= 9:
            window_width, window_height = 3, 3
        else:
            window_width, window_height = 4, 2

    elif tech == "bolt-caster":
        window_width, window_height = 4, 3

    elif tech == "jetpack":
        window_width, window_height = 3, 3

    elif tech == "neutron":
        window_width, window_height = 3, 3

    elif tech == "pulse-spitter":
        if module_count < 7:
            window_width, window_height = 3, 3
        else:
            window_width, window_height = 4, 2

    elif tech == "pulse":
        if module_count == 6:
            window_width, window_height = 3, 2
        elif module_count < 9:
            window_width, window_height = 4, 2
        else:
            window_width, window_height = 4, 3

    # --- Generic fallback rules ---
    elif module_count < 1:
        logging.warning(f"Module count is {module_count}. Returning default 1x1 window.")
        return 1, 1
    elif module_count < 3:
        window_width, window_height = 2, 1
    elif module_count < 5:
        window_width, window_height = 2, 2
    elif module_count < 7:
        window_width, window_height = 3, 2
    elif module_count == 7:
        window_width, window_height = 4, 2
    elif module_count <= 9:
        window_width, window_height = 3, 3
    else:  # module_count >= 10
        window_width, window_height = 4, 3

    return window_width, window_height


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
