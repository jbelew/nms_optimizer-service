"""
Windowing module for localized grid optimization.

This module provides functions for identifying and extracting localized regions
(windows/opportunities) within a larger grid for focused optimization. It includes:
- Window scanning and opportunity detection
- Localized grid creation (preserving other tech modules)
- Window scoring based on supercharged slots and adjacency

Key Functions:
- find_supercharged_opportunities(): Scans grid for high-value windows
- _scan_grid_with_window(): Helper for window-based grid scanning
- calculate_window_score(): Evaluates window quality
- create_localized_grid(): Extracts subgrid for SA/permutation refinement
- create_localized_grid_ml(): Extracts subgrid for ML-based refinement
"""

import logging
from copy import deepcopy
from typing import Optional, Tuple

from src.grid_utils import Grid
from src.modules_utils import get_tech_modules
from src.module_placement import clear_all_modules_of_tech
from .helpers import determine_window_dimensions


def _scan_grid_with_window(
    grid_copy: Grid,
    window_width: int,
    window_height: int,
    module_count: int,
    tech: str,
    require_supercharge: bool = True,
) -> Tuple[float, Optional[Tuple[int, int]]]:
    """
    Scans a grid with a fixed window size to find the best placement opportunity.

    Slides a window of specified dimensions across the entire grid, evaluating each
    position based on available supercharged slots and empty cells. Optionally
    filters for windows containing at least one supercharged slot.

    Args:
        grid_copy (Grid): The grid to scan (typically with target tech cleared).
        window_width (int): Width of the scanning window in cells.
        window_height (int): Height of the scanning window in cells.
        module_count (int): Number of modules to place (used for availability check).
        tech (str): Technology identifier being optimized.
        require_supercharge (bool, optional): If True, only considers windows with
                                             at least one available supercharged slot.
                                             Defaults to True.

    Returns:
        Tuple[float, Optional[Tuple[int, int]]]: A tuple containing:
            - best_score (float): The best score found (-1 if invalid window size or no match)
            - best_start_pos (Tuple[int, int] | None): (x, y) of best window's top-left corner,
                                                       or None if no valid window found

    Validation:
        - Window dimensions must fit within grid bounds
        - Window must have at least module_count available cells
        - If require_supercharge=True, must have ≥1 unoccupied supercharged slot

    Scoring:
        - Higher score indicates better placement opportunity
        - Based on supercharged slot availability and empty cell count
        - Calculated via calculate_window_score()

    Notes:
        - Returns (-1, None) if window is too large for grid
        - Iteration order is top-to-bottom, left-to-right
        - Empty cells outside the window are not considered
    """
    best_score = -1
    best_start_pos = None

    # Check if window dimensions are valid for the grid size
    if window_width > grid_copy.width or window_height > grid_copy.height:
        logging.warning(
            f"Window size ({window_width}x{window_height}) is larger than grid ({grid_copy.width}x{grid_copy.height}). Cannot scan with this size."
        )
        return -1, None

    # Iterate through possible top-left corners for the window
    for start_y in range(grid_copy.height - window_height + 1):
        for start_x in range(grid_copy.width - window_width + 1):
            # Create a temporary window grid reflecting the current slice
            window_grid = Grid(window_width, window_height)
            for y in range(window_height):
                for x in range(window_width):
                    grid_x = start_x + x
                    grid_y = start_y + y
                    # Bounds check (should be handled by outer loops, but safety)
                    if 0 <= grid_x < grid_copy.width and 0 <= grid_y < grid_copy.height:
                        cell = grid_copy.get_cell(grid_x, grid_y)
                        # Copy relevant data to the window grid cell
                        window_grid.cells[y][x]["active"] = cell["active"]
                        window_grid.cells[y][x]["supercharged"] = cell["supercharged"]
                        window_grid.cells[y][x]["module"] = cell["module"]  # Keep module info for checks
                        window_grid.cells[y][x]["tech"] = cell["tech"]
                    else:
                        window_grid.cells[y][x]["active"] = False  # Mark as inactive if out of bounds

            if require_supercharge:
                # Check if the window has at least one available supercharged slot
                has_available_supercharged = False
                for y in range(window_height):
                    for x in range(window_width):
                        cell = window_grid.get_cell(x, y)
                        if cell["supercharged"] and cell["module"] is None and cell["active"]:
                            has_available_supercharged = True
                            break
                    if has_available_supercharged:
                        break
                if not has_available_supercharged:
                    continue  # Skip this window

            # Check if the number of available cells in the current window is sufficient
            available_cells_in_window = 0
            for y in range(window_height):
                for x in range(window_width):
                    cell = window_grid.get_cell(x, y)
                    if cell["active"] and cell["module"] is None:
                        available_cells_in_window += 1
            if available_cells_in_window < module_count:
                continue  # Skip this window

            # Calculate score and update best if this window is better
            window_score = calculate_window_score(window_grid, tech, grid_copy, start_x, start_y)
            if window_score > best_score:
                best_score = window_score
                best_start_pos = (start_x, start_y)

    return best_score, best_start_pos


def find_supercharged_opportunities(
    grid: Grid,
    modules: dict,
    ship: str,
    tech: str,
    tech_modules: Optional[list] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Finds the highest-scoring window with available supercharged slots.

    Dynamically sizes windows based on module count, scans with both original and
    rotated dimensions (if non-square), and returns the position and size of the
    best window. Falls back to non-supercharged windows if no supercharged
    opportunities exist.

    Args:
        grid (Grid): The current grid layout.
        modules (dict): Module definitions indexed by ship and tech.
        ship (str): Ship identifier (e.g., "corvette", "freighter").
        tech (str): Technology identifier (e.g., "trails", "photon", "pulse").
        tech_modules (Optional[list], optional): Pre-fetched list of module definitions.
                                                If None, fetched from modules dict.
                                                Defaults to None.

    Returns:
        Optional[Tuple[int, int, int, int]]: A tuple (opportunity_x, opportunity_y, width, height)
                                            representing:
                                            - opportunity_x: X-coordinate of top-left corner
                                            - opportunity_y: Y-coordinate of top-left corner
                                            - width: Window width in cells
                                            - height: Window height in cells
                                            Returns None if:
                                            - No modules found for tech
                                            - All supercharged slots are occupied
                                            - No suitable window found even as fallback

    Window Sizing:
        - Dynamically determined by determine_window_dimensions() based on module_count
        - Non-square windows are scanned in both orientations (original + rotated)
        - Rotated dimensions use (height, width) instead of (width, height)

    Scoring Strategy:
        1. Prefer windows with available supercharged slots (require_supercharge=True)
        2. Compare original vs rotated dimensions and select best score
        3. If no supercharged window found, retry without supercharge requirement
        4. Return position and dimensions of overall best window

    Notes:
        - If rotated dimensions equal original (square window), rotation is skipped
        - Logs selection rationale (which dimensions were best)
        - Returns None if grid has no unoccupied supercharged slots AND no fallback window
        - Clears target tech modules before scanning to evaluate placement potential
    """
    grid_copy = grid.copy()
    # Clear the target tech modules to evaluate potential placement areas
    clear_all_modules_of_tech(grid_copy, tech)

    # Determine Dynamic Window Size (needed first to check sc_eligible modules)
    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech)
    if tech_modules is None:
        logging.error(f"No modules found for ship '{ship}' and tech '{tech}' in find_supercharged_opportunities.")
        return None
    module_count = len(tech_modules)
    
    # Check if all modules are non-sc_eligible
    all_non_sc_eligible = all(not m.get("sc_eligible", False) for m in tech_modules)
    
    # Check if there are any unoccupied supercharged slots
    unoccupied_supercharged_slots = False
    for y in range(grid_copy.height):
        for x in range(grid_copy.width):
            cell = grid_copy.get_cell(x, y)
            if cell["supercharged"] and cell["module"] is None and cell["active"]:
                unoccupied_supercharged_slots = True
                break
        if unoccupied_supercharged_slots:
            break
    
    # If all modules are non-sc_eligible, we can't use supercharged windows
    if all_non_sc_eligible and unoccupied_supercharged_slots:
        logging.info("All modules are non-sc_eligible. Skipping supercharged window search, looking for regular active windows.")
        unoccupied_supercharged_slots = False
    
    if not unoccupied_supercharged_slots:
        logging.info("No suitable supercharged windows found or all modules are non-sc_eligible.")
        return None
    window_width, window_height = determine_window_dimensions(module_count, tech, ship)
    logging.info(f"Using dynamic window size {window_width}x{window_height} for {tech} ({module_count} modules).")

    # --- Scan with Original Dimensions ---
    best_score1, best_pos1 = _scan_grid_with_window(grid_copy, window_width, window_height, module_count, tech)

    # --- Scan with Rotated Dimensions (if needed) ---
    best_score2 = -1
    best_pos2 = None
    rotated_needed = window_width != window_height  # Check if width and height are different
    rotated_width, rotated_height = 0, 0  # Initialize for print statement clarity

    if rotated_needed:
        rotated_width, rotated_height = window_height, window_width  # Swap dimensions
        # print(f"INFO -- Also checking rotated window size {rotated_width}x{rotated_height}.")
        best_score2, best_pos2 = _scan_grid_with_window(grid_copy, rotated_width, rotated_height, module_count, tech)

    # --- Compare Results and Determine Best Dimensions ---
    overall_best_score = -1
    overall_best_pos = None
    overall_best_width = 0  # <<< Store best width
    overall_best_height = 0  # <<< Store best height

    # Check original scan result
    if best_score1 > overall_best_score:
        overall_best_score = best_score1
        overall_best_pos = best_pos1
        overall_best_width = window_width  # <<< Store original dimensions
        overall_best_height = window_height  # <<< Store original dimensions

    # Check rotated scan result (if performed)
    if best_score2 > overall_best_score:
        overall_best_score = best_score2
        overall_best_pos = best_pos2
        overall_best_width = rotated_width  # <<< Store rotated dimensions
        overall_best_height = rotated_height  # <<< Store rotated dimensions
        logging.info(
            f"Rotated window ({rotated_width}x{rotated_height}) provided a better score ({overall_best_score:.2f})."
        )
    elif best_score1 > -1 and rotated_needed:  # Only print if original scan found something and rotation was checked
        logging.info(
            f"Original window ({window_width}x{window_height}) provided the best score ({overall_best_score:.2f})."
        )
    elif best_score1 > -1 and not rotated_needed:  # Square window case
        logging.info(f"Best score found with square window ({window_width}x{window_height}): {overall_best_score:.2f}.")

    # --- Fallback: If no supercharged window found, try again without requiring supercharge ---
    # This block attempts to find a suitable window even if it doesn't contain a supercharged slot.
    # To disable this fallback, comment out or remove the entire 'if overall_best_pos is None:' block below.
    if overall_best_pos is None:
        logging.info("No supercharged window found. Retrying without supercharge requirement.")
        best_score1, best_pos1 = _scan_grid_with_window(
            grid_copy, window_width, window_height, module_count, tech, require_supercharge=False
        )
        if rotated_needed:
            best_score2, best_pos2 = _scan_grid_with_window(
                grid_copy, rotated_width, rotated_height, module_count, tech, require_supercharge=False
            )

        # Re-compare scores
        if best_score1 > overall_best_score:
            overall_best_score = best_score1
            overall_best_pos = best_pos1
            overall_best_width = window_width
            overall_best_height = window_height

        if best_score2 > overall_best_score:
            overall_best_score = best_score2
            overall_best_pos = best_pos2
            overall_best_width = rotated_width
            overall_best_height = rotated_height

    # --- Return the Overall Best Result ---
    if overall_best_pos is not None:
        best_x, best_y = overall_best_pos
        # logging.info(f"Best opportunity window found starting at: ({best_x}, {best_y}) with dimensions {overall_best_width}x{overall_best_height}")
        # <<< Return position AND dimensions >>>
        return best_x, best_y, overall_best_width, overall_best_height
    else:
        logging.info(f"No suitable opportunity window found for {tech} after scanning (original and rotated).")
        return None


def calculate_window_score(
    window_grid: Grid, tech: str, full_grid: Optional[Grid] = None, window_start_x: int = 0, window_start_y: int = 0
) -> float:
    """
    Calculates a quality score for a placement window.

    Evaluates window desirability based on supercharged slot availability, empty
    cells, and adjacency to existing modules. Prioritizes supercharged slots while
    penalizing edge placements. For single-cell windows, adjacency bonus is applied.

    Args:
        window_grid (Grid): The window being evaluated.
        tech (str): Technology identifier being optimized.
        full_grid (Optional[Grid], optional): Complete grid for adjacency context.
                                             Required for single-module windows.
                                             Defaults to None.
        window_start_x (int, optional): X-coordinate of window's top-left in full grid.
                                       Defaults to 0.
        window_start_y (int, optional): Y-coordinate of window's top-left in full grid.
                                       Defaults to 0.

    Returns:
        float: A numeric score representing window quality. Higher = better.
               Score components:
               - Supercharged slots: multiplied by 3
               - Edge penalty: -0.25 per horizontal edge supercharged slot
               - Empty cells (no supercharge): multiplied by 1
               - Adjacency (single cells): bonus per neighboring module

    Scoring Logic:
        1. Count active supercharged slots (empty or current tech)
        2. Count empty cells (unoccupied, active)
        3. Count edge penalties (supercharged on left/right edge of window)
        4. If no supercharged slots:
           - Single cells: prioritize adjacency
           - Multi-cell: use empty count
        5. If supercharged slots exist: return supercharged_count * 3

    Adjacency Bonus (Single-Cell Windows):
        - Applied when window_grid.width == 1 and window_grid.height == 1
        - +3.0 points for each adjacent cell with a module in full_grid
        - Only evaluated when full_grid is provided

    Notes:
        - Only counts active cells (inactive cells are completely ignored)
        - Edge penalty applies to supercharged cells on window edges (not corner edges)
        - Supercharged scoring dominates (ignores empty_count in that case)
        - Returns 0 if window contains no active cells
        - Used for both opportunity scanning and window selection
    """
    supercharged_count = 0
    empty_count = 0
    edge_penalty = 0
    adjacency_score = 0

    for y in range(window_grid.height):
        for x in range(window_grid.width):
            cell = window_grid.get_cell(x, y)
            if cell["active"]:  # Only consider active cells
                if cell["supercharged"]:
                    # Check if the supercharged cell is empty or occupied by the current tech
                    if cell["module"] is None or cell["tech"] == tech:
                        supercharged_count += 1
                        # Check if the supercharged slot is on the horizontal edge of the window
                    if window_grid.width > 1 and (x == 0 or x == window_grid.width - 1):
                        edge_penalty += 1
                if cell["module"] is None:
                    empty_count += 1

                    # For single-module placement (1x1 window), score based on adjacency to existing modules
                    if window_grid.width == 1 and window_grid.height == 1 and full_grid is not None:
                        # Get actual grid coordinates
                        grid_x = window_start_x + x
                        grid_y = window_start_y + y

                        # Check neighbors in full grid
                        adjacent_positions = [
                            (grid_x - 1, grid_y),
                            (grid_x + 1, grid_y),
                            (grid_x, grid_y - 1),
                            (grid_x, grid_y + 1),
                        ]
                        for adj_x, adj_y in adjacent_positions:
                            if 0 <= adj_x < full_grid.width and 0 <= adj_y < full_grid.height:
                                adj_cell = full_grid.get_cell(adj_x, adj_y)
                                if adj_cell["module"] is not None:
                                    adjacency_score += 3.0  # Weight for adjacency to other modules

    if supercharged_count > 0:
        return supercharged_count * 3  # + (empty_count * 1)
    else:
        # For single cells with no supercharge, prioritize adjacency
        if window_grid.width == 1 and window_grid.height == 1:
            return empty_count * 1 + adjacency_score  # Adjacency bonus dominates for single modules
        return (supercharged_count * 3) + (empty_count * 1) + (edge_penalty * 0.25)


def create_localized_grid(
    grid: Grid, opportunity_x: int, opportunity_y: int, tech: str, localized_width: int, localized_height: int
) -> Tuple[Grid, int, int]:
    """
    Creates a localized grid for SA/permutation-based refinement.

    Extracts a rectangular subgrid around a specified opportunity, clamping to grid
    bounds if necessary. Preserves all module data including other tech types.
    Used for windowed simulated annealing and permutation-based optimization.

    Args:
        grid (Grid): The main grid to extract from.
        opportunity_x (int): X-coordinate (column) of window's top-left corner.
        opportunity_y (int): Y-coordinate (row) of window's top-left corner.
        tech (str): Technology being optimized (for reference, not filtering).
        localized_width (int): Desired width of extracted window in cells.
        localized_height (int): Desired height of extracted window in cells.

    Returns:
        Tuple[Grid, int, int]: A tuple containing:
            - localized_grid (Grid): Extracted subgrid with all modules preserved
            - start_x (int): Actual X-coordinate of extraction (may differ if clamped)
            - start_y (int): Actual Y-coordinate of extraction (may differ if clamped)

    Clamping Behavior:
        - start_x/start_y clamped to [0, grid.width-1] and [0, grid.height-1]
        - end coordinates clamped to grid bounds
        - Resulting localized_grid may be smaller than requested if near grid edges

    Data Preservation:
        - All module properties copied for existing modules
        - Other tech modules are preserved (not removed)
        - Supercharged status preserved
        - Active/inactive status preserved
        - All cell attributes copied (bonus, adjacency, type, etc.)

    Uses:
        - SA/permutation refinement: works with all tech modules in window
        - Window sizing: actual extracted dimensions returned for reference

    Notes:
        - Does NOT clear other tech modules (unlike create_localized_grid_ml)
        - Does NOT mark other tech cells as inactive
        - Actual grid dimensions may be smaller than requested if window extends beyond bounds
        - start_x and start_y indicate where extraction started (for coordinate mapping)
    """
    # <<< Remove hardcoded dimensions >>>
    # localized_width = 4
    # localized_height = 3

    # Directly use opportunity_x and opportunity_y as the starting position
    start_x = opportunity_x
    start_y = opportunity_y

    # Clamp the starting position to ensure it's within the grid bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    # Calculate the end position based on the clamped start position and desired dimensions
    end_x = min(grid.width, start_x + localized_width)
    end_y = min(grid.height, start_y + localized_height)

    # Adjust the localized grid size based on the clamped bounds
    actual_localized_width = end_x - start_x
    actual_localized_height = end_y - start_y

    # Create the localized grid with the actual calculated size
    localized_grid = Grid(actual_localized_width, actual_localized_height)

    # Copy the grid structure and module data (logic remains the same)
    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            localized_x = x - start_x
            localized_y = y - start_y
            cell = grid.get_cell(x, y)
            localized_grid.cells[localized_y][localized_x]["active"] = cell["active"]
            localized_grid.cells[localized_y][localized_x]["supercharged"] = cell["supercharged"]

            # Copy module data if a module exists
            if cell["module"] is not None:
                localized_grid.cells[localized_y][localized_x]["module"] = cell["module"]
                localized_grid.cells[localized_y][localized_x]["label"] = cell["label"]
                localized_grid.cells[localized_y][localized_x]["tech"] = cell["tech"]
                localized_grid.cells[localized_y][localized_x]["type"] = cell["type"]
                localized_grid.cells[localized_y][localized_x]["bonus"] = cell["bonus"]
                localized_grid.cells[localized_y][localized_x]["adjacency"] = cell["adjacency"]
                localized_grid.cells[localized_y][localized_x]["sc_eligible"] = cell["sc_eligible"]
                localized_grid.cells[localized_y][localized_x]["image"] = cell["image"]
                if "module_position" in cell:
                    localized_grid.cells[localized_y][localized_x]["module_position"] = cell["module_position"]

    return localized_grid, start_x, start_y


def create_localized_grid_ml(
    grid: Grid, opportunity_x: int, opportunity_y: int, tech: str, localized_width: int, localized_height: int
) -> Tuple[Grid, int, int, dict]:
    """
    Creates a localized grid for ML-based refinement.

    Extracts a rectangular subgrid around a specified opportunity, removing modules
    of other tech types and marking their cells as inactive. This creates an isolated
    environment for ML optimization while preserving original state for restoration.

    Args:
        grid (Grid): The main grid to extract from.
        opportunity_x (int): X-coordinate (column) of window's top-left corner.
        opportunity_y (int): Y-coordinate (row) of window's top-left corner.
        tech (str): Technology being optimized (modules of this tech are preserved).
        localized_width (int): Desired width of extracted window in cells.
        localized_height (int): Desired height of extracted window in cells.

    Returns:
        Tuple[Grid, int, int, dict]: A tuple containing:
            - localized_grid (Grid): Extracted subgrid with other tech removed
            - start_x (int): Actual X-coordinate of extraction (may differ if clamped)
            - start_y (int): Actual Y-coordinate of extraction (may differ if clamped)
            - original_state_map (dict): Mapping {(x, y): original_cell_data} for cells
                                        that were modified (other tech modules removed)

    Clamping Behavior:
        - start_x/start_y clamped to [0, grid.width-1] and [0, grid.height-1]
        - end coordinates clamped to grid bounds
        - Resulting localized_grid may be smaller than requested if near grid edges

    Cell Handling:
        1. **Target Tech Modules**: Preserved as-is
        2. **Other Tech Modules**: Removed, cell marked inactive, original state saved
        3. **Inactive Cells**: Preserved as inactive (empty)
        4. **Empty Active Cells**: Copied as-is (available for placement)
        5. **Supercharged Status**: Always preserved

    Original State Restoration:
        - original_state_map contains deepcopy of main grid cells that were modified
        - Used by refinement handlers to restore grid after ML processing
        - Only includes cells with other tech modules (not inactive cells)

    ML Isolation:
        - Other tech modules are removed to give ML model pure search space
        - Cells with other tech are marked inactive (unavailable for new modules)
        - Allows ML to focus optimization on target tech placement

    Notes:
        - Differs from create_localized_grid by removing other tech modules
        - The deepcopy in original_state_map ensures safe restoration
        - Inactive cells are NOT added to original_state_map (not modified)
        - Used exclusively by _handle_ml_opportunity() for refinement
    """
    # <<< Remove hardcoded dimensions >>>
    # localized_width = 4
    # localized_height = 3

    # Directly use opportunity_x and opportunity_y as the starting position
    start_x = opportunity_x
    start_y = opportunity_y

    # Clamp the starting position to ensure it's within the grid bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    # Calculate the end position based on the clamped start position and desired dimensions
    end_x = min(grid.width, start_x + localized_width)
    end_y = min(grid.height, start_y + localized_height)

    # Adjust the localized grid size based on the clamped bounds
    actual_localized_width = end_x - start_x
    actual_localized_height = end_y - start_y

    # Create the localized grid with the actual calculated size
    localized_grid = Grid(actual_localized_width, actual_localized_height)
    original_state_map = {}  # To store original state of modified cells

    # Copy the grid structure and module data, modifying for other techs (logic remains the same)
    for y_main in range(start_y, end_y):
        for x_main in range(start_x, end_x):
            localized_x = x_main - start_x
            localized_y = y_main - start_y
            main_cell = grid.get_cell(x_main, y_main)
            local_cell = localized_grid.get_cell(localized_x, localized_y)

            # Always copy basic structure like supercharged status
            local_cell["supercharged"] = main_cell["supercharged"]

            # Check the module and its tech in the main grid
            if main_cell["module"] is not None and main_cell["tech"] != tech:
                # --- Other Tech Found ---
                original_state_map[(x_main, y_main)] = deepcopy(main_cell)
                local_cell["module"] = None
                local_cell["label"] = ""
                local_cell["tech"] = None
                local_cell["type"] = ""
                local_cell["bonus"] = 0.0
                local_cell["adjacency"] = False
                local_cell["sc_eligible"] = False
                local_cell["image"] = None
                local_cell["total"] = 0.0
                local_cell["adjacency_bonus"] = 0.0
                if "module_position" in local_cell:
                    del local_cell["module_position"]
                local_cell["active"] = False
            elif not main_cell["active"]:
                # --- Inactive Cell in Main Grid ---
                local_cell["active"] = False
                local_cell["module"] = None
                local_cell["label"] = ""
                local_cell["tech"] = None
                local_cell["type"] = ""
                local_cell["bonus"] = 0.0
                local_cell["adjacency"] = False
                local_cell["sc_eligible"] = False
                local_cell["image"] = None
                local_cell["total"] = 0.0
                local_cell["adjacency_bonus"] = 0.0
                if "module_position" in local_cell:
                    del local_cell["module_position"]
            else:
                # --- Target Tech, Empty Active Cell ---
                local_cell.update(deepcopy(main_cell))
                local_cell["active"] = True

    return localized_grid, start_x, start_y, original_state_map
