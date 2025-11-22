# optimization/windowing.py
import logging
from copy import deepcopy

from src.grid_utils import Grid
from src.modules_utils import get_tech_modules
from src.module_placement import clear_all_modules_of_tech
from .helpers import determine_window_dimensions


def _scan_grid_with_window(
    grid_copy, window_width, window_height, module_count, tech, require_supercharge: bool = True
):
    """
    Helper function to scan the grid with a specific window size and find the best opportunity.

    Args:
        grid_copy (Grid): A copy of the main grid (with target tech cleared).
        window_width (int): The width of the scanning window.
        window_height (int): The height of the scanning window.
        module_count (int): The number of modules for the tech.
        tech (str): The technology key.

    Returns:
        tuple: (best_score, best_start_pos) where best_start_pos is (x, y) or None.
               Returns (-1, None) if the window size is invalid for the grid.
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
    grid,
    modules,
    ship,
    tech,
    tech_modules=None,
):
    """
    Scans the entire grid with a sliding window (dynamically sized, including rotation
    if non-square) to find the highest-scoring window containing available
    supercharged slots.

    Args:
        grid (Grid): The current grid layout.
        modules (dict): The module data.
        ship (str): The ship type.
        tech (str): The technology type.

    Returns:
        tuple or None: A tuple (opportunity_x, opportunity_y, best_width, best_height)
                       representing the top-left corner and dimensions of the best window,
                       or None if no suitable window is found or if all supercharged slots
                       are occupied.
    """
    grid_copy = grid.copy()
    # Clear the target tech modules to evaluate potential placement areas
    clear_all_modules_of_tech(grid_copy, tech)

    # Check if there are any unoccupied supercharged slots (no change needed)
    unoccupied_supercharged_slots = False
    for y in range(grid_copy.height):
        for x in range(grid_copy.width):
            cell = grid_copy.get_cell(x, y)
            if cell["supercharged"] and cell["module"] is None and cell["active"]:
                unoccupied_supercharged_slots = True
                break
        if unoccupied_supercharged_slots:
            break
    if not unoccupied_supercharged_slots:
        logging.info("No unoccupied supercharged slots found.")
        return None

    # Determine Dynamic Window Size (no change needed)
    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech)
    if tech_modules is None:
        logging.error(f"No modules found for ship '{ship}' and tech '{tech}' in find_supercharged_opportunities.")
        return None
    module_count = len(tech_modules)
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


def calculate_window_score(window_grid, tech, full_grid=None, window_start_x=0, window_start_y=0):
     """
     Calculates a score for a given window based on supercharged and empty slots,
     excluding inactive cells. Prioritizes supercharged slots away from the horizontal edges of the window.
     For single-cell windows, also considers adjacency to existing modules in the full grid.
     
     Args:
         window_grid (Grid): The window being evaluated
         tech (str): The technology type
         full_grid (Grid): Optional full grid for adjacency scoring in single-module placement
         window_start_x (int): Starting x coordinate of window in full grid
         window_start_y (int): Starting y coordinate of window in full grid
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
                         adjacent_positions = [(grid_x - 1, grid_y), (grid_x + 1, grid_y), (grid_x, grid_y - 1), (grid_x, grid_y + 1)]
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


def create_localized_grid(grid, opportunity_x, opportunity_y, tech, localized_width, localized_height):
    """
    Creates a localized grid around a given opportunity, ensuring it stays within
    the bounds of the main grid and preserves modules of other tech types.
    Uses the provided dimensions for the localized grid size.

    Args:
        grid (Grid): The main grid.
        opportunity_x (int): The x-coordinate of the opportunity (top-left corner).
        opportunity_y (int): The y-coordinate of the opportunity (top-left corner).
        tech (str): The technology type being optimized.
        localized_width (int): The desired width of the localized window. # <<< New
        localized_height (int): The desired height of the localized window. # <<< New

    Returns:
        tuple: A tuple containing:
            - localized_grid (Grid): The localized grid.
            - start_x (int): The starting x-coordinate of the localized grid in the main grid.
            - start_y (int): The starting y-coordinate of the localized grid in the main grid.
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


# --- NEW ML-Specific Function ---
def create_localized_grid_ml(grid, opportunity_x, opportunity_y, tech, localized_width, localized_height):
    """
    Creates a localized grid around a given opportunity for ML processing,
    using the provided dimensions.

    Modules of *other* tech types within the localized area are temporarily
    removed, and their corresponding cells in the localized grid are marked
    as inactive. The original state of these modified cells (from the main grid)
    is stored for later restoration.

    Args:
        grid (Grid): The main grid.
        opportunity_x (int): The x-coordinate of the opportunity (top-left corner).
        opportunity_y (int): The y-coordinate of the opportunity (top-left corner).
        tech (str): The technology type being optimized (modules of this tech are kept).
        localized_width (int): The desired width of the localized window. # <<< New
        localized_height (int): The desired height of the localized window. # <<< New

    Returns:
        tuple: A tuple containing:
            - localized_grid (Grid): The localized grid prepared for ML.
            - start_x (int): The starting x-coordinate of the localized grid in the main grid.
            - start_y (int): The starting y-coordinate of the localized grid in the main grid.
            - original_state_map (dict): A dictionary mapping main grid coordinates
                                         (x, y) to their original cell data for cells
                                         that were modified (other tech removed).
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
