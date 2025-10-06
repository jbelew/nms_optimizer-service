# grid_display.py
"""
This module provides functions for displaying the grid in the console for debugging.

It includes utilities to print both a detailed and a compact version of the grid,
using ANSI color codes to represent different technologies and supercharged slots.
The display functions are designed to only output when the logging level is set
to DEBUG.
"""
import logging
from .grid_utils import Grid
from .data_loader import get_all_module_data

# Load all module data once for display purposes.
modules = get_all_module_data()

# --- Define ANSI color codes ---
# Map color names from modules.py to ANSI escape codes
# (Using bright versions for better visibility on dark backgrounds)
COLOR_MAP = {
    "purple": "\033[38;5;135m",  # Soft purple / orchid
    "red": "\033[38;5;196m",  # Bright red
    "green": "\033[38;5;40m",  # True green
    "cyan": "\033[38;5;51m",  # Vibrant cyan
    "amber": "\033[38;5;214m",  # Orange-amber
    "iris": "\033[38;5;63m",  # Blue-violet / iris
    "yellow": "\033[38;5;226m",  # True yellow
    "sky": "\033[38;5;117m",  # Sky blue
    "jade": "\033[38;5;35m",  # Jade green / teal
    "orange": "\033[38;5;202m",  # True orange
    "gray": "\033[38;5;245m",  # Medium gray
}
RESET_COLOR = "\033[0m"
SUPERCHARGED_COLOR = "\033[93m"  # Keep supercharged distinct (Light Yellow)


# --- Helper to get tech color ---
def get_tech_color_code(tech_key: str) -> str:
    """Finds the ANSI color code for a given technology key.

    Args:
        tech_key (str): The technology key (e.g., "pulse").

    Returns:
        str: The ANSI escape code for the color, or an empty string if not found.
    """
    if not tech_key:
        return ""  # Default color if no tech

    for ship_data in modules.values():
        for category_list in ship_data.get("types", {}).values():
            for tech_info in category_list:
                if tech_info.get("key") == tech_key:
                    color_name = tech_info.get("color")
                    return COLOR_MAP.get(color_name, "")  # Return ANSI code or default
    return ""  # Default if tech_key not found


# --- Updated Print Functions ---


def print_grid(grid: Grid) -> None:
    """Displays a detailed, color-coded representation of the grid for debugging.

    This function prints the grid to the debug log, showing the module ID,
    total bonus, base bonus, and adjacency bonus for each cell. Cells are
    colored based on their technology type.

    Args:
        grid (Grid): The grid object to display.
    """
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    logging.debug("--- Grid Display ---")
    for y, row in enumerate(grid.cells):
        formatted_row = []

        for x, cell in enumerate(row):
            # Use grid.get_cell for safety, no need for deepcopy here as we read values
            cell_data = grid.get_cell(x, y)
            is_supercharged = cell_data["supercharged"]
            tech_key = cell_data["tech"]
            module_id = cell_data["module"]
            is_active = cell_data["active"]

            active_state = "++" if is_active else "--"
            display_text = active_state if module_id is None else module_id

            # Determine color
            color_code = SUPERCHARGED_COLOR if is_supercharged else get_tech_color_code(tech_key)

            # Format the cell string
            formatted_cell = (
                f"{color_code}"
                f"{display_text} "
                f"(T: {cell_data['total']:.3f}) "
                f"(B: {cell_data['bonus']:.3f}) "
                f"(A: {cell_data['adjacency_bonus']:.3f})"
                f"{RESET_COLOR}"
            )
            formatted_row.append(formatted_cell)

        logging.debug(" | ".join(formatted_row))
    logging.debug("--------------------")


def print_grid_compact(grid: Grid) -> None:
    """Displays a compact, color-coded representation of the grid for debugging.

    This function prints a simplified view of the grid to the debug log,
    showing only the module IDs, colored by their technology type.

    Args:
        grid (Grid): The grid object to display.
    """
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    logging.debug("--- Compact Grid Display ---")
    for y, row in enumerate(grid.cells):
        formatted_row = []

        for x, cell in enumerate(row):
            # Use grid.get_cell for safety
            cell_data = grid.get_cell(x, y)
            is_supercharged = cell_data["supercharged"]
            tech_key = cell_data["tech"]
            module_id = cell_data["module"]
            is_active = cell_data["active"]

            active_state = "++" if is_active else "--"
            display_text = active_state if module_id is None else module_id

            # Determine color
            color_code = SUPERCHARGED_COLOR if is_supercharged else get_tech_color_code(tech_key)

            # Format the cell string
            formatted_cell = f"{color_code}{display_text}{RESET_COLOR}"
            formatted_row.append(formatted_cell)

        logging.debug(" | ".join(formatted_row))
    logging.debug("--------------------------")
