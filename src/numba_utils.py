# src/numba_utils.py
import numpy as np
from .grid_utils import Grid

# --- Numba-compatible constants ---

# Using integers for enums
# AdjacencyType
ADJACENCY_NONE = 0
ADJACENCY_LESSER = 1
ADJACENCY_GREATER = 2

# ModuleType
MODULE_TYPE_NONE = 0
MODULE_TYPE_CORE = 1
MODULE_TYPE_BONUS = 2


def _get_adjacency_int(adjacency_str: str) -> int:
    """Converts adjacency string to its integer representation."""
    if not adjacency_str or not isinstance(adjacency_str, str):
        return ADJACENCY_NONE
    if adjacency_str.startswith("greater"):
        return ADJACENCY_GREATER
    if adjacency_str.startswith("lesser"):
        return ADJACENCY_LESSER
    return ADJACENCY_NONE

def _get_module_type_int(type_str: str) -> int:
    """Converts module type string to its integer representation."""
    if not type_str:
        return MODULE_TYPE_NONE
    if type_str == "core":
        return MODULE_TYPE_CORE
    if type_str == "bonus":
        return MODULE_TYPE_BONUS
    return MODULE_TYPE_NONE


def grid_to_numba_data(grid: Grid, tech: str):
    """
    Converts a Grid object into a set of NumPy arrays that are suitable
    for use with Numba-jitted functions.
    """
    width, height = grid.width, grid.height

    # Initialize NumPy arrays
    module_grid = np.zeros((height, width), dtype=np.bool_)
    adjacency_grid = np.zeros((height, width), dtype=np.int8)
    type_grid = np.zeros((height, width), dtype=np.int8)
    bonus_grid = np.zeros((height, width), dtype=np.float64)
    supercharged_grid = np.zeros((height, width), dtype=np.bool_)
    sc_eligible_grid = np.zeros((height, width), dtype=np.bool_)

    tech_module_coords = []

    for y in range(height):
        for x in range(width):
            cell = grid.get_cell(x, y)

            # Check if the module exists and belongs to the specified tech
            if cell.get("module") is not None and cell.get("tech") == tech:
                module_grid[y, x] = True
                tech_module_coords.append((x, y))

                adjacency_grid[y, x] = _get_adjacency_int(cell.get("adjacency"))
                type_grid[y, x] = _get_module_type_int(cell.get("type"))
                bonus_grid[y, x] = cell.get("bonus", 0.0)
                supercharged_grid[y, x] = cell.get("supercharged", False)
                sc_eligible_grid[y, x] = cell.get("sc_eligible", False)

    return (
        module_grid,
        adjacency_grid,
        type_grid,
        bonus_grid,
        supercharged_grid,
        sc_eligible_grid,
        tuple(tech_module_coords) # Pass coordinates as a tuple of tuples
    )
