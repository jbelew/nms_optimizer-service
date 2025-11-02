# bonus_calculations.py
"""This module handles the calculation of technology bonuses in the grid.

It includes functions for determining adjacency bonuses, applying supercharge
multipliers, and calculating the overall score for a given technology layout.
The core logic revolves around the concept of "adjacency factors," which are
weights applied to modules based on their neighbors.
"""
import logging
from .grid_utils import Grid
from enum import Enum

# --- Constants ---


# Adjacency Types using Enum
class AdjacencyType(Enum):
    """Enumeration for module adjacency types."""

    GREATER = "greater"
    LESSER = "lesser"


# Module Types using Enum
class ModuleType(Enum):
    """Enumeration for module types."""

    CORE = "core"
    BONUS = "bonus"
    UPGRADE = "upgrade"
    COSMETIC = "cosmetic"
    REACTOR = "reactor"
    ATLANTID = "atlantid"


# --- Weights (Based on Neighbor Type and Adjacency) ---
# Weight applied TO the current module FROM the neighbor
WEIGHT_FROM_GREATER_BONUS = 0.06
WEIGHT_FROM_LESSER_BONUS = 0.03
WEIGHT_FROM_GREATER_CORE = 0.07
WEIGHT_FROM_LESSER_CORE = 0.04

# Supercharge Multiplier
SUPERCHARGE_MULTIPLIER = 1.25


# --- Helper Functions ---
def _get_orthogonal_neighbors(grid: Grid, x: int, y: int) -> list[dict]:
    """Gets valid orthogonal neighbor cells with modules of the same tech.

    Args:
        grid: The Grid object representing the layout.
        x: The x-coordinate of the center cell.
        y: The y-coordinate of the center cell.

    Returns:
        A list of dictionaries, where each dictionary represents a neighbor
        and contains its data and coordinates.
    """
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    center_cell = grid.get_cell(x, y)
    center_cell_tech = center_cell.get("tech")

    if not center_cell_tech or center_cell.get("module") is None:
        return []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            neighbor_cell = grid.get_cell(nx, ny)
            if neighbor_cell.get("module") is not None and neighbor_cell.get("tech") == center_cell_tech:
                neighbor_data = neighbor_cell.copy()
                neighbor_data["x"] = nx
                neighbor_data["y"] = ny
                neighbors.append(neighbor_data)
    return neighbors


# --- Core Calculation Functions ---




def clear_scores(grid: Grid, tech: str) -> None:
    """
    Clears calculated scores for modules of a specific technology.

    Resets the 'total' and 'adjacency_bonus' fields to 0.0 for all
    modules of the given tech.

    Args:
        grid: The Grid object.
        tech: The technology type to clear.
    """
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell.get("tech") == tech:
                grid.set_total(x, y, 0.0)
                cell["adjacency_bonus"] = 0.0


from rust_scorer import calculate_grid_score as rust_calculate_grid_score, Grid as RustGrid, Cell as RustCell, AdjacencyType as RustAdjacencyType, ModuleType as RustModuleType

def calculate_grid_score(grid: Grid, tech: str, apply_supercharge_first: bool = False) -> float:
    """
    Calculates the total grid score for a given technology using the Rust implementation.
    """
    if grid is None:
        logging.warning("calculate_grid_score called with None grid.")
        return 0.0

    # Convert Python Grid to Rust Grid
    rust_cells = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            py_cell = grid.get_cell(x, y)
            adjacency = None
            if py_cell["adjacency"] == "greater":
                adjacency = RustAdjacencyType.Greater
            elif py_cell["adjacency"] == "lesser":
                adjacency = RustAdjacencyType.Lesser

            module_type = None
            if py_cell["type"] == "bonus":
                module_type = RustModuleType.Bonus
            elif py_cell["type"] == "core":
                module_type = RustModuleType.Core

            rust_cell = RustCell(
                py_cell["value"],
                py_cell["total"],
                py_cell["adjacency_bonus"],
                py_cell["bonus"],
                py_cell["active"],
                py_cell["supercharged"],
                py_cell["sc_eligible"],
                module=py_cell["module"],
                label=py_cell["label"],
                module_type=module_type,
                adjacency=adjacency,
                tech=py_cell["tech"],
                image=py_cell["image"],
            )
            row.append(rust_cell)
        rust_cells.append(row)

    rust_grid = RustGrid(
        width=grid.width,
        height=grid.height,
        cells=rust_cells,
    )

    # Call the Rust function
    return rust_calculate_grid_score(rust_grid, tech, apply_supercharge_first)


def calculate_score_delta(grid: Grid, changed_cells_info: list, tech: str) -> float:
    """
    Calculates the change in grid score based on a move.
    """
    old_grid = grid.copy()
    for (x, y), original_cell_data in changed_cells_info:
        old_grid.cells[y][x].update(original_cell_data)

    old_score = calculate_grid_score(old_grid, tech)
    new_score = calculate_grid_score(grid, tech)

    return new_score - old_score
