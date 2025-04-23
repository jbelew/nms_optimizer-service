# bonus_calculations.py
from grid_utils import Grid
import math
from enum import Enum
from copy import deepcopy

# --- Constants ---

# Adjacency Types using Enum
class AdjacencyType(Enum):
    GREATER = "greater"
    LESSER = "lesser"

# Module Types using Enum
class ModuleType(Enum):
    CORE = "core"
    BONUS = "bonus"

# --- Weights (Based on Neighbor Type and Adjacency) ---
# Weight applied TO the current module FROM the neighbor
WEIGHT_FROM_GREATER_BONUS = 0.07
WEIGHT_FROM_LESSER_BONUS = 0.04
WEIGHT_FROM_GREATER_CORE = 0.06
WEIGHT_FROM_LESSER_CORE = 0.04

# Supercharge Multiplier
SUPERCHARGE_MULTIPLIER = 1.25

# --- Helper Functions ---

def _get_orthogonal_neighbors(grid: Grid, x: int, y: int) -> list[dict]:
    """Gets valid orthogonal neighbor cells with modules of the same tech."""
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
                neighbor_data['x'] = nx
                neighbor_data['y'] = ny
                neighbors.append(neighbor_data)
    return neighbors

# --- Core Calculation Functions ---

def _calculate_adjacency_factor(grid: Grid, x: int, y: int) -> float:
    """
    Calculates the total adjacency boost *factor* for a module based on the
    type and adjacency of its neighbors, respecting the receiver's adjacency rules.

    Args:
        grid: The Grid object.
        x: The x-coordinate of the module.
        y: The y-coordinate of the module.

    Returns:
        The total adjacency boost factor (sum of weights from neighbors).
    """
    cell = grid.get_cell(x, y)
    if cell.get("module") is None:
        return 0.0

    # Get the receiver's adjacency type
    cell_adj_type = cell.get("adjacency")

    total_adjacency_boost_factor = 0.0
    adjacent_cells = _get_orthogonal_neighbors(grid, x, y) # Neighbors of the same tech

    for adj_cell in adjacent_cells:
        # Get the neighbor's (giver's) type and adjacency type
        adj_cell_type = adj_cell.get("type")
        adj_cell_adj_type = adj_cell.get("adjacency")

        weight_from_this_neighbor = 0.0

        # Check if both giver and receiver have defined adjacency types
        if adj_cell_adj_type and cell_adj_type:

            # Rule: Greater neighbor cannot give bonus to Lesser receiver
            if cell_adj_type == AdjacencyType.LESSER.value and adj_cell_adj_type == AdjacencyType.GREATER.value:
                weight_from_this_neighbor = 0.0
            else:
                # Determine weight based on the NEIGHBOR's type and adjacency
                if adj_cell_type == ModuleType.CORE.value:
                    if adj_cell_adj_type == AdjacencyType.GREATER.value:
                        weight_from_this_neighbor = WEIGHT_FROM_GREATER_CORE
                    elif adj_cell_adj_type == AdjacencyType.LESSER.value:
                        weight_from_this_neighbor = WEIGHT_FROM_LESSER_CORE
                elif adj_cell_type == ModuleType.BONUS.value:
                    if adj_cell_adj_type == AdjacencyType.GREATER.value:
                        weight_from_this_neighbor = WEIGHT_FROM_GREATER_BONUS
                    elif adj_cell_adj_type == AdjacencyType.LESSER.value:
                        weight_from_this_neighbor = WEIGHT_FROM_LESSER_BONUS
                # If neighbor type is unknown or adjacency is 'none', weight remains 0.0

        total_adjacency_boost_factor += weight_from_this_neighbor

    return total_adjacency_boost_factor


def populate_all_module_bonuses(grid: Grid, tech: str, apply_supercharge_first: bool = False) -> None:
    """
    Calculates and populates the total bonuses for all modules of a
    given tech in the grid using a non-iterative approach.
    Updates 'adjacency_bonus' to store the raw adjacency factor.

    Args:
        grid: The Grid object.
        tech: The technology type to calculate for.
        apply_supercharge_first (bool):
            - If False (default): Calculates boost on base, then applies supercharge
              multiplier to the final sum (base + boost).
            - If True: Applies supercharge multiplier to the base bonus *before*
              calculating adjacency boost amount. The final total is then
              base_bonus + boost_amount (calculated on supercharged base).
    """
    tech_module_coords = []
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell.get("module") is not None and cell.get("tech") == tech:
                tech_module_coords.append((x, y))
                # Reset scores before calculation
                grid.set_total(x, y, 0.0)
                cell["adjacency_bonus"] = 0.0 # Reset adjacency factor display

    if not tech_module_coords:
        return

    # Pre-calculate adjacency factors for all relevant modules
    module_adj_factors = {}
    for x, y in tech_module_coords:
         module_adj_factors[(x, y)] = _calculate_adjacency_factor(grid, x, y)

    # Calculate final bonuses using pre-calculated factors
    for x, y in tech_module_coords:
        cell = grid.get_cell(x, y)
        base_bonus = cell.get("bonus", 0.0)
        is_supercharged = cell.get("supercharged", False)
        is_sc_eligible = cell.get("sc_eligible", False)
        adj_factor = module_adj_factors[(x, y)] # This is the raw factor (sum of weights)

        total_bonus = 0.0
        # final_adjacency_boost_display = 0.0 # No longer needed for this purpose

        if apply_supercharge_first:
            # Apply SC to base *before* calculating boost amount
            calculation_base = base_bonus
            if is_supercharged and is_sc_eligible:
                calculation_base *= SUPERCHARGE_MULTIPLIER
            adjacency_boost_amount = calculation_base * adj_factor
            # Final total is original base + boost (calculated on potentially SC base)
            total_bonus = base_bonus + adjacency_boost_amount
            # final_adjacency_boost_display = adjacency_boost_amount # Old logic
        else:
            # Calculate boost on original base *first* (DEFAULT BEHAVIOR)
            adjacency_boost_amount_on_base = base_bonus * adj_factor
            # Preliminary total
            total_bonus = base_bonus + adjacency_boost_amount_on_base
            # Apply SC multiplier to the combined total *after* adding boost
            if is_supercharged and is_sc_eligible:
                total_bonus *= SUPERCHARGE_MULTIPLIER
            # final_adjacency_boost_display = adjacency_boost_amount_on_base # Old logic

        grid.set_total(x, y, total_bonus)
        # --- Store the raw adjacency factor for display ---
        grid.get_cell(x, y)["adjacency_bonus"] = adj_factor
        # --- End change --


def populate_module_bonuses(grid: Grid, x: int, y: int, apply_supercharge_first: bool = False) -> float:
    """
    Calculates the total bonus for a single module. Consistent with populate_all_module_bonuses.
    (Kept for potential single-module updates, but generally populate_all is preferred).

    Args:
        grid: The Grid object.
        x: The x-coordinate of the module.
        y: The y-coordinate of the module.
        apply_supercharge_first (bool): Controls supercharge order. Defaults to False.

    Returns:
        The calculated total bonus for the module.
    """
    cell = grid.get_cell(x, y)
    if cell.get("module") is None:
        grid.set_total(x, y, 0.0)
        cell["adjacency_bonus"] = 0.0
        return 0.0

    base_bonus = cell.get("bonus", 0.0)
    is_supercharged = cell.get("supercharged", False)
    is_sc_eligible = cell.get("sc_eligible", False)

    adj_factor = _calculate_adjacency_factor(grid, x, y)

    total_bonus = 0.0
    final_adjacency_boost_display = 0.0

    if apply_supercharge_first:
        calculation_base = base_bonus
        if is_supercharged and is_sc_eligible:
            calculation_base *= SUPERCHARGE_MULTIPLIER
        adjacency_boost_amount = calculation_base * adj_factor
        total_bonus = base_bonus + adjacency_boost_amount
        final_adjacency_boost_display = adjacency_boost_amount
    else: # Default behavior
        adjacency_boost_amount_on_base = base_bonus * adj_factor
        total_bonus = base_bonus + adjacency_boost_amount_on_base
        if is_supercharged and is_sc_eligible:
            total_bonus *= SUPERCHARGE_MULTIPLIER
        final_adjacency_boost_display = adjacency_boost_amount_on_base

    grid.set_total(x, y, total_bonus)
    cell["adjacency_bonus"] = final_adjacency_boost_display

    return total_bonus


def clear_scores(grid: Grid, tech: str) -> None:
    """
    Clears calculated 'total' and the 'adjacency_bonus' (boost amount)
    for modules of a given tech.
    """
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell.get("tech") == tech:
                grid.set_total(x, y, 0.0)
                cell["adjacency_bonus"] = 0.0


def calculate_grid_score(grid: Grid, tech: str, apply_supercharge_first: bool = False) -> float:
    """
    Calculates the total grid score for a given technology by summing module totals.
    Relies on populate_all_module_bonuses to handle adjacency boost internally,
    passing the apply_supercharge_first flag.

    Args:
        grid: The Grid object.
        tech: The technology type (e.g., "pulse", "hyper") to score.
        apply_supercharge_first (bool): If True, applies supercharge multiplier
                                        to the base bonus before calculating
                                        adjacency boost amount. Defaults to False.

    Returns:
        The total calculated score for the specified technology, rounded.
    """
    if grid is None:
        print("Warning: calculate_grid_score called with None grid.")
        return 0.0

    # Ensure all bonuses are calculated/updated before summing
    populate_all_module_bonuses(grid, tech, apply_supercharge_first)

    total_grid_score = 0.0
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            # Sum only modules of the specified tech that have been placed
            if cell.get("module") is not None and cell.get("tech") == tech:
                total_grid_score += cell.get("total", 0.0)

    return round(total_grid_score, 8)
