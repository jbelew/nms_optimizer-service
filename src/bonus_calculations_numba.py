# src/bonus_calculations_numba.py
import numba
import numpy as np
from .numba_utils import (
    ADJACENCY_LESSER,
    ADJACENCY_GREATER,
    ADJACENCY_NONE,
    MODULE_TYPE_CORE,
    MODULE_TYPE_BONUS,
)

# --- Constants ---
WEIGHT_FROM_GREATER_BONUS = 0.06
WEIGHT_FROM_LESSER_BONUS = 0.03
WEIGHT_FROM_GREATER_CORE = 0.07
WEIGHT_FROM_LESSER_CORE = 0.04
SUPERCHARGE_MULTIPLIER = 1.25

@numba.jit(nopython=True)
def _calculate_adjacency_factor_numba(
    x, y, tech_str, module_grid, adjacency_grid, type_grid
):
    height, width = module_grid.shape
    total_adjacency_boost_factor = 0.0

    cell_adj_type = adjacency_grid[y, x]

    # Directions for orthogonal neighbors
    directions = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])

    for i in range(directions.shape[0]):
        dx, dy = directions[i]
        nx, ny = x + dx, y + dy

        if 0 <= nx < width and 0 <= ny < height and module_grid[ny, nx]:
            adj_cell_type = type_grid[ny, nx]
            adj_cell_adj_type = adjacency_grid[ny, nx]

            weight_from_this_neighbor = 0.0

            if adj_cell_adj_type != ADJACENCY_NONE and cell_adj_type != ADJACENCY_NONE:
                # Rule: Greater neighbor cannot give bonus to Lesser receiver
                if (cell_adj_type == ADJACENCY_LESSER and adj_cell_adj_type == ADJACENCY_GREATER):
                    if tech_str == "pulse" or tech_str == "photonix":
                        weight_from_this_neighbor = -0.01
                    else:
                        weight_from_this_neighbor = 0.0001
                else:
                    if adj_cell_type == MODULE_TYPE_CORE:
                        if adj_cell_adj_type == ADJACENCY_GREATER:
                            weight_from_this_neighbor = WEIGHT_FROM_GREATER_CORE
                        elif adj_cell_adj_type == ADJACENCY_LESSER:
                            weight_from_this_neighbor = WEIGHT_FROM_LESSER_CORE
                    elif adj_cell_type == MODULE_TYPE_BONUS:
                        if adj_cell_adj_type == ADJACENCY_GREATER:
                            weight_from_this_neighbor = WEIGHT_FROM_GREATER_BONUS
                        elif adj_cell_adj_type == ADJACENCY_LESSER:
                            weight_from_this_neighbor = WEIGHT_FROM_LESSER_BONUS

            total_adjacency_boost_factor += weight_from_this_neighbor

    return total_adjacency_boost_factor

@numba.jit(nopython=True)
def calculate_bonuses_numba(
    tech_str,
    module_grid,
    adjacency_grid,
    type_grid,
    bonus_grid,
    supercharged_grid,
    sc_eligible_grid,
    tech_module_coords,
    apply_supercharge_first,
):
    num_modules = len(tech_module_coords)
    adj_factors = np.zeros(num_modules, dtype=np.float64)
    total_bonuses = np.zeros(num_modules, dtype=np.float64)

    # Pre-calculate adjacency factors
    for i in range(num_modules):
        x, y = tech_module_coords[i]
        adj_factors[i] = _calculate_adjacency_factor_numba(
            x, y, tech_str, module_grid, adjacency_grid, type_grid
        )

    # Calculate final bonuses
    for i in range(num_modules):
        x, y = tech_module_coords[i]
        base_bonus = bonus_grid[y, x]
        is_supercharged = supercharged_grid[y, x]
        is_sc_eligible = sc_eligible_grid[y, x]
        adj_factor = adj_factors[i]
        module_type = type_grid[y, x]

        total_bonus = 0.0

        if apply_supercharge_first:
            calculation_base = base_bonus
            if is_supercharged and is_sc_eligible:
                calculation_base *= SUPERCHARGE_MULTIPLIER

            adjacency_boost_amount = 0.0
            if module_type == MODULE_TYPE_CORE:
                adjacency_boost_amount = adj_factor
            else:
                adjacency_boost_amount = calculation_base * adj_factor

            total_bonus = base_bonus + adjacency_boost_amount
        else:
            adjacency_boost_amount_on_base = 0.0
            if module_type == MODULE_TYPE_CORE:
                adjacency_boost_amount_on_base = adj_factor
            else:
                adjacency_boost_amount_on_base = base_bonus * adj_factor

            total_bonus = base_bonus + adjacency_boost_amount_on_base

            if is_supercharged and is_sc_eligible:
                total_bonus *= SUPERCHARGE_MULTIPLIER

        total_bonuses[i] = total_bonus

    return total_bonuses, adj_factors
