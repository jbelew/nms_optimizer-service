# bonus_calculations.py
import logging
from .grid_utils import Grid
from enum import Enum
import numpy as np
from numba import jit

# --- Constants ---
WEIGHT_FROM_GREATER_BONUS = 0.06
WEIGHT_FROM_LESSER_BONUS = 0.03
WEIGHT_FROM_GREATER_CORE = 0.07
WEIGHT_FROM_LESSER_CORE = 0.04
SUPERCHARGE_MULTIPLIER = 1.25

# --- Enums and Mappings for Numba ---
class AdjacencyType(Enum):
    NONE = 0
    GREATER = 1
    LESSER = 2

class ModuleType(Enum):
    NONE = 0
    CORE = 1
    BONUS = 2

def _get_grid_data_for_numba(grid: Grid, tech: str):
    height, width = grid.height, grid.width

    # Data arrays
    bonuses = np.zeros((height, width), dtype=np.float64)
    techs = np.zeros((height, width), dtype=np.int64)
    superchargeds = np.zeros((height, width), dtype=np.bool_)
    sc_eligibles = np.zeros((height, width), dtype=np.bool_)
    adjacencies = np.zeros((height, width), dtype=np.int64)
    module_types = np.zeros((height, width), dtype=np.int64)
    modules = np.zeros((height, width), dtype=np.int64)

    # Mappings
    tech_map = {tech: 1}
    tech_id_counter = 2

    adjacency_map = {"none": AdjacencyType.NONE.value, "greater": AdjacencyType.GREATER.value, "lesser": AdjacencyType.LESSER.value}
    module_type_map = {"core": ModuleType.CORE.value, "bonus": ModuleType.BONUS.value}

    for y in range(height):
        for x in range(width):
            cell = grid.get_cell(x, y)
            if cell["module"] is not None:
                bonuses[y, x] = cell.get("bonus", 0.0)

                cell_tech = cell.get("tech")
                if cell_tech:
                    if cell_tech not in tech_map:
                        tech_map[cell_tech] = tech_id_counter
                        tech_id_counter += 1
                    techs[y, x] = tech_map[cell_tech]

                superchargeds[y, x] = cell.get("supercharged", False)
                sc_eligibles[y, x] = cell.get("sc_eligible", False)

                adj_str = cell.get("adjacency", "none")
                if isinstance(adj_str, bool): # Handle the old boolean format
                    adjacencies[y,x] = AdjacencyType.GREATER.value if adj_str else AdjacencyType.NONE.value
                else:
                    adjacencies[y, x] = adjacency_map.get(adj_str.split('_')[0], AdjacencyType.NONE.value)

                module_types[y, x] = module_type_map.get(cell.get("type"), ModuleType.NONE.value)
                modules[y,x] = 1 # 1 if module is present, 0 otherwise

    tech_id = tech_map.get(tech, 0)
    return bonuses, techs, superchargeds, sc_eligibles, adjacencies, module_types, modules, width, height, tech_id

@jit(nopython=True)
def _calculate_adjacency_factor_numba(x, y, tech_id, techs, module_types, adjacencies, width, height):
    total_adjacency_boost_factor = 0.0
    cell_adj_type = adjacencies[y, x]

    if cell_adj_type == AdjacencyType.NONE.value:
        return 0.0

    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy

        if 0 <= nx < width and 0 <= ny < height:
            if techs[ny, nx] == tech_id:
                adj_cell_type = module_types[ny, nx]
                adj_cell_adj_type = adjacencies[ny, nx]
                weight_from_this_neighbor = 0.0

                if adj_cell_adj_type != AdjacencyType.NONE.value and cell_adj_type != AdjacencyType.NONE.value:
                    if not (cell_adj_type == AdjacencyType.LESSER.value and adj_cell_adj_type == AdjacencyType.GREATER.value):
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

                total_adjacency_boost_factor += weight_from_this_neighbor

    return total_adjacency_boost_factor

@jit(nopython=True)
def _populate_all_module_bonuses_numba(tech_id, bonuses, techs, superchargeds, sc_eligibles, adjacencies, module_types, modules, width, height):
    adjacency_factors = np.zeros_like(bonuses)
    totals = np.zeros_like(bonuses)

    for y in range(height):
        for x in range(width):
            if techs[y, x] == tech_id:
                adjacency_factors[y,x] = _calculate_adjacency_factor_numba(x, y, tech_id, techs, module_types, adjacencies, width, height)

    for y in range(height):
        for x in range(width):
            if techs[y, x] == tech_id:
                base_bonus = bonuses[y, x]
                is_supercharged = superchargeds[y, x]
                is_sc_eligible = sc_eligibles[y, x]
                adj_factor = adjacency_factors[y, x]
                module_type = module_types[y, x]

                total_bonus = 0.0
                if module_type == ModuleType.CORE.value:
                    adjacency_boost_amount_on_base = adj_factor
                else:
                    adjacency_boost_amount_on_base = base_bonus * adj_factor

                total_bonus = base_bonus + adjacency_boost_amount_on_base

                if is_supercharged and is_sc_eligible:
                    total_bonus *= SUPERCHARGE_MULTIPLIER

                totals[y,x] = total_bonus

    return totals, adjacency_factors


def calculate_grid_score(
    grid: Grid, tech: str, apply_supercharge_first: bool = False
) -> float:
    if grid is None:
        logging.warning("calculate_grid_score called with None grid.")
        return 0.0

    (bonuses, techs, superchargeds,
     sc_eligibles, adjacencies, module_types,
     modules, width, height, tech_id) = _get_grid_data_for_numba(grid, tech)

    if tech_id == 0: # Tech not found
        return 0.0

    totals, adjacency_factors = _populate_all_module_bonuses_numba(
        tech_id, bonuses, techs, superchargeds, sc_eligibles, adjacencies, module_types, modules, width, height
    )

    # Update the grid with the calculated values (side effect)
    for y in range(height):
        for x in range(width):
            if techs[y, x] == tech_id:
                cell = grid.get_cell(x, y)
                cell["total"] = round(totals[y, x], 4)
                cell["adjacency_bonus"] = round(adjacency_factors[y, x], 4)

    return round(np.sum(totals), 4)
