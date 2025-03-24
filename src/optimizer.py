# optimizer.py
from grid_utils import Grid
from modules_data import get_tech_modules, get_tech_modules_for_training, get_tech_tree, get_tech_tree_json
from optimization_algorithms import optimize_placement
from bonus_calculations import calculate_adjacency_bonus, populate_adjacency_bonuses, calculate_module_bonus, populate_module_bonuses, calculate_core_bonus, populate_core_bonus
from module_placement import place_module
from grid_display import print_grid, print_grid_compact

# Re-export the functions that other parts of your project use
__all__ = [
    "Grid",
    "get_tech_modules",
    "get_tech_modules_for_training",
    "get_tech_tree",
    "get_tech_tree_json",
    "optimize_placement",
    "simulated_annealing_optimization",
    "calculate_adjacency_bonus",
    "populate_adjacency_bonuses",
    "calculate_module_bonus",
    "populate_module_bonuses",
    "calculate_core_bonus",
    "populate_core_bonus",
    "place_module",
    "print_grid",
    "print_grid_compact",
    "find_best_available_position",
    "perturb_grid"
]
