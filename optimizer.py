# optimizer.py
from grid_utils import Grid
from modules_data import get_tech_modules, get_tech_modules_for_training, get_tech_tree, get_tech_tree_json
# --- Remove these imports to break the cycle ---
# from optimization_algorithms import optimize_placement, refine_placement, refine_placement_for_training, calculate_grid_score
# --- Keep imports for functions defined elsewhere or used directly if any ---
from bonus_calculations import populate_module_bonuses, calculate_grid_score # Keep calculate_grid_score only if optimizer.py uses it directly
from grid_display import print_grid, print_grid_compact


# Re-export the functions that other parts of your project use
__all__ = [
    "Grid",
    "get_tech_modules",
    "get_tech_modules_for_training",
    "get_tech_tree",
    "get_tech_tree_json",
    "optimize_placement",
    "refine_placement",
    "refine_placement_for_training",
    "calculate_grid_score",
    "populate_adjacency_count",
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
