# optimizer.py
from src.grid_utils import Grid
from src.modules_utils import get_tech_modules, get_tech_modules_for_training, get_tech_tree, get_tech_tree_json
# --- Remove these imports to break the cycle ---
# from optimization_algorithms import optimize_placement, refine_placement, refine_placement_for_training, calculate_grid_score
# --- Keep imports for functions defined elsewhere or used directly if any ---
from src.bonus_calculations import calculate_grid_score # Keep calculate_grid_score only if optimizer.py uses it directly
from src.grid_display import print_grid, print_grid_compact


# Re-export the functions that other parts of your project use
__all__ = [
    "Grid",
    "get_tech_modules",
    "get_tech_modules_for_training",
    "get_tech_tree",
    "get_tech_tree_json",
    "calculate_grid_score",
    "print_grid",
    "print_grid_compact",
]
