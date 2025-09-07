# optimizer.py
from .grid_utils import Grid
from .modules_utils import get_tech_modules, get_tech_modules_for_training, get_tech_tree, get_tech_tree_json
from .bonus_calculations import (
    calculate_grid_score,
)
from .grid_display import print_grid, print_grid_compact
from .optimization.core import optimize_placement
from .optimization.refinement import refine_placement
from .optimization.training import refine_placement_for_training
from .module_placement import (
    place_module,
)


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
    "place_module",
    "print_grid",
    "print_grid_compact",
]
