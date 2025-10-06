# optimizer.py
"""
This module serves as a central hub for optimization-related utilities.

It re-exports key functions and classes from other modules to provide a
single, convenient access point for other parts of the application, such as
the main Flask app. This also helps in managing imports and avoiding
circular dependency issues.
"""
from src.grid_utils import Grid
from src.modules_utils import get_tech_modules, get_tech_modules_for_training, get_tech_tree, get_tech_tree_json
from src.bonus_calculations import calculate_grid_score
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
