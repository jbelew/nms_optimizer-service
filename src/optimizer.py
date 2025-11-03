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
from rust_scorer import calculate_grid_score as rust_calculate_grid_score

__all__ = [
    "Grid",
    "get_tech_modules",
    "get_tech_modules_for_training",
    "get_tech_tree",
    "get_tech_tree_json",
    "rust_calculate_grid_score",
]
