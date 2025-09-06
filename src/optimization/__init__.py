# src/optimization/__init__.py

# Expose the main optimization function directly from the package
from .core import optimize_placement

# Expose the refinement and training functions
from .refinement import refine_placement, simulated_annealing
from .training import refine_placement_for_training

# You can choose whether to expose helpers and windowing functions.
# If they are only used internally by the optimization module, it's better not to expose them.
# If they are needed by other parts of the application, you can expose them here.
# For now, I will not expose them to keep the package interface clean.

__all__ = [
    "optimize_placement",
    "refine_placement",
    "simulated_annealing",
    "refine_placement_for_training",
]
