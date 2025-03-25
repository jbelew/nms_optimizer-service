# refine_placement_debug.py
import sys
import os
from copy import deepcopy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from optimizer import (
    refine_placement,
    print_grid,
    print_grid_compact,
    Grid,
    calculate_grid_score
)
from modules import modules
from module_placement import place_module

def create_test_grid_1(grid):
    """
    Creates a test grid with a core module and some bonus modules.
    """
    # Place some modules
    place_module(grid, 0, 0, "IK", "Infraknife Accelerator", "infra", "core", 1.0, True, True, "infra.png")
    place_module(grid, 1, 0, "Xa", "Infraknife Accelerator Upgrade Sigma", "infra", "bonus", 0.40, True, True, "infra-upgrade.png")
    place_module(grid, 0, 1, "QR", "Q-Resonator", "infra", "bonus", 0.04, True, True, "q-resonator.png")
    place_module(grid, 1, 1, "Xb", "Infraknife Accelerator Upgrade Tau", "infra", "bonus", 0.39, True, True, "infra-upgrade.png")
    place_module(grid, 2, 1, "Xc", "Infraknife Accelerator Upgrade Theta", "infra", "bonus", 0.38, True, True, "infra-upgrade.png")
    return grid

def create_test_grid_2(grid):
    """
    Creates a test grid with a core module and some bonus modules.
    """
    # Place some modules
    place_module(grid, 0, 0, "IK", "Infraknife Accelerator", "infra", "core", 1.0, True, True, "infra.png")
    place_module(grid, 1, 0, "Xa", "Infraknife Accelerator Upgrade Sigma", "infra", "bonus", 0.40, True, True, "infra-upgrade.png")
    place_module(grid, 0, 1, "QR", "Q-Resonator", "infra", "bonus", 0.04, True, True, "q-resonator.png")
    place_module(grid, 1, 1, "Xb", "Infraknife Accelerator Upgrade Tau", "infra", "bonus", 0.39, True, True, "infra-upgrade.png")
    place_module(grid, 2, 1, "Xc", "Infraknife Accelerator Upgrade Theta", "infra", "bonus", 0.38, True, True, "infra-upgrade.png")
    grid.set_supercharged(0,0, True)
    return grid

def create_test_grid_3(grid):
    """
    Creates a test grid with a core module and some bonus modules.
    """
    # Place some modules
    place_module(grid, 0, 0, "IK", "Infraknife Accelerator", "infra", "core", 1.0, True, True, "infra.png")
    place_module(grid, 1, 0, "Xa", "Infraknife Accelerator Upgrade Sigma", "infra", "bonus", 0.40, True, True, "infra-upgrade.png")
    place_module(grid, 0, 1, "QR", "Q-Resonator", "infra", "bonus", 0.04, True, True, "q-resonator.png")
    place_module(grid, 1, 1, "Xb", "Infraknife Accelerator Upgrade Tau", "infra", "bonus", 0.39, True, True, "infra-upgrade.png")
    place_module(grid, 2, 1, "Xc", "Infraknife Accelerator Upgrade Theta", "infra", "bonus", 0.38, True, True, "infra-upgrade.png")
    grid.set_supercharged(0,0, True)
    grid.set_supercharged(1,0, True)
    return grid

def create_test_grid_4(grid):
    """
    Creates a test grid with a core module and some bonus modules.
    """
    # Place some modules
    place_module(grid, 0, 0, "IK", "Infraknife Accelerator", "infra", "core", 1.0, True, True, "infra.png")
    place_module(grid, 1, 0, "Xa", "Infraknife Accelerator Upgrade Sigma", "infra", "bonus", 0.40, True, True, "infra-upgrade.png")
    place_module(grid, 0, 1, "QR", "Q-Resonator", "infra", "bonus", 0.04, True, True, "q-resonator.png")
    place_module(grid, 1, 1, "Xb", "Infraknife Accelerator Upgrade Tau", "infra", "bonus", 0.39, True, True, "infra-upgrade.png")
    place_module(grid, 2, 1, "Xc", "Infraknife Accelerator Upgrade Theta", "infra", "bonus", 0.38, True, True, "infra-upgrade.png")
    grid.set_supercharged(0,0, True)
    grid.set_supercharged(1,0, True)
    grid.set_supercharged(2,0, True)
    return grid

def create_test_grid_5(grid):
    """
    Creates a test grid with a core module and some bonus modules.
    """
    # Place some modules
    place_module(grid, 3, 0, "IK", "Infraknife Accelerator", "infra", "core", 1.0, True, True, "infra.png")
    place_module(grid, 1, 0, "Xa", "Infraknife Accelerator Upgrade Sigma", "infra", "bonus", 0.40, True, True, "infra-upgrade.png")
    place_module(grid, 1, 1, "QR", "Q-Resonator", "infra", "bonus", 0.04, True, True, "q-resonator.png")
    place_module(grid, 2, 0, "Xb", "Infraknife Accelerator Upgrade Tau", "infra", "bonus", 0.39, True, True, "infra-upgrade.png")
    place_module(grid, 2, 1, "Xc", "Infraknife Accelerator Upgrade Theta", "infra", "bonus", 0.38, True, True, "infra-upgrade.png")
    grid.set_supercharged(0,0, False)
    grid.set_supercharged(1,0, True)
    grid.set_supercharged(2,0, False)
    grid.set_supercharged(3,0, True)
    return grid

def test_refine_placement(grid_creator, ship, tech):
    """
    Tests the refine_placement function on a given grid.
    """
    # Create a new, empty grid for each test case
    grid = Grid(4, 3)
    grid = grid_creator(grid)

    # Create a deep copy of the grid for manual scoring
    grid_copy = grid.copy()

    print("--- Initial Grid ---")
    print_grid(grid)

    manual_score = calculate_grid_score(grid_copy, tech)
    print(f"Manual Score: {manual_score}")

    # Create a second deep copy for refine_placement
    grid_for_refine = grid.copy()

    # Clear all modules from the grid before calling refine_placement
    clear_all_modules(grid_for_refine)

    optimal_grid, highest_bonus = refine_placement(grid_for_refine, ship, modules, tech)

    print("--- Refined Grid ---")
    if optimal_grid:
        print_grid(optimal_grid)
        print(f"Refined Score: {highest_bonus}")
    else:
        print("Refine placement returned None")

    print("-" * 20)

def clear_all_modules(grid):
    """Clears all modules from the entire grid."""
    for y in range(grid.height):
        for x in range(grid.width):
            grid.cells[y][x]["module"] = None
            grid.cells[y][x]["label"] = ""
            grid.cells[y][x]["tech"] = None
            grid.cells[y][x]["type"] = ""
            grid.cells[y][x]["bonus"] = 0
            grid.cells[y][x]["adjacency"] = False
            grid.cells[y][x]["sc_eligible"] = False
            grid.cells[y][x]["image"] = None

if __name__ == "__main__":
    ship = "Exotic"
    tech = "infra"

    test_grid_creators = [
        create_test_grid_1,
        create_test_grid_2,
        create_test_grid_3,
        create_test_grid_4,
        create_test_grid_5,
    ]

    for i, grid_creator in enumerate(test_grid_creators):
        print(f"--- Test Case {i+1} ---")
        test_refine_placement(grid_creator, ship, tech)
