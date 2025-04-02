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
from module_placement import place_module, clear_all_modules_of_tech
from bonus_calculations import clear_scores

def reset_grid_state(grid):
    """Resets the state of the grid cells to their initial, empty condition."""
    for y in range(grid.height):
        for x in range(grid.width):
            grid.cells[y][x]["module"] = None
            grid.cells[y][x]["label"] = ""
            grid.cells[y][x]["tech"] = None
            grid.cells[y][x]["type"] = ""
            grid.cells[y][x]["bonus"] = 0
            grid.cells[y][x]["total"] = 0
            grid.cells[y][x]["adjacency_bonus"] = 0
            grid.cells[y][x]["adjacency"] = False  # Explicitly reset adjacency to False
            grid.cells[y][x]["sc_eligible"] = False
            grid.cells[y][x]["image"] = None
            grid.cells[y][x]["module_position"] = None  # Explicitly reset module_position
            grid.cells[y][x]["active"] = True  # Reset to active
            grid.cells[y][x]["supercharged"] = False  # Reset to not supercharged

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
    place_module(grid, 0, 0, "Xa", "Infraknife Accelerator", "infra", "core", 1.0, True, True, "infra.png")
    place_module(grid, 1, 0, "IK", "Infraknife Accelerator Upgrade Sigma", "infra", "bonus", 0.40, True, True, "infra-upgrade.png")
    place_module(grid, 2, 0, "QR", "Q-Resonator", "infra", "bonus", 0.04, True, True, "q-resonator.png")
    place_module(grid, 0, 1, "Xb", "Infraknife Accelerator Upgrade Tau", "infra", "bonus", 0.39, True, True, "infra-upgrade.png")
    place_module(grid, 1, 1, "Xc", "Infraknife Accelerator Upgrade Theta", "infra", "bonus", 0.38, True, True, "infra-upgrade.png")
    grid.set_supercharged(1,0, True)
    grid.set_supercharged(3,0, True)
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
    # Reset grid state before calculating the manual score
    reset_grid_state(grid_copy)
    # Re-place the modules to ensure scores are calculated correctly
    grid_copy = grid_creator(grid_copy)
    # Clear the scores before calculating the manual score
    clear_scores(grid_copy, tech)
    # Calculate the manual score AFTER placing modules and clearing scores
    manual_score = calculate_grid_score(grid_copy, tech)
    print_grid(grid_copy)
    print(f"Manual Score: {manual_score}")

    # Create a second deep copy for refine_placement
    grid_for_refine = grid.copy()

    # Clear all modules of the specified tech from the grid before calling refine_placement
    clear_all_modules_of_tech(grid_for_refine, tech)

    optimal_grid, highest_bonus = refine_placement(grid_for_refine, ship, modules, tech)

    print("--- Refined Grid ---")
    if optimal_grid:
        print_grid(optimal_grid)
        print(f"Refined Score: {highest_bonus}")
    else:
        print("Refine placement returned None")

    print("-" * 20)

if __name__ == "__main__":
    ship = "standard"
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
