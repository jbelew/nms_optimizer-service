# module_placement.py
"""
This module provides utility functions for placing and clearing modules on the grid.

These functions directly manipulate the grid's cell data, serving as helpers
for higher-level optimization and placement logic.
"""


def place_module(
    grid,
    x,
    y,
    module_id,
    label,
    tech,
    module_type,
    bonus,
    adjacency,
    sc_eligible,
    image,
):
    """Populates a grid cell with the data for a specific module.

    This function sets all the relevant properties for a module at the given
    (x, y) coordinates in the grid.

    Args:
        grid (Grid): The grid object to modify.
        x (int): The x-coordinate where the module will be placed.
        y (int): The y-coordinate where the module will be placed.
        module_id (str): The unique identifier of the module.
        label (str): The display name of the module.
        tech (str): The technology type of the module (e.g., "pulse").
        module_type (str): The category of the module (e.g., "core", "bonus").
        bonus (float): The base bonus value of the module.
        adjacency (str): The adjacency requirement (e.g., "greater", "lesser").
        sc_eligible (bool): Whether the module is eligible for supercharging.
        image (str): The image URL or identifier for the module.
    """
    grid.cells[y][x]["module"] = module_id
    grid.cells[y][x]["label"] = label
    grid.cells[y][x]["tech"] = tech
    grid.cells[y][x]["type"] = module_type
    grid.cells[y][x]["bonus"] = bonus
    grid.cells[y][x]["adjacency"] = adjacency
    grid.cells[y][x]["sc_eligible"] = sc_eligible
    grid.cells[y][x]["image"] = image
    grid.cells[y][x]["module_position"] = (x, y)


def clear_all_modules_of_tech(grid, tech):
    """Removes all modules of a specific technology from the grid.

    This function iterates through every cell in the grid and resets the
    properties of any cell that contains a module of the specified tech type.

    Args:
        grid (Grid): The grid object to modify.
        tech (str): The technology type to clear from the grid.
    """
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.get_cell(x, y)["tech"] == tech:
                grid.cells[y][x]["module"] = None
                grid.cells[y][x]["label"] = ""
                grid.cells[y][x]["tech"] = None
                grid.cells[y][x]["type"] = ""
                grid.cells[y][x]["bonus"] = 0
                grid.cells[y][x]["total"] = 0
                grid.cells[y][x]["adjacency_bonus"] = 0
                grid.cells[y][x]["adjacency"] = False
                grid.cells[y][x]["group_adjacent"] = False
                grid.cells[y][x]["sc_eligible"] = False
                grid.cells[y][x]["image"] = None
                grid.cells[y][x]["module_position"] = None
