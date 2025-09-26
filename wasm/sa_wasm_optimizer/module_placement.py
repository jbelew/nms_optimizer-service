# module_placement.py
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
    """Places a module on the grid at the specified position."""
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
    """Clears all modules of the specified tech type from the entire grid."""
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
                grid.cells[y][x]["sc_eligible"] = False
                grid.cells[y][x]["image"] = None
                grid.cells[y][x]["module_position"] = None
