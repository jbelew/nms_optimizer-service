from grid_utils import Grid


def calculate_adjacency_bonus(grid: Grid, x: int, y: int) -> float:
    """Calculates the adjacency bonus for a module at a given position."""
    # cell = grid.get_cell(x, y)
    # if not cell["adjacency"]:
    #     return 0.0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Only orthogonal directions
    adjacency_bonus = 0.0

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            neighbor = grid.get_cell(nx, ny)
            if neighbor["module"] is not None and neighbor["tech"] == cell["tech"]:
                adjacency_bonus += 1
                grid.set_adjacency_bonus(x, y, adjacency_bonus)

    return adjacency_bonus


def populate_adjacency_bonuses(grid: Grid, tech: str) -> None:
    """Populates the adjacency bonuses for all modules in the grid."""
    for row in range(grid.height):
        for col in range(grid.width):
            current_cell = grid.get_cell(col, row)
            if current_cell["module"] is not None and current_cell["tech"] == tech:
                calculate_adjacency_bonus(grid, col, row)


def calculate_module_bonus(grid: Grid, x: int, y: int) -> float:
    """Calculates the total bonus for a module at a given position."""
    cell = grid.get_cell(x, y)

    base_bonus = cell["bonus"]
    adjacency_bonus = cell["adjacency_bonus"]
    is_supercharged = cell["supercharged"]
    is_sc_eligible = cell["sc_eligible"]

    total_bonus = base_bonus * (1 + adjacency_bonus)

    if is_supercharged and is_sc_eligible:
        total_bonus *= 1.25

    grid.set_total(x, y, total_bonus)
    return total_bonus


def populate_module_bonuses(grid: Grid, tech: str) -> None:
    """Populates the total bonuses for all modules in the grid."""
    for row in range(grid.height):
        for col in range(grid.width):
            current_cell = grid.get_cell(col, row)
            if current_cell["module"] is not None and current_cell["tech"] == tech:
                calculate_module_bonus(grid, col, row)


def calculate_core_bonus(grid: Grid, tech: str) -> float:
    """Calculates the core bonus for the grid, using ONLY base bonuses."""
    core_bonus = 0
    core_x, core_y = None, None

    # Find core module
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["type"] == "core" and cell["tech"] == tech:
                core_x, core_y = x, y
                core_bonus = cell["bonus"]  # Use only the base bonus of the core module
                break

    # Add base bonuses of adjacent bonus modules
    if core_x is not None and core_y is not None:
        for dy in [-1, 1]:
            y = core_y + dy
            if 0 <= y < grid.height:
                cell = grid.get_cell(core_x, y)
                if cell["type"] == "bonus" and cell["tech"] == tech:
                    core_bonus += cell["bonus"]
        for dx in [-1, 1]:
            x = core_x + dx
            if 0 <= x < grid.width:
                cell = grid.get_cell(x, core_y)
                if cell["type"] == "bonus" and cell["tech"] == tech:
                    core_bonus += cell["bonus"]

    return core_bonus


def populate_core_bonus(grid: Grid, tech: str) -> float:
    """Populates the core bonus for the grid."""
    core_bonus = calculate_core_bonus(grid, tech)
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["type"] == "core" and cell["tech"] == tech:
                grid.set_total(x, y, core_bonus)  # Set total bonus for the core module
    return core_bonus


def calculate_potential_adjacency_bonus(grid: Grid, x: int, y: int, tech: str) -> int:
    """Calculate the potential adjacency bonus if a module were placed at (x, y)."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    adjacency_bonus = 0

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            neighbor = grid.get_cell(nx, ny)
            if neighbor["module"] is not None and neighbor["tech"] == tech:
                adjacency_bonus += 1
    return adjacency_bonus

