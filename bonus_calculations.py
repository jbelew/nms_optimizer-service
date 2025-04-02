# bonus_calculations.py
from grid_utils import Grid


def calculate_adjacency_bonus(grid: Grid, x: int, y: int) -> float:
    """Calculates the adjacency bonus multiplier for a module at a given position."""
    cell = grid.get_cell(x, y)
    if not cell["adjacency"]:
        grid.set_adjacency_bonus(x, y, 1.0)  # No adjacency bonus if not adjacency-eligible
        return 1.0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Only orthogonal directions
    adjacency_count = 0

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            neighbor = grid.get_cell(nx, ny)
            if neighbor["module"] is not None and neighbor["tech"] == cell["tech"]:
                adjacency_count += 1

    adjacency_multiplier = 1 + (adjacency_count * 0.1)  # 10% bonus per adjacent module
    grid.set_adjacency_bonus(x, y, adjacency_multiplier)
    return adjacency_multiplier


def populate_adjacency_bonuses(grid: Grid, tech: str) -> None:
    """Populates the adjacency bonuses for all modules of a given tech in the grid."""
    for row in range(grid.height):
        for col in range(grid.width):
            current_cell = grid.get_cell(col, row)
            if current_cell["module"] is not None and current_cell["tech"] == tech:
                calculate_adjacency_bonus(grid, col, row)


def calculate_module_bonus(grid: Grid, x: int, y: int) -> float:
    """Calculates the total bonus for a module at a given position, considering supercharging and adjacency."""
    cell = grid.get_cell(x, y)

    base_bonus = cell["bonus"]
    adjacency_multiplier = cell["adjacency_bonus"]  # This is now a multiplier
    is_supercharged = cell["supercharged"]
    is_sc_eligible = cell["sc_eligible"]

    total_bonus = base_bonus

    if is_supercharged and is_sc_eligible:
        total_bonus *= 1.5  # Apply supercharge multiplier

    # Apply adjacency bonus as a multiplier
    if cell["adjacency"]:
        total_bonus *= adjacency_multiplier

    grid.set_total(x, y, total_bonus)
    return total_bonus


def populate_module_bonuses(grid: Grid, tech: str) -> None:
    """Populates the total bonuses for all modules of a given tech in the grid."""
    for row in range(grid.height):
        for col in range(grid.width):
            current_cell = grid.get_cell(col, row)
            if current_cell["module"] is not None and current_cell["tech"] == tech:
                calculate_module_bonus(grid, col, row)


def calculate_grid_score(grid: Grid, tech: str) -> float:
    """Calculates the total grid score for a given technology."""
    clear_scores(grid, tech)
    calculate_all_bonuses(grid, tech)

    total_grid_score = 0

    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech:
                total_grid_score += cell["total"]  # Sum all module total bonuses

    return total_grid_score


def calculate_all_bonuses(grid: Grid, tech: str) -> None:
    """Calculates and populates all bonus types for a given technology in the grid."""
    populate_adjacency_bonuses(grid, tech)
    populate_module_bonuses(grid, tech)


def clear_scores(grid: Grid, tech: str) -> None:
    """Clears the total scores and adjacency bonuses for all modules of a given technology."""
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech:
                grid.set_total(x, y, 0)
                grid.set_adjacency_bonus(x, y, 0)
