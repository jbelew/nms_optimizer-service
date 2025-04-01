# bonus_calculations.py
from grid_utils import Grid


def calculate_adjacency_bonus(grid: Grid, x: int, y: int) -> float:
    """Calculates the adjacency bonus for a module at a given position."""
    cell = grid.get_cell(x, y)
    if not cell["adjacency"]:
        return 0.0

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

    total_bonus = base_bonus  # Default value if no adjacency

    if cell["adjacency"]:
        total_bonus = base_bonus + (1 * adjacency_bonus)

    if is_supercharged and is_sc_eligible:
        total_bonus *= 1.25

    grid.set_total(x, y, total_bonus)
    return total_bonus


def populate_module_bonuses(grid: Grid, tech: str) -> None:
    """Populates the total bonuses for all modules in the grid."""
    for row in range(grid.height):
        for col in range(grid.width):
            current_cell = grid.get_cell(col, row)
            if current_cell["module"] is None:
                continue
            if current_cell["module"] is not None and current_cell["tech"] == tech:
                calculate_module_bonus(grid, col, row)


def calculate_core_bonus(grid: Grid, tech: str) -> float:
    """Calculates the core bonus for the grid, considering adjacency, base bonus, and supercharged status."""
    core_bonus = 0
    core_x, core_y = None, None

    # Find core module and calculate its bonus
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["type"] == "core" and cell["tech"] == tech:
                core_x, core_y = x, y

                # Calculate adjacency bonus for the core module
                adjacency_bonus = calculate_adjacency_bonus(grid, x, y)

                # Calculate the total bonus for the core module
                base_bonus = cell["bonus"]
                total_bonus = base_bonus

                if cell["adjacency"]:
                    total_bonus = base_bonus + (1 * adjacency_bonus)

                # Apply supercharged modifier if applicable
                if cell["sc_eligible"] and cell["supercharged"]:
                    total_bonus *= 1.25  # Apply supercharged modifier
                
                core_bonus = total_bonus
                break  # Assume only one core module

    return core_bonus


def populate_core_bonus(grid: Grid, tech: str) -> float:
    """Populates the core bonus for the grid."""
    core_bonus = calculate_core_bonus(grid, tech)

    for row in range(grid.height):
        for col in range(grid.width):
            cell = grid.get_cell(col, row)
            if cell["type"] == "core" and cell["tech"] == tech:
                grid.set_total(col, row, core_bonus)

    return core_bonus


def calculate_grid_score(grid: Grid, tech: str) -> float:
    """Calculates the total grid score for a given technology."""
    clear_scores(grid, tech)  # Clear existing scores before calculating
    calculate_all_bonuses(grid, tech)  # Ensure all bonuses are calculated first
    total_grid_score = 0
    core_bonus = 0
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["type"] == "core" and cell["tech"] == tech:
                core_bonus = cell["total"]  # Get the core bonus
            elif cell["type"] == "bonus" and cell["tech"] == tech:
                total_grid_score += cell["total"]  # Sum all bonus module total bonuses

    return total_grid_score + core_bonus  # Add the core bonus to the total


def calculate_all_bonuses(grid: Grid, tech: str) -> None:
    """Calculates and populates all bonus types for a given technology in the grid."""
    populate_adjacency_bonuses(grid, tech)
    populate_module_bonuses(grid, tech)
    populate_core_bonus(grid, tech)


def clear_scores(grid: Grid, tech: str) -> None:
    """Clears the total scores for all modules of a given technology."""
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech:
                grid.set_total(x, y, 0)
                grid.set_adjacency_bonus(x, y, 0)
