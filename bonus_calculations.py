# bonus_calculations.py
from grid_utils import Grid
import math

# Global weights
core_weight = 0.15  # Default core weight
bonus_weight = 0.04  # Default bonus weight

def calculate_adjacency_count(grid: Grid, x: int, y: int) -> list:
    """
    Identifies adjacent modules and their types, including adjacency type.

    Returns:
        list: A list of dictionaries, each containing the coordinates, type, and adjacency type of an adjacent module.
    """
    cell = grid.get_cell(x, y)
    if not cell["adjacency"]:
        return []  # No adjacency if not adjacency-eligible

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Only orthogonal directions
    adjacent_modules = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            neighbor = grid.get_cell(nx, ny)
            if neighbor["module"] is not None:
                adjacent_modules.append(
                    {
                        "x": nx,
                        "y": ny,
                        "type": neighbor["type"],
                        "adjacency": neighbor["adjacency"],
                    }
                )

    return adjacent_modules


def calculate_adjacency_bonus(grid: Grid, tech: str) -> None:
    """Calculates and applies adjacency bonuses to modules in the grid."""
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if (
                cell["module"] is not None
                and cell["tech"] == tech
                and cell["type"] == "bonus"
            ):
                adjacent_modules = calculate_adjacency_count(grid, x, y)
                for adjacent in adjacent_modules:
                    adjacent_cell = grid.get_cell(adjacent["x"], adjacent["y"])
                    if adjacent["type"] == "bonus":
                        if (
                            adjacent["adjacency"] == "lesser"
                            and cell["adjacency"] == "lesser"
                        ):
                            adjacent_cell["adjacency_bonus"] += (
                                cell["bonus"] * bonus_weight
                            )
                            cell["adjacency_bonus"] += (
                                adjacent_cell["bonus"] * bonus_weight
                            )
                        elif (
                            adjacent["adjacency"] == "greater"
                            and cell["adjacency"] == "greater"
                        ):
                            adjacent_cell["adjacency_bonus"] += (
                                cell["bonus"] * bonus_weight
                            )
                            cell["adjacency_bonus"] += (
                                adjacent_cell["bonus"] * bonus_weight
                            )
                        elif (
                            adjacent["adjacency"] == "greater"
                            and cell["adjacency"] == "lesser"
                        ):
                            adjacent_cell["adjacency_bonus"] += (
                                cell["bonus"] * bonus_weight
                            )
                        elif (
                            adjacent["adjacency"] == "lesser"
                            and cell["adjacency"] == "greater"
                        ):
                            pass # Do nothing, per the rules
                    elif adjacent["type"] == "core":
                        adjacent_cell["adjacency_bonus"] += (
                            cell["bonus"] * core_weight
                        )


def populate_module_bonuses(grid: Grid, x: int, y: int) -> float:
    """Calculates the total bonus for a module at a given position, considering supercharging."""
    cell = grid.get_cell(x, y)

    base_bonus = cell["bonus"]
    is_supercharged = cell["supercharged"]
    is_sc_eligible = cell["sc_eligible"]
    adjacency_bonus = cell["adjacency_bonus"]

    total_bonus = base_bonus + adjacency_bonus

    if is_supercharged and is_sc_eligible:
        total_bonus *= 1.25  # Apply supercharge multiplie

    grid.set_total(x, y, total_bonus)
    return total_bonus


def populate_all_module_bonuses(grid: Grid, tech: str) -> None:
    """Populates the total bonuses for all modules of a given tech in the grid."""
    for row in range(grid.height):
        for col in range(grid.width):
            current_cell = grid.get_cell(col, row)
            if current_cell["module"] is not None and current_cell["tech"] == tech:
                populate_module_bonuses(grid, col, row)


def calculate_grid_score(grid: Grid, tech: str) -> float:
    """Calculates the total grid score for a given technology."""
    clear_scores(grid, tech)
    calculate_adjacency_bonus(grid, tech)
    populate_all_module_bonuses(grid, tech)

    total_grid_score = 0
    bonus_sum = 0

    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech:
                bonus_sum += cell["bonus"]
                total_grid_score += cell["total"]  # Sum all module total bonuses

    # Apply the bonus_sum to the total_grid_score
    total_grid_score *= 1.1**bonus_sum

    return total_grid_score


def clear_scores(grid: Grid, tech: str) -> None:
    """Clears the total scores for all modules of a given technology."""
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech:
                grid.set_total(x, y, 0)
                cell["adjacency_bonus"] = 0
