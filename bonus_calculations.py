# bonus_calculations.py
from grid_utils import Grid
import math

# Global weights
core_weight_greater = 0.06  # Core weight for greater adjacency
core_weight_lesser = 0.037  # Core weight for lesser adjacency
greater_weight = 0.04  # Default greater adjacency weight (formerly bonus_weight)
lesser_weight = 0.030  # Default lesser adjacency weight

# Constants for adjacency types
ADJACENCY_GREATER = "greater"
ADJACENCY_LESSER = "lesser"
ADJACENCY_CORE = "core"
ADJACENCY_BONUS = "bonus"


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
                        "tech": neighbor["tech"],  # Include the tech type
                    }
                )

    return adjacent_modules


def calculate_adjacency_bonus(grid: Grid, tech: str) -> None:
    """Calculates and applies adjacency bonuses to modules in the grid."""
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["module"] is not None and cell["tech"] == tech:
                adjacent_modules = calculate_adjacency_count(grid, x, y)
                for adjacent in adjacent_modules:
                    adjacent_cell = grid.get_cell(adjacent["x"], adjacent["y"])
                    if adjacent["tech"] == tech:
                        bonus_type = (cell["adjacency"], adjacent["adjacency"])
                        bonus_map = {
                            (ADJACENCY_LESSER, ADJACENCY_LESSER): lesser_weight,
                            (ADJACENCY_GREATER, ADJACENCY_GREATER): greater_weight,
                            (ADJACENCY_GREATER, ADJACENCY_LESSER): lesser_weight,
                            (ADJACENCY_LESSER, ADJACENCY_GREATER): lesser_weight,
                        }
                        weight = bonus_map.get(bonus_type)
                        if weight:
                            if cell["type"] == ADJACENCY_BONUS and adjacent["type"] == ADJACENCY_BONUS:
                                adjacent_cell["adjacency_bonus"] += cell["bonus"] * weight
                                if bonus_type == (ADJACENCY_LESSER, ADJACENCY_LESSER) or bonus_type == (
                                    ADJACENCY_GREATER,
                                    ADJACENCY_GREATER,
                                ):
                                    cell["adjacency_bonus"] += adjacent_cell["bonus"] * weight
                            elif cell["type"] == ADJACENCY_BONUS and adjacent["type"] == ADJACENCY_CORE:
                                if cell["adjacency"] == ADJACENCY_GREATER:
                                    adjacent_cell["adjacency_bonus"] += cell["bonus"] * core_weight_greater
                                elif cell["adjacency"] == ADJACENCY_LESSER:
                                    adjacent_cell["adjacency_bonus"] += cell["bonus"] * core_weight_lesser
                            elif cell["type"] == ADJACENCY_CORE and adjacent["type"] == ADJACENCY_BONUS:
                                if cell["adjacency"] == ADJACENCY_GREATER:
                                    adjacent_cell["adjacency_bonus"] += cell["bonus"] * core_weight_greater
                                elif cell["adjacency"] == ADJACENCY_LESSER:
                                    adjacent_cell["adjacency_bonus"] += cell["bonus"] * core_weight_lesser
                            elif cell["type"] == ADJACENCY_CORE and adjacent["type"] == ADJACENCY_CORE:
                                if bonus_type == (ADJACENCY_LESSER, ADJACENCY_LESSER) or bonus_type == (
                                    ADJACENCY_GREATER,
                                    ADJACENCY_GREATER,
                                ):
                                    adjacent_cell["adjacency_bonus"] += cell["bonus"] * weight
                                    cell["adjacency_bonus"] += adjacent_cell["bonus"] * weight
                                elif bonus_type == (ADJACENCY_LESSER, ADJACENCY_GREATER):
                                    cell["adjacency_bonus"] += adjacent_cell["bonus"] * weight
                                elif bonus_type == (ADJACENCY_GREATER, ADJACENCY_LESSER):
                                    adjacent_cell["adjacency_bonus"] += cell["bonus"] * weight


def populate_module_bonuses(grid: Grid, x: int, y: int) -> float:
    """Calculates the total bonus for a module at a given position, considering supercharging."""
    cell = grid.get_cell(x, y)

    base_bonus = cell["bonus"]
    is_supercharged = cell["supercharged"]
    is_sc_eligible = cell["sc_eligible"]
    adjacency_bonus = cell["adjacency_bonus"]

    total_bonus = base_bonus + adjacency_bonus

    if is_supercharged and is_sc_eligible:
        total_bonus *= 1.50  # Apply supercharge multiplie

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
    if grid is None:
        return 0.0  # Or handle the error in a way that makes sense for your application

    clear_scores(grid, tech)
    calculate_adjacency_bonus(grid, tech)
    populate_all_module_bonuses(grid, tech)

    total_grid_score = 0
    base_bonus_sum = 0  # Initialize the sum of base bonuses
    bonus_multiplier = 1.0  # Initialize as 1.0 for multiplication

    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech:
                base_bonus_sum += cell["bonus"]  # Sum the base bonuses
                bonus_multiplier *= 1.0 + cell["bonus"]  # Multiply by (1 + bonus) to handle 0 bonuses
                total_grid_score += cell["total"]  # Sum all module total bonuses

    # Apply the bonus_multiplier to the total_grid_score
    total_grid_score *= bonus_multiplier
    total_grid_score += base_bonus_sum  # Add the base bonus sum to the total grid score

    return round(total_grid_score, 8)  # Round to 8 decimal places


def clear_scores(grid: Grid, tech: str) -> None:
    """Clears the total scores for all modules of a given technology."""
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech:
                grid.set_total(x, y, 0)
                cell["adjacency_bonus"] = 0
