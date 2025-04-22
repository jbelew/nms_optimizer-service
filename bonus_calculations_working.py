# bonus_calculations.py
from grid_utils import Grid
import math
from enum import Enum # Import Enum

# --- Constants ---

# Adjacency Types using Enum for better organization and type safety
class AdjacencyType(Enum):
    GREATER = "greater"
    LESSER = "lesser"

# Module Types using Enum
class ModuleType(Enum):
    CORE = "core"
    BONUS = "bonus"

# Weights (These now represent the *factor* by which the base bonus is boosted per interaction)
CORE_WEIGHT_GREATER = 0.080
CORE_WEIGHT_LESSER = 0.02
BONUS_BONUS_GREATER_WEIGHT = 0.060
BONUS_BONUS_LESSER_WEIGHT = 0.020
BONUS_BONUS_MIXED_WEIGHT = 0.02 # Lesser GIVES TO Greater (Boost factor applied to Greater)

# Supercharge Multiplier
SUPERCHARGE_MULTIPLIER = 1.25

# --- Helper Functions ---

def _get_orthogonal_neighbors(grid: Grid, x: int, y: int) -> list[dict]:
    """Gets valid orthogonal neighbor cells with modules."""
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            neighbor_cell = grid.get_cell(nx, ny)
            if neighbor_cell.get("module") is not None:
                # Add coordinates for convenience
                neighbor_cell['x'] = nx
                neighbor_cell['y'] = ny
                neighbors.append(neighbor_cell)
    return neighbors

# --- Core Calculation Functions ---

# REMOVED: calculate_adjacency_bonus function is no longer needed.

def populate_module_bonuses(grid: Grid, x: int, y: int) -> float:
    """
    Calculates the total bonus for a single module.
    REVISED AGAIN: Adjacency effects now BOOST the module's own base bonus.
    The boost amount = base_bonus * interaction_weight_determined_by_neighbor.

    Args:
        grid: The Grid object.
        x: The x-coordinate of the module.
        y: The y-coordinate of the module.

    Returns:
        The calculated total bonus for the module.
    """
    cell = grid.get_cell(x, y)
    if cell.get("module") is None:
        grid.set_total(x, y, 0.0)
        cell["adjacency_bonus"] = 0.0 # Keep clearing unused field
        return 0.0

    base_bonus = cell.get("bonus", 0.0)
    tech = cell.get("tech")
    cell_type = cell.get("type")
    cell_adj_type = cell.get("adjacency")

    # --- Calculate Adjacency Boost ---
    total_adjacency_boost_factor = 0.0 # Accumulate the weights

    adjacent_cells = _get_orthogonal_neighbors(grid, x, y)

    for adj_cell in adjacent_cells:
        # Skip if neighbor is not the same tech
        if adj_cell.get("tech") != tech:
            continue

        # Get neighbor properties needed to determine interaction rules
        adj_cell_adj_type = adj_cell.get("adjacency")
        adj_cell_module_type = adj_cell.get("type")
        # NOTE: adj_cell_base_bonus is NOT needed to calculate the weight anymore

        # --- Determine weight representing the boost factor FROM this neighbor ---
        weight_adj_gives_to_cell = 0.0

        # Interaction is defined from the perspective of the GIVER (adj_cell) to the RECEIVER (cell)
        module_interaction = (adj_cell_module_type, cell_type)

        # --- Case 1: Bonus -> Bonus ---
        if module_interaction == (ModuleType.BONUS.value, ModuleType.BONUS.value):
            # Both giver (adj) and receiver (cell) must have adjacency types defined
            if adj_cell_adj_type and cell_adj_type:
                interaction_key = (adj_cell_adj_type, cell_adj_type) # Giver, Receiver

                if interaction_key == (AdjacencyType.LESSER.value, AdjacencyType.LESSER.value):
                    # Lesser neighbor gives boost factor to Lesser cell
                    weight_adj_gives_to_cell = BONUS_BONUS_LESSER_WEIGHT
                elif interaction_key == (AdjacencyType.GREATER.value, AdjacencyType.GREATER.value):
                    # Greater neighbor gives boost factor to Greater cell
                    weight_adj_gives_to_cell = BONUS_BONUS_GREATER_WEIGHT
                elif interaction_key == (AdjacencyType.LESSER.value, AdjacencyType.GREATER.value):
                    # Lesser neighbor gives boost factor to Greater cell
                    weight_adj_gives_to_cell = BONUS_BONUS_MIXED_WEIGHT
                # elif interaction_key == (AdjacencyType.GREATER.value, AdjacencyType.LESSER.value):
                    # Greater neighbor does NOT give boost factor to Lesser cell, weight remains 0.0

        # --- Case 2: Core -> Bonus ---
        elif module_interaction == (ModuleType.CORE.value, ModuleType.BONUS.value):
            # Check if the receiving Bonus module (cell) can accept adjacency
            if cell_adj_type:
                # Boost factor depends on the GIVING Core's (adj_cell) adjacency type
                if adj_cell_adj_type == AdjacencyType.GREATER.value:
                    weight_adj_gives_to_cell = CORE_WEIGHT_GREATER
                elif adj_cell_adj_type == AdjacencyType.LESSER.value:
                    weight_adj_gives_to_cell = CORE_WEIGHT_LESSER

        # --- Case 3: Bonus -> Core ---
        elif module_interaction == (ModuleType.BONUS.value, ModuleType.CORE.value):
            # Core always receives boost factor based on the GIVING Bonus module's (adj_cell) type
            if adj_cell_adj_type == AdjacencyType.GREATER.value:
                 weight_adj_gives_to_cell = CORE_WEIGHT_GREATER
            elif adj_cell_adj_type == AdjacencyType.LESSER.value:
                 weight_adj_gives_to_cell = CORE_WEIGHT_LESSER

        # --- Accumulate the weight (boost factor) ---
        total_adjacency_boost_factor += weight_adj_gives_to_cell
        # print(f"DEBUG: {cell.get('module')}({x},{y}) gets weight +{weight_adj_gives_to_cell:.4f} from {adj_cell.get('module')}({adj_cell.get('x')},{adj_cell.get('y')})")

    # --- Calculate the final boosted bonus ---
    # Total Boost Amount = Base_Bonus * Sum_of_Weights
    adjacency_boost_amount = base_bonus * total_adjacency_boost_factor
    effective_bonus = base_bonus + adjacency_boost_amount

    # --- End Adjacency Boost Calculation ---

    # Calculate total bonus including supercharge, starting from the effective_bonus
    total_bonus = effective_bonus
    is_supercharged = cell.get("supercharged", False)
    is_sc_eligible = cell.get("sc_eligible", False)
    if is_supercharged and is_sc_eligible:
        total_bonus *= SUPERCHARGE_MULTIPLIER

    # Update the cell's total bonus in the grid
    grid.set_total(x, y, total_bonus)
    # Store the calculated boost amount in the (otherwise unused) adjacency_bonus field for display/debugging
    cell["adjacency_bonus"] = adjacency_boost_amount

    return total_bonus


def populate_all_module_bonuses(grid: Grid, tech: str) -> None:
    """
    Populates the total bonuses for all modules of a given tech in the grid.
    Calls the modified populate_module_bonuses which now includes adjacency calcs.
    """
    # First, reset all totals and the (now repurposed) adjacency_bonus field
    for y_clear in range(grid.height):
        for x_clear in range(grid.width):
            cell_to_clear = grid.get_cell(x_clear, y_clear)
            if cell_to_clear.get("tech") == tech:
                 grid.set_total(x_clear, y_clear, 0.0)
                 cell_to_clear["adjacency_bonus"] = 0.0 # Clear the boost amount field

    # Now, calculate the new total for each module, including adjacency boost effects
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell.get("module") is not None and cell.get("tech") == tech:
                populate_module_bonuses(grid, x, y) # Calculate and set total


def clear_scores(grid: Grid, tech: str) -> None:
    """
    Clears calculated 'total' and the 'adjacency_bonus' (boost amount)
    for modules of a given tech.
    """
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell.get("tech") == tech:
                grid.set_total(x, y, 0.0)
                cell["adjacency_bonus"] = 0.0


def calculate_grid_score(grid: Grid, tech: str) -> float:
    """
    Calculates the total grid score for a given technology by summing module totals.
    Relies on populate_all_module_bonuses to handle adjacency boost internally.

    Args:
        grid: The Grid object.
        tech: The technology type (e.g., "pulse", "hyper") to score.

    Returns:
        The total calculated score for the specified technology, rounded.
    """
    if grid is None:
        print("Warning: calculate_grid_score called with None grid.")
        return 0.0

    # 1. Calculate the final 'total' for each module, including adjacency boost and supercharging.
    populate_all_module_bonuses(grid, tech)

    # 2. Calculate the final grid score by summing the 'total' of each module
    total_grid_score = 0.0
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell.get("tech") == tech:
                total_grid_score += cell.get("total", 0.0) # Sum the final totals

    # Rounding
    return round(total_grid_score, 8)

