# bonus_calculations.py
import logging
from .grid_utils import Grid

# Numba-related imports
from .numba_utils import grid_to_numba_data
from .bonus_calculations_numba import calculate_bonuses_numba


def populate_all_module_bonuses(
    grid: Grid, tech: str, apply_supercharge_first: bool = False
) -> None:
    """
    Calculates and populates the total bonuses for all modules of a
    given tech in the grid using a high-performance Numba implementation.
    Updates 'adjacency_bonus' to store the raw adjacency factor.

    Args:
        grid: The Grid object.
        tech: The technology type to calculate for.
        apply_supercharge_first (bool):
            - If False (default): Calculates boost on base, then applies supercharge
              multiplier to the final sum (base + boost).
            - If True: Applies supercharge multiplier to the base bonus *before*
              calculating adjacency boost amount. The final total is then
              base_bonus + boost_amount (calculated on supercharged base).
    """
    # 1. Convert Grid to Numba-friendly format
    (
        module_grid,
        adjacency_grid,
        type_grid,
        bonus_grid,
        supercharged_grid,
        sc_eligible_grid,
        tech_module_coords,
    ) = grid_to_numba_data(grid, tech)

    if not tech_module_coords:
        # Clear any stale scores if no modules of this tech are found
        clear_scores(grid, tech)
        return

    # Reset scores in the original grid before calculation
    for x, y in tech_module_coords:
        grid.set_total(x, y, 0.0)
        grid.get_cell(x, y)["adjacency_bonus"] = 0.0

    # 2. Call the Numba-jitted calculation function
    total_bonuses, adj_factors = calculate_bonuses_numba(
        tech,
        module_grid,
        adjacency_grid,
        type_grid,
        bonus_grid,
        supercharged_grid,
        sc_eligible_grid,
        tech_module_coords,
        apply_supercharge_first,
    )

    # 3. Update the original Grid object with the results
    for i, (x, y) in enumerate(tech_module_coords):
        grid.set_total(x, y, round(total_bonuses[i], 4))
        grid.get_cell(x, y)["adjacency_bonus"] = round(adj_factors[i], 4)


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


def calculate_grid_score(
    grid: Grid, tech: str, apply_supercharge_first: bool = False
) -> float:
    """
    Calculates the total grid score for a given technology by summing module totals.
    Relies on populate_all_module_bonuses to handle adjacency boost internally,
    passing the apply_supercharge_first flag.

    Args:
        grid: The Grid object.
        tech: The technology type (e.g., "pulse", "hyper") to score.
        apply_supercharge_first (bool): If True, applies supercharge multiplier
                                        to the base bonus before calculating
                                        adjacency boost amount. Defaults to False.

    Returns:
        The total calculated score for the specified technology, rounded.
    """
    if grid is None:
        logging.warning("calculate_grid_score called with None grid.")
        return 0.0

    # Ensure all bonuses are calculated/updated before summing
    populate_all_module_bonuses(grid, tech, apply_supercharge_first)

    total_grid_score = 0.0
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            # Sum only modules of the specified tech that have been placed
            if cell.get("module") is not None and cell.get("tech") == tech:
                total_grid_score += cell.get("total", 0.0)

    return round(total_grid_score, 4)
