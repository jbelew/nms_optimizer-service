"""
This module provides functions for pattern matching and manipulation.

It includes utilities for rotating and mirroring patterns, applying them to a
grid, generating all unique variations of a pattern, and calculating a
heuristic score for a pattern's placement. This is used to find optimal
layouts based on pre-defined "solve maps".
"""

from .module_placement import place_module, clear_all_modules_of_tech
from .modules_utils import get_tech_modules
import logging


def rotate_pattern(pattern):
    """Rotates a pattern 90 degrees clockwise.

    Args:
        pattern (dict): A dictionary where keys are (x, y) tuples representing
            coordinates and values are module IDs.

    Returns:
        dict: The rotated pattern.
    """
    x_coords = [coord[0] for coord in pattern.keys()]
    y_coords = [coord[1] for coord in pattern.keys()]
    if not x_coords or not y_coords:
        return {}  # Return empty dict if pattern is empty
    max_x = max(x_coords)
    rotated_pattern = {}
    for (x, y), module_label in pattern.items():
        new_x = y
        new_y = max_x - x
        rotated_pattern[(new_x, new_y)] = module_label
    return rotated_pattern


def mirror_pattern_horizontally(pattern):
    """Mirrors a pattern horizontally.

    Args:
        pattern (dict): A dictionary where keys are (x, y) tuples representing
            coordinates and values are module IDs.

    Returns:
        dict: The horizontally mirrored pattern.
    """
    x_coords = [coord[0] for coord in pattern.keys()]
    if not x_coords:
        return {}
    max_x = max(x_coords)
    mirrored_pattern = {}
    for (x, y), module_label in pattern.items():
        new_x = max_x - x
        mirrored_pattern[(new_x, y)] = module_label
    return mirrored_pattern


def mirror_pattern_vertically(pattern):
    """Mirrors a pattern vertically.

    Args:
        pattern (dict): A dictionary where keys are (x, y) tuples representing
            coordinates and values are module IDs.

    Returns:
        dict: The vertically mirrored pattern.
    """
    y_coords = [coord[1] for coord in pattern.keys()]
    if not y_coords:
        return {}
    max_y = max(y_coords)
    mirrored_pattern = {}
    for (x, y), module_label in pattern.items():
        new_y = max_y - y
        mirrored_pattern[(x, new_y)] = module_label
    return mirrored_pattern


def apply_pattern_to_grid(
    grid,
    pattern,
    modules,
    tech,
    start_x,
    start_y,
    ship,
    tech_modules=None,
):
    """Applies a pattern to a copy of the grid at a given starting position.

    This function checks if a given pattern can be placed on the grid without
    conflicts (e.g., overlapping existing modules of a different tech, placing
    on inactive cells). It only places modules that the player owns.

    Args:
        grid (Grid): The current grid state.
        pattern (dict): The pattern to apply, with (x, y) coordinates as keys.
        modules (dict): The complete module data for the ship.
        tech (str): The technology key for the modules being placed.
        start_x (int): The starting x-coordinate on the grid to apply the pattern.
        start_y (int): The starting y-coordinate on the grid to apply the pattern.
        ship (str): The ship key.
        tech_modules (list, optional): A pre-filtered list of available modules for this tech.

    Returns:
        tuple[Grid, int] or tuple[None, int]: A tuple containing the new grid
            with the pattern applied and the calculated adjacency score, or
            (None, 0) if the pattern cannot be applied.
    """
    # Create a deep copy of the grid to avoid modifying the original
    new_grid = grid.copy()

    if tech_modules is None:
        tech_modules_available = get_tech_modules(modules, ship, tech)
    else:
        tech_modules_available = tech_modules
    if tech_modules_available is None:  # Should not happen if filter_solves worked correctly
        logging.error(f"No modules found for ship '{ship}' and tech '{tech}'.")
        return None, 0
    # Create a mapping from module id to module data
    available_module_ids_map = {m["id"]: m for m in tech_modules_available}

    # --- Pre-check 1: Determine if the pattern is even placeable by the player and fits basic constraints ---
    # Count how many modules the pattern *intends* to place that the player actually owns.
    expected_module_placements_in_pattern = 0
    for module_id_in_pattern_val in pattern.values():  # Iterate through values (module IDs)
        if module_id_in_pattern_val is not None and module_id_in_pattern_val in available_module_ids_map:
            expected_module_placements_in_pattern += 1

    # If the pattern has defined module IDs, but none are owned by the player, this pattern is not applicable.
    if expected_module_placements_in_pattern == 0 and any(pid is not None for pid in pattern.values()):
        return None, 0

    # --- Pre-check 2: Check for overlaps, off-grid, or inactive cells for REQUIRED modules ---
    for pattern_x, pattern_y in pattern.keys():  # Iterate through keys (coordinates)
        module_id_in_pattern = pattern.get((pattern_x, pattern_y))
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y

        # Is this part of the pattern trying to place an owned module?
        if module_id_in_pattern is not None and module_id_in_pattern in available_module_ids_map:
            if not (0 <= grid_x < new_grid.width and 0 <= grid_y < new_grid.height):
                # A required module (owned, non-None) would be off-grid. This pattern variation doesn't fit.
                return None, 0

            current_cell_on_new_grid = new_grid.get_cell(grid_x, grid_y)
            if not current_cell_on_new_grid["active"]:
                # Cannot place a required module on an inactive cell.
                return None, 0

            if current_cell_on_new_grid["module"] is not None and current_cell_on_new_grid["tech"] != tech:
                # Overlap with a module of a *different* technology.
                return None, 0
        # If module_id_in_pattern is None, or not in available_module_ids_map, we don't check its target cell strictly here,
        # as it won't be placed anyway or it's an intentionally empty slot.

    # If all pre-checks pass, proceed with actual placement attempt
    clear_all_modules_of_tech(new_grid, tech)  # Clear target tech modules for a clean placement

    successfully_placed_this_variation = 0
    for (pattern_x, pattern_y), module_id_in_pattern in pattern.items():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y

        if not (0 <= grid_x < new_grid.width and 0 <= grid_y < new_grid.height):
            continue  # Skip parts of the pattern that are off-grid

        if module_id_in_pattern is None:  # Intentionally empty slot in pattern
            continue

        if module_id_in_pattern in available_module_ids_map:
            module_data = available_module_ids_map[module_id_in_pattern]
            # Cell activity and non-overlap with *other* tech already confirmed by pre-checks for these modules.
            # The cell is also guaranteed to be empty of the *target* tech due to clear_all_modules_of_tech.
            place_module(
                new_grid,
                grid_x,
                grid_y,
                module_data["id"],
                module_data["label"],
                tech,
                module_data["type"],
                module_data["bonus"],
                module_data["adjacency"],
                module_data["sc_eligible"],
                module_data["image"],
            )
            successfully_placed_this_variation += 1

    # --- Post-placement Check: Did we place all *expected* modules? ---
    # This ensures that if the pattern expected to place modules (that player owns), they were all placed.
    # If `expected_module_placements_in_pattern` is 0 (e.g. pattern was all "None", or all unowned modules), this check is skipped.
    if (
        expected_module_placements_in_pattern > 0
        and successfully_placed_this_variation < expected_module_placements_in_pattern
    ):
        # This implies some expected modules (owned, non-None in pattern) couldn't be placed.
        # This should ideally be caught by pre-checks (e.g. inactive cell, off-grid).
        return None, 0

    # If we reach here, the pattern (or what's placeable from it according to ownership and grid constraints) was applied.
    adjacency_score = calculate_pattern_adjacency_score(new_grid, tech)
    return new_grid, adjacency_score  # Return the new grid


def get_all_unique_pattern_variations(original_pattern):
    """Generates all unique variations of a pattern (rotations and mirrors).

    This function creates a comprehensive list of all possible orientations
    for a given pattern to ensure all potential fits are checked.

    Args:
        original_pattern (dict): The base pattern to transform.

    Returns:
        list[dict]: A list of unique pattern dictionaries.
    """
    patterns_to_try = [original_pattern]
    rotated_patterns = set()
    mirrored_patterns = set()

    rotated_pattern_90 = rotate_pattern(original_pattern)
    if rotated_pattern_90 != original_pattern:
        if tuple(rotated_pattern_90.items()) not in rotated_patterns:
            patterns_to_try.append(rotated_pattern_90)
            rotated_patterns.add(tuple(rotated_pattern_90.items()))
            rotated_pattern_180 = rotate_pattern(rotated_pattern_90)
            if rotated_pattern_180 != original_pattern and tuple(rotated_pattern_180.items()) not in rotated_patterns:
                patterns_to_try.append(rotated_pattern_180)
                rotated_patterns.add(tuple(rotated_pattern_180.items()))
                rotated_pattern_270 = rotate_pattern(rotated_pattern_180)
                if (
                    rotated_pattern_270 != original_pattern
                    and tuple(rotated_pattern_270.items()) not in rotated_patterns
                ):
                    patterns_to_try.append(rotated_pattern_270)
                    rotated_patterns.add(tuple(rotated_pattern_270.items()))

    # Add mirrored patterns
    for pattern in list(patterns_to_try):
        mirrored_horizontal = mirror_pattern_horizontally(pattern)
        if tuple(mirrored_horizontal.items()) not in mirrored_patterns:
            patterns_to_try.append(mirrored_horizontal)
            mirrored_patterns.add(tuple(mirrored_horizontal.items()))
        mirrored_vertical = mirror_pattern_vertically(pattern)
        if tuple(mirrored_vertical.items()) not in mirrored_patterns:
            patterns_to_try.append(mirrored_vertical)
            mirrored_patterns.add(tuple(mirrored_vertical.items()))

    return patterns_to_try


def calculate_pattern_adjacency_score(grid, tech):
    """Calculates a heuristic adjacency score for a placed pattern.

    This score is not the final bonus calculation but a heuristic used to
    quickly evaluate the quality of a pattern's placement during the
    pattern matching phase. It rewards adjacencies to other modules, grid edges,
    and modules within the same adjacency group.

    Args:
        grid (Grid): The grid with the pattern applied.
        tech (str): The technology type of the modules to consider.

    Returns:
        float: The calculated adjacency score.
    """
    module_edge_weight = 3.0  # Weight for adjacency to other modules
    grid_edge_weight = 0.5  # Weight for adjacency to grid edges
    group_adjacency_weight = 5.0  # Weight for adjacency to modules in the same group

    total_adjacency_score = 0

    # Iterate through the grid to find modules of the specified tech
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["module"] is not None and cell["tech"] == tech:
                # Check each edge individually and apply grid_edge_weight
                if x == 0:
                    total_adjacency_score += grid_edge_weight  # Left edge
                if x == grid.width - 1:
                    total_adjacency_score += grid_edge_weight  # Right edge
                if y == 0:
                    total_adjacency_score += grid_edge_weight  # Top edge
                if y == grid.height - 1:
                    total_adjacency_score += grid_edge_weight  # Bottom edge

                # --- Adjacency Checks ---
                num_adjacent_same_group_other_tech = 0
                adjacency_rule = cell.get("adjacency")
                adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

                for adj_x, adj_y in adjacent_positions:
                    if 0 <= adj_x < grid.width and 0 <= adj_y < grid.height:
                        adjacent_cell = grid.get_cell(adj_x, adj_y)
                        if adjacent_cell["module"] is not None:
                            # Standard adjacency bonus for being next to other modules of a different tech
                            if adjacent_cell["tech"] != tech:
                                total_adjacency_score += module_edge_weight

                                # Count for group adjacency bonus if adjacency rules match
                                if adjacency_rule and adjacent_cell.get("adjacency") == adjacency_rule:
                                    num_adjacent_same_group_other_tech += 1

                # Check for group adjacency bonus from "greater_n" or "lesser_n" rules
                if isinstance(adjacency_rule, str) and "_" in adjacency_rule:
                    parts = adjacency_rule.split("_")
                    if len(parts) == 2 and parts[1].isdigit():
                        rule_type = parts[0]
                        rule_value = int(parts[1])

                        # Apply bonus based on the rule
                        if rule_type == "greater" and num_adjacent_same_group_other_tech > rule_value:
                            total_adjacency_score += group_adjacency_weight * num_adjacent_same_group_other_tech
                        elif rule_type == "lesser" and num_adjacent_same_group_other_tech < rule_value:
                            total_adjacency_score += group_adjacency_weight * (
                                rule_value - num_adjacent_same_group_other_tech
                            )

    return total_adjacency_score


def _extract_pattern_from_grid(grid, tech):
    """
    Extracts a normalized pattern (relative coordinates) from a grid for a specific technology.

    Args:
        grid (Grid): The grid from which to extract the pattern.
        tech (str): The technology key to filter modules.

    Returns:
        dict: A dictionary where keys are (x, y) tuples representing relative
              coordinates and values are module IDs, or an empty dict if no modules found.
    """
    modules_in_pattern = {}
    min_x, min_y = float("inf"), float("inf")

    # Find all modules of the specified tech and determine min_x, min_y
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["module"] is not None and cell["tech"] == tech:
                modules_in_pattern[(x, y)] = cell["module"]
                min_x = min(min_x, x)
                min_y = min(min_y, y)

    if not modules_in_pattern:
        return {}  # No modules of this tech found

    # Normalize coordinates to be relative to (min_x, min_y)
    normalized_pattern = {}
    for (x, y), module_id in modules_in_pattern.items():
        normalized_pattern[(x - min_x, y - min_y)] = module_id

    return normalized_pattern
