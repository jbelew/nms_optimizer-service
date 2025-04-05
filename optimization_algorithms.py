# optimization_algorithms.py
from grid_utils import Grid
from modules_data import get_tech_modules
from grid_display import print_grid_compact, print_grid
from bonus_calculations import calculate_grid_score
from module_placement import (
    place_module,
    clear_all_modules_of_tech,
)  # Import from module_placement
from simulated_annealing import simulated_annealing
from itertools import permutations
import random
from copy import deepcopy
from modules import (
    solves,
)
from solve_map_utils import filter_solves  # Import the new function
from sse_events import sse_message


def refine_placement(grid, ship, modules, tech, player_owned_rewards=None):
    optimal_grid = None
    highest_bonus = 0.0
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)

    if tech_modules is None:
        print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
        return None, 0.0

    # Precompute available positions for fast access
    available_positions = [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.get_cell(x, y)["module"] is None and grid.get_cell(x, y)["active"]
    ]

    # Check if there are enough available positions for all modules
    if len(available_positions) < len(tech_modules):
        return None, 0.0

    # Initialize the iteration counter
    iteration_count = 0

    # Generate all permutations of module placements
    for placement in permutations(available_positions, len(tech_modules)):
        # Shuffle the tech_modules list for each permutation
        shuffled_tech_modules = tech_modules[:]  # Create a copy to avoid modifying the original list
        random.shuffle(shuffled_tech_modules)

        # Increment the iteration counter
        iteration_count += 1

        # Clear all modules of the selected technology - MOVED BEFORE PLACEMENT
        clear_all_modules_of_tech(grid, tech)

        # Place all modules in the current permutation
        for index, (x, y) in enumerate(placement):
            module = shuffled_tech_modules[index]
            place_module(
                grid,
                x,
                y,
                module["id"],
                module["label"],
                tech,
                module["type"],
                module["bonus"],
                module["adjacency"],
                module["sc_eligible"],
                module["image"],
            )

        # Calculate the score for the current arrangement - MOVED OUTSIDE THE INNER LOOP
        grid_bonus = calculate_grid_score(grid, tech)

        # Update the best grid if a better score is found - MOVED OUTSIDE THE INNER LOOP
        if grid_bonus > highest_bonus:
            highest_bonus = grid_bonus
            optimal_grid = deepcopy(grid)
            # print(highest_bonus)
            # print_grid_compact(optimal_grid)

    # Print the total number of iterations
    print(f"INFO -- refine_placement completed {iteration_count} iterations for ship: '{ship}' -- tech: '{tech}'")

    return optimal_grid, highest_bonus


def rotate_pattern(pattern):
    """Rotates a pattern 90 degrees clockwise."""
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
    """Mirrors a pattern horizontally."""
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
    """Mirrors a pattern vertically."""
    y_coords = [coord[1] for coord in pattern.keys()]
    if not y_coords:
        return {}
    max_y = max(y_coords)
    mirrored_pattern = {}
    for (x, y), module_label in pattern.items():
        new_y = max_y - y
        mirrored_pattern[(x, new_y)] = module_label
    return mirrored_pattern


def apply_pattern_to_grid(grid, pattern, modules, tech, start_x, start_y, ship, player_owned_rewards=None):
    """Applies a pattern to a *copy* of the grid at a given starting position.

    Returns a new grid with the pattern applied, or None if the pattern cannot be applied.
    """
    # Create a deep copy of the grid to avoid modifying the original
    new_grid = grid.copy()

    # Check for overlap before applying the pattern
    for pattern_x, pattern_y in pattern.keys():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y
        if 0 <= grid_x < new_grid.width and 0 <= grid_y < new_grid.height:
            if (
                new_grid.get_cell(grid_x, grid_y)["module"] is not None
                and new_grid.get_cell(grid_x, grid_y)["tech"] != tech
            ):
                return None, 0  # Indicate a bad pattern with a score of 0

    # Clear existing modules of the selected technology in the new grid
    clear_all_modules_of_tech(new_grid, tech)

    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None:
        print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
        return None, 0

    # Create a mapping from module id to module data
    module_id_map = {module["id"]: module for module in tech_modules}

    for (pattern_x, pattern_y), module_id in pattern.items():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y

        if 0 <= grid_x < new_grid.width and 0 <= grid_y < new_grid.height:
            if module_id is None:
                continue
            if module_id in module_id_map:
                module_data = module_id_map[module_id]
                if new_grid.get_cell(grid_x, grid_y)["active"] and new_grid.get_cell(grid_x, grid_y)["module"] is None:
                    place_module(
                        new_grid,  # Apply changes to the new grid
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

    adjacency_score = calculate_pattern_adjacency_score(new_grid, tech)
    return new_grid, adjacency_score  # Return the new grid


def get_all_unique_pattern_variations(original_pattern):
    """
    Generates all unique variations of a pattern (rotations and mirrors).

    Args:
        original_pattern (dict): The original pattern.

    Returns:
        list: A list of unique pattern variations.
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


def count_adjacent_occupied(grid, x, y):
    """Counts the number of adjacent occupied slots to a given cell."""
    count = 0
    adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    for nx, ny in adjacent_positions:
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            if grid.get_cell(nx, ny)["module"] is not None:
                count += 1
    return count


def calculate_pattern_adjacency_score(grid, tech):
    """
    Calculates the adjacency score for modules of a specific tech on the grid.

    Args:
        grid (Grid): The grid.
        tech (str): The tech type of the modules to consider.

    Returns:
        int: The adjacency score.
    """
    module_edge_weight = 2.0  # Weight for adjacency to other modules
    grid_edge_weight = 1.0  # Weight for adjacency to grid edges

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

                # Check adjacent positions for modules of different techs
                adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                for adj_x, adj_y in adjacent_positions:
                    if 0 <= adj_x < grid.width and 0 <= adj_y < grid.height:
                        adjacent_cell = grid.get_cell(adj_x, adj_y)
                        if adjacent_cell["module"] is not None and adjacent_cell["tech"] != tech:
                            total_adjacency_score += module_edge_weight

    return total_adjacency_score


def optimize_placement(grid, ship, modules, tech, player_owned_rewards=None, message_queue=None):
    """
    Optimizes the placement of modules in a grid for a specific ship and technology.
    ... (rest of the docstring)
    """
    print(f"INFO -- Attempting solve for ship: '{ship}' -- tech: '{tech}'")

    if player_owned_rewards is None:
        player_owned_rewards = []

    # --- Early Check: Any Empty, Active Slots? ---
    has_empty_active_slots = False
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.get_cell(x, y)["active"] and grid.get_cell(x, y)["module"] is None:
                has_empty_active_slots = True
                break
        if has_empty_active_slots:
            break
    if not has_empty_active_slots:
        raise ValueError(f"No empty, active slots available on the grid for ship: '{ship}' -- tech: '{tech}'.")

    best_grid = Grid.from_dict(grid.to_dict())
    best_bonus = -float("inf")

    solved_grid = Grid.from_dict(grid.to_dict())
    solved_bonus = -float("inf")

    best_pattern_grid = Grid.from_dict(grid.to_dict())
    highest_pattern_bonus = -float("inf")
    best_pattern_adjacency_score = 0

    # Filter the solves dictionary based on player-owned rewards
    filtered_solves = filter_solves(solves, ship, modules, tech, player_owned_rewards)

    # --- Special Case: No Solve Available ---
    if ship not in filtered_solves or (ship in filtered_solves and tech not in filtered_solves[ship]):
        print(f"INFO -- No solve found for ship: '{ship}' -- tech: '{tech}'. Placing modules in empty slots.")
        solved_grid = place_all_modules_in_empty_slots(grid, modules, ship, tech, player_owned_rewards)
        solved_bonus = calculate_grid_score(solved_grid, tech)
        solve_score = 0
        pattern_applied = True
        return solved_grid, solved_bonus  # Add this line to exit early

    else:
        pattern_applied = False
        solve_data = filtered_solves[ship][tech]
        original_pattern = solve_data["map"]
        solve_score = solve_data["score"]

        # Generate all unique pattern variations
        patterns_to_try = get_all_unique_pattern_variations(original_pattern)

        # Create a temporary grid outside the pattern loop
        grid_dict = grid.to_dict()
        if grid_dict is None:
            print("Error: grid.to_dict() returned None")
            return best_grid, best_bonus
        temp_grid = Grid.from_dict(grid_dict)
        if temp_grid is None:
            print("Error: Grid.from_dict() returned None")
            return best_grid, best_bonus

        for pattern in patterns_to_try:
            x_coords = [coord[0] for coord in pattern.keys()]
            y_coords = [coord[1] for coord in pattern.keys()]
            if not x_coords or not y_coords:
                continue
            pattern_width = max(x_coords) + 1
            pattern_height = max(y_coords) + 1

            # Try placing the pattern in all possible positions
            for start_x in range(grid.width - pattern_width + 1):
                for start_y in range(grid.height - pattern_height + 1):
                    # Apply the pattern to the persistent temp_grid
                    temp_result_grid, adjacency_score = apply_pattern_to_grid(
                        temp_grid,
                        pattern,
                        modules,
                        tech,
                        start_x,
                        start_y,
                        ship,
                        player_owned_rewards,
                    )
                    if temp_result_grid is not None:
                        current_pattern_bonus = calculate_grid_score(temp_result_grid, tech)


                        if current_pattern_bonus > highest_pattern_bonus:
                            highest_pattern_bonus = current_pattern_bonus
                            best_pattern_grid = temp_result_grid.copy()
                            best_pattern_adjacency_score = adjacency_score
                        elif (
                            current_pattern_bonus == highest_pattern_bonus
                            and adjacency_score >= best_pattern_adjacency_score # Changed > to >=
                        ):
                            best_pattern_grid = temp_result_grid.copy()
                            best_pattern_adjacency_score = adjacency_score


        # Reset temp_grid for the next pattern
        temp_grid = Grid.from_dict(grid_dict)

        if best_pattern_grid:
            # Initialize solved_grid with best_pattern_grid
            solved_grid = best_pattern_grid
            # solved_bonus = highest_pattern_bonus
            solved_bonus = calculate_grid_score(solved_grid, tech)
            print(f"INFO -- Best pattern score: {solved_bonus} for ship: '{ship}' -- tech: '{tech}' that fits.")
            pattern_applied = True
        else:
            print(
                f"WARNING -- No best pattern definition found for ship: '{ship}' -- tech: '{tech}' that fits. Falling back to simulated_annealing."
            )

    # --- 1. Supercharged Opportunity Refinement (Moved to the Beginning) ---
    opportunity = find_supercharged_opportunities(solved_grid, modules, ship, tech)

    if opportunity:
        print(f"INFO -- Found opportunity: {opportunity}")
        # Create a localized grid
        opportunity_x, opportunity_y = opportunity

        # Deep copy solved_grid before clearing and creating localized grid
        temp_solved_grid = deepcopy(solved_grid)
        clear_all_modules_of_tech(temp_solved_grid, tech)
        localized_grid, start_x, start_y = create_localized_grid(temp_solved_grid, opportunity_x, opportunity_y, tech)

        # Refine the localized grid - Surround with id statement
        print_grid(localized_grid)

        # Get the number of modules for the given tech
        tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
        num_modules = len(tech_modules) if tech_modules else 0

        if num_modules < 6:
            print(f"INFO -- {tech} has less than 6 modules, running refine_placement")
            temp_refined_grid, temp_refined_bonus = refine_placement(
                localized_grid, ship, modules, tech, player_owned_rewards
            )
        else:
            print(f"INFO -- {tech} has 6 or more modules, running simulated_annealing")
            temp_refined_grid, temp_refined_bonus = simulated_annealing(
                localized_grid, ship, modules, tech, player_owned_rewards
            )

        temp_refined_bonus = calculate_grid_score(temp_refined_grid, tech)  # Recalculate score after annealing

        if temp_refined_grid is not None:
            print(f"INFO -- Refined grid score: {temp_refined_bonus}")
            print_grid(temp_refined_grid)
            # Apply changes to temp_solved_grid
            apply_localized_grid_changes(temp_solved_grid, temp_refined_grid, tech, start_x, start_y)
            # Calculate the new score of the entire grid
            new_solved_bonus = calculate_grid_score(temp_solved_grid, tech)
            # Compare bonuses and apply changes if the refined bonus is higher

            ######
            #
            #    TODO: Ugly hack for a bugt I can't find!
            #
            ######

            if new_solved_bonus > (solved_bonus * 0.99):
                # if new_solved_bonus > solved_bonus:
                # Copy temp_solved_grid to solved_grid
                solved_grid = temp_solved_grid.copy()
                print(f"INFO -- Better refined grid found for ship: '{ship}' -- tech: '{tech}'")
                solved_bonus = new_solved_bonus
            else:
                print(
                    f"INFO -- Refined grid did not improve the score. Solved Bonus: {solved_bonus} vs Refined Bonus: {temp_refined_bonus}"
                )
        else:
            print("INFO -- Opportunity refinement failed. Not enough space to apply the opportunity.")

    # --- 3. Simulated Annealing (Fallback, Moved to the End) ---
    if not pattern_applied:
        print(
            f"WARNING -- No pattern was applied for ship: '{ship}' -- tech: '{tech}'. Falling back to simulated annealing."
        )
        solved_grid = Grid.from_dict(grid.to_dict())
        clear_all_modules_of_tech(solved_grid, tech)
        temp_solved_grid, temp_solved_bonus = simulated_annealing(
            solved_grid,
            ship,
            modules,
            tech,
            player_owned_rewards,
            initial_temperature=4000,
            cooling_rate=0.98,
            iterations_per_temp=30,
            initial_swap_probability=0.40,
            final_swap_probability=0.3,
        )
        if temp_solved_grid is not None:
            solved_grid = temp_solved_grid
            solved_bonus = calculate_grid_score(solved_grid, tech)  # Recalculate score after annealing
        else:
            print(
                f"ERROR -- simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
            )
            raise ValueError(
                f"simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
            )

    # Check if all modules were placed - Only if a pattern was applied
    all_modules_placed = check_all_modules_placed(solved_grid, modules, ship, tech, player_owned_rewards)

    if not all_modules_placed:
        print(
            f"WARNING! -- Not all modules were placed in grid for ship: '{ship}' -- tech: '{tech}'. Running simulated_annealing solver."
        )

        clear_all_modules_of_tech(solved_grid, tech)
        temp_solved_grid, temp_solved_bonus = simulated_annealing(
            solved_grid,
            ship,
            modules,
            tech,
            player_owned_rewards,
            initial_temperature=2000,
            cooling_rate=0.98,
            iterations_per_temp=20,
        )
        if temp_solved_grid is not None:
            solved_grid = temp_solved_grid
            solved_bonus = calculate_grid_score(solved_grid, tech)  # Recalculate score after annealing
        else:
            print(
                f"ERROR -- simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
            )
            raise ValueError(
                f"simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
            )

    # Recalculate solved_bonus after applying the pattern or falling back to simulated_annealing
    solved_bonus = calculate_grid_score(solved_grid, tech)

    # Calculate the percentage of the solve score achieved
    if solve_score > 0:
        percentage = (solved_bonus / solve_score) * 100  # Use solved_bonus
    else:
        percentage = 0

    if solved_grid is not None:
        best_grid = solved_grid
        best_bonus = solved_bonus
        print(
            f"SUCCESS -- Percentage of Solve Score Achieved: {percentage:.2f}% (Current Score: {best_bonus:.2f}, Adjacency Score: {best_pattern_adjacency_score:.2f}) for ship: '{ship}' -- tech: '{tech}'"
        )
    else:
        print(f"ERROR -- No valid grid could be generated for ship: '{ship}' -- tech: '{tech}'")

    return best_grid, percentage



def place_all_modules_in_empty_slots(grid, modules, ship, tech, player_owned_rewards=None):
    """Places all modules of a given tech in any remaining empty slots, going column by column."""
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None:
        print(f"ERROR --  No modules found for ship: '{ship}' -- tech: '{tech}'")
        return grid

    module_index = 0  # Keep track of the current module to place

    for x in range(grid.width):  # Iterate through columns first
        for y in range(grid.height):  # Then iterate through rows
            if module_index >= len(tech_modules):
                return grid  # All modules placed, exit early

            if grid.get_cell(x, y)["module"] is None and grid.get_cell(x, y)["active"]:
                module = tech_modules[module_index]
                place_module(
                    grid,
                    x,
                    y,
                    module["id"],
                    module["label"],
                    tech,
                    module["type"],
                    module["bonus"],
                    module["adjacency"],
                    module["sc_eligible"],
                    module["image"],
                )
                module_index += 1  # Move to the next module

    if module_index < len(tech_modules) and len(tech_modules) > 0:
        print(f"WARNING -- Not enough space to place all modules for ship: '{ship}' -- tech: '{tech}'")

    return grid


def count_empty_in_localized(localized_grid):
    """Counts the number of truly empty slots in a localized grid."""
    count = 0
    for y in range(localized_grid.height):
        for x in range(localized_grid.width):
            cell = localized_grid.get_cell(x, y)
            if cell["module"] is None:  # Only count if the module slot is empty
                count += 1
    return count


def find_supercharged_opportunities(grid, modules, ship, tech):
    """
    Scans the entire grid with a sliding window to find the highest-scoring
    window containing available supercharged slots. Prioritizes supercharged
    slots away from the edges of the window.

    Args:
        grid (Grid): The current grid layout.
        modules (dict): The module data.
        ship (str): The ship type.
        tech (str): The technology type.

    Returns:
        tuple or None: A tuple (opportunity_x, opportunity_y) representing the
                       top-left corner of the best window, or None if no
                       suitable window is found or if all supercharged slots
                       are occupied.
    """
    grid_copy = grid.copy()
    clear_all_modules_of_tech(grid_copy, tech)

    # Check if there are any unoccupied supercharged slots
    unoccupied_supercharged_slots = False
    for y in range(grid_copy.height):
        for x in range(grid_copy.width):
            cell = grid_copy.get_cell(x, y)
            if cell["supercharged"] and cell["module"] is None and cell["active"]:
                unoccupied_supercharged_slots = True
                break
        if unoccupied_supercharged_slots:
            break

    if not unoccupied_supercharged_slots:
        print("INFO -- No unoccupied supercharged slots found.")
        return None  # Return None if all supercharged slots are occupied

    window_width = 4
    window_height = 3

    best_window_score = -1
    best_window_start_x, best_window_start_y = None, None

    for start_y in range(grid_copy.height - window_height + 1):
        for start_x in range(grid_copy.width - window_width + 1):
            window_grid = Grid(window_width, window_height)
            for y in range(window_height):
                for x in range(window_width):
                    grid_x = start_x + x
                    grid_y = start_y + y
                    cell = grid_copy.get_cell(grid_x, grid_y)
                    window_grid.cells[y][x] = cell.copy()

            # Check if the window has at least one available supercharged slot
            has_available_supercharged = False
            for y in range(window_height):
                for x in range(window_width):
                    cell = window_grid.get_cell(x, y)
                    if cell["supercharged"] and cell["module"] is None and cell["active"]:
                        has_available_supercharged = True
                        break
                if has_available_supercharged:
                    break

            if not has_available_supercharged:
                continue  # Skip this window if it doesn't have an available supercharged slot

            # Check if the number of available cells in the current window is less than the number of modules
            tech_modules = get_tech_modules(modules, ship, tech)
            if tech_modules is None:
                print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
                return None

            available_cells_in_window = 0
            for y in range(window_height):
                for x in range(window_width):
                    cell = window_grid.get_cell(x, y)
                    if cell["active"] and cell["module"] is None:
                        available_cells_in_window += 1

            if available_cells_in_window < len(tech_modules):
                # print(f"INFO -- Not enough available cells in the window ({available_cells_in_window}) for all modules ({len(tech_modules)}). Skipping this window.")
                continue  # Skip this window and move to the next one

            window_score = calculate_window_score(window_grid, tech)
            if window_score > best_window_score:
                best_window_score = window_score
                best_window_start_x, best_window_start_y = start_x, start_y

    if best_window_start_x is not None and best_window_start_y is not None:
        return best_window_start_x, best_window_start_y  # Return the top-left of the best window
    else:
        return None

def calculate_window_score(window_grid, tech):
    """Calculates a score for a given window based on supercharged and empty slots,
    excluding inactive cells. Prioritizes supercharged slots away from the horizontal edges of the window.
    """
    supercharged_count = 0
    empty_count = 0
    edge_penalty = 0
    for y in range(window_grid.height):
        for x in range(window_grid.width):
            cell = window_grid.get_cell(x, y)
            if cell["active"]:  # Only consider active cells
                if cell["supercharged"]:
                    # Check if the supercharged cell is empty or occupied by the current tech
                    if cell["module"] is None or cell["tech"] == tech:
                        supercharged_count += 1
                        # Check if the supercharged slot is on the horizontal edge of the window
                        if x == 0 or x == window_grid.width - 1:
                            edge_penalty += 1  # Apply a penalty for edge supercharged slots
                if cell["module"] is None:
                    empty_count += 1

    # Prioritize supercharged slots, then empty slots, and penalize edge supercharged slots
    return (supercharged_count * 3) + (empty_count * 1) - (edge_penalty * 0.5)


def create_localized_grid(grid, opportunity_x, opportunity_y, tech):
    """
    Creates a localized grid around a given opportunity, ensuring it stays within
    the bounds of the main grid and preserves modules of other tech types.
    Now directly uses the opportunity point as the top-left corner, with clamping.

    Args:
        grid (Grid): The main grid.
        opportunity_x (int): The x-coordinate of the opportunity (top-left corner).
        opportunity_y (int): The y-coordinate of the opportunity (top-left corner).
        tech (str): The technology type being optimized.

    Returns:
        tuple: A tuple containing:
            - localized_grid (Grid): The localized grid.
            - start_x (int): The starting x-coordinate of the localized grid in the main grid.
            - start_y (int): The starting y-coordinate of the localized grid in the main grid.
    """
    localized_width = 4
    localized_height = 3

    # Directly use opportunity_x and opportunity_y as the starting position
    start_x = opportunity_x
    start_y = opportunity_y

    # Clamp the starting position to ensure it's within the grid bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    # Calculate the end position based on the clamped start position
    end_x = min(grid.width, start_x + localized_width)
    end_y = min(grid.height, start_y + localized_height)

    # Adjust the localized grid size based on the clamped bounds
    actual_localized_width = end_x - start_x
    actual_localized_height = end_y - start_y

    # Create the localized grid with the adjusted size
    localized_grid = Grid(actual_localized_width, actual_localized_height)

    # Copy the grid structure (active/inactive, supercharged) AND module data
    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            localized_x = x - start_x
            localized_y = y - start_y
            cell = grid.get_cell(x, y)
            localized_grid.cells[localized_y][localized_x]["active"] = cell["active"]
            localized_grid.cells[localized_y][localized_x]["supercharged"] = cell["supercharged"]

            # Copy module data if a module exists
            if cell["module"] is not None:
                localized_grid.cells[localized_y][localized_x]["module"] = cell["module"]
                localized_grid.cells[localized_y][localized_x]["label"] = cell["label"]
                localized_grid.cells[localized_y][localized_x]["tech"] = cell["tech"]
                localized_grid.cells[localized_y][localized_x]["type"] = cell["type"]
                localized_grid.cells[localized_y][localized_x]["bonus"] = cell["bonus"]
                localized_grid.cells[localized_y][localized_x]["adjacency"] = cell["adjacency"]
                localized_grid.cells[localized_y][localized_x]["sc_eligible"] = cell["sc_eligible"]
                localized_grid.cells[localized_y][localized_x]["image"] = cell["image"]
                # Only copy module_position if it exists
                if "module_position" in cell:
                    localized_grid.cells[localized_y][localized_x]["module_position"] = cell["module_position"]

    return localized_grid, start_x, start_y


def apply_localized_grid_changes(grid, localized_grid, tech, start_x, start_y):
    """Applies changes from the localized grid back to the main grid."""
    localized_width = localized_grid.width
    localized_height = localized_grid.height

    # Copy module placements from the localized grid back to the main grid
    for y in range(localized_height):
        for x in range(localized_width):
            main_x = start_x + x
            main_y = start_y + y
            if 0 <= main_x < grid.width and 0 <= main_y < grid.height:
                # Only copy if the cell is empty or of the same tech
                if grid.get_cell(main_x, main_y)["tech"] == tech or grid.get_cell(main_x, main_y)["module"] is None:
                    grid.cells[main_y][main_x].update(localized_grid.cells[y][x])


def check_all_modules_placed(grid, modules, ship, tech, player_owned_rewards=None):
    """
    Checks if all modules for a given tech have been placed in the grid.

    Args:
        grid (Grid): The grid layout.
        modules (dict): The module data.
        ship (str): The ship type.
        tech (str): The technology type.
        player_owned_rewards (list, optional): Rewards owned by the player. Defaults to None.

    Returns:
        bool: True if all modules are placed, False otherwise.
    """
    if player_owned_rewards is None:
        player_owned_rewards = []

    # Get the filtered list of modules
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)

    placed_module_ids = set()

    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech and cell["module"]:
                placed_module_ids.add(cell["module"])

    all_module_ids = {module["id"] for module in tech_modules}
    return placed_module_ids == all_module_ids
