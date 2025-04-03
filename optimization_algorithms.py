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
        print(
            f"Not enough available positions to place all modules for ship '{ship}' and tech '{tech}'."
        )
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
            # print_grid(optimal_grid)
    # Print the total number of iterations
    print(
        f"INFO -- refine_placement completed {iteration_count} iterations for ship: '{ship}' -- tech: '{tech}' with score of {highest_bonus}."
    )

    print(f"Refined bonus -- {highest_bonus}")
    print_grid(optimal_grid)

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


def apply_pattern_to_grid(
    grid, pattern, modules, tech, start_x, start_y, ship, player_owned_rewards=None
):
    """Applies a pattern to an existing grid at a given starting position,
    preserving the original grid's state (except for modules of the same tech)
    and only filling empty, active slots with the pattern's modules.
    """
    
    clear_all_modules_of_tech(grid, tech)
    
    # Check for overlap before applying the pattern
    for pattern_x, pattern_y in pattern.keys():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y
        if 0 <= grid_x < grid.width and 0 <= grid_y < grid.height:
            if (
                grid.get_cell(grid_x, grid_y)["module"] is not None
                and grid.get_cell(grid_x, grid_y)["tech"] != tech
            ):
                return 0, 0  # Indicate a bad pattern with a score of 0

    # Clear existing modules of the selected technology
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.get_cell(x, y)["tech"] == tech:
                grid.cells[y][x]["module"] = None
                grid.cells[y][x]["tech"] = None
                grid.cells[y][x]["type"] = None
                grid.cells[y][x]["bonus"] = 0
                grid.cells[y][x]["adjacency"] = False
                grid.cells[y][x]["sc_eligible"] = False
                grid.cells[y][x]["image"] = None

    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None:
        print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
        return 0, 0
    # Create a mapping from module id to module data
    module_id_map = {module["id"]: module for module in tech_modules}

    for (pattern_x, pattern_y), module_id in pattern.items():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y

        if 0 <= grid_x < grid.width and 0 <= grid_y < grid.height:
            if module_id is None:
                continue
            if module_id in module_id_map:
                module_data = module_id_map[module_id]
                if (
                    grid.get_cell(grid_x, grid_y)["active"]
                    and grid.get_cell(grid_x, grid_y)["module"] is None
                ):
                    place_module(
                        grid,  # Apply changes directly to the input grid
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

    adjacency_score = calculate_pattern_adjacency_score(grid, pattern, start_x, start_y)
    return 1, adjacency_score  # Return 1 to indicate a good pattern


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
            if (
                rotated_pattern_180 != original_pattern
                and tuple(rotated_pattern_180.items()) not in rotated_patterns
            ):
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
    mirrored_horizontal = mirror_pattern_horizontally(original_pattern)
    if mirrored_horizontal != original_pattern:
        if tuple(mirrored_horizontal.items()) not in mirrored_patterns:
            patterns_to_try.append(mirrored_horizontal)
            mirrored_patterns.add(tuple(mirrored_horizontal.items()))
    mirrored_vertical = mirror_pattern_vertically(original_pattern)
    if mirrored_vertical != original_pattern:
        if tuple(mirrored_vertical.items()) not in mirrored_patterns:
            patterns_to_try.append(mirrored_vertical)
            mirrored_patterns.add(tuple(mirrored_vertical.items()))

    # Add mirrored and rotated patterns
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


def calculate_pattern_adjacency_score(grid, pattern, start_x, start_y):
    """
    Calculates the adjacency score for a pattern placed on the grid.

    Args:
        grid (Grid): The grid.
        pattern (dict): The pattern.
        start_x (int): The starting x-coordinate of the pattern.
        start_y (int): The starting y-coordinate of the pattern.

    Returns:
        int: The adjacency score.
    """
    total_adjacency_score = 0
    pattern_positions = set()
    #grid_edges = set() # Remove this line

    # Find all positions occupied by the pattern
    for pattern_x, pattern_y in pattern.keys():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y
        if 0 <= grid_x < grid.width and 0 <= grid_y < grid.height:
            pattern_positions.add((grid_x, grid_y))
            #if grid_x == 0 or grid_x == grid.width - 1 or grid_y == 0 or grid_y == grid.height - 1:
            #    grid_edges.add((grid_x, grid_y)) # Remove this line
            
            # Check each edge individually
            if grid_x == 0:
                total_adjacency_score += 1  # Left edge
            if grid_x == grid.width - 1:
                total_adjacency_score += 1  # Right edge
            if grid_y == 0:
                total_adjacency_score += 1  # Top edge
            if grid_y == grid.height - 1:
                total_adjacency_score += 1  # Bottom edge

    #total_adjacency_score += len(grid_edges) # Remove this line

    # Iterate through the pattern's positions and check for adjacent occupied slots
    for grid_x, grid_y in pattern_positions:
        adjacent_positions = [
            (grid_x - 1, grid_y),
            (grid_x + 1, grid_y),
            (grid_x, grid_y - 1),
            (grid_x, grid_y + 1),
        ]
        for adj_x, adj_y in adjacent_positions:
            if (
                0 <= adj_x < grid.width
                and 0 <= adj_y < grid.height
                and grid.get_cell(adj_x, adj_y)["module"] is not None
                and (
                    adj_x,
                    adj_y,
                )
                not in pattern_positions
            ):
                total_adjacency_score += 1

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
        raise ValueError(
            f"No empty, active slots available on the grid for ship: '{ship}' -- tech: '{tech}'."
        )

    best_grid = Grid.from_dict(grid.to_dict())
    best_bonus = -float("inf")

    solved_grid = Grid.from_dict(grid.to_dict())
    solved_bonus = -float("inf")

    refined_grid = Grid.from_dict(grid.to_dict())
    refined_bonus = -float("inf")

    best_pattern_grid = Grid.from_dict(grid.to_dict())
    highest_pattern_bonus = -float("inf")
    best_pattern_adjacency_score = 0

    # Filter the solves dictionary based on player-owned rewards
    filtered_solves = filter_solves(solves, ship, modules, tech, player_owned_rewards)

    if ship in filtered_solves and tech in filtered_solves[ship]:
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

        pattern_applied = False  # Flag to check if any pattern was successfully applied
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
                    pattern_result, adjacency_score = apply_pattern_to_grid(
                        temp_grid,
                        pattern,
                        modules,
                        tech,
                        start_x,
                        start_y,
                        ship,
                        player_owned_rewards,
                    )
                    if pattern_result == 1:
                        pattern_applied = True  # A pattern was attempted
                        # Recalculate the score after attempting to apply the pattern
                        current_pattern_bonus = calculate_grid_score(temp_grid, tech)

                        if current_pattern_bonus > highest_pattern_bonus:
                            highest_pattern_bonus = current_pattern_bonus
                            best_pattern_grid = Grid.from_dict(temp_grid.to_dict())
                            best_pattern_adjacency_score = adjacency_score
                        elif (
                            current_pattern_bonus == highest_pattern_bonus
                            and adjacency_score > best_pattern_adjacency_score
                        ):
                            highest_pattern_bonus = current_pattern_bonus
                            best_pattern_grid = Grid.from_dict(temp_grid.to_dict())

                            best_pattern_adjacency_score = adjacency_score

            # Reset temp_grid for the next pattern
            temp_grid = Grid.from_dict(grid_dict)

        if best_pattern_grid:
            # Initialize solved_grid with best_pattern_grid
            solved_grid = best_pattern_grid
            solved_bonus = highest_pattern_bonus
            print(
                f"INFO -- Best pattern score: {solved_bonus} for ship: '{ship}' -- tech: '{tech}' that fits."
            )
        else:
            print(
                f"WARNING -- No best pattern definition found for ship: '{ship}' -- tech: '{tech}' that fits. Falling back to simulated_annealing."
            )
            solved_grid = Grid.from_dict(grid.to_dict())
            clear_all_modules_of_tech(solved_grid, tech)
            temp_solved_grid, temp_solved_bonus = simulated_annealing(
                solved_grid, ship, modules, tech, player_owned_rewards
            )
            if temp_solved_grid is not None:
                solved_grid = temp_solved_grid
                solved_bonus = calculate_grid_score(solved_grid, tech) # Recalculate score after annealing
            else:
                print(
                    f"ERROR -- simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
                )
                raise ValueError(
                    f"simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
                )
            pattern_applied = True

        if not pattern_applied:
            print(
                f"WARNING -- No pattern was applied for ship: '{ship}' -- tech: '{tech}'. Falling back to simulated annealing."
            )
            solved_grid = Grid.from_dict(grid.to_dict())
            clear_all_modules_of_tech(solved_grid, tech)
            temp_solved_grid, temp_solved_bonus = simulated_annealing(
                solved_grid, ship, modules, tech, player_owned_rewards, initial_temperature=4000, cooling_rate=0.98, iterations_per_temp=30, initial_swap_probability=0.40,
    final_swap_probability=0.3
            )
            if temp_solved_grid is not None:
                solved_grid = temp_solved_grid
                solved_bonus = calculate_grid_score(solved_grid, tech) # Recalculate score after annealing
            else:
                print(
                    f"ERROR -- simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
                )
                raise ValueError(
                    f"simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
                )

    else:
        print(
            f"INFO -- No solve found for ship: '{ship}' -- tech: '{tech}'. Placing modules in empty slots."
        )
        solved_grid = place_all_modules_in_empty_slots(
            grid, modules, ship, tech, player_owned_rewards
        )
        solved_bonus = calculate_grid_score(solved_grid, tech)
        solve_score = 0

    # Check if all modules were placed
    all_modules_placed = check_all_modules_placed(
        solved_grid, modules, ship, tech, player_owned_rewards
    )

    # Check for supercharged opportunities
    opportunity = find_supercharged_opportunities(solved_grid, modules, ship, tech)

    if opportunity:
        print(f"INFO -- Found opportunity: {opportunity}")
        if message_queue:
            message_queue.put(sse_message(f"Attempting solve for ship: '{ship}' -- tech: '{tech}'", event='status'))
        # Create a localized grid
        opportunity_x, opportunity_y = opportunity

        # Deep copy solved_grid before clearing and creating localized grid
        temp_solved_grid = deepcopy(solved_grid)
        clear_all_modules_of_tech(temp_solved_grid, tech)
        localized_grid, start_x, start_y = create_localized_grid(
            temp_solved_grid, opportunity_x, opportunity_y, tech
        )

        # Refine the localized grid - Surround with id statement
        tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
        if tech_modules is not None and len(tech_modules) < 6:
            refined_grid, refined_bonus = refine_placement(
                localized_grid, ship, modules, tech, player_owned_rewards
            )
            refined_bonus = calculate_grid_score(refined_grid, tech) # Recalculate score after refine_placement
        else:
            refined_grid, refined_bonus = simulated_annealing(
                localized_grid, ship, modules, tech, player_owned_rewards
            )
            refined_bonus = calculate_grid_score(refined_grid, tech) # Recalculate score after annealing
            print_grid(refined_grid)

        if refined_grid is not None:
            # Apply changes to temp_solved_grid
            apply_localized_grid_changes(
                temp_solved_grid, refined_grid, tech, start_x, start_y
            )
            # Calculate the new score of the entire grid
            new_solved_bonus = calculate_grid_score(temp_solved_grid, tech)
            # Compare bonuses and apply changes if the refined bonus is higher
            if new_solved_bonus > solved_bonus:
                # Copy temp_solved_grid to solved_grid
                solved_grid = temp_solved_grid.copy()
                print(
                    f"INFO -- Better refined grid found for ship: '{ship}' -- tech: '{tech}'"
                )
                solved_bonus = new_solved_bonus
            else:
                print(
                    f"INFO -- Refined grid did not improve the score. Solved Bonus: {solved_bonus} vs Refined Bonus: {refined_bonus}"
                )
        else:
            print("simulated_annealing returned None. No changes made.")

    # Recalculate solved_bonus after applying the pattern or falling back to simulated_annealing
    solved_bonus = calculate_grid_score(solved_grid, tech)

    # Calculate the percentage of the solve score achieved
    if solve_score > 0:
        percentage = (solved_bonus / solve_score) * 100 # Use solved_bonus
    else:
        percentage = 0

    if solved_grid is not None:
        best_grid = solved_grid
        best_bonus = solved_bonus
        print(
            f"SUCCESS -- Percentage of Solve Score Achieved: {percentage:.2f}% (Current Score: {best_bonus:.2f}, Adjacency Score: {best_pattern_adjacency_score:.2f}) for ship: '{ship}' -- tech: '{tech}'"
        )
    else:
        print(
            f"ERROR -- No valid grid could be generated for ship: '{ship}' -- tech: '{tech}'"
        )

    return best_grid, percentage


def optimize_placement_old(grid, ship, modules, tech, player_owned_rewards=None):
    """
    Optimizes the placement of modules in a grid for a specific ship and technology.

    Args:
        grid (Grid): The initial grid layout.
        ship (str): The ship type.
        modules (dict): The module data.
        tech (str): The technology type.
        player_owned_rewards (list, optional): Rewards owned by the player. Defaults to None.

    Returns:
        tuple: A tuple containing the best grid found and the percentage of the solve score achieved.
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
                break  # Found at least one empty, active slot, no need to check further
        if has_empty_active_slots:
            break
    if not has_empty_active_slots:
        raise ValueError(
            f"No empty, active slots available on the grid for ship: '{ship}' -- tech: '{tech}'."
        )

    best_grid = Grid.from_dict(grid.to_dict())
    best_bonus = -float("inf")

    best_pattern_grid = Grid.from_dict(grid.to_dict())
    highest_pattern_bonus = -float("inf")
    best_pattern_adjacency_score = 0

    # Filter the solves dictionary based on player-owned rewards
    filtered_solves = filter_solves(solves, ship, modules, tech, player_owned_rewards)

    if ship in filtered_solves and tech in filtered_solves[ship]:
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

        pattern_applied = False  # Flag to check if any pattern was successfully applied
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
                    # Calculate the bonus before applying the pattern
                    pre_pattern_bonus = calculate_grid_score(temp_grid, tech)

                    # Apply the pattern to the persistent temp_grid
                    pattern_result, adjacency_score = apply_pattern_to_grid(
                        temp_grid,
                        pattern,
                        modules,
                        tech,
                        start_x,
                        start_y,
                        ship,
                        player_owned_rewards,
                    )

                    if pattern_result == 0:
                        current_pattern_bonus = pre_pattern_bonus
                    else:
                        current_pattern_bonus = calculate_grid_score(temp_grid, tech)
                        pattern_applied = True  # A pattern was successfully applied

                    if current_pattern_bonus > highest_pattern_bonus or (
                        current_pattern_bonus == highest_pattern_bonus
                        and adjacency_score > best_pattern_adjacency_score
                    ):
                        highest_pattern_bonus = current_pattern_bonus
                        best_pattern_grid = Grid.from_dict(temp_grid.to_dict())
                        best_pattern_adjacency_score = adjacency_score

            # Reset temp_grid for the next pattern
            temp_grid = Grid.from_dict(grid_dict)

        if best_pattern_grid:
            # Initialize best_grid with best_pattern_grid
            best_grid = best_pattern_grid
            best_bonus = highest_pattern_bonus
            print(
                f"INFO -- Best pattern score: {best_bonus} for ship: '{ship}' -- tech: '{tech}' that fits."
            )
        
        else:
            print(
                f"WARNING -- No best pattern definition found for ship: '{ship}' -- tech: '{tech}' that fits. Falling back to refine_placement."
            )
            best_grid = Grid.from_dict(grid.to_dict())
            clear_all_modules_of_tech(best_grid, tech)
            temp_best_grid, temp_best_bonus = simulated_annealing(
                best_grid, ship, modules, tech, player_owned_rewards
            )
            # temp_best_grid, temp_best_bonus = refine_placement(best_grid, ship, modules, tech, player_owned_rewards)
            if temp_best_grid is not None:
                best_grid = temp_best_grid
                best_bonus = temp_best_bonus
            else:
                print(
                    f"ERROR -- simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
                )
                raise ValueError(
                    f"simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
                )
            pattern_applied = True

        if not pattern_applied:
            print(
                f"WARNING -- No pattern was applied for ship: '{ship}' -- tech: '{tech}'. Falling back to simulated annealing."
            )
            best_grid = Grid.from_dict(grid.to_dict())
            clear_all_modules_of_tech(best_grid, tech)
            temp_best_grid, temp_best_bonus = simulated_annealing(
                best_grid, ship, modules, tech, player_owned_rewards
            )
            if temp_best_grid is not None:
                best_grid = temp_best_grid
                best_bonus = temp_best_bonus
            else:
                print(
                    f"ERROR -- simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
                )
                raise ValueError(
                    f"simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
                )

    else:
        print(
            f"INFO -- No solve found for ship: '{ship}' -- tech: '{tech}'. Placing modules in empty slots."
        )
        best_grid = place_all_modules_in_empty_slots(
            grid, modules, ship, tech, player_owned_rewards
        )
        best_bonus = calculate_grid_score(best_grid, tech)
        solve_score = 0

    solved_bonus = calculate_grid_score(best_grid, tech)

    # Check if all modules were placed
    all_modules_placed = check_all_modules_placed(
        best_grid, modules, ship, tech, player_owned_rewards
    )
    if not all_modules_placed:
        print(
            f"WARNING -- Not all modules were placed in grid for ship: '{ship}' -- tech: '{tech}'. Running simulated_annealing solver."
        )

        clear_all_modules_of_tech(best_grid, tech)
        temp_best_grid, temp_best_bonus = simulated_annealing(
            best_grid,
            ship,
            modules,
            tech,
            player_owned_rewards,
            initial_temperature=2000,
            cooling_rate=0.98,
            iterations_per_temp=20,
        )
        if temp_best_grid is not None:
            best_grid = temp_best_grid
            best_bonus = temp_best_bonus
            solved_bonus = best_bonus
        else:
            print(
                f"ERROR -- simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
            )
            raise ValueError(
                f"simulated_annealing solver failed to find a valid placement for ship: '{ship}' -- tech: '{tech}'."
            )
    else:
        # Check for supercharged opportunities
        opportunity = find_supercharged_opportunities(best_grid, modules, ship, tech)

        if opportunity:
            print(f"INFO -- Found opportunity: {opportunity}")
            # Create a localized grid
            opportunity_x, opportunity_y = opportunity

            # Deep copy best_grid before clearing and creating localized grid
            temp_best_grid = deepcopy(best_grid)
            clear_all_modules_of_tech(temp_best_grid, tech)
            localized_grid, start_x, start_y = create_localized_grid(
                temp_best_grid, opportunity_x, opportunity_y, tech
            )

            # Refine the localized grid - Surround with id statement
            tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
            if tech_modules is not None and len(tech_modules) < 6:
                optimized_localized_grid, refined_bonus = refine_placement(
                    localized_grid, ship, modules, tech, player_owned_rewards
                )
            else:
                optimized_localized_grid, refined_bonus = simulated_annealing(
                    localized_grid, ship, modules, tech, player_owned_rewards
                )

            if optimized_localized_grid is not None:
                # Compare bonuses and apply changes if the refined bonus is higher
                if refined_bonus > solved_bonus:
                    # Apply changes to temp_best_grid
                    apply_localized_grid_changes(
                        temp_best_grid, optimized_localized_grid, tech, start_x, start_y
                    )
                    # Copy temp_best_grid to best_grid
                    best_grid = temp_best_grid.copy()
                    print(
                        f"INFO -- Better refined grid found for ship: '{ship}' -- tech: '{tech}'"
                    )
                    solved_bonus = refined_bonus
                    best_bonus = refined_bonus
                else:
                    print(
                        f"INFO -- Refined grid did not improve the score. Solved Bonus: {solved_bonus} vs Refined Bonus: {refined_bonus}"
                    )
            else:
                print("simulated_annealing returned None. No changes made.")

    # Calculate the percentage of the solve score achieved
    if solve_score > 0:
        percentage = (best_bonus / solve_score) * 100
    else:
        percentage = 0

    if best_grid is not None:
        print(
            f"SUCCESS -- Percentage of Solve Score Achieved: {percentage:.2f}% (Current Score: {best_bonus:.2f}, Adjacency Score: {best_pattern_adjacency_score:.2f}) for ship: '{ship}' -- tech: '{tech}'"
        )
    else:
        print(
            f"ERROR -- No valid grid could be generated for ship: '{ship}' -- tech: '{tech}'"
        )

    return best_grid, percentage


def place_all_modules_in_empty_slots(
    grid, modules, ship, tech, player_owned_rewards=None
):
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
        print(
            f"WARNING -- Not enough space to place all modules for ship: '{ship}' -- tech: '{tech}'"
        )

    return grid


def find_supercharged_opportunities(grid, modules, ship, tech):
    """
    Checks if there are any opportunities to utilize unused supercharged slots or swap modules
    with supercharged slots, focusing on the outer boundary of the solve.
    Also checks for supercharged slots within the bounds of the current tech.

    Args:
        grid (Grid): The current grid layout.
        modules (dict): The module data.
        ship (str): The ship type.
        tech (str): The technology type.

    Returns:
        tuple or None: A tuple (opportunity_x, opportunity_y) if an opportunity is found,
                       None otherwise.
    """
    occupied_positions = [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.get_cell(x, y)["module"] is not None
    ]
    supercharged_positions = [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.get_cell(x, y)["supercharged"]
    ]
    occupied_supercharged_count = sum(
        1 for x, y in occupied_positions if (x, y) in supercharged_positions
    )

    # --- Helper Functions ---
    def is_valid_position(x, y):
        """Checks if a position is within the grid bounds."""
        return 0 <= x < grid.width and 0 <= y < grid.height

    def get_adjacent_positions(grid, x, y):
        """Gets the valid adjacent positions to a given position."""
        adjacent = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [(nx, ny) for nx, ny in adjacent if is_valid_position(nx, ny)]

    def is_on_boundary(grid, x, y):
        """Checks if a cell is on the outer boundary of the solve."""
        if grid.get_cell(x, y)["module"] is not None:
            return False
        for nx, ny in get_adjacent_positions(grid, x, y):
            if grid.get_cell(nx, ny)["module"] is not None:
                return True
        return False

    def count_supercharged_in_localized(localized_grid):
        """Counts the number of supercharged slots in a localized grid."""
        count = 0
        for y in range(localized_grid.height):
            for x in range(localized_grid.width):
                if localized_grid.get_cell(x, y)["supercharged"]:
                    count += 1
        return count

    # --- Main Logic ---
    boundary_supercharged_slots = [
        (x, y) for x, y in supercharged_positions if is_on_boundary(grid, x, y)
    ]
    unused_boundary_supercharged_slots = [
        (x, y)
        for x, y in boundary_supercharged_slots
        if grid.get_cell(x, y)["module"] is None
    ]

    if unused_boundary_supercharged_slots:
        # There's an unused supercharged slot on the boundary, so there's an opportunity
        opportunity_x, opportunity_y = unused_boundary_supercharged_slots[0]
        localized_grid, _, _ = create_localized_grid(
            grid, opportunity_x, opportunity_y, tech
        )  # Unpack the tuple
        if (
            count_supercharged_in_localized(localized_grid)
            > occupied_supercharged_count
        ):
            return opportunity_x, opportunity_y

    modules_in_boundary_supercharged = [
        (x, y) for x, y in occupied_positions if (x, y) in boundary_supercharged_slots
    ]
    if not modules_in_boundary_supercharged:
        pass
    else:
        for sx, sy in boundary_supercharged_slots:
            for nx, ny in get_adjacent_positions(grid, sx, sy):
                if (nx, ny) in occupied_positions and (
                    nx,
                    ny,
                ) not in modules_in_boundary_supercharged:
                    # There's a module adjacent to a supercharged slot on the boundary, so there's a swap opportunity
                    localized_grid, _, _ = create_localized_grid(
                        grid, sx, sy
                    )  # Unpack the tuple
                    if (
                        count_supercharged_in_localized(localized_grid)
                        > occupied_supercharged_count
                    ):
                        return sx, sy

    # Check for supercharged slots within the bounds of the current tech
    tech_occupied_supercharged_slots = [
        (x, y) for x, y in supercharged_positions if grid.get_cell(x, y)["tech"] == tech
    ]
    if tech_occupied_supercharged_slots:
        return tech_occupied_supercharged_slots[0]

    # No opportunities found
    return None


def find_supercharged_opportunities_old(grid, modules, ship, tech):
    """
    Checks if there are any opportunities to utilize unused supercharged slots or swap modules
    with supercharged slots, focusing on the outer boundary of the solve.

    Args:
        grid (Grid): The current grid layout.
        modules (dict): The module data.
        ship (str): The ship type.
        tech (str): The technology type.

    Returns:
        tuple or None: A tuple (opportunity_x, opportunity_y) if an opportunity is found,
                       None otherwise.
    """
    occupied_positions = [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.get_cell(x, y)["module"] is not None
    ]
    supercharged_positions = [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.get_cell(x, y)["supercharged"]
    ]
    occupied_supercharged_count = sum(
        1 for x, y in occupied_positions if (x, y) in supercharged_positions
    )

    # --- Helper Functions ---
    def is_valid_position(x, y):
        """Checks if a position is within the grid bounds."""
        return 0 <= x < grid.width and 0 <= y < grid.height

    def get_adjacent_positions(grid, x, y):
        """Gets the valid adjacent positions to a given position."""
        adjacent = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [(nx, ny) for nx, ny in adjacent if is_valid_position(nx, ny)]

    def is_on_boundary(grid, x, y):
        """Checks if a cell is on the outer boundary of the solve."""
        if grid.get_cell(x, y)["module"] is not None:
            return False
        for nx, ny in get_adjacent_positions(grid, x, y):
            if grid.get_cell(nx, ny)["module"] is not None:
                return True
        return False

    def count_supercharged_in_localized(localized_grid):
        """Counts the number of supercharged slots in a localized grid."""
        count = 0
        for y in range(localized_grid.height):
            for x in range(localized_grid.width):
                if localized_grid.get_cell(x, y)["supercharged"]:
                    count += 1
        return count

    # --- Main Logic ---
    boundary_supercharged_slots = [
        (x, y) for x, y in supercharged_positions if is_on_boundary(grid, x, y)
    ]
    unused_boundary_supercharged_slots = [
        (x, y)
        for x, y in boundary_supercharged_slots
        if grid.get_cell(x, y)["module"] is None
    ]

    if unused_boundary_supercharged_slots:
        # There's an unused supercharged slot on the boundary, so there's an opportunity
        opportunity_x, opportunity_y = unused_boundary_supercharged_slots[0]
        localized_grid, _, _ = create_localized_grid(
            grid, opportunity_x, opportunity_y
        )  # Unpack the tuple
        if (
            count_supercharged_in_localized(localized_grid)
            > occupied_supercharged_count
        ):
            return opportunity_x, opportunity_y

    modules_in_boundary_supercharged = [
        (x, y) for x, y in occupied_positions if (x, y) in boundary_supercharged_slots
    ]
    if not modules_in_boundary_supercharged:
        return None
    for sx, sy in boundary_supercharged_slots:
        for nx, ny in get_adjacent_positions(grid, sx, sy):
            if (nx, ny) in occupied_positions and (
                nx,
                ny,
            ) not in modules_in_boundary_supercharged:
                # There's a module adjacent to a supercharged slot on the boundary, so there's a swap opportunity
                localized_grid, _, _ = create_localized_grid(
                    grid, sx, sy
                )  # Unpack the tuple
                if (
                    count_supercharged_in_localized(localized_grid)
                    > occupied_supercharged_count
                ):
                    return sx, sy

    # No opportunities found
    return None


def create_localized_grid(grid, opportunity_x, opportunity_y, tech):
    """
    Creates a localized grid around a given opportunity, ensuring it stays within
    the bounds of the main grid and preserves modules of other tech types.

    Args:
        grid (Grid): The main grid.
        opportunity_x (int): The x-coordinate of the opportunity.
        opportunity_y (int): The y-coordinate of the opportunity.
        tech (str): The technology type being optimized.

    Returns:
        tuple: A tuple containing:
            - localized_grid (Grid): The localized grid.
            - start_x (int): The starting x-coordinate of the localized grid in the main grid.
            - start_y (int): The starting y-coordinate of the localized grid in the main grid.
    """
    localized_width = 3
    localized_height = 3

    # Calculate the bounds of the localized grid, clamping to the main grid's edges
    start_x = max(0, opportunity_x - localized_width // 2)
    start_y_unclamped = opportunity_y - localized_height // 2
    start_y = max(0, start_y_unclamped)

    # Calculate how much start_y was clamped
    clamped_diff_y = start_y - start_y_unclamped

    end_x = min(
        grid.width, opportunity_x + localized_width // 2 + (localized_width % 2)
    )
    # Adjust end_y based on how much start_y was clamped
    end_y = min(
        grid.height,
        opportunity_y + localized_height // 2 + (localized_height % 2) + clamped_diff_y,
    )

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
            localized_grid.cells[localized_y][localized_x]["supercharged"] = cell[
                "supercharged"
            ]

            # Copy module data if a module exists
            if cell["module"] is not None:
                localized_grid.cells[localized_y][localized_x]["module"] = cell[
                    "module"
                ]
                localized_grid.cells[localized_y][localized_x]["label"] = cell["label"]
                localized_grid.cells[localized_y][localized_x]["tech"] = cell["tech"]
                localized_grid.cells[localized_y][localized_x]["type"] = cell["type"]
                localized_grid.cells[localized_y][localized_x]["bonus"] = cell["bonus"]
                localized_grid.cells[localized_y][localized_x]["adjacency"] = cell[
                    "adjacency"
                ]
                localized_grid.cells[localized_y][localized_x]["sc_eligible"] = cell[
                    "sc_eligible"
                ]
                localized_grid.cells[localized_y][localized_x]["image"] = cell["image"]
                # Only copy module_position if it exists
                if "module_position" in cell:
                    localized_grid.cells[localized_y][localized_x][
                        "module_position"
                    ] = cell["module_position"]

    return localized_grid, start_x, start_y


def apply_localized_grid_changes(grid, localized_grid, tech, start_x, start_y):
    """Applies changes from the localized grid back to the main grid."""
    localized_width = localized_grid.width
    localized_height = localized_grid.height

    # Clear all existing modules of the specified tech in the main grid
    # clear_all_modules_of_tech(grid, tech) # This is no longer needed

    # Copy module placements from the localized grid back to the main grid
    for y in range(localized_height):
        for x in range(localized_width):
            main_x = start_x + x
            main_y = start_y + y
            if 0 <= main_x < grid.width and 0 <= main_y < grid.height:
                # Only copy if the cell is empty or of the same tech
                if (
                    grid.get_cell(main_x, main_y)["tech"] == tech
                    or grid.get_cell(main_x, main_y)["module"] is None
                ):                    
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
