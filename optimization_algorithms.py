# optimization_algorithms.py
from grid_utils import Grid
from modules_data import get_tech_modules
from grid_display import print_grid_compact, print_grid
from bonus_calculations import calculate_grid_score
from module_placement import place_module
from itertools import permutations
from itertools import combinations
from modules import (
    solves,
)  
from solve_map_utils import filter_solves # Import the new function
import json
import queue

def send_sse_message(message_queue, client_id, message):
    """Sends an SSE message if the message queue is set."""
    if message_queue and client_id:
        message_queue.put(json.dumps({"clientId": client_id, "status": "info", "message": message}))


def refine_placement(grid, ship, modules, tech, player_owned_rewards=None):
    optimal_grid = None
    highest_bonus = 0.0
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)

    if tech_modules is None:
        print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
        return None, 0.0

    core_modules = [module for module in tech_modules if module["type"] == "core"]
    bonus_modules = [module for module in tech_modules if module["type"] == "bonus"]

    if not core_modules:
        raise ValueError("No core modules specified")

    core_module = core_modules[0]  # Assuming one core module

    # Precompute available positions for fast access
    available_positions = [
        (x, y) for y in range(grid.height) for x in range(grid.width)
        if grid.get_cell(x, y)["module"] is None and grid.get_cell(x, y)["active"]
    ]

    for core_x, core_y in available_positions:
        temp_grid = grid.copy()  # Use a dedicated copy method instead of to_dict/from_dict

        place_module(
            temp_grid, core_x, core_y,
            core_module["id"], core_module["label"], tech,
            core_module["type"], core_module["bonus"],
            core_module["adjacency"], core_module["sc_eligible"],
            core_module["image"],
        )

        # Get new available positions after placing core
        new_available_positions = [
            (x, y) for x, y in available_positions if temp_grid.get_cell(x, y)["module"] is None
        ]

        # Use combinations for all possible bonus placements
        for bonus_placement in combinations(new_available_positions, min(len(bonus_modules), len(new_available_positions))):
            temp_grid_inner = temp_grid.copy()  # Avoid full grid recreation

            for index, (x, y) in enumerate(bonus_placement):
                bonus_module = bonus_modules[index]
                place_module(
                    temp_grid_inner, x, y,
                    bonus_module["id"], bonus_module["label"], tech,
                    bonus_module["type"], bonus_module["bonus"],
                    bonus_module["adjacency"], bonus_module["sc_eligible"],
                    bonus_module["image"],
                )

            core_bonus = calculate_grid_score(temp_grid_inner, tech)

            if core_bonus > highest_bonus:
                highest_bonus = core_bonus
                optimal_grid = temp_grid_inner.copy()  # Store the best-found grid

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
    """Applies a pattern to an existing grid at a given starting position,
    preserving the original grid's state (except for modules of the same tech)
    and only filling empty, active slots with the pattern's modules.
    """
    # Check for overlap before applying the pattern
    for (pattern_x, pattern_y) in pattern.keys():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y
        if 0 <= grid_x < grid.width and 0 <= grid_y < grid.height:
            if grid.get_cell(grid_x, grid_y)["module"] is not None and grid.get_cell(grid_x, grid_y)["tech"] != tech:
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
    adjacent_positions = [
        (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)
    ]
    for nx, ny in adjacent_positions:
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            if grid.get_cell(nx, ny)["module"] is not None:
                count += 1
    return count

def calculate_pattern_adjacency_score(grid, pattern, start_x, start_y):
    """
    Calculates the secondary adjacency score for a pattern placed on the grid.

    Args:
        grid (Grid): The grid.
        pattern (dict): The pattern.
        start_x (int): The starting x-coordinate of the pattern.
        start_y (int): The starting y-coordinate of the pattern.

    Returns:
        int: The secondary adjacency score.
    """
    total_adjacency_score = 0
    pattern_positions = set()

    # Find all positions occupied by the pattern
    for (pattern_x, pattern_y) in pattern.keys():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y
        if 0 <= grid_x < grid.width and 0 <= grid_y < grid.height:
            pattern_positions.add((grid_x, grid_y))

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
                and (adj_x, adj_y) not in pattern_positions
                and grid.get_cell(adj_x, adj_y)["module"] is not None
            ):
                total_adjacency_score += 1

    return total_adjacency_score

def optimize_placement(
    grid,
    ship,
    modules,
    tech,
    client_id=None,  # Optional client_id
    message_queue=None,  # Optional message_queue
    player_owned_rewards=None
):
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
                        temp_grid, pattern, modules, tech, start_x, start_y, ship, player_owned_rewards
                    )
                    if pattern_result == 0:
                        current_pattern_bonus = 0
                    else:
                        current_pattern_bonus = calculate_grid_score(temp_grid, tech)

                    if current_pattern_bonus > highest_pattern_bonus:
                        highest_pattern_bonus = current_pattern_bonus
                        best_pattern_grid = Grid.from_dict(temp_grid.to_dict())
                        best_pattern_adjacency_score = adjacency_score

            # Reset temp_grid for the next pattern
            temp_grid = Grid.from_dict(grid_dict)

        if best_pattern_grid:
            # Initialize best_grid with best_pattern_grid
            best_grid = best_pattern_grid
            best_bonus = highest_pattern_bonus
        else:
            print(
                f"No best pattern definition found for ship: {ship}, tech: {tech}. Starting with the initial grid."
            )
            send_sse_message(message_queue, client_id, "No best pattern definition found. Starting with the initial grid.")
    else:
        print(f"No solve found for {ship} {tech}, placing modules in empty slots.")
        send_sse_message(message_queue, client_id, "No solve found. Placing modules in empty slots.")
        best_grid = place_all_modules_in_empty_slots(grid, modules, ship, tech, player_owned_rewards)
        best_bonus = calculate_grid_score(best_grid, tech)
        solve_score = 0

    solved_bonus = calculate_grid_score(best_grid, tech)

    # Check if all modules were placed
    all_modules_placed = check_all_modules_placed(best_grid, modules, ship, tech)
    if not all_modules_placed:
        print("WARNING: Not all modules for this tech were placed in the grid. Running brute-force solver.")
        send_sse_message(message_queue, client_id, "Not all modules for this tech were placed in the grid. Running brute-force solver.")
        
        clear_all_modules_of_tech(best_grid, tech)
        temp_best_grid, temp_best_bonus = refine_placement(best_grid, ship, modules, tech, player_owned_rewards)
        if temp_best_grid is not None:
            best_grid = temp_best_grid
            best_bonus = temp_best_bonus
            solved_bonus = best_bonus
        else:
            print("Brute-force solver failed to find a valid placement.")
    else:
        # Check for supercharged opportunities
        opportunity = find_supercharged_opportunities(best_grid, modules, ship, tech)

        if opportunity:
            print(f"Found opportunity: {opportunity}")
            send_sse_message(message_queue, client_id, f"Found possble supercharger opportunity. Hold on a moment.")
            # Create a localized grid
            opportunity_x, opportunity_y = opportunity
            localized_grid, start_x, start_y = create_localized_grid(
                best_grid, opportunity_x, opportunity_y
            )

            # Refine the localized grid
            optimized_localized_grid, refined_bonus = refine_placement(
                localized_grid, ship, modules, tech, player_owned_rewards
            )

            if optimized_localized_grid is not None:
                # Compare bonuses and apply changes if the refined bonus is higher
                if refined_bonus > solved_bonus:
                    apply_localized_grid_changes(
                        best_grid, optimized_localized_grid, tech, start_x, start_y
                    )
                    print("BETTER, REFINED GRID FOUND!")
                    send_sse_message(message_queue, client_id, "Better, refined grid found!")
                    solved_bonus = refined_bonus
                    best_bonus = refined_bonus
                else:
                    print("Refined grid did not improve the score.")
            else:
                print("refine_placement returned None. No changes made.")

    # Calculate the percentage of the solve score achieved
    if solve_score > 0:
        percentage = (best_bonus / solve_score) * 100
    else:
        percentage = 0


    if best_grid is not None:
        print(f"Percentage of Solve Score Achieved: {percentage:.2f}% (Current Score: {best_bonus:.2f}, Adjacency Score: {best_pattern_adjacency_score:.2f})")
        # print_grid_compact(best_grid)
    else:
        print("No valid grid could be generated.")
        send_sse_message(message_queue, client_id, "No valid grid could be generated.")

    return best_grid, percentage

def place_all_modules_in_empty_slots(grid, modules, ship, tech, player_owned_rewards=None):
    """Places all modules of a given tech in any remaining empty slots, going column by column."""
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None:
        print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
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
        print(f"Warning: Not enough space to place all modules for ship '{ship}' and tech '{tech}'.")

    return grid

def find_supercharged_opportunities(grid, modules, ship, tech):
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


def create_localized_grid(grid, opportunity_x, opportunity_y):
    """
    Creates a localized grid around a given opportunity, ensuring it stays within
    the bounds of the main grid.

    Args:
        grid (Grid): The main grid.
        opportunity_x (int): The x-coordinate of the opportunity.
        opportunity_y (int): The y-coordinate of the opportunity.

    Returns:
        tuple: A tuple containing:
            - localized_grid (Grid): The localized grid.
            - start_x (int): The starting x-coordinate of the localized grid in the main grid.
            - start_y (int): The starting y-coordinate of the localized grid in the main grid.
    """
    localized_width = 4
    localized_height = 4

    # Calculate the bounds of the localized grid, clamping to the main grid's edges
    start_x = max(0, opportunity_x - localized_width // 2)
    start_y = max(0, opportunity_y - localized_height // 2)
    end_x = min(
        grid.width, opportunity_x + localized_width // 2 + (localized_width % 2)
    )
    end_y = min(
        grid.height, opportunity_y + localized_height // 2 + (localized_height % 2)
    )

    # Adjust the localized grid size based on the clamped bounds
    actual_localized_width = end_x - start_x
    actual_localized_height = end_y - start_y

    # Create the localized grid with the adjusted size
    localized_grid = Grid(actual_localized_width, actual_localized_height)

    # Copy only the grid structure (active/inactive, supercharged)
    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            localized_x = x - start_x
            localized_y = y - start_y
            cell = grid.get_cell(x, y)
            localized_grid.cells[localized_y][localized_x]["active"] = cell["active"]
            localized_grid.cells[localized_y][localized_x]["supercharged"] = cell[
                "supercharged"
            ]

    return localized_grid, start_x, start_y


def apply_localized_grid_changes(grid, localized_grid, tech, start_x, start_y):
    """Applies changes from the localized grid back to the main grid."""
    localized_width = localized_grid.width
    localized_height = localized_grid.height

    # Clear all existing modules of the specified tech in the main grid
    clear_all_modules_of_tech(grid, tech)

    # Copy module placements from the localized grid back to the main grid
    for y in range(localized_height):
        for x in range(localized_width):
            main_x = start_x + x
            main_y = start_y + y
            if 0 <= main_x < grid.width and 0 <= main_y < grid.height:
                grid.cells[main_y][main_x]["module"] = localized_grid.cells[y][x][
                    "module"
                ]
                grid.cells[main_y][main_x]["label"] = localized_grid.cells[y][x][
                    "label"
                ]
                grid.cells[main_y][main_x]["tech"] = localized_grid.cells[y][x]["tech"]
                grid.cells[main_y][main_x]["type"] = localized_grid.cells[y][x]["type"]
                grid.cells[main_y][main_x]["bonus"] = localized_grid.cells[y][x][
                    "bonus"
                ]
                grid.cells[main_y][main_x]["adjacency"] = localized_grid.cells[y][x][
                    "adjacency"
                ]
                grid.cells[main_y][main_x]["sc_eligible"] = localized_grid.cells[y][x][
                    "sc_eligible"
                ]
                grid.cells[main_y][main_x]["image"] = localized_grid.cells[y][x][
                    "image"
                ]

def check_all_modules_placed(grid, modules, ship, tech):
    """
    Checks if all modules for a given tech have been placed in the grid.

    Args:
        grid (Grid): The grid layout.
        modules (dict): The module data.
        ship (str): The ship type.
        tech (str): The technology type.

    Returns:
        bool: True if all modules are placed, False otherwise.
    """

    tech_modules = get_tech_modules(modules, ship, tech)
    placed_module_ids = set()

    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech and cell["module"]:
                placed_module_ids.add(cell["module"])

    all_module_ids = {module["id"] for module in tech_modules}
    return placed_module_ids == all_module_ids


def clear_all_modules_of_tech(grid, tech):
    """Clears all modules of the specified tech type from the entire grid."""
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.get_cell(x, y)["tech"] == tech:
                grid.cells[y][x]["module"] = None
                grid.cells[y][x]["label"] = ""
                grid.cells[y][x]["tech"] = None
                grid.cells[y][x]["type"] = ""
                grid.cells[y][x]["bonus"] = 0
                grid.cells[y][x]["adjacency"] = False
                grid.cells[y][x]["sc_eligible"] = False
                grid.cells[y][x]["image"] = None
