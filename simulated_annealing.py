# simulated_annealing.py
import random
import math
import time
from grid_display import print_grid_compact, print_grid
from modules_data import get_tech_modules
from bonus_calculations import calculate_grid_score
from module_placement import place_module, clear_all_modules_of_tech


def simulated_annealing(
    grid,
    ship,
    modules,
    tech,
    player_owned_rewards=None,
    initial_temperature=4000,
    cooling_rate=0.995,
    stopping_temperature=1.5,
    iterations_per_temp=35,
    initial_swap_probability=0.55,
    final_swap_probability=0.4,
    # <<< Add a flag to indicate polishing mode >>>
    start_from_current_grid: bool = False,
    max_processing_time: float = 360.0 # <<< Make max time an argument
):
    """
    Performs simulated annealing to optimize module placement on a grid.

    Args:
        # ... (other args remain the same)
        start_from_current_grid (bool): If True, skips the initial clearing and
                                        placement, starting optimization from the
                                        provided grid state. Ideal for polishing.
        max_processing_time (float): Maximum time in seconds allowed for annealing.
    """
    start_time = time.time()  # Start timing
    # max_processing_time = 360  # Maximum processing time in seconds # <<< Use argument instead

    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None:
        print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
        return grid, 0.0

    # --- Determine modules to consider (based on available slots if not polishing) ---
    modules_to_consider = []
    if not start_from_current_grid:
        # Original logic to figure out which modules fit in empty slots
        supercharged_slots = [
            (x, y) for y in range(grid.height) for x in range(grid.width)
            if grid.get_cell(x, y)["module"] is None and grid.get_cell(x, y)["supercharged"] and grid.get_cell(x, y)["active"]
        ]
        active_slots = [
            (x, y) for y in range(grid.height) for x in range(grid.width)
            if grid.get_cell(x, y)["module"] is None and grid.get_cell(x, y)["active"] and not grid.get_cell(x, y)["supercharged"]
        ]
        num_available_positions = len(supercharged_slots) + len(active_slots)

        core_module = next((m for m in tech_modules if m["type"] == "core"), None)
        bonus_modules = [m for m in tech_modules if m["type"] != "core"]
        bonus_modules.sort(key=lambda m: m["bonus"], reverse=True)

        if core_module:
            modules_to_consider.append(core_module)
            # Adjust available positions count if core is definitely placed
            if num_available_positions > 0:
                 num_available_positions -= 1

        modules_to_consider.extend(bonus_modules[:num_available_positions])
    else:
        # If polishing, consider all modules currently placed on the grid for swaps/moves
        modules_to_consider = []
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                if cell["tech"] == tech and cell["module"] is not None:
                    # Find the full module definition
                    module_def = next((m for m in tech_modules if m["id"] == cell["module"]), None)
                    if module_def:
                        modules_to_consider.append(module_def)
        # If modules_to_consider is empty after checking grid, fall back to tech_modules
        if not modules_to_consider:
             modules_to_consider = tech_modules


    # --- Initialize the current state ---
    current_grid = grid.copy() # Always start with a copy

    # <<< Conditional Initial Placement >>>
    if not start_from_current_grid:
        # Clear any existing modules of the same tech from the grid.
        clear_all_modules_of_tech(current_grid, tech)
        # Place modules with priority (original behavior)
        place_modules_with_supercharged_priority(current_grid, modules_to_consider, tech)
    # else: If start_from_current_grid is True, we use the grid as passed in.

    current_score = calculate_grid_score(current_grid, tech)
    best_grid = current_grid.copy()
    best_score = current_score

    temperature = initial_temperature
    swap_probability = initial_swap_probability

    # --- Annealing Loop ---
    while temperature > stopping_temperature:
        # Check if the maximum processing time has been exceeded
        if time.time() - start_time > max_processing_time:
            print(f"INFO -- SA: Max processing time ({max_processing_time}s) exceeded. Returning best found.")
            break # Exit the loop

        # Calculate the adaptive swap probability
        swap_probability = get_swap_probability(
            temperature, initial_temperature, stopping_temperature,
            initial_swap_probability, final_swap_probability
        )

        for _ in range(iterations_per_temp):
            neighbor_grid = current_grid.copy()
            # Ensure modules_to_consider is passed correctly
            if random.random() < swap_probability:
                swap_modules(neighbor_grid, tech, modules_to_consider)
            else:
                move_module(neighbor_grid, tech, modules_to_consider)
            neighbor_score = calculate_grid_score(neighbor_grid, tech)

            delta_e = neighbor_score - current_score
            if delta_e > 0 or random.random() < math.exp(delta_e / temperature):
                current_grid = neighbor_grid # Keep the copy
                current_score = neighbor_score
                if current_score > best_score:
                    best_grid = current_grid.copy()
                    best_score = current_score
            # else: discard neighbor_grid (it was just a copy)

        temperature *= cooling_rate
    # --- End Annealing Loop ---

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"INFO -- SA finished ({'Polish' if start_from_current_grid else 'Full'}). Best score: {best_score:.4f}. Time: {elapsed_time:.4f}s")

    # Check for failure (important for polishing case too)
    if best_grid is None or (best_score == 0 and any(m["bonus"] > 0 for m in modules_to_consider)):
         # Check if score is 0 despite having modules with potential bonus
         print(f"WARNING -- SA solver might have failed to find a valid placement for {ship}/{tech}. Score is {best_score}.")
         # Return the initial grid state if polishing failed badly
         if start_from_current_grid:
             initial_score = calculate_grid_score(grid, tech)
             print(f"INFO -- Returning original grid state with score {initial_score:.4f} due to potential SA failure.")
             return grid.copy(), initial_score
         else:
             # For a full run, maybe return the cleared grid or raise error
             cleared_grid = grid.copy()
             clear_all_modules_of_tech(cleared_grid, tech)
             return cleared_grid, 0.0

    return best_grid, best_score

def get_swap_probability(
    temperature,
    initial_temperature,
    stopping_temperature,
    initial_swap_probability,
    final_swap_probability,
):
    """
    Calculates the swap probability based on the current temperature.
    """
    # Example: Linearly decrease swap_probability from initial_swap_probability to final_swap_probability
    if temperature >= initial_temperature:
        return initial_swap_probability
    if temperature <= stopping_temperature:
        return final_swap_probability

    return initial_swap_probability - (initial_swap_probability - final_swap_probability) * (
        (initial_temperature - temperature) / (initial_temperature - stopping_temperature)
    )


def place_modules_with_supercharged_priority(grid, tech_modules, tech):
    """
    Places modules with priority on supercharged slots, ensuring the core module is always placed.

    Args:
        grid (Grid): The grid to place modules on.
        tech_modules (list): The list of modules to place.
        tech (str): The technology type of the modules.
    """
    # Identify supercharged and active slots - DO NOT SHUFFLE THESE LISTS
    supercharged_slots = [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.get_cell(x, y)["module"] is None
        and grid.get_cell(x, y)["supercharged"]
        and grid.get_cell(x, y)["active"]
    ]
    active_slots = [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.get_cell(x, y)["module"] is None
        and grid.get_cell(x, y)["active"]
        and not grid.get_cell(x, y)["supercharged"]
    ]

    # Sort modules by bonus in descending order, ensuring core module is first
    core_module = next((m for m in tech_modules if m["type"] == "core"), None)
    bonus_modules = [m for m in tech_modules if m["type"] != "core"]
    bonus_modules.sort(key=lambda m: m["bonus"], reverse=True)
    sorted_modules = []
    if core_module:
        sorted_modules.append(core_module)
    sorted_modules.extend(bonus_modules)

    # Limit the number of modules to place if there are not enough slots
    num_available_positions = len(supercharged_slots) + len(active_slots)
    modules_to_place = sorted_modules[: min(len(sorted_modules), num_available_positions)]

    # Separate sc_eligible and non_sc_eligible modules
    sc_eligible_modules = [m for m in modules_to_place if m["sc_eligible"]]
    non_sc_eligible_modules = [m for m in modules_to_place if not m["sc_eligible"]]

    # Handle core module placement
    if core_module:
        if core_module["sc_eligible"]:
            sc_eligible_modules.insert(0, core_module)  # Ensure core is first if sc_eligible
        else:
            non_sc_eligible_modules.insert(0, core_module)  # Ensure core is first if not sc_eligible

    # Sort sc_eligible modules by bonus (descending)
    sc_eligible_modules.sort(key=lambda m: m["bonus"], reverse=True)
    # Sort non_sc_eligible modules by bonus (descending)
    non_sc_eligible_modules.sort(key=lambda m: m["bonus"], reverse=True)

    # Place sc_eligible modules in supercharged slots first
    placed_module_ids = set()
    for module in sc_eligible_modules:
        if module["id"] not in placed_module_ids:
            placed = False
            for index, (x, y) in enumerate(supercharged_slots):
                if grid.get_cell(x, y)["module"] is None:
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
                    placed_module_ids.add(module["id"])
                    supercharged_slots.pop(index)
                    placed = True
                    break
            if placed:
                continue

    # Place non_sc_eligible core module in a supercharged slot if available
    if core_module and not core_module["sc_eligible"]:
        if core_module["id"] not in placed_module_ids:
            placed = False
            for index, (x, y) in enumerate(supercharged_slots):
                if grid.get_cell(x, y)["module"] is None:
                    place_module(
                        grid,
                        x,
                        y,
                        core_module["id"],
                        core_module["label"],
                        tech,
                        core_module["type"],
                        core_module["bonus"],
                        core_module["adjacency"],
                        core_module["sc_eligible"],
                        core_module["image"],
                    )
                    placed_module_ids.add(core_module["id"])
                    supercharged_slots.pop(index)
                    placed = True
                    break
            if placed:
                non_sc_eligible_modules.pop(0)  # Remove the core module from the list.

    # Place remaining non_sc_eligible modules in remaining supercharged slots
    for module in non_sc_eligible_modules:
        if module["id"] not in placed_module_ids:
            placed = False
            for index, (x, y) in enumerate(supercharged_slots):
                if grid.get_cell(x, y)["module"] is None:
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
                    placed_module_ids.add(module["id"])
                    supercharged_slots.pop(index)
                    placed = True
                    break
            if placed:
                continue

    # Place remaining modules in any active slots
    remaining_modules = [m for m in modules_to_place if m["id"] not in placed_module_ids]
    # random.shuffle(active_slots)  # Shuffle active slots for random placement - NO LONGER SHUFFLING SLOTS
    for index, (x, y) in enumerate(active_slots):
        if index < len(remaining_modules):
            module = remaining_modules[index]
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


def swap_modules(grid, tech, tech_modules):
    """
    Swaps the positions of two randomly selected modules on the grid,
    considering only modules in tech_modules and respecting modules of other techs.
    """
    module_positions = []
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech:
                module_positions.append((x, y))

    if len(module_positions) < 2:
        return  # Not enough modules of this tech to swap

    pos1, pos2 = random.sample(module_positions, 2)

    x1, y1 = pos1
    x2, y2 = pos2

    # Check if either position is occupied by a module of a different tech
    if grid.get_cell(x1, y1)["tech"] != tech or grid.get_cell(x2, y2)["tech"] != tech:
        return  # Do not swap if other tech modules are involved

    # Get the module data from each cell
    module_data_1 = grid.get_cell(x1, y1).copy()
    module_data_2 = grid.get_cell(x2, y2).copy()

    # Swap the modules, preserving only the module data
    grid.cells[y1][x1].update(
        {
            "module": module_data_2["module"],
            "label": module_data_2["label"],
            "tech": module_data_2["tech"],
            "type": module_data_2["type"],
            "bonus": module_data_2["bonus"],
            "adjacency": module_data_2["adjacency"],
            "sc_eligible": module_data_2["sc_eligible"],
            "image": module_data_2["image"],
            "module_position": (x1, y1),
        }
    )

    grid.cells[y2][x2].update(
        {
            "module": module_data_1["module"],
            "label": module_data_1["label"],
            "tech": module_data_1["tech"],
            "type": module_data_1["type"],
            "bonus": module_data_1["bonus"],
            "adjacency": module_data_1["adjacency"],
            "sc_eligible": module_data_1["sc_eligible"],
            "image": module_data_1["image"],
            "module_position": (x2, y2),
        }
    )


def move_module(grid, tech, tech_modules):
    """
    Moves a randomly selected module to any empty slot,
    considering only modules in tech_modules and respecting modules of other techs.
    """
    module_positions = []
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech:
                module_positions.append((x, y))

    if not module_positions:
        return  # No modules of this tech to move

    x, y = random.choice(module_positions)

    # Check if the selected cell is occupied by a module of a different tech
    if grid.get_cell(x, y)["tech"] != tech:
        return  # Do not move if other tech modules are involved

    empty_positions = [
        (ex, ey)
        for ey in range(grid.height)
        for ex in range(grid.width)
        if grid.get_cell(ex, ey)["module"] is None and grid.get_cell(ex, ey)["active"]
    ]

    if empty_positions:
        # Prioritize moves that increase adjacency
        best_new_x, best_new_y = None, None
        best_adjacency_change = -float("inf")

        # Get the module data from the original cell
        module_data = grid.get_cell(x, y).copy()

        for new_x, new_y in empty_positions:
            # Check if the new position is occupied by a module of a different tech
            if grid.get_cell(new_x, new_y)["tech"] is not None and grid.get_cell(new_x, new_y)["tech"] != tech:
                continue  # Skip this position if it's occupied by a different tech

            adjacency_change = calculate_adjacency_change(grid, x, y, new_x, new_y, module_data, tech)
            if adjacency_change > best_adjacency_change:
                best_adjacency_change = adjacency_change
                best_new_x, best_new_y = new_x, new_y

        # If we found a move that increases adjacency, use it
        if best_new_x is not None and best_new_y is not None:
            new_x, new_y = best_new_x, best_new_y
        else:
            # Otherwise, choose a random empty slot
            new_x, new_y = random.choice(empty_positions)

        # Move the module data to the new cell
        grid.cells[new_y][new_x].update(
            {
                "module": module_data["module"],
                "label": module_data["label"],
                "tech": module_data["tech"],
                "type": module_data["type"],
                "bonus": module_data["bonus"],
                "adjacency": module_data["adjacency"],
                "sc_eligible": module_data["sc_eligible"],
                "image": module_data["image"],
                "module_position": (new_x, new_y),
            }
        )

        # Clear the old position, preserving active and supercharged status
        grid.cells[y][x]["module"] = None
        grid.cells[y][x]["label"] = ""
        grid.cells[y][x]["tech"] = None
        grid.cells[y][x]["type"] = ""
        grid.cells[y][x]["bonus"] = 0
        grid.cells[y][x]["adjacency"] = False
        grid.cells[y][x]["sc_eligible"] = False
        grid.cells[y][x]["image"] = None
        grid.cells[y][x]["module_position"] = None


def is_adjacent(x1, y1, x2, y2):
    """Checks if two positions are adjacent."""
    return (abs(x1 - x2) == 1 and y1 == y2) or (abs(y1 - y2) == 1 and x1 == x2)


def get_adjacent_empty_positions(grid, x, y):
    """Gets a list of adjacent empty positions to a given position."""
    adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    empty_positions = []
    for ax, ay in adjacent_positions:
        if 0 <= ax < grid.width and 0 <= ay < grid.height:
            if grid.get_cell(ax, ay)["module"] is None and grid.get_cell(ax, ay)["active"]:
                empty_positions.append((ax, ay))
    return empty_positions


def get_unplaced_modules(grid, modules, ship, tech):
    """
    Gets a list of modules that have not been placed on the grid.
    """
    tech_modules = get_tech_modules(modules, ship, tech)
    placed_module_ids = set()
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech and cell["module"]:
                placed_module_ids.add(cell["module"])

    unplaced_modules = [m for m in tech_modules if m["id"] not in placed_module_ids]
    return unplaced_modules


def calculate_adjacency_change(grid, x1, y1, x2, y2, module_data, tech):
    """
    Calculates the change in adjacency bonus if a module is moved or swapped.
    """
    original_adjacency = 0
    new_adjacency = 0

    # Helper function to check adjacency for a single cell
    def check_cell_adjacency(x, y, tech):
        adjacency_count = 0
        adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for ax, ay in adjacent_positions:
            if 0 <= ax < grid.width and 0 <= ay < grid.height:
                adjacent_cell = grid.get_cell(ax, ay)
                if adjacent_cell["tech"] == tech and adjacent_cell["module"] is not None:
                    adjacency_count += 1
        return adjacency_count

    # Calculate original adjacency
    original_adjacency += check_cell_adjacency(x1, y1, tech)

    # Calculate new adjacency
    new_adjacency += check_cell_adjacency(x2, y2, tech)

    return new_adjacency - original_adjacency


def check_all_modules_placed(grid, modules, ship, tech):
    """
    Checks if all modules of a given tech have been placed on the grid.
    """
    tech_modules = get_tech_modules(modules, ship, tech)
    placed_module_ids = set()
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech and cell["module"]:
                placed_module_ids.add(cell["module"])

    return len(placed_module_ids) == len(tech_modules)
