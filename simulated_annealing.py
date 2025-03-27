# simulated_annealing.py
import random
import math
import time  # Import the time module
from grid_utils import Grid
from modules_data import get_tech_modules
from bonus_calculations import calculate_grid_score
from module_placement import place_module, clear_all_modules_of_tech

def simulated_annealing(grid, ship, modules, tech, player_owned_rewards=None, initial_temperature=3500, cooling_rate=0.99, stopping_temperature=0.1, iterations_per_temp=45, initial_swap_probability=0.6, final_swap_probability=0.4):
    """
    Performs simulated annealing to optimize module placement on a grid,
    prioritizing adjacency bonuses.

    Args:
        grid (Grid): The initial grid layout.
        ship (str): The ship type.
        modules (dict): The module data.
        tech (str): The technology type.
        player_owned_rewards (list, optional): Rewards owned by the player. Defaults to None.
        initial_temperature (float): The starting temperature.
        cooling_rate (float): The rate at which the temperature decreases.
        stopping_temperature (float): The temperature at which the algorithm stops.
        iterations_per_temp (int): The number of iterations at each temperature.
        initial_swap_probability (float): The starting swap probability.
        final_swap_probability (float): The ending swap probability.

    Returns:
        tuple: A tuple containing the best grid found and its score.
    """
    start_time = time.time()  # Start timing
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None:
        print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
        return grid, 0.0

    # Check if there are enough available positions for all modules
    available_positions = [
        (x, y) for y in range(grid.height) for x in range(grid.width)
        if grid.get_cell(x, y)["module"] is None and grid.get_cell(x, y)["active"]
    ]
    if len(available_positions) < len(tech_modules):
        print(f"Not enough available positions to place all modules for ship '{ship}' and tech '{tech}'.")
        return grid, 0.0

    # Initialize the current state with a placement that prioritizes supercharged slots
    current_grid = grid.copy()
    place_modules_with_supercharged_priority(current_grid, tech_modules, tech)
    current_score = calculate_grid_score(current_grid, tech)

    best_grid = current_grid.copy()
    best_score = current_score

    temperature = initial_temperature
    while temperature > stopping_temperature:
        # Calculate the adaptive swap probability
        swap_probability = get_swap_probability(temperature, initial_temperature, stopping_temperature, initial_swap_probability, final_swap_probability)

        # print(f"DEBUG -- Current temperature: {temperature:.2f}, Current Score: {current_score:.2f}, Best Score: {best_score:.2f}, Swap Probability: {swap_probability:.2f}")
        for _ in range(iterations_per_temp):
            # Create a neighbor by either swapping or moving a module
            neighbor_grid = current_grid.copy()
            if random.random() < swap_probability:  # Use the adaptive swap_probability
                swap_modules(neighbor_grid, tech)
            else:
                move_module(neighbor_grid, tech)
            neighbor_score = calculate_grid_score(neighbor_grid, tech)

            # Decide whether to accept the neighbor
            delta_e = neighbor_score - current_score
            if delta_e > 0:
                # Accept better solutions
                current_grid = neighbor_grid.copy()
                current_score = neighbor_score
                if current_score > best_score:
                    best_grid = current_grid.copy()
                    best_score = current_score
            else:
                # Accept worse solutions with a probability
                acceptance_probability = math.exp(delta_e / temperature)
                if random.random() < acceptance_probability:
                    current_grid = neighbor_grid.copy()
                    current_score = neighbor_score

        # Cool down
        temperature *= cooling_rate

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"DEBUG -- Simulated annealing finished. Best score found: {best_score:.2f} -- Time: {elapsed_time:.4f}s")
    return best_grid, best_score

def get_swap_probability(temperature, initial_temperature, stopping_temperature, initial_swap_probability, final_swap_probability):
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
    Places modules with priority on supercharged slots, but with some randomness.

    Args:
        grid (Grid): The grid to place modules on.
        tech_modules (list): The list of modules to place.
        tech (str): The technology type of the modules.
    """
    # Clear any existing modules of the same tech from the grid.
    clear_all_modules_of_tech(grid, tech)

    # Identify supercharged and active slots
    supercharged_slots = [
        (x, y) for y in range(grid.height) for x in range(grid.width)
        if grid.get_cell(x, y)["supercharged"] and grid.get_cell(x, y)["active"]
    ]
    active_slots = [
        (x, y) for y in range(grid.height) for x in range(grid.width)
        if grid.get_cell(x, y)["module"] is None and grid.get_cell(x, y)["active"]
    ]

    # Sort modules by bonus in descending order
    sorted_modules = sorted(tech_modules, key=lambda m: m["bonus"], reverse=True)

    # Randomly select a subset of the highest-bonus modules to place in supercharged slots
    num_supercharged_modules = min(len(supercharged_slots), random.randint(1, len(sorted_modules)))
    supercharged_modules = random.sample(sorted_modules[:len(sorted_modules)], num_supercharged_modules)

    # Place the selected modules in random supercharged slots
    random.shuffle(supercharged_slots)
    for index, (x, y) in enumerate(supercharged_slots):
        if index < len(supercharged_modules):
            module = supercharged_modules[index]
            place_module(
                grid, x, y,
                module["id"], module["label"], tech,
                module["type"], module["bonus"],
                module["adjacency"], module["sc_eligible"],
                module["image"],
            )

    # Place remaining modules in active slots, prioritizing adjacency
    remaining_modules = [m for m in sorted_modules if m not in supercharged_modules]
    random.shuffle(remaining_modules)
    
    # Get a list of occupied positions
    occupied_positions = [(x, y) for y in range(grid.height) for x in range(grid.width) if grid.get_cell(x, y)["module"] is not None]

    # Place remaining modules, prioritizing adjacency
    for module in remaining_modules:
        # Find adjacent empty slots
        adjacent_empty_slots = []
        for ox, oy in occupied_positions:
            for ax, ay in get_adjacent_empty_positions(grid, ox, oy):
                if (ax, ay) not in adjacent_empty_slots and (ax, ay) in active_slots:
                    adjacent_empty_slots.append((ax, ay))

        # If there are adjacent empty slots, choose one randomly
        if adjacent_empty_slots:
            x, y = random.choice(adjacent_empty_slots)
        else:
            # Otherwise, choose a random empty slot
            if active_slots:
                x, y = random.choice(active_slots)
            else:
                continue  # No more slots available

        place_module(
            grid, x, y,
            module["id"], module["label"], tech,
            module["type"], module["bonus"],
            module["adjacency"], module["sc_eligible"],
            module["image"],
        )
        occupied_positions.append((x,y))
        active_slots.remove((x,y))

def swap_modules(grid, tech):
    """
    Swaps the positions of two randomly selected modules on the grid,
    prioritizing swaps between adjacent modules.
    """
    module_positions = []
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.get_cell(x, y)["tech"] == tech:
                module_positions.append((x, y))

    if len(module_positions) < 2:
        return  # Not enough modules to swap

    # Prioritize adjacent swaps
    adjacent_pairs = []
    for i in range(len(module_positions)):
        for j in range(i + 1, len(module_positions)):
            x1, y1 = module_positions[i]
            x2, y2 = module_positions[j]
            if is_adjacent(x1, y1, x2, y2):
                adjacent_pairs.append((module_positions[i], module_positions[j]))

    if adjacent_pairs:
        pos1, pos2 = random.choice(adjacent_pairs)
    else:
        pos1, pos2 = random.sample(module_positions, 2)

    x1, y1 = pos1
    x2, y2 = pos2

    # Get the module data from each cell
    module_data_1 = grid.get_cell(x1, y1).copy()
    module_data_2 = grid.get_cell(x2, y2).copy()

    # Swap the modules, preserving only the module data
    grid.cells[y1][x1].update({
        "module": module_data_2["module"],
        "label": module_data_2["label"],
        "tech": module_data_2["tech"],
        "type": module_data_2["type"],
        "bonus": module_data_2["bonus"],
        "adjacency": module_data_2["adjacency"],
        "sc_eligible": module_data_2["sc_eligible"],
        "image": module_data_2["image"],
        "module_position": (x1, y1)
    })

    grid.cells[y2][x2].update({
        "module": module_data_1["module"],
        "label": module_data_1["label"],
        "tech": module_data_1["tech"],
        "type": module_data_1["type"],
        "bonus": module_data_1["bonus"],
        "adjacency": module_data_1["adjacency"],
        "sc_eligible": module_data_1["sc_eligible"],
        "image": module_data_1["image"],
        "module_position": (x2, y2)
    })

def move_module(grid, tech):
    """
    Moves a randomly selected module to an adjacent empty slot, if possible.
    """
    module_positions = []
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.get_cell(x, y)["tech"] == tech:
                module_positions.append((x, y))

    if not module_positions:
        return  # No modules to move

    x, y = random.choice(module_positions)
    adjacent_empty_positions = get_adjacent_empty_positions(grid, x, y)

    if adjacent_empty_positions:
        new_x, new_y = random.choice(adjacent_empty_positions)
        
        # Get the module data from the original cell
        module_data = grid.get_cell(x, y).copy()

        # Move the module data to the new cell
        grid.cells[new_y][new_x].update({
            "module": module_data["module"],
            "label": module_data["label"],
            "tech": module_data["tech"],
            "type": module_data["type"],
            "bonus": module_data["bonus"],
            "adjacency": module_data["adjacency"],
            "sc_eligible": module_data["sc_eligible"],
            "image": module_data["image"],
            "module_position": (new_x, new_y)
        })

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
