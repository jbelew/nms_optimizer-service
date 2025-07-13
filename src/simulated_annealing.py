# simulated_annealing.py
import random
import math
import time
from grid_display import print_grid_compact, print_grid
from modules_utils import get_tech_modules
from bonus_calculations import calculate_grid_score
from module_placement import place_module, clear_all_modules_of_tech


def place_modules_with_supercharged_priority(grid, tech_modules, tech):
    """
    Places modules with priority on supercharged slots, ensuring the core module
    is always placed and ONLY sc_eligible modules go into supercharged slots.

    Args:
        grid (Grid): The grid to place modules on.
        tech_modules (list): The list of modules to place.
        tech (str): The technology type of the modules.
    """
    # Identify supercharged and active slots
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

    # Sort modules by bonus in descending order, ensuring core module is considered
    core_module = next((m for m in tech_modules if m["type"] == "core"), None)
    bonus_modules = [m for m in tech_modules if m["type"] != "core"]
    bonus_modules.sort(key=lambda m: m["bonus"], reverse=True)

    # Determine the actual modules to place based on available slots
    num_available_positions = len(supercharged_slots) + len(active_slots)
    modules_to_place_candidates = []
    if core_module:
        modules_to_place_candidates.append(core_module)
    modules_to_place_candidates.extend(bonus_modules)

    modules_to_place = modules_to_place_candidates[: min(len(modules_to_place_candidates), num_available_positions)]

    # Separate sc_eligible and non_sc_eligible modules *from the ones we intend to place*
    sc_eligible_modules = [m for m in modules_to_place if m.get("sc_eligible", False)]
    non_sc_eligible_modules = [m for m in modules_to_place if not m.get("sc_eligible", False)]

    # Sort eligible modules by bonus (descending) - Core might not be highest bonus but needs placement
    sc_eligible_modules.sort(key=lambda m: (m["type"] != "core", -m["bonus"])) # Place core first among eligible if present
    non_sc_eligible_modules.sort(key=lambda m: (m["type"] != "core", -m["bonus"])) # Place core first among non-eligible if present

    placed_module_ids = set()
    remaining_sc_eligible = []

    # --- Placement Pass 1: Place SC_ELIGIBLE modules into SUPERCHARGED slots ---
    # Shuffle slots to add randomness to initial placement within supercharged zone
    random.shuffle(supercharged_slots)
    for module in sc_eligible_modules:
        if not supercharged_slots: # No more supercharged slots left
            remaining_sc_eligible.append(module)
            continue

        placed_in_sc = False
        # Try placing in an available supercharged slot
        slot_index_to_remove = -1
        for i, (x, y) in enumerate(supercharged_slots):
             # Double check the cell is still empty (should be, but safety)
            if grid.get_cell(x, y)["module"] is None:
                place_module(
                    grid, x, y,
                    module["id"], module["label"], tech, module["type"],
                    module["bonus"], module["adjacency"], module["sc_eligible"], module["image"]
                )
                placed_module_ids.add(module["id"])
                slot_index_to_remove = i
                placed_in_sc = True
                break # Module placed, move to next module

        if placed_in_sc:
            if slot_index_to_remove != -1:
                supercharged_slots.pop(slot_index_to_remove) # Remove the used slot
        else:
            # Could not place in any remaining supercharged slot (shouldn't happen if slots exist)
            remaining_sc_eligible.append(module)

    # --- Placement Pass 2: Place ALL remaining modules into ACTIVE slots ---
    # Combine non_sc_eligible and any sc_eligible that didn't fit in supercharged slots
    modules_for_active_slots = non_sc_eligible_modules + remaining_sc_eligible
    # Sort the combined list (optional, e.g., by bonus, keeping core priority)
    modules_for_active_slots.sort(key=lambda m: (m["type"] != "core", -m["bonus"]))

    # Shuffle active slots for randomness
    random.shuffle(active_slots)
    for module in modules_for_active_slots:
        # Ensure we haven't somehow already placed this module
        if module["id"] in placed_module_ids:
            continue

        if not active_slots: # No more active slots left
            print(f"Warning: Ran out of active slots while trying to place {module['id']}")
            break # Stop trying to place

        placed_in_active = False
        slot_index_to_remove = -1
        # Try placing in an available active slot
        for i, (x, y) in enumerate(active_slots):
            if grid.get_cell(x, y)["module"] is None:
                place_module(
                    grid, x, y,
                    module["id"], module["label"], tech, module["type"],
                    module["bonus"], module["adjacency"], module["sc_eligible"], module["image"]
                )
                placed_module_ids.add(module["id"])
                slot_index_to_remove = i
                placed_in_active = True
                break # Module placed, move to next module

        if placed_in_active:
             if slot_index_to_remove != -1:
                active_slots.pop(slot_index_to_remove) # Remove the used slot
        # else: Module could not be placed (should only happen if out of slots)

    # Final check (optional): Verify if all intended modules were placed
    if len(placed_module_ids) < len(modules_to_place):
        unplaced_ids = [m["id"] for m in modules_to_place if m["id"] not in placed_module_ids]
        print(f"Warning: Could not place all intended modules during initial placement. Unplaced: {unplaced_ids}")


# --- Rest of the simulated_annealing.py file ---

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
    final_swap_probability=0.25,
    start_from_current_grid: bool = False,
    max_processing_time: float = 360.0
):
    """
    Performs simulated annealing to optimize module placement on a grid.

    Args:
        grid (Grid): The initial grid state.
        ship (str): The ship type key.
        modules (dict): The main modules dictionary.
        tech (str): The technology key.
        player_owned_rewards (list, optional): List of reward module IDs owned. Defaults to None.
        initial_temperature (float): Starting temperature.
        cooling_rate (float): Rate at which temperature decreases (e.g., 0.995).
        stopping_temperature (float): Temperature at which annealing stops.
        iterations_per_temp (int): Number of iterations at each temperature level.
        initial_swap_probability (float): Probability of attempting a swap vs. a move at high temp.
        final_swap_probability (float): Probability of attempting a swap vs. a move at low temp.
        start_from_current_grid (bool): If True, skips the initial clearing and
                                        placement, starting optimization from the
                                        provided grid state. Ideal for polishing.
        max_processing_time (float): Maximum time in seconds allowed for annealing.

    Returns:
        tuple: (best_grid, best_score) or (None, 0.0) on failure.
    """
    start_time = time.time()

    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None:
        print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
        # Return a copy of the original grid and 0 score if modules aren't found
        return grid.copy(), 0.0

    current_grid = grid.copy() # Work on a copy from the start

    # --- Determine modules to consider ---
    modules_to_consider = []
    if not start_from_current_grid:
        # Clear the target tech from our working copy *before* counting slots
        clear_all_modules_of_tech(current_grid, tech)

        # Now, count empty active slots on the *cleared* current_grid
        active_slots_count = 0
        for y in range(current_grid.height): # Iterate over current_grid
            for x in range(current_grid.width):
                cell = current_grid.get_cell(x, y)
                # This condition is now correct because current_grid has target tech cleared
                if cell["module"] is None and cell["active"]: # Count empty & active
                    active_slots_count += 1

        print(f"Active slots count: {active_slots_count}")  

        core_module = next((m for m in tech_modules if m["type"] == "core"), None)
        bonus_modules = [m for m in tech_modules if m["type"] != "core"]
        bonus_modules.sort(key=lambda m: m["bonus"], reverse=True)

        num_to_take = active_slots_count
        if core_module:
            modules_to_consider.append(core_module)
            if num_to_take > 0:
                 num_to_take -= 1 # Account for core module taking a slot

        modules_to_consider.extend(bonus_modules[:num_to_take])

        if len(modules_to_consider) == 0 and len(tech_modules) > 0:
             print(f"Warning: No active empty slots available for SA initial placement of {tech}.")
             # Return a copy of the original grid if no slots available
             return grid.copy(), calculate_grid_score(grid, tech)

    else: # Polishing mode
        # Consider all modules of the target tech currently placed on the grid
        modules_on_grid_ids = set()
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                if cell["tech"] == tech and cell["module"] is not None:
                    modules_on_grid_ids.add(cell["module"])

        # Get the full definitions for these modules
        modules_to_consider = [m for m in tech_modules if m["id"] in modules_on_grid_ids]

        if not modules_to_consider:
             print(f"Info: SA Polishing - No modules of tech '{tech}' found on the grid to polish.")
             # Return the original grid as there's nothing to do
             return grid.copy(), calculate_grid_score(grid, tech)


    if not start_from_current_grid:
        # current_grid is already a copy and has been cleared of the target tech above.
        # Now, place the selected modules.
        # Place modules using the updated priority function
        place_modules_with_supercharged_priority(current_grid, modules_to_consider, tech)
    # else: If start_from_current_grid is True, we use the grid as passed in.

    # Check if any modules were actually placed/present
    if not any(current_grid.get_cell(x,y)['tech'] == tech for y in range(current_grid.height) for x in range(current_grid.width)):
        if len(modules_to_consider) > 0:
            print(f"Warning: SA - Initial placement failed or no modules placed for {tech}. Returning cleared grid.")
            cleared_grid = grid.copy()
            clear_all_modules_of_tech(cleared_grid, tech)
            return cleared_grid, 0.0
        else:
            # No modules were intended to be placed (e.g., no slots or no modules defined)
            print(f"Info: SA - No modules to place for {tech}. Returning original grid state.")
            return grid.copy(), calculate_grid_score(grid, tech) # Return original score


    current_score = calculate_grid_score(current_grid, tech)
    best_grid = current_grid.copy()
    best_score = current_score

    temperature = initial_temperature
    swap_probability = initial_swap_probability

    # --- Annealing Loop ---
    while temperature > stopping_temperature:
        if time.time() - start_time > max_processing_time:
            print(f"INFO -- SA: Max processing time ({max_processing_time}s) exceeded. Returning best found.")
            break

        swap_probability = get_swap_probability(
            temperature, initial_temperature, stopping_temperature,
            initial_swap_probability, final_swap_probability
        )

        for _ in range(iterations_per_temp):
            neighbor_grid = current_grid.copy()
            # Ensure modules_to_consider reflects the modules actually on the grid now
            current_modules_on_grid_defs = [m for m in tech_modules if m['id'] in {neighbor_grid.get_cell(x,y)['module'] for y in range(neighbor_grid.height) for x in range(neighbor_grid.width) if neighbor_grid.get_cell(x,y)['tech'] == tech}]

            if not current_modules_on_grid_defs: # Safety check if grid somehow became empty
                continue

            if random.random() < swap_probability:
                swap_modules(neighbor_grid, tech, current_modules_on_grid_defs)
            else:
                move_module(neighbor_grid, tech, current_modules_on_grid_defs)

            neighbor_score = calculate_grid_score(neighbor_grid, tech)

            delta_e = neighbor_score - current_score
            if delta_e > 0 or random.random() < math.exp(delta_e / temperature):
                current_grid = neighbor_grid
                current_score = neighbor_score
                if current_score > best_score:
                    best_grid = current_grid.copy()
                    # <<< Add your print statement here to check things >>>
                    print(f"DEBUG SA -- New best score for {tech}: {current_score:.4f} (Temp: {temperature:.2f})")
                    # print_grid_compact(best_grid)
                    best_score = current_score

        temperature *= cooling_rate
    # --- End Annealing Loop ---

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"INFO -- SA finished ({'Polish' if start_from_current_grid else 'Full'}). Best score: {best_score:.4f}. Time: {elapsed_time:.4f}s")

    # Final check for validity (especially important if polishing)
    final_modules_placed = check_all_modules_placed(best_grid, modules, ship, tech, player_owned_rewards, modules_to_consider)
    if not final_modules_placed:
         print(f"WARNING -- SA: Final grid for {tech} did not contain all expected modules. This might indicate an issue.")
         # Decide how to handle this - return best_grid anyway, or revert?
         # For now, let's return the best_grid found, but log the warning.

    # Check for zero score if modules should provide bonus
    if best_score < 1e-9 and any(m["bonus"] > 0 or m["type"] == "core" for m in modules_to_consider):
         print(f"WARNING -- SA solver resulted in zero score for {ship}/{tech} despite modules existing. Potential failure.")
         # If polishing failed badly, return the original grid state
         if start_from_current_grid:
             initial_score = calculate_grid_score(grid, tech) # Score of the grid passed in
             print(f"INFO -- Returning original grid state (score {initial_score:.4f}) due to zero score after SA polish.")
             return grid.copy(), initial_score
         # If a full run failed, maybe return a cleared grid
         # else:
         #     cleared_grid = grid.copy()
         #     clear_all_modules_of_tech(cleared_grid, tech)
         #     return cleared_grid, 0.0
         # Let's return the best_grid found for now, even if score is 0
         return best_grid, best_score

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
    if temperature >= initial_temperature:
        return initial_swap_probability
    if temperature <= stopping_temperature:
        return final_swap_probability

    # Linear interpolation between initial and final probabilities
    progress = (initial_temperature - temperature) / (initial_temperature - stopping_temperature)
    return initial_swap_probability - (initial_swap_probability - final_swap_probability) * progress


def swap_modules(grid, tech, tech_modules_on_grid):
    """
    Swaps the positions of two randomly selected modules of the specified tech
    that are currently on the grid.
    """
    module_positions = []
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            # Ensure the module is of the target tech AND is one we are considering
            if cell["tech"] == tech and cell["module"] in {m['id'] for m in tech_modules_on_grid}:
                module_positions.append((x, y))

    if len(module_positions) < 2:
        return # Not enough modules to swap

    pos1, pos2 = random.sample(module_positions, 2)
    x1, y1 = pos1
    x2, y2 = pos2

    # Get the full module data from each cell before swapping
    module_data_1 = grid.get_cell(x1, y1).copy()
    module_data_2 = grid.get_cell(x2, y2).copy()

    # Place module 2 data into cell 1
    place_module(
        grid, x1, y1,
        module_data_2["module"], module_data_2["label"], module_data_2["tech"],
        module_data_2["type"], module_data_2["bonus"], module_data_2["adjacency"],
        module_data_2["sc_eligible"], module_data_2["image"]
    )
    # Update module_position after placement
    grid.cells[y1][x1]["module_position"] = (x1, y1)


    # Place module 1 data into cell 2
    place_module(
        grid, x2, y2,
        module_data_1["module"], module_data_1["label"], module_data_1["tech"],
        module_data_1["type"], module_data_1["bonus"], module_data_1["adjacency"],
        module_data_1["sc_eligible"], module_data_1["image"]
    )
    # Update module_position after placement
    grid.cells[y2][x2]["module_position"] = (x2, y2)


def move_module(grid, tech, tech_modules_on_grid):
    """
    Moves a randomly selected module of the specified tech (that's on the grid)
    to a random empty active slot.
    """
    module_positions = []
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech and cell["module"] in {m['id'] for m in tech_modules_on_grid}:
                module_positions.append((x, y))

    if not module_positions:
        return # No modules to move

    x_from, y_from = random.choice(module_positions)
    module_data_to_move = grid.get_cell(x_from, y_from).copy()

    empty_active_positions = [
        (ex, ey)
        for ey in range(grid.height)
        for ex in range(grid.width)
        if grid.get_cell(ex, ey)["module"] is None and grid.get_cell(ex, ey)["active"]
    ]

    if not empty_active_positions:
        return # No empty slots to move to

    # --- Simple Random Move (Original Logic) ---
    x_to, y_to = random.choice(empty_active_positions)

    # Place the module in the new empty slot
    place_module(
        grid, x_to, y_to,
        module_data_to_move["module"], module_data_to_move["label"], module_data_to_move["tech"],
        module_data_to_move["type"], module_data_to_move["bonus"], module_data_to_move["adjacency"],
        module_data_to_move["sc_eligible"], module_data_to_move["image"]
    )
    # Update module_position after placement
    grid.cells[y_to][x_to]["module_position"] = (x_to, y_to)


    # Clear the original position (preserving active/supercharged status)
    grid.cells[y_from][x_from]["module"] = None
    grid.cells[y_from][x_from]["label"] = ""
    grid.cells[y_from][x_from]["tech"] = None
    grid.cells[y_from][x_from]["type"] = ""
    grid.cells[y_from][x_from]["bonus"] = 0.0
    grid.cells[y_from][x_from]["adjacency"] = False # Adjacency type is part of module def
    grid.cells[y_from][x_from]["sc_eligible"] = False
    grid.cells[y_from][x_from]["image"] = None
    grid.cells[y_from][x_from]["module_position"] = None
    grid.cells[y_from][x_from]["total"] = 0.0
    grid.cells[y_from][x_from]["adjacency_bonus"] = 0.0


# --- Helper functions (is_adjacent, get_adjacent_empty_positions, etc.) ---
# These seem okay, but calculate_adjacency_change is complex and might not be needed
# if we recalculate the full score anyway. Let's keep them for now but note they aren't used
# in the simplified move/swap logic above.

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

def get_unplaced_modules(grid, modules, ship, tech, player_owned_rewards=None):
    """
    Gets a list of modules that have not been placed on the grid for a specific tech.
    """
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None: return []

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
    (Note: This is complex and might be less reliable than just recalculating score)
    """
    original_adjacency = 0
    new_adjacency = 0

    def check_cell_adjacency(x, y, tech):
        adjacency_count = 0
        adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for ax, ay in adjacent_positions:
            if 0 <= ax < grid.width and 0 <= ay < grid.height:
                adjacent_cell = grid.get_cell(ax, ay)
                if adjacent_cell["tech"] == tech and adjacent_cell["module"] is not None:
                    adjacency_count += 1
        return adjacency_count

    original_adjacency += check_cell_adjacency(x1, y1, tech)
    new_adjacency += check_cell_adjacency(x2, y2, tech)

    return new_adjacency - original_adjacency


def check_all_modules_placed(grid, modules, ship, tech, player_owned_rewards=None, modules_expected=None):
    """
    Checks if all expected modules for a given tech have been placed in the grid.

    Args:
        grid (Grid): The grid layout.
        modules (dict): The module data.
        ship (str): The ship type.
        tech (str): The technology type.
        player_owned_rewards (list, optional): Rewards owned by the player. Defaults to None.
        modules_expected (list, optional): The specific list of module definitions
                                           that were intended to be placed. If None,
                                           it fetches based on ship/tech/rewards.

    Returns:
        bool: True if all expected modules are placed, False otherwise.
    """
    if modules_expected is None:
        modules_expected = get_tech_modules(modules, ship, tech, player_owned_rewards)
        if modules_expected is None:
            print(f"Warning: check_all_modules_placed - Could not get expected modules for {ship}/{tech}.")
            return False # Cannot verify if expected modules are unknown

    if not modules_expected:
        return True # If no modules were expected, then all (zero) are placed.

    placed_module_ids = set()
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech and cell["module"]:
                placed_module_ids.add(cell["module"])

    expected_module_ids = {module["id"] for module in modules_expected}

    # Check if the set of placed IDs matches the set of expected IDs
    return placed_module_ids == expected_module_ids
