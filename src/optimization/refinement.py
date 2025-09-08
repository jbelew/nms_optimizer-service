# optimization/refinement.py
import random
import math
import time
import gevent
import logging
from copy import deepcopy
from itertools import permutations
from typing import Optional

from src.grid_utils import restore_original_state, apply_localized_grid_changes
from src.modules_utils import get_tech_modules
from src.bonus_calculations import calculate_grid_score
from src.module_placement import place_module, clear_all_modules_of_tech
from .helpers import check_all_modules_placed
from .windowing import create_localized_grid, create_localized_grid_ml


def _handle_ml_opportunity(
    grid,
    modules,
    ship,
    tech,
    player_owned_rewards,
    opportunity_x,
    opportunity_y,
    window_width,
    window_height,
    progress_callback=None,
    run_id=None,
    stage=None,
    send_grid_updates=False,
    solve_type: Optional[str] = None,
    tech_modules=None,
):
    """Handles the ML-based refinement within an opportunity window."""
    from src.ml_placement import ml_placement  # Keep import local if possible

    if player_owned_rewards is None:
        player_owned_rewards = []

    logging.info(
        f"Using ML for opportunity refinement at ({opportunity_x}, {opportunity_y}) with window {window_width}x{window_height}"
    )
    # 1. Create localized grid specifically for ML
    # <<< Pass window dimensions to create_localized_grid_ml >>>
    localized_grid_ml, start_x, start_y, original_state_map = create_localized_grid_ml(
        grid,
        opportunity_x,
        opportunity_y,
        tech,
        window_width,
        window_height,  # <<< Pass dimensions
    )

    # 2. Run ML placement on the localized grid
    #    Make sure player_owned_rewards is passed correctly!
    # <<< Pass localized grid dimensions to ml_placement >>>
    ml_refined_grid, ml_refined_score_local = ml_placement(
        localized_grid_ml,
        ship,
        tech,
        full_grid_original=grid,  # Pass the original full grid
        start_x_original=opportunity_x,  # Pass the original start_x
        start_y_original=opportunity_y,  # Pass the original start_y
        player_owned_rewards=player_owned_rewards,
        model_grid_width=localized_grid_ml.width,  # <<< Use actual localized width
        model_grid_height=localized_grid_ml.height,  # <<< Use actual localized height
        polish_result=True,  # Usually don't polish within the main polish step
        progress_callback=progress_callback,
        run_id=run_id,
        stage=stage,
        send_grid_updates=send_grid_updates,
        original_state_map=original_state_map,
        solve_type=solve_type,
        tech_modules=tech_modules,  # type: ignore
    )

    # 3. Process ML result (logic remains the same)
    if ml_refined_grid is not None:
        # logging.info(f"ML refinement produced a grid. Applying changes...")
        grid_copy = grid.copy()
        clear_all_modules_of_tech(grid_copy, tech)
        apply_localized_grid_changes(grid_copy, ml_refined_grid, tech, start_x, start_y)
        restore_original_state(grid_copy, original_state_map)
        new_score_global = calculate_grid_score(grid_copy, tech)
        # logging.info(f"Score after ML refinement and restoration: {new_score_global:.4f}")
        return grid_copy, new_score_global
    else:
        # Handle ML failure (logic remains the same)
        logging.info("ML refinement failed or returned None. No changes applied.")
        grid_copy = grid.copy()
        restore_original_state(grid_copy, original_state_map)
        original_score = calculate_grid_score(grid_copy, tech)
        logging.info(
            f"Returning grid with original score after failed ML: {original_score:.4f}"
        )
        return None, 0.0


def _handle_sa_refine_opportunity(
    grid,
    modules,
    ship,
    tech,
    player_owned_rewards,
    opportunity_x,
    opportunity_y,
    window_width,
    window_height,
    progress_callback=None,
    run_id=None,
    stage=None,
    send_grid_updates=False,
    solve_type: Optional[str] = None,
    tech_modules=None,
):
    """Handles the SA/Refine-based refinement within an opportunity window."""
    logging.info(
        f"Using SA/Refine for opportunity refinement at ({opportunity_x}, {opportunity_y}) with window {window_width}x{window_height}"
    )

    # --- *** Modification Start *** ---
    # Create a copy of the grid *before* clearing, to preserve the original state for localization
    grid_copy_for_localization = grid.copy()

    # Clear the target tech from the original grid passed in (or a copy if preferred, but this modifies the copy passed from optimize_placement)
    clear_all_modules_of_tech(grid_copy_for_localization, tech)
    # --- *** Modification End *** ---

    # Create a localized grid (preserves other tech modules)
    # <<< Pass window dimensions to create_localized_grid >>>
    # <<< Use the copy made *before* clearing for localization info >>>
    localized_grid, start_x, start_y = create_localized_grid(
        grid_copy_for_localization,
        opportunity_x,
        opportunity_y,
        tech,
        window_width,
        window_height,  # <<< Pass dimensions
    )
    # print_grid(localized_grid) # Optional debug

    # Get the number of modules for the given tech (no change)
    if tech_modules is None:
        tech_modules = get_tech_modules(
            modules, ship, tech, player_owned_rewards, solve_type=solve_type
        )
    num_modules = len(tech_modules) if tech_modules else 0

    # Refine the localized grid (no change in logic here)
    if num_modules < 6:
        logging.info(f"{tech} has less than 6 modules, running refine_placement")
        temp_refined_grid, temp_refined_bonus_local = refine_placement(
            localized_grid,
            ship,
            modules,
            tech,
            player_owned_rewards,
            solve_type=solve_type,
            tech_modules=tech_modules,
        )
    else:
        logging.info(f"{tech} has 6 or more modules, running simulated_annealing")
        temp_refined_grid, temp_refined_bonus_local = simulated_annealing(
            localized_grid,
            ship,
            modules,
            tech,
            grid,  # full_grid
            player_owned_rewards,
            start_x=start_x,  # Pass start_x
            start_y=start_y,  # Pass start_y
            progress_callback=progress_callback,
            run_id=run_id,
            stage=stage,
            send_grid_updates=send_grid_updates,
            solve_type=solve_type,
            tech_modules=tech_modules or [],
        )

    # Process SA/Refine result (logic remains the same)
    if temp_refined_grid is not None:
        calculate_grid_score(temp_refined_grid, tech)
        # Apply changes back to the main grid copy (grid - which was cleared earlier)
        clear_all_modules_of_tech(grid, tech)
        apply_localized_grid_changes(grid, temp_refined_grid, tech, start_x, start_y)
        new_score_global = calculate_grid_score(grid, tech)
        logging.info(f"Score after SA/Refine refinement: {new_score_global:.4f}")
        # print_grid(grid) # Print the modified grid (grid)
        return grid, new_score_global
    else:
        logging.info("SA/Refine refinement failed. No changes applied.")
        # Return the grid (which was cleared of the tech) and indicate failure
        return grid, -1.0


def refine_placement(
    grid,
    ship,
    modules,
    tech,
    player_owned_rewards=None,
    solve_type: Optional[str] = None,
    tech_modules=None,
):
    optimal_grid = None
    highest_bonus = 0.0
    if tech_modules is None:
        tech_modules = get_tech_modules(
            modules, ship, tech, player_owned_rewards, solve_type=solve_type
        )

    if tech_modules is None:
        logging.error(f"No modules found for ship '{ship}' and tech '{tech}'.")
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
        shuffled_tech_modules = tech_modules[
            :
        ]  # Create a copy to avoid modifying the original list
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
    logging.info(
        f"refine_placement completed {iteration_count} iterations for ship: '{ship}' -- tech: '{tech}'"
    )

    return optimal_grid, highest_bonus


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

    modules_to_place = modules_to_place_candidates[
        : min(len(modules_to_place_candidates), num_available_positions)
    ]

    # Separate sc_eligible and non_sc_eligible modules *from the ones we intend to place*
    sc_eligible_modules = [m for m in modules_to_place if m.get("sc_eligible", False)]
    non_sc_eligible_modules = [
        m for m in modules_to_place if not m.get("sc_eligible", False)
    ]

    # Sort eligible modules by bonus (descending) - Core might not be highest bonus but needs placement
    sc_eligible_modules.sort(
        key=lambda m: (m["type"] != "core", -m["bonus"])
    )  # Place core first among eligible if present
    non_sc_eligible_modules.sort(
        key=lambda m: (m["type"] != "core", -m["bonus"])
    )  # Place core first among non-eligible if present

    placed_module_ids = set()
    remaining_sc_eligible = []

    # --- Placement Pass 1: Place SC_ELIGIBLE modules into SUPERCHARGED slots ---
    # Shuffle slots to add randomness to initial placement within supercharged zone
    random.shuffle(supercharged_slots)
    for module in sc_eligible_modules:
        if not supercharged_slots:  # No more supercharged slots left
            remaining_sc_eligible.append(module)
            continue

        placed_in_sc = False
        # Try placing in an available supercharged slot
        slot_index_to_remove = -1
        for i, (x, y) in enumerate(supercharged_slots):
            # Double check the cell is still empty (should be, but safety)
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
                slot_index_to_remove = i
                placed_in_sc = True
                break  # Module placed, move to next module

        if placed_in_sc:
            if slot_index_to_remove != -1:
                supercharged_slots.pop(slot_index_to_remove)  # Remove the used slot
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

        if not active_slots:  # No more active slots left
            logging.warning(
                f"SA: Ran out of active slots while trying to place {module['id']}"
            )
            break  # Stop trying to place

        placed_in_active = False
        slot_index_to_remove = -1
        # Try placing in an available active slot
        for i, (x, y) in enumerate(active_slots):
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
                slot_index_to_remove = i
                placed_in_active = True
                break  # Module placed, move to next module

        if placed_in_active:
            if slot_index_to_remove != -1:
                active_slots.pop(slot_index_to_remove)  # Remove the used slot
        # else: Module could not be placed (should only happen if out of slots)

    # Final check (optional): Verify if all intended modules were placed
    if len(placed_module_ids) < len(modules_to_place):
        unplaced_ids = [
            m["id"] for m in modules_to_place if m["id"] not in placed_module_ids
        ]
        logging.warning(
            f"SA: Could not place all intended modules during initial placement. Unplaced: {unplaced_ids}"
        )


# --- Rest of the simulated_annealing.py file ---


def simulated_annealing(
    grid,
    ship,
    modules,
    tech,
    full_grid,  # Added full_grid
    player_owned_rewards=None,
    initial_temperature=4000,
    cooling_rate=0.995,
    stopping_temperature=1.5,
    iterations_per_temp=35,
    initial_swap_probability=0.55,
    final_swap_probability=0.25,
    start_from_current_grid: bool = False,
    max_processing_time: float = 360.0,
    progress_callback=None,
    run_id=None,
    stage=None,
    progress_offset=0,
    progress_scale=100,
    send_grid_updates=False,
    start_x: int = 0,  # Added start_x
    start_y: int = 0,  # Added start_y
    solve_type: Optional[str] = None,
    tech_modules: list = None,
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
    logging.info(f"SA: Starting optimization for ship='{ship}', tech='{tech}'")
    logging.info(f"SA: Parameters initial_temp={initial_temperature}, cooling_rate={cooling_rate}, stopping_temp={stopping_temperature}, iterations_per_temp={iterations_per_temp}, max_processing_time={max_processing_time}")
    logging.info(f"SA: Mode: {'Polishing' if start_from_current_grid else 'Full Run'}")
    start_time = time.time()

    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards, solve_type=solve_type)
    if tech_modules is None:
        logging.error(f"SA: No modules found for ship '{ship}' and tech '{tech}'.")
        # Return a copy of the original grid and 0 score if modules aren't found
        return grid.copy(), 0.0

    current_grid = grid.copy()  # Work on a copy from the start

    # --- Determine modules to consider ---
    modules_to_consider = []
    if not start_from_current_grid:
        # Clear the target tech from our working copy *before* counting slots
        clear_all_modules_of_tech(current_grid, tech)

        # Now, count empty active slots on the *cleared* current_grid
        active_slots_count = 0
        for y in range(current_grid.height):  # Iterate over current_grid
            for x in range(current_grid.width):
                cell = current_grid.get_cell(x, y)
                # This condition is now correct because current_grid has target tech cleared
                if cell["module"] is None and cell["active"]:  # Count empty & active
                    active_slots_count += 1

        logging.info(f"SA: Active slots count: {active_slots_count}")

        core_module = next((m for m in tech_modules if m["type"] == "core"), None)
        bonus_modules = [m for m in tech_modules if m["type"] != "core"]
        bonus_modules.sort(key=lambda m: m["bonus"], reverse=True)

        num_to_take = active_slots_count
        if core_module:
            modules_to_consider.append(core_module)
            if num_to_take > 0:
                num_to_take -= 1  # Account for core module taking a slot

        modules_to_consider.extend(bonus_modules[:num_to_take])

        if len(modules_to_consider) == 0 and len(tech_modules) > 0:
            logging.warning(
                f"SA: No active empty slots available for SA initial placement of {tech}."
            )
            # Return a copy of the original grid if no slots available
            return grid.copy(), calculate_grid_score(grid, tech)

    else:  # Polishing mode
        # Consider all modules of the target tech currently placed on the grid
        modules_on_grid_ids = set()
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                if cell["tech"] == tech and cell["module"] is not None:
                    modules_on_grid_ids.add(cell["module"])

        # Get the full definitions for these modules
        if tech_modules is None:
            tech_modules = []
        modules_to_consider = [
            m for m in tech_modules if m["id"] in modules_on_grid_ids
        ]

        if not modules_to_consider:
            logging.info(
                f"SA: Polishing - No modules of tech '{tech}' found on the grid to polish."
            )
            # Return the original grid as there's nothing to do
            return grid.copy(), calculate_grid_score(grid, tech)

    if not start_from_current_grid:
        # current_grid is already a copy and has been cleared of the target tech above.
        # Now, place the selected modules.
        # Place modules using the updated priority function
        place_modules_with_supercharged_priority(
            current_grid, modules_to_consider, tech
        )
    # else: If start_from_current_grid is True, we use the grid as passed in.

    # Check if any modules were actually placed/present
    if not any(
        current_grid.get_cell(x, y)["tech"] == tech
        for y in range(current_grid.height)
        for x in range(current_grid.width)
    ):
        if len(modules_to_consider) > 0:
            logging.warning(
                f"SA: Initial placement failed or no modules placed for {tech}. Returning cleared grid."
            )
            cleared_grid = grid.copy()
            clear_all_modules_of_tech(cleared_grid, tech)
            return cleared_grid, 0.0
        else:
            # No modules were intended to be placed (e.g., no slots or no modules defined)
            logging.info(
                f"SA: No modules to place for {tech}. Returning original grid state."
            )
            return grid.copy(), calculate_grid_score(
                grid, tech
            )  # Return original score

    current_score = calculate_grid_score(current_grid, tech)
    best_grid = current_grid.copy()
    best_score = current_score
    best_score_time = start_time # Initialize best_score_time

    temperature = initial_temperature
    swap_probability = initial_swap_probability

    # --- Progress Reporting Setup ---
    try:
        total_steps = math.log(stopping_temperature / initial_temperature) / math.log(
            cooling_rate
        )
    except (ValueError, ZeroDivisionError):
        total_steps = 0  # Avoid division by zero if cooling_rate is 1 or invalid
    step = 0
    # --- End Progress Reporting Setup ---

    # --- Annealing Loop ---
    while temperature > stopping_temperature:
        step += 1
        if time.time() - start_time > max_processing_time:
            logging.info(
                f"SA: Max processing time ({max_processing_time}s) exceeded. Returning best found."
            )
            break

        swap_probability = get_swap_probability(
            temperature,
            initial_temperature,
            stopping_temperature,
            initial_swap_probability,
            final_swap_probability,
        )

        for _ in range(iterations_per_temp):
            # Ensure modules_to_consider reflects the modules actually on the grid now
            current_modules_on_grid_defs = [
                m
                for m in tech_modules
                if m["id"]
                in {
                    current_grid.get_cell(x, y)["module"]
                    for y in range(current_grid.height)
                    for x in range(current_grid.width)
                    if current_grid.get_cell(x, y)["tech"] == tech
                }
            ]

            if (
                not current_modules_on_grid_defs
            ):  # Safety check if grid somehow became empty
                continue

            modified_cells_info = [] # Initialize for each iteration

            if random.random() < swap_probability:
                pos1, original_cell_1_data, pos2, original_cell_2_data = swap_modules(current_grid, tech, current_modules_on_grid_defs)
                if pos1 is None: # Not enough modules to swap
                    continue
                modified_cells_info = [(pos1, original_cell_1_data), (pos2, original_cell_2_data)]
            else:
                pos_from, original_from_cell_data, pos_to, original_to_cell_data = move_module(current_grid, tech, current_modules_on_grid_defs)
                if pos_from is None: # No modules to move or no empty slots
                    continue
                modified_cells_info = [(pos_from, original_from_cell_data), (pos_to, original_to_cell_data)]

            neighbor_score = calculate_grid_score(current_grid, tech)

            delta_e = neighbor_score - current_score
            if delta_e > 0 or random.random() < math.exp(delta_e / temperature):
                current_score = neighbor_score
                if current_score > best_score:
                    best_grid = current_grid.copy()
                    best_score_time = time.time() # Update time when new best score is found
                    elapsed_time_at_best_score = best_score_time - start_time
                    logging.info(
                        f"SA: New best score for {tech}: {current_score:.4f} (Temp: {temperature:.2f}, Time: {elapsed_time_at_best_score:.2f}s)"
                    )
                    best_score = current_score
                    if progress_callback:
                        progress_data = {
                            "tech": tech,
                            "run_id": run_id,
                            "stage": stage,
                            "progress_percent": progress_offset
                            + (
                                (step / total_steps) * progress_scale
                                if total_steps > 0
                                else 0
                            ),
                            "current_temp": temperature,
                            "best_score": best_score,
                            "status": "new_best",
                        }
                        if send_grid_updates:
                            # Reconstitute the full grid from the localized best_grid
                            # Assuming start_x, start_y, localized_width, localized_height are available
                            # from the context where simulated_annealing is called.
                            # For now, we'll use a placeholder for these values.
                            # In a real scenario, these would need to be passed into simulated_annealing.
                            reconstituted_full_grid = full_grid.copy()
                            # Clear the tech modules from the full grid before applying localized changes
                            clear_all_modules_of_tech(reconstituted_full_grid, tech)
                            apply_localized_grid_changes(
                                reconstituted_full_grid,
                                best_grid,
                                tech,
                                # These values need to be passed from the calling function (optimize_placement)
                                # For now, using dummy values. This will cause an error if not handled.
                                start_x,  # Placeholder for start_x
                                start_y,  # Placeholder for start_y
                            )
                            progress_data["best_grid"] = (
                                reconstituted_full_grid.to_dict()
                            )
                        progress_callback(progress_data)
                        gevent.sleep(0)
            else:
                # Revert changes if the new state is not accepted
                for (x, y), original_data in modified_cells_info:
                    current_grid.cells[y][x].update(original_data)

        temperature *= cooling_rate
        if progress_callback and (step % 5 == 0 or temperature <= stopping_temperature):
            progress_data = {
                "tech": tech,
                "run_id": run_id,
                "stage": stage,
                "progress_percent": progress_offset
                + (
                    max(
                        ((time.time() - start_time) / max_processing_time)
                        if max_processing_time > 0
                        else 0,
                        ((step / total_steps) if total_steps > 0 else 0),
                    )
                    * progress_scale
                ),
                "current_temp": temperature,
                "best_score": best_score,
                "status": "in_progress",
            }

            progress_callback(progress_data)
            gevent.sleep(0)
    # --- End Annealing Loop ---

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_to_best_score = best_score_time - start_time
    logging.info(
        f"SA: Finished ({'Polish' if start_from_current_grid else 'Full'}). Best score: {best_score:.4f}. Time: {elapsed_time:.4f}s. Time to best score: {elapsed_time_to_best_score:.4f}s"
    )

    # Final check for validity (especially important if polishing)
    final_modules_placed = check_all_modules_placed(
        best_grid, modules, ship, tech, player_owned_rewards, tech_modules=modules_to_consider, solve_type=solve_type
    )
    if not final_modules_placed:
        logging.warning(
            f"SA: Final grid for {tech} did not contain all expected modules. This might indicate an issue."
        )
        # Decide how to handle this - return best_grid anyway, or revert?
        # For now, let's return the best_grid found, but log the warning.

    # Check for zero score if modules should provide bonus
    if best_score < 1e-9 and any(
        m["bonus"] > 0 or m["type"] == "core" for m in modules_to_consider
    ):
        logging.warning(
            f"SA: solver resulted in zero score for {ship}/{tech} despite modules existing. Potential failure."
        )
        # If polishing failed badly, return the original grid state
        if start_from_current_grid:
            initial_score = calculate_grid_score(
                grid, tech
            )  # Score of the grid passed in
            logging.info(
                f"SA: Returning original grid state (score {initial_score:.4f}) due to zero score after SA polish."
            )
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
    progress = (initial_temperature - temperature) / (
        initial_temperature - stopping_temperature
    )
    return (
        initial_swap_probability
        - (initial_swap_probability - final_swap_probability) * progress
    )


def swap_modules(grid, tech, tech_modules_on_grid):
    """
    Swaps the positions of two randomly selected modules of the specified tech
    that are currently on the grid. Returns the original state of the swapped cells.
    """
    module_positions = []
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            # Ensure the module is of the target tech AND is one we are considering
            if cell["tech"] == tech and cell["module"] in {
                m["id"] for m in tech_modules_on_grid
            }:
                module_positions.append((x, y))

    if len(module_positions) < 2:
        return None, None, None, None  # Not enough modules to swap

    pos1, pos2 = random.sample(module_positions, 2)
    x1, y1 = pos1
    x2, y2 = pos2

    # Get the full module data from each cell before swapping
    original_cell_1_data = grid.get_cell(x1, y1).copy()
    original_cell_2_data = grid.get_cell(x2, y2).copy()

    module_data_1 = original_cell_1_data.copy()
    module_data_2 = original_cell_2_data.copy()

    # Place module 2 data into cell 1
    place_module(
        grid,
        x1,
        y1,
        module_data_2["module"],
        module_data_2["label"],
        module_data_2["tech"],
        module_data_2["type"],
        module_data_2["bonus"],
        module_data_2["adjacency"],
        module_data_2["sc_eligible"],
        module_data_2["image"],
    )
    # Update module_position after placement
    grid.cells[y1][x1]["module_position"] = (x1, y1)

    # Place module 1 data into cell 2
    place_module(
        grid,
        x2,
        y2,
        module_data_1["module"],
        module_data_1["label"],
        module_data_1["tech"],
        module_data_1["type"],
        module_data_1["bonus"],
        module_data_1["adjacency"],
        module_data_1["sc_eligible"],
        module_data_1["image"],
    )
    # Update module_position after placement
    grid.cells[y2][x2]["module_position"] = (x2, y2)

    return (x1, y1), original_cell_1_data, (x2, y2), original_cell_2_data


def move_module(grid, tech, tech_modules_on_grid):
    """
    Moves a randomly selected module of the specified tech (that's on the grid)
    to a random empty active slot. Returns the original state of the modified cells.
    """
    module_positions = []
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech and cell["module"] in {
                m["id"] for m in tech_modules_on_grid
            }:
                module_positions.append((x, y))

    if not module_positions:
        return None, None, None, None  # No modules to move

    x_from, y_from = random.choice(module_positions)
    original_from_cell_data = grid.get_cell(x_from, y_from).copy()
    module_data_to_move = original_from_cell_data.copy()

    empty_active_positions = [
        (ex, ey)
        for ey in range(grid.height)
        for ex in range(grid.width)
        if grid.get_cell(ex, ey)["module"] is None and grid.get_cell(ex, ey)["active"]
    ]

    if not empty_active_positions:
        return None, None, None, None  # No empty slots to move to

    # --- Simple Random Move (Original Logic) ---
    x_to, y_to = random.choice(empty_active_positions)
    original_to_cell_data = grid.get_cell(x_to, y_to).copy()

    # Place the module in the new empty slot
    place_module(
        grid,
        x_to,
        y_to,
        module_data_to_move["module"],
        module_data_to_move["label"],
        module_data_to_move["tech"],
        module_data_to_move["type"],
        module_data_to_move["bonus"],
        module_data_to_move["adjacency"],
        module_data_to_move["sc_eligible"],
        module_data_to_move["image"],
    )
    # Update module_position after placement
    grid.cells[y_to][x_to]["module_position"] = (x_to, y_to)

    # Clear the original position (preserving active/supercharged status)
    grid.cells[y_from][x_from]["module"] = None
    grid.cells[y_from][x_from]["label"] = ""
    grid.cells[y_from][x_from]["tech"] = None
    grid.cells[y_from][x_from]["type"] = ""
    grid.cells[y_from][x_from]["bonus"] = 0.0
    grid.cells[y_from][x_from]["adjacency"] = (
        False  # Adjacency type is part of module def
    )
    grid.cells[y_from][x_from]["sc_eligible"] = False
    grid.cells[y_from][x_from]["image"] = None
    grid.cells[y_from][x_from]["module_position"] = None
    grid.cells[y_from][x_from]["total"] = 0.0
    grid.cells[y_from][x_from]["adjacency_bonus"] = 0.0

    return (x_from, y_from), original_from_cell_data, (x_to, y_to), original_to_cell_data
