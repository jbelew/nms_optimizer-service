# optimization_algorithms.py
from grid_utils import Grid
from modules_data import get_tech_modules
from grid_display import print_grid_compact, print_grid
from bonus_calculations import calculate_grid_score
from modules_data import get_tech_modules_for_training
from module_placement import (
    place_module,
    clear_all_modules_of_tech,
)  # Import from module_placement
from simulated_annealing import simulated_annealing

# from ml_placement import ml_placement # Keep commented unless ML is fully integrated
from itertools import permutations
import random
import math
import multiprocessing
import time
from copy import deepcopy
from modules import (
    solves,
)
from solve_map_utils import filter_solves  # Import the new function

# --- Constants ---
REFINEMENT_WINDOW_WIDTH = 4
REFINEMENT_WINDOW_HEIGHT = 3

# --- Helper Functions for optimize_placement ---


def _initial_placement_no_solve(grid, ship, modules, tech, player_owned_rewards):
    """Handles initial placement when no solve map is available."""
    print(f"INFO -- No solve found for ship: '{ship}' -- tech: '{tech}'. Placing modules in empty slots.")
    solved_grid = place_all_modules_in_empty_slots(grid, modules, ship, tech, player_owned_rewards)
    solved_bonus = calculate_grid_score(solved_grid, tech)
    solve_score = 0
    percentage = 100.0 if solved_bonus > 1e-9 else 0.0
    print(
        f"SUCCESS -- Final Score (No Solve Map): {solved_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) for ship: '{ship}' -- tech: '{tech}'"
    )
    print_grid_compact(solved_grid)
    # Return None for grid as this path terminates early in the main function
    return None, percentage, solved_bonus


def _initial_placement_with_solve(grid, ship, modules, tech, player_owned_rewards, filtered_solves):
    """
    Handles initial placement using pattern matching from solve maps,
    with SA fallback if no pattern fits.
    """
    solve_data = filtered_solves[ship][tech]
    original_pattern = solve_data["map"]
    solve_score = solve_data["score"]  # Official solve score

    print(f"INFO -- Found solve map for {ship}/{tech}. Score: {solve_score:.4f}. Attempting pattern matching.")
    patterns_to_try = get_all_unique_pattern_variations(original_pattern)
    grid_dict = grid.to_dict()  # Store original grid state

    best_pattern_grid = None
    highest_pattern_bonus = -float("inf")
    best_pattern_adjacency_score = -float("inf")  # Use float for consistency

    # Try applying all pattern variations
    for pattern in patterns_to_try:
        x_coords = [coord[0] for coord in pattern.keys()]
        y_coords = [coord[1] for coord in pattern.keys()]
        if not x_coords or not y_coords:
            continue
        pattern_width = max(x_coords) + 1
        pattern_height = max(y_coords) + 1

        # Check if pattern dimensions exceed grid dimensions
        if pattern_width > grid.width or pattern_height > grid.height:
            continue

        for start_x in range(grid.width - pattern_width + 1):
            for start_y in range(grid.height - pattern_height + 1):
                # Use a fresh copy of the original grid for each pattern attempt
                temp_grid_pattern = Grid.from_dict(grid_dict)
                temp_result_grid, adjacency_score = apply_pattern_to_grid(
                    temp_grid_pattern, pattern, modules, tech, start_x, start_y, ship, player_owned_rewards
                )
                if temp_result_grid is not None:
                    current_pattern_bonus = calculate_grid_score(temp_result_grid, tech)
                    # Track the best pattern found so far
                    if current_pattern_bonus > highest_pattern_bonus:
                        highest_pattern_bonus = current_pattern_bonus
                        best_pattern_grid = temp_result_grid.copy()
                        best_pattern_adjacency_score = adjacency_score
                    elif (
                        current_pattern_bonus == highest_pattern_bonus
                        and adjacency_score > best_pattern_adjacency_score  # Prefer better adjacency on tie
                    ):
                        best_pattern_grid = temp_result_grid.copy()
                        best_pattern_adjacency_score = adjacency_score

    # Check if any pattern was successfully applied
    if best_pattern_grid is not None:
        solved_grid = best_pattern_grid
        solved_bonus = highest_pattern_bonus
        print(
            f"INFO -- Best pattern score: {solved_bonus:.4f} (Adjacency: {best_pattern_adjacency_score:.2f}) for ship: '{ship}' -- tech: '{tech}' that fits."
        )
        print_grid_compact(solved_grid)
        return solved_grid, solved_bonus, solve_score, True, False  # pattern_applied=True, sa_was_initial=False
    else:
        # --- Case: Solve Map Exists, but No Pattern Fits ---
        print(
            f"WARNING -- Solve map exists for {ship}/{tech}, but no pattern variation fits the grid. Falling back to initial Simulated Annealing."
        )
        initial_sa_grid = grid.copy()
        clear_all_modules_of_tech(initial_sa_grid, tech)
        # Use slightly faster SA params for initial fallback
        solved_grid, solved_bonus = simulated_annealing(
            initial_sa_grid,
            ship,
            modules,
            tech,
            player_owned_rewards,
            initial_temperature=3500,
            cooling_rate=0.97,
            iterations_per_temp=25,
            initial_swap_probability=0.45,
            final_swap_probability=0.35,
            max_processing_time=10.0,  # Limit initial SA time
        )
        if solved_grid is None:
            raise ValueError(f"Fallback simulated_annealing failed for {ship}/{tech} when no pattern fit.")
        print(f"INFO -- Fallback SA score (no pattern fit): {solved_bonus:.4f}")
        return solved_grid, solved_bonus, solve_score, False, True  # pattern_applied=False, sa_was_initial=True


def _perform_opportunity_refinement(
    grid_to_refine, current_best_score, opportunity, ship, modules, tech, player_owned_rewards, experimental
):
    """Handles the refinement process within a found opportunity window."""
    opportunity_x, opportunity_y = opportunity
    refined_grid = None
    refined_score = -1.0
    sa_was_ml_fallback = False

    if experimental:
        print("INFO -- Experimental flag is True. Attempting ML refinement first.")
        # --- Try ML Refinement ---
        # Pass a copy to avoid modifying grid_to_refine if ML fails
        refined_grid, refined_score = _handle_ml_opportunity(
            grid_to_refine.copy(), modules, ship, tech, player_owned_rewards, opportunity_x, opportunity_y
        )

        # --- Fallback to SA/Refine if ML failed ---
        if refined_grid is None:
            print("INFO -- ML refinement failed or model not found. Falling back to SA/Refine refinement.")
            sa_was_ml_fallback = True
            # Pass a fresh copy of the original grid_to_refine
            refined_grid, refined_score = _handle_sa_refine_opportunity(
                grid_to_refine.copy(), modules, ship, tech, player_owned_rewards, opportunity_x, opportunity_y
            )
        # else: ML succeeded, proceed with its result.

    else:  # Not experimental
        print("INFO -- Experimental flag is False. Using SA/Refine refinement directly.")
        # Pass a copy to avoid modifying grid_to_refine if SA/Refine fails
        refined_grid, refined_score = _handle_sa_refine_opportunity(
            grid_to_refine.copy(), modules, ship, tech, player_owned_rewards, opportunity_x, opportunity_y
        )

    # --- Compare and Decide ---
    if refined_grid is not None and refined_score >= current_best_score:
        # --- Refinement Improved Score ---
        refinement_method = "ML/SA Fallback" if sa_was_ml_fallback else ("ML" if experimental else "SA/Refine")
        print(
            f"INFO -- Opportunity refinement (using {refinement_method}) improved score from {current_best_score:.4f} to {refined_score:.4f}"
        )
        return refined_grid, refined_score, False  # Return improved grid, score, sa_was_initial=False
    else:
        # --- Refinement Failed or Did Not Improve ---
        if refined_grid is not None:  # Refinement ran but didn't improve
            refinement_method = "ML/SA Fallback" if sa_was_ml_fallback else ("ML" if experimental else "SA/Refine")
            print(
                f"INFO -- Opportunity refinement (using {refinement_method}) did not improve score ({refined_score:.4f} vs {current_best_score:.4f})."
            )
        else:  # Refinement failed completely
            refinement_method = "ML/SA Fallback" if sa_was_ml_fallback else ("ML" if experimental else "SA/Refine")
            print(f"INFO -- Opportunity refinement (using {refinement_method}) failed completely.")

        # --- Final Fallback SA Logic (Only if experimental and NOT already an ML->SA fallback) ---
        if experimental and not sa_was_ml_fallback:
            print(
                "INFO -- Experimental flag is True AND refinement didn't improve/failed (and was not ML->SA fallback). Attempting final fallback Simulated Annealing."
            )
            # Run SA on the grid state *before* the failed/unimproved refinement attempt
            sa_fallback_grid, sa_fallback_score = _handle_sa_refine_opportunity(
                grid_to_refine.copy(), modules, ship, tech, player_owned_rewards, opportunity_x, opportunity_y
            )

            if sa_fallback_grid is not None and sa_fallback_score > current_best_score:
                print(
                    f"INFO -- Final fallback SA improved score from {current_best_score:.4f} to {sa_fallback_score:.4f}"
                )
                return sa_fallback_grid, sa_fallback_score, False  # Return fallback grid, score, sa_was_initial=False
            elif sa_fallback_grid is not None:
                print(
                    f"INFO -- Final fallback SA did not improve score ({sa_fallback_score:.4f} vs {current_best_score:.4f}). Keeping previous best."
                )
            else:
                print(f"ERROR -- Final fallback Simulated Annealing failed. Keeping previous best.")
        elif sa_was_ml_fallback:
            print("INFO -- Skipping final fallback SA because ML failed and its SA fallback didn't improve.")
        else:  # experimental is False
            print("INFO -- Experimental flag is False, keeping previous best grid without final fallback SA.")

        # If refinement failed or didn't improve, return the original grid and score
        return grid_to_refine, current_best_score, None  # Indicate no change, sa_was_initial status irrelevant here


def _perform_final_sa_check(
    solved_grid, solved_bonus, sa_was_initial_placement, ship, modules, tech, player_owned_rewards
):
    """Performs a final SA check if modules are not placed and initial placement wasn't SA."""
    all_modules_placed = check_all_modules_placed(solved_grid, modules, ship, tech, player_owned_rewards)

    if not all_modules_placed and not sa_was_initial_placement:
        print(f"WARNING! -- Not all modules placed AND initial placement wasn't SA. Running final SA.")
        grid_for_final_sa = solved_grid.copy()
        clear_all_modules_of_tech(grid_for_final_sa, tech)

        temp_solved_grid, _ = simulated_annealing(
            grid_for_final_sa,
            ship,
            modules,
            tech,
            player_owned_rewards,
            initial_temperature=4000,
            cooling_rate=0.98,
            iterations_per_temp=30,
            initial_swap_probability=0.40,
            final_swap_probability=0.3,
            max_processing_time=15.0,  # Limit final SA time
        )

        if temp_solved_grid is not None:
            final_sa_score = calculate_grid_score(temp_solved_grid, tech)
            if final_sa_score > solved_bonus:
                print(
                    f"INFO -- Final SA (due to unplaced modules) improved score from {solved_bonus:.4f} to {final_sa_score:.4f}"
                )
                return temp_solved_grid, final_sa_score  # Return updated grid and score
            else:
                print(
                    f"INFO -- Final SA (due to unplaced modules) did not improve score ({final_sa_score:.4f} vs {solved_bonus:.4f}). Keeping previous best."
                )
        else:
            print(
                f"ERROR -- Final simulated_annealing solver (due to unplaced modules) failed for ship: '{ship}' -- tech: '{tech}'. Returning previous best grid."
            )

    elif not all_modules_placed and sa_was_initial_placement:
        print(f"WARNING! -- Not all modules placed, but initial placement WAS SA. Skipping final SA check.")

    # Return original grid and score if no final SA was run or if it didn't improve
    return solved_grid, solved_bonus


# --- Main Optimization Function (Refactored) ---


def optimize_placement(grid, ship, modules, tech, player_owned_rewards=None, experimental=False):
    """
    Optimizes the placement of modules in a grid for a specific ship and technology.
    Uses pre-defined solve maps first, then refines based on supercharged opportunities.
    Includes an experimental path using ML placement with SA fallback.

    Args:
        grid (Grid): The initial grid state.
        ship (str): The ship type key.
        modules (dict): The main modules dictionary.
        tech (str): The technology key.
        player_owned_rewards (list, optional): List of reward module IDs owned. Defaults to None.
        experimental (bool): If True, attempts ML placement first during refinement,
                             with SA as a fallback. If False, uses SA/Refine directly.

    Returns:
        tuple: (best_grid, percentage, best_bonus)
               - best_grid (Grid): The optimized grid.
               - percentage (float): The percentage of the official solve score achieved.
               - best_bonus (float): The actual score achieved by the best grid.
    Raises:
        ValueError: If no empty, active slots are available or if critical steps fail.
    """
    print(f"INFO -- Attempting solve for ship: '{ship}' -- tech: '{tech}' -- Experimental: {experimental}")

    if player_owned_rewards is None:
        player_owned_rewards = []

    # --- 1. Early Check: Any Empty, Active Slots? ---
    if not any(
        grid.get_cell(x, y)["active"] and grid.get_cell(x, y)["module"] is None
        for y in range(grid.height)
        for x in range(grid.width)
    ):
        raise ValueError(f"No empty, active slots available on the grid for ship: '{ship}' -- tech: '{tech}'.")

    # --- 2. Filter Solves ---
    filtered_solves = filter_solves(solves, ship, modules, tech, player_owned_rewards)

    # --- 3. Initial Placement ---
    solved_grid = None
    solved_bonus = -float("inf")
    solve_score = 0.0
    pattern_applied = False
    sa_was_initial_placement = False

    if ship not in filtered_solves or tech not in filtered_solves.get(ship, {}):
        # --- 3a. No Solve Map Available ---
        # This path now returns early if no solve map exists
        return _initial_placement_no_solve(grid, ship, modules, tech, player_owned_rewards)
    else:
        # --- 3b. Solve Map Exists ---
        solved_grid, solved_bonus, solve_score, pattern_applied, sa_was_initial_placement = (
            _initial_placement_with_solve(grid, ship, modules, tech, player_owned_rewards, filtered_solves)
        )

    # --- 4. Opportunity Refinement ---
    grid_to_refine = solved_grid.copy()
    current_best_score = calculate_grid_score(grid_to_refine, tech)  # Score *before* refinement

    opportunity = find_supercharged_opportunities(grid_to_refine, modules, ship, tech)

    if opportunity:
        print(f"INFO -- Found opportunity for refinement at window starting: {opportunity}")
        # Pass sa_was_initial_placement status from initial placement phase
        refined_grid, refined_score, sa_status_after_refine = _perform_opportunity_refinement(
            grid_to_refine, current_best_score, opportunity, ship, modules, tech, player_owned_rewards, experimental
        )
        # Update main grid and score if refinement was successful
        solved_grid = refined_grid
        solved_bonus = refined_score
        # Update sa_was_initial_placement only if refinement changed the grid state
        if sa_status_after_refine is not None:
            sa_was_initial_placement = sa_status_after_refine
    else:
        print("INFO -- No supercharged opportunity found for refinement.")
        # solved_grid, solved_bonus, sa_was_initial_placement remain unchanged

    # --- 5. Final SA Check (if modules unplaced) ---
    solved_grid, solved_bonus = _perform_final_sa_check(
        solved_grid, solved_bonus, sa_was_initial_placement, ship, modules, tech, player_owned_rewards
    )

    # --- 6. Final Result Calculation ---
    best_grid = solved_grid
    best_bonus = calculate_grid_score(best_grid, tech)  # Recalculate final score

    if solve_score > 1e-9:
        percentage = (best_bonus / solve_score) * 100
    else:
        percentage = 100.0 if best_bonus > 1e-9 else 0.0

    print(
        f"SUCCESS -- Final Score: {best_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) for ship: '{ship}' -- tech: '{tech}'"
    )
    print_grid_compact(best_grid)

    return best_grid, percentage, best_bonus


def refine_placement(grid, ship, modules, tech, player_owned_rewards=None):
    # ... (Keep original implementation) ...
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
        # Return the grid as is if not enough space, score will be calculated later
        print(
            f"Warning: Not enough active slots ({len(available_positions)}) for modules ({len(tech_modules)}) for {tech}. Returning current grid state."
        )
        return grid.copy(), calculate_grid_score(grid, tech)  # Return copy and current score

    # Initialize the iteration counter
    iteration_count = 0
    best_grid_copy = None  # Store the best grid found

    # Generate all permutations of module placements
    # Limit permutations if too many for reasonable time
    num_modules = len(tech_modules)
    try:
        num_permutations = math.perm(len(available_positions), num_modules)
        max_permutations = 50000  # Adjust this limit as needed
        if num_permutations > max_permutations:
            print(f"Warning: Too many permutations ({num_permutations}) for refine_placement. Limiting iterations.")
            # Use random sampling instead of full permutations
            max_iterations = max_permutations
            is_sampling = True
        else:
            max_iterations = num_permutations
            is_sampling = False
            placement_iterator = permutations(available_positions, num_modules)
    except (ValueError, OverflowError):
        print(f"Warning: Could not calculate permutations for refine_placement. Using sampling.")
        max_iterations = 50000  # Fallback limit
        is_sampling = True

    if is_sampling:
        # Generate random samples of placements
        for _ in range(max_iterations):
            iteration_count += 1
            temp_grid = grid.copy()  # Use a fresh copy for each sample
            clear_all_modules_of_tech(temp_grid, tech)

            placement = random.sample(available_positions, num_modules)
            shuffled_tech_modules = random.sample(tech_modules, num_modules)  # Shuffle modules too

            placement_successful = True
            for index, (x, y) in enumerate(placement):
                module = shuffled_tech_modules[index]
                try:
                    place_module(
                        temp_grid,
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
                except IndexError:
                    placement_successful = False
                    break
            if not placement_successful:
                continue

            grid_bonus = calculate_grid_score(temp_grid, tech)
            if grid_bonus > highest_bonus:
                highest_bonus = grid_bonus
                best_grid_copy = temp_grid  # Keep the best grid found so far
    else:
        # Iterate through permutations
        for placement in placement_iterator:
            iteration_count += 1
            temp_grid = grid.copy()  # Use a fresh copy for each permutation
            clear_all_modules_of_tech(temp_grid, tech)

            # Place modules in the current permutation order
            placement_successful = True
            for index, (x, y) in enumerate(placement):
                # Use tech_modules directly as permutation handles position order
                module = tech_modules[index]
                try:
                    place_module(
                        temp_grid,
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
                except IndexError:
                    placement_successful = False
                    break
            if not placement_successful:
                continue

            grid_bonus = calculate_grid_score(temp_grid, tech)
            if grid_bonus > highest_bonus:
                highest_bonus = grid_bonus
                best_grid_copy = temp_grid  # Keep the best grid found so far

    print(f"INFO -- refine_placement completed {iteration_count} iterations for ship: '{ship}' -- tech: '{tech}'")

    # Return the best grid found, or the original if none was better/found
    optimal_grid = best_grid_copy if best_grid_copy is not None else grid.copy()
    # Ensure the returned score matches the returned grid
    final_score = calculate_grid_score(optimal_grid, tech)

    return optimal_grid, final_score


def _evaluate_permutation_worker(args):
    # ... (Keep original implementation) ...
    placement_indices, original_base_grid, tech_modules, available_positions, tech = args
    num_modules_to_place = len(tech_modules)
    working_grid = original_base_grid.copy()
    clear_all_modules_of_tech(working_grid, tech)
    try:
        placement_positions = [available_positions[i] for i in placement_indices]
    except IndexError:
        print(
            f"Error: Invalid placement index in worker. Indices: {placement_indices}, Available: {len(available_positions)}"
        )
        return (-1.0, None)
    placement_successful = True
    for index, (x, y) in enumerate(placement_positions):
        if index >= num_modules_to_place:
            placement_successful = False
            break
        module = tech_modules[index]
        try:
            place_module(
                working_grid,
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
        except IndexError:
            print(f"ERROR: IndexError during place_module at ({x},{y}) in worker. Skipping.")
            placement_successful = False
            break
        except Exception as e:
            print(f"ERROR: Exception during place_module at ({x},{y}) in worker: {e}. Skipping.")
            placement_successful = False
            break
    if not placement_successful:
        return (-1.0, None)
    grid_bonus = calculate_grid_score(working_grid, tech)
    return (grid_bonus, placement_indices)


def refine_placement_for_training(grid, ship, modules, tech, num_workers=None):
    # ... (Keep original implementation) ...
    start_time = time.time()
    optimal_grid = None
    highest_bonus = -1.0
    tech_modules = get_tech_modules_for_training(modules, ship, tech)
    if not tech_modules:
        print(f"Warning: No modules for {ship}/{tech}. Returning cleared grid.")
        cleared_grid = grid.copy()
        clear_all_modules_of_tech(cleared_grid, tech)
        return cleared_grid, 0.0
    num_modules_to_place = len(tech_modules)
    available_positions = [
        (x, y) for y in range(grid.height) for x in range(grid.width) if grid.get_cell(x, y)["active"]
    ]
    num_available = len(available_positions)
    if num_available < num_modules_to_place:
        print(
            f"Warning: Not enough active slots ({num_available}) for modules ({num_modules_to_place}) for {tech}. Returning cleared grid."
        )
        cleared_grid = grid.copy()
        clear_all_modules_of_tech(cleared_grid, tech)
        return cleared_grid, 0.0
    base_working_grid = grid.copy()
    clear_all_modules_of_tech(base_working_grid, tech)
    try:
        num_permutations = math.perm(num_available, num_modules_to_place)
        print(
            f"-- Training ({tech}): {num_available} slots, {num_modules_to_place} modules -> {num_permutations:,} permutations."
        )
    except (ValueError, OverflowError):
        print(
            f"-- Training ({tech}): {num_available} slots, {num_modules_to_place} modules -> Large number of permutations."
        )
        num_permutations = float("inf")
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    print(f"-- Using {num_workers} worker processes.")
    permutation_indices_iterator = permutations(range(num_available), num_modules_to_place)
    tasks = (
        (indices, base_working_grid, tech_modules, available_positions, tech)
        for indices in permutation_indices_iterator
    )
    best_placement_indices = None
    processed_count = 0
    chunksize = 1000
    if num_permutations != float("inf"):
        chunks_per_worker_target = 500
        calculated_chunksize = num_permutations // (num_workers * chunks_per_worker_target)
        chunksize = max(chunksize, calculated_chunksize)
        max_chunksize = 50000
        chunksize = min(chunksize, max_chunksize)
    print(f"-- Starting parallel evaluation with chunksize={chunksize}...")
    maxtasks = 2000
    print(f"-- Setting maxtasksperchild={maxtasks}")
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=maxtasks) as pool:
        results_iterator = pool.imap_unordered(_evaluate_permutation_worker, tasks, chunksize=chunksize)
        update_frequency = max(1, chunksize * num_workers // 4)
        for score, placement_indices in results_iterator:
            processed_count += 1
            if score == -1.0 and placement_indices is None:
                continue
            if placement_indices is not None and score > highest_bonus:
                highest_bonus = score
                best_placement_indices = placement_indices
            if processed_count % update_frequency == 0 or (
                num_permutations != float("inf") and processed_count == num_permutations
            ):
                elapsed = time.time() - start_time
                print(
                    f"\r-- Processed ~{processed_count // 1000}k permutations. Best: {highest_bonus:.4f} ({elapsed:.1f}s)",
                    end="",
                    flush=True,
                )
    print()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"-- Parallel evaluation finished in {total_time:.2f} seconds. Processed {processed_count} permutations.")
    if total_time > 0:
        perms_per_sec = processed_count / total_time
        print(f"-- Rate: {perms_per_sec:,.0f} permutations/sec")
    if best_placement_indices is not None:
        print(f"-- Reconstructing best grid with score: {highest_bonus:.4f}")
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)
        best_positions = [available_positions[i] for i in best_placement_indices]
        reconstruction_successful = True
        for index, (x, y) in enumerate(best_positions):
            if index >= len(tech_modules):
                reconstruction_successful = False
                break
            module = tech_modules[index]
            try:
                place_module(
                    optimal_grid,
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
            except Exception as e:
                print(f"ERROR during final grid reconstruction at ({x},{y}): {e}")
                reconstruction_successful = False
                break
        if not reconstruction_successful:
            print("Warning: Final grid reconstruction failed. Returning cleared grid.")
            optimal_grid = grid.copy()
            clear_all_modules_of_tech(optimal_grid, tech)
            highest_bonus = 0.0
        else:
            final_score = calculate_grid_score(optimal_grid, tech)
            if abs(final_score - highest_bonus) > 1e-6:
                print(
                    f"Warning: Final score ({final_score:.4f}) differs from tracked best ({highest_bonus:.4f}). Using final score."
                )
                highest_bonus = final_score
    elif num_modules_to_place > 0:
        print(f"Warning: No optimal grid found for {ship}/{tech}. Returning cleared grid.")
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)
        highest_bonus = 0.0
    else:
        print(f"-- No modules to place for {ship}/{tech}. Returning cleared grid.")
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)
        highest_bonus = 0.0
    return optimal_grid, highest_bonus


def rotate_pattern(pattern):
    # ... (Keep original implementation) ...
    x_coords = [coord[0] for coord in pattern.keys()]
    y_coords = [coord[1] for coord in pattern.keys()]
    if not x_coords or not y_coords:
        return {}
    max_x = max(x_coords)
    rotated_pattern = {}
    for (x, y), module_label in pattern.items():
        new_x = y
        new_y = max_x - x
        rotated_pattern[(new_x, new_y)] = module_label
    return rotated_pattern


def mirror_pattern_horizontally(pattern):
    # ... (Keep original implementation) ...
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
    # ... (Keep original implementation) ...
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
    # ... (Keep original implementation) ...
    new_grid = grid.copy()
    for pattern_x, pattern_y in pattern.keys():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y
        if 0 <= grid_x < new_grid.width and 0 <= grid_y < new_grid.height:
            if (
                new_grid.get_cell(grid_x, grid_y)["module"] is not None
                and new_grid.get_cell(grid_x, grid_y)["tech"] != tech
            ):
                return None, 0
    clear_all_modules_of_tech(new_grid, tech)
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None:
        print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
        return None, 0
    module_id_map = {module["id"]: module for module in tech_modules}
    for (pattern_x, pattern_y), module_id in pattern.items():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y
        if 0 <= grid_x < new_grid.width and 0 <= grid_y < new_grid.height:
            if module_id is None or module_id == "None":
                continue  # Allow "None" string from JSON
            if module_id in module_id_map:
                module_data = module_id_map[module_id]
                if new_grid.get_cell(grid_x, grid_y)["active"] and new_grid.get_cell(grid_x, grid_y)["module"] is None:
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
    adjacency_score = calculate_pattern_adjacency_score(new_grid, tech)
    return new_grid, adjacency_score


def get_all_unique_pattern_variations(original_pattern):
    # ... (Keep original implementation, ensure it handles string coords if needed) ...
    # Convert string keys like "(0, 0)" to tuples if necessary
    pattern_tuple_keys = {}
    for k, v in original_pattern.items():
        try:
            # Attempt to evaluate the string key as a tuple
            coord_tuple = eval(k)
            if isinstance(coord_tuple, tuple) and len(coord_tuple) == 2:
                pattern_tuple_keys[coord_tuple] = v
            else:
                print(f"Warning: Skipping invalid pattern key format: {k}")
        except:
            print(f"Warning: Skipping invalid pattern key format: {k}")

    if not pattern_tuple_keys:
        print("Warning: Original pattern resulted in no valid tuple keys.")
        return []

    unique_patterns = set()
    patterns_to_try = [pattern_tuple_keys]
    unique_patterns.add(tuple(sorted(pattern_tuple_keys.items())))

    current_patterns = [pattern_tuple_keys]
    for _ in range(3):  # Max 3 rotations needed
        next_rotation_batch = []
        for p in current_patterns:
            rotated = rotate_pattern(p)
            rotated_tuple = tuple(sorted(rotated.items()))
            if rotated_tuple not in unique_patterns:
                unique_patterns.add(rotated_tuple)
                patterns_to_try.append(rotated)
                next_rotation_batch.append(rotated)
        current_patterns = next_rotation_batch
        if not current_patterns:
            break  # Stop if no new rotations found

    # Add mirrored patterns
    mirrored_patterns_to_add = []
    for p in patterns_to_try:  # Iterate through all unique rotations found so far
        mirrored_h = mirror_pattern_horizontally(p)
        mirrored_h_tuple = tuple(sorted(mirrored_h.items()))
        if mirrored_h_tuple not in unique_patterns:
            unique_patterns.add(mirrored_h_tuple)
            mirrored_patterns_to_add.append(mirrored_h)

        mirrored_v = mirror_pattern_vertically(p)
        mirrored_v_tuple = tuple(sorted(mirrored_v.items()))
        if mirrored_v_tuple not in unique_patterns:
            unique_patterns.add(mirrored_v_tuple)
            mirrored_patterns_to_add.append(mirrored_v)

    patterns_to_try.extend(mirrored_patterns_to_add)
    print(f"Generated {len(patterns_to_try)} unique pattern variations.")
    return patterns_to_try


def count_adjacent_occupied(grid, x, y):
    # ... (Keep original implementation) ...
    count = 0
    adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    for nx, ny in adjacent_positions:
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            if grid.get_cell(nx, ny)["module"] is not None:
                count += 1
    return count


def calculate_pattern_adjacency_score(grid, tech):
    # ... (Keep original implementation) ...
    module_edge_weight = 3.0
    grid_edge_weight = 0.5
    total_adjacency_score = 0
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["module"] is not None and cell["tech"] == tech:
                if x == 0:
                    total_adjacency_score += grid_edge_weight
                if x == grid.width - 1:
                    total_adjacency_score += grid_edge_weight
                if y == 0:
                    total_adjacency_score += grid_edge_weight
                if y == grid.height - 1:
                    total_adjacency_score += grid_edge_weight
                adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                for adj_x, adj_y in adjacent_positions:
                    if 0 <= adj_x < grid.width and 0 <= adj_y < grid.height:
                        adjacent_cell = grid.get_cell(adj_x, adj_y)
                        if adjacent_cell["module"] is not None and adjacent_cell["tech"] != tech:
                            total_adjacency_score += module_edge_weight
    return total_adjacency_score


def _handle_ml_opportunity(grid, modules, ship, tech, player_owned_rewards, opportunity_x, opportunity_y):
    # ... (Keep original implementation, ensure ml_placement import is handled) ...
    try:
        from ml_placement import ml_placement  # Try importing here
    except ImportError:
        print("ERROR -- ml_placement module not found. Cannot use ML refinement.")
        return None, 0.0

    if player_owned_rewards is None:
        player_owned_rewards = []
    print(f"INFO -- Using ML for opportunity refinement at ({opportunity_x}, {opportunity_y})")
    localized_grid_ml, start_x, start_y, original_state_map = create_localized_grid_ml(
        grid, opportunity_x, opportunity_y, tech
    )
    ml_refined_grid, ml_refined_score_local = ml_placement(
        localized_grid_ml,
        ship,
        modules,
        tech,
        player_owned_rewards=player_owned_rewards,
        model_grid_width=REFINEMENT_WINDOW_WIDTH,
        model_grid_height=REFINEMENT_WINDOW_HEIGHT,
        polish_result=False,  # Turn off SA polishing within ML call during refinement
    )
    if ml_refined_grid is not None:
        print(f"INFO -- ML refinement produced a grid. Applying changes...")
        grid_copy = grid.copy()
        clear_all_modules_of_tech(grid_copy, tech)
        apply_localized_grid_changes(grid_copy, ml_refined_grid, tech, start_x, start_y)
        restore_original_state(grid_copy, original_state_map)
        new_score_global = calculate_grid_score(grid_copy, tech)
        print(f"INFO -- Score after ML refinement and restoration: {new_score_global:.4f}")
        return grid_copy, new_score_global
    else:
        print("INFO -- ML refinement failed or returned None. No changes applied.")
        grid_copy = grid.copy()
        restore_original_state(grid_copy, original_state_map)
        original_score = calculate_grid_score(grid_copy, tech)
        print(f"INFO -- Returning grid with original score after failed ML: {original_score:.4f}")
        return None, 0.0  # Return None grid to indicate failure


def _handle_sa_refine_opportunity(grid, modules, ship, tech, player_owned_rewards, opportunity_x, opportunity_y):
    # ... (Keep original implementation) ...
    print(f"INFO -- Using SA/Refine for opportunity refinement at ({opportunity_x}, {opportunity_y})")
    # Create a localized grid (preserves other tech modules)
    localized_grid, start_x, start_y = create_localized_grid(grid, opportunity_x, opportunity_y, tech)
    # Get the number of modules for the given tech
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    num_modules = len(tech_modules) if tech_modules else 0

    # Refine the localized grid
    temp_refined_grid = None
    temp_refined_bonus_local = -1.0
    if num_modules < 6:
        print(f"INFO -- {tech} has less than 6 modules, running refine_placement")
        temp_refined_grid, temp_refined_bonus_local = refine_placement(
            localized_grid, ship, modules, tech, player_owned_rewards
        )
    else:
        print(f"INFO -- {tech} has 6 or more modules, running simulated_annealing")
        # Use faster SA params for refinement
        temp_refined_grid, temp_refined_bonus_local = simulated_annealing(
            localized_grid,
            ship,
            modules,
            tech,
            player_owned_rewards,
            initial_temperature=1500,
            cooling_rate=0.96,
            stopping_temperature=1.5,
            iterations_per_temp=25,
            initial_swap_probability=0.40,
            final_swap_probability=0.25,
            start_from_current_grid=True,  # Start SA from the localized state
            max_processing_time=5.0,  # Limit refinement SA time
        )

    # Process SA/Refine result
    if temp_refined_grid is not None:
        # Recalculate local score
        temp_refined_bonus_local = calculate_grid_score(temp_refined_grid, tech)
        # Apply changes back to the main grid copy (passed as 'grid' argument)
        apply_localized_grid_changes(grid, temp_refined_grid, tech, start_x, start_y)
        # Calculate the new score of the entire grid
        new_score_global = calculate_grid_score(grid, tech)
        print(f"INFO -- Score after SA/Refine refinement: {new_score_global:.4f}")
        return grid, new_score_global  # Return the modified grid and its new global score
    else:
        print("INFO -- SA/Refine refinement failed. No changes applied.")
        return grid, -1.0  # Indicate failure


def place_all_modules_in_empty_slots(grid, modules, ship, tech, player_owned_rewards=None):
    # ... (Keep original implementation) ...
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None:
        print(f"ERROR --  No modules found for ship: '{ship}' -- tech: '{tech}'")
        return grid
    module_index = 0
    for x in range(grid.width):
        for y in range(grid.height):
            if module_index >= len(tech_modules):
                return grid
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
                module_index += 1
    if module_index < len(tech_modules) and len(tech_modules) > 0:
        print(f"WARNING -- Not enough space to place all modules for ship: '{ship}' -- tech: '{tech}'")
    return grid


def count_empty_in_localized(localized_grid):
    # ... (Keep original implementation) ...
    count = 0
    for y in range(localized_grid.height):
        for x in range(localized_grid.width):
            if localized_grid.get_cell(x, y)["module"] is None:
                count += 1
    return count


def find_supercharged_opportunities(grid, modules, ship, tech):
    # ... (Keep original implementation) ...
    grid_copy = grid.copy()
    clear_all_modules_of_tech(grid_copy, tech)
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
        return None
    window_width = REFINEMENT_WINDOW_WIDTH
    window_height = REFINEMENT_WINDOW_HEIGHT
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
                continue
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
                continue
            window_score = calculate_window_score(window_grid, tech)
            if window_score > best_window_score:
                best_window_score = window_score
                best_window_start_x, best_window_start_y = start_x, start_y
    if best_window_start_x is not None and best_window_start_y is not None:
        return best_window_start_x, best_window_start_y
    else:
        return None


def calculate_window_score(window_grid, tech):
    # ... (Keep original implementation) ...
    supercharged_count = 0
    empty_count = 0
    edge_penalty = 0
    for y in range(window_grid.height):
        for x in range(window_grid.width):
            cell = window_grid.get_cell(x, y)
            if cell["active"]:
                if cell["supercharged"]:
                    if cell["module"] is None or cell["tech"] == tech:
                        supercharged_count += 1
                        if x == 0 or x == window_grid.width - 1:
                            edge_penalty += 1
                if cell["module"] is None:
                    empty_count += 1
    return (supercharged_count * 3) + (empty_count * 1) - (edge_penalty * 0.5)


def create_localized_grid(grid, opportunity_x, opportunity_y, tech):
    # ... (Keep original implementation) ...
    localized_width = REFINEMENT_WINDOW_WIDTH
    localized_height = REFINEMENT_WINDOW_HEIGHT
    start_x = opportunity_x
    start_y = opportunity_y
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(grid.width, start_x + localized_width)
    end_y = min(grid.height, start_y + localized_height)
    actual_localized_width = end_x - start_x
    actual_localized_height = end_y - start_y
    localized_grid = Grid(actual_localized_width, actual_localized_height)
    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            localized_x = x - start_x
            localized_y = y - start_y
            cell = grid.get_cell(x, y)
            localized_grid.cells[localized_y][localized_x]["active"] = cell["active"]
            localized_grid.cells[localized_y][localized_x]["supercharged"] = cell["supercharged"]
            if cell["module"] is not None:
                localized_grid.cells[localized_y][localized_x]["module"] = cell["module"]
                localized_grid.cells[localized_y][localized_x]["label"] = cell["label"]
                localized_grid.cells[localized_y][localized_x]["tech"] = cell["tech"]
                localized_grid.cells[localized_y][localized_x]["type"] = cell["type"]
                localized_grid.cells[localized_y][localized_x]["bonus"] = cell["bonus"]
                localized_grid.cells[localized_y][localized_x]["adjacency"] = cell["adjacency"]
                localized_grid.cells[localized_y][localized_x]["sc_eligible"] = cell["sc_eligible"]
                localized_grid.cells[localized_y][localized_x]["image"] = cell["image"]
                if "module_position" in cell:
                    localized_grid.cells[localized_y][localized_x]["module_position"] = cell["module_position"]
    return localized_grid, start_x, start_y


def create_localized_grid_ml(grid, opportunity_x, opportunity_y, tech):
    # ... (Keep original implementation) ...
    localized_width = REFINEMENT_WINDOW_WIDTH
    localized_height = REFINEMENT_WINDOW_HEIGHT
    start_x = opportunity_x
    start_y = opportunity_y
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(grid.width, start_x + localized_width)
    end_y = min(grid.height, start_y + localized_height)
    actual_localized_width = end_x - start_x
    actual_localized_height = end_y - start_y
    localized_grid = Grid(actual_localized_width, actual_localized_height)
    original_state_map = {}
    for y_main in range(start_y, end_y):
        for x_main in range(start_x, end_x):
            localized_x = x_main - start_x
            localized_y = y_main - start_y
            main_cell = grid.get_cell(x_main, y_main)
            local_cell = localized_grid.get_cell(localized_x, localized_y)
            local_cell["supercharged"] = main_cell["supercharged"]
            if main_cell["module"] is not None and main_cell["tech"] != tech:
                original_state_map[(x_main, y_main)] = deepcopy(main_cell)
                local_cell["module"] = None
                local_cell["label"] = ""
                local_cell["tech"] = None
                local_cell["type"] = ""
                local_cell["bonus"] = 0.0
                local_cell["adjacency"] = False
                local_cell["sc_eligible"] = False
                local_cell["image"] = None
                local_cell["total"] = 0.0
                local_cell["adjacency_bonus"] = 0.0
                if "module_position" in local_cell:
                    del local_cell["module_position"]
                local_cell["active"] = False
            elif not main_cell["active"]:
                local_cell["active"] = False
                local_cell["module"] = None
                local_cell["label"] = ""
                local_cell["tech"] = None
                local_cell["type"] = ""
                local_cell["bonus"] = 0.0
                local_cell["adjacency"] = False
                local_cell["sc_eligible"] = False
                local_cell["image"] = None
                local_cell["total"] = 0.0
                local_cell["adjacency_bonus"] = 0.0
                if "module_position" in local_cell:
                    del local_cell["module_position"]
            else:
                local_cell.update(deepcopy(main_cell))
                local_cell["active"] = True
    return localized_grid, start_x, start_y, original_state_map


def restore_original_state(grid, original_state_map):
    # ... (Keep original implementation) ...
    if not original_state_map:
        return
    print(f"INFO -- Restoring original state for {len(original_state_map)} cells.")
    for (x, y), original_cell_data in original_state_map.items():
        if 0 <= x < grid.width and 0 <= y < grid.height:
            grid.cells[y][x].update(deepcopy(original_cell_data))
        else:
            print(f"Warning -- Coordinate ({x},{y}) from original_state_map is out of bounds for the main grid.")


def apply_localized_grid_changes(grid, localized_grid, tech, start_x, start_y):
    # ... (Keep original implementation) ...
    localized_width = localized_grid.width
    localized_height = localized_grid.height
    for y in range(localized_height):
        for x in range(localized_width):
            main_x = start_x + x
            main_y = start_y + y
            if 0 <= main_x < grid.width and 0 <= main_y < grid.height:
                # Check if the cell in the main grid is either empty or belongs to the target tech
                main_cell = grid.get_cell(main_x, main_y)
                if main_cell["module"] is None or main_cell["tech"] == tech:
                    # Update the main grid cell with data from the localized grid cell
                    # Use deepcopy to avoid potential shared references if localized_grid.cells contains complex objects
                    grid.cells[main_y][main_x].update(deepcopy(localized_grid.cells[y][x]))


def check_all_modules_placed(grid, modules, ship, tech, player_owned_rewards=None):
    # ... (Keep original implementation) ...
    if player_owned_rewards is None:
        player_owned_rewards = []
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if not tech_modules:
        return True  # No modules to place means all are placed
    placed_module_ids = set()
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech and cell["module"]:
                placed_module_ids.add(cell["module"])
    all_module_ids = {module["id"] for module in tech_modules}
    return placed_module_ids == all_module_ids
