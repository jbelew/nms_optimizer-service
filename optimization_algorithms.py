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

# from ml_placement import ml_placement
from itertools import permutations
import random
import math
import multiprocessing
import time
from copy import deepcopy
from modules import (
    solves,
)
from solve_map_utils import filter_solves  # Import the new functio


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
            # print_grid(optimal_grid)

    # Print the total number of iterations
    print(f"INFO -- refine_placement completed {iteration_count} iterations for ship: '{ship}' -- tech: '{tech}'")

    return optimal_grid, highest_bonus


# --- Worker Function (Define at top level) ---
def _evaluate_permutation_worker(args):
    """
    Worker function to evaluate a single permutation.
    Takes a tuple of arguments to be easily used with pool.map.
    Creates its own grid copy internally to avoid modifying the shared base.
    """
    # 1. Unpack arguments
    placement_indices, original_base_grid, tech_modules, available_positions, tech = args
    num_modules_to_place = len(tech_modules)

    # 2. Create a local copy *inside* the worker for this specific permutation
    # This is crucial: the grid received (original_base_grid) is pickled by multiprocessing,
    # but we need a distinct copy for *each* permutation evaluation within this worker.
    working_grid = original_base_grid.copy()  # Perform the copy here

    # 3. Clear the tech modules from the *local copy*
    clear_all_modules_of_tech(working_grid, tech)

    # 4. Determine placement positions
    try:
        placement_positions = [available_positions[i] for i in placement_indices]
    except IndexError:
        # Log error or handle appropriately
        print(
            f"Error: Invalid placement index in worker. Indices: {placement_indices}, Available: {len(available_positions)}"
        )
        return (-1.0, None)  # Indicate error

    # 5. Place modules onto the local working_grid
    placement_successful = True
    for index, (x, y) in enumerate(placement_positions):
        if index >= num_modules_to_place:  # Safety check
            placement_successful = False
            break
        module = tech_modules[index]
        try:
            place_module(
                working_grid,  # Use the local copy
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
        except Exception as e:  # Catch other potential errors
            print(f"ERROR: Exception during place_module at ({x},{y}) in worker: {e}. Skipping.")
            placement_successful = False
            break

    if not placement_successful:
        return (-1.0, None)  # Indicate error or skip

    # 6. Calculate score on the local working_grid
    grid_bonus = calculate_grid_score(working_grid, tech)

    # 7. Return score and the indices
    return (grid_bonus, placement_indices)


# --- Modified refine_placement_for_training ---
def refine_placement_for_training(grid, ship, modules, tech, num_workers=None):
    """
    Optimizes module placement using brute-force permutations with multiprocessing,
    intended for generating optimal ground truth for training data.
    Optimized copying strategy and added memory management safeguards.
    """
    start_time = time.time()
    optimal_grid = None
    highest_bonus = -1.0  # Use -1 to clearly distinguish from a valid 0 score

    tech_modules = get_tech_modules_for_training(modules, ship, tech)

    # --- Initial Checks (same as before) ---
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
    # --- End Initial Checks ---

    # --- Base Grid Setup ---
    # Create ONE base working grid here. It will be pickled and sent to workers.
    # Workers will create their own copies from this base.
    base_working_grid = grid.copy()
    clear_all_modules_of_tech(base_working_grid, tech)
    # --- End Base Grid Setup ---

    # --- Permutation Info & Worker Setup ---
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
        # Optional: Limit workers if permutations are truly astronomical, though fixing copying might be enough
        # if num_permutations > 100_000_000 and num_workers > 4:
        #     print(f"-- Limiting workers from {num_workers} to 4 due to extreme permutation count.")
        #     num_workers = 4
        print(f"-- Using {num_workers} worker processes.")
    # --- End Worker Setup ---

    # --- Task Preparation ---
    # Generate permutations of *indices* into available_positions
    permutation_indices_iterator = permutations(range(num_available), num_modules_to_place)

    # Package arguments: Pass the single base_working_grid. It gets pickled by the pool mechanism.
    tasks = (
        (indices, base_working_grid, tech_modules, available_positions, tech)
        for indices in permutation_indices_iterator
    )
    # --- End Task Preparation ---

    best_placement_indices = None
    processed_count = 0

    # --- Chunksize Calculation (heuristic) ---
    chunksize = 1000  # Minimum chunksize
    if num_permutations != float("inf"):
        # Aim for a moderate number of chunks per worker to balance overhead and load balancing
        chunks_per_worker_target = 500  # Tune this value
        calculated_chunksize = num_permutations // (num_workers * chunks_per_worker_target)
        chunksize = max(chunksize, calculated_chunksize)
        # Add an upper limit to prevent huge chunks consuming too much memory at once
        max_chunksize = 50000  # Tune this based on memory observations
        chunksize = min(chunksize, max_chunksize)
    # --- End Chunksize Calculation ---

    # --- Multiprocessing Pool Execution ---
    print(f"-- Starting parallel evaluation with chunksize={chunksize}...")
    # Add maxtasksperchild: Restarts worker after N tasks to help free memory
    maxtasks = 2000  # Tune this value (e.g., 1000-10000)
    print(f"-- Setting maxtasksperchild={maxtasks}")
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=maxtasks) as pool:
        # imap_unordered is good for performance when order doesn't matter
        results_iterator = pool.imap_unordered(_evaluate_permutation_worker, tasks, chunksize=chunksize)

        # Progress reporting frequency
        update_frequency = max(1, chunksize * num_workers // 4)  # Update reasonably often

        for score, placement_indices in results_iterator:
            processed_count += 1
            # Check for worker errors
            if score == -1.0 and placement_indices is None:
                continue

            # Track best score
            if placement_indices is not None and score > highest_bonus:
                highest_bonus = score
                best_placement_indices = placement_indices

            # Print progress periodically
            if processed_count % update_frequency == 0 or (
                num_permutations != float("inf") and processed_count == num_permutations
            ):
                elapsed = time.time() - start_time
                progress_percent = (processed_count / num_permutations * 100) if num_permutations != float("inf") else 0
                # Use \r and flush=True for inline updating
                print(
                    f"\r-- Processed ~{processed_count // 1000}k permutations. Best: {highest_bonus:.4f} ({elapsed:.1f}s)",
                    end="",
                    flush=True,
                )

    print()  # Ensure the next print starts on a new line after progress updates

    end_time = time.time()
    total_time = end_time - start_time
    print(f"-- Parallel evaluation finished in {total_time:.2f} seconds. Processed {processed_count} permutations.")
    if total_time > 0:
        perms_per_sec = processed_count / total_time
        print(f"-- Rate: {perms_per_sec:,.0f} permutations/sec")
    # --- End Pool Execution ---

    # --- Reconstruct Best Grid ---
    if best_placement_indices is not None:
        print(f"-- Reconstructing best grid with score: {highest_bonus:.4f}")
        # Start from the original grid structure again for safety
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)  # Clear only the target tech
        best_positions = [available_positions[i] for i in best_placement_indices]
        reconstruction_successful = True
        for index, (x, y) in enumerate(best_positions):
            # ... (rest of reconstruction logic is the same as your previous version) ...
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
            # Optional score verification
            final_score = calculate_grid_score(optimal_grid, tech)
            if abs(final_score - highest_bonus) > 1e-6:
                print(
                    f"Warning: Final score ({final_score:.4f}) differs from tracked best ({highest_bonus:.4f}). Using final score."
                )
                highest_bonus = final_score

    # --- Handle No Valid Placement Found ---
    elif num_modules_to_place > 0:  # Check if modules existed but no solution found
        print(f"Warning: No optimal grid found for {ship}/{tech}. Returning cleared grid.")
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)
        highest_bonus = 0.0  # Score is 0 for a cleared grid
    else:  # No modules to place initially
        print(f"-- No modules to place for {ship}/{tech}. Returning cleared grid.")
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)
        highest_bonus = 0.0
    # --- End Handling ---

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
    module_edge_weight = 3.0  # Weight for adjacency to other modules
    grid_edge_weight = 0.5  # Weight for adjacency to grid edges

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


# --- Helper function for ML opportunity refinement ---
def _handle_ml_opportunity(grid, modules, ship, tech, player_owned_rewards, opportunity_x, opportunity_y):
    """Handles the ML-based refinement within an opportunity window."""
    from ml_placement import ml_placement

    if player_owned_rewards is None:
        player_owned_rewards = []

    print(f"INFO -- Using ML for opportunity refinement at ({opportunity_x}, {opportunity_y})")
    # 1. Create localized grid specifically for ML (handles other techs)
    #    Pass the ORIGINAL grid here to gather localization info.
    localized_grid_ml, start_x, start_y, original_state_map = create_localized_grid_ml(
        grid, opportunity_x, opportunity_y, tech
    )

    # 2. Run ML placement on the localized grid
    #    Make sure player_owned_rewards is passed correctly!
    ml_refined_grid, ml_refined_score_local = ml_placement(
        localized_grid_ml,
        ship,
        modules,
        tech,
        player_owned_rewards=player_owned_rewards,  # <<< Pass player_owned_rewards
        # Pass model_dir, model_grid_width, model_grid_height if needed
    )

    # 3. Process ML result
    if ml_refined_grid is not None:
        print(f"INFO -- ML refinement produced a grid. Applying changes...")

        # --- *** Modification Start *** ---
        # Create the working copy NOW, *after* localization info is gathered
        # This copy will be modified and potentially returned.
        grid_copy = grid.copy()

        # Clear ALL modules of the target tech from the *entire* grid copy
        # This ensures we don't keep modules outside the refinement window.
        # print(f"DEBUG -- Clearing all '{tech}' modules from the full grid copy before applying ML changes.")
        clear_all_modules_of_tech(grid_copy, tech)
        # --- *** Modification End *** ---

        # Apply ML changes (from the small ml_refined_grid) to the cleared grid copy
        # This places the modules found by ML into the correct positions within the window
        # on the otherwise empty (for this tech) grid_copy.
        apply_localized_grid_changes(grid_copy, ml_refined_grid, tech, start_x, start_y)

        # Restore the original state of cells (other techs) that were temporarily removed during localization
        # Apply restoration to the grid_copy.
        restore_original_state(grid_copy, original_state_map)

        # Recalculate the score of the *entire* modified grid copy
        new_score_global = calculate_grid_score(grid_copy, tech)
        print(f"INFO -- Score after ML refinement and restoration: {new_score_global:.4f}")
        #print_grid_compact(grid_copy)
        return grid_copy, new_score_global  # Return the modified grid copy and its new global score
    else:
        # Handle ML failure
        print("INFO -- ML refinement failed or returned None. No changes applied.")
        # If ML failed, we still need to restore the original state onto a copy
        # to ensure consistency, especially if create_localized_grid_ml had side effects.
        grid_copy = grid.copy()  # Create a fresh copy from the original grid
        restore_original_state(grid_copy, original_state_map)
        # Return the restored grid and its score (which should be the original score)
        original_score = calculate_grid_score(grid_copy, tech)  # Recalculate score after restoration
        print(f"INFO -- Returning grid with original score after failed ML: {original_score:.4f}")
        # Return the grid_copy (which now matches the original state) and its score
        return None, 0.0


# --- Existing function for SA/Refine opportunity ---
def _handle_sa_refine_opportunity(grid, modules, ship, tech, player_owned_rewards, opportunity_x, opportunity_y):
    """Handles the SA/Refine-based refinement within an opportunity window."""
    print(f"INFO -- Using SA/Refine for opportunity refinement at ({opportunity_x}, {opportunity_y})")
    clear_all_modules_of_tech(grid, tech)
    # Create a localized grid (preserves other tech modules)
    localized_grid, start_x, start_y = create_localized_grid(grid, opportunity_x, opportunity_y, tech)
    # print_grid(localized_grid)

    # Get the number of modules for the given tech
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    num_modules = len(tech_modules) if tech_modules else 0

    # Refine the localized grid
    if num_modules < 6:
        print(f"INFO -- {tech} has less than 6 modules, running refine_placement")
        temp_refined_grid, temp_refined_bonus_local = refine_placement(
            localized_grid, ship, modules, tech, player_owned_rewards
        )
    else:
        print(f"INFO -- {tech} has 6 or more modules, running simulated_annealing")
        temp_refined_grid, temp_refined_bonus_local = simulated_annealing(
            localized_grid, ship, modules, tech, player_owned_rewards
        )

    # Process SA/Refine result
    if temp_refined_grid is not None:
        # Recalculate local score just to be sure
        temp_refined_bonus_local = calculate_grid_score(temp_refined_grid, tech)
        # print(f"INFO -- SA/Refine local score: {temp_refined_bonus_local:.4f}")
        # print_grid_compact(temp_refined_grid)

        # Apply changes back to the main grid copy (grid)
        apply_localized_grid_changes(grid, temp_refined_grid, tech, start_x, start_y)

        # Calculate the new score of the entire grid
        new_score_global = calculate_grid_score(grid, tech)
        print(f"INFO -- Score after SA/Refine refinement: {new_score_global:.4f}")
        return grid, new_score_global  # Return the modified grid and its new global score
    else:
        print("INFO -- SA/Refine refinement failed. No changes applied.")
        return grid, -1.0  # Indicate failure or no improvement


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
    print(
        f"INFO -- Attempting solve for ship: '{ship}' -- tech: '{tech}' -- Experimental: {experimental}"
    )

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

    # Initialize variables for tracking best results
    # best_grid = grid.copy() # Start with a copy of the input grid # Not needed, solved_grid tracks best
    # best_bonus = -float("inf") # Not needed, solved_bonus tracks best
    solved_grid = grid.copy() # Grid state after initial pattern/SA fallback
    solved_bonus = -float("inf")
    best_pattern_grid = grid.copy() # Best grid found using pattern matching
    highest_pattern_bonus = -float("inf")
    best_pattern_adjacency_score = 0
    solve_score = 0  # Official score from the solve map, if available
    pattern_applied = False # Flag if a pattern was successfully applied
    sa_was_initial_placement = False # Flag if SA was the direct initial placement method

    # Filter the solves dictionary based on player-owned rewards
    filtered_solves = filter_solves(solves, ship, modules, tech, player_owned_rewards)

    # --- Initial Placement Strategy ---
    # Check if a solve map exists for this ship/tech combination after filtering
    # --- Special Case: No Solve Available --- # <<< THIS BLOCK IS UNCHANGED >>>
    if ship not in filtered_solves or (ship in filtered_solves and tech not in filtered_solves[ship]):
        print(f"INFO -- No solve found for ship: '{ship}' -- tech: '{tech}'. Placing modules in empty slots.")
        solved_grid = place_all_modules_in_empty_slots(grid, modules, ship, tech, player_owned_rewards)
        solved_bonus = calculate_grid_score(solved_grid, tech)
        solve_score = 0
        percentage = 100.0 if solved_bonus > 1e-9 else 0.0
        print(
            f"SUCCESS -- Final Score (No Solve Map): {solved_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) for ship: '{ship}' -- tech: '{tech}'"
        )
        print_grid_compact(solved_grid)
        return solved_grid, percentage, solved_bonus # Adjusted return to match function signature

    # --- Case 2: Solve Map Exists ---
    else:
        solve_data = filtered_solves[ship][tech]
        original_pattern = solve_data["map"]
        solve_score = solve_data["score"] # Get the official solve score

        print(f"INFO -- Found solve map for {ship}/{tech}. Score: {solve_score:.4f}. Attempting pattern matching.")
        patterns_to_try = get_all_unique_pattern_variations(original_pattern)
        grid_dict = grid.to_dict() # Store original grid state

        # Try applying all pattern variations
        for pattern in patterns_to_try:
            x_coords = [coord[0] for coord in pattern.keys()]
            y_coords = [coord[1] for coord in pattern.keys()]
            if not x_coords or not y_coords: continue
            pattern_width = max(x_coords) + 1
            pattern_height = max(y_coords) + 1

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
                            and adjacency_score >= best_pattern_adjacency_score
                        ):
                            best_pattern_grid = temp_result_grid.copy()
                            best_pattern_adjacency_score = adjacency_score

        # Check if any pattern was successfully applied and resulted in a valid score
        if highest_pattern_bonus > -float("inf"):
            solved_grid = best_pattern_grid # Use the best pattern found
            solved_bonus = highest_pattern_bonus # Use its score
            print(
                f"INFO -- Best pattern score: {solved_bonus:.4f} (Adjacency: {best_pattern_adjacency_score:.2f}) for ship: '{ship}' -- tech: '{tech}' that fits."
            )
            print_grid_compact(solved_grid)
            pattern_applied = True
            sa_was_initial_placement = False # Pattern was applied, not SA
        else:
            # --- Case 2b: Solve Map Exists, but No Pattern Fits ---
            print(
                f"WARNING -- Solve map exists for {ship}/{tech}, but no pattern variation fits the grid. Falling back to initial Simulated Annealing."
            )
            # Clear the grid for the target tech before running SA
            initial_sa_grid = grid.copy()
            clear_all_modules_of_tech(initial_sa_grid, tech)
            # Run SA as the fallback placement method
            solved_grid, solved_bonus = simulated_annealing(
                initial_sa_grid,
                ship,
                modules,
                tech,
                player_owned_rewards,
                 # Consider using slightly faster SA params here too
                initial_temperature=3500,
                cooling_rate=0.97,
                iterations_per_temp=25,
                initial_swap_probability=0.45,
                final_swap_probability=0.35,
                max_processing_time=10.0 # Limit initial SA time
            )
            if solved_grid is None:
                raise ValueError(f"Fallback simulated_annealing failed for {ship}/{tech} when no pattern fit.")
            print(f"INFO -- Fallback SA score (no pattern fit): {solved_bonus:.4f}")
            pattern_applied = False # Mark that a pattern wasn't the final result
            sa_was_initial_placement = True # SA was the initial placement method

    # --- Opportunity Refinement Stage ---
    # Start refinement from the best state achieved so far (pattern or initial SA)
    grid_to_refine = solved_grid.copy()
    current_best_score = calculate_grid_score(grid_to_refine, tech) # Score *before* refinement

    opportunity = find_supercharged_opportunities(grid_to_refine, modules, ship, tech)

    if opportunity:
        print(f"INFO -- Found opportunity for refinement at window starting: {opportunity}")
        opportunity_x, opportunity_y = opportunity
        refined_grid_candidate = None
        refined_score_global = -1.0
        sa_was_ml_fallback = False # Flag for ML->SA fallback

        # --- Branch based on experimental flag ---
        if experimental:
            print("INFO -- Experimental flag is True. Attempting ML refinement first.")
            # --- Try ML Refinement ---
            refined_grid_candidate, refined_score_global = _handle_ml_opportunity(
                grid_to_refine.copy(), # Pass a copy
                modules, ship, tech, player_owned_rewards,
                opportunity_x, opportunity_y
            )

            # --- Fallback to SA/Refine if ML failed ---
            if refined_grid_candidate is None:
                print("INFO -- ML refinement failed or model not found. Falling back to SA/Refine refinement.")
                sa_was_ml_fallback = True # Mark that SA is running due to ML failure
                refined_grid_candidate, refined_score_global = _handle_sa_refine_opportunity(
                    grid_to_refine.copy(), # Pass a fresh copy
                    modules, ship, tech, player_owned_rewards,
                    opportunity_x, opportunity_y
                )
            # else: ML refinement succeeded (or returned original grid), proceed with its result.

        else: # Not experimental
            print("INFO -- Experimental flag is False. Using SA/Refine refinement directly.")
            # --- Use SA/Refine Refinement ---
            refined_grid_candidate, refined_score_global = _handle_sa_refine_opportunity(
                grid_to_refine.copy(), # Pass a copy
                modules, ship, tech, player_owned_rewards,
                opportunity_x, opportunity_y
            )

        # --- Compare and Update based on Refinement Result ---
        if refined_grid_candidate is not None and refined_score_global >= current_best_score:
            # --- Refinement Improved Score ---
            print(
                f"INFO -- Opportunity refinement (using {'ML/SA Fallback' if sa_was_ml_fallback else ('ML' if experimental else 'SA/Refine')}) improved score from {current_best_score:.4f} to {refined_score_global:.4f}"
            )
            solved_grid = refined_grid_candidate # Update solved_grid with the better one
            solved_bonus = refined_score_global # Update the score
            sa_was_initial_placement = False # Grid state changed by refinement
        else:
            # --- Refinement Failed or Did Not Improve ---
            if refined_grid_candidate is not None:
                 # Refinement ran but didn't improve
                 print(
                     f"INFO -- Opportunity refinement (using {'ML/SA Fallback' if sa_was_ml_fallback else ('ML' if experimental else 'SA/Refine')}) did not improve score ({refined_score_global:.4f} vs {current_best_score:.4f})."
                 )
            else:
                 # Refinement failed completely (either ML failed and its SA fallback failed, or standard SA failed)
                 print(f"INFO -- Opportunity refinement (using {'ML/SA Fallback' if sa_was_ml_fallback else ('ML' if experimental else 'SA/Refine')}) failed completely.")

            # --- Final Fallback SA Logic (Only if experimental and NOT already an ML->SA fallback) ---
            # Run this ONLY IF:
            # 1. Experimental flag is True
            # 2. AND the refinement attempt that failed/didn't improve was NOT the SA run triggered by ML failing.
            if experimental and not sa_was_ml_fallback:
                print("INFO -- Experimental flag is True AND refinement didn't improve/failed (and was not ML->SA fallback). Attempting final fallback Simulated Annealing.")
                # Run SA on the grid state *before* the failed/unimproved refinement attempt
                grid_for_sa_fallback = grid_to_refine.copy() # Use the grid state before this refinement attempt

                # Call _handle_sa_refine_opportunity again for the fallback
                sa_fallback_grid, sa_fallback_bonus = _handle_sa_refine_opportunity(
                    grid_for_sa_fallback.copy(), # Pass a copy
                    modules, ship, tech, player_owned_rewards,
                    opportunity_x, opportunity_y,
                    # Optional: different SA params
                )

                # Check if the final fallback improved the score compared to the pre-refinement state
                if sa_fallback_grid is not None and sa_fallback_bonus > current_best_score:
                    print(f"INFO -- Final fallback SA improved score from {current_best_score:.4f} to {sa_fallback_bonus:.4f}")
                    solved_grid = sa_fallback_grid
                    solved_bonus = sa_fallback_bonus
                    sa_was_initial_placement = False # Grid state changed by fallback SA
                elif sa_fallback_grid is not None:
                    print(f"INFO -- Final fallback SA did not improve score ({sa_fallback_bonus:.4f} vs {current_best_score:.4f}). Keeping previous best.")
                    # solved_grid, solved_bonus, sa_was_initial_placement remain unchanged
                else:
                     print(f"ERROR -- Final fallback Simulated Annealing failed. Keeping previous best.")
                     # solved_grid, solved_bonus, sa_was_initial_placement remain unchanged
            elif sa_was_ml_fallback:
                # Experimental is True, ML failed, SA fallback ran and didn't improve. Do nothing more.
                print("INFO -- Skipping final fallback SA because ML failed and its SA fallback didn't improve.")
                # sa_was_initial_placement remains unchanged
            else: # experimental is False
                # Experimental is False, refinement failed/didn't improve. Do nothing more.
                print("INFO -- Experimental flag is False, keeping previous best grid without final fallback SA.")
                # sa_was_initial_placement remains unchanged

    else: # No opportunity found
        print("INFO -- No supercharged opportunity found for refinement.")
        # solved_grid, solved_bonus, sa_was_initial_placement remain unchanged

    # --- Final Checks and Fallbacks (Simulated Annealing if modules not placed) ---
    # Check if all modules were placed after pattern matching and potential refinement
    all_modules_placed = check_all_modules_placed(solved_grid, modules, ship, tech, player_owned_rewards)

    # Run final SA check ONLY if modules aren't placed AND the current state didn't come directly from initial SA
    if not all_modules_placed and not sa_was_initial_placement:
        print(
            f"WARNING! -- Not all modules placed AND initial placement wasn't SA. Running final SA."
        )

        # Start SA from the current state of solved_grid, but clear the tech first
        grid_for_final_sa = solved_grid.copy()
        clear_all_modules_of_tech(grid_for_final_sa, tech)

        temp_solved_grid, temp_solved_bonus = simulated_annealing(
            grid_for_final_sa,
            ship,
            modules,
            tech,
            player_owned_rewards,
            # Use standard SA parameters for final check
            initial_temperature=4000,
            cooling_rate=0.98,
            iterations_per_temp=30,
            initial_swap_probability=0.40,
            final_swap_probability=0.3,
            max_processing_time=15.0 # Limit final SA time
        )

        if temp_solved_grid is not None:
            # Recalculate score after final SA
            final_sa_score = calculate_grid_score(temp_solved_grid, tech)
            # Only update if the final SA score is better than the current solved_bonus
            if final_sa_score > solved_bonus:
                print(f"INFO -- Final SA (due to unplaced modules) improved score from {solved_bonus:.4f} to {final_sa_score:.4f}")
                solved_grid = temp_solved_grid
                solved_bonus = final_sa_score
            else:
                print(
                    f"INFO -- Final SA (due to unplaced modules) did not improve score ({final_sa_score:.4f} vs {solved_bonus:.4f}). Keeping previous best."
                )
        else:
            print(
                f"ERROR -- Final simulated_annealing solver (due to unplaced modules) failed for ship: '{ship}' -- tech: '{tech}'. Returning previous best grid."
            )
            # Keep the existing solved_grid and solved_bonus
    elif not all_modules_placed and sa_was_initial_placement:
         print(
             f"WARNING! -- Not all modules placed, but initial placement WAS SA. Skipping final SA check."
         )
    # else: # All modules placed, no need for final SA check

    # --- Final Result Calculation ---
    best_grid = solved_grid # The final state of solved_grid is our best result
    best_bonus = calculate_grid_score(best_grid, tech) # Recalculate final score for certainty

    # Calculate the percentage of the official solve score achieved
    if solve_score > 1e-9: # Use a small epsilon to avoid division by zero/near-zero
        percentage = (best_bonus / solve_score) * 100
    else:
        # Handle cases where solve_score is 0 or very close to it
        percentage = 100.0 if best_bonus > 1e-9 else 0.0

    print(
        f"SUCCESS -- Final Score: {best_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) for ship: '{ship}' -- tech: '{tech}'"
    )
    print_grid_compact(best_grid) # Print the final best grid

    return best_grid, percentage, best_bonus # Return the final best grid, percentage, and score


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


# --- NEW ML-Specific Function ---
def create_localized_grid_ml(grid, opportunity_x, opportunity_y, tech):
    """
    Creates a localized grid (4x3) around a given opportunity for ML processing.

    Modules of *other* tech types within the localized area are temporarily
    removed, and their corresponding cells in the localized grid are marked
    as inactive. The original state of these modified cells (from the main grid)
    is stored for later restoration.

    Args:
        grid (Grid): The main grid.
        opportunity_x (int): The x-coordinate of the opportunity (top-left corner).
        opportunity_y (int): The y-coordinate of the opportunity (top-left corner).
        tech (str): The technology type being optimized (modules of this tech are kept).

    Returns:
        tuple: A tuple containing:
            - localized_grid (Grid): The localized grid prepared for ML.
            - start_x (int): The starting x-coordinate of the localized grid in the main grid.
            - start_y (int): The starting y-coordinate of the localized grid in the main grid.
            - original_state_map (dict): A dictionary mapping main grid coordinates
                                         (x, y) to their original cell data for cells
                                         that were modified (other tech removed).
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
    original_state_map = {}  # To store original state of modified cells

    # Copy the grid structure and module data, modifying for other techs
    for y_main in range(start_y, end_y):
        for x_main in range(start_x, end_x):
            localized_x = x_main - start_x
            localized_y = y_main - start_y
            main_cell = grid.get_cell(x_main, y_main)
            local_cell = localized_grid.get_cell(localized_x, localized_y)

            # Always copy basic structure like supercharged status
            local_cell["supercharged"] = main_cell["supercharged"]

            # Check the module and its tech in the main grid
            if main_cell["module"] is not None and main_cell["tech"] != tech:
                # --- Other Tech Found ---
                # 1. Store the original state using main grid coordinates
                #    Use deepcopy to ensure modifications to main_cell later don't affect the stored state
                original_state_map[(x_main, y_main)] = deepcopy(main_cell)

                # 2. Clear module info in the local grid cell
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
                    del local_cell["module_position"]  # Or set to None

                # 3. Mark the local cell as inactive
                local_cell["active"] = False

            elif not main_cell["active"]:
                # --- Inactive Cell in Main Grid ---
                # Keep it inactive in the local grid and ensure module info is cleared
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
                # --- Target Tech, Empty Active Cell ---
                # Copy the cell data directly (preserves target tech modules and empty active state)
                # Use deepcopy here as well for safety, although update might be sufficient
                local_cell.update(deepcopy(main_cell))
                # Ensure 'active' is explicitly True if copied from an active main cell
                local_cell["active"] = True

    return localized_grid, start_x, start_y, original_state_map


# --- Restoration Function (Keep as is from previous response) ---
def restore_original_state(grid, original_state_map):
    """
    Restores the original state of cells in the main grid that were temporarily
    modified (other tech modules removed and marked inactive) during localization.

    Args:
        grid (Grid): The main grid to restore.
        original_state_map (dict): The dictionary mapping main grid coordinates (x, y)
                                   to their original cell data, as returned by
                                   create_localized_grid_ml.
    """
    if not original_state_map:
        return  # Nothing to restore

    print(f"INFO -- Restoring original state for {len(original_state_map)} cells.")
    for (x, y), original_cell_data in original_state_map.items():
        if 0 <= x < grid.width and 0 <= y < grid.height:
            # Directly update the cell in the main grid with its original data
            # Use deepcopy of the stored data for safety when updating
            grid.cells[y][x].update(deepcopy(original_cell_data))
        else:
            print(f"Warning -- Coordinate ({x},{y}) from original_state_map is out of bounds for the main grid.")


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
