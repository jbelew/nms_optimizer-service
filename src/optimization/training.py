# optimization/training.py
import logging
import math
import multiprocessing
import time
from itertools import permutations

from ..bonus_calculations import calculate_grid_score
from ..module_placement import place_module, clear_all_modules_of_tech


def _evaluate_permutation_worker(args):
    """
    Worker function to evaluate a single permutation.
    Takes a tuple of arguments to be easily used with pool.map.
    Creates its own grid copy internally to avoid modifying the shared base.
    """
    # 1. Unpack arguments
    placement_indices, original_base_grid, tech_modules, available_positions, tech, solve_type = args
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
        logging.error(
            f"Invalid placement index in worker. Indices: {placement_indices}, Available: {len(available_positions)}"
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
            logging.error(f"IndexError during place_module at ({x},{y}) in worker. Skipping.")
            placement_successful = False
            break
        except Exception as e:  # Catch other potential errors
            logging.error(f"Exception during place_module at ({x},{y}) in worker: {e}. Skipping.")
            placement_successful = False
            break

    if not placement_successful:
        return (-1.0, None)  # Indicate error or skip

    # 6. Calculate score on the local working_grid
    grid_bonus = calculate_grid_score(working_grid, tech)

    # 7. Return score and the indices
    return (grid_bonus, placement_indices)


def refine_placement_for_training(grid, tech_modules, tech, num_workers=None, solve_type: str = "normal"):
    """
    Optimizes module placement using brute-force permutations with multiprocessing,
    intended for generating optimal ground truth for training data.
    Optimized copying strategy and added memory management safeguards.
    """
    start_time = time.time()
    optimal_grid = None
    highest_bonus = -1.0  # Use -1 to clearly distinguish from a valid 0 score

    # --- Initial Checks (same as before) ---
    if not tech_modules:
        logging.warning(f"No modules for {tech}. Returning cleared grid.")
        cleared_grid = grid.copy()
        clear_all_modules_of_tech(cleared_grid, tech)
        return cleared_grid, 0.0

    num_modules_to_place = len(tech_modules)
    available_positions = [
        (x, y) for y in range(grid.height) for x in range(grid.width) if grid.get_cell(x, y)["active"]
    ]
    num_available = len(available_positions)

    if num_available < num_modules_to_place:
        logging.warning(
            f"Not enough active slots ({num_available}) for modules ({num_modules_to_place}) for {tech}. Returning cleared grid."
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
        logging.info(
            f"Training ({tech}): {num_available} slots, {num_modules_to_place} modules -> {num_permutations:,} permutations."
        )
    except (ValueError, OverflowError):
        logging.info(
            f"Training ({tech}): {num_available} slots, {num_modules_to_place} modules -> Large number of permutations."
        )
        num_permutations = float("inf")

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        # Optional: Limit workers if permutations are truly astronomical, though fixing copying might be enough
        # if num_permutations > 100_000_000 and num_workers > 4:
        #     logging.info(f"Limiting workers from {num_workers} to 4 due to extreme permutation count.")
        #     num_workers = 4
        logging.info(f"Using {num_workers} worker processes.")
    # --- End Worker Setup ---

    # --- Task Preparation ---
    # Generate permutations of *indices* into available_positions
    permutation_indices_iterator = permutations(range(num_available), num_modules_to_place)

    # Package arguments: Pass the single base_working_grid. It gets pickled by the pool mechanism.
    tasks = (
        (indices, base_working_grid, tech_modules, available_positions, tech, solve_type)
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
        calculated_chunksize = int(
            num_permutations // (num_workers * chunks_per_worker_target)  # Ensure integer
        )
        chunksize = max(chunksize, calculated_chunksize)
        # Add an upper limit to prevent huge chunks consuming too much memory at once
        max_chunksize = 50000  # Tune this based on memory observations
        chunksize = min(chunksize, max_chunksize)
    # --- End Chunksize Calculation ---

    # --- Multiprocessing Pool Execution ---
    logging.info(f"Starting parallel evaluation with chunksize={chunksize}...")
    # Add maxtasksperchild: Restarts worker after N tasks to help free memory
    maxtasks = 2000  # Tune this value (e.g., 1000-10000)
    logging.info(f"Setting maxtasksperchild={maxtasks}")
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=maxtasks) as pool:
        # imap_unordered is good for performance when order doesn't matter
        results_iterator = pool.imap_unordered(_evaluate_permutation_worker, tasks, chunksize=chunksize)

        # Progress reporting frequency
        update_frequency = max(
            1,
            chunksize * num_workers // 4,  # Update reasonably often
        )

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
                (processed_count / num_permutations * 100) if num_permutations != float("inf") else 0
                # Use \r and flush=True for inline updating
                logging.debug(
                    f"Processed ~{processed_count // 1000}k permutations. Best: {highest_bonus:.4f} ({elapsed:.1f}s)"
                )

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Parallel evaluation finished in {total_time:.2f} seconds. Processed {processed_count} permutations.")
    if total_time > 0:
        perms_per_sec = processed_count / total_time
        logging.info(f"Rate: {perms_per_sec:,.0f} permutations/sec")
    # --- End Pool Execution ---

    # --- Reconstruct Best Grid ---
    if best_placement_indices is not None:
        logging.info(f"Reconstructing best grid with score: {highest_bonus:.4f}")
        # Start from the original grid structure again for safety
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)  # Clear only the target tech
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
                logging.error(f"ERROR during final grid reconstruction at ({x},{y}): {e}")
                reconstruction_successful = False
                break

        if not reconstruction_successful:
            logging.warning("Final grid reconstruction failed. Returning cleared grid.")
            optimal_grid = grid.copy()
            clear_all_modules_of_tech(optimal_grid, tech)
            highest_bonus = 0.0
        else:
            # Optional score verification
            final_score = calculate_grid_score(optimal_grid, tech)
            if abs(final_score - highest_bonus) > 1e-6:
                logging.warning(
                    f"Final score ({final_score:.4f}) differs from tracked best ({highest_bonus:.4f}). Using final score."
                )
                highest_bonus = final_score

    # --- Handle No Valid Placement Found ---
    elif num_modules_to_place > 0:  # Check if modules existed but no solution found
        logging.warning(f"No optimal grid found for {tech}. Returning cleared grid.")
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)
        highest_bonus = 0.0  # Score is 0 for a cleared grid
    else:  # No modules to place initially
        logging.info(f"No modules to place for {tech}. Returning cleared grid.")
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)
        highest_bonus = 0.0
    # --- End Handling ---

    return optimal_grid, highest_bonus
