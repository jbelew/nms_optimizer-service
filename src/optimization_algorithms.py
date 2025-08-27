# optimization_algorithms.py
from grid_utils import Grid, restore_original_state, apply_localized_grid_changes
from modules_utils import get_tech_modules, get_tech_modules_for_training
from grid_display import print_grid_compact
from bonus_calculations import calculate_grid_score
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
from data_loader import get_solve_map
from solve_map_utils import filter_solves


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
    print(
        f"INFO -- refine_placement completed {iteration_count} iterations for ship: '{ship}' -- tech: '{tech}'"
    )

    return optimal_grid, highest_bonus


# --- Worker Function (Define at top level) ---
def _evaluate_permutation_worker(args):
    """
    Worker function to evaluate a single permutation.
    Takes a tuple of arguments to be easily used with pool.map.
    Creates its own grid copy internally to avoid modifying the shared base.
    """
    # 1. Unpack arguments
    placement_indices, original_base_grid, tech_modules, available_positions, tech = (
        args
    )
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
            print(
                f"ERROR: IndexError during place_module at ({x},{y}) in worker. Skipping."
            )
            placement_successful = False
            break
        except Exception as e:  # Catch other potential errors
            print(
                f"ERROR: Exception during place_module at ({x},{y}) in worker: {e}. Skipping."
            )
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
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.get_cell(x, y)["active"]
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
    permutation_indices_iterator = permutations(
        range(num_available), num_modules_to_place
    )

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
        calculated_chunksize = int(
            num_permutations
            // (num_workers * chunks_per_worker_target)  # Ensure integer
        )
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
        results_iterator = pool.imap_unordered(
            _evaluate_permutation_worker, tasks, chunksize=chunksize
        )

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
                (processed_count / num_permutations * 100) if num_permutations != float(
                    "inf"
                ) else 0
                # Use \r and flush=True for inline updating
                print(
                    f"\r-- Processed ~{processed_count // 1000}k permutations. Best: {highest_bonus:.4f} ({elapsed:.1f}s)",
                    end="",
                    flush=True,
                )

    print()  # Ensure the next print starts on a new line after progress updates

    end_time = time.time()
    total_time = end_time - start_time
    print(
        f"-- Parallel evaluation finished in {total_time:.2f} seconds. Processed {processed_count} permutations."
    )
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
        print(
            f"Warning: No optimal grid found for {ship}/{tech}. Returning cleared grid."
        )
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


def determine_window_dimensions(module_count: int, tech) -> tuple[int, int]:
    """
    Determines the window width and height based on the number of modules.

    This defines the size of the grid window used for placement calculations.

    Args:
        module_count: The total number of modules for a given technology.

    Returns:
        A tuple containing the calculated window_width and window_height.
    """
    window_width, window_height = 3, 3

    if module_count < 1:
        # Handle cases with zero or negative modules (optional, but good practice)
        print(f"Warning: Module count is {module_count}. Returning default 1x1 window.")
        return 1, 1
    elif module_count < 3:
        window_width, window_height = 1, 2
    elif module_count < 4:
        window_width, window_height = 1, 3
    elif module_count < 7:
        window_width, window_height = 2, 3
    elif module_count < 8 and tech == "pulse-spitter" or tech == "jetpack":
        window_width, window_height = 3, 3
    elif module_count < 8:
        window_width, window_height = 4, 2
    elif module_count < 9:
        window_width, window_height = 4, 2
    elif module_count < 10 and tech == "bolt-caster":
        window_width, window_height = 4, 3
    elif module_count < 10:
        window_width, window_height = 3, 3
    elif module_count >= 10:
        window_width, window_height = 4, 3

    return window_width, window_height


def rotate_pattern(pattern):
    "Rotates a pattern 90 degrees clockwise."
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
    "Mirrors a pattern horizontally."
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
    "Mirrors a pattern vertically."
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
    """Applies a pattern to a *copy* of the grid at a given starting position.

    Returns a new grid with the pattern applied, or None if the pattern cannot be applied.
    """
    # Create a deep copy of the grid to avoid modifying the original
    new_grid = grid.copy()

    tech_modules_available = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if (
        tech_modules_available is None
    ):  # Should not happen if filter_solves worked correctly
        print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
        return None, 0
    # Create a mapping from module id to module data
    available_module_ids_map = {m["id"]: m for m in tech_modules_available}

    # --- Pre-check 1: Determine if the pattern is even placeable by the player and fits basic constraints ---
    # Count how many modules the pattern *intends* to place that the player actually owns.
    expected_module_placements_in_pattern = 0
    for (
        module_id_in_pattern_val
    ) in pattern.values():  # Iterate through values (module IDs)
        if (
            module_id_in_pattern_val is not None
            and module_id_in_pattern_val in available_module_ids_map
        ):
            expected_module_placements_in_pattern += 1

    # If the pattern has defined module IDs, but none are owned by the player, this pattern is not applicable.
    if expected_module_placements_in_pattern == 0 and any(
        pid is not None for pid in pattern.values()
    ):
        return None, 0

    # --- Pre-check 2: Check for overlaps, off-grid, or inactive cells for REQUIRED modules ---
    for pattern_x, pattern_y in pattern.keys():  # Iterate through keys (coordinates)
        module_id_in_pattern = pattern.get((pattern_x, pattern_y))
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y

        # Is this part of the pattern trying to place an owned module?
        if (
            module_id_in_pattern is not None
            and module_id_in_pattern in available_module_ids_map
        ):
            if not (0 <= grid_x < new_grid.width and 0 <= grid_y < new_grid.height):
                # A required module (owned, non-None) would be off-grid. This pattern variation doesn't fit.
                return None, 0

            current_cell_on_new_grid = new_grid.get_cell(grid_x, grid_y)
            if not current_cell_on_new_grid["active"]:
                # Cannot place a required module on an inactive cell.
                return None, 0

            if (
                current_cell_on_new_grid["module"] is not None
                and current_cell_on_new_grid["tech"] != tech
            ):
                # Overlap with a module of a *different* technology.
                return None, 0
        # If module_id_in_pattern is None, or not in available_module_ids_map, we don't check its target cell strictly here,
        # as it won't be placed anyway or it's an intentionally empty slot.

    # If all pre-checks pass, proceed with actual placement attempt
    clear_all_modules_of_tech(
        new_grid, tech
    )  # Clear target tech modules for a clean placement

    successfully_placed_this_variation = 0
    for (pattern_x, pattern_y), module_id_in_pattern in pattern.items():
        grid_x = start_x + pattern_x
        grid_y = start_y + pattern_y

        if not (0 <= grid_x < new_grid.width and 0 <= grid_y < new_grid.height):
            continue  # Skip parts of the pattern that are off-grid

        if module_id_in_pattern is None:  # Intentionally empty slot in pattern
            continue

        if module_id_in_pattern in available_module_ids_map:
            module_data = available_module_ids_map[module_id_in_pattern]
            # Cell activity and non-overlap with *other* tech already confirmed by pre-checks for these modules.
            # The cell is also guaranteed to be empty of the *target* tech due to clear_all_modules_of_tech.
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
            successfully_placed_this_variation += 1

    # --- Post-placement Check: Did we place all *expected* modules? ---
    # This ensures that if the pattern expected to place modules (that player owns), they were all placed.
    # If `expected_module_placements_in_pattern` is 0 (e.g. pattern was all "None", or all unowned modules), this check is skipped.
    if (
        expected_module_placements_in_pattern > 0
        and successfully_placed_this_variation < expected_module_placements_in_pattern
    ):
        # This implies some expected modules (owned, non-None in pattern) couldn't be placed.
        # This should ideally be caught by pre-checks (e.g. inactive cell, off-grid).
        return None, 0

    # If we reach here, the pattern (or what's placeable from it according to ownership and grid constraints) was applied.
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
    This includes a boost for modules with group adjacency requirements.

    Args:
        grid (Grid): The grid.
        tech (str): The tech type of the modules to consider.

    Returns:
        int: The adjacency score.
    """
    module_edge_weight = 3.0  # Weight for adjacency to other modules
    grid_edge_weight = 0.5  # Weight for adjacency to grid edges
    group_adjacency_weight = 5.0  # Weight for adjacency to modules in the same group

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

                # --- Adjacency Checks ---
                num_adjacent_same_group = 0
                adjacency_rule = cell.get("adjacency")
                adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

                for adj_x, adj_y in adjacent_positions:
                    if 0 <= adj_x < grid.width and 0 <= adj_y < grid.height:
                        adjacent_cell = grid.get_cell(adj_x, adj_y)
                        if adjacent_cell["module"] is not None:
                            # Standard adjacency bonus for being next to other modules of a different tech
                            if adjacent_cell["tech"] != tech:
                                total_adjacency_score += module_edge_weight

                            # Count for group adjacency bonus if adjacency rules match
                            if (
                                adjacency_rule
                                and adjacent_cell.get("adjacency") == adjacency_rule
                            ):
                                num_adjacent_same_group += 1

                # Check for group adjacency bonus from "greater_n" or "lesser_n" rules
                if isinstance(adjacency_rule, str) and "_" in adjacency_rule:
                    parts = adjacency_rule.split("_")
                    if len(parts) == 2 and parts[1].isdigit():
                        rule_type = parts[0]
                        rule_value = int(parts[1])

                        # Apply bonus based on the rule
                        if (
                            rule_type == "greater"
                            and num_adjacent_same_group > rule_value
                        ):
                            total_adjacency_score += (
                                group_adjacency_weight * num_adjacent_same_group
                            )
                        elif (
                            rule_type == "lesser"
                            and num_adjacent_same_group < rule_value
                        ):
                            total_adjacency_score += group_adjacency_weight * (
                                rule_value - num_adjacent_same_group
                            )

    return total_adjacency_score


# --- Helper function for ML opportunity refinement ---
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
):
    """Handles the ML-based refinement within an opportunity window."""
    from ml_placement import ml_placement  # Keep import local if possible

    if player_owned_rewards is None:
        player_owned_rewards = []

    print(
        f"INFO -- Using ML for opportunity refinement at ({opportunity_x}, {opportunity_y}) with window {window_width}x{window_height}"
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
        modules,
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
    )

    # 3. Process ML result (logic remains the same)
    if ml_refined_grid is not None:
        # print(f"INFO -- ML refinement produced a grid. Applying changes...")
        grid_copy = grid.copy()
        clear_all_modules_of_tech(grid_copy, tech)
        apply_localized_grid_changes(grid_copy, ml_refined_grid, tech, start_x, start_y)
        restore_original_state(grid_copy, original_state_map)
        new_score_global = calculate_grid_score(grid_copy, tech)
        # print(f"INFO -- Score after ML refinement and restoration: {new_score_global:.4f}")
        return grid_copy, new_score_global
    else:
        # Handle ML failure (logic remains the same)
        print("INFO -- ML refinement failed or returned None. No changes applied.")
        grid_copy = grid.copy()
        restore_original_state(grid_copy, original_state_map)
        original_score = calculate_grid_score(grid_copy, tech)
        print(
            f"INFO -- Returning grid with original score after failed ML: {original_score:.4f}"
        )
        return None, 0.0


# --- Existing function for SA/Refine opportunity ---
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
):
    """Handles the SA/Refine-based refinement within an opportunity window."""
    print(
        f"INFO -- Using SA/Refine for opportunity refinement at ({opportunity_x}, {opportunity_y}) with window {window_width}x{window_height}"
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
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    num_modules = len(tech_modules) if tech_modules else 0

    # Refine the localized grid (no change in logic here)
    if num_modules < 6:
        print(f"INFO -- {tech} has less than 6 modules, running refine_placement")
        temp_refined_grid, temp_refined_bonus_local = refine_placement(
            localized_grid, ship, modules, tech, player_owned_rewards
        )
    else:
        print(f"INFO -- {tech} has 6 or more modules, running simulated_annealing")
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
        )

    # Process SA/Refine result (logic remains the same)
    if temp_refined_grid is not None:
        calculate_grid_score(temp_refined_grid, tech)
        # Apply changes back to the main grid copy (grid - which was cleared earlier)
        clear_all_modules_of_tech(grid, tech)
        apply_localized_grid_changes(grid, temp_refined_grid, tech, start_x, start_y)
        new_score_global = calculate_grid_score(grid, tech)
        print(f"INFO -- Score after SA/Refine refinement: {new_score_global:.4f}")
        # print_grid(grid) # Print the modified grid (grid)
        return grid, new_score_global
    else:
        print("INFO -- SA/Refine refinement failed. No changes applied.")
        # Return the grid (which was cleared of the tech) and indicate failure
        return grid, -1.0


def optimize_placement(
    grid,
    ship,
    modules,
    tech,
    player_owned_rewards=None,
    forced=False,
    experimental_window_sizing=False,
    progress_callback=None,
    run_id=None,
    send_grid_updates=False,
):
    """
    Optimizes the placement of modules in a grid for a specific ship and technology.
    Uses pre-defined solve maps first, then refines based on supercharged opportunities,
    prioritizing the pattern's location if its opportunity score is high and it contains
    an available supercharged slot.
    Includes an experimental path using ML placement with SA fallback.

    Args:
        grid (Grid): The initial grid state.
        ship (str): The ship type key.
        modules (dict): The main modules dictionary.
        tech (str): The technology key.
        player_owned_rewards (list, optional): List of reward module IDs owned. Defaults to None.
        forced (bool): If True and no pattern fits a solve map, forces SA.
                       If False, returns "Pattern No Fit" to allow UI intervention.
        experimental_window_sizing (bool): If True and tech is 'pulse', dynamically chooses
                                           between a 4x3 and 4x2 window for refinement.

    Returns:
        tuple: (best_grid, percentage, best_bonus, solve_method)
               - best_grid (Grid): The optimized grid.
               - percentage (float): The percentage of the official solve score achieved.
               - best_bonus (float): The actual score achieved by the best grid.
               - solve_method (str): The name of the method used to generate the final grid.
    Raises:
        ValueError: If no empty, active slots are available or if critical steps fail.
    """
    print(  # <<< KEEP: Start of process >>>
        f"INFO -- Attempting solve for ship: '{ship}' -- tech: '{tech}' -- Exp. Window: {experimental_window_sizing}"
    )

    if player_owned_rewards is None:
        player_owned_rewards = []

    # --- Get modules for the current tech ---
    # This list is used to determine module_count for experimental window sizing
    # and for the check_all_modules_placed function.
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if not tech_modules:
        # This case should ideally be caught by has_empty_active_slots or other checks,
        # but as a safeguard if get_tech_modules returns None or empty for a valid tech key.
        print(
            f"Warning: No modules retrieved for ship '{ship}', tech '{tech}'. Cannot proceed with optimization."
        )
        # Return a grid that's essentially empty for this tech, with 0 score.
        cleared_grid_on_fail = grid.copy()
        clear_all_modules_of_tech(cleared_grid_on_fail, tech)
        return cleared_grid_on_fail, 0.0, 0.0, "Module Definition Error"

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

    # Initialize variables for tracking best results
    solved_grid = grid.copy()  # Grid state after initial pattern/SA fallback
    solved_bonus = -float("inf")
    best_pattern_grid = grid.copy()  # Best grid found using pattern matching
    highest_pattern_bonus = -float("inf")
    best_pattern_adjacency_score = 0
    best_pattern_start_x = -1
    best_pattern_start_y = -1
    best_pattern_width = -1
    best_pattern_height = -1
    solve_score = 0
    sa_was_initial_placement = False
    solve_method = "Unknown"  # <<< Initialize solve_method >>>

    # --- Load Solves On-Demand ---
    all_solves_for_ship = get_solve_map(ship)
    # Create the structure that filter_solves expects
    solves_for_filtering = {ship: all_solves_for_ship} if all_solves_for_ship else {}
    filtered_solves = filter_solves(
        solves_for_filtering, ship, modules, tech, player_owned_rewards
    )
    # --- End On-Demand Loading ---

    # --- Initial Placement Strategy ---
    if ship not in filtered_solves or (
        ship in filtered_solves and tech not in filtered_solves[ship]
    ):
        # --- Special Case: No Solve Available ---
        print(
            f"INFO -- No solve found for ship: '{ship}' -- tech: '{tech}'. Placing modules in empty slots."
        )  # <<< KEEP: Important outcome >>>
        solve_method = "Initial Placement (No Solve)"  # <<< Set method >>>
        # Assuming place_all_modules_in_empty_slots is defined elsewhere
        solved_grid = place_all_modules_in_empty_slots(
            grid, modules, ship, tech, player_owned_rewards
        )
        solved_bonus = calculate_grid_score(solved_grid, tech)
        solve_score = 0
        percentage = 100.0 if solved_bonus > 1e-9 else 0.0
        # <<< KEEP: Final result for this path >>>
        print(
            f"SUCCESS -- Final Score (No Solve Map): {solved_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) using method '{solve_method}' for ship: '{ship}' -- tech: '{tech}'"
        )
        # print_grid_compact(solved_grid) # Optional: Can be noisy for many calls
        return solved_grid, round(percentage, 4), solved_bonus, solve_method
    else:
        # --- Case 2: Solve Map Exists ---
        solve_data = filtered_solves[ship][tech]
        original_pattern = solve_data["map"]
        solve_score = solve_data["score"]

        # print(f"INFO -- Found solve map for {ship}/{tech}. Score: {solve_score:.4f}. Attempting pattern matching.") # <<< KEEP: Important outcome >>>
        # Assuming get_all_unique_pattern_variations is defined elsewhere
        patterns_to_try = get_all_unique_pattern_variations(original_pattern)
        grid_dict = grid.to_dict()

        for pattern in patterns_to_try:
            x_coords = [coord[0] for coord in pattern.keys()]
            y_coords = [coord[1] for coord in pattern.keys()]
            if not x_coords or not y_coords:
                continue
            pattern_width = max(x_coords) + 1
            pattern_height = max(y_coords) + 1

            for start_x in range(grid.width - pattern_width + 1):
                for start_y in range(grid.height - pattern_height + 1):
                    temp_grid_pattern = Grid.from_dict(grid_dict)
                    # Assuming apply_pattern_to_grid is defined elsewhere
                    temp_result_grid, adjacency_score = apply_pattern_to_grid(
                        temp_grid_pattern,
                        pattern,
                        modules,
                        tech,
                        start_x,
                        start_y,
                        ship,
                        player_owned_rewards,
                    )
                    if temp_result_grid is not None:
                        current_pattern_bonus = calculate_grid_score(
                            temp_result_grid, tech
                        )
                        if current_pattern_bonus > highest_pattern_bonus:
                            highest_pattern_bonus = current_pattern_bonus
                            best_pattern_grid = temp_result_grid.copy()
                            best_pattern_adjacency_score = adjacency_score
                            best_pattern_start_x = start_x
                            best_pattern_start_y = start_y
                            best_pattern_width = pattern_width
                            best_pattern_height = pattern_height
                        elif (
                            current_pattern_bonus == highest_pattern_bonus
                            and adjacency_score
                            > best_pattern_adjacency_score  # Change >= to > to prefer earlier (top-left) positions on tie
                        ):
                            best_pattern_grid = temp_result_grid.copy()
                            best_pattern_adjacency_score = adjacency_score
                            best_pattern_start_x = start_x
                            best_pattern_start_y = start_y
                            best_pattern_width = pattern_width
                            best_pattern_height = pattern_height

        if highest_pattern_bonus > -float("inf"):
            solved_grid = best_pattern_grid
            solved_bonus = highest_pattern_bonus
            solve_method = "Pattern Match"  # <<< Set method >>>
            # <<< KEEP: Best pattern result >>>
            print(
                f"INFO -- Best pattern score: {solved_bonus:.4f} (Adjacency: {best_pattern_adjacency_score:.2f}) found at ({best_pattern_start_x},{best_pattern_start_y}) with size {best_pattern_width}x{best_pattern_height} for ship: '{ship}' -- tech: '{tech}' that fits."
            )
            # print_grid_compact(solved_grid)
            sa_was_initial_placement = False
        else:
            # --- Case 2b: No Pattern Fits ---
            if not forced:
                print(
                    f"INFO -- Solve map exists for {ship}/{tech}, but no pattern variation fits. Returning 'Pattern No Fit'. UI can prompt to force SA."
                )  # <<< KEEP: Important outcome >>>
                return None, 0.0, 0.0, "Pattern No Fit"
            else:
                # <<< KEEP: Important fallback >>>
                print(
                    f"WARNING -- Solve map exists for {ship}/{tech}, but no pattern variation fits the grid. FORCING initial Simulated Annealing."
                )
                initial_sa_grid = grid.copy()
                clear_all_modules_of_tech(initial_sa_grid, tech)
                solved_grid, solved_bonus = simulated_annealing(
                    initial_sa_grid,
                    ship,
                    modules,
                    tech,
                    grid,  # full_grid
                    player_owned_rewards,
                    cooling_rate=0.999,
                    iterations_per_temp=35,
                    initial_swap_probability=0.55,
                    max_processing_time=20.0,
                    progress_callback=progress_callback,
                    run_id=run_id,
                    stage="initial_placement",
                    send_grid_updates=send_grid_updates,
                )
                if solved_grid is None:
                    raise ValueError(
                        f"Forced fallback simulated_annealing failed for {ship}/{tech} when no pattern fit."
                    )
                print(
                    f"INFO -- Forced fallback SA score (no pattern fit): {solved_bonus:.4f}"
                )  # <<< KEEP: Result of fallback >>>
                solve_method = (
                    "Forced Initial SA (No Pattern Fit)"  # <<< Set method >>>
                )
                sa_was_initial_placement = True

    # --- Opportunity Refinement Stage ---
    # Ensure solved_grid is not None before proceeding (it could be if "Pattern No Fit" was returned and this logic is ever reached without a prior assignment)
    if solved_grid is None:
        # This case should ideally not be hit if "Pattern No Fit" returns early.
        # However, as a safeguard:
        raise ValueError(
            "optimize_placement: solved_grid is None before opportunity refinement, indicating an unexpected state."
        )
    grid_after_initial_placement = (
        solved_grid.copy()
    )  # Grid state before refinement starts
    current_best_score = calculate_grid_score(grid_after_initial_placement, tech)

    # Prepare grid for opportunity scanning (clear target tech)
    grid_for_opportunity_scan = grid_after_initial_placement.copy()
    clear_all_modules_of_tech(grid_for_opportunity_scan, tech)

    # --- Calculate Pattern Window Score (if applicable) ---
    pattern_window_score = 0.0
    pattern_opportunity_result = None
    if (
        highest_pattern_bonus > -float("inf")
        and best_pattern_start_x != -1
        and best_pattern_width > 0
    ):
        try:
            # Assuming create_localized_grid and calculate_window_score are defined elsewhere
            pattern_window_grid, _, _ = create_localized_grid(
                grid_for_opportunity_scan,  # Use the grid with tech cleared
                best_pattern_start_x,
                best_pattern_start_y,
                tech,
                best_pattern_width,
                best_pattern_height,
            )
            pattern_window_score = calculate_window_score(pattern_window_grid, tech)
            pattern_opportunity_result = (
                best_pattern_start_x,
                best_pattern_start_y,
                best_pattern_width,
                best_pattern_height,
            )
            # print(f"DEBUG -- Calculated opportunity score for pattern area ({best_pattern_width}x{best_pattern_height} at {best_pattern_start_x},{best_pattern_start_y}): {pattern_window_score:.2f}") # <<< COMMENT OUT/DEBUG >>>
        except Exception as e:
            print(f"Warning: Error calculating pattern window score: {e}")
            pattern_window_score = -1.0
            pattern_opportunity_result = None

    # --- Find Scanned Supercharged Opportunities ---
    # Assuming find_supercharged_opportunities is defined elsewhere
    scanned_opportunity_result = find_supercharged_opportunities(
        grid_for_opportunity_scan, modules, ship, tech, player_owned_rewards
    )

    # --- Calculate Scanned Window Score (if applicable) ---
    scanned_window_score = -1.0
    if scanned_opportunity_result:
        scan_x, scan_y, scan_w, scan_h = scanned_opportunity_result
        try:
            # Assuming create_localized_grid and calculate_window_score are defined elsewhere
            scanned_window_grid, _, _ = create_localized_grid(
                grid_for_opportunity_scan,
                scan_x,
                scan_y,
                tech,
                scan_w,
                scan_h,  # Use the grid with tech cleared
            )
            scanned_window_score = calculate_window_score(scanned_window_grid, tech)
            # print(f"DEBUG -- Calculated opportunity score for best scanned area ({scan_w}x{scan_h} at {scan_x},{scan_y}): {scanned_window_score:.2f}") # <<< COMMENT OUT/DEBUG >>>
        except Exception as e:
            print(f"Warning: Error calculating scanned window score: {e}")
            scanned_window_score = -1.0
            scanned_opportunity_result = None  # Invalidate if score calculation failed

    # --- Compare Scores and Select Final Opportunity ---
    final_opportunity_result = None
    opportunity_source = "None"  # For logging
    if (
        pattern_window_score >= scanned_window_score
        and pattern_opportunity_result is not None
    ):
        print(
            "INFO -- Using pattern location as the refinement opportunity window (score >= scanned)."
        )  # <<< COMMENT OUT >>>
        final_opportunity_result = pattern_opportunity_result
        opportunity_source = "Pattern"
    elif scanned_opportunity_result is not None:
        print(
            "INFO -- Using scanned location as the refinement opportunity window (score > pattern or pattern invalid)."
        )  # <<< COMMENT OUT >>>
        final_opportunity_result = scanned_opportunity_result
        opportunity_source = "Scan"
    elif (
        pattern_opportunity_result is not None
    ):  # Fallback if scanning failed but pattern exists
        print(
            "INFO -- Using pattern location as the refinement opportunity window (scanning failed)."
        )  # <<< COMMENT OUT >>>
        final_opportunity_result = pattern_opportunity_result
        opportunity_source = "Pattern (Fallback)"
        # else: # <<< COMMENT OUT >>>
        print("INFO -- No suitable opportunity window found from pattern or scanning.")

    # --- Perform Refinement using the Selected Opportunity ---
    # --- Experimental Window Sizing for 'pulse' tech ---
    if (
        experimental_window_sizing
        and tech == "pulse"
        and final_opportunity_result
        and len(tech_modules) >= 8
    ):
        print("INFO -- Experimental window sizing active for 'pulse' tech.")
        opp_x_anchor, opp_y_anchor, current_opp_w, current_opp_h = (
            final_opportunity_result
        )

        # Determine the score of the current best opportunity (before considering 4x3 override)
        score_of_current_best_opportunity = -1.0
        if opportunity_source in ["Pattern", "Pattern (Fallback)"]:
            score_of_current_best_opportunity = pattern_window_score
        elif opportunity_source == "Scan":
            score_of_current_best_opportunity = scanned_window_score
        else:
            # Fallback: if opportunity_source is unknown but final_opportunity_result exists,
            # calculate its score based on its current dimensions.
            # This ensures we have a baseline for comparison.
            print(
                f"Warning: Unknown opportunity_source '{opportunity_source}' for experimental sizing. Recalculating score for current best."
            )
            temp_loc_grid, _, _ = create_localized_grid(
                grid_for_opportunity_scan.copy(),
                opp_x_anchor,
                opp_y_anchor,
                tech,
                current_opp_w,
                current_opp_h,
            )
            score_of_current_best_opportunity = calculate_window_score(
                temp_loc_grid, tech
            )

        print(
            f"INFO -- Experimental: Current best opportunity ({current_opp_w}x{current_opp_h} from {opportunity_source}) score: {score_of_current_best_opportunity:.4f}"
        )

        # Scan the entire grid for the best 4x3 window using _scan_grid_with_window
        # grid_for_opportunity_scan is the grid with the target tech cleared
        # tech_modules is available from the top of optimize_placement
        best_4x3_score_from_scan, best_4x3_pos_from_scan = _scan_grid_with_window(
            grid_for_opportunity_scan.copy(),  # Scan on the grid with tech cleared
            4,
            3,
            len(tech_modules),
            tech,  # Pass fixed dimensions, module count, tech
        )

        if best_4x3_pos_from_scan:
            # The score returned by _scan_grid_with_window is already the window score
            print(
                f"INFO -- Experimental: Best 4x3 window found by scan: score {best_4x3_score_from_scan:.4f} at ({best_4x3_pos_from_scan[0]},{best_4x3_pos_from_scan[1]})."
            )

            # Compare the best 4x3 score (from scan) with the score of the current best opportunity
            if best_4x3_score_from_scan > score_of_current_best_opportunity:
                print(
                    f"INFO -- Experimental: Scanned 4x3 window (score {best_4x3_score_from_scan:.4f}) is better than current best ({score_of_current_best_opportunity:.4f}). Selecting 4x3."
                )
                # Override final_opportunity_result with the 4x3 window's location and dimensions
                final_opportunity_result = (
                    best_4x3_pos_from_scan[0],
                    best_4x3_pos_from_scan[1],
                    4,
                    3,
                )
            else:
                print(
                    f"INFO -- Experimental: Current best opportunity (score {score_of_current_best_opportunity:.4f}) is better or equal to scanned 4x3 ({best_4x3_score_from_scan:.4f}). Keeping original dimensions ({current_opp_w}x{current_opp_h})."
                )
                # final_opportunity_result remains unchanged
        else:
            print(
                "INFO -- Experimental: No suitable 4x3 window found by full scan. Keeping original dimensions."
            )
            # final_opportunity_result remains unchanged
    # --- End Experimental Window Sizing ---
    if final_opportunity_result:
        opportunity_x, opportunity_y, window_width, window_height = (
            final_opportunity_result
        )
        # <<< KEEP: Selected window info >>>
        # print(f"INFO -- Selected opportunity window ({opportunity_source}): Start ({opportunity_x}, {opportunity_y}), Size {window_width}x{window_height}")

        # <<< --- Add Check for Available Supercharged Slot in Window --- >>>
        window_has_available_sc = False
        # Use grid_for_opportunity_scan as it has the tech cleared
        for y_win in range(opportunity_y, opportunity_y + window_height):
            for x_win in range(opportunity_x, opportunity_x + window_width):
                # Bounds check for safety
                if (
                    0 <= x_win < grid_for_opportunity_scan.width
                    and 0 <= y_win < grid_for_opportunity_scan.height
                ):
                    cell = grid_for_opportunity_scan.get_cell(x_win, y_win)
                    if (
                        cell["active"]
                        and cell["supercharged"]
                        and cell["module"] is None
                    ):
                        window_has_available_sc = True
                        break
            if window_has_available_sc:
                break
        # <<< --- End Check --- >>>

        # <<< --- Only proceed if the check passes --- >>>
        if window_has_available_sc:
            # print("INFO -- Opportunity window contains at least one available supercharged slot. Proceeding with refinement.") # <<< COMMENT OUT >>>

            refined_grid_candidate = None
            refined_score_global = -1.0
            sa_was_ml_fallback = False
            refinement_method = ""  # For logging

            # --- Branch based on experimental flag ---
            # Default path: Try ML refinement first
            refinement_method = "ML"
            # Assuming _handle_ml_opportunity is defined elsewhere
            refined_grid_candidate, refined_score_global = _handle_ml_opportunity(
                grid_after_initial_placement.copy(),  # Pass a copy of the pre-cleared state
                modules,
                ship,
                tech,
                player_owned_rewards,
                opportunity_x,
                opportunity_y,
                window_width,
                window_height,
                progress_callback=progress_callback,
                run_id=run_id,
                stage="refinement_ml",
                send_grid_updates=send_grid_updates,
            )
            if refined_grid_candidate is None:
                print(
                    "INFO -- ML refinement failed or model not found. Falling back to SA/Refine refinement."
                )  # <<< KEEP: Important fallback >>>
                sa_was_ml_fallback = True
                refinement_method = "ML->SA/Refine Fallback"
                # Assuming _handle_sa_refine_opportunity is defined elsewhere - THIS WAS THE BUG
                refined_grid_candidate, refined_score_global = (
                    _handle_sa_refine_opportunity(
                        grid_after_initial_placement.copy(),  # Pass a fresh copy
                        modules,
                        ship,
                        tech,
                        player_owned_rewards,
                        opportunity_x,
                        opportunity_y,
                        window_width,
                        window_height,
                        progress_callback=progress_callback,
                        run_id=run_id,
                        stage="refinement_sa_fallback",
                        send_grid_updates=send_grid_updates,
                    )
                )

            # --- Compare and Update based on Refinement Result ---
            if (
                refined_grid_candidate is not None
                and refined_score_global >= current_best_score
            ):
                # <<< KEEP: Score improvement >>>
                print(
                    f"INFO -- Opportunity refinement ({refinement_method}) improved score from {current_best_score:.4f} to {refined_score_global:.4f}"
                )
                solved_grid = refined_grid_candidate
                solved_bonus = refined_score_global
                solve_method = refinement_method  # <<< Update method based on successful refinement >>>
                sa_was_initial_placement = False
            else:  # Refinement didn't improve or failed, keep grid_after_initial_placement
                if refined_grid_candidate is not None:
                    # <<< KEEP: Score did not improve >>>
                    print(
                        f"INFO -- Opportunity refinement ({refinement_method}) did not improve score ({refined_score_global:.4f} vs {current_best_score:.4f}). Keeping previous best."
                    )
                else:
                    # <<< KEEP: Refinement failed >>>
                    print(
                        f"INFO -- Opportunity refinement ({refinement_method}) failed completely. Keeping previous best."
                    )
                # solved_grid remains grid_after_initial_placement
                solved_bonus = current_best_score
                # solve_method remains what it was before refinement

                # --- Final Fallback SA Logic ---
                if (
                    not sa_was_ml_fallback
                ):  # Only run if the previous SA wasn't already a fallback from ML
                    # <<< KEEP: Important fallback >>>
                    print(
                        "INFO -- Refinement didn't improve/failed (and was not ML->SA fallback). Attempting final fallback Simulated Annealing."
                    )
                    grid_for_sa_fallback = grid_after_initial_placement.copy()
                    # Assuming _handle_sa_refine_opportunity is defined elsewhere
                    sa_fallback_grid, sa_fallback_bonus = _handle_sa_refine_opportunity(
                        grid_for_sa_fallback.copy(),  # Pass a copy
                        modules,
                        ship,
                        tech,
                        player_owned_rewards,
                        opportunity_x,
                        opportunity_y,
                        window_width,
                        window_height,
                        progress_callback=progress_callback,
                        run_id=run_id,
                        stage="final_fallback_sa",
                        send_grid_updates=send_grid_updates,
                    )
                    if (
                        sa_fallback_grid is not None
                        and sa_fallback_bonus > current_best_score
                    ):
                        # <<< KEEP: Score improvement >>>
                        print(
                            f"INFO -- Final fallback SA improved score from {current_best_score:.4f} to {sa_fallback_bonus:.4f}"
                        )
                        solved_grid = sa_fallback_grid
                        solved_bonus = sa_fallback_bonus
                        solve_method = "Final Fallback SA"  # <<< Update method >>>
                        sa_was_initial_placement = False
                    elif sa_fallback_grid is not None:
                        # <<< KEEP: Score did not improve >>>
                        print(
                            f"INFO -- Final fallback SA did not improve score ({sa_fallback_bonus:.4f} vs {current_best_score:.4f}). Keeping previous best."
                        )
                    else:
                        print(
                            "ERROR -- Final fallback Simulated Annealing failed. Keeping previous best."
                        )

        # <<< --- Else block for the supercharged check --- >>>
        else:
            # <<< KEEP: Reason for skipping refinement >>>
            print(
                "INFO -- Skipping refinement: Selected opportunity window does not contain any available supercharged slots."
            )
            # solved_grid remains grid_after_initial_placement
            solved_bonus = current_best_score
            # solve_method remains what it was before refinement
        # <<< --- End supercharged check block --- >>>

    else:  # No final_opportunity_result found
        # <<< KEEP: Reason for skipping refinement >>>
        print(
            "INFO -- No refinement performed as no suitable opportunity window was selected."
        )
        # solved_grid remains grid_after_initial_placement
        solved_bonus = current_best_score
        # solve_method remains what it was before refinement

    # --- Final Checks and Fallbacks (Simulated Annealing if modules not placed) ---
    # Assuming check_all_modules_placed is defined elsewhere (or imported)
    all_modules_placed = check_all_modules_placed(
        solved_grid, modules, ship, tech, player_owned_rewards
    )
    if not all_modules_placed and not sa_was_initial_placement:
        # <<< KEEP: Important fallback >>>
        print(
            "WARNING! -- Not all modules placed AND initial placement wasn't SA. Running final SA."
        )
        grid_for_final_sa = solved_grid.copy()
        clear_all_modules_of_tech(grid_for_final_sa, tech)
        temp_solved_grid, temp_solved_bonus_sa = simulated_annealing(
            grid_for_final_sa,
            ship,
            modules,
            tech,
            grid,  # full_grid
            player_owned_rewards,
            iterations_per_temp=25,
            initial_swap_probability=0.70,
            final_swap_probability=0.25,
            max_processing_time=20.0,
            progress_callback=progress_callback,
            run_id=run_id,
            stage="final_sa_unplaced_modules",
            send_grid_updates=send_grid_updates,
        )
        if temp_solved_grid is not None:
            final_sa_score = calculate_grid_score(temp_solved_grid, tech)
            # Use solved_bonus which holds the score *after* potential refinement
            if final_sa_score > solved_bonus:
                # <<< KEEP: Score improvement >>>
                print(
                    f"INFO -- Final SA (due to unplaced modules) improved score from {solved_bonus:.4f} to {final_sa_score:.4f}"
                )
                solved_grid = temp_solved_grid
                solved_bonus = final_sa_score
                solve_method = "Final SA (Unplaced Modules)"  # <<< Update method >>>
            else:
                # <<< KEEP: Score did not improve >>>
                print(
                    f"INFO -- Final SA (due to unplaced modules) did not improve score ({final_sa_score:.4f} vs {solved_bonus:.4f}). Keeping previous best."
                )
        else:
            print(
                f"ERROR -- Final simulated_annealing solver (due to unplaced modules) failed for ship: '{ship}' -- tech: '{tech}'. Returning previous best grid."
            )
    elif not all_modules_placed and sa_was_initial_placement:
        # <<< KEEP: Important warning >>>
        print(
            "WARNING! -- Not all modules placed, but initial placement WAS SA. Skipping final SA check."
        )

    # --- Final Result Calculation ---
    best_grid = solved_grid
    # Use the final solved_bonus calculated through the process
    best_bonus = solved_bonus  # This holds the score after refinement/fallbacks

    # Recalculate just in case, but should match solved_bonus
    final_check_score = calculate_grid_score(best_grid, tech)
    if abs(final_check_score - best_bonus) > 1e-6:
        print(
            f"Warning: Final check score {final_check_score:.4f} differs from tracked best_bonus {best_bonus:.4f}. Using check score."
        )
        best_bonus = final_check_score

    if solve_score > 1e-9:
        percentage = (best_bonus / solve_score) * 100
    else:
        percentage = 100.0 if best_bonus > 1e-9 else 0.0

    # <<< KEEP: Final result >>>
    print(
        f"SUCCESS -- Final Score: {best_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) using method '{solve_method}' for ship: '{ship}' -- tech: '{tech}'"
    )
    print_grid_compact(best_grid)

    return best_grid, round(percentage, 2), best_bonus, solve_method


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


def count_empty_in_localized(localized_grid):
    """Counts the number of truly empty slots in a localized grid."""
    count = 0
    for y in range(localized_grid.height):
        for x in range(localized_grid.width):
            cell = localized_grid.get_cell(x, y)
            if cell["module"] is None:  # Only count if the module slot is empty
                count += 1
    return count


def _scan_grid_with_window(grid_copy, window_width, window_height, module_count, tech):
    """
    Helper function to scan the grid with a specific window size and find the best opportunity.

    Args:
        grid_copy (Grid): A copy of the main grid (with target tech cleared).
        window_width (int): The width of the scanning window.
        window_height (int): The height of the scanning window.
        module_count (int): The number of modules for the tech.
        tech (str): The technology key.

    Returns:
        tuple: (best_score, best_start_pos) where best_start_pos is (x, y) or None.
               Returns (-1, None) if the window size is invalid for the grid.
    """
    best_score = -1
    best_start_pos = None

    # Check if window dimensions are valid for the grid size
    if window_width > grid_copy.width or window_height > grid_copy.height:
        print(
            f"Warning: Window size ({window_width}x{window_height}) is larger than grid ({grid_copy.width}x{grid_copy.height}). Cannot scan with this size."
        )
        return -1, None

    # Iterate through possible top-left corners for the window
    for start_y in range(grid_copy.height - window_height + 1):
        for start_x in range(grid_copy.width - window_width + 1):
            # Create a temporary window grid reflecting the current slice
            window_grid = Grid(window_width, window_height)
            for y in range(window_height):
                for x in range(window_width):
                    grid_x = start_x + x
                    grid_y = start_y + y
                    # Bounds check (should be handled by outer loops, but safety)
                    if 0 <= grid_x < grid_copy.width and 0 <= grid_y < grid_copy.height:
                        cell = grid_copy.get_cell(grid_x, grid_y)
                        # Copy relevant data to the window grid cell
                        window_grid.cells[y][x]["active"] = cell["active"]
                        window_grid.cells[y][x]["supercharged"] = cell["supercharged"]
                        window_grid.cells[y][x]["module"] = cell[
                            "module"
                        ]  # Keep module info for checks
                        window_grid.cells[y][x]["tech"] = cell["tech"]
                    else:
                        window_grid.cells[y][x]["active"] = (
                            False  # Mark as inactive if out of bounds
                        )

            # Check if the window has at least one available supercharged slot
            has_available_supercharged = False
            for y in range(window_height):
                for x in range(window_width):
                    cell = window_grid.get_cell(x, y)
                    if (
                        cell["supercharged"]
                        and cell["module"] is None
                        and cell["active"]
                    ):
                        has_available_supercharged = True
                        break
                if has_available_supercharged:
                    break
            if not has_available_supercharged:
                continue  # Skip this window

            # Check if the number of available cells in the current window is sufficient
            available_cells_in_window = 0
            for y in range(window_height):
                for x in range(window_width):
                    cell = window_grid.get_cell(x, y)
                    if cell["active"] and cell["module"] is None:
                        available_cells_in_window += 1
            if available_cells_in_window < module_count:
                continue  # Skip this window

            # Calculate score and update best if this window is better
            window_score = calculate_window_score(window_grid, tech)
            if window_score > best_score:
                best_score = window_score
                best_start_pos = (start_x, start_y)

    return best_score, best_start_pos


def find_supercharged_opportunities(
    grid, modules, ship, tech, player_owned_rewards=None
):
    """
    Scans the entire grid with a sliding window (dynamically sized, including rotation
    if non-square) to find the highest-scoring window containing available
    supercharged slots.

    Args:
        grid (Grid): The current grid layout.
        modules (dict): The module data.
        ship (str): The ship type.
        tech (str): The technology type.
        player_owned_rewards (list, optional): List of reward module IDs owned. Defaults to None.

    Returns:
        tuple or None: A tuple (opportunity_x, opportunity_y, best_width, best_height)
                       representing the top-left corner and dimensions of the best window,
                       or None if no suitable window is found or if all supercharged slots
                       are occupied.
    """
    grid_copy = grid.copy()
    # Clear the target tech modules to evaluate potential placement areas
    clear_all_modules_of_tech(grid_copy, tech)

    # Check if there are any unoccupied supercharged slots (no change needed)
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

    # Determine Dynamic Window Size (no change needed)
    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
    if tech_modules is None:
        print(
            f"Error: No modules found for ship '{ship}' and tech '{tech}' in find_supercharged_opportunities."
        )
        return None
    module_count = len(tech_modules)
    window_width, window_height = determine_window_dimensions(module_count, tech)
    print(
        f"INFO -- Using dynamic window size {window_width}x{window_height} for {tech} ({module_count} modules)."
    )

    # --- Scan with Original Dimensions ---
    best_score1, best_pos1 = _scan_grid_with_window(
        grid_copy, window_width, window_height, module_count, tech
    )

    # --- Scan with Rotated Dimensions (if needed) ---
    best_score2 = -1
    best_pos2 = None
    rotated_needed = (
        window_width != window_height
    )  # Check if width and height are different
    rotated_width, rotated_height = 0, 0  # Initialize for print statement clarity

    if rotated_needed:
        rotated_width, rotated_height = window_height, window_width  # Swap dimensions
        # print(f"INFO -- Also checking rotated window size {rotated_width}x{rotated_height}.")
        best_score2, best_pos2 = _scan_grid_with_window(
            grid_copy, rotated_width, rotated_height, module_count, tech
        )

    # --- Compare Results and Determine Best Dimensions ---
    overall_best_score = -1
    overall_best_pos = None
    overall_best_width = 0  # <<< Store best width
    overall_best_height = 0  # <<< Store best height

    # Check original scan result
    if best_score1 > overall_best_score:
        overall_best_score = best_score1
        overall_best_pos = best_pos1
        overall_best_width = window_width  # <<< Store original dimensions
        overall_best_height = window_height  # <<< Store original dimensions

    # Check rotated scan result (if performed)
    if best_score2 > overall_best_score:
        overall_best_score = best_score2
        overall_best_pos = best_pos2
        overall_best_width = rotated_width  # <<< Store rotated dimensions
        overall_best_height = rotated_height  # <<< Store rotated dimensions
        print(
            f"INFO -- Rotated window ({rotated_width}x{rotated_height}) provided a better score ({overall_best_score:.2f})."
        )
    elif (
        best_score1 > -1 and rotated_needed
    ):  # Only print if original scan found something and rotation was checked
        print(
            f"INFO -- Original window ({window_width}x{window_height}) provided the best score ({overall_best_score:.2f})."
        )
    elif best_score1 > -1 and not rotated_needed:  # Square window case
        print(
            f"INFO -- Best score found with square window ({window_width}x{window_height}): {overall_best_score:.2f}."
        )

    # --- Return the Overall Best Result ---
    if overall_best_pos is not None:
        best_x, best_y = overall_best_pos
        # print(f"INFO -- Best opportunity window found starting at: ({best_x}, {best_y}) with dimensions {overall_best_width}x{overall_best_height}")
        # <<< Return position AND dimensions >>>
        return best_x, best_y, overall_best_width, overall_best_height
    else:
        print(
            f"INFO -- No suitable opportunity window found for {tech} after scanning (original and rotated)."
        )
        return None


def calculate_window_score(window_grid, tech):
    """
    Calculates a score for a given window based on supercharged and empty slots,
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
                    if window_grid.width > 1 and (x == 0 or x == window_grid.width - 1):
                        edge_penalty += 1
                if cell["module"] is None:
                    empty_count += 1

    if supercharged_count > 0:
        return supercharged_count * 3  # + (empty_count * 1)
    else:
        return (supercharged_count * 3) + (empty_count * 1) + (edge_penalty * 0.25)


def create_localized_grid(
    grid, opportunity_x, opportunity_y, tech, localized_width, localized_height
):
    """
    Creates a localized grid around a given opportunity, ensuring it stays within
    the bounds of the main grid and preserves modules of other tech types.
    Uses the provided dimensions for the localized grid size.

    Args:
        grid (Grid): The main grid.
        opportunity_x (int): The x-coordinate of the opportunity (top-left corner).
        opportunity_y (int): The y-coordinate of the opportunity (top-left corner).
        tech (str): The technology type being optimized.
        localized_width (int): The desired width of the localized window. # <<< New
        localized_height (int): The desired height of the localized window. # <<< New

    Returns:
        tuple: A tuple containing:
            - localized_grid (Grid): The localized grid.
            - start_x (int): The starting x-coordinate of the localized grid in the main grid.
            - start_y (int): The starting y-coordinate of the localized grid in the main grid.
    """
    # <<< Remove hardcoded dimensions >>>
    # localized_width = 4
    # localized_height = 3

    # Directly use opportunity_x and opportunity_y as the starting position
    start_x = opportunity_x
    start_y = opportunity_y

    # Clamp the starting position to ensure it's within the grid bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    # Calculate the end position based on the clamped start position and desired dimensions
    end_x = min(grid.width, start_x + localized_width)
    end_y = min(grid.height, start_y + localized_height)

    # Adjust the localized grid size based on the clamped bounds
    actual_localized_width = end_x - start_x
    actual_localized_height = end_y - start_y

    # Create the localized grid with the actual calculated size
    localized_grid = Grid(actual_localized_width, actual_localized_height)

    # Copy the grid structure and module data (logic remains the same)
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
                if "module_position" in cell:
                    localized_grid.cells[localized_y][localized_x][
                        "module_position"
                    ] = cell["module_position"]

    return localized_grid, start_x, start_y


# --- NEW ML-Specific Function ---
def create_localized_grid_ml(
    grid, opportunity_x, opportunity_y, tech, localized_width, localized_height
):
    """
    Creates a localized grid around a given opportunity for ML processing,
    using the provided dimensions.

    Modules of *other* tech types within the localized area are temporarily
    removed, and their corresponding cells in the localized grid are marked
    as inactive. The original state of these modified cells (from the main grid)
    is stored for later restoration.

    Args:
        grid (Grid): The main grid.
        opportunity_x (int): The x-coordinate of the opportunity (top-left corner).
        opportunity_y (int): The y-coordinate of the opportunity (top-left corner).
        tech (str): The technology type being optimized (modules of this tech are kept).
        localized_width (int): The desired width of the localized window. # <<< New
        localized_height (int): The desired height of the localized window. # <<< New

    Returns:
        tuple: A tuple containing:
            - localized_grid (Grid): The localized grid prepared for ML.
            - start_x (int): The starting x-coordinate of the localized grid in the main grid.
            - start_y (int): The starting y-coordinate of the localized grid in the main grid.
            - original_state_map (dict): A dictionary mapping main grid coordinates
                                         (x, y) to their original cell data for cells
                                         that were modified (other tech removed).
    """
    # <<< Remove hardcoded dimensions >>>
    # localized_width = 4
    # localized_height = 3

    # Directly use opportunity_x and opportunity_y as the starting position
    start_x = opportunity_x
    start_y = opportunity_y

    # Clamp the starting position to ensure it's within the grid bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    # Calculate the end position based on the clamped start position and desired dimensions
    end_x = min(grid.width, start_x + localized_width)
    end_y = min(grid.height, start_y + localized_height)

    # Adjust the localized grid size based on the clamped bounds
    actual_localized_width = end_x - start_x
    actual_localized_height = end_y - start_y

    # Create the localized grid with the actual calculated size
    localized_grid = Grid(actual_localized_width, actual_localized_height)
    original_state_map = {}  # To store original state of modified cells

    # Copy the grid structure and module data, modifying for other techs (logic remains the same)
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
                # --- Inactive Cell in Main Grid ---
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
                local_cell.update(deepcopy(main_cell))
                local_cell["active"] = True

    return localized_grid, start_x, start_y, original_state_map


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
        player_owned_rewards = []  # Ensure it's an empty list if None

    tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)

    if tech_modules is None:
        print(
            f"Warning: check_all_modules_placed (opt_alg) - Could not get expected modules for {ship}/{tech}. Assuming not all modules are placed."
        )
        return False  # If modules for the tech couldn't be retrieved, assume they aren't all placed.

    if not tech_modules:  # Handles empty list case (no modules defined for this tech)
        return True  # All zero modules are considered placed.

    placed_module_ids = set()
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get_cell(x, y)
            if cell["tech"] == tech and cell["module"]:
                placed_module_ids.add(cell["module"])

    all_module_ids = {module["id"] for module in tech_modules}
    return placed_module_ids == all_module_ids
