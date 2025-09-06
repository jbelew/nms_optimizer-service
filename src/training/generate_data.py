# training/generate_data.py
import random
import numpy as np  # type: ignore
from copy import deepcopy  # Import deepcopy
import sys
import os
import time
import argparse
import uuid

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
from grid_utils import Grid
from modules_utils import get_tech_modules_for_training
from optimization.training import refine_placement_for_training
from optimization.helpers import determine_window_dimensions
from optimization.windowing import _scan_grid_with_window, calculate_window_score
from optimization.refinement import simulated_annealing
from pattern_matching import get_all_unique_pattern_variations
from bonus_calculations import (
    calculate_grid_score,
)
from data_definitions.modules import modules
from data_definitions.solves import solves
from grid_display import print_grid
from module_placement import place_module

# --- Configuration for Data Storage ---
GENERATED_BATCH_DIR = "generated_batches"
# --- End Configuration ---


# --- Data Generation Function (Saves Unique Batches) ---
def generate_training_batch(
    num_samples,
    # <<< Remove grid_width, grid_height as direct parameters >>>
    # grid_width,
    # grid_height,
    max_supercharged,
    max_inactive_cells,
    ship,
    tech,
    base_output_dir,  # <<< Renamed for clarity
    solve_map_prob=0.0,
    experimental: bool = False,  # <<< Renamed flag
):
    """
    Generates a batch of training data and saves it into ship/tech subdirectories.
    Uses dynamic window sizing based on the number of modules for the tech.
    Optionally uses pre-defined solve maps variations. Includes mirrored data augmentation.
    Always uses solve map if num_supercharged is 0 and a map exists.
    Ensures solve map variations fit the active cells of the grid.

    Args:
        num_samples (int): Number of original samples to generate (before augmentation).
        max_supercharged (int): Maximum number of supercharged cells to add.
        max_inactive_cells (int): Maximum number of inactive cells to add.
        ship (str): Ship type key.
        tech (str): Technology key.
        base_output_dir (str): Base directory to save the generated .npz file (e.g., 'generated_batches').
        solve_map_prob (float): Probability (0.0-1.0) of using a pre-defined solve map variation
                                (only applies if num_supercharged > 0).
        experimental (bool): If True, applies experimental data generation logic.
                             Currently, this means only generating 4x3 data for pulse tech if SC config meets criteria.

    Returns:
        tuple: (generated_count, num_output_classes, saved_filepath) or (0, 0, None) on failure.
               generated_count reflects total count including augmentations.
    """  # <<< Corrected docstring indentation
    start_time_tech = time.time()

    tech_modules = get_tech_modules_for_training(modules, ship, tech)
    if not tech_modules:
        print(
            f"Error: No tech modules found for ship='{ship}', tech='{tech}'. Cannot determine grid size or generate data."
        )
        return 0, 0, None
    module_count = len(tech_modules)

    # --- Initial default grid dimensions (might be overridden by experimental logic) ---
    # This call uses the *production* logic of determine_window_dimensions
    default_grid_width, default_grid_height = determine_window_dimensions(
        module_count, tech
    )

    # --- End Determine Dynamic Grid Dimensions ---

    # --- Construct specific output directory ---
    tech_output_dir = os.path.join(base_output_dir, ship, tech)
    # --- End Construct specific output directory ---

    # --- 1. Generate Samples ---
    new_X_supercharge_list = []
    new_X_inactive_mask_list = []
    new_y_list = []

    tech_modules.sort(key=lambda m: m["id"])
    module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(tech_modules)}
    num_output_classes = len(tech_modules) + 1  # <<< Use fetched tech_modules

    # --- Check if Solve Map Exists and Generate Variations ---
    solve_map_exists = False
    solve_map_variations = []

    # This variable will hold the actual grid dimensions for the current sample
    # It's determined after the experimental filter.
    # Initialize here to avoid undefined variable errors if loops are skipped.
    current_sample_grid_w = default_grid_width
    current_sample_grid_h = default_grid_height

    if ship in solves and tech in solves[ship]:
        original_pattern = solves[ship][tech].get("map")
        if original_pattern:
            solve_map_exists = True
            # Convert string keys like "(0, 0)" to tuples if necessary
            pattern_tuple_keys = {}
            for k, v in original_pattern.items():
                try:
                    # Ensure keys are tuples, handle potential string format from storage
                    coord_tuple = eval(k) if isinstance(k, str) else k
                    if isinstance(coord_tuple, tuple) and len(coord_tuple) == 2:
                        # <<< Check if pattern fits within dynamic grid dimensions >>>
                        px, py = (
                            coord_tuple  # Use default_grid_width/height for initial solve map check
                        )
                        if (
                            0 <= px < default_grid_width
                            and 0 <= py < default_grid_height
                        ):
                            pattern_tuple_keys[coord_tuple] = v
                        else:
                            print(
                                f"Warning: Solve map key {k} is outside default grid dimensions ({default_grid_width}x{default_grid_height}). Skipping key."
                            )
                            # If any key is outside, the original pattern itself doesn't fit
                            pattern_tuple_keys = {}  # Clear it to prevent use
                            break  # No need to check other keys for this pattern
                    else:
                        print(f"Warning: Skipping invalid pattern key format: {k}")
                except Exception as e:
                    print(
                        f"Warning: Skipping invalid pattern key format: {k} due to error: {e}"
                    )

            if pattern_tuple_keys:  # Only generate variations if the pattern fits
                solve_map_variations = get_all_unique_pattern_variations(
                    pattern_tuple_keys
                )
                if not solve_map_variations:
                    print(
                        f"Warning: Solve map exists for {ship}/{tech} and fits, but generated no variations."
                    )
                    solve_map_exists = False  # Treat as non-existent if no variations
            else:
                # Pattern didn't fit or had invalid keys
                print(
                    f"Warning: Solve map exists for {ship}/{tech}, but pattern doesn't fit dynamic grid or has invalid keys."
                )
                solve_map_exists = False  # Treat as non-existent
    # --- End Solve Map Check ---

    original_generated_count = 0
    attempt_count = 0
    max_attempts = num_samples * 15 + 100  # Keep max attempts logic

    while original_generated_count < num_samples and attempt_count < max_attempts:
        attempt_count += 1

        # --- Determine grid dimensions for THIS sample ---
        if experimental:
            current_sample_grid_w, current_sample_grid_h = 4, 3
            # print(f"DEBUG: Experimental mode active. Grid set to 4x3 for tech '{tech}'.")
        else:
            current_sample_grid_w, current_sample_grid_h = (
                default_grid_width,
                default_grid_height,
            )

        # Create the grid for the current sample with its determined dimensions
        original_grid_layout = Grid(current_sample_grid_w, current_sample_grid_h)
        inactive_positions_set = set()

        # Populate SC/inactive on original_grid_layout (using current_sample_grid_w/h)
        total_cells_current_sample = current_sample_grid_w * current_sample_grid_h
        all_positions_current_sample = [
            (x, y)
            for y in range(current_sample_grid_h)
            for x in range(current_sample_grid_w)
        ]

        num_inactive = random.randint(
            0, min(max_inactive_cells, total_cells_current_sample)
        )
        if num_inactive > 0 and num_inactive <= len(all_positions_current_sample):
            inactive_positions = random.sample(
                all_positions_current_sample, num_inactive
            )
            inactive_positions_set = set(inactive_positions)
            for x, y in inactive_positions:
                original_grid_layout.set_active(x, y, False)

        active_positions_current_sample = [
            pos
            for pos in all_positions_current_sample
            if pos not in inactive_positions_set
        ]
        num_active_cells_current_sample = len(active_positions_current_sample)
        if num_active_cells_current_sample == 0:
            continue

        supercharged_positions_set = set()
        max_possible_supercharged = min(
            max_supercharged, num_active_cells_current_sample
        )
        num_supercharged_actual = random.randint(0, max_possible_supercharged)
        if num_supercharged_actual > 0 and num_supercharged_actual <= len(
            active_positions_current_sample
        ):
            supercharged_positions = random.sample(
                active_positions_current_sample, num_supercharged_actual
            )
            supercharged_positions_set = set(supercharged_positions)
            for x, y in supercharged_positions:
                original_grid_layout.set_supercharged(x, y, True)

        # --- Early Skip for Experimental with Zero SC Slots ---
        if experimental and num_supercharged_actual == 0:
            # print(f"DEBUG: Experimental mode and 0 SC slots for tech '{tech}'. Skipping this attempt.")
            continue  # Skip this attempt entirely, don't even run the experimental filter below
        # --- End Early Skip ---

        print_grid(
            original_grid_layout
        )  # Optional: for debugging the layout before filters

        # --- Experimental 4x3 Data Generation Filter ---
        # This filter runs if 'experimental' is true, which also means current_sample_grid_w/h will be 4x3.
        if experimental:  # By this point, current_sample_grid_w and current_sample_grid_h are 4,3 if experimental is true
            # This filter's purpose is to generate 4x3 data only if its raw SC score is "better"
            # than what could be achieved on the tech's standard (often smaller) grid.
            # If the tech's standard grid IS ALREADY 4x3 (i.e., default_grid_width/height match current_sample_grid_w/h),
            # this specific comparison is not meaningful and can be problematic if _scan_grid_with_window
            # returns an unexpected value (like -1.0) when the scan window equals the grid size.
            if (default_grid_width, default_grid_height) != (
                current_sample_grid_w,
                current_sample_grid_h,
            ):
                print(
                    f"\nDEBUG_EXPERIMENTAL_FILTER (Tech: {tech}, Attempt: {attempt_count}, Experimental Flag: True):"
                )
                print(f"  - Module Count: {module_count}")
                print(
                    f"  - Current Sample Grid (Experimental): {current_sample_grid_w}x{current_sample_grid_h}"
                )
                print(
                    f"  - Default Grid for Comparison (Standard for this tech): {default_grid_width}x{default_grid_height}"
                )
                print(f"  - SC slots in 4x3: {supercharged_positions_set}")
                print(f"  - Inactive slots in 4x3: {inactive_positions_set}")

                # `original_grid_layout` is 4x3 here, with SC slots populated.

                # Calculate the raw score for the full 4x3 experimental layout
                # This score is based purely on SC/empty/edge, ignoring if all modules fit.
                raw_score_4x3 = calculate_window_score(original_grid_layout, tech)
                print(f"  - Raw Score for 4x3 Layout: {raw_score_4x3:.4f}")

                # Scan the 4x3 layout to find the best raw score for a default-sized window (e.g., 4x2) within it.
                best_default_slice_raw_score, best_default_slice_pos = (
                    _scan_grid_with_window(
                        deepcopy(original_grid_layout),
                        default_grid_width,
                        default_grid_height,
                        module_count,
                        tech,
                    )
                )
                print(
                    f"  - Best Raw Score for {default_grid_width}x{default_grid_height} slice within 4x3: {best_default_slice_raw_score:.4f} (at pos: {best_default_slice_pos})"
                )

                # Now compare the raw scores: Generate sample only if 4x3 raw score is strictly better
                if not (raw_score_4x3 > best_default_slice_raw_score):
                    print(
                        f"  - Decision: SKIP 4x3. Raw score 4x3 ({raw_score_4x3:.4f}) is NOT > best default slice raw score ({best_default_slice_raw_score:.4f})."
                    )
                    continue  # Skip this sample
                print(
                    f"  - Decision: KEEP 4x3. Raw score 4x3 ({raw_score_4x3:.4f}) > best default slice raw score ({best_default_slice_raw_score:.4f}). Proceeding with 4x3 data gen."
                )
            else:
                # Experimental mode is on, AND the default grid size for this tech IS 4x3.
                # The comparative filter is not needed/meaningful in this case. Proceed with 4x3 generation.
                print(
                    f"\nDEBUG_EXPERIMENTAL_INFO (Tech: {tech}, Attempt: {attempt_count}, Experimental Flag: True):"
                )
                print(
                    f"  - Current Sample Grid (Experimental): {current_sample_grid_w}x{current_sample_grid_h}"
                )
                print(
                    f"  - Default Grid for this tech IS ALSO {default_grid_width}x{default_grid_height}."
                )
                print(
                    f"  - Skipping comparative raw score filter. Proceeding with 4x3 data generation as per experimental flag."
                )

        # SC and inactive cells are already populated on original_grid_layout by the block above.
        # The inactive_positions_set and supercharged_positions_set are correctly defined.
        # --- Decide: Use Solve Map Variation or Optimize ---
        if experimental:
            # If experimental mode is on AND there are no supercharged slots,
            # always skip using the solve map for this sample.
            if num_supercharged_actual == 0:
                use_solve_map = False
                # print(f"DEBUG: Experimental mode with 0 SC slots. Forcing NO solve map for this sample.")
            else:  # Experimental mode, but SC slots > 0
                # Probabilistic use of solve map
                use_solve_map = solve_map_exists and random.random() < solve_map_prob
                # if use_solve_map:
                #     print(f"DEBUG: Experimental mode, >0 SC. Probabilistically chose solve map (prob: {solve_map_prob}).")
        else:  # Not experimental
            if num_supercharged_actual == 0 and solve_map_exists:
                # If not experimental, 0 SC slots, and a map exists, always use it.
                use_solve_map = True
                # print(f"DEBUG: Not experimental, 0 SC slots, map exists. Forcing solve map.")
            else:  # Not experimental, and (num_supercharged_actual > 0 or not solve_map_exists)
                # Probabilistic use of solve map
                use_solve_map = solve_map_exists and random.random() < solve_map_prob
                # if use_solve_map:
                #     print(f"DEBUG: Not experimental, >0 SC or no map. Probabilistically chose solve map (prob: {solve_map_prob}).")
        # --- End Decision ---
        # --- Create Input and Output Matrices ---
        # <<< Use dynamic grid dimensions >>>
        input_supercharge_np = np.zeros(
            (current_sample_grid_h, current_sample_grid_w), dtype=np.int8
        )
        input_inactive_mask_np = np.zeros(
            (current_sample_grid_h, current_sample_grid_w), dtype=np.int8
        )
        output_matrix = np.zeros(
            (current_sample_grid_h, current_sample_grid_w), dtype=np.int8
        )
        sample_valid = True  # Assume valid initially

        # Create Input Matrices
        # <<< Use dynamic grid dimensions >>>
        for y in range(current_sample_grid_h):
            for x in range(current_sample_grid_w):
                pos = (x, y)
                is_active = pos not in inactive_positions_set
                is_supercharged = pos in supercharged_positions_set
                input_supercharge_np[y, x] = int(is_active and is_supercharged)
                input_inactive_mask_np[y, x] = int(
                    not is_active
                )  # 1 if inactive, 0 if active

        # Create Output Matrix
        if use_solve_map:
            if not solve_map_variations:
                print(
                    f"ERROR: Trying to use solve map for {ship}/{tech}, but variations list is empty. Skipping sample."
                )
                sample_valid = False
                continue  # Skip this attempt

            # --- Try to find a fitting variation ---
            fitting_variation_found = False
            tried_variations_indices = set()
            max_variation_attempts = len(
                solve_map_variations
            )  # Try all variations once
            current_pattern_data = {}  # Initialize to an empty dict

            for _ in range(max_variation_attempts):
                available_indices = [
                    i
                    for i, _ in enumerate(solve_map_variations)
                    if i not in tried_variations_indices
                ]
                if not available_indices:
                    break
                chosen_variation_index = random.choice(available_indices)
                tried_variations_indices.add(chosen_variation_index)
                current_pattern_data = solve_map_variations[chosen_variation_index]

                # Check if this variation fits the current grid's active cells
                pattern_fits = True
                for (px, py), module_id in current_pattern_data.items():
                    if module_id is None or module_id == "None":
                        continue

                    # Bounds check (already done when creating variations, but safe)
                    if not (
                        0 <= px < current_sample_grid_w
                        and 0 <= py < current_sample_grid_h
                    ):
                        pattern_fits = False
                        break

                    # Check if the target cell in the original grid is active
                    if not original_grid_layout.get_cell(px, py)["active"]:
                        pattern_fits = False
                        break
                if pattern_fits:
                    fitting_variation_found = True
                    # <<< Use dynamic grid dimensions >>>
                    for y in range(
                        current_sample_grid_h
                    ):  # <<< Use current sample dimensions
                        for x in range(
                            current_sample_grid_w
                        ):  # <<< Use current sample dimensions
                            if not original_grid_layout.get_cell(x, y)["active"]:
                                output_matrix[y, x] = 0
                                continue
                            module_id = current_pattern_data.get((x, y))
                            if module_id is None or module_id == "None":
                                output_matrix[y, x] = 0
                            else:
                                mapped_class = module_id_mapping.get(module_id)
                                if mapped_class is None:
                                    print(
                                        f"\nWarning: Module ID '{module_id}' from FITTING solve map variation not found. Skipping sample."
                                    )
                                    sample_valid = False
                                    break
                                output_matrix[y, x] = mapped_class
                        if not sample_valid:
                            break
                    break  # Found fitting variation

            if not fitting_variation_found:
                use_solve_map = False  # Fallback to optimization
                sample_valid = True  # Reset validity
            elif not sample_valid:
                continue  # Skip sample if mapping failed

            # --- Optional: Add print_grid for solve map case ---
            if fitting_variation_found and sample_valid:
                print(
                    f"\n--- Using Fitting Solve Map Variation for Sample {original_generated_count + 1} (Tech: {tech}) ---"
                )
                # <<< Use dynamic grid dimensions >>>
                temp_display_grid = Grid(current_sample_grid_w, current_sample_grid_h)
                tech_module_defs_map = {m["id"]: m for m in tech_modules}
                # <<< Use dynamic grid dimensions >>>
                for y_disp in range(current_sample_grid_h):
                    for x_disp in range(current_sample_grid_w):
                        temp_display_grid.set_active(
                            x_disp,
                            y_disp,
                            original_grid_layout.get_cell(x_disp, y_disp)["active"],
                        )
                        temp_display_grid.set_supercharged(
                            x_disp,
                            y_disp,
                            original_grid_layout.get_cell(x_disp, y_disp)[
                                "supercharged"
                            ],
                        )
                        module_id_disp = current_pattern_data.get((x_disp, y_disp))

                        # --- Start of Corrected Block ---
                        if (
                            module_id_disp
                            and module_id_disp != "None"
                            and temp_display_grid.get_cell(x_disp, y_disp)["active"]
                        ):
                            module_data = tech_module_defs_map.get(module_id_disp)
                            if module_data:
                                try:
                                    # Explicitly pass arguments expected by place_module
                                    # Ensure these keyword names match your place_module signature
                                    place_module(
                                        temp_display_grid,
                                        x_disp,
                                        y_disp,
                                        module_id=module_data.get("id"),
                                        label=module_data.get("label"),  # Added label
                                        tech=module_data.get(
                                            "tech"
                                        ),  # Use module_tech if that's the param name
                                        module_type=module_data.get(
                                            "type"
                                        ),  # Added module_type
                                        bonus=module_data.get(
                                            "bonus"
                                        ),  # Use module_bonus if that's the param name
                                        adjacency=module_data.get(
                                            "adjacency"
                                        ),  # Use module_adjacency if that's the param name
                                        sc_eligible=module_data.get(
                                            "sc_eligible"
                                        ),  # Added sc_eligible
                                        image=module_data.get("image"),  # Added image
                                    )
                                except TypeError as te:
                                    # Catch signature mismatches specifically
                                    print(
                                        f"Warning: Error placing module {module_id_disp} for display. TypeError: {te}. Check place_module signature and arguments passed."
                                    )
                                except Exception as e:
                                    print(
                                        f"Warning: Error placing module {module_id_disp} for display: {e}"
                                    )
                            else:
                                print(
                                    f"Warning: Module data for ID '{module_id_disp}' not found for display."
                                )
                        else:
                            # Ensure cells without modules are cleared properly for display grid
                            temp_display_grid.set_module(x_disp, y_disp, None)
                            temp_display_grid.set_tech(x_disp, y_disp, None)
                        # --- End of Corrected Block ---

                display_score = calculate_grid_score(temp_display_grid, tech)
                print(f"Solve Map Variation Score: {display_score:.4f}")
                print_grid(temp_display_grid)
            # --- End print_grid ---

        # --- Optimization Path ---
        if not use_solve_map:
            optimized_grid = None
            best_bonus = -1.0

            try:
                # module_count is len(tech_modules), tech_modules is already fetched
                if module_count < 8:
                    print(
                        f"INFO -- DataGen ({tech}): {module_count} modules < 9. Using refine_placement_for_training (brute-force)."
                    )
                    # refine_placement_for_training will use brute-force for <10 modules
                    optimized_grid, best_bonus = refine_placement_for_training(
                        original_grid_layout,
                        ship,
                        modules,
                        tech,  # <<< Pass modules from modules_for_training.py
                    )
                else:  # module_count >= 10
                    print(
                        f"INFO -- DataGen ({tech}): {module_count} modules >= 9. Using direct Simulated Annealing for ground truth."
                    )
                    # Use robust SA parameters, similar to those in refine_placement_for_training's internal SA call
                    # when it handles >=10 modules.
                    sa_params_for_ground_truth = {
                        "initial_temperature": 5000,
                        "cooling_rate": 0.999,
                        "stopping_temperature": 0.1,
                        "iterations_per_temp": 35,
                        "initial_swap_probability": 0.55,
                        "final_swap_probability": 0.25,
                        "start_from_current_grid": False,
                        "max_processing_time": 600.0,
                    }
                    # Ensure 'modules' (modules_for_training.modules) is passed as modules_data_dict
                    # and tech_modules (the list of module dicts) is passed as tech_modules_list_override
                    optimized_grid, sa_score = simulated_annealing(
                        original_grid_layout,
                        ship,
                        modules,  # <<< Pass modules from modules_for_training.py
                        tech,
                        player_owned_rewards=None,  # Not needed when overriding modules
                        **sa_params_for_ground_truth,
                    )
                    best_bonus = sa_score  # Assign the score from SA

                if optimized_grid is None:
                    sample_valid = False
            except ValueError as ve:
                print(
                    f"\nOptimization failed for attempt {attempt_count} with ValueError: {ve}. Skipping."
                )
                sample_valid = False
            except Exception as e:
                print(
                    f"\nUnexpected error during optimization for attempt {attempt_count}: {e}"
                )
                sample_valid = False

            if sample_valid and optimized_grid is not None:
                print(
                    f"\n--- Using Optimized Layout for Sample {original_generated_count + 1} (Tech: {tech}) ---"
                )
                print(f"Optimized Score: {best_bonus:.4f}")
                print_grid(optimized_grid)
                # <<< Use dynamic grid dimensions >>>
                for y in range(current_sample_grid_h):
                    for x in range(
                        current_sample_grid_w
                    ):  # <<< Corrected loop variable
                        if not original_grid_layout.get_cell(x, y)["active"]:
                            output_matrix[y, x] = 0
                            continue
                        cell_data = optimized_grid.get_cell(x, y)
                        module_id = cell_data["module"]
                        if module_id is None:
                            output_matrix[y, x] = 0
                        else:
                            mapped_class = module_id_mapping.get(module_id)
                            if mapped_class is None:
                                print(
                                    f"\nWarning: Module ID '{module_id}' from OPTIMIZED grid not found. Skipping sample."
                                )
                                sample_valid = False
                                break
                            output_matrix[y, x] = mapped_class
                    if not sample_valid:
                        continue
            elif sample_valid and optimized_grid is None:
                sample_valid = False
        # --- End Optimization Path ---

        # --- Add Valid Sample AND AUGMENTATIONS ---
        if sample_valid:
            # (Augmentation logic remains the same)
            new_X_supercharge_list.append(input_supercharge_np)
            new_X_inactive_mask_list.append(input_inactive_mask_np)
            new_y_list.append(output_matrix)
            new_X_supercharge_list.append(np.fliplr(input_supercharge_np))
            new_X_inactive_mask_list.append(np.fliplr(input_inactive_mask_np))
            new_y_list.append(np.fliplr(output_matrix))
            new_X_supercharge_list.append(np.flipud(input_supercharge_np))
            new_X_inactive_mask_list.append(np.flipud(input_inactive_mask_np))
            new_y_list.append(np.flipud(output_matrix))
            new_X_supercharge_list.append(np.flipud(np.fliplr(input_supercharge_np)))
            new_X_inactive_mask_list.append(
                np.flipud(np.fliplr(input_inactive_mask_np))
            )
            new_y_list.append(np.flipud(np.fliplr(output_matrix)))

            original_generated_count += 1
            total_samples_in_list = len(new_y_list)
            if (
                original_generated_count % 1 == 0
                or original_generated_count == num_samples
            ):
                print(
                    f"-- Generated {original_generated_count}/{num_samples} original samples for tech '{tech}' (Attempt {attempt_count}, Total in list: {total_samples_in_list}) --"
                )
        # --- End Add Sample ---

    # --- Final Count and Warnings ---
    final_generated_count = len(new_y_list)
    if original_generated_count < num_samples:
        print(
            f"\nWarning: Only generated {original_generated_count}/{num_samples} *original* samples after {max_attempts} attempts for tech '{tech}'. Final count with augmentation: {final_generated_count}."
        )

    if not new_y_list:
        print("No valid samples generated for this batch. Skipping save.")
        return 0, num_output_classes, None

    # --- Convert to NumPy Arrays ---
    try:
        new_X_supercharge_np = np.array(new_X_supercharge_list, dtype=np.int8)
        new_X_inactive_mask_np = np.array(new_X_inactive_mask_list, dtype=np.int8)
        new_y_np = np.array(new_y_list, dtype=np.int8)
    except Exception as e:
        print(f"Error converting lists to NumPy arrays: {e}")
        return 0, num_output_classes, None

    # --- Save Batch ---
    saved_filepath = None
    if new_y_np.size > 0:
        try:
            os.makedirs(tech_output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            # <<< Use dynamic grid dimensions in filename >>>
            filename = f"data_{ship}_{tech}_{current_sample_grid_w}x{current_sample_grid_h}_{timestamp}_{unique_id}.npz"
            saved_filepath = os.path.join(tech_output_dir, filename)
            print(
                f"Saving {len(new_y_np)} generated samples (incl. augmentations) to {saved_filepath}..."
            )
            np.savez_compressed(
                saved_filepath,
                X_supercharge=new_X_supercharge_np,
                X_inactive_mask=new_X_inactive_mask_np,
                y=new_y_np,
            )
            print("Save complete.")
        except Exception as e:
            print(f"Error saving data batch: {e}")
            saved_filepath = None
            final_generated_count = 0
    else:
        print("No data generated in this batch, skipping save.")
        final_generated_count = 0

    elapsed_time_tech = time.time() - start_time_tech
    print(
        f"Data generation process finished for tech '{tech}'. Generated: {final_generated_count} (incl. augmentations). Time: {elapsed_time_tech:.2f}s"
    )

    return final_generated_count, num_output_classes, saved_filepath


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a batch of training data for NMS Optimizer."
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Tech category (used if --tech is not specified).",
        metavar="CATEGORY_NAME",
    )
    parser.add_argument(
        "--tech",
        type=str,
        default=None,
        help="Specific tech key to process (optional). If provided, only this tech is processed.",
        metavar="TECH_KEY",
    )
    parser.add_argument(
        "--ship", type=str, default="standard", help="Ship type.", metavar="SHIP_TYPE"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Original samples per tech (before augmentation).",
        metavar="NUM_SAMPLES",
    )
    parser.add_argument(
        "--max_sc", type=int, default=4, help="Max supercharged.", metavar="MAX_SC"
    )
    parser.add_argument(
        "--max_inactive",
        type=int,
        default=0,
        help="Max inactive.",
        metavar="MAX_INACTIVE",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=GENERATED_BATCH_DIR,
        help="Base output directory (ship/tech subdirs will be created).",
        metavar="DIR_PATH",
    )
    parser.add_argument(
        "--solve_prob",
        type=float,
        default=0.0,
        help="Probability (0.0-1.0) of using solve map (if supercharged > 0).",
        metavar="PROB",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="If set, applies experimental data generation logic (e.g., for pulse 4x3).",
    )
    args = parser.parse_args()
    if not 0.0 <= args.solve_prob <= 1.0:
        parser.error("--solve_prob must be between 0.0 and 1.0")

    config = {
        "num_samples_per_tech": args.samples,
        "max_supercharged": args.max_sc,
        "max_inactive_cells": args.max_inactive,
        "ship": args.ship,
        "tech_category_to_process": args.category,
        "specific_tech_to_process": args.tech,
        "output_dir": args.output_dir,
        "solve_map_prob": args.solve_prob,
        "experimental": args.experimental,  # <<< Updated in config
    }

    start_time_all = time.time()
    print(f"Starting data batch generation process...")
    print(f"Configuration: {config}")

    # (Logic to determine tech_keys_to_process remains the same)
    tech_keys_to_process = []
    try:
        if config["specific_tech_to_process"]:
            tech_keys_to_process = [config["specific_tech_to_process"]]
            print(f"Processing specific tech: {tech_keys_to_process[0]}")
        else:
            ship_data = modules.get(config["ship"])
            if (
                not ship_data
                or "types" not in ship_data
                or not isinstance(ship_data["types"], dict)
            ):
                raise KeyError(
                    f"Ship '{config['ship']}' or its 'types' dictionary not found/invalid."
                )
            category_data = ship_data["types"].get(config["tech_category_to_process"])
            if not category_data or not isinstance(category_data, list):
                raise KeyError(
                    f"Category '{config['tech_category_to_process']}' not found or invalid for ship '{config['ship']}'."
                )
            tech_keys_to_process = [
                t["key"] for t in category_data if isinstance(t, dict) and "key" in t
            ]
            if not tech_keys_to_process:
                raise ValueError(
                    f"No valid tech keys found for ship '{config['ship']}', category '{config['tech_category_to_process']}'."
                )
            print(
                f"Planning to generate data for techs in category '{config['tech_category_to_process']}': {tech_keys_to_process}"
            )
    except (KeyError, ValueError) as e:
        print(f"Error determining techs to process: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while getting tech keys: {e}")
        exit()

    total_samples_generated_overall = 0
    for tech in tech_keys_to_process:
        print(f"\n{'=' * 10} Processing Tech: {tech} {'=' * 10}")
        # <<< Update function call to remove width/height >>>
        generated_count, _, _ = generate_training_batch(
            num_samples=config["num_samples_per_tech"],
            # grid_width=config["grid_width"], # Removed
            # grid_height=config["grid_height"], # Removed
            max_supercharged=config["max_supercharged"],
            max_inactive_cells=config["max_inactive_cells"],
            ship=config["ship"],
            tech=tech,
            base_output_dir=config["output_dir"],
            solve_map_prob=config["solve_map_prob"],
            experimental=config["experimental"],  # <<< Pass updated flag
        )
        total_samples_generated_overall += generated_count

    end_time_all = time.time()
    print(f"\n{'=' * 20} Data Batch Generation Complete {'=' * 20}")
    print(
        f"Total samples generated across all techs (incl. augmentations) in this run: {total_samples_generated_overall}"
    )
    print(f"Total time: {end_time_all - start_time_all:.2f} seconds.")
    print(
        f"Generated files saved in subdirectories under: {os.path.abspath(config['output_dir'])}"
    )
