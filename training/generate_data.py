# training/generate_data.py
import random
import numpy as np
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
try:
    # Try importing Grid from optimizer first (if it exists there)
    from optimizer import Grid
except ImportError:
    # Fallback to importing Grid from grid_utils
    from grid_utils import Grid

# <<< Import determine_window_dimensions >>>
from modules_data import get_tech_modules_for_training
from optimization_algorithms import (
    get_all_unique_pattern_variations,
    refine_placement_for_training,
    calculate_pattern_adjacency_score, # <<< Import needed for print_grid logic
    determine_window_dimensions # <<< Import the dynamic sizing function
)
from bonus_calculations import calculate_grid_score # <<< Import needed for print_grid logic
from modules_for_training import modules, solves
from simulated_annealing import simulated_annealing
from grid_display import print_grid # <<< Import print_grid
from module_placement import place_module # <<< Import place_module

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
    base_output_dir, # <<< Renamed for clarity
    solve_map_prob=0.0,
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

    Returns:
        tuple: (generated_count, num_output_classes, saved_filepath) or (0, 0, None) on failure.
               generated_count reflects total count including augmentations.
    """
    start_time_tech = time.time()

    # --- Determine Dynamic Grid Dimensions ---
    tech_modules = get_tech_modules_for_training(modules, ship, tech)
    if not tech_modules:
        print(f"Error: No tech modules found for ship='{ship}', tech='{tech}'. Cannot determine grid size or generate data.")
        return 0, 0, None

    module_count = len(tech_modules)
    grid_width, grid_height = determine_window_dimensions(module_count, tech)
    print(f"INFO -- Determined dynamic grid size for {tech} ({module_count} modules): {grid_width}x{grid_height}")
    # --- End Determine Dynamic Grid Dimensions ---

    # --- Construct specific output directory ---
    tech_output_dir = os.path.join(base_output_dir, ship, tech)
    # --- End Construct specific output directory ---

    # --- 1. Generate Samples ---
    new_X_supercharge_list = []
    new_X_inactive_mask_list = []
    new_y_list = []

    # tech_modules = get_tech_modules_for_training(modules, ship, tech) # Already fetched above
    # if not tech_modules: # Already checked above
    #     print(f"Error: No tech modules found for ship='{ship}', tech='{tech}'. Cannot generate data.")
    #     return 0, 0, None

    tech_modules.sort(key=lambda m: m['id'])
    module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(tech_modules)}
    num_output_classes = len(tech_modules) + 1 # <<< Use fetched tech_modules

    # --- Check if Solve Map Exists and Generate Variations ---
    solve_map_exists = False
    solve_map_variations = []
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
                         px, py = coord_tuple
                         if 0 <= px < grid_width and 0 <= py < grid_height:
                             pattern_tuple_keys[coord_tuple] = v
                         else:
                             print(f"Warning: Solve map key {k} is outside dynamic grid dimensions ({grid_width}x{grid_height}). Skipping key.")
                             # If any key is outside, the original pattern itself doesn't fit
                             pattern_tuple_keys = {} # Clear it to prevent use
                             break # No need to check other keys for this pattern
                    else:
                        print(f"Warning: Skipping invalid pattern key format: {k}")
                except Exception as e:
                    print(f"Warning: Skipping invalid pattern key format: {k} due to error: {e}")

            if pattern_tuple_keys: # Only generate variations if the pattern fits
                solve_map_variations = get_all_unique_pattern_variations(pattern_tuple_keys)
                if not solve_map_variations:
                    print(f"Warning: Solve map exists for {ship}/{tech} and fits, but generated no variations.")
                    solve_map_exists = False # Treat as non-existent if no variations
            else:
                # Pattern didn't fit or had invalid keys
                print(f"Warning: Solve map exists for {ship}/{tech}, but pattern doesn't fit dynamic grid or has invalid keys.")
                solve_map_exists = False # Treat as non-existent
    # --- End Solve Map Check ---

    original_generated_count = 0
    attempt_count = 0
    max_attempts = num_samples * 15 + 100 # Keep max attempts logic

    # <<< Use dynamic grid dimensions >>>
    total_cells = grid_width * grid_height
    all_positions = [(x, y) for y in range(grid_height) for x in range(grid_width)]

    while original_generated_count < num_samples and attempt_count < max_attempts:
        attempt_count += 1
        # <<< Use dynamic grid dimensions >>>
        original_grid_layout = Grid(grid_width, grid_height) # <<< Keep track of the initial state
        inactive_positions_set = set()

        # --- Add Inactive/Supercharged Cells ---
        inactive_positions = []
        # <<< Use dynamic total_cells >>>
        num_inactive = random.randint(0, min(max_inactive_cells, total_cells))
        if num_inactive > 0:
            if num_inactive <= len(all_positions):
                inactive_positions = random.sample(all_positions, num_inactive)
                inactive_positions_set = set(inactive_positions)
                for x, y in inactive_positions:
                    original_grid_layout.set_active(x, y, False)
            else:
                # This case should be less likely now, but keep the warning
                print(f"Warning: Cannot sample {num_inactive} inactive positions from {len(all_positions)} total positions. Setting 0 inactive.")
                inactive_positions = []
                inactive_positions_set = set()

        active_positions = [pos for pos in all_positions if pos not in inactive_positions_set]
        num_active_cells = len(active_positions)
        if num_active_cells == 0: continue # Skip if no active cells

        supercharged_positions_set = set()
        # <<< Use dynamic num_active_cells >>>
        max_possible_supercharged = min(max_supercharged, num_active_cells)
        num_supercharged = random.randint(0, max_possible_supercharged) # <<< This is the key variable
        if num_supercharged > 0:
            # Ensure we don't try to sample more than available active positions
            if num_supercharged <= len(active_positions):
                supercharged_positions = random.sample(active_positions, num_supercharged)
                supercharged_positions_set = set(supercharged_positions)
                for x, y in supercharged_positions:
                    original_grid_layout.set_supercharged(x, y, True) # <<< Update original_grid_layout state
            else:
                 print(f"Warning: Cannot sample {num_supercharged} supercharged positions from {len(active_positions)} active positions. Setting 0 supercharged.")
                 num_supercharged = 0 # Correct the count
                 supercharged_positions_set = set()

        # --- End Add Cells ---

        # --- Decide: Use Solve Map Variation or Optimize ---
        # (Logic remains the same, but relies on updated solve_map_exists flag)
        if num_supercharged == 0 and solve_map_exists:
            use_solve_map = True
        else:
            use_solve_map = solve_map_exists and random.random() < solve_map_prob
        # --- End Decision ---

        # --- Create Input and Output Matrices ---
        # <<< Use dynamic grid dimensions >>>
        input_supercharge_np = np.zeros((grid_height, grid_width), dtype=np.int8)
        input_inactive_mask_np = np.zeros((grid_height, grid_width), dtype=np.int8)
        output_matrix = np.zeros((grid_height, grid_width), dtype=np.int8)
        sample_valid = True # Assume valid initially

        # Create Input Matrices
        # <<< Use dynamic grid dimensions >>>
        for y in range(grid_height):
            for x in range(grid_width):
                pos = (x, y)
                is_active = pos not in inactive_positions_set
                is_supercharged = pos in supercharged_positions_set
                input_supercharge_np[y, x] = int(is_active and is_supercharged)
                input_inactive_mask_np[y, x] = int(not is_active)

        # Create Output Matrix
        if use_solve_map:
            if not solve_map_variations:
                 print(f"ERROR: Trying to use solve map for {ship}/{tech}, but variations list is empty. Skipping sample.")
                 sample_valid = False
                 continue # Skip this attempt

            # --- Try to find a fitting variation ---
            fitting_variation_found = False
            tried_variations_indices = set()
            max_variation_attempts = len(solve_map_variations) # Try all variations once

            for _ in range(max_variation_attempts):
                available_indices = [i for i, _ in enumerate(solve_map_variations) if i not in tried_variations_indices]
                if not available_indices: break
                chosen_variation_index = random.choice(available_indices)
                tried_variations_indices.add(chosen_variation_index)
                current_pattern_data = solve_map_variations[chosen_variation_index]

                # Check if this variation fits the current grid's active cells
                pattern_fits = True
                for (px, py), module_id in current_pattern_data.items():
                    if module_id is None or module_id == "None": continue

                    # Bounds check (already done when creating variations, but safe)
                    if not (0 <= px < grid_width and 0 <= py < grid_height):
                        pattern_fits = False; break

                    # Check if the target cell in the original grid is active
                    if not original_grid_layout.get_cell(px, py)["active"]:
                        pattern_fits = False; break

                if pattern_fits:
                    fitting_variation_found = True
                    # <<< Use dynamic grid dimensions >>>
                    for y in range(grid_height):
                        for x in range(grid_width):
                            if not original_grid_layout.get_cell(x, y)["active"]:
                                output_matrix[y, x] = 0; continue
                            module_id = current_pattern_data.get((x, y))
                            if module_id is None or module_id == "None":
                                output_matrix[y, x] = 0
                            else:
                                mapped_class = module_id_mapping.get(module_id)
                                if mapped_class is None:
                                    print(f"\nWarning: Module ID '{module_id}' from FITTING solve map variation not found. Skipping sample.")
                                    sample_valid = False; break
                                output_matrix[y, x] = mapped_class
                        if not sample_valid: break
                    break # Found fitting variation

            if not fitting_variation_found:
                use_solve_map = False # Fallback to optimization
                sample_valid = True # Reset validity
            elif not sample_valid:
                continue # Skip sample if mapping failed

            # --- Optional: Add print_grid for solve map case ---
            if fitting_variation_found and sample_valid:
                print(f"\n--- Using Fitting Solve Map Variation for Sample {original_generated_count+1} (Tech: {tech}) ---")
                # <<< Use dynamic grid dimensions >>>
                temp_display_grid = Grid(grid_width, grid_height)
                tech_module_defs_map = {m['id']: m for m in tech_modules}
                # <<< Use dynamic grid dimensions >>>
                for y_disp in range(grid_height):
                    for x_disp in range(grid_width):
                        temp_display_grid.set_active(x_disp, y_disp, original_grid_layout.get_cell(x_disp, y_disp)["active"])
                        temp_display_grid.set_supercharged(x_disp, y_disp, original_grid_layout.get_cell(x_disp, y_disp)["supercharged"])
                        module_id_disp = current_pattern_data.get((x_disp, y_disp))

                        # --- Start of Corrected Block ---
                        if module_id_disp and module_id_disp != "None" and temp_display_grid.get_cell(x_disp, y_disp)["active"]:
                            module_data = tech_module_defs_map.get(module_id_disp)
                            if module_data:
                                try:
                                    # Explicitly pass arguments expected by place_module
                                    # Ensure these keyword names match your place_module signature
                                    place_module(
                                        temp_display_grid,
                                        x_disp,
                                        y_disp,
                                        module_id=module_data.get('id'),
                                        label=module_data.get('label'), # Added label
                                        tech=module_data.get('tech'), # Use module_tech if that's the param name
                                        module_type=module_data.get('type'), # Added module_type
                                        bonus=module_data.get('bonus'), # Use module_bonus if that's the param name
                                        adjacency=module_data.get('adjacency'), # Use module_adjacency if that's the param name
                                        sc_eligible=module_data.get('sc_eligible'), # Added sc_eligible
                                        image=module_data.get('image') # Added image
                                    )
                                except TypeError as te:
                                    # Catch signature mismatches specifically
                                    print(f"Warning: Error placing module {module_id_disp} for display. TypeError: {te}. Check place_module signature and arguments passed.")
                                except Exception as e:
                                    print(f"Warning: Error placing module {module_id_disp} for display: {e}")
                            else:
                                print(f"Warning: Module data for ID '{module_id_disp}' not found for display.")
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
            
            sa_params = {
                "initial_temperature": 5000,
                "cooling_rate": 0.999,
                "stopping_temperature": 0.1,
                "iterations_per_temp": 50,
                "initial_swap_probability": 0.6,
                "final_swap_probability": 0.1,
                "start_from_current_grid": False,
                "max_processing_time": 600.0
            }
            
            try:
                optimized_grid, optimized_score = simulated_annealing(
                    original_grid_layout, # Use the correctly prepared grid for this sample
                    ship,                 # Use the 'ship' variable
                    modules,
                    tech,
                    None,                 # Pass None for player_owned_rewards in this context
                    **sa_params
                )
                best_bonus = optimized_score # Update best_bonus with the score from SA
                
                # optimized_grid, best_bonus = refine_placement_for_training(
                #     original_grid_layout, ship, modules, tech
                # )
                
                if optimized_grid is None: sample_valid = False
            except ValueError as ve:
                 print(f"\nOptimization failed for attempt {attempt_count} with ValueError: {ve}. Skipping.")
                 sample_valid = False
            except Exception as e:
                print(f"\nUnexpected error during optimization for attempt {attempt_count}: {e}")
                sample_valid = False

            if sample_valid and optimized_grid is not None:
                print(f"\n--- Using Optimized Layout for Sample {original_generated_count+1} (Tech: {tech}) ---")
                print(f"Optimized Score: {best_bonus:.4f}")
                print_grid(optimized_grid)
                # <<< Use dynamic grid dimensions >>>
                for y in range(grid_height):
                    for x in range(grid_width):
                        if not original_grid_layout.get_cell(x, y)["active"]:
                             output_matrix[y, x] = 0; continue
                        cell_data = optimized_grid.get_cell(x, y)
                        module_id = cell_data["module"]
                        if module_id is None:
                            output_matrix[y, x] = 0
                        else:
                            mapped_class = module_id_mapping.get(module_id)
                            if mapped_class is None:
                                print(f"\nWarning: Module ID '{module_id}' from OPTIMIZED grid not found. Skipping sample.")
                                sample_valid = False; break
                            output_matrix[y, x] = mapped_class
                    if not sample_valid: continue
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
            new_X_inactive_mask_list.append(np.flipud(np.fliplr(input_inactive_mask_np)))
            new_y_list.append(np.flipud(np.fliplr(output_matrix)))

            original_generated_count += 1
            total_samples_in_list = len(new_y_list)
            if original_generated_count % 1 == 0 or original_generated_count == num_samples:
                 print(f"-- Generated {original_generated_count}/{num_samples} original samples for tech '{tech}' (Attempt {attempt_count}, Total in list: {total_samples_in_list}) --")
        # --- End Add Sample ---

    # --- Final Count and Warnings ---
    final_generated_count = len(new_y_list)
    if original_generated_count < num_samples:
        print(f"\nWarning: Only generated {original_generated_count}/{num_samples} *original* samples after {max_attempts} attempts for tech '{tech}'. Final count with augmentation: {final_generated_count}.")

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
            filename = f"data_{ship}_{tech}_{grid_width}x{grid_height}_{timestamp}_{unique_id}.npz"
            saved_filepath = os.path.join(tech_output_dir, filename)
            print(f"Saving {len(new_y_np)} generated samples (incl. augmentations) to {saved_filepath}...")
            np.savez_compressed(
                saved_filepath,
                X_supercharge=new_X_supercharge_np,
                X_inactive_mask=new_X_inactive_mask_np,
                y=new_y_np
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
    print(f"Data generation process finished for tech '{tech}'. Generated: {final_generated_count} (incl. augmentations). Time: {elapsed_time_tech:.2f}s")

    return final_generated_count, num_output_classes, saved_filepath


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a batch of training data for NMS Optimizer.")
    parser.add_argument("--category", type=str, required=True, help="Tech category (used if --tech is not specified).", metavar="CATEGORY_NAME")
    parser.add_argument("--tech", type=str, default=None, help="Specific tech key to process (optional). If provided, only this tech is processed.", metavar="TECH_KEY")
    parser.add_argument("--ship", type=str, default="standard", help="Ship type.", metavar="SHIP_TYPE")
    parser.add_argument("--samples", type=int, default=64, help="Original samples per tech (before augmentation).", metavar="NUM_SAMPLES")
    # <<< Remove width and height arguments >>>
    # parser.add_argument("--width", type=int, default=4, help="Grid width.", metavar="WIDTH")
    # parser.add_argument("--height", type=int, default=3, help="Grid height.", metavar="HEIGHT")
    parser.add_argument("--max_sc", type=int, default=4, help="Max supercharged.", metavar="MAX_SC")
    parser.add_argument("--max_inactive", type=int, default=0, help="Max inactive.", metavar="MAX_INACTIVE")
    parser.add_argument("--output_dir", type=str, default=GENERATED_BATCH_DIR, help="Base output directory (ship/tech subdirs will be created).", metavar="DIR_PATH")
    parser.add_argument("--solve_prob", type=float, default=0.0, help="Probability (0.0-1.0) of using solve map (if supercharged > 0).", metavar="PROB")
    args = parser.parse_args()
    if not 0.0 <= args.solve_prob <= 1.0: parser.error("--solve_prob must be between 0.0 and 1.0")

    config = {
        "num_samples_per_tech": args.samples,
        # <<< Remove grid_width, grid_height from config >>>
        # "grid_width": args.width,
        # "grid_height": args.height,
        "max_supercharged": args.max_sc, "max_inactive_cells": args.max_inactive, "ship": args.ship,
        "tech_category_to_process": args.category,
        "specific_tech_to_process": args.tech,
        "output_dir": args.output_dir, "solve_map_prob": args.solve_prob,
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
            if not ship_data or "types" not in ship_data or not isinstance(ship_data["types"], dict):
                raise KeyError(f"Ship '{config['ship']}' or its 'types' dictionary not found/invalid.")
            category_data = ship_data["types"].get(config["tech_category_to_process"])
            if not category_data or not isinstance(category_data, list):
                raise KeyError(f"Category '{config['tech_category_to_process']}' not found or invalid for ship '{config['ship']}'.")
            tech_keys_to_process = [t["key"] for t in category_data if isinstance(t, dict) and "key" in t]
            if not tech_keys_to_process:
                raise ValueError(f"No valid tech keys found for ship '{config['ship']}', category '{config['tech_category_to_process']}'.")
            print(f"Planning to generate data for techs in category '{config['tech_category_to_process']}': {tech_keys_to_process}")
    except (KeyError, ValueError) as e:
        print(f"Error determining techs to process: {e}"); exit()
    except Exception as e:
        print(f"An unexpected error occurred while getting tech keys: {e}"); exit()

    total_samples_generated_overall = 0
    for tech in tech_keys_to_process:
        print(f"\n{'='*10} Processing Tech: {tech} {'='*10}")
        # <<< Update function call to remove width/height >>>
        generated_count, _, _ = generate_training_batch(
            num_samples=config["num_samples_per_tech"],
            # grid_width=config["grid_width"], # Removed
            # grid_height=config["grid_height"], # Removed
            max_supercharged=config["max_supercharged"],
            max_inactive_cells=config["max_inactive_cells"], ship=config["ship"], tech=tech,
            base_output_dir=config["output_dir"],
            solve_map_prob=config["solve_map_prob"],
        )
        total_samples_generated_overall += generated_count

    end_time_all = time.time()
    print(f"\n{'='*20} Data Batch Generation Complete {'='*20}")
    print(f"Total samples generated across all techs (incl. augmentations) in this run: {total_samples_generated_overall}")
    print(f"Total time: {end_time_all - start_time_all:.2f} seconds.")
    print(f"Generated files saved in subdirectories under: {os.path.abspath(config['output_dir'])}")
