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
# Import Grid and get_tech_modules_for_training from optimizer
from optimizer import Grid, get_tech_modules_for_training

# Import optimization functions directly from optimization_algorithms
from optimization_algorithms import (
    get_all_unique_pattern_variations,
    refine_placement_for_training # <<<--- Moved this import here
)
# Import other necessary modules
from modules import modules, solves
from simulated_annealing import simulated_annealing
from grid_display import print_grid
from module_placement import place_module

# --- Configuration for Data Storage ---
GENERATED_BATCH_DIR = "generated_batches"
# --- End Configuration ---


# --- Data Generation Function (Saves Unique Batches) ---
def generate_training_batch(
    num_samples,
    grid_width,
    grid_height,
    max_supercharged,
    max_inactive_cells,
    ship,
    tech,
    output_dir,
    solve_map_prob=0.0,
):
    """
    Generates a batch of training data and saves it to a unique .npz file.
    Optionally uses pre-defined solve maps (including rotated/mirrored variations)
    for a portion of the samples. Includes mirrored data augmentation.

    Args:
        num_samples (int): The number of *original* samples to generate before augmentation.
                           The final batch size will be larger due to augmentation.
        grid_width (int): The width of the grid.
        grid_height (int): The height of the grid.
        max_supercharged (int): Max supercharged slots (among active cells).
        max_inactive_cells (int): Max inactive cells.
        ship (str): The type of ship.
        tech (str): The technology to generate data for.
        output_dir (str): Directory to save the generated .npz batch file.
        solve_map_prob (float): Probability (0.0 to 1.0) of using a pre-defined
                                solve map variation instead of optimization.

    Returns:
        tuple: (generated_count, num_output_classes, saved_filepath) or (0, 0, None) on failure.
               generated_count now reflects the total count including augmentations.
    """
    start_time_tech = time.time()

    # --- 1. Generate Samples ---
    new_X_list = []
    new_y_list = []
    # Note: num_samples is the target for *original* generations.
    print(f"Generating {num_samples} original samples for ship='{ship}', tech='{tech}' (Solve Map Prob: {solve_map_prob:.2f})...")
    print("Data augmentation (mirroring) will increase the final sample count.")

    tech_modules = get_tech_modules_for_training(modules, ship, tech)
    if not tech_modules:
        print(f"Error: No tech modules found for ship='{ship}', tech='{tech}'. Cannot generate data.")
        return 0, 0, None

    tech_modules.sort(key=lambda m: m['id'])
    module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(tech_modules)}
    num_output_classes = len(tech_modules) + 1

    # --- Mappings for Visualization ---
    class_id_to_module = {v: k for k, v in module_id_mapping.items()}
    # Build a comprehensive map of ALL module details for the ship for visualization
    all_ship_modules_list = []
    if ship in modules:
        for cat_data in modules[ship].get("types", {}).values():
            if isinstance(cat_data, list):
                for tech_info in cat_data:
                    if isinstance(tech_info, dict) and "modules" in tech_info:
                        all_ship_modules_list.extend(tech_info["modules"])
    module_details_map = {m['id']: m for m in all_ship_modules_list}
    # --- End Mappings ---

    # --- Solve Map Variations ---
    solve_map_variations = []
    solve_map_exists = ship in solves and tech in solves[ship]
    if solve_map_exists:
        original_solve_map_data = solves.get(ship, {}).get(tech, {}).get("map")
        if original_solve_map_data:
            try:
                # Generate all unique variations (rotations/mirrors)
                solve_map_variations = get_all_unique_pattern_variations(original_solve_map_data)
                print(f"-- Found {len(solve_map_variations)} unique solve map variations for {tech}.")
            except Exception as e:
                print(f"Error generating solve map variations for {tech}: {e}")
                solve_map_variations = [] # Fallback to empty list
        else:
            print(f"Warning: Solve map entry exists for {ship}/{tech}, but 'map' key is missing or empty.")
            solve_map_exists = False # Treat as non-existent if map data is bad

    if not solve_map_variations: # If generation failed or no map existed
        solve_map_exists = False
    # --- End Solve Map Variations ---

    original_generated_count = 0 # Track original samples generated
    attempt_count = 0
    # Adjust max_attempts if needed, as each success yields more samples now
    max_attempts = num_samples * 15 + 100

    total_cells = grid_width * grid_height
    all_positions = [(x, y) for y in range(grid_height) for x in range(grid_width)]

    while original_generated_count < num_samples and attempt_count < max_attempts:
        attempt_count += 1
        original_grid_layout = Grid(grid_width, grid_height)

        # --- Add Inactive/Supercharged Cells (same as before) ---
        inactive_positions = []
        if random.random() < 0.5:
            num_inactive = random.randint(0, min(max_inactive_cells, total_cells))
            if num_inactive > 0:
                if num_inactive <= len(all_positions):
                    inactive_positions = random.sample(all_positions, num_inactive)
                    for x, y in inactive_positions:
                        original_grid_layout.set_active(x, y, False)
                else:
                    print(f"Warning: Cannot sample {num_inactive} inactive positions from {len(all_positions)} total positions. Setting 0 inactive.")
                    inactive_positions = []

        active_positions = [pos for pos in all_positions if pos not in inactive_positions]
        num_active_cells = len(active_positions)
        if num_active_cells == 0: continue

        max_possible_supercharged = min(max_supercharged, num_active_cells)
        num_supercharged = random.randint(0, max_possible_supercharged)
        if num_supercharged > 0:
            supercharged_positions = random.sample(active_positions, num_supercharged)
            for x, y in supercharged_positions:
                original_grid_layout.set_supercharged(x, y, True)
        # --- End Add Cells ---

        # --- Decide: Use Solve Map Variation or Optimize ---
        use_solve_map = solve_map_exists and random.random() < solve_map_prob

        input_matrix = np.zeros((grid_height, grid_width), dtype=np.int8)
        output_matrix = np.zeros((grid_height, grid_width), dtype=np.int8)
        sample_valid = True

        # --- Create Input Matrix (same as before) ---
        for y in range(grid_height):
            for x in range(grid_width):
                cell = original_grid_layout.get_cell(x, y)
                # Input matrix represents supercharged status (1 if active AND supercharged, 0 otherwise)
                input_matrix[y, x] = int(cell["active"] and cell["supercharged"])
        # --- End Input Matrix ---

        # --- Create Output Matrix ---
        if use_solve_map:
            # --- Method 1: Use Random Pre-defined Solve Map Variation ---
            # print(f"-- Attempt {attempt_count}: Using pre-defined solve map variation for tech '{tech}'") # Less verbose
            current_pattern_data = random.choice(solve_map_variations)

            for y in range(grid_height):
                for x in range(grid_width):
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
                                f"\nWarning: Module ID '{module_id}' from SOLVE MAP variation not found in mapping for tech '{tech}'. Skipping sample."
                            )
                            sample_valid = False
                            break
                        output_matrix[y, x] = mapped_class
                if not sample_valid:
                    break

            # --- Visualization (Optional, can be removed/commented for speed) ---
            # if sample_valid:
            #     output_grid_vis = Grid(grid_width, grid_height)
            #     # ... (visualization code remains the same) ...
            #     print(f"Output Grid (Solve Map Variation Placement - Tech: {tech})")
            #     print_grid(output_grid_vis)
            # --- End Visualization ---

        else:
            # --- Method 2: Use Optimization (same as before) ---
            optimized_grid = None
            try:
                num_modules_for_tech = len(tech_modules)
                # Choose optimization based on module count
                if num_modules_for_tech < 7:
                    optimized_grid, best_bonus = refine_placement_for_training(
                        original_grid_layout, ship, modules, tech
                    )
                else:
                    optimized_grid, best_bonus = simulated_annealing(
                        original_grid_layout, ship, modules, tech,
                        player_owned_rewards=["PC"], # Example, adjust if needed
                        initial_temperature=4000, cooling_rate=0.997,
                        stopping_temperature=1.0, iterations_per_temp=50,
                        initial_swap_probability=0.55, final_swap_probability=0.4,
                    )
                if optimized_grid is None: sample_valid = False
            except ValueError as ve:
                 print(f"\nOptimization failed for attempt {attempt_count} with ValueError: {ve}. Skipping.")
                 sample_valid = False
            except Exception as e:
                print(f"\nUnexpected error during optimization for attempt {attempt_count}: {e}")
                sample_valid = False

            if sample_valid and optimized_grid is not None:
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
                    if not sample_valid: continue # Check validity within the inner loop
            elif sample_valid and optimized_grid is None: # Handle case where optimization returns None but no exception
                sample_valid = False
            # --- End Optimization ---


        # --- Add Valid Sample AND AUGMENTATIONS ---
        if sample_valid:
            # 1. Add the original sample
            new_X_list.append(input_matrix)
            new_y_list.append(output_matrix)

            # 2. Add horizontally flipped sample
            input_matrix_hflip = np.fliplr(input_matrix)
            output_matrix_hflip = np.fliplr(output_matrix)
            new_X_list.append(input_matrix_hflip)
            new_y_list.append(output_matrix_hflip)

            # 3. Add vertically flipped sample
            input_matrix_vflip = np.flipud(input_matrix)
            output_matrix_vflip = np.flipud(output_matrix)
            new_X_list.append(input_matrix_vflip)
            new_y_list.append(output_matrix_vflip)

            # 4. Add horizontally AND vertically flipped sample (180 deg rotation)
            input_matrix_hvflip = np.flipud(np.fliplr(input_matrix))
            output_matrix_hvflip = np.flipud(np.fliplr(output_matrix))
            new_X_list.append(input_matrix_hvflip)
            new_y_list.append(output_matrix_hvflip)


            original_generated_count += 1 # Increment count of *original* samples
            total_samples_in_list = len(new_X_list) # Current total including augmentations

            # Update progress print based on original count
            if original_generated_count % 1 == 0 or original_generated_count == num_samples:
                 print(f"-- Generated {original_generated_count}/{num_samples} original samples for tech '{tech}' (Attempt {attempt_count}, Total in list: {total_samples_in_list}) --")
        # --- End Add Sample ---

    # --- Final Count and Warnings ---
    final_generated_count = len(new_X_list) # Total samples including augmentations
    if original_generated_count < num_samples:
        print(
            f"\nWarning: Only generated {original_generated_count}/{num_samples} *original* samples after {max_attempts} attempts for tech '{tech}'. Final count with augmentation: {final_generated_count}."
        )

    if not new_X_list:
        print("No valid samples generated for this batch. Skipping save.")
        return 0, num_output_classes, None

    # --- Convert to NumPy Arrays ---
    new_X_np = np.array(new_X_list, dtype=np.int8)
    new_y_np = np.array(new_y_list, dtype=np.int8)

    # --- Save Batch ---
    saved_filepath = None
    if new_X_np.size > 0:
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            # Include grid dimensions in filename for clarity
            filename = f"data_{ship}_{tech}_{grid_width}x{grid_height}_{timestamp}_{unique_id}.npz"
            saved_filepath = os.path.join(output_dir, filename)
            print(f"Saving {len(new_X_np)} generated samples (incl. augmentations) to {saved_filepath}...")
            np.savez_compressed(saved_filepath, X=new_X_np, y=new_y_np)
            print("Save complete.")
        except Exception as e:
            print(f"Error saving data batch: {e}")
            saved_filepath = None
            final_generated_count = 0 # Reset count if save fails
    else:
        print("No data generated in this batch, skipping save.")
        final_generated_count = 0

    elapsed_time_tech = time.time() - start_time_tech
    print(f"Data generation process finished for tech '{tech}'. Generated: {final_generated_count} (incl. augmentations). Time: {elapsed_time_tech:.2f}s")

    # Return the final count including augmentations
    return final_generated_count, num_output_classes, saved_filepath


# --- Main Execution (remains the same) ---
if __name__ == "__main__":
    # --- Argument Parsing (remains the same) ---
    parser = argparse.ArgumentParser(description="Generate a batch of training data for NMS Optimizer.")
    parser.add_argument("--category", type=str, required=True, help="Tech category.", metavar="CATEGORY_NAME")
    parser.add_argument("--ship", type=str, default="standard", help="Ship type.", metavar="SHIP_TYPE")
    parser.add_argument("--samples", type=int, default=64, help="Original samples per tech (before augmentation).", metavar="NUM_SAMPLES") # Clarified help text
    parser.add_argument("--width", type=int, default=4, help="Grid width.", metavar="WIDTH")
    parser.add_argument("--height", type=int, default=3, help="Grid height.", metavar="HEIGHT")
    parser.add_argument("--max_sc", type=int, default=4, help="Max supercharged.", metavar="MAX_SC")
    parser.add_argument("--max_inactive", type=int, default=3, help="Max inactive.", metavar="MAX_INACTIVE")
    parser.add_argument("--output_dir", type=str, default=GENERATED_BATCH_DIR, help="Output directory.", metavar="DIR_PATH")
    parser.add_argument("--solve_prob", type=float, default=0.25, help="Probability (0.0-1.0) of using solve map.", metavar="PROB")
    args = parser.parse_args()
    if not 0.0 <= args.solve_prob <= 1.0: parser.error("--solve_prob must be between 0.0 and 1.0")

    # --- Configuration (remains the same) ---
    config = {
        "num_samples_per_tech": args.samples, "grid_width": args.width, "grid_height": args.height,
        "max_supercharged": args.max_sc, "max_inactive_cells": args.max_inactive, "ship": args.ship,
        "tech_category_to_process": args.category, "output_dir": args.output_dir, "solve_map_prob": args.solve_prob,
    }

    start_time_all = time.time()
    print(f"Starting data batch generation process...")
    print(f"Configuration: {config}")

    # --- Directory Check (remains the same) ---
    try:
        os.makedirs(config["output_dir"], exist_ok=True)
        print(f"Output directory: {os.path.abspath(config['output_dir'])}")
    except OSError as e: print(f"Error creating output directory '{config['output_dir']}': {e}"); exit()

    # --- Get Tech Keys (remains the same) ---
    try:
        ship_data = modules.get(config["ship"])
        if not ship_data or "types" not in ship_data or not isinstance(ship_data["types"], dict):
            raise KeyError(f"Ship '{config['ship']}' or its 'types' dictionary not found/invalid.")
        category_data = ship_data["types"].get(config["tech_category_to_process"])
        if not category_data or not isinstance(category_data, list):
            raise KeyError(f"Category '{config['tech_category_to_process']}' not found or invalid for ship '{config['ship']}'.")
        tech_keys_to_process = [t["key"] for t in category_data if isinstance(t, dict) and "key" in t]
    except KeyError as e: print(f"Error accessing module data: {e}"); exit()
    except Exception as e: print(f"An unexpected error occurred while getting tech keys: {e}"); exit()
    if not tech_keys_to_process: print(f"Error: No valid tech keys found for ship '{config['ship']}', category '{config['tech_category_to_process']}'."); exit()
    print(f"Planning to generate data for techs: {tech_keys_to_process}")

    # --- Loop and Generate Data Batches (remains the same) ---
    total_samples_generated_overall = 0
    for tech in tech_keys_to_process:
        print(f"\n{'='*10} Processing Tech: {tech} {'='*10}")
        # generate_training_batch now returns the count *including* augmentations
        generated_count, _, _ = generate_training_batch(
            num_samples=config["num_samples_per_tech"], grid_width=config["grid_width"],
            grid_height=config["grid_height"], max_supercharged=config["max_supercharged"],
            max_inactive_cells=config["max_inactive_cells"], ship=config["ship"], tech=tech,
            output_dir=config["output_dir"], solve_map_prob=config["solve_map_prob"],
        )
        total_samples_generated_overall += generated_count

    end_time_all = time.time()
    print(f"\n{'='*20} Data Batch Generation Complete {'='*20}")
    print(f"Total samples generated across all techs (incl. augmentations) in this run: {total_samples_generated_overall}") # Updated message
    print(f"Total time: {end_time_all - start_time_all:.2f} seconds.")
    print(f"Generated files saved in: {os.path.abspath(config['output_dir'])}")

