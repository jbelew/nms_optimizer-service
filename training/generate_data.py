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
    from optimizer import Grid
except ImportError:
    from grid_utils import Grid

from modules_data import get_tech_modules_for_training
from optimization_algorithms import (
    get_all_unique_pattern_variations,
    refine_placement_for_training
)
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
    base_output_dir, # <<< Renamed for clarity
    solve_map_prob=0.0,
):
    """
    Generates a batch of training data and saves it into ship/tech subdirectories.
    Optionally uses pre-defined solve maps variations. Includes mirrored data augmentation.

    Args:
        num_samples (int): Number of original samples to generate (before augmentation).
        grid_width (int): Width of the grid.
        grid_height (int): Height of the grid.
        max_supercharged (int): Maximum number of supercharged cells to add.
        max_inactive_cells (int): Maximum number of inactive cells to add.
        ship (str): Ship type key.
        tech (str): Technology key.
        base_output_dir (str): Base directory to save the generated .npz file (e.g., 'generated_batches').
        solve_map_prob (float): Probability (0.0-1.0) of using a pre-defined solve map variation.

    Returns:
        tuple: (generated_count, num_output_classes, saved_filepath) or (0, 0, None) on failure.
               generated_count reflects total count including augmentations.
    """
    start_time_tech = time.time()

    # --- Construct specific output directory ---
    tech_output_dir = os.path.join(base_output_dir, ship, tech)
    # --- End Construct specific output directory ---

    # --- 1. Generate Samples ---
    new_X_supercharge_list = []
    new_X_inactive_mask_list = []
    new_y_list = []

    tech_modules = get_tech_modules_for_training(modules, ship, tech)
    if not tech_modules:
        print(f"Error: No tech modules found for ship='{ship}', tech='{tech}'. Cannot generate data.")
        return 0, 0, None

    tech_modules.sort(key=lambda m: m['id'])
    module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(tech_modules)}
    num_output_classes = len(tech_modules) + 1

    # --- Check if Solve Map Exists and Generate Variations ---
    solve_map_exists = False
    solve_map_variations = []
    if ship in solves and tech in solves[ship]:
        original_pattern = solves[ship][tech].get("map")
        if original_pattern:
            solve_map_exists = True
            solve_map_variations = get_all_unique_pattern_variations(original_pattern)
            if not solve_map_variations:
                 print(f"Warning: Solve map exists for {ship}/{tech}, but generated no variations.")
                 solve_map_exists = False
    # --- End Solve Map Check ---

    original_generated_count = 0
    attempt_count = 0
    max_attempts = num_samples * 15 + 100

    total_cells = grid_width * grid_height
    all_positions = [(x, y) for y in range(grid_height) for x in range(grid_width)]

    while original_generated_count < num_samples and attempt_count < max_attempts:
        attempt_count += 1
        original_grid_layout = Grid(grid_width, grid_height)
        inactive_positions_set = set()

        # --- Add Inactive/Supercharged Cells ---
        inactive_positions = []
        if random.random() < 0.10:
            print("INFO -- Adding inactive cells")
            num_inactive = random.randint(0, min(max_inactive_cells, total_cells))
            if num_inactive > 0:
                if num_inactive <= len(all_positions):
                    inactive_positions = random.sample(all_positions, num_inactive)
                    inactive_positions_set = set(inactive_positions)
                    for x, y in inactive_positions:
                        original_grid_layout.set_active(x, y, False)
                else:
                    print(f"Warning: Cannot sample {num_inactive} inactive positions from {len(all_positions)} total positions. Setting 0 inactive.")
                    inactive_positions = []
                    inactive_positions_set = set()

        active_positions = [pos for pos in all_positions if pos not in inactive_positions_set]
        num_active_cells = len(active_positions)
        if num_active_cells == 0: continue

        supercharged_positions_set = set()
        max_possible_supercharged = min(max_supercharged, num_active_cells)
        num_supercharged = random.randint(0, max_possible_supercharged)
        if num_supercharged > 0:
            supercharged_positions = random.sample(active_positions, num_supercharged)
            supercharged_positions_set = set(supercharged_positions)
            for x, y in supercharged_positions:
                original_grid_layout.set_supercharged(x, y, True)
        # --- End Add Cells ---

        # --- Decide: Use Solve Map Variation or Optimize ---
        use_solve_map = solve_map_exists and random.random() < solve_map_prob

        # --- Create Input and Output Matrices ---
        input_supercharge_np = np.zeros((grid_height, grid_width), dtype=np.int8)
        input_inactive_mask_np = np.zeros((grid_height, grid_width), dtype=np.int8)
        output_matrix = np.zeros((grid_height, grid_width), dtype=np.int8)
        sample_valid = True

        # Create Input Matrices
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
                 continue

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
                            print(f"\nWarning: Module ID '{module_id}' from SOLVE MAP variation not found. Skipping sample.")
                            sample_valid = False; break
                        output_matrix[y, x] = mapped_class
                if not sample_valid: continue
        else:
            # Optimize using refine_placement_for_training or simulated_annealing
            optimized_grid = None
            try:
                num_modules_for_tech = len(tech_modules)
                if num_modules_for_tech < 7:
                    optimized_grid, best_bonus = refine_placement_for_training(
                        original_grid_layout, ship, modules, tech
                    )
                else:
                    optimized_grid, best_bonus = simulated_annealing(
                        original_grid_layout, ship, modules, tech,
                        player_owned_rewards=[], # SA for training uses all modules
                        initial_temperature=4000, cooling_rate=0.995,
                        stopping_temperature=1.5, iterations_per_temp=40,
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
                print_grid(optimized_grid) # Optional: uncomment for debugging
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
        # --- End Output Matrix ---


        # --- Add Valid Sample AND AUGMENTATIONS ---
        if sample_valid:
            # 1. Original
            new_X_supercharge_list.append(input_supercharge_np)
            new_X_inactive_mask_list.append(input_inactive_mask_np)
            new_y_list.append(output_matrix)

            # 2. Horizontal Flip
            new_X_supercharge_list.append(np.fliplr(input_supercharge_np))
            new_X_inactive_mask_list.append(np.fliplr(input_inactive_mask_np))
            new_y_list.append(np.fliplr(output_matrix))

            # 3. Vertical Flip
            new_X_supercharge_list.append(np.flipud(input_supercharge_np))
            new_X_inactive_mask_list.append(np.flipud(input_inactive_mask_np))
            new_y_list.append(np.flipud(output_matrix))

            # 4. Horizontal + Vertical Flip (180 Rot)
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
            # <<< Use tech_output_dir >>>
            os.makedirs(tech_output_dir, exist_ok=True) # Ensure directory exists
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"data_{ship}_{tech}_{grid_width}x{grid_height}_{timestamp}_{unique_id}.npz"
            # <<< Use tech_output_dir >>>
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
    parser.add_argument("--category", type=str, required=True, help="Tech category.", metavar="CATEGORY_NAME")
    parser.add_argument("--ship", type=str, default="standard", help="Ship type.", metavar="SHIP_TYPE")
    parser.add_argument("--samples", type=int, default=64, help="Original samples per tech (before augmentation).", metavar="NUM_SAMPLES")
    parser.add_argument("--width", type=int, default=4, help="Grid width.", metavar="WIDTH")
    parser.add_argument("--height", type=int, default=3, help="Grid height.", metavar="HEIGHT")
    parser.add_argument("--max_sc", type=int, default=4, help="Max supercharged.", metavar="MAX_SC")
    parser.add_argument("--max_inactive", type=int, default=3, help="Max inactive.", metavar="MAX_INACTIVE")
    # <<< Changed help text for output_dir >>>
    parser.add_argument("--output_dir", type=str, default=GENERATED_BATCH_DIR, help="Base output directory (ship/tech subdirs will be created).", metavar="DIR_PATH")
    parser.add_argument("--solve_prob", type=float, default=0.05, help="Probability (0.0-1.0) of using solve map.", metavar="PROB")
    args = parser.parse_args()
    if not 0.0 <= args.solve_prob <= 1.0: parser.error("--solve_prob must be between 0.0 and 1.0")

    config = {
        "num_samples_per_tech": args.samples, "grid_width": args.width, "grid_height": args.height,
        "max_supercharged": args.max_sc, "max_inactive_cells": args.max_inactive, "ship": args.ship,
        "tech_category_to_process": args.category, "output_dir": args.output_dir, "solve_map_prob": args.solve_prob,
    }

    start_time_all = time.time()
    print(f"Starting data batch generation process...")
    print(f"Configuration: {config}")

    # <<< No need to create base_output_dir here, generate_training_batch handles it >>>
    # try:
    #     os.makedirs(config["output_dir"], exist_ok=True)
    #     print(f"Base output directory: {os.path.abspath(config['output_dir'])}")
    # except OSError as e: print(f"Error creating base output directory '{config['output_dir']}': {e}"); exit()

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

    total_samples_generated_overall = 0
    for tech in tech_keys_to_process:
        print(f"\n{'='*10} Processing Tech: {tech} {'='*10}")
        generated_count, _, _ = generate_training_batch(
            num_samples=config["num_samples_per_tech"], grid_width=config["grid_width"],
            grid_height=config["grid_height"], max_supercharged=config["max_supercharged"],
            max_inactive_cells=config["max_inactive_cells"], ship=config["ship"], tech=tech,
            base_output_dir=config["output_dir"], # <<< Pass base_output_dir
            solve_map_prob=config["solve_map_prob"],
        )
        total_samples_generated_overall += generated_count

    end_time_all = time.time()
    print(f"\n{'='*20} Data Batch Generation Complete {'='*20}")
    print(f"Total samples generated across all techs (incl. augmentations) in this run: {total_samples_generated_overall}")
    print(f"Total time: {end_time_all - start_time_all:.2f} seconds.")
    # <<< Updated final message >>>
    print(f"Generated files saved in subdirectories under: {os.path.abspath(config['output_dir'])}")

