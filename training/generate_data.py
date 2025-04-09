# training/generate_data.py
import random
import numpy as np
import sys
import os
import time
import argparse
import uuid # <-- Import uuid for unique filenames

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
from optimizer import refine_placement_for_training, Grid, get_tech_modules_for_training, print_grid
from modules import modules
from simulated_annealing import simulated_annealing

# --- Configuration for Data Storage ---
# Default directory for generated batches (can be overridden by CLI)
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
    output_dir, # <-- New argument for output directory
):
    """
    Generates a batch of training data and saves it to a unique .npz file.

    Args:
        num_samples (int): The number of samples to generate in this batch.
        grid_width (int): The width of the grid.
        grid_height (int): The height of the grid.
        max_supercharged (int): Max supercharged slots (among active cells).
        max_inactive_cells (int): Max inactive cells.
        ship (str): The type of ship.
        tech (str): The technology to generate data for.
        output_dir (str): Directory to save the generated .npz batch file.

    Returns:
        tuple: (generated_count, num_output_classes, saved_filepath) or (0, 0, None) on failure.
    """
    start_time_tech = time.time()

    # --- 1. Generate Samples (No Loading/Appending) ---
    new_X_list = []
    new_y_list = []
    print(f"Generating {num_samples} samples for ship='{ship}', tech='{tech}'...")

    tech_modules = get_tech_modules_for_training(modules, ship, tech)
    if not tech_modules:
        print(f"Error: No tech modules found for ship='{ship}', tech='{tech}'. Cannot generate data.")
        return 0, 0, None # Indicate failure

    # Sort modules by ID to ensure consistent mapping across runs
    tech_modules.sort(key=lambda m: m['id'])
    module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(tech_modules)}
    num_output_classes = len(tech_modules) + 1

    generated_count = 0
    attempt_count = 0
    max_attempts = num_samples * 15 + 100

    total_cells = grid_width * grid_height
    all_positions = [(x, y) for y in range(grid_height) for x in range(grid_width)]

    while generated_count < num_samples and attempt_count < max_attempts:
        attempt_count += 1
        grid = Grid(grid_width, grid_height)

        # --- Add Inactive Cells ---
        num_inactive = random.randint(0, min(max_inactive_cells, total_cells))
        if num_inactive > 0:
            inactive_positions = random.sample(all_positions, num_inactive)
            for x, y in inactive_positions:
                grid.set_active(x, y, False)
        else:
            inactive_positions = []

        # --- Determine Active Cells for Supercharging ---
        active_positions = [pos for pos in all_positions if pos not in inactive_positions]
        num_active_cells = len(active_positions)
        if num_active_cells == 0:
            # print(f"Attempt {attempt_count}: No active cells, skipping.") # Debug
            continue

        # --- Set Supercharged Cells ---
        max_possible_supercharged = min(max_supercharged, num_active_cells)
        num_supercharged = random.randint(0, max_possible_supercharged)
        if num_supercharged > 0:
            supercharged_positions = random.sample(active_positions, num_supercharged)
            for x, y in supercharged_positions:
                grid.set_supercharged(x, y, True)

        # --- Optimize Placement ---
        try:
            num_modules_for_tech = len(tech_modules)
            # print(f"-- Tech '{tech}' has {num_modules_for_tech} modules. Selecting algorithm...") # Less verbose

            if num_modules_for_tech < 7:
                # print(f"-- Using refine_placement_for_training (brute-force) for {tech}.") # Less verbose
                optimized_grid, best_bonus = refine_placement_for_training(
                    grid, ship, modules, tech
                )
            else:
                # print(f"-- Using simulated_annealing for {tech}.") # Less verbose
                optimized_grid, best_bonus = simulated_annealing(
                    grid,
                    ship,
                    modules,
                    tech,
                    player_owned_rewards=["PC"],
                    initial_temperature=4000,
                    cooling_rate=0.997,
                    stopping_temperature=1.0,
                    iterations_per_temp=50,
                    initial_swap_probability=0.55,
                    final_swap_probability=0.4,
                )

            # Optional: Keep the print_grid if useful for debugging generation
            # print_grid(optimized_grid)

            if optimized_grid is None:
                # print(f"-- Optimization returned None for attempt {attempt_count}. Skipping.") # Less verbose
                continue
        except ValueError as ve:
             print(f"\nOptimization failed for attempt {attempt_count} with ValueError: {ve}. Skipping.")
             continue
        except Exception as e:
            print(f"\nUnexpected error during optimization for attempt {attempt_count}: {e}")
            # import traceback
            # traceback.print_exc()
            continue

        # --- Create Input/Output Matrices ---
        input_matrix = np.zeros((grid_height, grid_width), dtype=np.int8)
        output_matrix = np.zeros((grid_height, grid_width), dtype=np.int8)
        module_found_in_mapping = True
        for y in range(grid_height):
            for x in range(grid_width):
                cell_data = optimized_grid.get_cell(x, y)
                input_matrix[y, x] = int(cell_data["supercharged"]) # 1 if SC, 0 otherwise
                # Ensure inactive cells are marked in input? No, model only gets SC layout.
                # Inactive cells are handled by the target output (class 0).

                module_id = cell_data["module"]
                if module_id is None:
                    output_matrix[y, x] = 0 # Background class for empty/inactive
                else:
                    mapped_class = module_id_mapping.get(module_id)
                    if mapped_class is None:
                        print(
                            f"\nWarning: Module ID '{module_id}' from optimized grid not found in mapping for tech '{tech}'. Skipping sample."
                        )
                        module_found_in_mapping = False
                        break
                    output_matrix[y, x] = mapped_class
            if not module_found_in_mapping:
                continue # Skip this sample
        # No need for the outer check, inner break handles it

        new_X_list.append(input_matrix)
        new_y_list.append(output_matrix)
        generated_count += 1
        # Progress print (less verbose)
        if generated_count % 1 == 0 or generated_count == num_samples:
             print(f"-- Generated {generated_count}/{num_samples} samples for tech '{tech}' (Attempt {attempt_count}) --")


    if generated_count < num_samples:
        print(
            f"\nWarning: Only generated {generated_count}/{num_samples} samples after {max_attempts} attempts for tech '{tech}'."
        )

    if not new_X_list:
        print("No valid samples generated for this batch. Skipping save.")
        return 0, num_output_classes, None

    new_X_np = np.array(new_X_list, dtype=np.int8)
    new_y_np = np.array(new_y_list, dtype=np.int8)

    # --- 2. Save This Batch to a Unique File ---
    saved_filepath = None
    if new_X_np.size > 0:
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"data_{ship}_{tech}_{timestamp}_{unique_id}.npz"
            saved_filepath = os.path.join(output_dir, filename)

            print(f"Saving {len(new_X_np)} generated samples to {saved_filepath}...")
            # Use savez_compressed to save space and bundle X and y
            np.savez_compressed(saved_filepath, X=new_X_np, y=new_y_np)
            print("Save complete.")
        except Exception as e:
            print(f"Error saving data batch: {e}")
            saved_filepath = None # Ensure None is returned on save error
            generated_count = 0 # Reflect that nothing was successfully saved
    else:
        print("No data generated in this batch, skipping save.")

    elapsed_time_tech = time.time() - start_time_tech
    print(
        f"Data generation process finished for tech '{tech}'. Generated: {generated_count}. Time: {elapsed_time_tech:.2f}s"
    )

    return generated_count, num_output_classes, saved_filepath


# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate a batch of training data for NMS Optimizer.")
    parser.add_argument(
        "--category", type=str, required=True,
        help="The technology category to generate data for (e.g., 'Hyperdrive', 'Weaponry').",
        metavar="CATEGORY_NAME"
    )
    parser.add_argument(
        "--ship", type=str, default="standard",
        help="The ship type to use (e.g., 'standard', 'sentinel').",
        metavar="SHIP_TYPE"
    )
    parser.add_argument(
        "--samples", type=int, default=64,
        help="Number of samples to generate per tech in this batch.",
        metavar="NUM_SAMPLES"
    )
    parser.add_argument(
        "--width", type=int, default=4, help="Grid width.", metavar="WIDTH"
    )
    parser.add_argument(
        "--height", type=int, default=3, help="Grid height.", metavar="HEIGHT"
    )
    parser.add_argument(
        "--max_sc", type=int, default=4,
        help="Maximum number of supercharged slots.", metavar="MAX_SC"
    )
    parser.add_argument(
        "--max_inactive", type=int, default=3,
        help="Maximum number of inactive cells.", metavar="MAX_INACTIVE"
    )
    parser.add_argument( # <-- New argument
        "--output_dir", type=str, default=GENERATED_BATCH_DIR,
        help="Directory to save the generated data batch files.",
        metavar="DIR_PATH"
    )

    args = parser.parse_args()
    # --- End Argument Parsing ---

    # --- Configuration (now uses parsed arguments) ---
    config = {
        "num_samples_per_tech": args.samples,
        "grid_width": args.width,
        "grid_height": args.height,
        "max_supercharged": args.max_sc,
        "max_inactive_cells": args.max_inactive,
        "ship": args.ship,
        "tech_category_to_process": args.category,
        "output_dir": args.output_dir, # <-- Use the parsed argument
    }
    # --- End Configuration ---

    start_time_all = time.time()
    print(f"Starting data batch generation process...")
    print(f"Configuration: {config}")

    # --- Ensure Output Directory Exists ---
    try:
        os.makedirs(config["output_dir"], exist_ok=True)
        print(f"Output directory: {os.path.abspath(config['output_dir'])}")
    except OSError as e:
        print(f"Error creating output directory '{config['output_dir']}': {e}")
        exit()
    # --- End Directory Check ---

    # --- Get Tech Keys ---
    try:
        ship_data = modules.get(config["ship"])
        if not ship_data or "types" not in ship_data or not isinstance(ship_data["types"], dict):
            raise KeyError(f"Ship '{config['ship']}' or its 'types' dictionary not found/invalid.")

        category_data = ship_data["types"].get(config["tech_category_to_process"])
        if not category_data or not isinstance(category_data, list):
            raise KeyError(
                f"Category '{config['tech_category_to_process']}' not found or invalid for ship '{config['ship']}'."
            )

        tech_keys_to_process = [
            tech_data["key"] for tech_data in category_data if isinstance(tech_data, dict) and "key" in tech_data
        ]
    except KeyError as e:
        print(f"Error accessing module data: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while getting tech keys: {e}")
        exit()

    if not tech_keys_to_process:
        print(
            f"Error: No valid tech keys found for ship '{config['ship']}', category '{config['tech_category_to_process']}'."
        )
        exit()

    print(f"Planning to generate data for techs: {tech_keys_to_process}")

    # --- Loop and Generate Data Batches ---
    total_samples_generated_overall = 0
    for tech in tech_keys_to_process:
        print(f"\n{'='*10} Processing Tech: {tech} {'='*10}")
        generated_count, _, _ = generate_training_batch( # Renamed function
            num_samples=config["num_samples_per_tech"],
            grid_width=config["grid_width"],
            grid_height=config["grid_height"],
            max_supercharged=config["max_supercharged"],
            max_inactive_cells=config["max_inactive_cells"],
            ship=config["ship"],
            tech=tech,
            output_dir=config["output_dir"], # Pass output dir
        )
        total_samples_generated_overall += generated_count

    end_time_all = time.time()
    print(f"\n{'='*20} Data Batch Generation Complete {'='*20}")
    print(f"Total samples generated across all techs in this run: {total_samples_generated_overall}")
    print(f"Total time: {end_time_all - start_time_all:.2f} seconds.")
    print(f"Generated files saved in: {os.path.abspath(config['output_dir'])}")

