# training/generate_data.py
import random
import numpy as np
import sys
import os
import time

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
from optimizer import (
    refine_placement_for_training,
    Grid,
    get_tech_modules_for_training,
)
from modules import modules

# --- Configuration for Data Storage ---
TRAINING_DATA_DIR = "training_data" # Directory to store .npy files
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
# --- End Configuration ---


# --- Data Generation Function (Loads/Appends) ---
def generate_or_update_training_data(
    num_new_samples,
    grid_width,
    grid_height,
    max_supercharged,
    max_inactive_cells,
    ship,
    tech,
    data_dir=TRAINING_DATA_DIR
):
    """
    Generates training data, appending to existing files if found.

    Args:
        num_new_samples (int): The number of *new* samples to generate in this run.
        grid_width (int): The width of the grid.
        grid_height (int): The height of the grid.
        max_supercharged (int): Max supercharged slots (among active cells).
        max_inactive_cells (int): Max inactive cells.
        ship (str): The type of ship.
        tech (str): The technology to generate data for.
        data_dir (str): Directory to load/save .npy data files.

    Returns:
        tuple: (total_samples_saved, num_output_classes) or (0, 0) on failure.
    """
    start_time_tech = time.time()
    X_existing, y_existing = None, None
    x_file_path = os.path.join(data_dir, f"X_{ship}_{tech}.npy")
    y_file_path = os.path.join(data_dir, f"y_{ship}_{tech}.npy")

    # --- 1. Load Existing Data ---
    if os.path.exists(x_file_path) and os.path.exists(y_file_path):
        try:
            print(f"Loading existing data from {x_file_path} and {y_file_path}...")
            X_existing = np.load(x_file_path)
            y_existing = np.load(y_file_path)
            print(f"Loaded {len(X_existing)} existing samples.")
            # Basic shape check
            if X_existing.shape[1:] != (grid_height, grid_width) or y_existing.shape[1:] != (grid_height, grid_width):
                 print(f"Warning: Existing data shape mismatch! Expected ({grid_height},{grid_width}), found X:{X_existing.shape[1:]}, y:{y_existing.shape[1:]}. Discarding existing data.")
                 X_existing, y_existing = None, None
        except Exception as e:
            print(f"Error loading existing data: {e}. Starting fresh.")
            X_existing, y_existing = None, None
    else:
        print("No existing data found. Generating fresh dataset.")

    # --- 2. Generate New Samples ---
    new_X_list = []
    new_y_list = []
    print(f"Generating {num_new_samples} new samples for ship='{ship}', tech='{tech}'...")

    tech_modules = get_tech_modules_for_training(modules, ship, tech)
    if not tech_modules:
        print(f"Error: No tech modules found for ship='{ship}', tech='{tech}'. Cannot generate data.")
        return 0, 0 # Indicate failure

    module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(tech_modules)}
    num_output_classes = len(tech_modules) + 1

    generated_count = 0
    attempt_count = 0
    # Adjust max attempts based on how many *new* samples are needed
    max_attempts = num_new_samples * 15 + 100 # Add a base number of attempts

    total_cells = grid_width * grid_height
    all_positions = [(x, y) for y in range(grid_height) for x in range(grid_width)]

    while generated_count < num_new_samples and attempt_count < max_attempts:
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
        if num_active_cells == 0: continue

        # --- Set Supercharged Cells ---
        max_possible_supercharged = min(max_supercharged, num_active_cells)
        num_supercharged = random.randint(0, max_possible_supercharged)
        if num_supercharged > 0:
            supercharged_positions = random.sample(active_positions, num_supercharged)
            for x, y in supercharged_positions:
                grid.set_supercharged(x, y, True)

        # --- Optimize Placement ---
        try:
            # Use the robust training version
            optimized_grid, best_bonus = refine_placement_for_training(grid, ship, modules, tech)
            if optimized_grid is None: continue
        except Exception as e:
            print(f"\nError during optimization for attempt {attempt_count}: {e}")
            continue

        # --- Create Input/Output Matrices ---
        input_matrix = np.zeros((grid_height, grid_width), dtype=np.int8)
        output_matrix = np.zeros((grid_height, grid_width), dtype=np.int8)
        module_found_in_mapping = True
        for y in range(grid_height):
            for x in range(grid_width):
                cell_data = optimized_grid.get_cell(x, y)
                input_matrix[y, x] = int(cell_data["supercharged"])
                module_id = cell_data["module"]
                if module_id is None:
                    output_matrix[y, x] = 0
                else:
                    mapped_class = module_id_mapping.get(module_id)
                    if mapped_class is None:
                        print(f"\nWarning: Module ID '{module_id}' from optimized grid not found in mapping for tech '{tech}'. Skipping sample.")
                        module_found_in_mapping = False; break
                    output_matrix[y, x] = mapped_class
            if not module_found_in_mapping: continue
        if not module_found_in_mapping: continue

        new_X_list.append(input_matrix)
        new_y_list.append(output_matrix)
        generated_count += 1
        # Progress print (optional, can be verbose)
        print(f"-- Generated new sample {generated_count}/{num_new_samples} for tech '{tech}' (Attempt {attempt_count}) --")

    # print() # Newline after loop

    if generated_count < num_new_samples:
        print(f"\nWarning: Only generated {generated_count}/{num_new_samples} new samples after {attempt_count} attempts for tech '{tech}'.")

    new_X_np = np.array(new_X_list, dtype=np.int8) if new_X_list else np.empty((0, grid_height, grid_width), dtype=np.int8)
    new_y_np = np.array(new_y_list, dtype=np.int8) if new_y_list else np.empty((0, grid_height, grid_width), dtype=np.int8)

    # --- 3. Concatenate Data ---
    if X_existing is not None and y_existing is not None:
        print(f"Appending {len(new_X_np)} new samples to {len(X_existing)} existing samples.")
        if X_existing.dtype != new_X_np.dtype:
             print(f"Warning: Casting existing X dtype from {X_existing.dtype} to {new_X_np.dtype}")
             X_existing = X_existing.astype(new_X_np.dtype)
        if y_existing.dtype != new_y_np.dtype:
             print(f"Warning: Casting existing y dtype from {y_existing.dtype} to {new_y_np.dtype}")
             y_existing = y_existing.astype(new_y_np.dtype)
        X_combined = np.concatenate((X_existing, new_X_np), axis=0)
        y_combined = np.concatenate((y_existing, new_y_np), axis=0)
    else:
        X_combined = new_X_np
        y_combined = new_y_np

    # --- 4. Save Combined Data ---
    total_samples_saved = 0
    if X_combined.size > 0:
        try:
            print(f"Saving {len(X_combined)} total samples to {x_file_path} and {y_file_path}...")
            np.save(x_file_path, X_combined)
            np.save(y_file_path, y_combined)
            total_samples_saved = len(X_combined)
            print("Save complete.")
        except Exception as e:
            print(f"Error saving data: {e}")
    else:
        print("No data generated or loaded, skipping save.")

    elapsed_time_tech = time.time() - start_time_tech
    print(f"Data generation/update process finished for tech '{tech}'. Total samples: {total_samples_saved}. Time: {elapsed_time_tech:.2f}s")

    return total_samples_saved, num_output_classes


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    config = {
        "num_new_samples_per_tech": 16, # Number of *new* samples to add per run
        "grid_width": 4,
        "grid_height": 3,
        "max_supercharged": 4,
        "max_inactive_cells": 3,
        "ship": "standard",
        "tech_category_to_process": "Hyperdrive", # Category to generate data for
        "data_dir": TRAINING_DATA_DIR
    }
    # --- End Configuration ---

    start_time_all = time.time()
    print(f"Starting data generation/update process...")
    print(f"Configuration: {config}")

    # --- Get Tech Keys ---
    try:
        ship_data = modules.get(config["ship"])
        if not ship_data or "types" not in ship_data or not isinstance(ship_data["types"], dict):
            raise KeyError(f"Ship '{config['ship']}' or its 'types' dictionary not found/invalid.")

        category_data = ship_data["types"].get(config["tech_category_to_process"])
        if not category_data or not isinstance(category_data, list):
            raise KeyError(f"Category '{config['tech_category_to_process']}' not found or invalid for ship '{config['ship']}'.")

        tech_keys_to_process = [
            tech_data["key"] for tech_data in category_data
            if isinstance(tech_data, dict) and "key" in tech_data
        ]
    except KeyError as e:
        print(f"Error accessing module data: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while getting tech keys: {e}")
        exit()

    if not tech_keys_to_process:
        print(f"Error: No valid tech keys found for ship '{config['ship']}', category '{config['tech_category_to_process']}'.")
        exit()

    print(f"Planning to generate/update data for techs: {tech_keys_to_process}")

    # --- Loop and Generate/Update Data ---
    total_samples_processed = 0
    for tech in tech_keys_to_process:
        print(f"\n{'='*10} Processing Tech: {tech} {'='*10}")
        saved_count, _ = generate_or_update_training_data(
            num_new_samples=config["num_new_samples_per_tech"],
            grid_width=config["grid_width"],
            grid_height=config["grid_height"],
            max_supercharged=config["max_supercharged"],
            max_inactive_cells=config["max_inactive_cells"],
            ship=config["ship"],
            tech=tech,
            data_dir=config["data_dir"]
        )
        total_samples_processed += saved_count # Or add num_new_samples? Depends on what you want to track

    end_time_all = time.time()
    print(f"\n{'='*20} Data Generation/Update Complete {'='*20}")
    print(f"Total time: {end_time_all - start_time_all:.2f} seconds.")
    # Consider logging total samples per tech if needed
