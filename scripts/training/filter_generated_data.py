import os
import numpy as np
import argparse
import glob
import random
import uuid
import itertools
import math
import time
from tqdm import tqdm

# --- Imports from your project ---
from src.grid_utils import Grid
from src.data_loader import get_all_module_data, get_training_module_ids
from src.bonus_calculations import calculate_grid_score
from src.optimization.helpers import determine_window_dimensions
from src.grid_display import print_grid_compact

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GENERATED_BATCH_DIR = os.path.join(PROJECT_ROOT, "scripts", "training", "generated_batches")
GROUND_TRUTH_DATA_DIR = os.path.join(PROJECT_ROOT, "scripts", "training", "ground_truth_data")
# --- End Configuration ---


def get_module_id_mapping(tech_modules):
    """Creates a mapping from module ID to its class index (1-based)."""
    tech_modules.sort(key=lambda m: m["id"])
    return {module["id"]: i + 1 for i, module in enumerate(tech_modules)}


def generate_all_input_grids(ship, tech, solve_type, max_supercharged, max_inactive_cells):
    """
    Generates all possible grid layouts with supercharged and inactive cells.
    Yields a Grid object and its numpy representations for comparison.
    """
    all_module_data = get_all_module_data()
    module_data_for_ship = all_module_data.get(ship)

    if not module_data_for_ship:
        raise ValueError(f"No module data found for ship '{ship}'.")

    training_module_ids = get_training_module_ids(ship, tech)
    if not training_module_ids:
        raise ValueError(f"No training modules found for {ship}/{tech}/{solve_type}.")

    module_count = len(training_module_ids)
    grid_w, grid_h = determine_window_dimensions(module_count, tech, ship)

    all_positions = [(x, y) for y in range(grid_h) for x in range(grid_w)]

    # Iterate through all possible numbers of inactive cells
    for num_inactive in range(max_inactive_cells + 1):
        inactive_combinations = itertools.combinations(all_positions, num_inactive)
        for inactive_positions_tuple in inactive_combinations:
            inactive_positions_set = set(inactive_positions_tuple)
            target_inactive_mask_np = np.zeros((grid_h, grid_w), dtype=np.int8)
            for x, y in inactive_positions_set:
                target_inactive_mask_np[y, x] = 1

            active_positions = [pos for pos in all_positions if pos not in inactive_positions_set]
            num_active_cells = len(active_positions)

            # Iterate through all possible numbers of supercharged slots
            for num_supercharged_actual in range(min(max_supercharged, num_active_cells) + 1):
                supercharged_combinations = itertools.combinations(active_positions, num_supercharged_actual)
                for supercharged_positions_tuple in supercharged_combinations:
                    original_grid_layout = Grid(grid_w, grid_h)
                    target_supercharge_np = np.zeros((grid_h, grid_w), dtype=np.int8)

                    for x, y in inactive_positions_set:
                        original_grid_layout.set_active(x, y, False)

                    for x, y in supercharged_positions_tuple:
                        original_grid_layout.set_supercharged(x, y, True)
                        target_supercharge_np[y, x] = 1

                    yield original_grid_layout, target_supercharge_np, target_inactive_mask_np


def _generate_random_input_grid(ship, tech, solve_type, max_supercharged, max_inactive_cells):
    """
    Generates a random grid layout with supercharged and inactive cells.
    Returns a Grid object and its numpy representations for comparison.
    """
    all_module_data = get_all_module_data()
    module_data_for_ship = all_module_data.get(ship)

    if not module_data_for_ship:
        raise ValueError(f"No module data found for ship '{ship}'.")

    training_module_ids = get_training_module_ids(ship, tech)
    if not training_module_ids:
        raise ValueError(f"No training modules found for {ship}/{tech}/{solve_type}.")

    module_count = len(training_module_ids)
    grid_w, grid_h = determine_window_dimensions(module_count, tech, ship)

    original_grid_layout = Grid(grid_w, grid_h)
    inactive_positions_set = set()
    total_cells = grid_w * grid_h
    all_positions = [(x, y) for y in range(grid_h) for x in range(grid_w)]

    num_inactive = random.randint(0, min(max_inactive_cells, total_cells))
    if num_inactive > 0 and num_inactive <= len(all_positions):
        inactive_positions = random.sample(all_positions, num_inactive)
        inactive_positions_set = set(inactive_positions)
        for x, y in inactive_positions:
            original_grid_layout.set_active(x, y, False)

    active_positions = [pos for pos in all_positions if pos not in inactive_positions_set]
    num_active_cells = len(active_positions)
    supercharged_positions_set = set()
    max_possible_supercharged = min(max_supercharged, num_active_cells)
    num_supercharged_actual = random.randint(0, max_possible_supercharged)

    if num_supercharged_actual > 0 and num_supercharged_actual <= len(active_positions):
        supercharged_positions = random.sample(active_positions, num_supercharged_actual)
        supercharged_positions_set = set(supercharged_positions)
        for x, y in supercharged_positions:
            original_grid_layout.set_supercharged(x, y, True)

    target_supercharge_np = np.zeros((grid_h, grid_w), dtype=np.int8)
    target_inactive_mask_np = np.zeros((grid_h, grid_w), dtype=np.int8)

    for y in range(grid_h):
        for x in range(grid_w):
            pos = (x, y)
            is_active = pos not in inactive_positions_set
            is_supercharged = pos in supercharged_positions_set
            target_supercharge_np[y, x] = int(is_active and is_supercharged)
            target_inactive_mask_np[y, x] = int(not is_active)

    return original_grid_layout, target_supercharge_np, target_inactive_mask_np


def _save_batch(
    batch_X_supercharge,
    batch_X_inactive_mask,
    batch_y,
    output_dir,
    ship,
    tech,
    solve_type,
    batch_file_counter,
    grid_w,
    grid_h,
):
    if not batch_X_supercharge:  # Check if batch is empty
        return

    output_subdir = os.path.join(output_dir, ship, tech)
    if solve_type:
        output_subdir = os.path.join(output_subdir, solve_type)
    os.makedirs(output_subdir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]

    filename_parts = ["data", ship, tech]
    filename_parts.append(f"{grid_w}x{grid_h}")
    filename_parts.append(timestamp)
    filename_parts.append(unique_id)
    filename = "_".join(filename_parts) + ".npz"

    output_filepath = os.path.join(output_subdir, filename)

    np.savez_compressed(
        output_filepath,
        X_supercharge=np.array(batch_X_supercharge),
        X_inactive_mask=np.array(batch_X_inactive_mask),
        y=np.array(batch_y),
    )
    print(f"Saved batch {batch_file_counter} to: {output_filepath} (Contains {len(batch_X_supercharge)} samples)")


def _calculate_total_grids(grid_w, grid_h, max_supercharged, max_inactive_cells):
    total_count = 0
    total_cells = grid_w * grid_h

    for num_inactive in range(max_inactive_cells + 1):
        # Number of ways to choose num_inactive cells from total_cells
        num_inactive_combinations = math.comb(total_cells, num_inactive)
        active_cells_after_inactive = total_cells - num_inactive

        for num_supercharged_actual in range(min(max_supercharged, active_cells_after_inactive) + 1):
            # Number of ways to choose num_supercharged_actual from active_cells_after_inactive
            num_supercharged_combinations = math.comb(active_cells_after_inactive, num_supercharged_actual)
            total_count += num_inactive_combinations * num_supercharged_combinations
    return total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a random input grid and find the highest-scoring matching layout from existing NPZ files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=GENERATED_BATCH_DIR,
        help="Base directory containing generated .npz files to search through.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=GROUND_TRUTH_DATA_DIR,
        help="Base directory to save the single highest-scoring ground truth .npz file.",
    )
    parser.add_argument(
        "--ship",
        type=str,
        required=True,
        help="Specific ship type (e.g., standard, corvette).",
    )
    parser.add_argument(
        "--tech",
        type=str,
        required=True,
        help="Specific tech type (e.g., hyper, pulse).",
    )
    parser.add_argument(
        "--solve_type",
        type=str,
        default=None,
        help="Optional: Specific solve type (e.g., 4x2, 3x3).",
    )
    parser.add_argument(
        "--max_sc",
        type=int,
        default=4,
        help="Maximum number of supercharged slots to randomly generate for the target input grid.",
    )
    parser.add_argument(
        "--max_inactive",
        type=int,
        default=0,
        help="Maximum number of inactive cells to randomly generate for the target input grid.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Starting Ground Truth Data Generation (Processing All Input Grid Permutations)")
    print("=" * 80 + "\n")

    BATCH_SIZE = 256
    batch_X_supercharge = []
    batch_X_inactive_mask = []
    batch_y = []
    current_batch_size = 0
    batch_file_counter = 0

    all_module_data = get_all_module_data()

    # Determine grid dimensions once for total count calculation
    module_data_for_ship_for_dims = all_module_data.get(args.ship)
    if not module_data_for_ship_for_dims:
        print(f"Error: No module data found for ship '{args.ship}' for dimension calculation.")
        exit(1)
    training_module_ids_for_dims = get_training_module_ids(args.ship, args.tech)
    if not training_module_ids_for_dims:
        print(
            f"Error: No training modules found for {args.ship}/{args.tech}/{args.solve_type} for dimension calculation."
        )
        exit(1)
    module_count_for_dims = len(training_module_ids_for_dims)
    grid_w, grid_h = determine_window_dimensions(module_count_for_dims, args.tech, args.ship)

    total_input_grids = _calculate_total_grids(grid_w, grid_h, args.max_sc, args.max_inactive)
    print(f"Total unique input grid permutations to process: {total_input_grids}")

    # Use glob to find all relevant .npz files based on ship/tech/solve_type filters
    search_pattern = os.path.join(args.input_dir, args.ship, args.tech)
    if args.solve_type:
        search_pattern = os.path.join(search_pattern, args.solve_type)
    search_pattern = os.path.join(search_pattern, "*.npz")
    all_npz_files = glob.glob(search_pattern, recursive=True)

    if not all_npz_files:
        print(f"No .npz files found in {os.path.dirname(search_pattern)} matching the criteria.")
        exit(0)

    print(f"Scanning {len(all_npz_files)} .npz files for matching layouts...")

    # Iterate through all possible input grid configurations
    grid_generator = generate_all_input_grids(args.ship, args.tech, args.solve_type, args.max_sc, args.max_inactive)

    for input_grid_idx, (random_grid_obj, target_supercharge_np, target_inactive_mask_np) in enumerate(grid_generator):
        print(f"\nProcessing Input Grid {input_grid_idx + 1} of {total_input_grids}:")
        print_grid_compact(random_grid_obj)

        best_overall_score = -1.0
        best_overall_y = None
        best_overall_filepath = None
        best_overall_sample_index = -1

        for filepath in tqdm(all_npz_files, desc=f"Searching for matches for Input Grid {input_grid_idx + 1}"):
            try:
                # Determine ship, tech, solve_type from path for module data loading
                relative_path_parts = os.path.relpath(os.path.dirname(filepath), args.input_dir).split(os.sep)
                current_ship_from_path = relative_path_parts[0] if len(relative_path_parts) > 0 else None
                current_tech_from_path = relative_path_parts[1] if len(relative_path_parts) > 1 else None
                current_solve_type_from_path = relative_path_parts[2] if len(relative_path_parts) > 2 else None

                tech_modules = []
                ship_data = all_module_data.get(current_ship_from_path)
                if ship_data and "types" in ship_data:
                    for category_name, category_data in ship_data["types"].items():
                        if not isinstance(category_data, list):
                            continue
                        for tech_info in category_data:
                            if tech_info.get("key") == current_tech_from_path:
                                if (
                                    current_solve_type_from_path
                                    and tech_info.get("type") == current_solve_type_from_path
                                ) or (not current_solve_type_from_path and tech_info.get("type") is None):
                                    for module in tech_info.get("modules", []):
                                        tech_modules.append(module)
                                    break
                        if tech_modules:
                            break

                if not tech_modules:
                    continue

                module_id_mapping = get_module_id_mapping(tech_modules)
                reverse_module_id_mapping = {v: k for k, v in module_id_mapping.items()}

                data = np.load(filepath)
                X_supercharge_batch = data["X_supercharge"]
                X_inactive_mask_batch = data["X_inactive_mask"]
                y_batch = data["y"]

                for i in range(len(y_batch)):
                    current_supercharge_np = X_supercharge_batch[i]
                    current_inactive_mask_np = X_inactive_mask_batch[i]
                    current_y_np = y_batch[i]

                    # Compare with the randomly generated input grid
                    if np.array_equal(current_supercharge_np, target_supercharge_np) and np.array_equal(
                        current_inactive_mask_np, target_inactive_mask_np
                    ):
                        # Reconstruct the grid to calculate its score
                        grid_h, grid_w = current_y_np.shape
                        current_grid_reconstructed = Grid(grid_w, grid_h)

                        for r in range(grid_h):
                            for c in range(grid_w):
                                is_active = current_inactive_mask_np[r, c] == 0
                                is_supercharged = current_supercharge_np[r, c] == 1
                                module_class = current_y_np[r, c]

                                current_grid_reconstructed.set_active(c, r, is_active)
                                current_grid_reconstructed.set_supercharged(c, r, is_supercharged)

                                if module_class > 0:
                                    module_id = reverse_module_id_mapping.get(module_class)
                                    if module_id:
                                        module_data = next((m for m in tech_modules if m["id"] == module_id), None)
                                        if module_data:
                                            current_grid_reconstructed.set_module(c, r, module_id)
                                            current_grid_reconstructed.set_tech(c, r, current_tech_from_path)
                                            current_grid_reconstructed.set_type(c, r, module_data.get("type"))
                                            base_bonus_value = module_data.get("bonus")
                                            current_grid_reconstructed.set_bonus(
                                                c, r, base_bonus_value if base_bonus_value is not None else 0.0
                                            )
                                            current_grid_reconstructed.set_value(
                                                c, r, int(base_bonus_value) if base_bonus_value is not None else 0
                                            )
                                            current_grid_reconstructed.set_adjacency(c, r, module_data.get("adjacency"))
                                            current_grid_reconstructed.set_sc_eligible(
                                                c, r, module_data.get("sc_eligible")
                                            )

                        if current_tech_from_path:
                            score = calculate_grid_score(current_grid_reconstructed, current_tech_from_path)

                            if score > best_overall_score:
                                best_overall_score = score
                                best_overall_y = current_y_np
                                best_overall_filepath = filepath
                                best_overall_sample_index = i

            except Exception as e:
                print(f"Error processing file {filepath}: {e}")

        # Save the single best layout found for THIS input grid
        if best_overall_y is not None:
            # Original
            batch_X_supercharge.append(target_supercharge_np)
            batch_X_inactive_mask.append(target_inactive_mask_np)
            batch_y.append(best_overall_y)

            # Flip Left-Right
            batch_X_supercharge.append(np.fliplr(target_supercharge_np))
            batch_X_inactive_mask.append(np.fliplr(target_inactive_mask_np))
            batch_y.append(np.fliplr(best_overall_y))

            # Flip Up-Down
            batch_X_supercharge.append(np.flipud(target_supercharge_np))
            batch_X_inactive_mask.append(np.flipud(target_inactive_mask_np))
            batch_y.append(np.flipud(best_overall_y))

            # Flip Up-Down then Left-Right
            batch_X_supercharge.append(np.flipud(np.fliplr(target_supercharge_np)))
            batch_X_inactive_mask.append(np.flipud(np.fliplr(target_inactive_mask_np)))
            batch_y.append(np.flipud(np.fliplr(best_overall_y)))

            current_batch_size += 4  # Increment by 4 for the 4 augmented samples

            print(f"Found best layout for Input Grid {input_grid_idx + 1}. Score: {best_overall_score:.4f} (Augmented)")

            if current_batch_size >= BATCH_SIZE:
                _save_batch(
                    batch_X_supercharge,
                    batch_X_inactive_mask,
                    batch_y,
                    args.output_dir,
                    args.ship,
                    args.tech,
                    args.solve_type,
                    batch_file_counter,
                    grid_w,
                    grid_h,
                )
                batch_X_supercharge = []
                batch_X_inactive_mask = []
                batch_y = []
                current_batch_size = 0
                batch_file_counter += 1
        else:
            print(f"No matching layout found for Input Grid {input_grid_idx + 1}.")

    # Save any remaining data in the last batch
    if batch_X_supercharge:
        _save_batch(
            batch_X_supercharge,
            batch_X_inactive_mask,
            batch_y,
            args.output_dir,
            args.ship,
            args.tech,
            args.solve_type,
            batch_file_counter,
            grid_w,
            grid_h,
        )

    print("\n" + "=" * 80)
    print("Ground Truth Data Generation Complete.")
    print("=" * 80 + "\n")
