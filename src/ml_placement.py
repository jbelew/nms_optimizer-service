# ml_placement.py
"""
This module contains the core logic for using a pre-trained machine learning
model to predict the optimal placement of technology modules within a grid.

The primary function, `ml_placement`, takes a grid state and technology context,
loads the appropriate PyTorch model, and generates a predicted layout. It also
includes an optional "polishing" step that uses a simulated annealing algorithm
to refine the model's output.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import logging
from typing import Optional, List, Tuple
from collections import Counter

# --- Add project root to sys.path if needed ---
# Ensure the project root is in the path if running this script directly
# or if imports fail in your environment.
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
try:
    from src.model_cache import get_model  # <<< Import the new model cache function

    # <<< Import both module definition sources >>>
    from src.modules_utils import get_tech_modules
    from src.data_loader import get_module_data, get_training_module_ids

    # from modules import modules as user_facing_modules # Keep if needed elsewhere, or pass in
    # <<< End import changes >>>
    from src.module_placement import place_module, clear_all_modules_of_tech
    from rust_scorer import calculate_grid_score as rust_calculate_grid_score
    from src.grid_display import print_grid_compact
    from src.grid_utils import Grid, apply_localized_grid_changes, restore_original_state
    from src.bonus_calculations import calculate_grid_score
    from src.data_definitions.model_mapping import (
        get_model_keys,
    )  # Import the modified get_model_keys
    from src.optimization.refinement import (
        simulated_annealing,
    )  # Import simulated_annealing
except ImportError as e:
    logging.error(f"ERROR in ml_placement.py: Failed to import dependencies - {e}")
    raise

# --- Constants ---
DEFAULT_MODEL_DIR = "trained_models"
DEFAULT_MODEL_GRID_WIDTH = 4
DEFAULT_MODEL_GRID_HEIGHT = 3
# --- End Constants ---


def ml_placement(
    grid: Grid,
    ship: str,  # This is the UI ship key
    tech: str,  # This is the UI tech key
    full_grid_original: Grid,  # The original full grid from optimize_placement
    start_x_original: int,  # The x-offset of this localized grid within the original full grid
    start_y_original: int,  # The y-offset of this localized grid within the original full grid
    original_state_map: dict,  # The map to restore original state of other tech modules
    player_owned_rewards: Optional[List[str]] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_grid_width: int = DEFAULT_MODEL_GRID_WIDTH,
    model_grid_height: int = DEFAULT_MODEL_GRID_HEIGHT,
    polish_result: bool = True,  # Flag for optional SA polishing
    progress_callback=None,
    run_id=None,
    stage=None,
    progress_offset=0,
    progress_scale=100,
    send_grid_updates=False,
    solve_type: Optional[str] = None,
    tech_modules: Optional[list] = None,
    available_modules: Optional[list] = None,
) -> Tuple[Optional[Grid], float]:
    """
    Uses a pre-trained Machine Learning model to predict module placement.
    Applies mapping (including reward variants and solve type) via get_model_keys.
    Uses UI-facing module definitions for the final grid construction.
    Optionally polishes the result using a quick simulated annealing run.

    Args:
        grid (Grid): The input grid state (active/inactive, supercharged).
        ship (str): The UI ship key.
        tech (str): The UI tech key.
        player_owned_rewards (Optional[List[str]]): List of reward module IDs owned.
        model_dir (str): Directory containing trained models.
        model_grid_width (int): Width the model expects.
        model_grid_height (int): Height the model expects.
        polish_result (bool): If True, run a quick SA polish on the ML output.
        solve_type (Optional[str]): The solve type, e.g., "max" or "normal".

    Returns:
        Tuple[Optional[Grid], float]: The predicted (and potentially polished) grid
                                       and its calculated score, or (None, 0.0) on failure.
    """
    start_time = time.time()
    # logging.info(f"INFO -- Attempting ML placement for UI keys: ship='{ship}', tech='{tech}' (Polish: {polish_result})") # Redundant with optimize_placement start

    if player_owned_rewards is None:
        player_owned_rewards = []
    set(player_owned_rewards)  # Use set for faster lookups

    # --- 1. Determine Model Keys using Mapping ---
    model_keys_info = get_model_keys(
        ship,
        tech,
        model_grid_width,
        model_grid_height,
        player_owned_rewards,  # pyright: ignore
        solve_type=solve_type,
        available_modules=available_modules,
    )
    filename_ship_key = model_keys_info["filename_ship_key"]
    filename_tech_key = model_keys_info["filename_tech_key"]
    module_def_ship_key = model_keys_info["module_def_ship_key"]
    module_def_tech_key = model_keys_info["module_def_tech_key"]

    # Important mapping info
    logging.info(f"ML Placement: UI keys ('{ship}', '{tech}') mapped to:")
    logging.info(f"       Filename keys: ('{filename_ship_key}', '{filename_tech_key}')")
    logging.info(f"       Module Def keys: ('{module_def_ship_key}', '{module_def_tech_key}')")

    # --- 2. Determine Model Path & Check Existence ---
    model_filename = f"model_{filename_ship_key}_{filename_tech_key}.pth"
    project_root = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(project_root, model_dir, model_filename)

    if not os.path.exists(model_path):
        # Important failure condition
        logging.warning(
            f"ML Placement: Model file for ('{filename_ship_key}', '{filename_tech_key}') not found at '{model_path}'. Cannot use ML."
        )
        return None, 0.0

    # --- 3. Get Module Mapping & Num Classes (using MODEL keys for model loading) ---
    # Use the new data loader to get the exact list of module IDs the model was trained on.
    training_module_ids = get_training_module_ids(module_def_ship_key, module_def_tech_key, solve_type=solve_type)

    if not training_module_ids:
        # Important failure condition
        logging.error(
            f"ML Placement: No TRAINING module IDs found for Module Def keys ('{module_def_ship_key}', '{module_def_tech_key}'). Cannot define model outputs."
        )
        return None, 0.0

    # --- Sort module IDs and create mappings ---
    training_module_ids.sort()  # Sort the IDs alphabetically for consistency
    module_id_mapping = {module_id: i + 1 for i, module_id in enumerate(training_module_ids)}
    num_output_classes = len(module_id_mapping) + 1  # +1 for background
    reverse_module_mapping = {v: k for k, v in module_id_mapping.items()}
    reverse_module_mapping[0] = "None"  # Background class is 0
    # logging.info(f"INFO -- ML Placement: Determined num_output_classes = {num_output_classes} based on Module Def keys ('{module_def_ship_key}', '{module_def_tech_key}').") # Less critical detail
    # --- End Model Loading Setup ---

    # --- 4. Get ACTUAL modules to place & their UI definitions (using UI keys + rewards) ---
    # Use the USER-FACING modules_data passed into the function
    modules_data = get_module_data(ship)
    if tech_modules is None:
        modules_to_place_list = get_tech_modules(modules_data, ship, tech, player_owned_rewards, solve_type=solve_type)
    else:
        modules_to_place_list = tech_modules

    if not modules_to_place_list:
        # Important warning
        logging.warning(
            f"ML Placement: No placeable modules found for UI keys '{ship}/{tech}' with rewards {player_owned_rewards}. Returning empty grid."
        )
        cleared_grid = grid.copy()
        clear_all_modules_of_tech(cleared_grid, tech)  # Clear the target tech
        return cleared_grid, 0.0

    # Create a map of module definitions based on the UI context for final placement
    ui_module_defs_map = {m["id"]: m for m in modules_to_place_list}

    # Count how many of each module ID we need to place
    modules_needed_count = Counter(m["id"] for m in modules_to_place_list)
    total_modules_to_place = len(modules_to_place_list)
    # logging.info(f"INFO -- ML Placement: Need to place {total_modules_to_place} modules based on UI keys/rewards: {dict(modules_needed_count)}") # Less critical detail
    # --- End UI Module Setup ---

    # --- 5. Load Model using Cache ---
    try:
        model = get_model(
            model_path=model_path,
            model_grid_width=model_grid_width,
            model_grid_height=model_grid_height,
            num_output_classes=num_output_classes,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # Ensure model is on the correct device
    except (FileNotFoundError, Exception):
        # Errors are already logged by get_model, so we can just return here.
        return None, 0.0

    # --- 6. Prepare Input Tensor ---
    input_supercharge_np = np.zeros((model_grid_height, model_grid_width), dtype=np.float32)
    # Create a simple mask of active cells to be used later, but not as a model input
    active_cell_mask = np.zeros((model_grid_height, model_grid_width), dtype=bool)

    for y in range(min(grid.height, model_grid_height)):
        for x in range(min(grid.width, model_grid_width)):
            try:
                cell = grid.get_cell(x, y)
                if cell and cell["active"]:
                    input_supercharge_np[y, x] = 1.0 if cell["supercharged"] else 0.0
                    active_cell_mask[y, x] = True
            except IndexError:
                logging.warning(f"Warning: Input grid access out of bounds at ({x},{y}) during tensor prep.")
                continue

    # Create a single-channel tensor for the supercharge grid.
    # The model now only expects this one channel.
    # Shape: (N, C, H, W) -> (1, 1, H, W)
    input_tensor = torch.tensor(input_supercharge_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # --- 7. Get Prediction ---
    # logging.info("ML Placement: Generating prediction...") # Less critical step
    try:
        with torch.no_grad():
            output_logits = model(input_tensor)
            output_probs = F.softmax(output_logits, dim=1)
    except Exception as e:
        # Important failure condition
        logging.error(f"ML Placement: Error during model prediction: {e}")
        return None, 0.0

    # --- 8. Process Output (Generate Potential Placements) ---
    output_scores = output_probs.squeeze(0).cpu().numpy()  # Shape: [num_classes, height, width]
    potential_placements = []
    for class_idx in range(1, num_output_classes):  # Skip background class 0
        predicted_module_id = reverse_module_mapping.get(class_idx)
        if predicted_module_id is None:
            continue

        # Check if this module_id is actually one we need to place for this user
        if predicted_module_id not in modules_needed_count:
            continue

        for y in range(model_grid_height):
            for x in range(model_grid_width):
                try:
                    if active_cell_mask[y, x]:  # Check if the cell is active
                        score = output_scores[class_idx, y, x]
                        potential_placements.append(
                            {
                                "score": score,
                                "x": x,
                                "y": y,
                                "class_idx": class_idx,
                                "module_id": predicted_module_id,
                            }
                        )
                except IndexError:
                    logging.warning(
                        f"Warning: Index out of bounds ({y},{x}) accessing input_inactive_mask_np or output_scores."
                    )
                    continue

    potential_placements.sort(key=lambda p: p["score"], reverse=True)
    # logging.info(f"INFO -- ML Placement: Generated {len(potential_placements)} potential placements for needed modules.") # Less critical detail

    # --- 9. Build Output Grid (Confidence-Based Assignment) ---
    predicted_grid = Grid(model_grid_width, model_grid_height)
    used_cells = set()
    placed_module_count = 0

    # Initialize grid structure (active/supercharged state) from input tensors
    for y in range(model_grid_height):
        for x in range(model_grid_width):
            try:
                is_active = active_cell_mask[y, x]
                is_supercharged = input_supercharge_np[y, x] == 1.0
                predicted_grid.set_active(x, y, is_active.item())
                predicted_grid.set_supercharged(x, y, bool(is_supercharged) and is_active.item())
                predicted_grid.set_module(x, y, None)
                predicted_grid.set_tech(x, y, None)
            except IndexError:
                logging.warning(f"Warning: Index out of bounds ({y},{x}) during predicted_grid initialization.")
                continue

    # logging.info(f"INFO -- ML Placement: Assigning modules based on confidence...") # Less critical step
    current_modules_needed = modules_needed_count.copy()

    for placement in potential_placements:
        if placed_module_count >= total_modules_to_place:
            break

        x, y = placement["x"], placement["y"]
        predicted_module_id = placement["module_id"]
        cell_coord = (x, y)

        if cell_coord in used_cells:
            continue
        if predicted_module_id is None or current_modules_needed.get(predicted_module_id, 0) <= 0:  # pyright: ignore
            continue

        try:
            if not predicted_grid.get_cell(x, y)["active"]:
                continue
        except IndexError:
            continue

        # --- Use UI Module Definition for Placement ---
        # Get the module definition based on the UI context (ship, tech, rewards)
        module_data_for_placement = ui_module_defs_map.get(predicted_module_id)

        if module_data_for_placement:
            try:
                # Use the original UI tech key when placing in the grid cell
                place_module(
                    predicted_grid,
                    x,
                    y,
                    module_data_for_placement["id"],
                    module_data_for_placement["label"],
                    tech,  # <<< Use original UI tech key here
                    module_data_for_placement.get("type"),
                    module_data_for_placement["bonus"],
                    module_data_for_placement["adjacency"],
                    module_data_for_placement["sc_eligible"],
                    module_data_for_placement["image"],  # This now comes from the UI definition
                )
                used_cells.add(cell_coord)
                current_modules_needed[predicted_module_id] -= 1
                placed_module_count += 1
                # logging.debug(f"ML Placement: Placed {predicted_module_id} at ({x},{y}) with score {placement['score']:.4f}") # DEBUG level

            except Exception as e:
                # Important error
                logging.error(f"ML Placement: Error placing module {predicted_module_id} at ({x},{y}): {e}")
                try:
                    predicted_grid.set_module(x, y, None)
                    predicted_grid.set_tech(x, y, None)
                except IndexError:
                    pass
        else:
            # This warning indicates the predicted module ID isn't in the list of modules
            # the user actually has available for this tech/reward combo.
            # Important warning
            logging.warning(
                f"ML Placement: Predicted module ID '{predicted_module_id}' not found in UI module definitions map. Skipping placement."
            )

    # --- Final Check and Logging ---
    # Important outcome
    logging.info(f"ML Placement: Placed {placed_module_count} modules.")
    if placed_module_count < total_modules_to_place:
        remaining_needed = {k: v for k, v in current_modules_needed.items() if v > 0}
        # Important warning
        logging.warning(
            f"ML Placement: Could only place {placed_module_count} out of {total_modules_to_place} expected modules for {tech}. "
            f"Remaining needed: {remaining_needed}"
        )

    # --- 10. Calculate Initial Score ---
    predicted_score = calculate_grid_score(predicted_grid, tech)
    # Important score info
    logging.info(f"ML Placement: Initial Score (before polish): {predicted_score:.4f}")
    # print_grid(predicted_grid) # Optional: print grid before polish

    # Emit the full, reconstituted grid state after ML placement, before polishing
    if send_grid_updates and progress_callback:
        reconstituted_grid_for_update = full_grid_original.copy()
        # Clear the tech modules from the full grid before applying localized changes
        clear_all_modules_of_tech(reconstituted_grid_for_update, tech)
        apply_localized_grid_changes(
            reconstituted_grid_for_update,
            predicted_grid,
            tech,
            start_x_original,
            start_y_original,
        )
        restore_original_state(reconstituted_grid_for_update, original_state_map)

        progress_data = {
            "tech": tech,
            "run_id": run_id,
            "stage": stage,
            "progress_percent": 0,
            "best_score": predicted_score,
            "status": "in_progress",
            "best_grid": reconstituted_grid_for_update.to_dict(),
        }
        progress_callback(progress_data)

    # --- 11. Optional Polishing Step ---
    if polish_result:
        # Indicates polishing is starting
        logging.info("ML Placement: Attempting SA polish on ML result...")
        grid_to_polish = predicted_grid.copy()
        print_grid_compact(grid_to_polish)  # Optional: print grid before polish

        polish_params = {
            "initial_temperature": 1500,  # Keep starting temp
            "cooling_rate": 0.98,  # Slow down cooling slightly (was 0.98)
            "stopping_temperature": 0.5,  # Keep stopping temp
            "iterations_per_temp": 35,  # More iterations per step (was 20)
            "initial_swap_probability": 0.40,  # Keep initial swap chance
            "final_swap_probability": 0.25,  # Keep final swap chance
            "start_from_current_grid": True,  # Keep this True for polishing
            "max_processing_time": 10.0,  # Increase max time slightly (was 5.0)
        }

        # logging.info(f"ML Placement: Using SA polish params: {polish_params}") # Less critical detail

        try:
            # Run SA starting from the ML-generated grid, using UI keys
            polished_grid, polished_bonus = simulated_annealing(
                grid_to_polish,
                ship,  # Use original UI ship key
                get_module_data(ship),
                tech,  # Use original UI tech key
                player_owned_rewards=player_owned_rewards,
                progress_callback=progress_callback,
                run_id=run_id,
                stage=stage,
                progress_offset=progress_offset,
                progress_scale=progress_scale,
                full_grid=full_grid_original,
                send_grid_updates=send_grid_updates,
                start_x=start_x_original,
                start_y=start_y_original,
                solve_type=solve_type,
                tech_modules=tech_modules,
                **polish_params,
            )

            print_grid_compact(polished_grid)

            if polished_grid is not None and polished_bonus > predicted_score:
                # Important score improvement
                logging.info(
                    f"ML Placement: SA polish improved score from {predicted_score:.4f} to {polished_bonus:.4f}. Updating grid."
                )
                # print_grid(polished_grid) # Optional: print grid after polish
                predicted_grid = polished_grid
                predicted_score = polished_bonus
            elif polished_grid is not None:
                # Important outcome
                logging.info(
                    f"ML Placement: SA polish did not improve score ({polished_bonus:.4f} vs {predicted_score:.4f}). Keeping ML result."
                )
            else:
                # Important failure condition
                logging.warning("ML Placement: SA polish failed or returned None. Keeping ML result.")

        except ValueError as e:
            # Important warning
            logging.warning(f"ML Placement: SA polishing step failed with ValueError: {e}. Skipping polish.")
        except Exception as e:
            # Important warning
            logging.warning(f"ML Placement: Unexpected error during SA polishing: {e}. Skipping polish.")
    # --- End Optional Polishing Step ---

    # --- 12. Return Result ---
    end_time = time.time()
    # Final result
    logging.info(f"ML Placement: Finished in {end_time - start_time:.2f} seconds. Final Score: {predicted_score:.4f}")
    # print_grid(predicted_grid) # Optional: print final grid

    return predicted_grid, predicted_score
