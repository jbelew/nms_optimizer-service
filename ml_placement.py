# ml_placement.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import logging
from typing import Optional, List, Tuple, Dict # <<< Keep Dict
from collections import Counter

# --- Add project root to sys.path if needed ---
# ... (keep if necessary)

# --- Imports from your project ---
try:
    from training.model_definition import ModulePlacementCNN
    from modules_data import get_tech_modules, get_tech_modules_for_training
    from module_placement import place_module
    from bonus_calculations import calculate_grid_score
    from grid_utils import Grid
    # <<< Import the modified get_model_keys >>>
    from model_mapping import get_model_keys
    # <<< Import simulated_annealing >>>
    from simulated_annealing import simulated_annealing
except ImportError as e:
    logging.error(f"ERROR in ml_placement.py: Failed to import dependencies - {e}")
    raise

# --- Constants ---
DEFAULT_MODEL_DIR = "training/trained_models"
DEFAULT_MODEL_GRID_WIDTH = 4
DEFAULT_MODEL_GRID_HEIGHT = 3
# --- End Constants ---

def ml_placement(
    grid: Grid,
    ship: str, # This is the UI ship key
    modules_data: dict,
    tech: str, # This is the UI tech key
    player_owned_rewards: Optional[List[str]] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_grid_width: int = DEFAULT_MODEL_GRID_WIDTH,
    model_grid_height: int = DEFAULT_MODEL_GRID_HEIGHT,
    polish_result: bool = True, # <<< Add new parameter for polishing
) -> Tuple[Optional[Grid], float]:
    """
    Uses a pre-trained Machine Learning model to predict module placement.
    Applies mapping (including reward variants) via get_model_keys.
    Optionally polishes the result using a quick simulated annealing run.

    Args:
        grid (Grid): The input grid state (active/inactive, supercharged).
        ship (str): The UI ship key.
        modules_data (dict): The main modules dictionary.
        tech (str): The UI tech key.
        player_owned_rewards (Optional[List[str]]): List of reward module IDs owned.
        model_dir (str): Directory containing trained models.
        model_grid_width (int): Width the model expects.
        model_grid_height (int): Height the model expects.
        polish_result (bool): If True, run a quick SA polish on the ML output.

    Returns:
        Tuple[Optional[Grid], float]: The predicted (and potentially polished) grid
                                       and its calculated score, or (None, 0.0) on failure.
    """
    start_time = time.time()
    logging.info(f"INFO -- Attempting ML placement for UI keys: ship='{ship}', tech='{tech}' (Polish: {polish_result})") # Log polish flag

    # --- Handle default player_owned_rewards ---
    if player_owned_rewards is None: player_owned_rewards = []
    player_owned_rewards_set = set(player_owned_rewards) # Keep the set for logging/other checks if needed
    logging.info(f"INFO -- ML Placement: Using player owned rewards: {player_owned_rewards_set}")

    # --- 1. Determine Model Keys using Mapping (Now includes reward logic) ---
    # <<< Pass player_owned_rewards to get_model_keys >>>
    # Note: Removed modules_data from the call as per the latest model_mapping.py version
    model_ship_key, model_tech_key = get_model_keys(
        ship, tech, player_owned_rewards
    )
    logging.info(f"INFO -- ML Placement: Mapping UI keys ('{ship}', '{tech}') to final model keys ('{model_ship_key}', '{model_tech_key}')")

    # --- 2. Determine Model Path & Check Existence ---
    # <<< Use the keys directly returned by get_model_keys >>>
    model_filename = f"model_{model_ship_key}_{model_tech_key}.pth" # No suffix needed here
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    model_path = os.path.join(project_root, model_dir, model_filename)

    # <<< Simplified Check: Just check if the specific model file exists >>>
    if not os.path.exists(model_path):
        logging.warning(f"INFO -- ML Placement: Model file not found at '{model_path}'. Cannot use ML.")
        # Optional: Add fallback logic here if you want to try the base 'pulse' model
        # if the 'photonix' model isn't found, but keeping it simple for now.
        # if model_tech_key == "photonix":
        #     logging.warning("INFO -- ML Placement: Photonix model not found, trying base 'pulse' model...")
        #     base_model_tech_key = "pulse" # Or get from mapping again without rewards
        #     base_model_filename = f"model_{model_ship_key}_{base_model_tech_key}.pth"
        #     base_model_path = os.path.join(project_root, model_dir, base_model_filename)
        #     if os.path.exists(base_model_path):
        #         model_path = base_model_path
        #         model_filename = base_model_filename
        #         logging.warning("INFO -- ML Placement: Found and using base 'pulse' model.")
        #     else:
        #         logging.warning("INFO -- ML Placement: Base 'pulse' model also not found.")
        #         return None, 0.0
        # else: # If it wasn't the special case, just fail
        return None, 0.0

    # --- 3. Get Module Mapping & Num Classes (using UI keys for consistency) ---
    # This part remains the same - based on *all* potential modules for the UI keys
    training_modules_list = get_tech_modules_for_training(modules_data, ship, tech) # Use UI keys
    if not training_modules_list:
        logging.error(f"ERROR -- ML Placement: No TRAINING modules found for UI keys {ship}/{tech}. Cannot define model outputs.")
        return None, 0.0
    # ... (rest of num_output_classes calculation remains the same) ...
    training_modules_list.sort(key=lambda m: m['id'])
    module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(training_modules_list)}
    num_output_classes = len(module_id_mapping) + 1
    reverse_module_mapping = {v: k for k, v in module_id_mapping.items()}
    reverse_module_mapping[0] = None
    logging.info(f"INFO -- ML Placement: Determined num_output_classes = {num_output_classes} based on UI keys '{ship}/{tech}'.")
    training_module_defs_map = {m['id']: m for m in training_modules_list}


    # --- 4. Get ACTUAL modules to place (using UI keys for runtime filtering) ---
    # This part remains the same
    modules_to_place_list = get_tech_modules(modules_data, ship, tech, player_owned_rewards) # Use UI keys
    # ... (rest of modules_to_place_list handling remains the same) ...
    if not modules_to_place_list:
         logging.warning(f"INFO -- ML Placement: No modules to place for UI keys {ship}/{tech} based on player rewards. Returning empty grid.")
         # ... (empty grid creation logic remains the same) ...
         empty_grid = Grid(model_grid_width, model_grid_height)
         for y in range(min(grid.height, model_grid_height)):
             for x in range(min(grid.width, model_grid_width)):
                 try:
                     cell = grid.get_cell(x, y)
                     empty_grid.set_active(x, y, cell["active"])
                     empty_grid.set_supercharged(x, y, cell["supercharged"] and cell["active"])
                 except IndexError: pass
         return empty_grid, 0.0
    modules_needed_count = Counter([m["id"] for m in modules_to_place_list])
    total_modules_to_place = len(modules_to_place_list)
    logging.info(f"INFO -- ML Placement: Need to place {total_modules_to_place} modules: {dict(modules_needed_count)}")


    # --- 5. Load Model (using the determined model_path) ---
    logging.info(f"INFO -- ML Placement: Loading model from {model_path}...") # Path now includes specific tech key via get_model_keys
    try:
        model = ModulePlacementCNN(
            input_channels=2, # Assuming 2 channels: supercharge + inactive mask
            grid_height=model_grid_height,
            grid_width=model_grid_width,
            num_output_classes=num_output_classes # Should be consistent
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logging.info(f"INFO -- ML Placement: Model '{model_filename}' loaded successfully onto {device}.") # Log loaded filename
    except FileNotFoundError:
        logging.error(f"ERROR -- ML Placement: Model file confirmed missing at {model_path}.")
        return None, 0.0
    except Exception as e:
        logging.error(f"ERROR -- ML Placement: Failed to load model state_dict from {model_path}: {e}")
        logging.error("       Check if model architecture (grid size, classes, channels) matches the saved file.")
        return None, 0.0

    # --- 6. Prepare Input Tensor ---
    input_supercharge_np = np.zeros((model_grid_height, model_grid_width), dtype=np.float32)
    input_inactive_mask_np = np.ones((model_grid_height, model_grid_width), dtype=np.int8) # 1=inactive
    for y in range(min(grid.height, model_grid_height)):
        for x in range(min(grid.width, model_grid_width)):
            try:
                cell = grid.get_cell(x, y)
                is_active = cell["active"]
                is_supercharged = cell["supercharged"]
                if is_active:
                    input_supercharge_np[y, x] = 1.0 if is_supercharged else 0.0
                    input_inactive_mask_np[y, x] = 0
            except IndexError:
                logging.warning(f"Warning: Input grid access out of bounds at ({x},{y}) during tensor prep.")
                continue
    input_supercharge_tensor = torch.tensor(input_supercharge_np, dtype=torch.float32).unsqueeze(0)
    input_inactive_tensor = torch.tensor(input_inactive_mask_np, dtype=torch.float32).unsqueeze(0)
    input_tensor = torch.stack([input_supercharge_tensor, input_inactive_tensor], dim=1).to(device)

    # --- 7. Get Prediction ---
    logging.info("INFO -- ML Placement: Generating prediction...")
    try:
        with torch.no_grad():
            output_logits = model(input_tensor)
            output_probs = F.softmax(output_logits, dim=1)
    except Exception as e:
        logging.error(f"ERROR -- ML Placement: Error during model prediction: {e}")
        return None, 0.0

    # --- 8. Process Output (Generate Potential Placements) ---
    output_scores = output_probs.squeeze(0).cpu().numpy()
    potential_placements = []
    for class_idx in range(1, num_output_classes):
        module_id = reverse_module_mapping.get(class_idx)
        if module_id is None: continue
        for y in range(model_grid_height):
            for x in range(model_grid_width):
                try:
                    if input_inactive_mask_np[y, x] == 0:
                        score = output_scores[class_idx, y, x]
                        potential_placements.append(
                            {"score": score, "x": x, "y": y, "class_idx": class_idx, "module_id": module_id}
                        )
                except IndexError:
                     logging.warning(f"Warning: Index out of bounds ({y},{x}) accessing input_inactive_mask_np or output_scores.")
                     continue
    potential_placements.sort(key=lambda p: p["score"], reverse=True)
    logging.info(f"INFO -- ML Placement: Generated {len(potential_placements)} potential placements.")

    # --- 9. Build Output Grid (Confidence-Based Assignment) ---
    predicted_grid = Grid(model_grid_width, model_grid_height)
    used_cells = set()
    placed_module_count = 0
    for y in range(model_grid_height): # Initialize grid structure
        for x in range(model_grid_width):
            try:
                is_active_np = (input_inactive_mask_np[y, x] == 0)
                is_supercharged_np = (input_supercharge_np[y, x] == 1.0)
                py_bool_is_active = bool(is_active_np)
                py_bool_is_supercharged = bool(is_supercharged_np and py_bool_is_active)
                predicted_grid.set_active(x, y, py_bool_is_active)
                predicted_grid.set_supercharged(x, y, py_bool_is_supercharged)
                predicted_grid.set_module(x, y, None)
                predicted_grid.set_tech(x, y, None)
            except IndexError:
                 logging.warning(f"Warning: Index out of bounds ({y},{x}) during predicted_grid initialization.")
                 continue

    logging.info(f"INFO -- ML Placement: Assigning modules based on confidence...")
    for placement in potential_placements: # Assign modules
        if placed_module_count >= total_modules_to_place: break
        x, y = placement["x"], placement["y"]
        predicted_module_id = placement["module_id"]
        cell_coord = (x, y)
        if cell_coord in used_cells: continue
        try:
            if not predicted_grid.get_cell(x, y)["active"]: continue
        except IndexError: continue

        predicted_module_def = training_module_defs_map.get(predicted_module_id)
        if predicted_module_def and predicted_module_def.get("type") == "reward":
            if predicted_module_id not in player_owned_rewards_set: continue

        if modules_needed_count.get(predicted_module_id, 0) > 0:
            module_data_for_placement = training_module_defs_map.get(predicted_module_id)
            if module_data_for_placement:
                try:
                    actual_type = module_data_for_placement["type"]
                    logging.debug(f"ML Placement: Placing {predicted_module_id} at ({x},{y}) with score {placement['score']:.4f}")
                    place_module(
                        predicted_grid, x, y,
                        module_data_for_placement["id"], module_data_for_placement["label"],
                        tech, # Use the original UI tech key for the grid cell
                        actual_type, module_data_for_placement["bonus"],
                        module_data_for_placement["adjacency"], module_data_for_placement["sc_eligible"],
                        module_data_for_placement["image"]
                    )
                    used_cells.add(cell_coord)
                    modules_needed_count[predicted_module_id] -= 1
                    placed_module_count += 1
                except Exception as e:
                    logging.error(f"ML Placement: Error placing module {predicted_module_id} at ({x},{y}): {e}")
                    try:
                        predicted_grid.set_module(x, y, None)
                        predicted_grid.set_tech(x, y, None)
                    except IndexError: pass
            else:
                logging.warning(f"ML Placement: Module data for '{predicted_module_id}' not found in training map during placement.")

    # --- Final Check and Logging ---
    logging.info(f"INFO -- ML Placement: Placed {placed_module_count} modules.")
    if placed_module_count < total_modules_to_place:
        remaining_needed = {k: v for k, v in modules_needed_count.items() if v > 0}
        logging.warning(
            f"WARNING -- ML Placement: Could only place {placed_module_count} out of {total_modules_to_place} expected modules for {tech}. "
            f"Remaining needed: {remaining_needed}"
        )

    # --- 10. Calculate Initial Score ---
    # Calculate score *before* potential polishing
    predicted_score = calculate_grid_score(predicted_grid, tech)
    logging.info(f"INFO -- ML Placement: Initial Score (before polish): {predicted_score:.4f}")

    # --- 11. Optional Polishing Step ---
    if polish_result:
        logging.info("INFO -- ML Placement: Attempting SA polish on ML result...")
        grid_to_polish = predicted_grid.copy() # Work on a copy

        # Define parameters for a *more thorough* SA polish run
        polish_params = {
            "initial_temperature": 3000,
            "cooling_rate": 0.97,
            "stopping_temperature": 2.0,
            "iterations_per_temp": 30,
            "initial_swap_probability": 0.45,
            "final_swap_probability": 0.30,
            "start_from_current_grid": True,
            "max_processing_time": 5.0, # e.g., max 5 seconds for polish
        }
        logging.info(f"INFO -- ML Placement: Using SA polish params: {polish_params}") # Log the params being used

        try:
            polished_grid, polished_bonus = simulated_annealing(
                grid_to_polish,
                ship,
                modules_data,
                tech,
                player_owned_rewards,
                **polish_params # Unpack parameters including the new flag
            )

            if polished_grid is not None and polished_bonus > predicted_score:
                logging.info(f"INFO -- ML Placement: SA polish improved score from {predicted_score:.4f} to {polished_bonus:.4f}. Updating grid.")
                predicted_grid = polished_grid # Update the main grid if improved
                predicted_score = polished_bonus # Update the score
            elif polished_grid is not None:
                logging.info(f"INFO -- ML Placement: SA polish did not improve score ({polished_bonus:.4f} vs {predicted_score:.4f}). Keeping ML result.")
            else:
                logging.info("INFO -- ML Placement: SA polish failed or returned None. Keeping ML result.")

        except ValueError as e:
            logging.warning(f"WARNING -- ML Placement: SA polishing step failed with ValueError: {e}. Skipping polish.")
        except Exception as e:
            logging.warning(f"WARNING -- ML Placement: Unexpected error during SA polishing: {e}. Skipping polish.")
    # --- End Optional Polishing Step ---

    # --- 12. Return Result ---
    end_time = time.time()
    logging.info(f"INFO -- ML Placement finished in {end_time - start_time:.2f} seconds. Final Score: {predicted_score:.4f}")

    return predicted_grid, predicted_score # Return the final grid and score

# Optional: Add to __all__
__all__ = ["ml_placement"]
