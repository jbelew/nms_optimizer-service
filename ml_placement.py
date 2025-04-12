# ml_placement.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import logging # Import logging
from typing import Optional, List, Tuple # Add typing for clarity
from collections import Counter # Import Counter

# --- Add project root to sys.path if needed (depending on execution context) ---
# import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
# Adjust relative paths if ml_placement.py is not in the root directory
try:
    from training.model_definition import ModulePlacementCNN
    # Import BOTH functions for getting modules
    from modules_data import get_tech_modules, get_tech_modules_for_training
    from module_placement import place_module # For placing modules on the grid
    from bonus_calculations import calculate_grid_score
    from grid_utils import Grid
except ImportError as e:
    logging.error(f"ERROR in ml_placement.py: Failed to import dependencies - {e}")
    logging.error("Ensure this file is placed correctly relative to other project modules.")
    # Depending on how critical this is, you might raise the error or exit
    raise

# --- Constants for ML Placement (Adjust if needed) ---
# Directory where trained models are saved (relative to project root)
DEFAULT_MODEL_DIR = "training/trained_models"
# Default grid dimensions the models were trained on
DEFAULT_MODEL_GRID_WIDTH = 4
DEFAULT_MODEL_GRID_HEIGHT = 3
# --- End Constants ---


def ml_placement(
    grid: Grid,
    ship: str,
    modules_data: dict,
    tech: str,
    player_owned_rewards: Optional[List[str]] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_grid_width: int = DEFAULT_MODEL_GRID_WIDTH,
    model_grid_height: int = DEFAULT_MODEL_GRID_HEIGHT,
) -> Tuple[Optional[Grid], float]: # Return type hint
    """
    Uses a pre-trained Machine Learning model to predict module placement.

    Applies a confidence-based assignment strategy: iterates through potential
    placements sorted by confidence, placing the first valid, owned, and required
    module instance encountered for an available cell.

    Args:
        grid (Grid): The initial grid layout (provides active/supercharged state).
        ship (str): The ship type (e.g., "standard").
        modules_data (dict): The main 'modules' dictionary containing all module info.
        tech (str): The technology key (e.g., "hyper").
        player_owned_rewards (list, optional): List of reward module IDs the player owns. Defaults to None.
        model_dir (str): Directory containing the trained PyTorch models (.pth files),
                         relative to the project root.
        model_grid_width (int): The width of the grid the model was trained on.
        model_grid_height (int): The height of the grid the model was trained on.

    Returns:
        tuple: (predicted_grid, predicted_score) containing the grid with ML-predicted
               placements and its calculated score. Returns (None, 0.0) if the model
               cannot be loaded or prediction fails.
    """
    start_time = time.time()
    logging.info(f"INFO -- Attempting ML placement for ship: '{ship}' -- tech: '{tech}'")

    # --- Handle default player_owned_rewards ---
    if player_owned_rewards is None:
        player_owned_rewards = []
    # Convert to set for faster lookups during assignment
    player_owned_rewards_set = set(player_owned_rewards)
    logging.info(f"INFO -- ML Placement: Using player owned rewards: {player_owned_rewards_set}")
    # --- End Handle default ---

    # --- 1. Determine Model Path & Check Existence ---
    model_filename = f"model_{ship}_{tech}.pth"
    # Construct path relative to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".")) # Assumes ml_placement.py is in root
    model_path = os.path.join(project_root, model_dir, model_filename)

    if not os.path.exists(model_path):
        logging.warning(f"INFO -- ML Placement: Model file not found at '{model_path}'. Cannot use ML.")
        return None, 0.0

    # --- 2. Get Module Mapping & Num Classes (FOR MODEL LOADING) ---
    # Use the list of ALL modules possible for this tech (including unowned rewards)
    # to define the model's output layer structure correctly.
    training_modules_list = get_tech_modules_for_training(modules_data, ship, tech)
    if not training_modules_list:
        logging.error(f"ERROR -- ML Placement: No TRAINING modules found for {ship}/{tech}. Cannot define model outputs.")
        return None, 0.0

    training_modules_list.sort(key=lambda m: m['id'])
    # This mapping is used to interpret the model's output classes
    module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(training_modules_list)}
    num_output_classes = len(module_id_mapping) + 1 # +1 for the background class (index 0)
    reverse_module_mapping = {v: k for k, v in module_id_mapping.items()}
    reverse_module_mapping[0] = None # Background class maps to no module
    logging.info(f"INFO -- ML Placement: Determined num_output_classes = {num_output_classes} based on training modules.")

    # Create a map of module definitions from the TRAINING list
    # This map is used to check the 'type' of the predicted module ID.
    training_module_defs_map = {m['id']: m for m in training_modules_list}
    logging.info(f"INFO -- ML Placement: Built training module definitions map.")

    # --- Get the ACTUAL modules to place for this specific request (FOR ASSIGNMENT LATER) ---
    # Use the list filtered by player_owned_rewards for the placement logic.
    modules_to_place_list = get_tech_modules(modules_data, ship, tech, player_owned_rewards)
    if not modules_to_place_list:
         logging.warning(f"INFO -- ML Placement: No modules to place for {ship}/{tech} based on player rewards. Returning empty grid.")
         # Return an empty grid matching model dimensions but with 0 score
         empty_grid = Grid(model_grid_width, model_grid_height)
         # Initialize active/supercharged based on input grid
         for y in range(min(grid.height, model_grid_height)):
             for x in range(min(grid.width, model_grid_width)):
                 try:
                     cell = grid.get_cell(x, y)
                     empty_grid.set_active(x, y, cell["active"])
                     empty_grid.set_supercharged(x, y, cell["supercharged"] and cell["active"])
                 except IndexError: pass # Ignore if input grid is smaller
         return empty_grid, 0.0

    # Use a Counter to track how many of each required module ID still need placing
    modules_needed_count = Counter([m["id"] for m in modules_to_place_list])
    total_modules_to_place = len(modules_to_place_list)
    logging.info(f"INFO -- ML Placement: Need to place {total_modules_to_place} modules: {dict(modules_needed_count)}")

    # --- 3. Load Model ---
    logging.info(f"INFO -- ML Placement: Loading model from {model_path}...")
    try:
        # <<< CHANGE: Set input_channels=2 >>>
        model = ModulePlacementCNN(
            input_channels=2, # Use 2 input channels
            grid_height=model_grid_height,
            grid_width=model_grid_width,
            num_output_classes=num_output_classes # Use the count derived from training modules
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the state dictionary onto the correct device first
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict) # Now the shapes should match
        model.to(device)
        model.eval() # Set model to evaluation mode
        logging.info(f"INFO -- ML Placement: Model loaded successfully onto {device}.")
    except FileNotFoundError:
        # This case should be caught by os.path.exists, but added for robustness
        logging.error(f"ERROR -- ML Placement: Model file confirmed missing at {model_path}.")
        return None, 0.0
    except Exception as e:
        logging.error(f"ERROR -- ML Placement: Failed to load model state_dict from {model_path}: {e}")
        logging.error("       Check if model architecture (grid size, classes, channels) matches the saved file.")
        return None, 0.0

    # --- 4. Prepare Input Tensor ---
    # Input 1: Supercharge status (1 if active AND supercharged, 0 otherwise)
    input_supercharge_np = np.zeros((model_grid_height, model_grid_width), dtype=np.float32)
    # Input 2: Inactive mask (1 if inactive, 0 if active)
    input_inactive_mask_np = np.ones((model_grid_height, model_grid_width), dtype=np.int8) # 1=inactive

    # Populate based on the input 'grid', clamping to model dimensions
    for y in range(min(grid.height, model_grid_height)):
        for x in range(min(grid.width, model_grid_width)):
            try: # Add try-except for safety
                cell = grid.get_cell(x, y)
                is_active = cell["active"]
                is_supercharged = cell["supercharged"]

                if is_active:
                    input_supercharge_np[y, x] = 1.0 if is_supercharged else 0.0
                    input_inactive_mask_np[y, x] = 0 # Mark as active in the mask
                # else: input_inactive_mask_np remains 1 (inactive)

            except IndexError:
                logging.warning(f"Warning: Input grid access out of bounds at ({x},{y}) during tensor prep.")
                continue # Skip if out of bounds

    # <<< CHANGE: Create the 2-channel input tensor >>>
    # Create individual tensors
    input_supercharge_tensor = torch.tensor(input_supercharge_np, dtype=torch.float32).unsqueeze(0) # Shape [1, H, W]
    input_inactive_tensor = torch.tensor(input_inactive_mask_np, dtype=torch.float32).unsqueeze(0) # Shape [1, H, W]

    # Stack along the channel dimension (dim=0) and add batch dimension (dim=0)
    input_tensor = torch.stack([input_supercharge_tensor, input_inactive_tensor], dim=1).to(device) # Shape [1, 2, H, W]
    # --- End Change ---

    # --- 5. Get Prediction ---
    logging.info("INFO -- ML Placement: Generating prediction...")
    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            output_logits = model(input_tensor) # Shape: [1, num_classes, height, width]
            output_probs = F.softmax(output_logits, dim=1) # Apply across class dimension
    except Exception as e:
        logging.error(f"ERROR -- ML Placement: Error during model prediction: {e}")
        return None, 0.0

    # --- 6. Process Output (Generate Potential Placements) ---
    output_scores = output_probs.squeeze(0).cpu().numpy() # Shape: [num_classes, height, width]

    potential_placements = []
    for class_idx in range(1, num_output_classes): # Skip background class 0
        module_id = reverse_module_mapping.get(class_idx)
        if module_id is None: continue # Should not happen if mapping is correct

        for y in range(model_grid_height):
            for x in range(model_grid_width):
                try: # Add try-except for safety
                    # Use the inactive mask derived from the input grid
                    if input_inactive_mask_np[y, x] == 0: # Only consider active cells
                        score = output_scores[class_idx, y, x]
                        potential_placements.append(
                            {"score": score, "x": x, "y": y, "class_idx": class_idx, "module_id": module_id}
                        )
                except IndexError:
                     logging.warning(f"Warning: Index out of bounds ({y},{x}) accessing input_inactive_mask_np or output_scores.")
                     continue # Skip if out of bounds

    # Sort all potential placements by score (confidence) in descending order
    potential_placements.sort(key=lambda p: p["score"], reverse=True)
    logging.info(f"INFO -- ML Placement: Generated {len(potential_placements)} potential placements.")

    # --- 7. Build Output Grid (Confidence-Based Assignment - REVISED) ---
    predicted_grid = Grid(model_grid_width, model_grid_height)
    used_cells = set() # Keep track of cells that have been assigned a module
    placed_module_count = 0

    # Initialize grid structure (active/supercharged) based on input masks
    for y in range(model_grid_height):
        for x in range(model_grid_width):
            try: # Add try-except for safety
                is_active_np = (input_inactive_mask_np[y, x] == 0)
                is_supercharged_np = (input_supercharge_np[y, x] == 1.0)
                py_bool_is_active = bool(is_active_np)
                py_bool_is_supercharged = bool(is_supercharged_np and py_bool_is_active)

                predicted_grid.set_active(x, y, py_bool_is_active)
                predicted_grid.set_supercharged(x, y, py_bool_is_supercharged) # SC only if active handled by mask
                predicted_grid.set_module(x, y, None) # Ensure grid starts empty
                predicted_grid.set_tech(x, y, None)
            except IndexError:
                 logging.warning(f"Warning: Index out of bounds ({y},{x}) during predicted_grid initialization.")
                 continue # Skip if out of bounds

    logging.info(f"INFO -- ML Placement: Assigning modules based on confidence...")

    # Iterate through sorted potential placements
    for placement in potential_placements:
        # Stop if all required modules have been placed
        if placed_module_count >= total_modules_to_place:
            break

        x, y = placement["x"], placement["y"]
        predicted_module_id = placement["module_id"] # The ID predicted by the model
        cell_coord = (x, y)

        # 1. Check if cell is already used
        if cell_coord in used_cells:
            # logging.debug(f"ML Placement: Skipping ({x},{y}) - cell already used.")
            continue

        # 2. Check if cell is active (redundant due to potential_placements creation, but safe)
        try:
            if not predicted_grid.get_cell(x, y)["active"]:
                # logging.debug(f"ML Placement: Skipping ({x},{y}) - cell inactive in predicted_grid.")
                continue
        except IndexError:
            logging.warning(f"ML Placement: Cell ({x},{y}) out of bounds during active check.")
            continue

        # 3. Check Ownership for Reward Modules
        # Use the training map to get the definition of the *predicted* module
        predicted_module_def = training_module_defs_map.get(predicted_module_id)
        if predicted_module_def and predicted_module_def.get("type") == "reward":
            if predicted_module_id not in player_owned_rewards_set:
                # logging.debug(f"ML Placement: Skipping placement of unowned reward {predicted_module_id} at ({x},{y})")
                continue # Skip this potential placement entirely if reward is not owned

        # 4. Check if this module ID is still needed (based on the runtime list)
        if modules_needed_count.get(predicted_module_id, 0) > 0:
            # Place the module
            # --- CRITICAL: Get module data from the TRAINING map for full details ---
            # We use the training map because the runtime list might have modified 'type' for rewards
            module_data_for_placement = training_module_defs_map.get(predicted_module_id)
            if module_data_for_placement:
                try:
                    # Use the correct 'type' from the original definition for placement
                    actual_type = module_data_for_placement["type"]
                    # If it was a reward that got converted to 'bonus' in the runtime list,
                    # we still want to place it using its original 'reward' type internally if needed,
                    # or just use the type from the training map. Let's use the training map's type.

                    logging.debug(f"ML Placement: Placing {predicted_module_id} at ({x},{y}) with score {placement['score']:.4f}")
                    place_module(
                        predicted_grid, x, y,
                        module_data_for_placement["id"],
                        module_data_for_placement["label"],
                        tech,
                        actual_type, # Use type from training map
                        module_data_for_placement["bonus"],
                        module_data_for_placement["adjacency"],
                        module_data_for_placement["sc_eligible"],
                        module_data_for_placement["image"]
                    )
                    used_cells.add(cell_coord)
                    modules_needed_count[predicted_module_id] -= 1 # Decrement needed count
                    placed_module_count += 1
                except Exception as e:
                    logging.error(f"ML Placement: Error placing module {predicted_module_id} at ({x},{y}): {e}")
                    # Ensure cell is cleared if placement fails mid-process
                    try:
                        predicted_grid.set_module(x, y, None)
                        predicted_grid.set_tech(x, y, None)
                    except IndexError: pass # Ignore if clearing fails
            else:
                # This warning should ideally not trigger if mappings are correct
                logging.warning(f"ML Placement: Module data for '{predicted_module_id}' not found in training map during placement.")
        # else: # Module not needed or all instances already placed
            # logging.debug(f"ML Placement: Skipping placement of {predicted_module_id} at ({x},{y}) - not needed.")

    # --- Final Check and Logging ---
    logging.info(f"INFO -- ML Placement: Placed {placed_module_count} modules.")
    if placed_module_count < total_modules_to_place:
        remaining_needed = {k: v for k, v in modules_needed_count.items() if v > 0}
        logging.warning(
            f"WARNING -- ML Placement: Could only place {placed_module_count} out of {total_modules_to_place} expected modules for {tech}. "
            f"Remaining needed: {remaining_needed}"
        )

    # --- 8. Calculate Score ---
    predicted_score = calculate_grid_score(predicted_grid, tech)
    logging.info(f"INFO -- ML Placement: Calculated Score for predicted grid: {predicted_score:.4f}")

    # --- 9. Return Result ---
    end_time = time.time()
    logging.info(f"INFO -- ML Placement finished in {end_time - start_time:.2f} seconds.")

    return predicted_grid, predicted_score

# Optional: Add to __all__ if you plan to import * from this file elsewhere
__all__ = ["ml_placement"]
