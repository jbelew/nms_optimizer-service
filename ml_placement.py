# ml_placement.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import time

# --- Add project root to sys.path if needed (depending on execution context) ---
# import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
# Adjust relative paths if ml_placement.py is not in the root directory
try:
    from training.train_model import ModulePlacementCNN
    # <<< Import BOTH functions >>>
    from modules_data import get_tech_modules, get_tech_modules_for_training
    from module_placement import place_module # Removed clear_all_modules_of_tech import (not used here)
    from bonus_calculations import calculate_grid_score
    from grid_utils import Grid
except ImportError as e:
    print(f"ERROR in ml_placement.py: Failed to import dependencies - {e}")
    print("Ensure this file is placed correctly relative to other project modules.")
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
    player_owned_rewards=None, # <<< Keep this argument
    model_dir: str = DEFAULT_MODEL_DIR,
    model_grid_width: int = DEFAULT_MODEL_GRID_WIDTH,
    model_grid_height: int = DEFAULT_MODEL_GRID_HEIGHT,
):
    """
    Uses a pre-trained Machine Learning model to predict module placement.

    Applies a confidence-based assignment strategy to ensure one module per type
    is placed in the best predicted active slot, respecting player-owned rewards.

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
    print(f"INFO -- Attempting ML placement for ship: '{ship}' -- tech: '{tech}'")

    # --- Handle default player_owned_rewards ---
    if player_owned_rewards is None:
        player_owned_rewards = []
    # Convert to set for faster lookups during assignment
    player_owned_rewards_set = set(player_owned_rewards)
    # --- End Handle default ---

    # --- 1. Determine Model Path & Check Existence ---
    model_filename = f"model_{ship}_{tech}.pth"
    # Construct path relative to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    model_path = os.path.join(project_root, model_dir, model_filename)

    if not os.path.exists(model_path):
        print(f"INFO -- ML Placement: Model file not found at '{model_path}'. Cannot use ML.")
        return None, 0.0

    # --- 2. Get Module Mapping & Num Classes (FOR MODEL LOADING) ---
    training_modules_list = get_tech_modules_for_training(modules_data, ship, tech)
    if not training_modules_list:
        print(f"ERROR -- ML Placement: No TRAINING modules found for {ship}/{tech}. Cannot define model outputs.")
        return None, 0.0

    training_modules_list.sort(key=lambda m: m['id'])
    # This mapping is used to interpret the model's output classes
    module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(training_modules_list)}
    # <<< Calculate num_output_classes based on the TRAINING list >>>
    num_output_classes = len(module_id_mapping) + 1 # +1 for the background class (index 0)
    reverse_module_mapping = {v: k for k, v in module_id_mapping.items()}
    reverse_module_mapping[0] = None # Background class maps to no module
    #print(f"DEBUG -- ml_placement: Determined num_output_classes = {num_output_classes} based on training modules.")

    # <<< Create a map of module definitions from the TRAINING list >>>
    # This map is used to check the 'type' of the predicted module ID.
    training_module_defs_map = {m['id']: m for m in training_modules_list}
    #print(f"DEBUG -- ml_placement: Built training module definitions map.")

    # --- Get the ACTUAL modules to place for this specific request (FOR ASSIGNMENT LATER) ---
    # <<< Use the RUNTIME list (with rewards) for placement logic >>>
    modules_to_place_list = get_tech_modules(modules_data, ship, tech, player_owned_rewards)
    if not modules_to_place_list:
         print(f"INFO -- ML Placement: No modules to place for {ship}/{tech} based on player rewards. Returning empty grid.")
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
    # No need for tech_module_defs_map based on this list, we use the training one

    # --- 3. Load Model ---
    print(f"INFO -- ML Placement: Loading model from {model_path}...")
    try:
        # <<< Instantiate model using num_output_classes from TRAINING list >>>
        model = ModulePlacementCNN(
            input_channels=1,
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
        print(f"INFO -- ML Placement: Model loaded successfully onto {device}.")
    except FileNotFoundError:
        # This case should be caught by os.path.exists, but added for robustness
        print(f"ERROR -- ML Placement: Model file confirmed missing at {model_path}.")
        return None, 0.0
    except Exception as e:
        print(f"ERROR -- ML Placement: Failed to load model state_dict from {model_path}: {e}")
        print("       Check if model architecture (grid size, classes) matches the saved file.")
        return None, 0.0

    # --- 4. Prepare Input Tensor ---
    input_supercharge_np = np.zeros((model_grid_height, model_grid_width), dtype=np.float32)
    input_inactive_mask_np = np.ones((model_grid_height, model_grid_width), dtype=np.int8) # 1=inactive

    # Populate based on the input 'grid', clamping to model dimensions
    for y in range(min(grid.height, model_grid_height)):
        for x in range(min(grid.width, model_grid_width)):
            try: # Add try-except for safety
                cell = grid.get_cell(x, y)
                if cell["active"]:
                    input_supercharge_np[y, x] = 1.0 if cell["supercharged"] else 0.0
                    input_inactive_mask_np[y, x] = 0 # Mark as active in the mask
            except IndexError:
                print(f"Warning: Input grid access out of bounds at ({x},{y}) during tensor prep.")
                continue # Skip if out of bounds

    input_tensor = torch.tensor(input_supercharge_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # --- 5. Get Prediction ---
    print("INFO -- ML Placement: Generating prediction...")
    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            output_logits = model(input_tensor) # Shape: [1, num_classes, height, width]
            output_probs = F.softmax(output_logits, dim=1) # Apply across class dimension
    except Exception as e:
        print(f"ERROR -- ML Placement: Error during model prediction: {e}")
        return None, 0.0

    # --- 6. Process Output (Confidence-Based Assignment) ---
    output_scores = output_probs.squeeze(0).cpu().numpy() # Shape: [num_classes, height, width]
    # potential_placements generation uses reverse_module_mapping from TRAINING list - this is correct

    potential_placements = []
    for class_idx in range(1, num_output_classes): # Skip background class 0
        module_id = reverse_module_mapping.get(class_idx)
        if module_id is None: continue

        for y in range(model_grid_height):
            for x in range(model_grid_width):
                try: # Add try-except for safety
                    if input_inactive_mask_np[y, x] == 0: # Only active cells
                        score = output_scores[class_idx, y, x]
                        potential_placements.append(
                            {"score": score, "x": x, "y": y, "class_idx": class_idx, "module_id": module_id}
                        )
                except IndexError:
                     print(f"Warning: Index out of bounds ({y},{x}) accessing input_inactive_mask_np or output_scores.")
                     continue # Skip if out of bounds

    potential_placements.sort(key=lambda p: p["score"], reverse=True)

    # --- 7. Build Output Grid (Enforcing Rules - Instance-Based Assignment) ---
    predicted_grid = Grid(model_grid_width, model_grid_height)

    # Initialize grid structure (active/supercharged)
    for y in range(model_grid_height):
        for x in range(model_grid_width):
            try: # Add try-except for safety
                is_active_np = (input_inactive_mask_np[y, x] == 0)
                is_supercharged_np = (input_supercharge_np[y, x] == 1.0)
                # Ensure boolean values are standard Python bool
                py_bool_is_active = bool(is_active_np)
                py_bool_is_supercharged = bool(is_supercharged_np and py_bool_is_active)
                predicted_grid.set_active(x, y, py_bool_is_active)
                predicted_grid.set_supercharged(x, y, py_bool_is_supercharged and py_bool_is_active)
                predicted_grid.set_module(x, y, None) # Ensure grid starts empty
                predicted_grid.set_tech(x, y, None)
            except IndexError:
                 print(f"Warning: Index out of bounds ({y},{x}) during predicted_grid initialization.")
                 continue # Skip if out of bounds

    # --- Modified Assignment Logic ---
    # <<< Use the RUNTIME list (modules_to_place_list) for assignment >>>
    num_modules_to_place = len(modules_to_place_list) # Count from the actual list
    placed_module_indices = set() # Track indices from modules_to_place_list
    used_cells = set()            # Track (x, y) coordinates
    assignments = {}              # Store final assignments: (x, y) -> module_index_in_modules_to_place_list

    print(f"INFO -- ML Placement: Assigning {num_modules_to_place} module instances based on confidence...")
    # <<< Use the RUNTIME list for debug print >>>
    #print(f"DEBUG -- ML Placement: modules_to_place_list (runtime) (len={num_modules_to_place}): {[m['id'] for m in modules_to_place_list]}")
    #print(f"DEBUG -- ML Placement: Player owned rewards set: {player_owned_rewards_set}")

    for placement in potential_placements:
        # Stop if all modules have been assigned a spot
        if len(placed_module_indices) == num_modules_to_place:
            break

        x, y = placement["x"], placement["y"]
        predicted_module_id = placement["module_id"] # The ID predicted by the model (from training map)
        cell_coord = (x, y)

        # Skip if cell is already used or inactive
        if cell_coord in used_cells: continue
        try:
            if not predicted_grid.get_cell(x, y)["active"]: continue
        except IndexError: continue

        # <<< --- CORRECTED CHECK: Reward Module Ownership --- >>>
        # 1. Get the definition of the predicted module (from training data map)
        predicted_module_def = training_module_defs_map.get(predicted_module_id)

        # 2. Check if it's a reward type AND if the player owns it
        if predicted_module_def and predicted_module_def.get("type") == "reward":
            if predicted_module_id not in player_owned_rewards_set:
                # print(f"DEBUG -- Skipping placement of reward module '{predicted_module_id}' at ({x},{y}) because player does not own it.")
                continue # Skip this potential placement if it's a reward and not owned
        # <<< --- END CORRECTED CHECK --- >>>

        # --- Find an unplaced instance in modules_to_place_list (RUNTIME list) ---
        found_module_instance_index = -1
        # Iterate through the ACTUAL list of modules for this request
        for idx, module_instance in enumerate(modules_to_place_list):
            # Check if this instance matches the predicted ID and hasn't been placed yet
            if idx not in placed_module_indices and module_instance["id"] == predicted_module_id:
                found_module_instance_index = idx
                break # Found the first available instance from the actual list

        # If an unplaced instance *from the actual list* was found, assign it
        if found_module_instance_index != -1:
            # print(f"DEBUG -- Assigning instance {found_module_instance_index} ('{predicted_module_id}') from actual list to ({x},{y}), score {placement['score']:.4f}")
            assignments[cell_coord] = found_module_instance_index # Store index from modules_to_place_list
            used_cells.add(cell_coord)
            placed_module_indices.add(found_module_instance_index) # Mark index from modules_to_place_list as placed

    # --- Place modules based on final assignments ---
    placed_module_count = 0
    for (x, y), module_idx in assignments.items():
        try:
            # --- CRITICAL CHANGE: Get data from modules_to_place_list (RUNTIME list) ---
            module_data = modules_to_place_list[module_idx]
            place_module(
                predicted_grid, x, y,
                module_data["id"], module_data["label"], tech, module_data["type"],
                module_data["bonus"], module_data["adjacency"],
                module_data["sc_eligible"], module_data["image"]
            )
            placed_module_count += 1
        except Exception as e:
            print(f"ERROR -- ML Placement: Error during final place_module for instance {module_idx} ('{module_data.get('id', 'N/A')}') at ({x},{y}): {e}")
            # Optionally clear the cell again if placement fails mid-way
            try:
                predicted_grid.set_module(x, y, None)
                predicted_grid.set_tech(x, y, None)
            except IndexError: pass

    # --- Logging update ---
    print(f"INFO -- ML Placement: Assigned and placed {placed_module_count} modules.")
    if placed_module_count < num_modules_to_place:
         unplaced_count = num_modules_to_place - placed_module_count
         print(f"WARNING -- ML Placement: Could only place {placed_module_count} out of {num_modules_to_place} expected modules for {tech}. {unplaced_count} module(s) remain unplaced.")
         # <<< Use the RUNTIME list for debug print >>>
         unplaced_indices = set(range(num_modules_to_place)) - placed_module_indices
         unplaced_ids = [modules_to_place_list[i]['id'] for i in unplaced_indices]
         print(f"       Unplaced module IDs: {unplaced_ids}")

    # --- 8. Calculate Score ---
    predicted_score = calculate_grid_score(predicted_grid, tech)
    print(f"INFO -- ML Placement: Calculated Score for predicted grid: {predicted_score:.4f}")

    # --- 9. Return Result ---
    end_time = time.time()
    print(f"INFO -- ML Placement finished in {end_time - start_time:.2f} seconds.")

    return predicted_grid, predicted_score

# Optional: Add to __all__ if you plan to import * from this file elsewhere
__all__ = ["ml_placement"]
