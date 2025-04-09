import torch
import torch.nn.functional as F # Import functional for softmax
import numpy as np
import random
import os
import sys

# --- Add project root to sys.path (adjust if needed) ---
# Assuming test_model.py is in the 'training' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
from training.train_model import ModulePlacementCNN# Import from the new training script
# Ensure Grid is imported correctly (assuming it's in optimizer or grid_utils)
try:
    from optimizer import Grid
except ImportError:
    from grid_utils import Grid # Fallback if Grid is in grid_utils.py

from optimizer import print_grid_compact, print_grid, get_tech_modules_for_training, calculate_grid_score
from modules import modules # Import your module definitions
from module_placement import place_module # To place modules on the grid

# --- Removed Logging Setup ---

def test_placement_model(
    model_path: str,
    input_supercharge_np: np.ndarray,
    input_inactive_mask_np: np.ndarray,
    module_id_mapping: dict,
    ship: str,
    tech: str,
    modules_data: dict,
    grid_height: int,
    grid_width: int,
    use_compact_print: bool = True
):
    """
    Loads a trained model, predicts placement for an input grid (with inactive cells),
    enforcing the one-module-per-type rule based on confidence, and prints it.

    Args:
        model_path (str): Path to the saved .pth model file.
        input_supercharge_np (np.ndarray): A 2D numpy array (height, width) representing
                                           the input grid (1 for supercharged, 0 otherwise).
                                           Shape should match model's expected input.
        input_inactive_mask_np (np.ndarray): A 2D numpy array (height, width) where
                                             1 indicates an inactive cell, 0 an active cell.
                                             Shape should match model's expected input.
        module_id_mapping (dict): The mapping from module ID to class index
                                  used during training for this tech.
        ship (str): The ship type (e.g., "standard").
        tech (str): The technology key (e.g., "hyper").
        modules_data (dict): The main 'modules' dictionary containing all module info.
        grid_height (int): The height of the grid the model was trained on.
        grid_width (int): The width of the grid the model was trained on.
        use_compact_print (bool): If True, use print_grid_compact, else print_grid.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    num_output_classes = len(module_id_mapping) + 1 # +1 for the background class

    # --- 1. Load Model ---
    print(f"Loading model from {model_path}...")
    try:
        model = ModulePlacementCNN(
            input_channels=1,
            grid_height=grid_height,
            grid_width=grid_width,
            num_output_classes=num_output_classes
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded successfully onto {device}.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Check if the model architecture (grid size, classes) matches the saved file.")
        return

    # --- 2. Prepare Input ---
    input_height, input_width = input_supercharge_np.shape
    if input_height != grid_height or input_width != grid_width:
        print(f"Warning: Input supercharge grid shape ({input_height}x{input_width}) differs from model's expected shape ({grid_height}x{grid_width}). Ensure input matches model dimensions.")

    input_tensor = torch.tensor(input_supercharge_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # --- 3. Get Prediction ---
    print("Generating prediction...")
    try:
        with torch.no_grad():
            # Get raw logits (scores) from the model
            output_logits = model(input_tensor) # Shape: [1, num_classes, height, width]
            # Optional: Convert logits to probabilities using Softmax for confidence comparison
            output_probs = F.softmax(output_logits, dim=1) # Apply softmax across the class dimension
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return

    # --- 4. Process Output - Confidence-Based Assignment ---
    # Move probabilities/logits back to CPU and remove batch dimension
    output_scores = output_probs.squeeze(0).cpu().numpy() # Shape: [num_classes, height, width]
    # If using logits directly: output_scores = output_logits.squeeze(0).cpu().numpy()

    reverse_module_mapping = {v: k for k, v in module_id_mapping.items()}
    reverse_module_mapping[0] = None # Background class

    tech_module_defs_list = get_tech_modules_for_training(modules_data, ship, tech)
    tech_module_defs_map = {m['id']: m for m in tech_module_defs_list} if tech_module_defs_list else {}
    if not tech_module_defs_map:
         print(f"Warning: No module definitions found for {ship}/{tech} in modules_data.")

    # Create a list of all possible non-background module placements with their scores
    potential_placements = []
    for class_idx in range(1, num_output_classes): # Skip background class 0
        module_id = reverse_module_mapping.get(class_idx)
        if module_id is None: continue # Should not happen if mapping is correct

        for y in range(grid_height):
            for x in range(grid_width):
                # Consider only active cells based on the input mask
                # Check bounds for safety
                if y >= input_inactive_mask_np.shape[0] or x >= input_inactive_mask_np.shape[1]:
                    continue
                is_active_from_mask = (input_inactive_mask_np[y, x] == 0)
                if is_active_from_mask:
                    score = output_scores[class_idx, y, x]
                    potential_placements.append({
                        "score": score,
                        "x": x,
                        "y": y,
                        "class_idx": class_idx,
                        "module_id": module_id
                    })

    # Sort potential placements by score (confidence) in descending order
    potential_placements.sort(key=lambda p: p["score"], reverse=True)

    # --- 5. Build Output Grid - Enforcing Rules ---
    output_grid = Grid(grid_width, grid_height)
    placed_module_ids = set()
    used_cells = set() # Keep track of cells that have been assigned a module
    placed_module_count = 0

    print("Building output grid (enforcing one module per type by confidence)...")

    # Initialize grid with active/inactive and supercharged status from input masks
    for y in range(grid_height):
        for x in range(grid_width):
             # Check bounds for safety
            if y >= input_inactive_mask_np.shape[0] or x >= input_inactive_mask_np.shape[1] or \
               y >= input_supercharge_np.shape[0] or x >= input_supercharge_np.shape[1]:
                continue # Skip if out of bounds for input arrays

            is_active = (input_inactive_mask_np[y, x] == 0)
            is_supercharged = (input_supercharge_np[y, x] == 1)
            output_grid.set_active(x, y, is_active)
            output_grid.set_supercharged(x, y, is_supercharged)
            # Initialize all cells as empty
            output_grid.set_module(x, y, None)
            output_grid.set_tech(x, y, None)

    # Iterate through sorted potential placements
    for placement in potential_placements:
        x, y = placement["x"], placement["y"]
        module_id = placement["module_id"]
        cell_coord = (x, y)

        # Check if module already placed OR cell already used
        if module_id in placed_module_ids or cell_coord in used_cells:
            continue

        # *** CORRECTION HERE ***
        # Check if cell is still active by accessing the grid's state
        try:
            if not output_grid.get_cell(x, y)["active"]:
                 continue
        except IndexError:
             print(f"Warning: Attempted to access cell ({x},{y}) out of bounds during placement check.")
             continue
        # *** END CORRECTION ***

        # Place the module
        module_data = tech_module_defs_map.get(module_id)
        if module_data:
            try:
                place_module(
                    output_grid, x, y, module_data["id"], module_data["label"],
                    tech, module_data["type"], module_data["bonus"],
                    module_data["adjacency"], module_data["sc_eligible"], module_data["image"]
                )
                placed_module_ids.add(module_id)
                used_cells.add(cell_coord)
                placed_module_count += 1
                # print(f"Placed {module_id} at ({x},{y}) with score {placement['score']:.4f}") # Debug print
            except Exception as e:
                print(f"Error placing module {module_id} at ({x},{y}): {e}")
                # Ensure cell is cleared if placement fails mid-process
                try:
                    output_grid.set_module(x, y, None)
                    output_grid.set_tech(x, y, None)
                except IndexError:
                     print(f"Warning: Attempted to clear cell ({x},{y}) out of bounds after placement error.")

        else:
            # This warning should ideally not trigger if mappings are correct
            print(f"Warning: Module data for '{module_id}' not found during placement.")

    print(f"Placed {placed_module_count} unique modules based on prediction confidence.")

    # --- 6. Print Grids ---
    # (Printing logic remains the same)
    print("\n--- Input Grid State (S=Supercharged, X=Inactive) ---")
    # Use input dimensions for the temporary input grid display
    input_height, input_width = input_supercharge_np.shape
    temp_input_grid = Grid(input_width, input_height)
    for y in range(input_height):
        for x in range(input_width):
            # Bounds check against input arrays
            if y >= input_inactive_mask_np.shape[0] or x >= input_inactive_mask_np.shape[1] or \
               y >= input_supercharge_np.shape[0] or x >= input_supercharge_np.shape[1]: continue

            is_inactive = (input_inactive_mask_np[y, x] == 1)
            is_supercharged = (input_supercharge_np[y, x] == 1)
            temp_input_grid.set_active(x, y, not is_inactive)
            temp_input_grid.set_supercharged(x, y, is_supercharged)
    print_grid_compact(temp_input_grid)

    print("\n--- Predicted Placement Grid (Rule Enforced) ---") # Updated title
    if use_compact_print:
        print_grid_compact(output_grid)
    else:
        try:
            score = calculate_grid_score(output_grid, tech)
            print(f"Calculated Score for Predicted Grid: {score:.4f}")
            print_grid(output_grid)
        except Exception as e:
            print(f"Error calculating score or printing full grid: {e}")
            print("Falling back to compact print:")
            print_grid_compact(output_grid)


# --- Example Usage ---
# (No changes needed in the __main__ block)
if __name__ == "__main__":
    # --- Consolidated Configuration ---
    config = {
        "test_ship": "standard",
        "test_tech": "cyclotron", # Technology to test
        "model_trained_grid_width": 4, # Grid width the model was trained on
        "model_trained_grid_height": 3, # Grid height the model was trained on
        "max_test_inactive": 1, # Max inactive cells for the random test input
        "max_test_supercharged": 4, # Max supercharged cells for the random test input
        "use_compact_print": False, # Use compact grid printing format?
        "model_dir": "../trained_models" # Relative path to model directory
    }
    # --- End Configuration ---

    # --- Model Path Setup ---
    script_dir = os.path.dirname(__file__)
    model_save_dir = os.path.abspath(os.path.join(script_dir, config["model_dir"]))
    model_filename = f"model_{config['test_ship']}_{config['test_tech']}.pth"
    full_model_path = os.path.join(model_save_dir, model_filename)

    # --- Get Module ID Mapping ---
    print(f"Generating module mapping for {config['test_ship']} / {config['test_tech']}...")
    tech_modules_list = get_tech_modules_for_training(modules, config['test_ship'], config['test_tech'])
    if not tech_modules_list:
        print(f"Error: No tech modules found for ship='{config['test_ship']}', tech='{config['test_tech']}'. Cannot test.")
        exit()
    tech_modules_list.sort(key=lambda m: m['id'])
    example_module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(tech_modules_list)}
    print(f"Using Module ID Mapping: {example_module_id_mapping}")
    expected_num_classes = len(tech_modules_list) + 1
    print(f"Expected number of output classes for model: {expected_num_classes}")

    # --- Generate a Random Input Grid (with inactive cells) ---
    test_input_grid_width = config["model_trained_grid_width"]
    test_input_grid_height = config["model_trained_grid_height"]
    print(f"Generating random {test_input_grid_height}x{test_input_grid_width} input grid for testing (with inactive cells)...")

    total_cells = test_input_grid_width * test_input_grid_height
    all_positions = [(y, x) for y in range(test_input_grid_height) for x in range(test_input_grid_width)]

    num_inactive = random.randint(0, min(config["max_test_inactive"], total_cells))
    inactive_indices = random.sample(all_positions, num_inactive) if num_inactive > 0 else []
    example_inactive_mask_np = np.zeros((test_input_grid_height, test_input_grid_width), dtype=np.int8)
    for y, x in inactive_indices:
        example_inactive_mask_np[y, x] = 1

    active_positions = [pos for pos in all_positions if pos not in inactive_indices]
    num_active_cells = len(active_positions)

    example_supercharge_np = np.zeros((test_input_grid_height, test_input_grid_width), dtype=np.float32)
    num_supercharged = 0
    if num_active_cells > 0:
        max_possible_supercharged = min(config["max_test_supercharged"], num_active_cells)
        num_supercharged = random.randint(0, max_possible_supercharged)
        if num_supercharged > 0:
            supercharged_indices = random.sample(active_positions, num_supercharged)
            for y, x in supercharged_indices:
                example_supercharge_np[y, x] = 1.0

    print(f"Input grid generated with {num_supercharged} supercharged slots and {num_inactive} inactive slots.")

    # --- Run the Test ---
    test_placement_model(
        full_model_path,
        example_supercharge_np,
        example_inactive_mask_np,
        example_module_id_mapping,
        config['test_ship'],
        config['test_tech'],
        modules,
        config['model_trained_grid_height'],
        config['model_trained_grid_width'],
        use_compact_print=config['use_compact_print']
    )

