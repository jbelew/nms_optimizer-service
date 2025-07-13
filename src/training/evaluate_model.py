# evaluate_model.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import glob
import time
import argparse
from tqdm import tqdm # For progress bar

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
from training.train_model import ModulePlacementCNN, PlacementDataset # Import model and dataset
from modules import modules # Your module definitions
from modules_data import get_tech_modules_for_training # To get module mapping
from grid_utils import Grid # Grid class
from bonus_calculations import calculate_grid_score # Scoring function
from module_placement import place_module # To build grids
from simulated_annealing import simulated_annealing # For comparison (optional)

# --- Import Metrics Library ---
try:
    import torchmetrics
except ImportError:
    print("torchmetrics not found. Please install it: pip install torchmetrics")
    sys.exit(1)

# --- Configuration ---
DEFAULT_TEST_DATA_DIR = "generated_batches" # Directory containing test .npz files
DEFAULT_MODEL_DIR = "training/trained_models" # Directory containing trained models

# --- Helper Function to Reconstruct Grid from Target Tensor ---
def reconstruct_grid_from_target(
    target_tensor: np.ndarray,
    input_supercharge_np: np.ndarray,
    input_inactive_mask_np: np.ndarray,
    reverse_module_mapping: dict,
    tech_module_defs_map: dict,
    tech: str,
    grid_height: int,
    grid_width: int,
) -> Grid:
    """
    Creates a Grid object based on the ground truth target tensor and input state.
    """
    grid = Grid(grid_width, grid_height)
    for y in range(grid_height):
        for x in range(grid_width):
            # Set active/supercharged state from input masks
            is_active = input_inactive_mask_np[y, x] == 0
            is_supercharged = input_supercharge_np[y, x] == 1.0
            grid.set_active(x, y, is_active)
            grid.set_supercharged(x, y, is_supercharged and is_active) # SC only if active

            # Get module ID from target tensor
            class_idx = target_tensor[y, x]
            module_id = reverse_module_mapping.get(class_idx)

            if module_id is not None and is_active:
                module_data = tech_module_defs_map.get(module_id)
                if module_data:
                    try:
                        place_module(
                            grid, x, y,
                            module_data["id"], module_data["label"], tech,
                            module_data["type"], module_data["bonus"],
                            module_data["adjacency"], module_data["sc_eligible"],
                            module_data["image"]
                        )
                    except Exception as e:
                        print(f"Warning: Error placing module {module_id} during ground truth reconstruction at ({x},{y}): {e}")
                        grid.set_module(x, y, None) # Ensure cell is cleared on error
                        grid.set_tech(x, y, None)
                else:
                    # Should not happen if mappings are correct
                    print(f"Warning: Module data for ground truth ID '{module_id}' not found.")
                    grid.set_module(x, y, None)
                    grid.set_tech(x, y, None)
            else:
                # Background class or inactive cell
                grid.set_module(x, y, None)
                grid.set_tech(x, y, None)
    return grid

# --- Helper Function to Build Predicted Grid (Confidence-Based) ---
def build_predicted_grid(
    output_probs: np.ndarray,
    input_supercharge_np: np.ndarray,
    input_inactive_mask_np: np.ndarray,
    reverse_module_mapping: dict,
    tech_module_defs_map: dict,
    tech: str,
    grid_height: int,
    grid_width: int,
) -> Grid:
    """
    Builds the final predicted grid using confidence-based assignment.
    (Similar logic to ml_placement.py and test_model.py)
    """
    num_output_classes = output_probs.shape[0]
    potential_placements = []
    for class_idx in range(1, num_output_classes): # Skip background class 0
        module_id = reverse_module_mapping.get(class_idx)
        if module_id is None: continue

        for y in range(grid_height):
            for x in range(grid_width):
                if input_inactive_mask_np[y, x] == 0: # Only active cells
                    score = output_probs[class_idx, y, x]
                    potential_placements.append(
                        {"score": score, "x": x, "y": y, "class_idx": class_idx, "module_id": module_id}
                    )

    potential_placements.sort(key=lambda p: p["score"], reverse=True)

    predicted_grid = Grid(grid_width, grid_height)
    placed_module_ids = set()
    used_cells = set()

    # Initialize grid structure
    for y in range(grid_height):
        for x in range(grid_width):
            is_active = input_inactive_mask_np[y, x] == 0
            is_supercharged = input_supercharge_np[y, x] == 1.0
            predicted_grid.set_active(x, y, is_active)
            predicted_grid.set_supercharged(x, y, is_supercharged and is_active)
            predicted_grid.set_module(x, y, None)
            predicted_grid.set_tech(x, y, None)

    # Assign modules based on confidence
    for placement in potential_placements:
        x, y = placement["x"], placement["y"]
        module_id = placement["module_id"]
        cell_coord = (x, y)

        if module_id in placed_module_ids or cell_coord in used_cells: continue
        try:
            if not predicted_grid.get_cell(x, y)["active"]: continue
        except IndexError: continue

        module_data = tech_module_defs_map.get(module_id)
        if module_data:
            try:
                place_module(
                    predicted_grid, x, y,
                    module_data["id"], module_data["label"], tech,
                    module_data["type"], module_data["bonus"],
                    module_data["adjacency"], module_data["sc_eligible"],
                    module_data["image"]
                )
                placed_module_ids.add(module_id)
                used_cells.add(cell_coord)
            except Exception as e:
                print(f"Warning: Error placing module {module_id} during prediction grid build at ({x},{y}): {e}")
                try:
                    predicted_grid.set_module(x, y, None)
                    predicted_grid.set_tech(x, y, None)
                except IndexError: pass
        else:
             print(f"Warning: Module data for predicted ID '{module_id}' not found.")

    return predicted_grid


# --- Main Evaluation Function ---
def evaluate_model(
    model_path: str,
    test_data_dir: str,
    ship: str,
    tech: str,
    grid_width: int,
    grid_height: int,
    batch_size: int = 32,
    run_sa_comparison: bool = False,
):
    """
    Evaluates a trained model on a test dataset.
    """
    print(f"\n{'='*10} Evaluating Model {'='*10}")
    print(f"Model: {model_path}")
    print(f"Test Data Dir: {test_data_dir}")
    print(f"Ship: {ship}, Tech: {tech}")
    print(f"Grid: {grid_width}x{grid_height}")
    print(f"Run SA Comparison: {run_sa_comparison}")

    # --- 1. Setup: Device, Mappings, Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get module mapping (consistent with training)
    tech_modules_list = get_tech_modules_for_training(modules, ship, tech)
    if not tech_modules_list:
        print(f"Error: No training modules found for {ship}/{tech}. Cannot evaluate.")
        return
    tech_modules_list.sort(key=lambda m: m['id'])
    module_id_mapping = {module["id"]: i + 1 for i, module in enumerate(tech_modules_list)}
    reverse_module_mapping = {v: k for k, v in module_id_mapping.items()}
    reverse_module_mapping[0] = None # Background class
    num_output_classes = len(module_id_mapping) + 1
    tech_module_defs_map = {m['id']: m for m in tech_modules_list}

    # Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    try:
        model = ModulePlacementCNN(
            input_channels=1, grid_height=grid_height, grid_width=grid_width, num_output_classes=num_output_classes
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Load Test Data ---
    file_pattern = os.path.join(test_data_dir, f"data_{ship}_{tech}_{grid_width}x{grid_height}_*.npz")
    test_files = glob.glob(file_pattern)
    if not test_files:
        print(f"Error: No test data files found matching pattern '{file_pattern}'.")
        return

    print(f"Found {len(test_files)} test data files. Loading...")
    all_X_test = []
    all_y_test = []
    for filepath in test_files:
        try:
            with np.load(filepath) as npz_file:
                # Assuming test data has same structure 'X', 'y' as training
                x_batch = npz_file["X"] # Input: Supercharge state (0 or 1)
                y_batch = npz_file["y"] # Target: Module class index (0 for background)
                # Basic validation
                if x_batch.shape[1:] != (grid_height, grid_width) or y_batch.shape[1:] != (grid_height, grid_width):
                    print(f"Warning: Shape mismatch in {filepath}. Skipping.")
                    continue
                all_X_test.append(x_batch)
                all_y_test.append(y_batch)
        except Exception as e:
            print(f"Error loading test data from {filepath}: {e}. Skipping.")

    if not all_X_test:
        print("Error: No valid test data could be loaded.")
        return

    try:
        X_test_np = np.concatenate(all_X_test, axis=0)
        y_test_np = np.concatenate(all_y_test, axis=0)
        print(f"Loaded {len(X_test_np)} total test samples.")
    except Exception as e:
        print(f"Error concatenating test data arrays: {e}")
        return

    # Create Dataset and DataLoader
    try:
        # Input X needs to be float for the model
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
        # Target y needs to be long for loss/metrics
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.long)
        test_dataset = PlacementDataset(X_test_tensor, y_test_tensor)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle test data
            num_workers=min(2, os.cpu_count()),
            pin_memory=torch.cuda.is_available(),
        )
    except Exception as e:
        print(f"Error creating test DataLoader: {e}")
        return

    # --- 3. Initialize Metrics ---
    criterion = torch.nn.CrossEntropyLoss()
    # Use 'macro' average for mIoU and Accuracy to treat all classes equally
    # Ignore index 0 (background) if desired, but usually included.
    test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_output_classes, average='macro').to(device)
    test_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_output_classes, average='macro').to(device)
    # Add other metrics if needed (Precision, Recall, F1)
    # test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_output_classes, average='macro').to(device)

    total_test_loss = 0.0
    all_predicted_scores = []
    all_ground_truth_scores = []
    all_sa_scores = [] # Optional

    # --- 4. Evaluation Loop ---
    print("Starting evaluation...")
    start_eval_time = time.time()
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)

            # --- Raw Prediction Metrics ---
            outputs_logits = model(inputs)
            loss = criterion(outputs_logits, targets)
            total_test_loss += loss.item()

            preds_classes = torch.argmax(outputs_logits, dim=1)
            test_accuracy.update(preds_classes, targets)
            test_iou.update(preds_classes, targets)
            # test_f1.update(preds_classes, targets)

            # --- Domain-Specific Score Calculation (Per Sample) ---
            output_probs_batch = F.softmax(outputs_logits, dim=1).cpu().numpy()
            inputs_np_batch = inputs.squeeze(1).cpu().numpy() # Remove channel dim
            targets_np_batch = targets.cpu().numpy()

            for j in range(inputs.size(0)): # Iterate through samples in the batch
                # --- Input Grid State Reconstruction ---
                # We need both supercharge state (from X) and inactive state.
                # The current X only stores supercharge. We need to infer inactive.
                # ASSUMPTION: The training data generation process ensures that
                # if a cell has a module in y (target), it must be active.
                # If y is 0 (background), it *could* be active or inactive.
                # Let's reconstruct the inactive mask based on the target 'y'.
                # A more robust way would be to save the inactive mask during data generation.
                input_supercharge_np = inputs_np_batch[j]
                target_np = targets_np_batch[j]
                # Infer inactive mask: 0 if target > 0, 1 otherwise (potential issue: empty active cells)
                # A better approach is needed if empty active cells are common and need distinction.
                # For now, let's assume target > 0 implies active.
                # Let's refine: Assume input 0 means not supercharged OR inactive.
                # Assume target 0 means background OR inactive.
                # If target > 0, cell MUST be active.
                # If target == 0 AND input == 0, cell MIGHT be inactive or just empty active.
                # --> This highlights a limitation: We can't perfectly reconstruct the inactive mask
                #     from only X (supercharge) and y (target classes).
                # --> BEST SOLUTION: Modify generate_data.py to save the inactive mask alongside X and y.
                # --> WORKAROUND (Less Accurate): Assume all cells are active unless proven otherwise.
                #     This might overestimate performance if the model places things in truly inactive spots.
                # Let's proceed with the WORKAROUND for now, but acknowledge its limitation.
                input_inactive_mask_np = np.zeros_like(target_np, dtype=np.int8) # Assume all active initially

                # Build Predicted Grid
                predicted_grid = build_predicted_grid(
                    output_probs_batch[j], input_supercharge_np, input_inactive_mask_np,
                    reverse_module_mapping, tech_module_defs_map, tech, grid_height, grid_width
                )
                predicted_score = calculate_grid_score(predicted_grid, tech)
                all_predicted_scores.append(predicted_score)

                # Build Ground Truth Grid
                ground_truth_grid = reconstruct_grid_from_target(
                    target_np, input_supercharge_np, input_inactive_mask_np,
                    reverse_module_mapping, tech_module_defs_map, tech, grid_height, grid_width
                )
                ground_truth_score = calculate_grid_score(ground_truth_grid, tech)
                all_ground_truth_scores.append(ground_truth_score)

                # Optional: Run Simulated Annealing for Comparison
                if run_sa_comparison:
                    # Create a base grid reflecting the input state for SA
                    sa_input_grid = Grid(grid_width, grid_height)
                    for y_sa in range(grid_height):
                        for x_sa in range(grid_width):
                             is_active = input_inactive_mask_np[y_sa, x_sa] == 0
                             is_supercharged = input_supercharge_np[y_sa, x_sa] == 1.0
                             sa_input_grid.set_active(x_sa, y_sa, is_active)
                             sa_input_grid.set_supercharged(x_sa, y_sa, is_supercharged and is_active)
                             sa_input_grid.set_module(x_sa, y_sa, None) # Start empty
                             sa_input_grid.set_tech(x_sa, y_sa, None)
                    try:
                        # Use default SA parameters or pass them in
                        sa_grid, sa_score = simulated_annealing(
                            sa_input_grid, ship, modules, tech,
                            player_owned_rewards=None # Assuming SA doesn't need rewards here, adjust if needed
                        )
                        if sa_grid is not None:
                            all_sa_scores.append(sa_score)
                        else:
                            all_sa_scores.append(0.0) # Append 0 if SA fails
                    except Exception as sa_e:
                        print(f"Warning: SA failed for sample {i*batch_size + j}: {sa_e}")
                        all_sa_scores.append(0.0) # Append 0 on error

    # --- 5. Compute and Print Final Metrics ---
    eval_time = time.time() - start_eval_time
    avg_test_loss = total_test_loss / len(test_loader)
    final_accuracy = test_accuracy.compute()
    final_iou = test_iou.compute()
    # final_f1 = test_f1.compute()

    avg_predicted_score = np.mean(all_predicted_scores) if all_predicted_scores else 0.0
    avg_ground_truth_score = np.mean(all_ground_truth_scores) if all_ground_truth_scores else 0.0
    avg_sa_score = np.mean(all_sa_scores) if all_sa_scores else 0.0

    print(f"\n{'='*10} Evaluation Results {'='*10}")
    print(f"Evaluation Time: {eval_time:.2f} seconds")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Pixel-wise Accuracy (Macro Avg): {final_accuracy:.4f}")
    print(f"Pixel-wise mIoU (Macro Avg): {final_iou:.4f}")
    # print(f"Pixel-wise F1-Score (Macro Avg): {final_f1:.4f}")
    print("-" * 30)
    print(f"Average Predicted Grid Score: {avg_predicted_score:.4f}")
    print(f"Average Ground Truth Grid Score: {avg_ground_truth_score:.4f}")
    if run_sa_comparison:
        print(f"Average Simulated Annealing Grid Score: {avg_sa_score:.4f}")
    print("-" * 30)
    # Calculate relative performance
    if avg_ground_truth_score > 1e-6: # Avoid division by zero
         score_ratio_vs_gt = avg_predicted_score / avg_ground_truth_score
         print(f"Ratio Predicted Score / Ground Truth Score: {score_ratio_vs_gt:.4f}")
    if run_sa_comparison and avg_sa_score > 1e-6:
         score_ratio_vs_sa = avg_predicted_score / avg_sa_score
         print(f"Ratio Predicted Score / SA Score: {score_ratio_vs_sa:.4f}")

    print(f"{'='*32}\n")

    # Clean up metrics from GPU memory
    test_accuracy.reset()
    test_iou.reset()
    # test_f1.reset()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained NMS Optimizer model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pth) file.")
    parser.add_argument("--category", type=str, required=True, help="Technology category the model was trained for.")
    parser.add_argument("--ship", type=str, default="standard", help="Ship type the model was trained for.")
    parser.add_argument("--width", type=int, default=4, help="Grid width the model was trained for.")
    parser.add_argument("--height", type=int, default=3, help="Grid height the model was trained for.")
    parser.add_argument("--test_data_dir", type=str, default=DEFAULT_TEST_DATA_DIR, help="Directory containing test .npz data files.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--compare_sa", action='store_true', help="Run Simulated Annealing on test inputs for comparison.")

    args = parser.parse_args()

    # --- Determine Tech Key from Category (if needed, or pass tech directly) ---
    # This logic assumes the model filename contains the specific tech key.
    # Example: model_standard_pulse.pth -> tech = "pulse"
    try:
        model_filename = os.path.basename(args.model)
        parts = model_filename.split('_')
        if len(parts) >= 3 and parts[0] == 'model':
            tech_key = parts[2].replace('.pth', '')
            print(f"Inferred tech key '{tech_key}' from model filename.")
        else:
            raise ValueError("Could not infer tech key from model filename.")
    except Exception as e:
        print(f"Error: Could not determine tech key from model filename '{args.model}'. {e}")
        print("Please ensure the model filename follows the pattern 'model_<ship>_<tech>.pth' or modify the script.")
        sys.exit(1)
    # --- End Tech Key Determination ---


    evaluate_model(
        model_path=args.model,
        test_data_dir=args.test_data_dir,
        ship=args.ship,
        tech=tech_key, # Use inferred tech key
        grid_width=args.width,
        grid_height=args.height,
        batch_size=args.batch_size,
        run_sa_comparison=args.compare_sa,
    )
