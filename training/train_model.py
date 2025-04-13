# training/train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import time
import glob
import argparse
import numpy as np  # Add numpy import
from model_definition import ModulePlacementCNN # Or just 'from model_definition import ...' if in root


# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
# Assuming optimizer.py is now refactored and doesn't cause circular imports
from modules_data import get_tech_modules_for_training
from modules import modules

# <<< Import necessary components for validation >>>
from sklearn.model_selection import train_test_split

try:
    import torchmetrics
except ImportError:
    print("torchmetrics not found. Please install it: pip install torchmetrics")
    # Decide if you want to exit or just disable metrics
    # sys.exit(1)
    torchmetrics = None  # Allow running without torchmetrics, but validation metrics won't work

# --- Configuration ---
DEFAULT_DATA_SOURCE_DIR = "generated_batches"
DEFAULT_MODEL_SAVE_DIR = "trained_models"  # Define default save dir
DEFAULT_LOG_DIR = "runs_placement_only"  # Define default log dir
# --- End Configuration ---


# --- Dataset ---
class PlacementDataset(data.Dataset):
    # <<< CHANGE: Accept both input arrays >>>
    def __init__(self, X_supercharge, X_inactive_mask, y):
        # <<< CHANGE: Store both input arrays >>>
        self.X_supercharge = X_supercharge
        self.X_inactive_mask = X_inactive_mask
        self.y = y

    def __len__(self):
        # <<< CHANGE: Use length of one of the input arrays >>>
        # Ensure data exists before getting length
        if self.X_supercharge is not None:
            return len(self.X_supercharge)
        elif self.X_inactive_mask is not None:
            return len(self.X_inactive_mask)
        elif self.y is not None:
            return len(self.y)
        else:
            return 0

    def __getitem__(self, idx):
        # <<< CHANGE: Get data from both input arrays >>>
        x_sc = self.X_supercharge[idx]
        x_inactive = self.X_inactive_mask[idx]
        target = self.y[idx]

        # <<< CHANGE: Stack the two input arrays along the channel dimension (dim=0) >>>
        # Ensure they are float tensors before stacking
        input_tensor = torch.stack(
            [torch.tensor(x_sc, dtype=torch.float32), torch.tensor(x_inactive, dtype=torch.float32)], dim=0
        )  # Shape becomes [2, height, width]

        # Ensure target is long tensor
        target_tensor = torch.tensor(target, dtype=torch.long)

        return input_tensor, target_tensor


# --- Training Function (Modified for Validation and Early Stopping) ---
def train_model(
    train_loader,
    val_loader,  # Can be None
    grid_height,
    grid_width,
    num_output_classes,
    num_epochs,
    learning_rate,
    weight_decay,
    log_dir,
    model_save_path,  # Path to save the best model
    scheduler_step_size,
    scheduler_gamma,
    early_stopping_patience: int = 10,
    early_stopping_metric: str = "val_loss",  # 'val_loss' or 'val_miou'
):
    """Initializes and trains a ModulePlacementCNN model with validation and early stopping."""
    print(f"Starting training for {log_dir}...")
    print(f"  Saving best model to: {model_save_path}")
    print(f"  Early Stopping: Patience={early_stopping_patience}, Metric='{early_stopping_metric}'")
    print(f"  Grid Dimensions: {grid_height}x{grid_width}")
    print(f"  Num Output Classes: {num_output_classes}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}, WD: {weight_decay}")
    print(f"  Scheduler Step: {scheduler_step_size}, Gamma: {scheduler_gamma}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training device: {device}")

    # <<< CHANGE: Set input_channels=2 >>>
    model = ModulePlacementCNN(
        input_channels=2, grid_height=grid_height, grid_width=grid_width, num_output_classes=num_output_classes
    ).to(device)
    criterion_placement = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure save directory exists
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # --- Initialize Metrics ---
    val_accuracy = None
    val_iou = None
    # <<< Use a local flag for metric availability >>>
    metrics_available = False

    # <<< Check the global torchmetrics variable (module or None) >>>
    if torchmetrics:
        try:
            # Use 'macro' average to treat all classes equally, adjust if needed
            val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_output_classes, average="macro").to(
                device
            )
            val_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_output_classes, average="macro").to(
                device
            )
            # <<< Set the flag to True if initialization succeeds >>>
            metrics_available = True
            print("Torchmetrics initialized successfully.")
        except Exception as e:
            print(f"Warning: Failed to initialize torchmetrics: {e}. Validation metrics will be unavailable.")
            # <<< Keep metrics_available as False >>>
            val_accuracy = None  # Ensure they are None if init fails
            val_iou = None
    else:
        print("Warning: torchmetrics not installed. Validation accuracy and mIoU will not be calculated.")
        # <<< metrics_available remains False >>>

    # --- Early Stopping Initialization ---
    # <<< Adjust check for es_metric based on metrics_available >>>
    if early_stopping_metric == "val_miou" and not metrics_available:
        print(
            f"Warning: Cannot monitor 'val_miou' as torchmetrics failed or is not installed. Defaulting to 'val_loss'."
        )
        early_stopping_metric = "val_loss"

    epochs_no_improve = 0
    best_metric_value = float("inf") if early_stopping_metric == "val_loss" else float("-inf")
    metric_mode = "min" if early_stopping_metric == "val_loss" else "max"  # Determine if lower or higher is better
    early_stop_triggered = False
    # --- End Early Stopping Initialization ---

    for epoch in range(num_epochs):
        # --- Training Loop ---
        model.train()
        running_train_loss = 0.0
        start_epoch_time = time.time()
        # <<< CHANGE: inputs will now have shape [batch, 2, height, width] >>>
        for i, (inputs, targets_placement) in enumerate(train_loader):
            inputs, targets_placement = inputs.to(device), targets_placement.to(device)
            optimizer.zero_grad()
            outputs_placement = model(inputs)
            loss_placement = criterion_placement(outputs_placement, targets_placement.long())
            loss = loss_placement
            loss.backward()
            optimizer.step()
            running_train_loss += loss_placement.item()
        # --- End Training Loop ---

        # Avoid division by zero if train_loader is empty
        avg_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Validation Loop (only if val_loader exists) ---
        avg_val_loss = float("nan")  # Default if no validation
        epoch_val_accuracy = 0.0
        epoch_val_iou = 0.0

        if val_loader:
            model.eval()
            running_val_loss = 0.0
            # <<< Reset metrics only if they exist >>>
            if metrics_available:
                if val_accuracy:
                    val_accuracy.reset()
                if val_iou:
                    val_iou.reset()

            with torch.no_grad():
                # <<< CHANGE: inputs will now have shape [batch, 2, height, width] >>>
                for i, (inputs, targets_placement) in enumerate(val_loader):
                    inputs, targets_placement = inputs.to(device), targets_placement.to(device)
                    outputs_placement = model(inputs)
                    loss_placement = criterion_placement(outputs_placement, targets_placement.long())
                    running_val_loss += loss_placement.item()

                    # <<< Update metrics only if available >>>
                    if metrics_available:
                        preds = torch.argmax(outputs_placement, dim=1)
                        if val_accuracy:
                            val_accuracy.update(preds, targets_placement)
                        if val_iou:
                            val_iou.update(preds, targets_placement)

            # Avoid division by zero if val_loader is empty
            avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            # <<< Compute metrics only if available >>>
            epoch_val_accuracy = val_accuracy.compute() if metrics_available and val_accuracy else 0.0
            epoch_val_iou = val_iou.compute() if metrics_available and val_iou else 0.0
        # --- End Validation Loop ---

        epoch_time = time.time() - start_epoch_time

        # --- Log Metrics ---
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "  # Will show 'nan' if no validation
            f"Val Acc: {epoch_val_accuracy:.4f}, "
            f"Val mIoU: {epoch_val_iou:.4f}, "
            f"LR: {current_lr:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        if val_loader:  # Only log validation metrics if validation was performed
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            if metrics_available:
                writer.add_scalar("Accuracy/validation", epoch_val_accuracy, epoch)
                writer.add_scalar("mIoU/validation", epoch_val_iou, epoch)
        writer.add_scalar("LearningRate/epoch", current_lr, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)

        # --- Early Stopping Check (only if val_loader exists) ---
        if val_loader:
            # <<< Use metrics_available flag here too >>>
            current_metric_value = avg_val_loss
            if early_stopping_metric == "val_miou":
                if metrics_available:
                    current_metric_value = epoch_val_iou
                else:
                    # Should have defaulted earlier, but double-check
                    print(
                        "Warning: Trying to use val_miou for early stopping, but metrics are unavailable. Using val_loss."
                    )
                    current_metric_value = avg_val_loss
                    early_stopping_metric = "val_loss"  # Ensure consistency
                    metric_mode = "min"
                    best_metric_value = min(best_metric_value, float("inf"))  # Reset best if switching metric

            improved = False
            if metric_mode == "min" and current_metric_value < best_metric_value:
                improved = True
            elif metric_mode == "max" and current_metric_value > best_metric_value:
                improved = True

            if improved:
                best_metric_value = current_metric_value
                epochs_no_improve = 0
                # Save the model *only* when the validation metric improves
                try:
                    # Move model to CPU before saving to avoid GPU memory issues on load
                    model.to("cpu")
                    torch.save(model.state_dict(), model_save_path)
                    model.to(device)  # Move back to original device
                    print(
                        f"  Saved new best model checkpoint (Epoch {epoch+1}, {early_stopping_metric}: {best_metric_value:.4f})"
                    )
                except Exception as e:
                    print(f"  Error saving model checkpoint: {e}")
            else:
                epochs_no_improve += 1
                print(
                    f"  No improvement in {early_stopping_metric} for {epochs_no_improve} epoch(s). Best: {best_metric_value:.4f}"
                )

            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                early_stop_triggered = True
                break  # Exit the training loop
        # --- End Early Stopping Check ---

        #scheduler.step()

    if not early_stop_triggered:
        print(f"Finished Training for {num_epochs} epochs.")
        # If training finished without early stopping and validation was performed,
        # ensure the *last* model is saved if it wasn't already the best.
        # This might be desired if the absolute best isn't strictly necessary.
        # However, the current logic only saves on improvement, which is typical.
        # If no validation was done, save the final model state.
        if not val_loader:
            try:
                print(f"Saving final model state (no validation performed) to {model_save_path}")
                model.to("cpu")
                torch.save(model.state_dict(), model_save_path)
                model.to(device)
            except Exception as e:
                print(f"  Error saving final model checkpoint: {e}")

    writer.close()

    # Load the best saved model state before returning (if validation was performed)
    if val_loader:
        try:
            print(
                f"Loading best model state from {model_save_path} (Metric: {early_stopping_metric}, Value: {best_metric_value:.4f})"
            )
            # Ensure the file exists before trying to load
            if os.path.exists(model_save_path):
                model.load_state_dict(torch.load(model_save_path, map_location=device))
                model.to(device)  # Ensure model is on the correct device
            else:
                print(f"Warning: Best model file {model_save_path} not found. Returning last state.")
        except Exception as e:
            print(f"Warning: Could not load best model state after training: {e}. Returning last state.")
    # If no validation, the model already holds the last state.

    return model


# --- Training Orchestration Function (Modified) ---
def run_training_from_files(
    ship,
    tech_category_to_train,
    grid_width,
    grid_height,
    learning_rate,
    weight_decay,
    num_epochs,
    batch_size,
    base_log_dir,
    model_save_dir,
    data_source_dir,
    scheduler_step_size,
    scheduler_gamma,
    validation_split=0.2,
    early_stopping_patience=10,
    early_stopping_metric="val_loss",
):
    """
    Orchestrates the training process, including data splitting, loading,
    and calling train_model with early stopping.
    """
    # --- Get Tech Keys ---
    try:
        ship_data = modules.get(ship)
        if not ship_data or "types" not in ship_data or not isinstance(ship_data["types"], dict):
            raise KeyError(f"Ship '{ship}' or its 'types' dictionary not found/invalid.")
        category_data = ship_data["types"].get(tech_category_to_train)
        if not category_data or not isinstance(category_data, list):
            raise KeyError(f"Category '{tech_category_to_train}' not found or invalid for ship '{ship}'.")
        tech_keys_to_train = [
            tech_data["key"] for tech_data in category_data if isinstance(tech_data, dict) and "key" in tech_data
        ]
    except KeyError as e:
        print(f"Error accessing module data: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while getting tech keys: {e}")
        return
    if not tech_keys_to_train:
        print(f"Error: No valid tech keys found for ship '{ship}', category '{tech_category_to_train}'.")
        return
    print(f"Planning to train models for techs: {tech_keys_to_train}")
    print(f"Looking for data files in: {os.path.abspath(data_source_dir)}")
    # --- End Get Tech Keys ---

    trained_models = {}

    for tech in tech_keys_to_train:
        print(f"\n{'='*10} Processing Tech: {tech} {'='*10}")

        # --- 1. Find and Load Data ---
        file_pattern = os.path.join(data_source_dir, f"data_{ship}_{tech}_{grid_width}x{grid_height}_*.npz")
        data_files = glob.glob(file_pattern)
        if not data_files:
            print(f"Warning: No data files found matching pattern '{file_pattern}'. Skipping tech '{tech}'.")
            continue
        print(f"Found {len(data_files)} data files for tech '{tech}'. Loading...")
        # <<< CHANGE: Lists for both input arrays >>>
        all_X_supercharge_data, all_X_inactive_mask_data, all_y_data = [], [], []
        total_loaded_samples = 0
        load_start_time = time.time()
        for filepath in data_files:
            try:
                with np.load(filepath) as npz_file:
                    # <<< CHANGE: Check for new keys >>>
                    if "X_supercharge" not in npz_file or "X_inactive_mask" not in npz_file or "y" not in npz_file:
                        print(
                            f"Warning: Keys 'X_supercharge', 'X_inactive_mask', or 'y' not found in {filepath}. Skipping file."
                        )
                        continue

                    # <<< CHANGE: Load both input arrays >>>
                    x_sc_batch, x_inactive_batch, y_batch = (
                        npz_file["X_supercharge"],
                        npz_file["X_inactive_mask"],
                        npz_file["y"],
                    )

                    # Basic shape validation (check all three arrays)
                    if len(x_sc_batch.shape) < 3 or len(x_inactive_batch.shape) < 3 or len(y_batch.shape) < 3:
                        print(f"Warning: Unexpected array dimensions in {filepath}. Skipping file.")
                        continue
                    if (
                        x_sc_batch.shape[1:] != (grid_height, grid_width)
                        or x_inactive_batch.shape[1:] != (grid_height, grid_width)
                        or y_batch.shape[1:] != (grid_height, grid_width)
                    ):
                        print(
                            f"Warning: Shape mismatch in {filepath}. Expected ({grid_height},{grid_width}), got X_sc:{x_sc_batch.shape[1:]}, X_in:{x_inactive_batch.shape[1:]}, y:{y_batch.shape[1:]}. Skipping file."
                        )
                        continue
                    if not (x_sc_batch.shape[0] == x_inactive_batch.shape[0] == y_batch.shape[0]):
                        print(
                            f"Warning: Sample count mismatch in {filepath} (X_sc:{x_sc_batch.shape[0]}, X_in:{x_inactive_batch.shape[0]}, y:{y_batch.shape[0]}). Skipping file."
                        )
                        continue

                    # <<< CHANGE: Append to respective lists >>>
                    all_X_supercharge_data.append(x_sc_batch)
                    all_X_inactive_mask_data.append(x_inactive_batch)
                    all_y_data.append(y_batch)
                    total_loaded_samples += x_sc_batch.shape[0]
            except Exception as e:
                print(f"Error loading data from {filepath}: {e}. Skipping file.")
                continue

        # <<< CHANGE: Check if any data was loaded >>>
        if not all_X_supercharge_data:
            print(f"Warning: No valid data could be loaded for tech '{tech}'. Skipping.")
            continue

        try:
            # <<< CHANGE: Concatenate all three lists >>>
            X_supercharge_data_np = np.concatenate(all_X_supercharge_data, axis=0)
            X_inactive_mask_data_np = np.concatenate(all_X_inactive_mask_data, axis=0)
            y_data_np = np.concatenate(all_y_data, axis=0)
            load_time = time.time() - load_start_time
            print(
                f"Loaded and concatenated {total_loaded_samples} total samples for tech '{tech}' in {load_time:.2f}s."
            )
        except Exception as e:
            print(f"Error concatenating data arrays for tech '{tech}': {e}. Skipping.")
            continue
        # --- End Load Data ---

        # --- Determine num_output_classes ---
        tech_modules = get_tech_modules_for_training(modules, ship, tech)
        if not tech_modules:
            print(f"Warning: Could not get modules for tech '{tech}'. Skipping.")
            continue
        num_output_classes = len(tech_modules) + 1
        if num_output_classes <= 1:
            print(f"Skipping tech '{tech}': Not enough output classes ({num_output_classes}).")
            continue
        # --- End Determine num_output_classes ---

        # --- 2. Perform Train/Validation Split ---
        # <<< CHANGE: Initialize all split arrays to None >>>
        X_train_sc_np, X_val_sc_np = None, None
        X_train_in_np, X_val_in_np = None, None
        y_train_np, y_val_np = None, None
        val_loader = None  # Initialize val_loader to None

        # <<< CHANGE: Use length of one input array for check >>>
        if len(X_supercharge_data_np) < 2:
            print(
                f"Warning: Not enough samples ({len(X_supercharge_data_np)}) for tech '{tech}' to create validation set. Training without validation/early stopping."
            )
            # <<< CHANGE: Assign all data to training sets >>>
            X_train_sc_np = X_supercharge_data_np
            X_train_in_np = X_inactive_mask_data_np
            y_train_np = y_data_np
        elif validation_split <= 0 or validation_split >= 1:
            print(
                f"Warning: Invalid validation_split ({validation_split}). Training without validation/early stopping."
            )
            # <<< CHANGE: Assign all data to training sets >>>
            X_train_sc_np = X_supercharge_data_np
            X_train_in_np = X_inactive_mask_data_np
            y_train_np = y_data_np
        else:
            try:
                # <<< CHANGE: Split all three arrays consistently >>>
                # Generate indices for splitting
                num_samples = len(X_supercharge_data_np)
                indices = np.arange(num_samples)
                train_indices, val_indices = train_test_split(
                    indices, test_size=validation_split, random_state=42, shuffle=True
                )

                # Use indices to split the arrays
                X_train_sc_np = X_supercharge_data_np[train_indices]
                X_val_sc_np = X_supercharge_data_np[val_indices]
                X_train_in_np = X_inactive_mask_data_np[train_indices]
                X_val_in_np = X_inactive_mask_data_np[val_indices]
                y_train_np = y_data_np[train_indices]
                y_val_np = y_data_np[val_indices]

                print(f"Split data: {len(X_train_sc_np)} train, {len(X_val_sc_np)} validation samples.")
            except Exception as e:
                print(
                    f"Error during train/val split for tech '{tech}': {e}. Training without validation/early stopping."
                )
                # <<< CHANGE: Fallback assignment >>>
                X_train_sc_np = X_supercharge_data_np
                X_train_in_np = X_inactive_mask_data_np
                y_train_np = y_data_np  # Fallback to using all data for training
        # --- End Split ---

        # --- 3. Prepare Datasets and DataLoaders ---
        try:
            # <<< CHANGE: Pass both input arrays to PlacementDataset >>>
            train_dataset = PlacementDataset(X_train_sc_np, X_train_in_np, y_train_np)

            # Create val_loader only if validation data exists
            # <<< CHANGE: Check validation arrays >>>
            if X_val_sc_np is not None and X_val_in_np is not None and y_val_np is not None and len(X_val_sc_np) > 0:
                # <<< CHANGE: Pass both input arrays to PlacementDataset >>>
                val_dataset = PlacementDataset(X_val_sc_np, X_val_in_np, y_val_np)
                val_batch_size = min(batch_size, len(val_dataset))  # Use same batch size or smaller
                if val_batch_size > 0:
                    val_loader = data.DataLoader(
                        val_dataset,
                        batch_size=val_batch_size,
                        shuffle=False,  # No shuffle for validation
                        num_workers=min(2, os.cpu_count()),
                        pin_memory=torch.cuda.is_available(),
                    )
                else:  # Should not happen if len(X_val_sc_np) > 0, but safety check
                    print(f"Warning: Validation dataset for tech '{tech}' is effectively empty. Skipping validation.")
                    val_loader = None  # Ensure val_loader is None
            else:
                print(f"Info: No validation data available for tech '{tech}'. Skipping validation.")
                val_loader = None  # Ensure val_loader is None

        except Exception as e:
            print(f"Error creating Datasets/Tensors for tech '{tech}': {e}. Skipping.")
            continue

        effective_batch_size = min(batch_size, len(train_dataset))
        if effective_batch_size == 0:
            print(f"Warning: Train dataset for tech '{tech}' is empty. Skipping.")
            continue

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count()),
            pin_memory=torch.cuda.is_available(),
        )
        # --- End Prepare DataLoaders ---

        # --- 4. Train Model ---
        log_dir = os.path.join(base_log_dir, ship, tech_category_to_train, tech)
        # Define path for saving the best model for this tech
        model_filename = f"model_{ship}_{tech}.pth"
        model_save_path = os.path.join(model_save_dir, model_filename)

        # <<< Check if val_loader exists before enabling early stopping >>>
        can_early_stop = val_loader is not None
        # If no validation loader, patience effectively becomes num_epochs (no early stopping)
        effective_patience = early_stopping_patience if can_early_stop else num_epochs
        if not can_early_stop:
            print("Info: No validation loader available. Early stopping disabled for this tech.")

        model = train_model(
            train_loader,
            val_loader,  # Pass val_loader (can be None)
            grid_height,
            grid_width,
            num_output_classes,
            num_epochs,
            learning_rate,
            weight_decay,
            log_dir,
            model_save_path,  # Pass save path
            scheduler_step_size,
            scheduler_gamma,
            early_stopping_patience=effective_patience,
            early_stopping_metric=early_stopping_metric,
        )
        # --- End Train Model ---

        # --- 5. Store Trained Model (Optional) ---
        # The best model is already saved by train_model.
        # We can store the final model object if needed elsewhere immediately.
        trained_models[tech] = model
        # --- End Store Model ---

    print(f"\n{'='*10} Training Pipeline Complete for Category: {tech_category_to_train} {'='*10}")


# --- Main Execution (Modified) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NMS Optimizer models with validation and early stopping.")
    # --- Basic Training Args ---
    parser.add_argument("--category", type=str, required=True, help="Technology category to train models for.")
    parser.add_argument("--ship", type=str, default="standard", help="Ship type.")
    parser.add_argument("--width", type=int, default=4, help="Grid width model was trained for.")
    parser.add_argument("--height", type=int, default=3, help="Grid height model was trained for.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--wd", type=float, default=1e-3, help="Weight decay (L2 regularization).")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum number of training epochs."
    )  # Increased default
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    # --- Scheduler Args ---
    parser.add_argument("--scheduler_step", type=int, default=30, help="StepLR: Number of epochs before decaying LR.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="StepLR: Multiplicative factor of LR decay.")
    # --- Data/Saving Args ---
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="Base directory for TensorBoard logs.")
    parser.add_argument(
        "--model_dir", type=str, default=DEFAULT_MODEL_SAVE_DIR, help="Directory to save best trained models."
    )
    parser.add_argument(
        "--data_source_dir",
        type=str,
        default=DEFAULT_DATA_SOURCE_DIR,
        help="Directory containing generated .npz data files.",
    )
    # --- Validation & Early Stopping Args ---
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="Fraction of data to use for validation (0.0 to 1.0)."
    )
    parser.add_argument(
        "--es_patience", type=int, default=15, help="Early Stopping: Number of epochs to wait for improvement."
    )  # Increased default
    parser.add_argument(
        "--es_metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_miou"],
        help="Early Stopping: Metric to monitor ('val_loss' or 'val_miou').",
    )

    args = parser.parse_args()

    # Validate val_split
    if not 0.0 <= args.val_split < 1.0:
        parser.error("--val_split must be between 0.0 and 1.0 (exclusive of 1.0)")

    # <<< Modify this check >>>
    # Validate es_metric based on initial import status
    # Note: This check happens *before* train_model runs, so it only catches
    # the case where torchmetrics wasn't installed at all. The check inside
    # train_model handles cases where initialization fails later.
    effective_es_metric = args.es_metric
    if args.es_metric == "val_miou" and not torchmetrics:
        print(
            "Warning: Cannot use 'val_miou' for early stopping as torchmetrics is not installed. Defaulting to 'val_loss'."
        )
        effective_es_metric = "val_loss"  # Use a different variable

    config = {
        "grid_width": args.width,
        "grid_height": args.height,
        "ship": args.ship,
        "tech_category_to_train": args.category,
        "learning_rate": args.lr,
        "weight_decay": args.wd,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "scheduler_step_size": args.scheduler_step,
        "scheduler_gamma": args.scheduler_gamma,
        "base_log_dir": args.log_dir,
        "model_save_dir": args.model_dir,
        "data_source_dir": args.data_source_dir,
        # <<< Add validation/early stopping config >>>
        "validation_split": args.val_split,
        "early_stopping_patience": args.es_patience,
        "early_stopping_metric": effective_es_metric,  # Use the potentially adjusted metric
    }

    start_time_all = time.time()
    print(f"Starting model training process...")
    print(f"Configuration: {config}")

    os.makedirs(config["model_save_dir"], exist_ok=True)

    run_training_from_files(
        ship=config["ship"],
        tech_category_to_train=config["tech_category_to_train"],
        grid_width=config["grid_width"],
        grid_height=config["grid_height"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        base_log_dir=config["base_log_dir"],
        model_save_dir=config["model_save_dir"],
        data_source_dir=config["data_source_dir"],
        scheduler_step_size=config["scheduler_step_size"],
        scheduler_gamma=config["scheduler_gamma"],
        # <<< Pass validation/early stopping config >>>
        validation_split=config["validation_split"],
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_metric=config["early_stopping_metric"],  # Pass the potentially adjusted metric
    )

    end_time_all = time.time()
    print(f"\n{'='*20} Model Training Complete {'='*20}")
    print(f"Total time: {end_time_all - start_time_all:.2f} seconds.")
    print(f"Best models saved in: {os.path.abspath(config['model_save_dir'])}")
