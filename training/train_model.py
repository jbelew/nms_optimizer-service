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
import numpy as np
from typing import Optional # Added for type hinting

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
from model_definition import ModulePlacementCNN # Import model definition
from modules_data import get_tech_modules_for_training
from modules import modules
from sklearn.model_selection import train_test_split

try:
    import torchmetrics
except ImportError:
    print("torchmetrics not found. Please install it: pip install torchmetrics")
    torchmetrics = None

# --- Configuration ---
DEFAULT_DATA_SOURCE_DIR = "generated_batches"
DEFAULT_MODEL_SAVE_DIR = "trained_models"
DEFAULT_LOG_DIR = "runs_placement_only"
# --- End Configuration ---


# --- Dataset ---
class PlacementDataset(data.Dataset):
    def __init__(self, X_supercharge, X_inactive_mask, y):
        self.X_supercharge = X_supercharge
        self.X_inactive_mask = X_inactive_mask
        self.y = y

    def __len__(self):
        # Use the length of the first available array
        if self.X_supercharge is not None:
            return len(self.X_supercharge)
        elif self.X_inactive_mask is not None:
            return len(self.X_inactive_mask)
        elif self.y is not None:
            return len(self.y)
        else:
            return 0

    def __getitem__(self, idx):
        x_sc = self.X_supercharge[idx]
        x_inactive = self.X_inactive_mask[idx]
        target = self.y[idx]

        # Ensure tensors are created correctly
        input_tensor = torch.stack(
            [torch.tensor(x_sc, dtype=torch.float32), torch.tensor(x_inactive, dtype=torch.float32)], dim=0
        )
        target_tensor = torch.tensor(target, dtype=torch.long)

        return input_tensor, target_tensor


# --- Training Function (Modified for Validation, Early Stopping, Class Weights, Scheduler) ---
def train_model(
    train_loader,
    val_loader,
    grid_height,
    grid_width,
    num_output_classes,
    num_epochs,
    learning_rate,
    weight_decay,
    log_dir,
    model_save_path,
    scheduler_step_size,
    scheduler_gamma,
    criterion_weights: Optional[torch.Tensor] = None, # <<< Added for class weights
    early_stopping_patience: int = 10,
    early_stopping_metric: str = "val_loss",
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

    model = ModulePlacementCNN(
        input_channels=2, grid_height=grid_height, grid_width=grid_width, num_output_classes=num_output_classes
    ).to(device)

    # <<< Use class weights if provided >>>
    if criterion_weights is not None:
        print("  Using provided class weights for loss.")
        criterion_placement = nn.CrossEntropyLoss(weight=criterion_weights.to(device))
    else:
        print("  Using default CrossEntropyLoss (no class weights).")
        criterion_placement = nn.CrossEntropyLoss()
    # <<< End class weights usage >>>

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # <<< Enable the scheduler >>>
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Ensure the specific model save directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # --- Initialize Metrics ---
    val_accuracy = None
    val_iou = None
    metrics_available = False

    if torchmetrics:
        try:
            val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_output_classes, average="macro").to(
                device
            )
            val_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_output_classes, average="macro").to(
                device
            )
            metrics_available = True
            print("Torchmetrics initialized successfully.")
        except Exception as e:
            print(f"Warning: Failed to initialize torchmetrics: {e}. Validation metrics will be unavailable.")
            val_accuracy = None
            val_iou = None
    else:
        print("Warning: torchmetrics not installed. Validation accuracy and mIoU will not be calculated.")

    # --- Early Stopping Initialization ---
    if early_stopping_metric == "val_miou" and not metrics_available:
        print(
            f"Warning: Cannot monitor 'val_miou' as torchmetrics failed or is not installed. Defaulting to 'val_loss'."
        )
        early_stopping_metric = "val_loss"

    epochs_no_improve = 0
    best_metric_value = float("inf") if early_stopping_metric == "val_loss" else float("-inf")
    metric_mode = "min" if early_stopping_metric == "val_loss" else "max"
    early_stop_triggered = False
    # --- End Early Stopping Initialization ---

    for epoch in range(num_epochs):
        # --- Training Loop ---
        model.train()
        running_train_loss = 0.0
        start_epoch_time = time.time()
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

        avg_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Validation Loop ---
        avg_val_loss = float("nan")
        epoch_val_accuracy = 0.0
        epoch_val_iou = 0.0

        if val_loader:
            model.eval()
            running_val_loss = 0.0
            if metrics_available:
                if val_accuracy: val_accuracy.reset()
                if val_iou: val_iou.reset()

            with torch.no_grad():
                for i, (inputs, targets_placement) in enumerate(val_loader):
                    inputs, targets_placement = inputs.to(device), targets_placement.to(device)
                    outputs_placement = model(inputs)
                    loss_placement = criterion_placement(outputs_placement, targets_placement.long())
                    running_val_loss += loss_placement.item()

                    if metrics_available:
                        preds = torch.argmax(outputs_placement, dim=1)
                        if val_accuracy: val_accuracy.update(preds, targets_placement)
                        if val_iou: val_iou.update(preds, targets_placement)

            avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            epoch_val_accuracy = val_accuracy.compute() if metrics_available and val_accuracy else 0.0
            epoch_val_iou = val_iou.compute() if metrics_available and val_iou else 0.0
        # --- End Validation Loop ---

        epoch_time = time.time() - start_epoch_time

        # --- Log Metrics ---
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {epoch_val_accuracy:.4f}, "
            f"Val mIoU: {epoch_val_iou:.4f}, "
            f"LR: {current_lr:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        if val_loader:
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            if metrics_available:
                writer.add_scalar("Accuracy/validation", epoch_val_accuracy, epoch)
                writer.add_scalar("mIoU/validation", epoch_val_iou, epoch)
        writer.add_scalar("LearningRate/epoch", current_lr, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)

        # --- Early Stopping Check ---
        if val_loader:
            current_metric_value = avg_val_loss
            if early_stopping_metric == "val_miou":
                if metrics_available:
                    current_metric_value = epoch_val_iou
                else:
                    print("Warning: Trying to use val_miou for early stopping, but metrics unavailable. Using val_loss.")
                    current_metric_value = avg_val_loss
                    early_stopping_metric = "val_loss"
                    metric_mode = "min"
                    best_metric_value = min(best_metric_value, float("inf"))

            improved = False
            if metric_mode == "min" and current_metric_value < best_metric_value:
                improved = True
            elif metric_mode == "max" and current_metric_value > best_metric_value:
                improved = True

            if improved:
                best_metric_value = current_metric_value
                epochs_no_improve = 0
                try:
                    # Save model to CPU to avoid GPU memory issues on load
                    model.to("cpu")
                    torch.save(model.state_dict(), model_save_path)
                    model.to(device) # Move back to original device
                    print(f"  Saved new best model checkpoint (Epoch {epoch+1}, {early_stopping_metric}: {best_metric_value:.4f})")
                except Exception as e:
                    print(f"  Error saving model checkpoint: {e}")
            else:
                epochs_no_improve += 1
                print(f"  No improvement in {early_stopping_metric} for {epochs_no_improve} epoch(s). Best: {best_metric_value:.4f}")

            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                early_stop_triggered = True
                break
        # --- End Early Stopping Check ---

        # <<< Step the scheduler after validation and logging >>>
        scheduler.step()

    if not early_stop_triggered:
        print(f"Finished Training for {num_epochs} epochs.")
        # Save the final model state if no validation was performed or early stopping didn't trigger
        if not val_loader:
            try:
                print(f"Saving final model state (no validation performed) to {model_save_path}")
                model.to("cpu")
                torch.save(model.state_dict(), model_save_path)
                model.to(device)
            except Exception as e:
                print(f"  Error saving final model checkpoint: {e}")

    writer.close()

    # --- Load Best Model State After Training (if validation was used) ---
    if val_loader:
        try:
            print(f"Loading best model state from {model_save_path} (Metric: {early_stopping_metric}, Value: {best_metric_value:.4f})")
            if os.path.exists(model_save_path):
                # Load state dict onto the correct device directly
                model.load_state_dict(torch.load(model_save_path, map_location=device))
                model.to(device) # Ensure model is on the correct device
            else:
                print(f"Warning: Best model file {model_save_path} not found. Returning last state.")
        except Exception as e:
            print(f"Warning: Could not load best model state after training: {e}. Returning last state.")

    return model


# --- Training Orchestration Function (Modified for Class Weights) ---
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
    base_model_save_dir,
    base_data_source_dir,
    scheduler_step_size,
    scheduler_gamma,
    validation_split=0.2,
    early_stopping_patience=10,
    early_stopping_metric="val_loss",
):
    """
    Orchestrates the training process, loading data and saving models
    using ship/tech subdirectories. Includes class weight calculation.
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
    print(f"Looking for data files in subdirectories under: {os.path.abspath(base_data_source_dir)}")
    # --- End Get Tech Keys ---

    trained_models = {}

    for tech in tech_keys_to_train:
        print(f"\n{'='*10} Processing Tech: {tech} {'='*10}")

        # --- 1. Find and Load Data ---
        tech_data_source_dir = os.path.join(base_data_source_dir, ship, tech)
        file_pattern = os.path.join(tech_data_source_dir, f"data_{ship}_{tech}_{grid_width}x{grid_height}_*.npz")
        data_files = glob.glob(file_pattern)
        if not data_files:
            print(f"Warning: No data files found matching pattern '{file_pattern}'. Skipping tech '{tech}'.")
            continue
        print(f"Found {len(data_files)} data files for tech '{tech}'. Loading...")

        all_X_supercharge_data, all_X_inactive_mask_data, all_y_data = [], [], []
        total_loaded_samples = 0
        load_start_time = time.time()
        for filepath in data_files:
            try:
                with np.load(filepath) as npz_file:
                    if "X_supercharge" not in npz_file or "X_inactive_mask" not in npz_file or "y" not in npz_file:
                        print(f"Warning: Required keys not found in {filepath}. Skipping file.")
                        continue

                    x_sc_batch, x_inactive_batch, y_batch = (
                        npz_file["X_supercharge"], npz_file["X_inactive_mask"], npz_file["y"]
                    )

                    # Basic shape and consistency validation
                    if len(x_sc_batch.shape) < 3 or len(x_inactive_batch.shape) < 3 or len(y_batch.shape) < 3:
                        print(f"Warning: Unexpected array dimensions in {filepath}. Skipping file.")
                        continue
                    if (
                        x_sc_batch.shape[1:] != (grid_height, grid_width)
                        or x_inactive_batch.shape[1:] != (grid_height, grid_width)
                        or y_batch.shape[1:] != (grid_height, grid_width)
                    ):
                        print(f"Warning: Shape mismatch in {filepath} (Expected {grid_height}x{grid_width}). Skipping file.")
                        continue
                    if not (x_sc_batch.shape[0] == x_inactive_batch.shape[0] == y_batch.shape[0]):
                        print(f"Warning: Sample count mismatch in {filepath}. Skipping file.")
                        continue

                    all_X_supercharge_data.append(x_sc_batch)
                    all_X_inactive_mask_data.append(x_inactive_batch)
                    all_y_data.append(y_batch)
                    total_loaded_samples += x_sc_batch.shape[0]
            except Exception as e:
                print(f"Error loading data from {filepath}: {e}. Skipping file.")
                continue

        if not all_X_supercharge_data:
            print(f"Warning: No valid data could be loaded for tech '{tech}'. Skipping.")
            continue

        try:
            X_supercharge_data_np = np.concatenate(all_X_supercharge_data, axis=0)
            X_inactive_mask_data_np = np.concatenate(all_X_inactive_mask_data, axis=0)
            y_data_np = np.concatenate(all_y_data, axis=0)
            load_time = time.time() - load_start_time
            print(f"Loaded and concatenated {total_loaded_samples} total samples for tech '{tech}' in {load_time:.2f}s.")
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
        X_train_sc_np, X_val_sc_np = None, None
        X_train_in_np, X_val_in_np = None, None
        y_train_np, y_val_np = None, None
        val_loader = None

        if len(X_supercharge_data_np) < 2:
            print(f"Warning: Not enough samples ({len(X_supercharge_data_np)}) for tech '{tech}' for validation.")
            X_train_sc_np, X_train_in_np, y_train_np = X_supercharge_data_np, X_inactive_mask_data_np, y_data_np
        elif validation_split <= 0 or validation_split >= 1:
            print(f"Warning: Invalid validation_split ({validation_split}). Training without validation.")
            X_train_sc_np, X_train_in_np, y_train_np = X_supercharge_data_np, X_inactive_mask_data_np, y_data_np
        else:
            try:
                num_samples = len(X_supercharge_data_np)
                indices = np.arange(num_samples)
                train_indices, val_indices = train_test_split(
                    indices, test_size=validation_split, random_state=42, shuffle=True
                )
                X_train_sc_np, X_val_sc_np = X_supercharge_data_np[train_indices], X_supercharge_data_np[val_indices]
                X_train_in_np, X_val_in_np = X_inactive_mask_data_np[train_indices], X_inactive_mask_data_np[val_indices]
                y_train_np, y_val_np = y_data_np[train_indices], y_data_np[val_indices]
                print(f"Split data: {len(X_train_sc_np)} train, {len(X_val_sc_np)} validation samples.")
            except Exception as e:
                print(f"Error during train/val split for tech '{tech}': {e}. Training without validation.")
                X_train_sc_np, X_train_in_np, y_train_np = X_supercharge_data_np, X_inactive_mask_data_np, y_data_np
        # --- End Split ---

        # --- Calculate Class Weights (using y_train_np) ---
        print("Calculating class weights...")
        class_weights_tensor = None
        if y_train_np is not None and y_train_np.size > 0:
            flat_labels = y_train_np.flatten()
            class_counts = np.bincount(flat_labels, minlength=num_output_classes)
            # Prevent division by zero and handle classes not present
            total_pixels = flat_labels.size
            # Inverse frequency weighting: weight = total / (num_classes * count)
            class_weights = np.where(
                class_counts > 0,
                total_pixels / (num_output_classes * class_counts),
                0 # Assign 0 weight if class is not present
            )
            # Optional: Normalize weights (can help stability)
            # class_weights = class_weights / np.sum(class_weights) * num_output_classes

            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            print(f"Calculated class weights: {class_weights_tensor.cpu().numpy()}")
        else:
            print("Warning: y_train_np not available or empty. Cannot calculate class weights.")
        # --- End Calculate Class Weights ---

        # --- 3. Prepare Datasets and DataLoaders ---
        try:
            train_dataset = PlacementDataset(X_train_sc_np, X_train_in_np, y_train_np)
            if X_val_sc_np is not None and X_val_in_np is not None and y_val_np is not None and len(X_val_sc_np) > 0:
                val_dataset = PlacementDataset(X_val_sc_np, X_val_in_np, y_val_np)
                val_batch_size = min(batch_size, len(val_dataset))
                if val_batch_size > 0:
                    val_loader = data.DataLoader(
                        val_dataset, batch_size=val_batch_size, shuffle=False,
                        num_workers=min(2, os.cpu_count()), pin_memory=torch.cuda.is_available()
                    )
                else:
                    print(f"Warning: Validation dataset for tech '{tech}' is empty. Skipping validation.")
                    val_loader = None
            else:
                print(f"Info: No validation data available for tech '{tech}'. Skipping validation.")
                val_loader = None
        except Exception as e:
            print(f"Error creating Datasets/Tensors for tech '{tech}': {e}. Skipping.")
            continue

        effective_batch_size = min(batch_size, len(train_dataset))
        if effective_batch_size == 0:
            print(f"Warning: Train dataset for tech '{tech}' is empty. Skipping.")
            continue

        train_loader = data.DataLoader(
            train_dataset, batch_size=effective_batch_size, shuffle=True,
            num_workers=min(4, os.cpu_count()), pin_memory=torch.cuda.is_available()
        )
        # --- End Prepare DataLoaders ---

        # --- 4. Train Model ---
        log_dir = os.path.join(base_log_dir, ship, tech)
        # Save model directly in the base model directory, named appropriately
        model_filename = f"model_{ship}_{tech}.pth"
        model_save_path = os.path.join(base_model_save_dir, model_filename)

        can_early_stop = val_loader is not None
        effective_patience = early_stopping_patience if can_early_stop else num_epochs
        if not can_early_stop:
            print("Info: No validation loader available. Early stopping disabled for this tech.")

        model = train_model(
            train_loader, val_loader, grid_height, grid_width, num_output_classes,
            num_epochs, learning_rate, weight_decay, log_dir, model_save_path,
            scheduler_step_size, scheduler_gamma,
            criterion_weights=class_weights_tensor, # <<< Pass calculated weights
            early_stopping_patience=effective_patience,
            early_stopping_metric=early_stopping_metric,
        )
        # --- End Train Model ---

        trained_models[tech] = model

    print(f"\n{'='*10} Training Pipeline Complete for Category: {tech_category_to_train} {'='*10}")


# --- Main Execution (Modified) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NMS Optimizer models with validation and early stopping.")
    parser.add_argument("--category", type=str, required=True, help="Technology category to train models for.")
    parser.add_argument("--ship", type=str, default="standard", help="Ship type.")
    parser.add_argument("--width", type=int, default=4, help="Grid width model was trained for.")
    parser.add_argument("--height", type=int, default=3, help="Grid height model was trained for.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay (L2 regularization). Try lower values like 1e-4 or 0.") # Adjusted default WD
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--scheduler_step", type=int, default=20, help="StepLR: Number of epochs before decaying LR.") # Adjusted default step
    parser.add_argument("--scheduler_gamma", type=float, default=0.5, help="StepLR: Multiplicative factor of LR decay.") # Adjusted default gamma
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="Base directory for TensorBoard logs (ship/tech subdirs will be created).")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_SAVE_DIR, help="Base directory to save best trained models.")
    parser.add_argument("--data_source_dir", type=str, default=DEFAULT_DATA_SOURCE_DIR, help="Base directory containing generated .npz data files (expects ship/tech subdirs).")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation (0.0 to 1.0).")
    parser.add_argument("--es_patience", type=int, default=15, help="Early Stopping: Number of epochs to wait for improvement.")
    parser.add_argument("--es_metric", type=str, default="val_loss", choices=["val_loss", "val_miou"], help="Early Stopping: Metric to monitor ('val_loss' or 'val_miou').")

    args = parser.parse_args()

    if not 0.0 <= args.val_split < 1.0:
        parser.error("--val_split must be between 0.0 and 1.0 (exclusive of 1.0)")

    effective_es_metric = args.es_metric
    if args.es_metric == "val_miou" and not torchmetrics:
        print("Warning: Cannot use 'val_miou' for early stopping as torchmetrics is not installed. Defaulting to 'val_loss'.")
        effective_es_metric = "val_loss"

    config = {
        "grid_width": args.width, "grid_height": args.height, "ship": args.ship,
        "tech_category_to_train": args.category, "learning_rate": args.lr,
        "weight_decay": args.wd, "num_epochs": args.epochs, "batch_size": args.batch_size,
        "scheduler_step_size": args.scheduler_step, "scheduler_gamma": args.scheduler_gamma,
        "base_log_dir": args.log_dir, "base_model_save_dir": args.model_dir,
        "base_data_source_dir": args.data_source_dir,
        "validation_split": args.val_split, "early_stopping_patience": args.es_patience,
        "early_stopping_metric": effective_es_metric,
    }

    start_time_all = time.time()
    print(f"Starting model training process...")
    print(f"Configuration: {config}")

    # Ensure the base model save directory exists
    os.makedirs(config["base_model_save_dir"], exist_ok=True)

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
        base_model_save_dir=config["base_model_save_dir"],
        base_data_source_dir=config["base_data_source_dir"],
        scheduler_step_size=config["scheduler_step_size"],
        scheduler_gamma=config["scheduler_gamma"],
        validation_split=config["validation_split"],
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_metric=config["early_stopping_metric"],
    )

    end_time_all = time.time()
    print(f"\n{'='*20} Model Training Complete {'='*20}")
    print(f"Total time: {end_time_all - start_time_all:.2f} seconds.")
    print(f"Best models saved directly in: {os.path.abspath(config['base_model_save_dir'])}")
