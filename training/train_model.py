# training/train_model.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
# <<< Import the scheduler >>>
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import time
import glob
import argparse

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
from optimizer import get_tech_modules_for_training
from modules import modules

# --- Configuration ---
DEFAULT_DATA_SOURCE_DIR = "generated_batches"
# --- End Configuration ---


# --- Model Definition (remains the same) ---
class ModulePlacementCNN(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_output_classes):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_output_classes = num_output_classes
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)
        self.output_conv = nn.Conv2d(64, num_output_classes, kernel_size=1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x_placement = self.output_conv(x)
        return x_placement


# --- Dataset (remains the same) ---
class PlacementDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]


# --- Training Function (Modified) ---
def train_model(
    train_loader,
    grid_height,
    grid_width,
    num_output_classes,
    num_epochs,
    learning_rate,
    weight_decay,
    log_dir,
    # <<< Add scheduler parameters >>>
    scheduler_step_size,
    scheduler_gamma,
):
    """Initializes and trains a ModulePlacementCNN model with LR scheduling."""
    print(f"Starting training for {log_dir}...")
    print(f"  Num output classes: {num_output_classes}")
    print(f"  Grid dimensions: {grid_height}x{grid_width}")
    print(f"  Total samples in loader: {len(train_loader.dataset)}")
    print(f"  Scheduler: StepLR (step_size={scheduler_step_size}, gamma={scheduler_gamma})") # Log scheduler info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training device: {device}")

    model = ModulePlacementCNN(
        input_channels=1, grid_height=grid_height, grid_width=grid_width, num_output_classes=num_output_classes
    ).to(device)
    criterion_placement = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # <<< Initialize the scheduler >>>
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss_placement = 0.0
        start_epoch_time = time.time()
        for i, (inputs, targets_placement) in enumerate(train_loader):
            inputs, targets_placement = inputs.to(device), targets_placement.to(device)
            optimizer.zero_grad()
            outputs_placement = model(inputs)
            loss_placement = criterion_placement(outputs_placement, targets_placement.long())
            loss = loss_placement
            loss.backward()
            optimizer.step()
            running_loss_placement += loss_placement.item()

        avg_loss_placement = running_loss_placement / len(train_loader)
        epoch_time = time.time() - start_epoch_time

        # <<< Get current learning rate for logging >>>
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Placement Loss: {avg_loss_placement:.4f}, "
            f"LR: {current_lr:.6f}, " # Log current LR
            f"Time: {epoch_time:.2f}s"
        )
        writer.add_scalar("Loss/epoch_placement", avg_loss_placement, epoch)
        writer.add_scalar("LearningRate/epoch", current_lr, epoch) # Log LR to TensorBoard
        writer.add_scalar("Time/epoch", epoch_time, epoch)

        # <<< Step the scheduler after each epoch >>>
        scheduler.step()

    print(f"Finished Training for {log_dir}")
    writer.close()
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
    # <<< Add scheduler parameters >>>
    scheduler_step_size,
    scheduler_gamma,
):
    """
    Orchestrates the training process, loading data and calling train_model.
    """
    # --- Get Tech Keys (remains the same) ---
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

    trained_models = {}

    for tech in tech_keys_to_train:
        print(f"\n{'='*10} Processing Tech: {tech} {'='*10}")

        # --- 1. Find and Load Data (remains the same) ---
        file_pattern = os.path.join(data_source_dir, f"data_{ship}_{tech}_*.npz")
        data_files = glob.glob(file_pattern)

        if not data_files:
            print(f"Warning: No data files found matching pattern '{file_pattern}'. Skipping tech '{tech}'.")
            continue

        print(f"Found {len(data_files)} data files for tech '{tech}'. Loading...")
        all_X_data = []
        all_y_data = []
        total_loaded_samples = 0
        load_start_time = time.time()

        for filepath in data_files:
            try:
                with np.load(filepath) as npz_file:
                    x_batch = npz_file["X"]
                    y_batch = npz_file["y"]
                    if x_batch.shape[1:] != (grid_height, grid_width) or y_batch.shape[1:] != (grid_height, grid_width):
                        print(f"Warning: Shape mismatch in {filepath}. Skipping file.")
                        continue
                    if x_batch.shape[0] != y_batch.shape[0]:
                        print(f"Warning: Sample count mismatch in {filepath}. Skipping file.")
                        continue
                    all_X_data.append(x_batch)
                    all_y_data.append(y_batch)
                    total_loaded_samples += x_batch.shape[0]
            except Exception as e:
                print(f"Error loading data from {filepath}: {e}. Skipping file.")
                continue

        if not all_X_data:
            print(f"Warning: No valid data could be loaded for tech '{tech}'. Skipping.")
            continue

        try:
            X_data_np = np.concatenate(all_X_data, axis=0)
            y_data_np = np.concatenate(all_y_data, axis=0)
            load_time = time.time() - load_start_time
            print(f"Loaded and concatenated {total_loaded_samples} total samples for tech '{tech}' in {load_time:.2f}s.")
        except Exception as e:
            print(f"Error concatenating data arrays for tech '{tech}': {e}. Skipping.")
            continue

        # --- Determine num_output_classes (remains the same) ---
        tech_modules = get_tech_modules_for_training(modules, ship, tech)
        if not tech_modules:
            print(f"Warning: Could not get modules for tech '{tech}'. Skipping.")
            continue
        num_output_classes = len(tech_modules) + 1
        if num_output_classes <= 1:
            print(f"Skipping tech '{tech}': Not enough output classes ({num_output_classes}).")
            continue

        # --- 2. Prepare Dataset and DataLoader (remains the same) ---
        try:
            X_train_tensor = torch.tensor(X_data_np, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_data_np, dtype=torch.long)
        except Exception as e:
            print(f"Error converting NumPy arrays to Tensors for tech '{tech}': {e}. Skipping.")
            continue

        train_dataset = PlacementDataset(X_train_tensor, y_train_tensor)
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

        # --- 3. Train Model ---
        log_dir = os.path.join(base_log_dir, ship, tech_category_to_train, tech)
        model = train_model(
            train_loader,
            grid_height,
            grid_width,
            num_output_classes,
            num_epochs,
            learning_rate,
            weight_decay,
            log_dir,
            # <<< Pass scheduler parameters >>>
            scheduler_step_size,
            scheduler_gamma,
        )

        # --- 4. Save Model (remains the same) ---
        model_filename = f"model_{ship}_{tech}.pth"
        model_save_path = os.path.join(model_save_dir, model_filename)
        try:
            model.to("cpu")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved trained model for tech '{tech}' to {model_save_path}")
            trained_models[tech] = model
        except Exception as e:
            print(f"Error saving model for tech '{tech}': {e}")

    print(f"\n{'='*10} Training Pipeline Complete for Category: {tech_category_to_train} {'='*10}")


# --- Main Execution (Modified) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NMS Optimizer models from generated data batches.")
    parser.add_argument("--category", type=str, required=True, help="Technology category to train models for.")
    parser.add_argument("--ship", type=str, default="standard", help="Ship type.")
    parser.add_argument("--width", type=int, default=4, help="Grid width model was trained for.")
    parser.add_argument("--height", type=int, default=3, help="Grid height model was trained for.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate.") # Clarified help
    parser.add_argument("--wd", type=float, default=1e-3, help="Weight decay (L2 regularization).")
    parser.add_argument("--epochs", type=int, default=96, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--log_dir", type=str, default="runs_placement_only", help="Base directory for TensorBoard logs.")
    parser.add_argument("--model_dir", type=str, default="trained_models", help="Directory to save trained models.")
    parser.add_argument("--data_source_dir", type=str, default=DEFAULT_DATA_SOURCE_DIR, help="Directory containing generated .npz data files.")
    # <<< Add scheduler arguments >>>
    parser.add_argument("--scheduler_step", type=int, default=30, help="StepLR: Number of epochs before decaying LR.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="StepLR: Multiplicative factor of LR decay.")

    args = parser.parse_args()

    config = {
        "grid_width": args.width,
        "grid_height": args.height,
        "ship": args.ship,
        "tech_category_to_train": args.category,
        "learning_rate": args.lr,
        "weight_decay": args.wd,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        # <<< Add scheduler config >>>
        "scheduler_step_size": args.scheduler_step,
        "scheduler_gamma": args.scheduler_gamma,
        "base_log_dir": args.log_dir,
        "model_save_dir": args.model_dir,
        "data_source_dir": args.data_source_dir,
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
        # <<< Pass scheduler config >>>
        scheduler_step_size=config["scheduler_step_size"],
        scheduler_gamma=config["scheduler_gamma"],
    )

    end_time_all = time.time()
    print(f"\n{'='*20} Model Training Complete {'='*20}")
    print(f"Total time: {end_time_all - start_time_all:.2f} seconds.")

