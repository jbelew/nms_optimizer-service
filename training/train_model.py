# training/train_model_script.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import time

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Imports from your project ---
from optimizer import get_tech_modules_for_training # Only need this utility
from modules import modules

# --- Configuration ---
# Shared directory config (could be moved to a shared config file later)
TRAINING_DATA_DIR = "training_data"
# --- End Configuration ---


# --- Model Definition ---
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
        self.output_conv = nn.Conv2d(64, num_output_classes, kernel_size=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        features = self.relu2(self.conv2(x))
        x_placement = self.output_conv(features)
        return x_placement


# --- Dataset ---
class PlacementDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Add channel dimension to X, return only X and y
        return self.X[idx].unsqueeze(0), self.y[idx]


# --- Training Function ---
def train_model(train_loader, grid_height, grid_width, num_output_classes, num_epochs, learning_rate, log_dir):
    """Initializes and trains a ModulePlacementCNN model for placement prediction only."""
    print(f"Starting training for {log_dir}...")
    print(f"  Num output classes: {num_output_classes}")
    print(f"  Grid dimensions: {grid_height}x{grid_width}")
    print(f"  Total samples in loader: {len(train_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training device: {device}")

    model = ModulePlacementCNN(
        input_channels=1, grid_height=grid_height, grid_width=grid_width, num_output_classes=num_output_classes
    ).to(device)
    criterion_placement = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Consider adding weight_decay here

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # --- Early Stopping Setup (Optional but Recommended) ---
    # best_val_loss = float('inf')
    # patience = 5 # Number of epochs to wait for improvement
    # patience_counter = 0
    # --- End Early Stopping Setup ---

    for epoch in range(num_epochs):
        model.train()
        running_loss_placement = 0.0
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
        print(f"Epoch [{epoch + 1}/{num_epochs}], " f"Placement Loss: {avg_loss_placement:.4f}")
        writer.add_scalar("Loss/epoch_placement", avg_loss_placement, epoch)

        # --- Early Stopping Check (Optional) ---
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for val_inputs, val_targets in val_loader: # Requires a validation loader
        #         val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
        #         val_outputs = model(val_inputs)
        #         val_loss += criterion_placement(val_outputs, val_targets.long()).item()
        # avg_val_loss = val_loss / len(val_loader)
        # writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        # print(f"  Validation Loss: {avg_val_loss:.4f}")
        #
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     patience_counter = 0
        #     # Optionally save the best model here
        #     # torch.save(model.state_dict(), best_model_path)
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping triggered at epoch {epoch + 1}")
        #         break
        # --- End Early Stopping Check ---


    print(f"Finished Training for {log_dir}")
    writer.close()
    return model


# --- Training Orchestration Function (Loads from Files) ---
def run_training_from_files(
    ship,
    tech_category_to_train,
    grid_width,
    grid_height,
    learning_rate,
    num_epochs,
    batch_size,
    base_log_dir,
    model_save_dir,
    data_dir=TRAINING_DATA_DIR
):
    """
    Orchestrates the training process by loading data from .npy files.
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
            tech_data["key"] for tech_data in category_data
            if isinstance(tech_data, dict) and "key" in tech_data
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

    trained_models = {}

    for tech in tech_keys_to_train:
        print(f"\n{'='*10} Processing Tech: {tech} {'='*10}")

        # --- 1. Load Data from .npy files ---
        x_file_path = os.path.join(data_dir, f"X_{ship}_{tech}.npy")
        y_file_path = os.path.join(data_dir, f"y_{ship}_{tech}.npy")

        if not os.path.exists(x_file_path) or not os.path.exists(y_file_path):
            print(f"Error: Data files not found for tech '{tech}' at {data_dir}. Skipping.")
            continue

        try:
            print(f"Loading data for tech '{tech}' from {data_dir}...")
            X_data_np = np.load(x_file_path)
            y_data_np = np.load(y_file_path)
            print(f"Loaded {len(X_data_np)} samples.")
        except Exception as e:
            print(f"Error loading data for tech '{tech}': {e}. Skipping.")
            continue

        if X_data_np.size == 0 or y_data_np.size == 0:
            print(f"Warning: Loaded data for tech '{tech}' is empty. Skipping.")
            continue

        # --- Determine num_output_classes ---
        tech_modules = get_tech_modules_for_training(modules, ship, tech)
        if not tech_modules:
             print(f"Warning: Could not get modules for tech '{tech}' to determine class count. Skipping.")
             continue
        num_output_classes = len(tech_modules) + 1
        if num_output_classes <= 1:
            print(f"Skipping tech '{tech}': Not enough output classes ({num_output_classes}). Needs > 1.")
            continue

        # --- 2. Prepare Dataset and DataLoader ---
        try:
            X_train_tensor = torch.tensor(X_data_np, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_data_np, dtype=torch.long) # Target needs to be Long
        except Exception as e:
            print(f"Error converting loaded NumPy arrays to Tensors for tech '{tech}': {e}. Skipping.")
            continue

        train_dataset = PlacementDataset(X_train_tensor, y_train_tensor)
        effective_batch_size = min(batch_size, len(train_dataset))
        if effective_batch_size == 0:
            print(f"Warning: Train dataset for tech '{tech}' is effectively empty. Skipping training.")
            continue

        # Consider adding num_workers and pin_memory if loading is slow
        train_loader = data.DataLoader(
            train_dataset, batch_size=effective_batch_size, shuffle=True
        )

        # --- 3. Train Model ---
        log_dir = os.path.join(base_log_dir, ship, tech_category_to_train, tech)
        model = train_model(
            train_loader, grid_height, grid_width, num_output_classes, num_epochs, learning_rate, log_dir
        )

        # --- 4. Save Model ---
        model_filename = f"model_{ship}_{tech}.pth"
        model_save_path = os.path.join(model_save_dir, model_filename)
        try:
            model.to("cpu") # Ensure model is on CPU before saving state_dict
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved trained model for tech '{tech}' to {model_save_path}")
            trained_models[tech] = model # Store the CPU version if needed
        except Exception as e:
            print(f"Error saving model for tech '{tech}': {e}")

    print(f"\n{'='*10} Training Pipeline Complete for Category: {tech_category_to_train} {'='*10}")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    
    config = {
        # Data/Model Structure
        "grid_width": 4,
        "grid_height": 3,
        "ship": "standard",
        "tech_category_to_train": "Weaponry", # Category to train models for

        # Training Hyperparameters (STARTING POINTS FOR ~8k SAMPLES)
        "learning_rate": 0.001,  # Slightly higher than for 80k
        "num_epochs": 50,       # More epochs, but monitor closely!
        "batch_size": 32,       # Keep as is or try 64

        # Paths
        "base_log_dir": "runs_placement_only",
        "model_save_dir": "trained_models",
        "data_dir": TRAINING_DATA_DIR
    }
    # --- End Configuration ---

    start_time_all = time.time()
    print(f"Starting model training process...")
    print(f"Configuration: {config}")

    # Ensure model save directory exists
    os.makedirs(config["model_save_dir"], exist_ok=True)

    # Run the training pipeline using the configuration
    run_training_from_files(
        ship=config["ship"],
        tech_category_to_train=config["tech_category_to_train"],
        grid_width=config["grid_width"],
        grid_height=config["grid_height"],
        learning_rate=config["learning_rate"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        base_log_dir=config["base_log_dir"],
        model_save_dir=config["model_save_dir"],
        data_dir=config["data_dir"]
    )

    end_time_all = time.time()
    print(f"\n{'='*20} Model Training Complete {'='*20}")
    print(f"Total time: {end_time_all - start_time_all:.2f} seconds.")

