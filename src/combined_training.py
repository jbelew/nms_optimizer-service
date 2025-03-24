import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
# Assuming data_generator.py is in the same directory
# from data_generator import generate_training_data  # Remove this import
from optimizer import simulated_annealing_optimization, Grid, modules, get_tech_modules_for_training, print_grid_compact

# --- Data Generation ---
def generate_training_data(num_samples, grid_width, grid_height, max_supercharged, ship, num_techs):
    X_train = []
    y_train = []
    bonus_scores = []  # Store bonus scores
    for i in range(num_samples):
        grid = Grid(grid_width, grid_height)
        num_supercharged = random.randint(0, max_supercharged)
        supercharged_positions = random.sample(
            [(x, y) for y in range(grid_height) for x in range(grid_width)],
            num_supercharged,
        )
        for x, y in supercharged_positions:
            grid.set_supercharged(x, y, True)

        # Randomly select a technology
        tech_keys = [tech_data["key"] for tech_data in modules[ship]["types"]["weapons"]]
        if not tech_keys:
            print(f"Error: No tech keys found for ship {ship}")
            continue

        tech = random.choice(tech_keys)

        try:
            optimized_grid, best_bonus = simulated_annealing_optimization(grid, ship, modules, tech)
            print(f"Sample {i + 1} - Best Bonus: {best_bonus}")
            print_grid_compact(optimized_grid)
        except Exception as e:
            print(f"Error during optimization for sample {i + 1}: {e}")
            continue  # Skip this sample if optimization fails

        # Encode the grid and solution (integer matrix encoding)
        input_matrix = np.zeros((grid_height, grid_width), dtype=int)
        output_matrix = np.zeros((grid_height, grid_width), dtype=int)
        tech_modules = get_tech_modules_for_training(modules, ship, tech)
        if not tech_modules:
            print(f"Warning: No tech modules found for ship='{ship}', tech='{tech}'")
            continue
        module_id_mapping = {module["unique_id"]: i + 1 for i, module in enumerate(tech_modules)}

        # Check for empty tech_modules
        if not tech_modules:
            print(f"Warning: No tech modules found for ship='{ship}', tech='{tech}'")
            continue  # Skip sample if no modules

        none_count = 0  # Count 'None' modules
        for y in range(grid_height):
            for x in range(grid_width):
                input_matrix[y, x] = int(grid.get_cell(x, y)["supercharged"])
                module_id = optimized_grid.get_cell(x, y)["module"]
                if module_id is None:
                    none_count += 1
                    output_matrix[y, x] = 0  # Map None to class 0 (background)
                else:
                    output_matrix[y, x] = module_id_mapping.get(f"{ship}-{tech}-{module_id}", 0)
                # Print debugging info:
                print(
                    f"Sample {i + 1}, position ({x}, {y}): module_id='{module_id}', mapped to class index {output_matrix[y, x]}, mapping: {module_id_mapping}"
                )
        if none_count > 3:  # tune this threshold
            print(f"Skipping sample {i+1} because it has too many None modules ({none_count})")
            continue

        X_train.append(input_matrix)
        y_train.append(output_matrix)
        bonus_scores.append(best_bonus)  # Store the bonus score

    # Determine the maximum number of output classes
    max_output_classes = len(module_id_mapping) if module_id_mapping else 0
    if not X_train:
        print("X_train is empty. Check data generation logic.")
        return [], [], [], 0

    print(f"Number of output classes: {max_output_classes}")  # Debugging
    print(f"Length of X_train: {len(X_train)}")
    print(f"Length of y_train: {len(y_train)}")
    print(f"Length of bonus_scores: {len(bonus_scores)}")

    return X_train, y_train, bonus_scores, max_output_classes



# Example usage:  Moved outside the function definition
num_samples = 32  # Adjust this to a larger number for better results
grid_width = 10
grid_height = 6
max_supercharged = 4
ship = "Exotic"
num_techs = len([tech_data["key"] for tech_data in modules[ship]["types"]["weapons"]])

X_train, y_train, bonus_scores, num_output_classes = generate_training_data(num_samples, grid_width, grid_height, max_supercharged, ship, num_techs)

# Check if data was generated
if not X_train:
    print("Error: No training data generated. Exiting.")
    exit()

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train), dtype=torch.long)
bonus_scores = torch.tensor(np.array(bonus_scores), dtype=torch.float32)  # Convert bonus scores to tensor

# Create a custom dataset that includes bonus scores
class BonusDataset(data.Dataset):
    def __init__(self, X, y, bonus):
        self.X = X
        self.y = y
        self.bonus = bonus

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.bonus[idx]

train_dataset = BonusDataset(X_train, y_train, bonus_scores)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



class ModulePlacementCNN(nn.Module):
    def __init__(self, input_channels, num_output_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Calculate flattened size dynamically based on input size.  This is CRUCIAL.
        self._to_linear = None
        self.fc1_placement = nn.Linear(32 * 1 * 2, 128)  # Adjusted, but will be overwritten
        self.fc2_placement = nn.Linear(128, num_output_classes)
        self.fc1_bonus = nn.Linear(32 * 1 * 2, 128)  # Adjusted, but will be overwritten
        self.fc2_bonus = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: [batch_size, input_channels, height, width]  e.g., [32, 1, 6, 10]
        print(f"Initial input shape: {x.shape}")
        x = self.pool(self.relu(self.conv1(x)))
        print(f"After conv1 and pool: {x.shape}")
        x = self.pool(self.relu(self.conv2(x)))
        print(f"After conv2 and pool: {x.shape}")

        if self._to_linear is None:
            self._to_linear = x.size()[1] * x.size()[2] * x.size()[3]
            print(f"Calculated _to_linear: {self._to_linear}")
            self.fc1_placement = nn.Linear(self._to_linear, 128)
            self.fc1_bonus = nn.Linear(self._to_linear, 128)

        x = x.view(x.size(0), -1)  # Flatten
        print(f"After flatten: {x.shape}")
        x_placement = self.relu(self.fc1_placement(x))
        x_placement = self.fc2_placement(x_placement)
        print(f"Output placement shape: {x_placement.shape}")
        x_bonus = self.relu(self.fc1_bonus(x))
        x_bonus = self.fc2_bonus(x_bonus)
        print(f"Output bonus shape: {x_bonus.shape}")
        return x_placement, x_bonus



# Hyperparameters
learning_rate = 0.001
num_epochs = 100
batch_size = 32

# Initialize model, loss function, and optimizer
model = ModulePlacementCNN(input_channels=1, num_output_classes=num_output_classes)
criterion_placement = nn.CrossEntropyLoss()
criterion_bonus = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize TensorBoard writer (optional, but recommended)
writer = SummaryWriter()

# Training loop
for epoch in range(num_epochs):
    running_loss_placement = 0.0
    running_loss_bonus = 0.0
    for i, (inputs, targets, bonus_scores) in enumerate(train_loader):
        print(f"Batch {i+1}, Input shape from data loader: {inputs.shape}")
        inputs = inputs.view(-1, 1, grid_height, grid_width)
        print(f"Batch {i+1}, Reshaped input shape: {inputs.shape}")

        print(f"Batch {i+1}, Target shape from data loader: {targets.shape}")
        print(f"Batch {i+1}, Target dtype: {targets.dtype}")  # Print data type
        print(
            f"Batch {i+1}, Target min: {targets.min()}, max: {targets.max()}"
        )  # Print min/max values


        bonus_scores = bonus_scores.view(-1, 1).float()
        print(f"Batch {i+1}, Bonus scores shape: {bonus_scores.shape}")

        optimizer.zero_grad()
        outputs_placement, outputs_bonus = model(inputs)
        print(f"Batch {i+1}, Output placement shape: {outputs_placement.shape}")

        # Calculate loss
        loss_placement = criterion_placement(outputs_placement, targets.long()) # Ensure targets are long type
        loss_bonus = criterion_bonus(outputs_bonus, bonus_scores)
        loss = loss_placement + loss_bonus
        loss.backward()
        optimizer.step()

        running_loss_placement += loss_placement.item()
        running_loss_bonus += loss_bonus.item()

        if i % 100 == 99:
            print(
                f"[{epoch + 1}, {i + 1:5d}] loss_placement: {running_loss_placement / 100:.3f} loss_bonus: {running_loss_bonus / 100:.3f}"
            )
            writer.add_scalar("Loss/train_placement", running_loss_placement / 100, epoch * len(train_loader) + i)
            writer.add_scalar("Loss/train_bonus", running_loss_bonus / 100, epoch * len(train_loader) + i)
            running_loss_placement = 0.0
            running_loss_bonus = 0.0

print("Finished Training")
writer.close()
