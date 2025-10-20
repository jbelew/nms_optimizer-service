# training/model_definition.py
"""
This module defines the PyTorch neural network architecture used for predicting
module placements.

The `ModulePlacementCNN` class implements a Convolutional Neural Network (CNN)
designed to take a grid's supercharge layout as input and produce a probability
distribution for the placement of each possible module in each cell.
"""
import torch
import torch.nn as nn


class ModulePlacementCNN(nn.Module):
    """A Convolutional Neural Network for module placement prediction.

    This network uses a series of convolutional layers, batch normalization,
    and a residual block to learn the spatial relationships in the grid.
    The final output is a set of logits for each possible module class in each
    cell of the grid.

    Attributes:
        grid_height (int): The height of the input grid.
        grid_width (int): The width of the input grid.
        num_output_classes (int): The number of possible output classes,
            including the "empty" class.
    """

    def __init__(self, grid_height, grid_width, num_output_classes):
        """Initializes the ModulePlacementCNN model.

        Args:
            grid_height (int): The height of the input grid.
            grid_width (int): The width of the input grid.
            num_output_classes (int): The number of possible module classes
                to predict, including the background/empty class.
        """
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_output_classes = num_output_classes

        # --- Block 1 ---
        # Takes a single channel input (the supercharge grid)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.relu1 = nn.ReLU()

        # --- Block 2 ---
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-5)
        self.relu2 = nn.ReLU()

        # --- Residual Block 3 (Replaces old Block 3) ---
        self.res_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res_bn1 = nn.BatchNorm2d(128, eps=1e-5)
        self.res_relu1 = nn.ReLU()
        self.res_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res_bn2 = nn.BatchNorm2d(128, eps=1e-5)
        self.res_relu2 = nn.ReLU()
        # --- End Block 3 ---

        # --- Dropout ---
        self.dropout = nn.Dropout(0.5)  # Increased dropout slightly for the deeper model

        # --- Final Output Layer ---
        self.output_conv = nn.Conv2d(128, num_output_classes, kernel_size=1)

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor, representing the supercharged
                layout of the grid. Shape: (N, 1, H, W).

        Returns:
            torch.Tensor: The output logits for each class at each grid cell.
                Shape: (N, num_output_classes, H, W).
        """
        # --- Input Check ---
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AT MODEL INPUT !!!")
            return torch.full_like(x, float("nan"))

        # --- Block 1 ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # --- Block 2 ---
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # --- Residual Block 3 ---
        # Store the input for the skip connection
        residual = x

        # Pass through the residual block layers
        out = self.res_conv1(x)
        out = self.res_bn1(out)
        out = self.res_relu1(out)
        out = self.res_conv2(out)
        out = self.res_bn2(out)

        # Add the skip connection
        out += residual
        # Apply the final activation for the block
        out = self.res_relu2(out)

        # --- Dropout ---
        out = self.dropout(out)

        # --- Final Output Layer ---
        x_placement = self.output_conv(out)

        # Final check (catches any NaNs that propagated)
        if torch.isnan(x_placement).any():
            print("!!! NaN DETECTED AT FINAL OUTPUT !!!")
            # Return NaN so the check in train_model catches it cleanly
            nan_output_shape = (
                x_placement.shape[0],
                self.num_output_classes,
                self.grid_height,
                self.grid_width,
            )
            # Ensure NaN tensor is on the same device and dtype as expected output
            return torch.full(nan_output_shape, float("nan"), device=x.device, dtype=x.dtype)

        return x_placement


# Optional: Add to __all__ if needed
__all__ = ["ModulePlacementCNN"]
