# training/model_definition.py
import torch
import torch.nn as nn


class ModulePlacementCNN(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_output_classes):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_output_classes = num_output_classes

        # --- Block 1 ---
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.relu1 = nn.ReLU()

        # --- Block 2 ---
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-5)
        self.relu2 = nn.ReLU()

        # --- Block 3 (Added Depth) ---
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-5)  # Match conv3 output
        self.relu3 = nn.ReLU()
        # --- End Block 3 ---

        # --- Dropout ---
        self.dropout = nn.Dropout(0.4)

        # --- Final Output Layer ---
        self.output_conv = nn.Conv2d(128, num_output_classes, kernel_size=1)

    def forward(self, x):
        # --- Input Check ---
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AT MODEL INPUT !!!")
            return torch.full_like(x, float("nan"))  # Return NaN to trigger outer check

        # --- Block 1 ---
        x = self.conv1(x)
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AFTER conv1 !!!")
        x = self.bn1(x)
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AFTER bn1 !!!")
        x = self.relu1(x)
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AFTER relu1 !!!")

        # --- Block 2 ---
        x = self.conv2(x)
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AFTER conv2 !!!")
        x = self.bn2(x)
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AFTER bn2 !!!")
        x = self.relu2(x)
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AFTER relu2 !!!")

        # --- Block 3 ---
        # <<< Check input to conv3 >>>
        x_before_conv3 = x
        if torch.isnan(x_before_conv3).any():
            print(
                "!!! NaN DETECTED *BEFORE* conv3 !!!"
            )  # Should not happen based on logs, but good check
        # <<< End Check >>>

        x = self.conv3(x)
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AFTER conv3 !!!")
            # <<< Print stats of input that caused NaN in conv3 >>>
            print(
                f"  Input to conv3 min/max/mean: {x_before_conv3.min():.4f} / {x_before_conv3.max():.4f} / {x_before_conv3.mean():.4f}"
            )
            # <<< End Print >>>

        x = self.bn3(x)
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AFTER bn3 !!!")
        x = self.relu3(x)
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AFTER relu3 !!!")

        # --- Dropout ---
        x = self.dropout(x)
        if torch.isnan(x).any():
            print("!!! NaN DETECTED AFTER dropout !!!")

        # --- Final Output Layer ---
        x_placement = self.output_conv(x)
        if torch.isnan(x_placement).any():
            print("!!! NaN DETECTED AFTER output_conv !!!")

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
            return torch.full(
                nan_output_shape, float("nan"), device=x.device, dtype=x.dtype
            )

        return x_placement


# Optional: Add to __all__ if needed
__all__ = ["ModulePlacementCNN"]
