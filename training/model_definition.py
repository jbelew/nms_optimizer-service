# training/model_definition.py
import torch
import torch.nn as nn

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

# Optional: Add to __all__ if needed
__all__ = ["ModulePlacementCNN"]
