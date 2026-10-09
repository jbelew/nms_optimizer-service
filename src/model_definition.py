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
    """A Convolutional Neural Network with Spatial Self-Attention for module placement prediction.

    This network uses a series of convolutional layers, batch normalization,
    a residual block, and an optional multi-head spatial self-attention block to learn
    both local spatial relationships and global multi-module synergies across the grid.
    The final output is a set of logits for each possible module class in each
    cell of the grid.

    Attributes:
        grid_height (int): The height of the input grid.
        grid_width (int): The width of the input grid.
        num_output_classes (int): The number of possible output classes,
            including the "empty" class.
        in_channels (int): The number of input channels (default 4).
        use_attention (bool): Whether to include the self-attention layer.
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_output_classes: int,
        in_channels: int = 4,
        use_attention: bool = True,
    ):
        """Initializes the ModulePlacementCNN model.

        Args:
            grid_height (int): The height of the input grid.
            grid_width (int): The width of the input grid.
            num_output_classes (int): The number of possible module classes
                to predict, including the background/empty class.
            in_channels (int): Number of input channels (default 4).
            use_attention (bool): Whether to include spatial self-attention.
        """
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_output_classes = num_output_classes
        self.in_channels = in_channels
        self.use_attention = use_attention

        # --- Block 1 ---
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.relu1 = nn.ReLU()

        # --- Block 2 ---
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-5)
        self.relu2 = nn.ReLU()

        # --- Residual Block 3 ---
        self.res_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res_bn1 = nn.BatchNorm2d(128, eps=1e-5)
        self.res_relu1 = nn.ReLU()
        self.res_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res_bn2 = nn.BatchNorm2d(128, eps=1e-5)
        self.res_relu2 = nn.ReLU()
        # --- End Block 3 ---

        # --- Spatial Self-Attention ---
        if self.use_attention:
            self.self_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
            self.attn_norm = nn.LayerNorm(128)

        # --- Dropout ---
        self.dropout = nn.Dropout(0.5)

        # --- Final Output Layer ---
        self.output_conv = nn.Conv2d(128, num_output_classes, kernel_size=1)

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (N, C, H, W).

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
        residual = x
        out = self.res_conv1(x)
        out = self.res_bn1(out)
        out = self.res_relu1(out)
        out = self.res_conv2(out)
        out = self.res_bn2(out)
        out += residual
        out = self.res_relu2(out)

        # --- Spatial Self-Attention ---
        if self.use_attention:
            b, c, h, w = out.shape
            tokens = out.flatten(2).transpose(1, 2)
            attn_out, _ = self.self_attn(tokens, tokens, tokens)
            tokens = self.attn_norm(tokens + attn_out)
            out = tokens.transpose(1, 2).view(b, c, h, w)

        # --- Dropout ---
        out = self.dropout(out)

        # --- Final Output Layer ---
        x_placement = self.output_conv(out)

        # Final check
        if torch.isnan(x_placement).any():
            print("!!! NaN DETECTED AT FINAL OUTPUT !!!")
            nan_output_shape = (
                x_placement.shape[0],
                self.num_output_classes,
                self.grid_height,
                self.grid_width,
            )
            return torch.full(nan_output_shape, float("nan"), device=x.device, dtype=x.dtype)

        return x_placement


def add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """Adds normalized X and Y coordinate channels in [-1, 1] to input tensor x.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).

    Returns:
        torch.Tensor: Augmented tensor of shape (N, C + 2, H, W).
    """
    b, _, h, w = x.shape
    device = x.device
    dtype = x.dtype
    y_coords = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    x_coords = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([x, x_coords, y_coords], dim=1)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss.

    Down-weights easy well-classified examples to focus training on difficult
    decisions (such as ambiguous module boundaries).

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        """Initializes FocalLoss.

        Args:
            gamma (float): Focusing parameter. Default: 2.0.
            weight (Optional[torch.Tensor]): Class weights of shape (num_classes,).
            reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'.
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass for FocalLoss.

        Args:
            inputs (torch.Tensor): Logits of shape (N, C, ...) or (N, C).
            targets (torch.Tensor): Ground truth class indices of shape (N, ...).

        Returns:
            torch.Tensor: Computed focal loss.
        """
        ce_loss_raw = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss_raw)
        modulating_factor = (1.0 - pt) ** self.gamma
        focal_loss = modulating_factor * ce_loss_raw

        if self.weight is not None:
            weight_on_device = self.weight.to(inputs.device)
            alpha = weight_on_device[targets]
            focal_loss = alpha * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


__all__ = ["ModulePlacementCNN", "add_coord_channels", "FocalLoss"]
