"""
Tests for ModulePlacementCNN architecture, multi-channel support, and attention.
"""

import unittest
import torch
from src.model_definition import ModulePlacementCNN, FocalLoss


class TestModulePlacementCNNArchitecture(unittest.TestCase):
    """Test suite for ModulePlacementCNN enhancements."""

    def test_model_supports_multi_channel_input(self):
        """Model should accept in_channels parameter (default 4: SC, Active, CoordX, CoordY)."""
        model = ModulePlacementCNN(grid_height=3, grid_width=4, num_output_classes=5, in_channels=4)
        x = torch.randn(2, 4, 3, 4)
        output = model(x)
        self.assertEqual(output.shape, (2, 5, 3, 4))

    def test_model_backwards_compatibility_single_channel(self):
        """Model should still support in_channels=1 for legacy saved checkpoints."""
        model = ModulePlacementCNN(grid_height=3, grid_width=4, num_output_classes=5, in_channels=1)
        x = torch.randn(2, 1, 3, 4)
        output = model(x)
        self.assertEqual(output.shape, (2, 5, 3, 4))

    def test_model_has_spatial_attention_layer(self):
        """Model should include a spatial self-attention mechanism."""
        model = ModulePlacementCNN(grid_height=3, grid_width=4, num_output_classes=5, in_channels=4)
        self.assertTrue(hasattr(model, "attention") or hasattr(model, "self_attn"))

    def test_model_gradient_flow_through_attention(self):
        """Gradients should propagate backwards through the attention and convolution blocks."""
        model = ModulePlacementCNN(grid_height=3, grid_width=4, num_output_classes=5, in_channels=4)
        x = torch.randn(2, 4, 3, 4, requires_grad=True)
        target = torch.randint(0, 5, (2, 3, 4))

        criterion = torch.nn.CrossEntropyLoss()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()

        self.assertIsNotNone(x.grad)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")


class TestFocalLoss(unittest.TestCase):
    """Test suite for FocalLoss implementation."""

    def test_focal_loss_reduces_easy_example_loss(self):
        """Focal loss should down-weight well-classified examples compared to standard CE."""
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        focal = FocalLoss(gamma=2.0, reduction="none")

        # Confident correct prediction: logit 10.0 for class 1
        logits = torch.tensor([[0.0, 10.0, 0.0]])
        target = torch.tensor([1])

        loss_ce = ce(logits, target)
        loss_focal = focal(logits, target)

        self.assertLess(loss_focal.item(), loss_ce.item())

    def test_focal_loss_supports_class_weights(self):
        """Focal loss should properly apply alpha/class weights."""
        weights = torch.tensor([1.0, 5.0, 1.0])
        focal = FocalLoss(gamma=2.0, weight=weights, reduction="mean")

        logits = torch.randn(4, 3, 2, 2)
        target = torch.randint(0, 3, (4, 2, 2))
        loss = focal(logits, target)

        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_focal_loss_backward(self):
        """Focal loss should compute valid gradients during backpropagation."""
        focal = FocalLoss(gamma=2.0, reduction="mean")
        logits = torch.randn(2, 4, 3, 3, requires_grad=True)
        target = torch.randint(0, 4, (2, 3, 3))

        loss = focal(logits, target)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())


if __name__ == "__main__":
    unittest.main()
