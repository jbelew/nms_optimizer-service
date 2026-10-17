"""Unit tests for the batch model training discovery and pipeline (scripts/training/train_all_models.py)."""

import os
import tempfile
import unittest
import numpy as np

from scripts.training.train_all_models import (
    DatasetTarget,
    discover_dataset_targets,
    filter_targets,
    load_and_preprocess_target,
    compute_model_filename,
)


class TestTrainAllModels(unittest.TestCase):
    """Test suite for batch training target discovery, dimension resolution, and preprocessing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.temp_dir.name, "batches")
        self.model_dir = os.path.join(self.temp_dir.name, "models")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_sample_npz(self, subpath: str, filename: str, shape: tuple, num_classes: int = 5):
        """Helper to create dummy .npz files with X_supercharge, y, and X_inactive_mask."""
        dir_path = os.path.join(self.data_dir, subpath)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, filename)

        n, h, w = shape
        sc = np.zeros((n, h, w), dtype=np.int8)
        sc[:, 0, 0] = 1
        inact = np.zeros((n, h, w), dtype=np.int8)
        y = np.random.randint(0, num_classes, size=(n, h, w), dtype=np.int8)

        np.savez_compressed(file_path, X_supercharge=sc, X_inactive_mask=inact, y=y)
        return file_path

    def test_compute_model_filename(self):
        """Tests canonical model filename generation for standard and pre-suffixed techs."""
        # Standard tech
        name = compute_model_filename("corvette", "launch", 4, 3)
        self.assertEqual(name, "model_corvette_launch_4x3.pth")

        # Transposed tech (file was 2x3, model is 3x2)
        name = compute_model_filename("standard", "cyclotron", 3, 2)
        self.assertEqual(name, "model_standard_cyclotron_3x2.pth")

        # Tech that already contains dimensions suffix (e.g., solar/photonix_4x3)
        name = compute_model_filename("solar", "photonix_4x3", 4, 3)
        self.assertEqual(name, "model_solar_photonix_4x3.pth")

    def test_discover_dataset_targets(self):
        """Tests discovery of datasets and detection of canonical dimensions and transpose requirement."""
        # Target 1: width >= height (4x3), no transpose needed
        self._create_sample_npz(
            "corvette/launch",
            "data_corvette_launch_4x3_20261010_1.npz",
            shape=(16, 3, 4),  # H=3, W=4
            num_classes=13,
        )

        # Target 2: width < height (2x3), transpose needed -> model becomes 3x2
        self._create_sample_npz(
            "standard/cyclotron",
            "data_standard_cyclotron_2x3_20261010_1.npz",
            shape=(16, 3, 2),  # H=3, W=2
            num_classes=6,
        )

        targets = discover_dataset_targets(self.data_dir, self.model_dir)
        self.assertEqual(len(targets), 2)

        target_map = {f"{t.ship}/{t.tech}": t for t in targets}

        # Check corvette/launch
        launch_target = target_map["corvette/launch"]
        self.assertEqual(launch_target.ship, "corvette")
        self.assertEqual(launch_target.tech, "launch")
        self.assertEqual(launch_target.file_width, 4)
        self.assertEqual(launch_target.file_height, 3)
        self.assertEqual(launch_target.model_width, 4)
        self.assertEqual(launch_target.model_height, 3)
        self.assertFalse(launch_target.needs_transpose)
        self.assertEqual(launch_target.num_classes, 13)
        self.assertEqual(launch_target.model_filename, "model_corvette_launch_4x3.pth")

        # Check standard/cyclotron
        cyclo_target = target_map["standard/cyclotron"]
        self.assertEqual(cyclo_target.ship, "standard")
        self.assertEqual(cyclo_target.tech, "cyclotron")
        self.assertEqual(cyclo_target.file_width, 2)
        self.assertEqual(cyclo_target.file_height, 3)
        self.assertEqual(cyclo_target.model_width, 3)
        self.assertEqual(cyclo_target.model_height, 2)
        self.assertTrue(cyclo_target.needs_transpose)
        self.assertEqual(cyclo_target.num_classes, 6)
        self.assertEqual(cyclo_target.model_filename, "model_standard_cyclotron_3x2.pth")

    def test_filter_targets(self):
        """Tests filtering targets by ship, tech, and skip_existing flag."""
        t1 = DatasetTarget(
            ship="corvette",
            tech="launch",
            dir_path="dummy/1",
            file_count=5,
            file_width=4,
            file_height=3,
            model_width=4,
            model_height=3,
            needs_transpose=False,
            num_classes=13,
            model_filename="model_corvette_launch_4x3.pth",
            model_path=os.path.join(self.model_dir, "model_corvette_launch_4x3.pth"),
        )
        t2 = DatasetTarget(
            ship="corvette",
            tech="pulse",
            dir_path="dummy/2",
            file_count=5,
            file_width=4,
            file_height=3,
            model_width=4,
            model_height=3,
            needs_transpose=False,
            num_classes=11,
            model_filename="model_corvette_pulse_4x3.pth",
            model_path=os.path.join(self.model_dir, "model_corvette_pulse_4x3.pth"),
        )
        t3 = DatasetTarget(
            ship="standard",
            tech="cyclotron",
            dir_path="dummy/3",
            file_count=5,
            file_width=2,
            file_height=3,
            model_width=3,
            model_height=2,
            needs_transpose=True,
            num_classes=6,
            model_filename="model_standard_cyclotron_3x2.pth",
            model_path=os.path.join(self.model_dir, "model_standard_cyclotron_3x2.pth"),
        )
        all_targets = [t1, t2, t3]

        # Ship filter
        corvette_only = filter_targets(all_targets, ships=["corvette"])
        self.assertEqual(len(corvette_only), 2)

        # Tech filter
        pulse_only = filter_targets(all_targets, techs=["pulse"])
        self.assertEqual(len(pulse_only), 1)
        self.assertEqual(pulse_only[0].tech, "pulse")

        # Skip existing
        # Create t1 model file on disk
        with open(t1.model_path, "w") as f:
            f.write("dummy")

        not_existing = filter_targets(all_targets, skip_existing=True)
        self.assertEqual(len(not_existing), 2)
        self.assertNotIn(t1, not_existing)

    def test_load_and_preprocess_target(self):
        """Tests loading, transposing arrays, and constructing DataLoaders."""
        # Create a transposed dataset (2x3 on disk, model will be 3x2)
        self._create_sample_npz(
            "standard/cyclotron",
            "data_1.npz",
            shape=(32, 3, 2),  # H=3, W=2
            num_classes=6,
        )
        self._create_sample_npz(
            "standard/cyclotron",
            "data_2.npz",
            shape=(32, 3, 2),  # H=3, W=2
            num_classes=6,
        )

        target = DatasetTarget(
            ship="standard",
            tech="cyclotron",
            dir_path=os.path.join(self.data_dir, "standard/cyclotron"),
            file_count=2,
            file_width=2,
            file_height=3,
            model_width=3,
            model_height=2,
            needs_transpose=True,
            num_classes=6,
            model_filename="model_standard_cyclotron_3x2.pth",
            model_path=os.path.join(self.model_dir, "model_standard_cyclotron_3x2.pth"),
        )

        train_loader, val_loader, class_weights = load_and_preprocess_target(
            target,
            batch_size=16,
            val_split=0.25,
            in_channels=4,
        )

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertEqual(len(class_weights), 6)

        # Inspect a batch from train_loader
        inputs, targets = next(iter(train_loader))
        # Inputs shape: (batch_size, in_channels=4, model_height=2, model_width=3)
        self.assertEqual(inputs.shape, (16, 4, 2, 3))
        # Targets shape: (batch_size, model_height=2, model_width=3)
        self.assertEqual(targets.shape, (16, 2, 3))


if __name__ == "__main__":
    unittest.main()
