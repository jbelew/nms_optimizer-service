#!/usr/bin/env python3
"""Batch Model Training Orchestration Script for No Man's Sky Optimizer Service.

Discovers all generated dataset batches in `scripts/training/generated_batches/`
and trains production machine learning models with the improved architecture:
- 4-channel CoordConv input (supercharged, active mask, CoordConv X, CoordConv Y)
- Spatial Self-Attention layer
- Class-weighted Focal Loss to balance rare modules and background cells
- Cosine Annealing learning rate schedule with validation early stopping

Usage examples:
    # Dry run: discover all targets and inspect parameters without training
    python -m scripts.training.train_all_models --dry_run

    # Train all discovered targets sequentially, saving models to src/trained_models
    python -m scripts.training.train_all_models

    # Train only corvette technologies, skipping already existing models
    python -m scripts.training.train_all_models --ship corvette --skip_existing

    # Train specific technologies with custom epochs and learning rate
    python -m scripts.training.train_all_models --ship corvette --tech launch,pulse --epochs 40 --lr 2e-4
"""

import argparse
import glob
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from scripts.training.train_model import PlacementDataset, train_model
from src.data_loader import get_all_module_data, get_training_module_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class DatasetTarget:
    """Represents a discovered training dataset for a specific ship and technology.

    Attributes:
        ship (str): Ship or platform identifier (e.g., 'corvette', 'standard').
        tech (str): Technology identifier (e.g., 'launch', 'pulse', 'mining').
        dir_path (str): Filesystem path to the folder containing .npz batches.
        file_count (int): Number of .npz batch files available.
        file_width (int): Grid width from array shape in .npz files.
        file_height (int): Grid height from array shape in .npz files.
        model_width (int): Canonical grid width for model (max of width, height).
        model_height (int): Canonical grid height for model (min of width, height).
        needs_transpose (bool): True if array must be transposed to match canonical width >= height.
        num_classes (int): Total output classes including background class 0.
        model_filename (str): Canonical model checkpoint filename.
        model_path (str): Full target path for saved model checkpoint.
    """

    ship: str
    tech: str
    dir_path: str
    file_count: int
    file_width: int
    file_height: int
    model_width: int
    model_height: int
    needs_transpose: bool
    num_classes: int
    model_filename: str
    model_path: str


def compute_model_filename(ship: str, tech: str, model_width: int, model_height: int) -> str:
    """Calculates canonical model checkpoint filename following project naming conventions.

    Args:
        ship (str): Ship platform key (e.g., 'corvette', 'solar').
        tech (str): Technology key (e.g., 'launch', 'photonix_4x3').
        model_width (int): Model grid width.
        model_height (int): Model grid height.

    Returns:
        str: Model filename formatted like 'model_corvette_launch_4x3.pth'.
    """
    dim_suffix = f"_{model_width}x{model_height}"
    if tech.endswith(dim_suffix):
        return f"model_{ship}_{tech}.pth"
    return f"model_{ship}_{tech}_{model_width}x{model_height}.pth"


def discover_dataset_targets(data_dir: str, model_dir: str) -> List[DatasetTarget]:
    """Scans the batch data directory and discovers all trainable technology targets.

    Inspects a sample .npz file from each directory to extract grid dimensions,
    active mask presence, and output class counts. Automatically canonicalizes
    dimensions so that model_width >= model_height and flags datasets needing
    spatial transposition.

    Args:
        data_dir (str): Base directory containing generated batch subdirectories.
        model_dir (str): Output directory where trained models will be saved.

    Returns:
        List[DatasetTarget]: Discovered dataset targets sorted by ship and tech.
    """
    if not os.path.exists(data_dir):
        logging.warning(f"Batch data directory '{data_dir}' does not exist.")
        return []

    npz_files = glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)
    unique_dirs = sorted(list(set(os.path.dirname(f) for f in npz_files)))

    targets: List[DatasetTarget] = []

    for d in unique_dirs:
        rel = os.path.relpath(d, data_dir)
        parts = rel.split(os.sep)
        if len(parts) < 2:
            continue

        ship = parts[0]
        tech = parts[1]

        files = glob.glob(os.path.join(d, "*.npz"))
        if not files:
            continue

        # Inspect the first file for dimension and class metadata
        try:
            with np.load(files[0]) as sample:
                if "X_supercharge" not in sample or "y" not in sample:
                    logging.warning(f"Skipping {d}: Missing required keys in sample file {files[0]}")
                    continue

                sc_arr = sample["X_supercharge"]
                y_arr = sample["y"]

                file_h, file_w = sc_arr.shape[1], sc_arr.shape[2]
                max_class = int(np.max(y_arr))
                num_classes = max_class + 1

                # Verify against data_loader if available
                loader_ids = get_training_module_ids(ship, tech)
                if loader_ids and (len(loader_ids) + 1) >= num_classes:
                    num_classes = len(loader_ids) + 1

        except Exception as e:
            logging.warning(f"Error reading sample from {d}: {e}")
            continue

        model_w = max(file_w, file_h)
        model_h = min(file_w, file_h)
        needs_transpose = file_w < file_h

        model_filename = compute_model_filename(ship, tech, model_w, model_h)
        model_path = os.path.join(model_dir, model_filename)

        target = DatasetTarget(
            ship=ship,
            tech=tech,
            dir_path=d,
            file_count=len(files),
            file_width=file_w,
            file_height=file_h,
            model_width=model_w,
            model_height=model_h,
            needs_transpose=needs_transpose,
            num_classes=num_classes,
            model_filename=model_filename,
            model_path=model_path,
        )
        targets.append(target)

    targets.sort(key=lambda t: (t.ship, t.tech))
    return targets


def filter_targets(
    targets: List[DatasetTarget],
    ships: Optional[List[str]] = None,
    techs: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    skip_existing: bool = False,
    model_dir: Optional[str] = None,
) -> List[DatasetTarget]:
    """Filters discovered dataset targets based on CLI criteria.

    Args:
        targets (List[DatasetTarget]): List of targets to filter.
        ships (Optional[List[str]]): Ship names to keep.
        techs (Optional[List[str]]): Technology names to keep.
        categories (Optional[List[str]]): Technology categories to keep.
        skip_existing (bool): If True, omit targets whose model checkpoint already exists.
        model_dir (Optional[str]): Directory path to check for existing model files.

    Returns:
        List[DatasetTarget]: Filtered list of targets.
    """
    filtered = targets

    if ships:
        ship_set = set(ships)
        filtered = [t for t in filtered if t.ship in ship_set]

    if techs:
        tech_set = set(techs)
        filtered = [t for t in filtered if t.tech in tech_set]

    if categories:
        cat_set = set(categories)
        all_modules = get_all_module_data()
        valid_tech_keys = set()
        for ship_key, s_data in all_modules.items():
            types_dict = s_data.get("types", {})
            for cat_name, tech_list in types_dict.items():
                if cat_name in cat_set and isinstance(tech_list, list):
                    for item in tech_list:
                        if isinstance(item, dict) and "key" in item:
                            valid_tech_keys.add(item["key"])
        filtered = [t for t in filtered if t.tech in valid_tech_keys]

    if skip_existing:
        filtered = [t for t in filtered if not os.path.exists(t.model_path)]

    return filtered


def load_and_preprocess_target(
    target: DatasetTarget,
    batch_size: int = 128,
    val_split: float = 0.2,
    max_files: Optional[int] = None,
    in_channels: int = 4,
    num_workers: int = 0,
) -> Tuple[data.DataLoader, Optional[data.DataLoader], torch.Tensor]:
    """Loads all batch files for a target, applies spatial transposition if needed, and builds DataLoaders.

    Args:
        target (DatasetTarget): The dataset target metadata.
        batch_size (int): Batch size for training and validation loaders.
        val_split (float): Fraction of data reserved for validation.
        max_files (Optional[int]): Maximum number of .npz files to load.
        in_channels (int): Input channel count (4 for SC + active mask + CoordConv).
        num_workers (int): Number of worker threads for DataLoader.

    Returns:
        Tuple[DataLoader, Optional[DataLoader], torch.Tensor]: (train_loader, val_loader, class_weights_tensor).

    Raises:
        ValueError: If no valid samples could be loaded from the target directory.
    """
    files = sorted(glob.glob(os.path.join(target.dir_path, "*.npz")))
    if max_files and max_files > 0:
        files = files[:max_files]

    sc_list, inact_list, y_list = [], [], []

    for filepath in files:
        try:
            with np.load(filepath) as npz:
                if "X_supercharge" not in npz or "y" not in npz:
                    continue
                sc = npz["X_supercharge"]
                y = npz["y"]
                inact = (
                    npz["X_inactive_mask"]
                    if "X_inactive_mask" in npz
                    else np.zeros_like(sc)
                )

                # Validate expected dimensions before transpose
                if sc.shape[1] != target.file_height or sc.shape[2] != target.file_width:
                    continue

                if target.needs_transpose:
                    sc = np.swapaxes(sc, 1, 2)
                    inact = np.swapaxes(inact, 1, 2)
                    y = np.swapaxes(y, 1, 2)

                sc_list.append(sc)
                inact_list.append(inact)
                y_list.append(y)
        except Exception as e:
            logging.debug(f"Skipping corrupted file {filepath}: {e}")

    if not sc_list:
        raise ValueError(f"No valid data could be loaded for target {target.ship}/{target.tech}")

    X_sc_all = np.concatenate(sc_list, axis=0)
    X_inact_all = np.concatenate(inact_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    # Class weight calculation to balance background and rare modules
    all_labels = y_all.flatten()
    unique_present = np.unique(all_labels)

    weights_present = compute_class_weight("balanced", classes=unique_present, y=all_labels)
    full_weights = np.ones(target.num_classes, dtype=np.float32)
    for cls_val, w_val in zip(unique_present, weights_present):
        if cls_val < target.num_classes:
            full_weights[cls_val] = float(w_val)

    class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32)

    # Split train and validation
    if val_split > 0.0 and len(X_sc_all) >= 10:
        idx_train, idx_val = train_test_split(
            np.arange(len(X_sc_all)), test_size=val_split, random_state=42
        )
        train_sc, val_sc = X_sc_all[idx_train], X_sc_all[idx_val]
        train_inact, val_inact = X_inact_all[idx_train], X_inact_all[idx_val]
        train_y, val_y = y_all[idx_train], y_all[idx_val]
    else:
        train_sc, val_sc = X_sc_all, None
        train_inact, val_inact = X_inact_all, None
        train_y, val_y = y_all, None

    train_dataset = PlacementDataset(
        train_sc, train_y, X_inactive_mask=train_inact, in_channels=in_channels
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_sc is not None and len(val_sc) > 0:
        val_dataset = PlacementDataset(
            val_sc, val_y, X_inactive_mask=val_inact, in_channels=in_channels
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=min(batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, class_weights_tensor


def train_all(
    targets: List[DatasetTarget],
    model_dir: str,
    log_dir: str,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 2e-4,
    wd: float = 5e-5,
    es_patience: int = 12,
    es_metric: str = "val_miou",
    in_channels: int = 4,
    use_attention: bool = True,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    max_files: Optional[int] = None,
) -> Dict[str, dict]:
    """Sequentially trains models for all provided targets with summary reporting.

    Args:
        targets (List[DatasetTarget]): List of targets to train.
        model_dir (str): Base destination directory for trained models.
        log_dir (str): Directory for TensorBoard metrics.
        epochs (int): Maximum training epochs per model.
        batch_size (int): Training batch size.
        lr (float): Initial learning rate for AdamW.
        wd (float): Weight decay (L2 penalty).
        es_patience (int): Number of epochs without improvement before early stopping.
        es_metric (str): Metric monitored for early stopping ('val_miou' or 'val_loss').
        in_channels (int): Input channel count (4 for CoordConv + SC + active).
        use_attention (bool): Whether to enable spatial self-attention.
        use_focal_loss (bool): Whether to use Focal Loss instead of CrossEntropy.
        focal_gamma (float): Focusing parameter gamma for Focal Loss.
        max_files (Optional[int]): Maximum number of batch files to load per tech.

    Returns:
        Dict[str, dict]: Summary dictionary mapping target keys to results.
    """
    results: Dict[str, dict] = {}
    total_start = time.time()

    print("\n" + "=" * 80)
    print(f"BATCH MODEL TRAINING: {len(targets)} targets queued")
    print(f"Model destination: {os.path.abspath(model_dir)}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, patience={es_patience}")
    print("=" * 80 + "\n")

    for i, target in enumerate(targets, 1):
        target_key = f"{target.ship}/{target.tech}"
        print(f"\n[{i}/{len(targets)}] >>> Training {target_key} ({target.model_width}x{target.model_height}, {target.num_classes} classes)")
        print(f"    Data source: {target.dir_path} ({target.file_count} files)")
        print(f"    Target model: {target.model_filename}")
        if target.needs_transpose:
            print(f"    Orientation: Transposing from {target.file_width}x{target.file_height} to canonical {target.model_width}x{target.model_height}")

        target_start = time.time()
        try:
            train_loader, val_loader, class_weights = load_and_preprocess_target(
                target=target,
                batch_size=batch_size,
                val_split=0.2,
                max_files=max_files,
                in_channels=in_channels,
            )

            target_log_dir = os.path.join(log_dir, target.ship, target.tech)
            os.makedirs(model_dir, exist_ok=True)

            _ = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                grid_height=target.model_height,
                grid_width=target.model_width,
                num_output_classes=target.num_classes,
                num_epochs=epochs,
                learning_rate=lr,
                weight_decay=wd,
                log_dir=target_log_dir,
                model_save_path=target.model_path,
                criterion_weights=class_weights,
                early_stopping_patience=es_patience,
                early_stopping_metric=es_metric,
                in_channels=in_channels,
                use_attention=use_attention,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
            )

            elapsed = time.time() - target_start
            results[target_key] = {
                "status": "SUCCESS",
                "time_sec": elapsed,
                "model_path": target.model_path,
                "model_filename": target.model_filename,
            }
            print(f"[{i}/{len(targets)}] SUCCESS: {target_key} trained in {elapsed:.1f}s -> {target.model_filename}")

        except Exception as e:
            elapsed = time.time() - target_start
            logging.error(f"[{i}/{len(targets)}] FAILED: {target_key} encountered error: {e}", exc_info=True)
            results[target_key] = {
                "status": "FAILED",
                "time_sec": elapsed,
                "error": str(e),
            }

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"BATCH TRAINING FINISHED: Completed in {total_elapsed:.1f}s")
    print("=" * 80)
    print(f"{'Target':<32} | {'Status':<9} | {'Time (s)':<8} | {'Model File':<35}")
    print("-" * 88)
    for key, res in sorted(results.items()):
        status = res.get("status", "UNKNOWN")
        sec = f"{res.get('time_sec', 0.0):.1f}"
        fname = res.get("model_filename", res.get("error", ""))
        print(f"{key:<32} | {status:<9} | {sec:<8} | {fname:<35}")
    print("=" * 80 + "\n")

    return results


def main() -> None:
    """CLI entrypoint for batch model training."""
    parser = argparse.ArgumentParser(
        description="Batch train NMS Optimizer ML placement models from generated batches."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="scripts/training/generated_batches",
        help="Base directory containing generated batch folders.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="src/trained_models",
        help="Target directory to save trained model checkpoints.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs_placement_only",
        help="Base directory for TensorBoard logging.",
    )
    parser.add_argument(
        "--ship",
        type=str,
        default=None,
        help="Filter by ship/platform key (comma-separated, e.g. 'corvette' or 'corvette,standard').",
    )
    parser.add_argument(
        "--tech",
        type=str,
        default=None,
        help="Filter by technology key (comma-separated, e.g. 'launch,pulse').",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by technology category (comma-separated, e.g. 'Weaponry,Hyperdrive').",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip models that already exist in model_dir.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover targets and print plan without training.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of batch .npz files to load per tech.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Maximum epochs per target.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=5e-5, help="Weight decay.")
    parser.add_argument("--es_patience", type=int, default=12, help="Early stopping patience.")
    parser.add_argument(
        "--es_metric",
        type=str,
        default="val_miou",
        choices=["val_miou", "val_loss"],
        help="Early stopping metric.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
        help="Number of input channels (4 = SC + active mask + CoordConv X/Y).",
    )
    parser.add_argument(
        "--no_attention",
        action="store_true",
        help="Disable spatial self-attention in model architecture.",
    )
    parser.add_argument(
        "--no_focal",
        action="store_true",
        help="Disable Focal Loss (use CrossEntropy instead).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter gamma for Focal Loss.",
    )

    args = parser.parse_args()

    ships = [s.strip() for s in args.ship.split(",") if s.strip()] if args.ship else None
    techs = [t.strip() for t in args.tech.split(",") if t.strip()] if args.tech else None
    categories = [c.strip() for c in args.category.split(",") if c.strip()] if args.category else None

    # Discover all targets
    all_targets = discover_dataset_targets(args.data_dir, args.model_dir)
    selected_targets = filter_targets(
        targets=all_targets,
        ships=ships,
        techs=techs,
        categories=categories,
        skip_existing=args.skip_existing,
        model_dir=args.model_dir,
    )

    if not selected_targets:
        print("No matching dataset targets found with current filters.")
        sys.exit(0)

    if args.dry_run:
        print("\n" + "=" * 92)
        print(f"DRY RUN: {len(selected_targets)} targets matched in '{args.data_dir}'")
        print("=" * 92)
        print(f"{'Target':<28} | {'Files':<5} | {'Raw Dim':<7} | {'Model Dim':<9} | {'Trans':<5} | {'Cls':<4} | {'Model Filename':<32}")
        print("-" * 92)
        for t in selected_targets:
            t_key = f"{t.ship}/{t.tech}"
            raw_dim = f"{t.file_width}x{t.file_height}"
            mod_dim = f"{t.model_width}x{t.model_height}"
            trans = "YES" if t.needs_transpose else "NO"
            print(f"{t_key:<28} | {t.file_count:<5} | {raw_dim:<7} | {mod_dim:<9} | {trans:<5} | {t.num_classes:<4} | {t.model_filename:<32}")
        print("=" * 92 + "\n")
        return

    train_all(
        targets=selected_targets,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        es_patience=args.es_patience,
        es_metric=args.es_metric,
        in_channels=args.in_channels,
        use_attention=not args.no_attention,
        use_focal_loss=not args.no_focal,
        focal_gamma=args.focal_gamma,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
