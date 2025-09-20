import os
import sys
import argparse
import re

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.training.train_model import run_training_from_files

def retrain_all_models(
    base_log_dir,
    base_model_save_dir,
    base_data_source_dir,
    learning_rate,
    weight_decay,
    num_epochs,
    batch_size,
    validation_split,
    early_stopping_patience,
    early_stopping_metric,
):
    """
    Retrains all models found in the specified models directory.
    """
    model_files = [f for f in os.listdir(base_model_save_dir) if f.endswith('.pth')]

    # Regex to parse model filenames. Examples:
    # model_standard_pulse.pth -> ship='standard', tech='pulse', solve_type=None
    # model_corvette_cyclotron_max.pth -> ship='corvette', tech='cyclotron', solve_type='max'
    # model_standard-mt_blaze-javelin.pth -> ship='standard-mt', tech='blaze-javelin', solve_type=None
    # model_solar_pulse_4x3.pth -> ship='solar', tech='pulse', solve_type=None (ignores _4x3)
    pattern = re.compile(r"model_([a-z0-9-]+)_([a-z0-9-]+)(?:_([a-z0-9]+))?(?:_\dx\d)?\.pth")

    for filename in model_files:
        match = pattern.match(filename)
        if not match:
            print(f"Warning: Could not parse filename '{filename}'. Skipping.")
            continue

        ship, tech, solve_type = match.groups()

        print(f"--- Retraining model from file: {filename} ---")
        print(f"  - Ship: {ship}")
        print(f"  - Tech: {tech}")
        print(f"  - Solve Type: {solve_type or 'default'}")

        run_training_from_files(
            ship=ship,
            tech_category_to_train=None,
            specific_techs_to_train=[tech],
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            batch_size=batch_size,
            base_log_dir=base_log_dir,
            base_model_save_dir=base_model_save_dir,
            base_data_source_dir=base_data_source_dir,
            solve_type=solve_type,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain all NMS Optimizer models.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--wd", type=float, default=5e-5, help="Weight decay (L2 regularization).")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--log_dir", type=str, default="scripts/training/runs_placement_only", help="Base directory for TensorBoard logs.")
    parser.add_argument("--model_dir", type=str, default="src/trained_models", help="Base directory to save best trained models.")
    parser.add_argument("--data_source_dir", type=str, default="scripts/training/generated_batches", help="Base directory containing generated .npz data files.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation.")
    parser.add_argument("--es_patience", type=int, default=16, help="Early Stopping: Number of epochs to wait for improvement.")
    parser.add_argument("--es_metric", type=str, default="val_miou", choices=["val_loss", "val_miou"], help="Early Stopping: Metric to monitor.")

    args = parser.parse_args()

    retrain_all_models(
        base_log_dir=args.log_dir,
        base_model_save_dir=args.model_dir,
        base_data_source_dir=args.data_source_dir,
        learning_rate=args.lr,
        weight_decay=args.wd,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        early_stopping_patience=args.es_patience,
        early_stopping_metric=args.es_metric,
    )
