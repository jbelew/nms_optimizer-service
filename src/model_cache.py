import torch
import logging
import os
from training.model_definition import ModulePlacementCNN

# Global cache for storing loaded models.
# The key is the model path, and the value is the loaded model object.
_model_cache = {}


def get_model(
    model_path: str,
    model_grid_width: int,
    model_grid_height: int,
    num_output_classes: int,
) -> torch.nn.Module:
    """
    Retrieves a model from the cache or loads it from disk if not already cached.

    Args:
        model_path (str): The full path to the trained model file (.pth).
        model_grid_width (int): The grid width the model was trained on.
        model_grid_height (int): The grid height the model was trained on.
        num_output_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The loaded and initialized model, ready for evaluation.

    Raises:
        FileNotFoundError: If the model file does not exist at the given path.
        Exception: For other errors during model loading or initialization.
    """
    # Use the absolute path as the cache key to ensure uniqueness.
    absolute_model_path = os.path.abspath(model_path)

    if absolute_model_path in _model_cache:
        logging.info(f"INFO -- Retrieving model from cache: {absolute_model_path}")
        return _model_cache[absolute_model_path]

    logging.info(f"INFO -- Model not in cache. Loading model from: {absolute_model_path}")

    if not os.path.exists(absolute_model_path):
        logging.error(f"ERROR -- Model file not found at: {absolute_model_path}")
        raise FileNotFoundError(
            f"Model file not found at the specified path: {absolute_model_path}"
        )

    try:
        # 1. Instantiate the model architecture.
        model = ModulePlacementCNN(
            input_channels=2,
            grid_height=model_grid_height,
            grid_width=model_grid_width,
            num_output_classes=num_output_classes,
        )

        # 2. Determine the device and load the state dictionary.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(absolute_model_path, map_location=device)

        # 3. Load the state dictionary into the model.
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Set the model to evaluation mode.

        logging.info(
            f"INFO -- Model loaded and cached successfully. Device: {device}"
        )

        # 4. Store the loaded model in the cache.
        _model_cache[absolute_model_path] = model

        return model

    except Exception as e:
        logging.error(
            f"ERROR -- Failed to load model state_dict from {absolute_model_path}: {e}"
        )
        logging.error(
            "       Check if model architecture (grid size, channels, classes) matches the saved file."
        )
        # Re-raise the exception to be handled by the caller.
        raise e
