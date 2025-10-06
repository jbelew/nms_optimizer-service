"""
This module provides a caching mechanism for loading PyTorch models.

It uses an in-memory LRU (Least Recently Used) cache to store pre-trained
models, avoiding the overhead of reading from disk every time a model is
needed. This is particularly useful in a server environment where the same
models may be requested frequently.
"""
import torch
import logging
import os
from functools import lru_cache
from .model_definition import ModulePlacementCNN

# A conservative cache size, suitable for memory-constrained environments.
# This can be overridden by an environment variable if needed.
CACHE_SIZE = int(os.environ.get("MODEL_CACHE_SIZE", 5))

@lru_cache(maxsize=CACHE_SIZE)
def _load_model_from_disk(model_path, model_grid_width, model_grid_height, num_output_classes):
    """
    Internal function to load a model from disk.
    This function is decorated with @lru_cache to cache its results.
    """
    logging.info(f"--- CACHE MISS --- Loading model from disk: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = ModulePlacementCNN(
        grid_height=model_grid_height,
        grid_width=model_grid_width,
        num_output_classes=num_output_classes,
    )

    # Always load to CPU first to ensure the cached object is stored in main memory.
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_model(model_path, model_grid_width, model_grid_height, num_output_classes):
    """
    Retrieves a model, utilizing an LRU cache to avoid reloading from disk.
    """
    absolute_model_path = os.path.abspath(model_path)

    try:
        # Call the cached loader function
        model = _load_model_from_disk(
            absolute_model_path,
            model_grid_width,
            model_grid_height,
            num_output_classes
        )

        # Determine the target device and move the model if necessary.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        logging.info(f"Model {os.path.basename(model_path)} ready on device: {device}")
        return model

    except Exception as e:
        logging.error(f"Failed to get or load model '{model_path}': {e}")
        raise
