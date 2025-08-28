# modules_for_training.py

"""
This file contains the specific lists of module IDs that were used to train
the machine learning models.

In most cases, the training list is the same as the full list of modules
available for a technology. However, in some cases, a model was trained
on a subset of modules (e.g., before a new module was added to the game).

To ensure the `ModulePlacementCNN` class is instantiated with the correct
number of output channels to match the saved model, the data loader for
the ML service should use the lists defined here.

If a technology is not present in this file, the loader should assume
the training set is the same as the main module definition.
"""

MODULES_FOR_TRAINING = {
    "standard": {
        "pulse": [
            "PE",  # Pulse Engine
            "FA",  # Flight Assist Override
            "SL",  # Sub-Light Amplifier
            "ID",  # Instability Drive
            "Xa",  # Pulse Engine Upgrade Theta
            "Xb",  # Pulse Engine Upgrade Tau
            "Xc",  # Pulse Engine Upgrade Sigma
        ],
        "photonix": [
            "PE",  # Pulse Engine
            "PC",  # Photonix Core
            "FA",  # Flight Assist Override
            "SL",  # Sub-Light Amplifier
            "ID",  # Instability Drive
            "Xa",  # Pulse Engine Upgrade Theta
            "Xb",  # Pulse Engine Upgrade Tau
            "Xc",  # Pulse Engine Upgrade Sigma
        ],
    }
}
