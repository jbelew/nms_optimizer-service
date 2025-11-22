import json
import logging
import pytest

from rust_scorer import simulated_annealing, Module, ModuleType, AdjacencyType


# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def progress_callback(progress_data):
    """Callback function passed to Rust simulated_annealing (not a pytest test)"""
    logging.info(f"Python Callback: {progress_data}")


def run_test():
    logging.info("Starting Rust callback test...")

    # Create a dummy grid
    grid_data = {
        "width": 10,
        "height": 6,
        "cells": [
            [
                {
                    "active": True,
                    "supercharged": False,
                    "module": None,
                    "tech": None,
                    "bonus": 0.0,
                    "adjacency": "no_adjacency",
                    "sc_eligible": False,
                    "value": 0,
                    "total": 0.0,
                    "adjacency_bonus": 0.0,
                    "label": None,
                    "module_type": "bonus",
                    "image": None,
                }
                for _ in range(10)
            ]
            for _ in range(6)
        ],
    }
    # Make one cell supercharged
    grid_data["cells"][2][2]["supercharged"] = True
    grid_data["cells"][2][2]["active"] = True

    # Create dummy tech modules
    tech_modules = [
        Module(
            id="M1",
            label="Module 1",
            tech="test",
            module_type=ModuleType.Bonus,
            bonus=1.0,
            adjacency=AdjacencyType.NoAdjacency,
            sc_eligible=True,
            image=None,
        ),
        Module(
            id="M2",
            label="Module 2",
            tech="test",
            module_type=ModuleType.Bonus,
            bonus=1.0,
            adjacency=AdjacencyType.NoAdjacency,
            sc_eligible=True,
            image=None,
        ),
        Module(
            id="M3",
            label="Module 3",
            tech="test",
            module_type=ModuleType.Bonus,
            bonus=1.0,
            adjacency=AdjacencyType.NoAdjacency,
            sc_eligible=True,
            image=None,
        ),
        Module(
            id="M4",
            label="Module 4",
            tech="test",
            module_type=ModuleType.Bonus,
            bonus=1.0,
            adjacency=AdjacencyType.NoAdjacency,
            sc_eligible=True,
            image=None,
        ),
        Module(
            id="M5",
            label="Module 5",
            tech="test",
            module_type=ModuleType.Bonus,
            bonus=1.0,
            adjacency=AdjacencyType.NoAdjacency,
            sc_eligible=True,
            image=None,
        ),
        Module(
            id="M6",
            label="Module 6",
            tech="test",
            module_type=ModuleType.Bonus,
            bonus=1.0,
            adjacency=AdjacencyType.NoAdjacency,
            sc_eligible=True,
            image=None,
        ),
        Module(
            id="M7",
            label="Module 7",
            tech="test",
            module_type=ModuleType.Bonus,
            bonus=1.0,
            adjacency=AdjacencyType.NoAdjacency,
            sc_eligible=True,
            image=None,
        ),
        Module(
            id="M8",
            label="Module 8",
            tech="test",
            module_type=ModuleType.Bonus,
            bonus=1.0,
            adjacency=AdjacencyType.NoAdjacency,
            sc_eligible=True,
            image=None,
        ),
        Module(
            id="M9",
            label="Module 9",
            tech="test",
            module_type=ModuleType.Bonus,
            bonus=1.0,
            adjacency=AdjacencyType.NoAdjacency,
            sc_eligible=True,
            image=None,
        ),
    ]

    tech = "test"
    initial_temperature = 100.0
    cooling_rate = 0.95
    stopping_temperature = 0.1
    iterations_per_temp = 10

    # Test with callback
    logging.info("Running simulated_annealing with progress_callback...")
    best_grid_json, best_score = simulated_annealing(
        json.dumps(grid_data),
        tech_modules,
        tech,
        initial_temperature,
        cooling_rate,
        stopping_temperature,
        iterations_per_temp,
        progress_callback,
    )
    logging.info(f"Simulated Annealing (with callback) finished. Best score: {best_score}")

    # Test without callback (passing None)
    logging.info("Running simulated_annealing without progress_callback (passing None)...")
    best_grid_json_none, best_score_none = simulated_annealing(
        json.dumps(grid_data),
        tech_modules,
        tech,
        initial_temperature,
        cooling_rate,
        stopping_temperature,
        iterations_per_temp,
        None,  # Pass None for progress_callback
    )
    logging.info(f"Simulated Annealing (without callback) finished. Best score: {best_score_none}")


if __name__ == "__main__":
    run_test()
