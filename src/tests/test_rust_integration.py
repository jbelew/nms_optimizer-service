import unittest
from src.grid_utils import Grid as PythonGrid
from src.bonus_calculations import calculate_grid_score


class TestRustIntegration(unittest.TestCase):
    def test_rust_scoring(self):
        # Create a Python Grid object to hold the initial data
        python_grid = PythonGrid(3, 3)
        pulse_module_data = {
            "id": "PULSE",
            "label": "Pulse Engine",
            "tech": "pulse",
            "type": "bonus",
            "bonus": 10.0,
            "adjacency": "greater",
            "sc_eligible": True,
            "image": None,
        }
        python_grid.get_cell(1, 1).update(pulse_module_data)
        python_grid.get_cell(1, 1)["module"] = pulse_module_data["id"]

        # Call the scoring function with the Python grid
        # The function internally converts to Rust grid
        score = calculate_grid_score(python_grid, "pulse", False)

        # Verify the results
        self.assertAlmostEqual(score, 10.0)
