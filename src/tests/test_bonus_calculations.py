import unittest
from src.grid_utils import Grid
from src.bonus_calculations import calculate_grid_score, AdjacencyType, ModuleType


class TestBonusCalculations(unittest.TestCase):
    """
    Test suite for the bonus calculation logic in bonus_calculations.py.
    """

    def setUp(self):
        """
        Set up a common grid and module data for the tests.
        """
        self.grid = Grid(3, 3)
        # Basic pulse module for testing
        self.pulse_module = {
            "id": "PULSE",
            "label": "Pulse Engine",
            "tech": "pulse",
            "type": ModuleType.BONUS.value,
            "bonus": 10.0,
            "adjacency": AdjacencyType.GREATER.value,
            "sc_eligible": True,
            "image": None,
        }

    def test_simple_bonus_calculation(self):
        """
        Test the bonus calculation for a single module with no adjacencies.
        """
        # Place a single module
        cell = self.grid.get_cell(1, 1)
        cell.update(self.pulse_module)
        cell["module"] = self.pulse_module["id"]  # Set the module ID to indicate placement

        # Calculate the score
        score = calculate_grid_score(self.grid, "pulse")

        # The score should be just the base bonus of the module
        self.assertEqual(score, 10.0)

    def test_supercharged_bonus(self):
        """
        Test that a supercharged module receives the correct bonus multiplier.
        """
        # Place a single supercharged module
        cell = self.grid.get_cell(1, 1)
        cell.update(self.pulse_module)
        cell["module"] = self.pulse_module["id"]  # Set the module ID to indicate placement
        cell["supercharged"] = True

        # Calculate the score
        score = calculate_grid_score(self.grid, "pulse")

        # The score should be the base bonus * supercharge multiplier
        self.assertAlmostEqual(score, 10.0 * 1.25)


if __name__ == "__main__":
    unittest.main()
