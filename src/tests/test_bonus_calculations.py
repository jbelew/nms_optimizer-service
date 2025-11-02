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
        score = calculate_grid_score(self.grid, "pulse", apply_supercharge_first=False)

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
        score = calculate_grid_score(self.grid, "pulse", apply_supercharge_first=False)

        # The score should be the base bonus * supercharge multiplier
        self.assertAlmostEqual(score, 10.0 * 1.25)

    def test_adjacent_greater_modules(self):
        """
        Test the bonus calculation for two adjacent 'greater' modules.
        """
        # Place two adjacent 'greater' modules
        cell1 = self.grid.get_cell(1, 1)
        cell1.update(self.pulse_module)
        cell1["module"] = self.pulse_module["id"]

        cell2 = self.grid.get_cell(1, 2)
        cell2.update(self.pulse_module)
        cell2["module"] = self.pulse_module["id"]

        # Calculate the score
        score = calculate_grid_score(self.grid, "pulse", apply_supercharge_first=False)

        # Each module gets a 6% bonus from its neighbor
        expected_score = (10.0 + 10.0 * 0.06) + (10.0 + 10.0 * 0.06)
        self.assertAlmostEqual(score, expected_score)

    def test_adjacent_lesser_modules(self):
        """
        Test the bonus calculation for two adjacent 'lesser' modules.
        """
        lesser_module = self.pulse_module.copy()
        lesser_module["adjacency"] = AdjacencyType.LESSER.value

        # Place two adjacent 'lesser' modules
        cell1 = self.grid.get_cell(1, 1)
        cell1.update(lesser_module)
        cell1["module"] = lesser_module["id"]

        cell2 = self.grid.get_cell(1, 2)
        cell2.update(lesser_module)
        cell2["module"] = lesser_module["id"]

        # Calculate the score
        score = calculate_grid_score(self.grid, "pulse", apply_supercharge_first=False)

        # Each module gets a 3% bonus from its neighbor
        expected_score = (10.0 + 10.0 * 0.03) + (10.0 + 10.0 * 0.03)
        self.assertAlmostEqual(score, expected_score)

    def test_greater_adjacent_to_lesser(self):
        """
        Test that a 'greater' module adjacent to a 'lesser' module penalizes the 'lesser' module.
        """
        lesser_module = self.pulse_module.copy()
        lesser_module["adjacency"] = AdjacencyType.LESSER.value

        # Place a 'greater' and a 'lesser' module adjacent to each other
        cell1 = self.grid.get_cell(1, 1)
        cell1.update(self.pulse_module)
        cell1["module"] = self.pulse_module["id"]

        cell2 = self.grid.get_cell(1, 2)
        cell2.update(lesser_module)
        cell2["module"] = lesser_module["id"]

        # Calculate the score
        score = calculate_grid_score(self.grid, "pulse", apply_supercharge_first=False)

        # The 'greater' module gets a 3% bonus from the 'lesser' module (10.0 * 0.03)
        greater_bonus = 10.0 + 0.3
        # The 'lesser' module gets a penalty factor from the 'greater' module (10.0 * -0.01)
        lesser_bonus = 10.0 - 0.1
        expected_score = greater_bonus + lesser_bonus
        self.assertAlmostEqual(score, expected_score)

    def test_core_module_bonus(self):
        """
        Test the bonus calculation for a 'core' module adjacent to other modules.
        """
        core_module = {
            "id": "CORE",
            "label": "Core Module",
            "tech": "pulse",
            "type": ModuleType.CORE.value,
            "bonus": 0.0,
            "adjacency": AdjacencyType.GREATER.value,
            "sc_eligible": False,
            "image": None,
        }

        # Place a core module and two 'greater' modules
        cell1 = self.grid.get_cell(1, 1)
        cell1.update(core_module)
        cell1["module"] = core_module["id"]

        cell2 = self.grid.get_cell(1, 2)
        cell2.update(self.pulse_module)
        cell2["module"] = self.pulse_module["id"]

        cell3 = self.grid.get_cell(0, 1)
        cell3.update(self.pulse_module)
        cell3["module"] = self.pulse_module["id"]

        # Calculate the score
        score = calculate_grid_score(self.grid, "pulse", apply_supercharge_first=False)

        # Core module gets 0.06 bonus from each BONUS neighbor
        core_bonus = 0.06 + 0.06
        # Other modules get 0.07 bonus from the CORE neighbor
        module2_bonus = 10.0 + 10.0 * 0.07
        module3_bonus = 10.0 + 10.0 * 0.07
        expected_score = core_bonus + module2_bonus + module3_bonus
        self.assertAlmostEqual(score, expected_score)

    def test_surrounded_module(self):
        """
        Test a module surrounded by four other modules.
        """
        # Center module
        cell_center = self.grid.get_cell(1, 1)
        cell_center.update(self.pulse_module)
        cell_center["module"] = self.pulse_module["id"]

        # Surrounding modules
        for x, y in [(0, 1), (2, 1), (1, 0), (1, 2)]:
            cell = self.grid.get_cell(x, y)
            cell.update(self.pulse_module)
            cell["module"] = self.pulse_module["id"]

        # Calculate the score
        score = calculate_grid_score(self.grid, "pulse", apply_supercharge_first=False)

        # Center module gets 4x bonus
        center_bonus = 10.0 + 4 * (10.0 * 0.06)
        # Each surrounding module gets 1x bonus
        surrounding_bonus = 4 * (10.0 + 1 * (10.0 * 0.06))
        expected_score = center_bonus + surrounding_bonus
        self.assertAlmostEqual(score, expected_score)

    def test_apply_supercharge_first(self):
        """
        Test the 'apply_supercharge_first' parameter.
        """
        # Place two adjacent, supercharged modules
        cell1 = self.grid.get_cell(1, 1)
        cell1.update(self.pulse_module)
        cell1["module"] = self.pulse_module["id"]
        cell1["supercharged"] = True

        cell2 = self.grid.get_cell(1, 2)
        cell2.update(self.pulse_module)
        cell2["module"] = self.pulse_module["id"]
        cell2["supercharged"] = True

        # Calculate score with apply_supercharge_first=True
        score = calculate_grid_score(self.grid, "pulse", apply_supercharge_first=True)

        # Base bonus is supercharged first
        supercharged_base = 10.0 * 1.25
        # Adjacency bonus is calculated on the supercharged base
        adj_bonus = supercharged_base * 0.06
        # Total for one module
        total_per_module = 10.0 + adj_bonus
        expected_score = total_per_module * 2
        self.assertAlmostEqual(score, expected_score)


if __name__ == "__main__":
    unittest.main()
