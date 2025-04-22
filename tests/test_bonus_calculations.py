import unittest
from unittest.mock import patch, MagicMock
from grid_utils import Grid
from bonus_calculations import (
    AdjacencyType,
    ModuleType,
    CORE_WEIGHT_GREATER,
    CORE_WEIGHT_LESSER,
    BONUS_BONUS_GREATER_WEIGHT,
    BONUS_BONUS_LESSER_WEIGHT,
    BONUS_BONUS_MIXED_WEIGHT,
    SUPERCHARGE_MULTIPLIER,
    _get_orthogonal_neighbors,
    populate_module_bonuses,
    populate_all_module_bonuses,
    clear_scores,
    calculate_grid_score
)


class TestBonusCalculations(unittest.TestCase):
    def setUp(self):
        """Set up common test resources."""
        # Create a test grid
        self.grid = Grid(3, 3)
        self.tech = "pulse"

    def test_basic_module_bonus_calculation(self):
        """Test basic bonus calculation for a single module without adjacency."""
        # Set up a module with a base bonus
        self.grid.set_module(1, 1, "TestModule")
        self.grid.set_tech(1, 1, self.tech)
        self.grid.set_type(1, 1, ModuleType.BONUS.value)
        self.grid.set_bonus(1, 1, 10.0)
        
        # Calculate bonus
        result = populate_module_bonuses(self.grid, 1, 1)
        
        # Assert the result equals the base bonus (no adjacency boost)
        self.assertEqual(result, 10.0)
        # Check that the total was set correctly
        self.assertEqual(self.grid.get_cell(1, 1)["total"], 10.0)
        # Check that adjacency_bonus is 0
        self.assertEqual(self.grid.get_cell(1, 1)["adjacency_bonus"], 0.0)

    def test_adjacency_boost_between_modules(self):
        """Test adjacency boost calculation between modules."""
        # Set up a core module
        self.grid.set_module(0, 1, "CoreModule")
        self.grid.set_tech(0, 1, self.tech)
        self.grid.set_type(0, 1, ModuleType.CORE.value)
        self.grid.set_adjacency(0, 1, AdjacencyType.GREATER.value)
        
        # Set up a bonus module adjacent to the core
        self.grid.set_module(1, 1, "BonusModule")
        self.grid.set_tech(1, 1, self.tech)
        self.grid.set_type(1, 1, ModuleType.BONUS.value)
        self.grid.set_bonus(1, 1, 10.0)
        self.grid.set_adjacency(1, 1, AdjacencyType.GREATER.value)
        
        # Calculate bonus for the bonus module
        result = populate_module_bonuses(self.grid, 1, 1)
        
        # Expected: base_bonus + (base_bonus * CORE_WEIGHT_GREATER)
        expected = 10.0 + (10.0 * CORE_WEIGHT_GREATER)
        
        # Assert the result matches expected
        self.assertAlmostEqual(result, expected, places=6)
        # Check that the total was set correctly
        self.assertAlmostEqual(self.grid.get_cell(1, 1)["total"], expected, places=6)
        # Check that adjacency_bonus is calculated correctly
        self.assertAlmostEqual(self.grid.get_cell(1, 1)["adjacency_bonus"], 10.0 * CORE_WEIGHT_GREATER, places=6)

    def test_supercharge_multiplier_application(self):
        """Test application of supercharge multiplier to a module."""
        # Set up a module with a base bonus and supercharge
        self.grid.set_module(1, 1, "TestModule")
        self.grid.set_tech(1, 1, self.tech)
        self.grid.set_type(1, 1, ModuleType.BONUS.value)
        self.grid.set_bonus(1, 1, 10.0)
        self.grid.set_supercharged(1, 1, True)
        self.grid.set_sc_eligible(1, 1, True)
        
        # Calculate bonus with supercharge applied after adjacency
        result_after = populate_module_bonuses(self.grid, 1, 1, apply_supercharge_first=False)
        
        # Expected: base_bonus * SUPERCHARGE_MULTIPLIER (no adjacency)
        expected_after = 10.0 * SUPERCHARGE_MULTIPLIER
        
        # Assert the result matches expected
        self.assertAlmostEqual(result_after, expected_after, places=6)
        
        # Reset and test with supercharge applied before adjacency
        self.grid.set_total(1, 1, 0.0)
        self.grid.set_adjacency_bonus(1, 1, 0.0)
        
        # Calculate bonus with supercharge applied before adjacency
        result_before = populate_module_bonuses(self.grid, 1, 1, apply_supercharge_first=True)
        
        # Expected: base_bonus * SUPERCHARGE_MULTIPLIER (no adjacency)
        expected_before = 10.0 * SUPERCHARGE_MULTIPLIER
        
        # Assert the result matches expected
        self.assertAlmostEqual(result_before, expected_before, places=6)

    def test_grid_score_calculation(self):
        """Test calculation of total grid score."""
        # Set up multiple modules with bonuses
        # Module 1
        self.grid.set_module(0, 0, "Module1")
        self.grid.set_tech(0, 0, self.tech)
        self.grid.set_type(0, 0, ModuleType.BONUS.value)
        self.grid.set_bonus(0, 0, 5.0)
        
        # Module 2
        self.grid.set_module(1, 1, "Module2")
        self.grid.set_tech(1, 1, self.tech)
        self.grid.set_type(1, 1, ModuleType.BONUS.value)
        self.grid.set_bonus(1, 1, 10.0)
        
        # Module 3 (different tech, should be ignored)
        self.grid.set_module(2, 2, "Module3")
        self.grid.set_tech(2, 2, "different_tech")
        self.grid.set_type(2, 2, ModuleType.BONUS.value)
        self.grid.set_bonus(2, 2, 15.0)
        
        # Calculate total grid score
        result = calculate_grid_score(self.grid, self.tech)
        
        # Expected: sum of Module1 and Module2 bonuses (no adjacency)
        expected = 15.0
        
        # Assert the result matches expected
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()