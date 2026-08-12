"""
Tests for adjacency-based single and multi-module placement without solve maps.
Focuses on the core placement logic using calculate_pattern_adjacency_score.
"""

import unittest
from src.grid_utils import Grid
from src.module_placement import place_module, clear_all_modules_of_tech
from src.pattern_matching import calculate_pattern_adjacency_score
from src.bonus_calculations import calculate_grid_score


class TestAdjacencyBasedPlacement(unittest.TestCase):
    """Tests for adjacency-based module placement logic"""

    def setUp(self):
        """Set up test grid"""
        self.grid = Grid(5, 5)
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                self.grid.cells[y][x]["active"] = True
        self.tech = "test_tech"

    def test_center_location_scores_zero_no_adjacency(self):
        """Center location with no neighbors should score 0 adjacency"""
        grid = self.grid.copy()

        # Find a center location away from edges and test its score
        test_grid = grid.copy()
        place_module(test_grid, 2, 2, "m1", "M1", self.tech, "bonus", 0.5, "no_adjacency", False, None)

        score = calculate_pattern_adjacency_score(test_grid, self.tech)
        self.assertEqual(score, 0.0, "Single module in center with no neighbors should score 0")

    def test_module_at_grid_edge_gets_bonus(self):
        """Module at grid edge should get grid edge bonus"""
        # Top-left corner (0, 0) should get 2 edge bonuses (left + top)
        test_grid = self.grid.copy()
        place_module(test_grid, 0, 0, "m1", "M1", self.tech, "bonus", 0.5, "no_adjacency", False, None)

        score_corner = calculate_pattern_adjacency_score(test_grid, self.tech)

        # Center location (2, 2) should get no edge bonus
        test_grid2 = self.grid.copy()
        place_module(test_grid2, 2, 2, "m1", "M1", self.tech, "bonus", 0.5, "no_adjacency", False, None)

        score_center = calculate_pattern_adjacency_score(test_grid2, self.tech)

        self.assertGreater(score_corner, score_center, "Corner placement should score higher due to grid edge bonus")

    def test_adjacency_to_other_modules_scores_higher(self):
        """Module adjacent to other modules should score higher"""
        # Create two grids - one with adjacency, one without
        # Grid 1: isolated module
        grid1 = self.grid.copy()
        place_module(grid1, 0, 0, "m1", "M1", self.tech, "bonus", 0.5, "no_adjacency", False, None)
        score1 = calculate_pattern_adjacency_score(grid1, self.tech)

        # Grid 2: module with neighbor (different tech)
        grid2 = self.grid.copy()
        place_module(grid2, 2, 2, "other1", "O1", "other_tech", "bonus", 0.5, "no_adjacency", False, None)
        place_module(grid2, 2, 3, "m1", "M1", self.tech, "bonus", 0.5, "no_adjacency", False, None)
        score2 = calculate_pattern_adjacency_score(grid2, self.tech)

        # Module adjacent to other tech should score higher (module_edge_weight = 3.0)
        self.assertGreater(score2, score1, "Module adjacent to other tech should score higher than isolated")

    def test_multiple_modules_combined_score(self):
        """Multiple modules should sum their individual adjacency scores"""
        grid = self.grid.copy()
        # Place 2 modules at grid edges
        place_module(grid, 0, 0, "m1", "M1", self.tech, "bonus", 0.5, "no_adjacency", False, None)
        place_module(grid, 4, 4, "m2", "M2", self.tech, "bonus", 0.5, "no_adjacency", False, None)

        score = calculate_pattern_adjacency_score(grid, self.tech)

        # Each corner gets 2 edge bonuses (0.5 weight), so total = 4 * 0.5 = 2.0
        self.assertEqual(score, 2.0, "Two corner modules should score 2.0 total")

    def test_adjacency_rules_group_bonus(self):
        """Modules with matching adjacency rules should get group bonus when adjacent to other tech"""
        grid = self.grid.copy()

        # Place a module with "greater_2" rule
        place_module(grid, 2, 2, "m1", "M1", self.tech, "bonus", 0.5, "greater_2", False, None)
        # Place another different-tech module with same rule adjacent to it
        place_module(grid, 2, 3, "m2", "M2", "other_tech", "bonus", 0.5, "greater_2", False, None)

        score = calculate_pattern_adjacency_score(grid, self.tech)

        # m1 should get module_edge_weight (3.0) for being adjacent to m2
        self.assertGreater(score, 0.0, "Module adjacent to different tech should score")

    def test_greater_n_adjacency_scaling(self):
        """Test that greater_3 module score scales correctly with number of matching neighbors"""
        # Test E = 1
        grid = self.grid.copy()
        place_module(grid, 1, 1, "m1", "M1", self.tech, "bonus", 0.5, "greater_3", False, None)
        place_module(grid, 1, 2, "n1", "N1", "other_tech", "bonus", 0.5, "greater_3", False, None)
        self.assertAlmostEqual(calculate_pattern_adjacency_score(grid, self.tech), 3.0 * 1 + 2.5)

        # Test E = 2
        place_module(grid, 1, 0, "n2", "N2", "other_tech", "bonus", 0.5, "greater_3", False, None)
        self.assertAlmostEqual(calculate_pattern_adjacency_score(grid, self.tech), 3.0 * 2 + 5.0)

        # Test E = 3
        place_module(grid, 0, 1, "n3", "N3", "other_tech", "bonus", 0.5, "greater_3", False, None)
        self.assertAlmostEqual(calculate_pattern_adjacency_score(grid, self.tech), 3.0 * 3 + 15.0)

        # Test E = 4
        place_module(grid, 2, 1, "n4", "N4", "other_tech", "bonus", 0.5, "greater_3", False, None)
        self.assertAlmostEqual(calculate_pattern_adjacency_score(grid, self.tech), 3.0 * 4 + 20.0)

    def test_lesser_n_adjacency_scaling(self):
        """Test that lesser_2 module score scales correctly with number of matching neighbors, penalizing above 2"""
        # Test E = 1
        grid = self.grid.copy()
        place_module(grid, 1, 1, "m1", "M1", self.tech, "bonus", 0.5, "lesser_2", False, None)
        place_module(grid, 1, 2, "n1", "N1", "other_tech", "bonus", 0.5, "lesser_2", False, None)
        self.assertAlmostEqual(calculate_pattern_adjacency_score(grid, self.tech), 3.0 * 1 + 5.0)

        # Test E = 2
        place_module(grid, 1, 0, "n2", "N2", "other_tech", "bonus", 0.5, "lesser_2", False, None)
        self.assertAlmostEqual(calculate_pattern_adjacency_score(grid, self.tech), 3.0 * 2 + 10.0)

        # Test E = 3
        place_module(grid, 0, 1, "n3", "N3", "other_tech", "bonus", 0.5, "lesser_2", False, None)
        self.assertAlmostEqual(calculate_pattern_adjacency_score(grid, self.tech), 3.0 * 3 + 5.0)

        # Test E = 4
        place_module(grid, 2, 1, "n4", "N4", "other_tech", "bonus", 0.5, "lesser_2", False, None)
        self.assertAlmostEqual(calculate_pattern_adjacency_score(grid, self.tech), 3.0 * 4 + 0.0)

    def test_cross_rule_family_adjacency(self):
        """Test that different rule values in the same family (e.g. greater_3 and greater_2) do NOT match (they are different groups)"""
        grid = self.grid.copy()
        # Place module with greater_3
        place_module(grid, 1, 1, "m1", "M1", self.tech, "bonus", 0.5, "greater_3", False, None)
        # Place different-tech neighbor with greater_2
        place_module(grid, 1, 2, "n1", "N1", "other_tech", "bonus", 0.5, "greater_2", False, None)

        # They should NOT match, so group bonus is 0. Only standard module edge weight applies (3.0 * 1 = 3.0).
        self.assertAlmostEqual(calculate_pattern_adjacency_score(grid, self.tech), 3.0 * 1)

    def test_grid_score_float_precision_rounding(self):
        """Test that calculate_grid_score rounds the score to 4 decimal places to prevent float summation inequalities from impacting decisions."""
        grid_a = self.grid.copy()
        grid_b = self.grid.copy()

        # Place a 3-module horizontal block of identical modules on grid_a
        place_module(grid_a, 1, 1, "m1", "M1", self.tech, "upgrade", 0.3, "greater_2", False, None)
        place_module(grid_a, 2, 1, "m2", "M2", self.tech, "bonus", 1.0, "greater_2", False, None)
        place_module(grid_a, 3, 1, "m3", "M3", self.tech, "upgrade", 0.29, "greater_2", False, None)

        # Place the same block at a different position on grid_b to shift the traversal/addition order
        place_module(grid_b, 3, 1, "m3", "M3", self.tech, "upgrade", 0.29, "greater_2", False, None)
        place_module(grid_b, 2, 1, "m2", "M2", self.tech, "bonus", 1.0, "greater_2", False, None)
        place_module(grid_b, 1, 1, "m1", "M1", self.tech, "upgrade", 0.3, "greater_2", False, None)

        score_a = calculate_grid_score(grid_a, self.tech)
        score_b = calculate_grid_score(grid_b, self.tech)

        # Ensure both grids score exactly the same up to float precision (due to rounding)
        self.assertEqual(score_a, score_b)

    def test_no_modules_placed_scores_zero(self):
        """Grid with no modules of tech should score 0"""
        grid = self.grid.copy()

        score = calculate_pattern_adjacency_score(grid, self.tech)
        self.assertEqual(score, 0.0, "Empty grid should score 0")

    def test_wrong_tech_ignored(self):
        """Modules of different tech should be ignored in scoring"""
        grid = self.grid.copy()

        # Place a module of different tech
        place_module(grid, 2, 2, "m1", "M1", "other_tech", "bonus", 0.5, "no_adjacency", False, None)

        score = calculate_pattern_adjacency_score(grid, self.tech)
        self.assertEqual(score, 0.0, "Modules of different tech should not contribute")


class TestPlacementAlgorithmLogic(unittest.TestCase):
    """Tests for the logic of the placement algorithm itself"""

    def setUp(self):
        """Set up test grid"""
        self.grid = Grid(5, 5)
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                self.grid.cells[y][x]["active"] = True
        self.tech = "test_tech"

    def test_best_position_selection_loop(self):
        """Simulating the placement algorithm to verify best position selection"""
        grid = self.grid.copy()

        # Pre-place a module to create adjacency opportunities
        place_module(grid, 2, 2, "seed", "SEED", "seed_tech", "bonus", 0.5, "no_adjacency", False, None)

        best_score = -float("inf")
        best_pos = None

        # Simulate scanning all positions like the algorithm does
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                if cell["active"] and cell["module"] is None:
                    # Try placing module here
                    test_grid = grid.copy()
                    place_module(test_grid, x, y, "m1", "M1", self.tech, "bonus", 0.5, "no_adjacency", False, None)

                    score = calculate_pattern_adjacency_score(test_grid, self.tech)

                    if score > best_score:
                        best_score = score
                        best_pos = (x, y)

        # Algorithm should find a position
        self.assertIsNotNone(best_pos, "Should find a valid position")

        # Should prefer position adjacent to the seed module
        if best_pos is not None:
            x, y = best_pos
            seed_x, seed_y = 2, 2
            is_adjacent = abs(x - seed_x) + abs(y - seed_y) == 1
            self.assertTrue(is_adjacent, f"Position {best_pos} should be adjacent to seed at (2,2)")

    def test_sequential_placement_considers_previous(self):
        """Multi-module placement should consider already-placed modules"""
        grid = self.grid.copy()

        # Place first module of this tech at corner (gets edge bonus)
        place_module(grid, 0, 0, "m1", "M1", self.tech, "bonus", 0.5, "no_adjacency", False, None)
        score1 = calculate_pattern_adjacency_score(grid, self.tech)

        # Place second module of different tech adjacent to first
        place_module(grid, 0, 1, "m2", "M2", "other_tech", "bonus", 0.5, "no_adjacency", False, None)
        score2 = calculate_pattern_adjacency_score(grid, self.tech)

        # Second placement should increase score (m1 gets adjacency bonus for m2)
        self.assertGreater(score2, score1, "Adding adjacent different-tech module should increase adjacency score")

    def test_clear_modules_before_rescoring(self):
        """Clearing modules of a tech should reset scoring"""
        grid = self.grid.copy()

        # Place module at corner (gets edge bonus)
        place_module(grid, 0, 0, "m1", "M1", self.tech, "bonus", 0.5, "no_adjacency", False, None)
        score_before = calculate_pattern_adjacency_score(grid, self.tech)

        # Clear it
        clear_all_modules_of_tech(grid, self.tech)
        score_after = calculate_pattern_adjacency_score(grid, self.tech)

        self.assertGreater(score_before, 0.0, "Should have score with module at corner")
        self.assertEqual(score_after, 0.0, "Should have no score after clearing")

    def test_inactive_cells_not_selected(self):
        """Algorithm should skip inactive cells"""
        grid = self.grid.copy()
        grid.cells[0][0]["active"] = False
        grid.cells[0][1]["active"] = False

        best_score = -float("inf")
        best_pos = None

        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                if cell["active"] and cell["module"] is None:
                    test_grid = grid.copy()
                    place_module(test_grid, x, y, "m1", "M1", self.tech, "bonus", 0.5, "no_adjacency", False, None)
                    score = calculate_pattern_adjacency_score(test_grid, self.tech)

                    if score > best_score:
                        best_score = score
                        best_pos = (x, y)

        # Should never pick (0, 0) or (0, 1) since they're inactive
        if best_pos:
            self.assertNotEqual(best_pos, (0, 0))
            self.assertNotEqual(best_pos, (0, 1))


if __name__ == "__main__":
    unittest.main()
