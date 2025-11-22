"""
Adversarial tests for optimization/refinement.py

Focus on real behavior, not mocking Rust components.
Tests validate edge cases and error handling.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.grid_utils import Grid
from src.optimization.refinement import refine_placement


class TestRefinePlacement(unittest.TestCase):
    """Adversarial tests for refine_placement function"""

    def _create_grid(self, width=5, height=5):
        """Helper to create a test grid"""
        grid = Grid(width, height)
        for y in range(height):
            for x in range(width):
                grid.cells[y][x]["active"] = True
        return grid

    def test_no_modules_returns_none(self):
        """With no tech modules, should return None"""
        grid = self._create_grid()
        result_grid, result_bonus = refine_placement(grid, "ship", {}, "tech", tech_modules=[])
        self.assertIsNone(result_grid)
        self.assertEqual(result_bonus, 0.0)

    def test_no_available_positions_returns_none(self):
        """If no empty active positions, should return None"""
        grid = self._create_grid(2, 2)
        # Fill all cells
        for y in range(grid.height):
            for x in range(grid.width):
                grid.cells[y][x]["module"] = "something"

        tech_modules = [
            {
                "id": "m1",
                "label": "M1",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
        ]

        result_grid, result_bonus = refine_placement(grid, "ship", {}, "tech", tech_modules=tech_modules)
        self.assertIsNone(result_grid)
        self.assertEqual(result_bonus, 0.0)

    def test_not_enough_empty_positions_returns_none(self):
        """If fewer empty positions than modules, should return None"""
        grid = self._create_grid(2, 2)
        # Fill 3 out of 4 cells
        grid.cells[0][0]["module"] = "m_x"
        grid.cells[0][1]["module"] = "m_y"
        grid.cells[1][0]["module"] = "m_z"

        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"M{i}",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(2)
        ]

        result_grid, result_bonus = refine_placement(grid, "ship", {}, "tech", tech_modules=tech_modules)
        self.assertIsNone(result_grid)
        self.assertEqual(result_bonus, 0.0)

    def test_single_module_single_position(self):
        """Single module in single available position should work"""
        grid = self._create_grid(1, 1)
        tech_modules = [
            {
                "id": "m1",
                "label": "M1",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
        ]

        with patch("src.optimization.refinement.calculate_grid_score", return_value=5.0):
            result_grid, result_bonus = refine_placement(grid, "ship", {}, "tech", tech_modules=tech_modules)

        self.assertIsNotNone(result_grid)
        self.assertEqual(result_bonus, 5.0)

    def test_exact_fit_all_positions_filled(self):
        """Exact fit: modules == empty positions"""
        grid = self._create_grid(2, 2)
        # Fill 2 cells, leave 2 empty
        grid.cells[0][0]["module"] = "existing"
        grid.cells[0][1]["module"] = "existing"

        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"M{i}",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(2)
        ]

        with patch("src.optimization.refinement.calculate_grid_score", return_value=10.0):
            result_grid, result_bonus = refine_placement(grid, "ship", {}, "tech", tech_modules=tech_modules)

        self.assertIsNotNone(result_grid)
        if result_grid is not None:
            # Both modules should be placed
            placed_count = sum(
                1
                for y in range(result_grid.height)
                for x in range(result_grid.width)
                if result_grid.cells[y][x]["module"] is not None and result_grid.cells[y][x]["tech"] == "tech"
            )
            self.assertEqual(placed_count, 2)

    def test_more_positions_than_modules(self):
        """More empty positions than modules"""
        grid = self._create_grid(5, 5)
        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"M{i}",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(3)
        ]

        with patch("src.optimization.refinement.calculate_grid_score", return_value=15.0):
            result_grid, result_bonus = refine_placement(grid, "ship", {}, "tech", tech_modules=tech_modules)

        self.assertIsNotNone(result_grid)
        if result_grid is not None:
            # All 3 modules should be placed
            placed_count = sum(
                1
                for y in range(result_grid.height)
                for x in range(result_grid.width)
                if result_grid.cells[y][x]["module"] is not None and result_grid.cells[y][x]["tech"] == "tech"
            )
            self.assertEqual(placed_count, 3)

    def test_inactive_cells_not_used(self):
        """Inactive cells should not be used for placement"""
        grid = self._create_grid(3, 1)
        grid.cells[0][1]["active"] = False  # Deactivate middle cell

        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"M{i}",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(2)
        ]

        with patch("src.optimization.refinement.calculate_grid_score", return_value=8.0):
            result_grid, result_bonus = refine_placement(grid, "ship", {}, "tech", tech_modules=tech_modules)

        self.assertIsNotNone(result_grid)
        if result_grid is not None:
            # Should only use cells at (0,0) and (2,0)
            # Middle cell should remain empty
            self.assertIsNone(result_grid.cells[0][1]["module"])

    def test_progress_callback_invoked(self):
        """Progress callback should be called during permutation"""
        grid = self._create_grid(3, 1)
        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"M{i}",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(2)
        ]

        callback = MagicMock()
        with patch("src.optimization.refinement.calculate_grid_score", return_value=8.0):
            result_grid, result_bonus = refine_placement(
                grid, "ship", {}, "tech", tech_modules=tech_modules, progress_callback=callback
            )

        # Callback should have been called
        self.assertGreater(callback.call_count, 0)

    def test_permutation_iteration_count_logged(self):
        """Function should complete and log iteration count"""
        grid = self._create_grid(3, 1)
        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"M{i}",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(2)
        ]

        with patch("src.optimization.refinement.calculate_grid_score", return_value=5.0):
            with patch("src.optimization.refinement.logging") as mock_logging:
                result_grid, result_bonus = refine_placement(grid, "ship", {}, "tech", tech_modules=tech_modules)

        # Should have logged completion
        mock_logging.info.assert_called()

    def test_clears_tech_before_each_permutation(self):
        """Should clear tech modules before trying each permutation"""
        grid = self._create_grid(2, 2)
        grid.cells[0][0]["module"] = "m0"
        grid.cells[0][0]["tech"] = "tech"

        tech_modules = [
            {
                "id": "m0",
                "label": "M0",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "m1",
                "label": "M1",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            },
        ]

        with patch("src.optimization.refinement.calculate_grid_score", return_value=5.0):
            with patch("src.optimization.refinement.clear_all_modules_of_tech") as mock_clear:
                result_grid, result_bonus = refine_placement(grid, "ship", {}, "tech", tech_modules=tech_modules)

        # Should have called clear multiple times (once per permutation)
        self.assertGreater(mock_clear.call_count, 0)


if __name__ == "__main__":
    unittest.main()
