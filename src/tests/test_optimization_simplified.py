import unittest
from unittest.mock import patch

from src.data_loader import get_all_module_data, get_all_solve_data
from src.grid_utils import Grid
from src.optimization import optimize_placement

sample_modules = get_all_module_data()
sample_solves = get_all_solve_data()


class TestOptimizationSimplified(unittest.TestCase):

    def setUp(self):
        self.ship = "standard"
        self.tech = "pulse"
        self.modules = sample_modules[self.ship]
        self.sc_grid = Grid(4, 3)
        self.sc_grid.set_supercharged(1, 1, True)

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.get_all_unique_pattern_variations")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.core.find_supercharged_opportunities")
    @patch("src.optimization.core._handle_ml_opportunity")
    @patch("src.optimization.core.calculate_grid_score")
    def test_refinement_improves_score_simplified(
        self,
        mock_calculate_grid_score,
        mock_handle_ml,
        mock_find_opportunities,
        mock_apply_pattern_to_grid,
        mock_get_all_unique_pattern_variations,
        mock_get_solve_map,
    ):
        """Simplified test for refinement improving score."""
        mock_get_solve_map.return_value = sample_solves[self.ship]
        mock_get_all_unique_pattern_variations.return_value = [sample_solves[self.ship][self.tech].get("map")]

        pattern_grid = self.sc_grid.copy()
        pattern_grid.set_module(0, 0, "PE")
        pattern_grid.set_tech(0, 0, self.tech)

        def apply_pattern_side_effect(*args, **kwargs):
            return pattern_grid, 10

        mock_apply_pattern_to_grid.side_effect = apply_pattern_side_effect
        mock_calculate_grid_score.side_effect = [
            20.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
            25.0,
        ]
        mock_find_opportunities.return_value = (0, 0, 4, 3)

        refined_grid = self.sc_grid.copy()
        refined_grid.set_module(1, 1, "PE")
        refined_grid.set_tech(1, 1, self.tech)
        mock_handle_ml.return_value = (refined_grid, 25.0)

        _, _, _, solve_method = optimize_placement(self.sc_grid, self.ship, self.modules, self.tech)

        self.assertEqual(solve_method, "ML")

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.get_all_unique_pattern_variations")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.core.find_supercharged_opportunities")
    @patch("src.optimization.core._handle_ml_opportunity")
    @patch("src.optimization.core._handle_sa_refine_opportunity")
    @patch("src.optimization.core.calculate_grid_score")
    def test_final_fallback_sa_improves_score_simplified(
        self,
        mock_calculate_grid_score,
        mock_handle_sa,
        mock_handle_ml,
        mock_find_opportunities,
        mock_apply_pattern_to_grid,
        mock_get_all_unique_pattern_variations,
        mock_get_solve_map,
    ):
        """Simplified test for final fallback SA improving score."""
        mock_get_solve_map.return_value = sample_solves[self.ship]
        mock_get_all_unique_pattern_variations.return_value = [sample_solves[self.ship][self.tech].get("map")]

        pattern_grid = self.sc_grid.copy()
        pattern_grid.set_module(0, 0, "PE")
        pattern_grid.set_tech(0, 0, self.tech)

        def apply_pattern_side_effect(*args, **kwargs):
            return pattern_grid, 10

        mock_apply_pattern_to_grid.side_effect = apply_pattern_side_effect
        mock_calculate_grid_score.side_effect = [
            20.0,
            15.0,
            15.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,
        ]
        mock_find_opportunities.return_value = (0, 0, 4, 3)

        mock_handle_ml.return_value = (None, 0.0)  # ML fails

        refined_grid = self.sc_grid.copy()
        refined_grid.set_module(2, 2, "PE")
        refined_grid.set_tech(2, 2, self.tech)
        mock_handle_sa.side_effect = [(None, 0.0), (refined_grid, 30.0)]

        _, _, _, solve_method = optimize_placement(self.sc_grid, self.ship, self.modules, self.tech)

        self.assertEqual(mock_handle_sa.call_count, 2)
        self.assertEqual(solve_method, "Final Fallback SA")
