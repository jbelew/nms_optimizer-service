import unittest
from unittest.mock import patch

# --- Imports from your project ---
from src.data_loader import get_all_module_data, get_all_solve_data
from src.grid_utils import Grid, apply_localized_grid_changes
from src.module_placement import clear_all_modules_of_tech
from src.optimization import optimize_placement
from src.optimization.helpers import (
    place_all_modules_in_empty_slots,
)
from src.optimization.windowing import find_supercharged_opportunities
from src.pattern_matching import (
    get_all_unique_pattern_variations,
    mirror_pattern_horizontally,
    mirror_pattern_vertically,
    rotate_pattern,
)
from src.bonus_calculations import calculate_grid_score

# Load all data for testing purposes
sample_modules = get_all_module_data()
sample_solves = get_all_solve_data()


# --- Test Class ---
class TestOptimization(unittest.TestCase):
    # --- Merged setUp method ---
    def setUp(self):
        """Set up common test resources."""
        # Existing setup
        self.grid = Grid(4, 3)  # Keep this if other tests use it directly
        self.ship = "standard"
        self.tech = "pulse"  # A tech with a known solve map
        self.modules = sample_modules[self.ship]  # Use the ship-specific data

        # Added setup from TestOptimizePlacement
        self.grid_width = 4
        self.grid_height = 3
        # self.ship = "standard" # Already defined
        # self.tech = "pulse" # Already defined
        # self.rewards = ["PC"] # Renamed to player_owned_rewards
        self.empty_grid = Grid(self.grid_width, self.grid_height)

        # Example of a grid with some pre-filled cells or inactive cells
        self.partial_grid = Grid(self.grid_width, self.grid_height)
        self.partial_grid.set_active(0, 0, False)  # Make one cell inactive

        # Example of a grid with supercharged slots
        self.sc_grid = Grid(self.grid_width, self.grid_height)
        self.sc_grid.set_supercharged(1, 1, True)
        self.sc_grid.set_supercharged(2, 1, True)

        # Mocked return value for successful SA/ML refinement
        self.mock_refined_grid = Grid(self.grid_width, self.grid_height)
        self.mock_refined_grid.set_module(1, 1, "PE")  # Example module
        self.mock_refined_grid.set_tech(1, 1, self.tech)
        self.mock_refined_score = 15.0

    # --- End Merged setUp ---

    # --- Existing Tests ---
    def test_rotate_pattern(self):
        pattern = {(0, 0): "A", (1, 0): "B", (0, 1): "C"}
        rotated_pattern = rotate_pattern(pattern)
        self.assertEqual(rotated_pattern, {(0, 1): "A", (0, 0): "B", (1, 1): "C"})

    def test_rotate_empty_pattern(self):
        pattern = {}
        rotated_pattern = rotate_pattern(pattern)
        self.assertEqual(rotated_pattern, {})

    def test_mirror_pattern_horizontally(self):
        pattern = {(0, 0): "A", (1, 0): "B", (0, 1): "C"}
        mirrored_pattern = mirror_pattern_horizontally(pattern)
        self.assertEqual(mirrored_pattern, {(1, 0): "A", (0, 0): "B", (1, 1): "C"})

    def test_mirror_pattern_horizontally_empty(self):
        pattern = {}
        mirrored_pattern = mirror_pattern_horizontally(pattern)
        self.assertEqual(mirrored_pattern, {})

    def test_mirror_pattern_vertically(self):
        pattern = {(0, 0): "A", (1, 0): "B", (0, 1): "C"}
        mirrored_pattern = mirror_pattern_vertically(pattern)
        self.assertEqual(mirrored_pattern, {(0, 1): "A", (1, 1): "B", (0, 0): "C"})

    def test_mirror_pattern_vertically_empty(self):
        pattern = {}
        mirrored_pattern = mirror_pattern_vertically(pattern)
        self.assertEqual(mirrored_pattern, {})

    def test_get_all_unique_pattern_variations(self):
        # Use a real pattern from solves for a more meaningful test
        pattern_str_keys = sample_solves.get("standard", {}).get("infra", {}).get("map", {})
        if pattern_str_keys:  # Ensure the pattern exists
            variations = get_all_unique_pattern_variations(pattern_str_keys)
            # Check that variations were generated and are dicts
            self.assertIsInstance(variations, list)
            self.assertGreater(len(variations), 0)
            self.assertIsInstance(variations[0], dict)
            # Check if keys are tuples
            first_key = next(iter(variations[0].keys()), None)
            self.assertIsInstance(first_key, tuple)
        else:
            self.skipTest("Skipping test_get_all_unique_pattern_variations: infra pattern not found in solves.")

    def test_place_all_modules_in_empty_slots(self):
        # Use the actual infra modules for standard ship
        infra_modules = [m for t in sample_modules["standard"]["types"].values() for m in t if m["key"] == "infra"][0][
            "modules"
        ]

        with patch("src.optimization.helpers.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = infra_modules
            # Use self.empty_grid which is guaranteed empty
            result_grid = place_all_modules_in_empty_slots(
                self.empty_grid.copy(),
                self.modules,
                self.ship,
                self.tech,
            )
            # Check if the correct number of modules were placed
            placed_count = 0
            for y in range(result_grid.height):
                for x in range(result_grid.width):
                    if result_grid.get_cell(x, y)["module"] is not None:
                        placed_count += 1
            self.assertEqual(placed_count, len(infra_modules))
            # Check if the first few slots are filled (assuming column-major order)
            self.assertIsNotNone(result_grid.get_cell(0, 0)["module"])
            self.assertIsNotNone(result_grid.get_cell(0, 1)["module"])
            self.assertIsNotNone(result_grid.get_cell(0, 2)["module"])
            self.assertIsNotNone(result_grid.get_cell(1, 0)["module"])

    def test_find_supercharged_opportunities_no_opportunity(self):
        # Use self.empty_grid which has no SC slots
        result = find_supercharged_opportunities(self.empty_grid, self.modules, self.ship, self.tech)
        self.assertIsNone(result)

    def test_find_supercharged_opportunities_opportunity(self):
        # Use self.sc_grid which has SC slots
        sc_grid_test = self.sc_grid.copy()
        # Place a module of the *same* tech outside the best window to ensure it's cleared
        sc_grid_test.set_module(3, 2, "IK")
        sc_grid_test.set_tech(3, 2, self.tech)
        # Place a module of a *different* tech within the window
        sc_grid_test.set_module(1, 0, "OTHER")
        sc_grid_test.set_tech(1, 0, "other_tech")

        result = find_supercharged_opportunities(
            sc_grid_test,
            self.modules,
            self.ship,
            self.tech,
        )
        # Assuming the best window starts at (0,0) for a 4x3 grid with SC at (1,1), (2,1)
        self.assertIsNotNone(result)
        self.assertEqual(result, (0, 0, 4, 3))

    def test_apply_localized_grid_changes(self):
        localized_grid = Grid(2, 2)  # Smaller localized grid
        localized_grid.set_module(0, 0, "IK")
        localized_grid.set_tech(0, 0, "infra")
        localized_grid.set_module(1, 0, "Xa")
        localized_grid.set_tech(1, 0, "infra")

        main_grid = Grid(4, 3)
        main_grid.set_module(0, 0, "OTHER")  # Existing module of different tech
        main_grid.set_tech(0, 0, "other_tech")

        start_x, start_y = 1, 1  # Apply the 2x2 localized grid starting at (1,1)

        apply_localized_grid_changes(main_grid, localized_grid, "infra", start_x, start_y)

        # Check changes within the applied area
        self.assertEqual(main_grid.get_cell(1, 1)["module"], "IK")
        self.assertEqual(main_grid.get_cell(1, 1)["tech"], "infra")
        self.assertEqual(main_grid.get_cell(2, 1)["module"], "Xa")
        self.assertEqual(main_grid.get_cell(2, 1)["tech"], "infra")

        # Check that other cells were not affected
        self.assertEqual(main_grid.get_cell(0, 0)["module"], "OTHER")  # Original module untouched
        self.assertIsNone(main_grid.get_cell(0, 1)["module"])  # Outside area untouched

    def test_clear_all_modules_of_tech(self):
        self.grid.set_module(0, 0, "IK")
        self.grid.set_tech(0, 0, "infra")
        self.grid.set_module(1, 1, "OTHER")  # Add another tech
        self.grid.set_tech(1, 1, "other_tech")
        clear_all_modules_of_tech(self.grid, "infra")
        # Check infra module is gone
        self.assertIsNone(self.grid.get_cell(0, 0)["module"])
        self.assertIsNone(self.grid.get_cell(0, 0)["tech"])
        # Check other tech module remains
        self.assertEqual(self.grid.get_cell(1, 1)["module"], "OTHER")
        self.assertEqual(self.grid.get_cell(1, 1)["tech"], "other_tech")

    # --- Added Tests for optimize_placement ---

    def test_optimize_no_empty_active_slots(self):
        """Test optimize_placement raises ValueError when no empty, active slots exist."""
        full_grid = Grid(self.grid_width, self.grid_height)
        # Fill the grid or make all slots inactive
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                full_grid.set_active(x, y, False)  # Make all inactive

        with self.assertRaisesRegex(ValueError, "No empty, active slots available"):
            optimize_placement(
                full_grid,
                self.ship,
                self.modules,
                self.tech,
            )

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.calculate_pattern_adjacency_score")
    @patch("src.optimization.core.calculate_grid_score")
    def test_optimize_no_solve_map_available(self, mock_calculate_score, mock_adjacency_score, mock_get_solves):
        """Test behavior when no solve map is found for the ship (uses adjacency scoring)."""
        mock_get_solves.return_value = {}

        # Mock adjacency scoring for placement selection (one per cell in grid for multi-module)
        # Standard grid is 4x3, so we need enough scores for all cells × number of modules
        mock_adjacency_score.return_value = 5.0
        mock_calculate_score.return_value = 8.5

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.empty_grid,
            self.ship,
            self.modules,
            self.tech,
        )

        mock_get_solves.assert_called_once_with(self.ship)
        # Should use adjacency scoring for placement selection
        self.assertTrue(mock_adjacency_score.called)
        # Final score should be calculated
        mock_calculate_score.assert_called()
        self.assertEqual(solved_bonus, 8.5)
        self.assertEqual(percentage, 100.0)
        # Should indicate fallback method
        self.assertIn("No Solve", solve_method)

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.core.simulated_annealing")
    def test_optimize_solve_map_no_pattern_fits_returns_indicator_when_not_forced(
        self, mock_simulated_annealing, mock_apply_pattern_to_grid, mock_get_solve_map
    ):
        """Test returns 'Pattern No Fit' when solve map exists, no pattern fits, and not forced."""
        mock_get_solve_map.return_value = sample_solves[self.ship]
        mock_apply_pattern_to_grid.return_value = (None, 0)

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.empty_grid,
            self.ship,
            self.modules,
            self.tech,
            forced=False,
        )

        mock_apply_pattern_to_grid.assert_called()
        mock_simulated_annealing.assert_not_called()
        self.assertIsNone(result_grid)
        self.assertEqual(percentage, 0.0)
        self.assertEqual(solved_bonus, 0.0)
        self.assertEqual(solve_method, "Pattern No Fit")

    @patch("src.optimization.core.calculate_grid_score")
    @patch("src.optimization.core.simulated_annealing")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.core.get_solve_map")
    def test_optimize_solve_map_no_pattern_fits_falls_back_to_sa_when_forced(
        self, mock_get_solve_map, mock_apply_pattern_to_grid, mock_simulated_annealing, mock_calculate_grid_score
    ):
        """Test fallback to initial SA when solve map exists, no pattern fits, and forced=True."""
        mock_get_solve_map.return_value = sample_solves[self.ship]
        mock_apply_pattern_to_grid.return_value = (None, 0)

        initial_sa_grid = self.empty_grid.copy()
        initial_sa_grid.set_module(0, 1, "PE")
        initial_sa_grid.set_tech(0, 1, self.tech)
        mock_simulated_annealing.return_value = (initial_sa_grid, 10.0)
        mock_calculate_grid_score.return_value = 10.0

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.empty_grid,
            self.ship,
            self.modules,
            self.tech,
            forced=True,
        )

        mock_apply_pattern_to_grid.assert_called()
        mock_simulated_annealing.assert_called_once()
        self.assertEqual(solve_method, "Forced Initial SA (No Pattern Fit)")
        self.assertEqual(solved_bonus, 10.0)
        # The score may be calculated multiple times, so we check the last call.
        mock_calculate_grid_score.assert_called_with(initial_sa_grid, self.tech, apply_supercharge_first=False)

    @patch("src.optimization.core.get_tech_modules")
    @patch("src.optimization.core.find_supercharged_opportunities")
    @patch("src.optimization.core.determine_window_dimensions")
    def test_optimize_partial_set_no_window_no_forced_returns_indicator(
        self, mock_determine_window_dimensions, mock_find_opportunities, mock_get_tech_modules
    ):
        """Test returns 'Pattern No Fit' when a partial set has no window and is not forced."""
        # Simulate a partial module set
        full_module_list = [
            {
                "id": "A",
                "label": "Module A",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "B",
                "label": "Module B",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "C",
                "label": "Module C",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "D",
                "label": "Module D",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "E",
                "label": "Module E",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "F",
                "label": "Module F",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
        ]
        partial_module_list = [
            {
                "id": "A",
                "label": "Module A",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "B",
                "label": "Module B",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "C",
                "label": "Module C",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
        ]
        mock_get_tech_modules.side_effect = [
            full_module_list,
            partial_module_list,
        ]
        # Simulate no suitable window found
        mock_find_opportunities.return_value = None
        # Mock determine_window_dimensions to return a size that won't fit in an empty grid
        # This will force the 'Could not find any suitable window' path
        mock_determine_window_dimensions.return_value = (100, 100)  # A window larger than the grid

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.empty_grid,
            self.ship,
            self.modules,
            self.tech,
            forced=False,
            available_modules=["A", "B", "C"],
        )

        mock_get_tech_modules.assert_called()
        mock_find_opportunities.assert_called_once()
        mock_determine_window_dimensions.assert_called_once()
        self.assertIsNone(result_grid)
        self.assertEqual(percentage, 0.0)
        self.assertEqual(solved_bonus, 0.0)
        self.assertEqual(solve_method, "Pattern No Fit")

    @patch("src.optimization.core.simulated_annealing")
    @patch("src.optimization.core.determine_window_dimensions")
    @patch("src.optimization.core.find_supercharged_opportunities")
    @patch("src.optimization.core.get_tech_modules")
    def test_optimize_partial_set_no_window_forced_falls_back_to_full_sa(
        self, mock_get_tech_modules, mock_find_opportunities, mock_determine_window_dimensions, mock_simulated_annealing
    ):
        """Test fallback to full SA for a partial set with no window when forced=True."""
        # Simulate a partial module set
        full_module_list = [
            {
                "id": "A",
                "label": "Module A",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "B",
                "label": "Module B",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "C",
                "label": "Module C",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "D",
                "label": "Module D",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
        ]
        partial_module_list = [
            {
                "id": "A",
                "label": "Module A",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
            {
                "id": "B",
                "label": "Module B",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": False,
                "sc_eligible": False,
                "image": None,
            },
        ]
        mock_get_tech_modules.side_effect = [full_module_list, partial_module_list, partial_module_list]

        # Simulate no suitable window found
        mock_find_opportunities.return_value = None
        mock_determine_window_dimensions.return_value = (100, 100)  # A window larger than the grid

        # Mock the fallback SA
        sa_grid = self.empty_grid.copy()
        sa_grid.set_module(0, 0, "A")
        sa_grid.set_tech(0, 0, self.tech)
        mock_simulated_annealing.return_value = (sa_grid, 20.0)

        _, _, solved_bonus, solve_method = optimize_placement(
            self.empty_grid,
            self.ship,
            self.modules,
            self.tech,
            forced=True,
            available_modules=["A", "B"],
        )

        mock_find_opportunities.assert_called_once()
        mock_determine_window_dimensions.assert_called_once()
        mock_simulated_annealing.assert_called_once()
        self.assertEqual(solve_method, "Partial Set SA Pattern Match")
        self.assertEqual(solved_bonus, 1.0)

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.get_all_unique_pattern_variations")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.core.find_supercharged_opportunities")
    @patch("src.optimization.core._handle_ml_opportunity")
    @patch("src.optimization.core._handle_sa_refine_opportunity")
    @patch("src.optimization.core.calculate_grid_score")
    def test_successful_pattern_match_no_refinement(
        self,
        mock_calculate_grid_score,
        mock_handle_sa,
        mock_handle_ml,
        mock_find_opportunities,
        mock_apply_pattern_to_grid,
        mock_get_all_unique_pattern_variations,
        mock_get_solve_map,
    ):
        """Test a successful pattern match with no refinement opportunity."""
        mock_get_solve_map.return_value = sample_solves[self.ship]
        mock_get_all_unique_pattern_variations.return_value = [sample_solves[self.ship][self.tech].get("map")]

        pattern_grid = self.empty_grid.copy()
        pattern_grid.set_module(0, 0, "PE")
        pattern_grid.set_tech(0, 0, self.tech)

        def apply_pattern_side_effect(*args, **kwargs):
            if args[4] == 0 and args[5] == 0:
                return pattern_grid, 10
            return None, 0

        mock_apply_pattern_to_grid.side_effect = apply_pattern_side_effect
        mock_calculate_grid_score.return_value = 20.0
        mock_find_opportunities.return_value = None  # No SC opportunity

        result_grid, _, solved_bonus, solve_method = optimize_placement(
            self.empty_grid, self.ship, self.modules, self.tech
        )

        mock_apply_pattern_to_grid.assert_called()
        mock_find_opportunities.assert_called_once()
        mock_handle_ml.assert_not_called()
        mock_handle_sa.assert_not_called()
        self.assertEqual(solve_method, "Pattern Match")
        self.assertEqual(solved_bonus, 20.0)
        self.assertEqual(result_grid.get_cell(0, 0)["module"], "PE")  # type: ignore[union-attr]

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.get_all_unique_pattern_variations")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.core.find_supercharged_opportunities")
    @patch("src.optimization.core._handle_ml_opportunity")
    @patch("src.optimization.core.calculate_grid_score")
    def test_refinement_improves_score(
        self,
        mock_calculate_grid_score,
        mock_handle_ml,
        mock_find_opportunities,
        mock_apply_pattern_to_grid,
        mock_get_all_unique_pattern_variations,
        mock_get_solve_map,
    ):
        """Test that refinement improves the score."""
        mock_get_solve_map.return_value = sample_solves[self.ship]
        mock_get_all_unique_pattern_variations.return_value = [sample_solves[self.ship][self.tech].get("map")]

        pattern_grid = self.sc_grid.copy()
        pattern_grid.set_module(0, 0, "PE")
        pattern_grid.set_tech(0, 0, self.tech)

        def apply_pattern_side_effect(*args, **kwargs):
            if args[4] == 0 and args[5] == 0:
                return pattern_grid, 10
            return None, 0

        mock_apply_pattern_to_grid.side_effect = apply_pattern_side_effect
        mock_calculate_grid_score.side_effect = [20.0, 25.0, 25.0]
        mock_find_opportunities.return_value = (0, 0, 4, 3)

        refined_grid = self.sc_grid.copy()
        refined_grid.set_module(1, 1, "PE")
        refined_grid.set_tech(1, 1, self.tech)
        mock_handle_ml.return_value = (refined_grid, 25.0)

        result_grid, _, solved_bonus, solve_method = optimize_placement(
            self.sc_grid, self.ship, self.modules, self.tech
        )

        mock_handle_ml.assert_called_once()
        self.assertEqual(solve_method, "ML")
        self.assertEqual(solved_bonus, 25.0)
        self.assertEqual(result_grid.get_cell(1, 1)["module"], "PE")  # type: ignore[union-attr]

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.get_all_unique_pattern_variations")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.core.find_supercharged_opportunities")
    @patch("src.optimization.core._handle_ml_opportunity")
    @patch("src.optimization.core._handle_sa_refine_opportunity")
    @patch("src.optimization.core.calculate_grid_score")
    def test_refinement_does_not_improve_score(
        self,
        mock_calculate_grid_score,
        mock_handle_sa,
        mock_handle_ml,
        mock_find_opportunities,
        mock_apply_pattern_to_grid,
        mock_get_all_unique_pattern_variations,
        mock_get_solve_map,
    ):
        """Test that refinement does not improve the score."""
        mock_get_solve_map.return_value = sample_solves[self.ship]
        mock_get_all_unique_pattern_variations.return_value = [sample_solves[self.ship][self.tech].get("map")]

        pattern_grid = self.sc_grid.copy()
        pattern_grid.set_module(0, 0, "PE")
        pattern_grid.set_tech(0, 0, self.tech)

        def apply_pattern_side_effect(*args, **kwargs):
            if args[4] == 0 and args[5] == 0:
                return pattern_grid, 10
            return None, 0

        mock_apply_pattern_to_grid.side_effect = apply_pattern_side_effect

        def calculate_grid_score_side_effect(grid_arg, tech_arg, apply_supercharge_first=False):
            if grid_arg.get_cell(0, 0)["module"] == "PE":
                return 20.0
            if grid_arg.get_cell(1, 1)["module"] == "PE":
                return 15.0
            return 0.0

        mock_calculate_grid_score.side_effect = calculate_grid_score_side_effect
        mock_find_opportunities.return_value = (0, 0, 4, 3)

        refined_grid = self.sc_grid.copy()
        refined_grid.set_module(1, 1, "PE")
        refined_grid.set_tech(1, 1, self.tech)
        mock_handle_ml.return_value = (refined_grid, 15.0)  # ML succeeds but with a lower score
        mock_handle_sa.return_value = (None, 0.0)  # Final fallback SA also fails to improve

        result_grid, _, solved_bonus, solve_method = optimize_placement(
            self.sc_grid, self.ship, self.modules, self.tech
        )

        mock_handle_ml.assert_called_once()
        mock_handle_sa.assert_called_once()  # Final fallback SA should be called once
        self.assertEqual(solve_method, "Pattern Match")
        self.assertEqual(solved_bonus, 20.0)
        self.assertEqual(result_grid.get_cell(0, 0)["module"], "PE")  # type: ignore[union-attr]

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.get_all_unique_pattern_variations")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.core.find_supercharged_opportunities")
    @patch("src.optimization.core._handle_ml_opportunity")
    @patch("src.optimization.core._handle_sa_refine_opportunity")
    @patch("src.optimization.core.calculate_grid_score")
    def test_final_fallback_sa_improves_score(
        self,
        mock_calculate_grid_score,
        mock_handle_sa,
        mock_handle_ml,
        mock_find_opportunities,
        mock_apply_pattern_to_grid,
        mock_get_all_unique_pattern_variations,
        mock_get_solve_map,
    ):
        """Test that final fallback SA improves the score."""
        mock_get_solve_map.return_value = sample_solves[self.ship]
        mock_get_all_unique_pattern_variations.return_value = [sample_solves[self.ship][self.tech].get("map")]

        pattern_grid = self.sc_grid.copy()
        pattern_grid.set_module(0, 0, "PE")
        pattern_grid.set_tech(0, 0, self.tech)

        def apply_pattern_side_effect(*args, **kwargs):
            if args[4] == 0 and args[5] == 0:
                return pattern_grid, 10
            return None, 0

        mock_apply_pattern_to_grid.side_effect = apply_pattern_side_effect

        def calculate_grid_score_side_effect(grid_arg, tech_arg, apply_supercharge_first=False):
            if grid_arg.get_cell(0, 0)["module"] == "PE":
                return 20.0
            if grid_arg.get_cell(2, 2)["module"] == "PE":
                return 30.0
            return 0.0

        mock_calculate_grid_score.side_effect = calculate_grid_score_side_effect
        mock_find_opportunities.return_value = (0, 0, 4, 3)

        mock_handle_ml.return_value = (None, 0.0)  # ML fails

        refined_grid = self.sc_grid.copy()
        refined_grid.set_module(2, 2, "PE")
        refined_grid.set_tech(2, 2, self.tech)
        # First SA (ML fallback) fails, second (final fallback) succeeds
        mock_handle_sa.side_effect = [(None, 0.0), (refined_grid, 30.0)]

        result_grid, _, solved_bonus, solve_method = optimize_placement(
            self.sc_grid, self.ship, self.modules, self.tech
        )

        self.assertEqual(mock_handle_sa.call_count, 2)
        self.assertEqual(solve_method, "Final Fallback SA")
        self.assertEqual(solved_bonus, 30.0)
        self.assertEqual(result_grid.get_cell(2, 2)["module"], "PE")  # type: ignore[union-attr]

    @patch("src.optimization.core.get_tech_modules")
    def test_optimize_no_modules_found_returns_error(self, mock_get_tech_modules):
        """Test that optimize_placement returns an error if no modules are found for the tech."""
        mock_get_tech_modules.return_value = []

        # Create a grid that has some modules of the tech to ensure they are cleared
        grid_with_modules = self.empty_grid.copy()
        grid_with_modules.set_module(0, 0, "PE")
        grid_with_modules.set_tech(0, 0, self.tech)

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            grid_with_modules,
            self.ship,
            self.modules,
            self.tech,
        )

        mock_get_tech_modules.assert_called()
        self.assertEqual(solve_method, "Module Definition Error")
        self.assertEqual(percentage, 0.0)
        self.assertEqual(solved_bonus, 0.0)

        # Check that the module of that tech has been cleared
        self.assertIsNone(result_grid.get_cell(0, 0)["module"])  # type: ignore[union-attr]

    def test_get_tech_modules_no_solve_type_returns_untyped_modules(self):
        # Mock ship_modules data
        mock_ship_modules = {
            "types": {
                "some_category": [
                    {
                        "key": "test_tech",
                        "label": "Test Tech Normal",
                        "modules": [{"id": "MOD_A", "type": "normal"}],
                        "type": "normal",
                    },
                    {
                        "key": "test_tech",
                        "label": "Test Tech Untyped",
                        "modules": [{"id": "MOD_B", "type": "untyped"}],
                        # No 'type' key here
                    },
                    {
                        "key": "test_tech",
                        "label": "Test Tech Max",
                        "modules": [{"id": "MOD_C", "type": "max"}],
                        "type": "max",
                    },
                ]
            }
        }

        # Call get_tech_modules
        from src.modules_utils import get_tech_modules

        result_modules = get_tech_modules(mock_ship_modules, "test_ship", "test_tech")

        # Assert that it returns modules from the untyped definition
        self.assertIsNotNone(result_modules)
        if isinstance(result_modules, list):
            self.assertEqual(len(result_modules), 1)
            self.assertEqual(result_modules[0]["id"], "MOD_B")

    def test_simulated_annealing_improves_score(self):
        """
        Test that simulated_annealing improves the score of a suboptimal layout.
        """
        import random
        from src.module_placement import place_module

        random.seed(42)  # for deterministic results

        # 1. Create a suboptimal grid layout
        grid = Grid(4, 3)
        tech = "pulse"
        ship = "standard"
        modules = sample_modules[ship]
        tech_modules = [m for t in modules["types"].values() for m in t if m["key"] == tech][0]["modules"]

        # Place modules in a simple, non-optimal way (e.g., in order of appearance)
        clear_all_modules_of_tech(grid, tech)
        x, y = 0, 0
        for module in tech_modules:
            while not grid.get_cell(x, y)["active"] or grid.get_cell(x, y)["module"] is not None:
                x += 1
                if x >= grid.width:
                    x = 0
                    y += 1
            place_module(
                grid,
                x,
                y,
                module["id"],
                module["label"],
                tech,
                module["type"],
                module["bonus"],
                module["adjacency"],
                module["sc_eligible"],
                module["image"],
            )

        initial_score = calculate_grid_score(grid, tech, apply_supercharge_first=False)
        self.assertGreater(initial_score, 0)  # Ensure we have a starting score

        # 2. Run simulated annealing
        from src.optimization.refinement import simulated_annealing

        best_grid, best_score = simulated_annealing(
            grid,
            ship,
            modules,
            tech,
            full_grid=grid,
            tech_modules=tech_modules,
            initial_temperature=10,  # Lower temp for faster test
            cooling_rate=0.9,
            iterations_per_temp=10,
        )

        # 3. Assert that the score has improved
        self.assertIsNotNone(best_grid)

    @patch("src.optimization.core._handle_sa_refine_opportunity")
    @patch("src.optimization.core.get_tech_modules")
    def test_partial_set_chooses_best_rotated_window(
        self,
        mock_get_tech_modules,
        mock_handle_sa_refine_opportunity,
    ):
        """
        Tests that for a partial set with non-sc_eligible modules,
        the code finds windows without supercharged cells.
        """
        # 1. Setup: Grid where non-eligible modules must avoid supercharged cells.
        grid = Grid(5, 5)
        # Place supercharged cells at column 3, rows 0-2
        grid.set_supercharged(3, 0, True)
        grid.set_supercharged(3, 1, True)
        grid.set_supercharged(3, 2, True)
        # With 6 non-sc_eligible modules, the code should find a 3x2 or 2x3
        # window in the non-supercharged region (columns 0-2 or 4)

        num_modules = 6
        full_module_list = [
            {
                "id": f"M{i}",
                "label": f"Module {i}",
                "type": "bonus",
                "bonus": 1.0,
                "adjacency": "none",
                "sc_eligible": False,  # Non-eligible for supercharge
                "image": None,
            }
            for i in range(num_modules)
        ]
        partial_module_list = full_module_list[:]
        mock_get_tech_modules.side_effect = [full_module_list, partial_module_list, partial_module_list]

        mock_handle_sa_refine_opportunity.return_value = (grid, 100.0)

        # 2. Run optimization
        optimize_placement(
            grid,
            "standard",
            self.modules,
            "pulse",
            forced=True,
            available_modules=[m["id"] for m in partial_module_list],
        )

        # 3. Assertions
        mock_handle_sa_refine_opportunity.assert_called_once()
        args, kwargs = mock_handle_sa_refine_opportunity.call_args

        window_width = args[6]
        window_height = args[7]

        # For 6 modules with no supercharged cells in the window,
        # determine_window_dimensions should return 3x2 (or 2x3 if rotated)
        # The key is that neither dimension should be 3 (which was looking for supercharge)
        # and the window should NOT span column 3
        self.assertIn(window_width, [2, 3])
        self.assertIn(window_height, [2, 3])

    def test_optimize_prioritizes_supercharged_scanned_window_over_non_sc_3x2_pattern(self):
        """
        Regression test for issue #813: when a 3x2 pattern matches in a non-supercharged
        region (window score 6.0) and a valid 3x2 Solver Window with 1 or 2 open
        supercharged slots exists elsewhere on the grid (window score 2.75-5.75),
        optimize_placement must select the supercharged scanned window and refine using
        the supercharged slots.
        """
        grid = Grid(7, 2)
        # Separate left 3x2 region (0..2, 0..1) from right 3x2 region (4..6, 0..1)
        grid.set_active(3, 0, False)
        grid.set_active(3, 1, False)
        # In the right 3x2 region, deactivate middle cell (5, 0) so the rigid shield
        # 3x2 pattern (which only has a corner empty slot) cannot match there, while 5
        # active slots remain for the 5 shield modules around 2 supercharged slots.
        grid.set_active(5, 0, False)
        grid.set_supercharged(4, 1, True)
        grid.set_supercharged(5, 1, True)

        result_grid, _, solved_bonus, solve_method = optimize_placement(
            grid,
            "standard",
            self.modules,
            "shield",
        )

        self.assertIsNotNone(result_grid)
        if result_grid is not None:
            # Both open supercharged slots in the scanned window should be occupied by shield modules
            self.assertIsNotNone(result_grid.get_cell(4, 1)["module"])
            self.assertEqual(result_grid.get_cell(4, 1)["tech"], "shield")
            self.assertIsNotNone(result_grid.get_cell(5, 1)["module"])
            self.assertEqual(result_grid.get_cell(5, 1)["tech"], "shield")
            self.assertNotEqual(solve_method, "Pattern Match")
            self.assertGreater(solved_bonus, 2.0649)

    def _assert_ml_refines_window(
        self,
        grid: Grid,
        tech: str,
        expected_window: tuple[int, int, int, int],
        sc_coord: tuple[int, int],
    ) -> None:
        """
        Helper to run optimize_placement with a stubbed ML refinement outcome and
        verify that refinement targets expected_window and adopts the refined result.
        """
        sc_x, sc_y = sc_coord
        refined_grid = grid.copy()
        refined_grid.set_module(sc_x, sc_y, "Xa")
        refined_grid.set_tech(sc_x, sc_y, tech)

        with patch(
            "src.optimization.core._handle_ml_opportunity",
            return_value=(refined_grid, 50.0),
        ) as mock_handle_ml:
            with patch(
                "src.optimization.core.calculate_grid_score",
                side_effect=lambda g, t, **kw: 50.0 if g.get_cell(sc_x, sc_y)["module"] == "Xa" else 10.0,
            ):
                result_grid, _, solved_bonus, solve_method = optimize_placement(
                    grid,
                    "standard",
                    self.modules,
                    tech,
                )

        mock_handle_ml.assert_called_once()
        ml_args = mock_handle_ml.call_args[0]
        self.assertEqual((ml_args[4], ml_args[5], ml_args[6], ml_args[7]), expected_window)
        self.assertEqual(solve_method, "ML")
        self.assertEqual(solved_bonus, 50.0)
        self.assertIsNotNone(result_grid)

    def test_optimize_prioritizes_supercharged_scanned_window_over_non_sc_3x3_pattern(self):
        """
        Regression test for issue #813: when a 3x3 pattern matches in a non-supercharged
        region (window score 9.0) and a valid Solver Window with 1 open supercharged slot
        exists elsewhere on the grid (window score 3.0), optimize_placement must select
        the supercharged scanned window (4, 0, 3, 3) and refine using the supercharged slot.
        """
        # Use hyper (3x3 pattern, 9 modules) where the pattern matches at (0, 0) in a
        # non-supercharged 3x3 region (window score 9.0) and a 3x3 supercharged window
        # at (4, 0) has 1 open supercharged slot at (5, 1) (window score 3.0).
        grid = Grid(7, 3)
        grid.set_active(3, 0, False)
        grid.set_active(3, 1, False)
        grid.set_active(3, 2, False)
        grid.set_supercharged(5, 1, True)

        orig_apply = __import__("src.pattern_matching", fromlist=["apply_pattern_to_grid"]).apply_pattern_to_grid

        def apply_only_left(g, pattern, modules, tech, start_x, start_y, ship, tech_modules=None):
            if start_x == 0 and start_y == 0:
                return orig_apply(g, pattern, modules, tech, start_x, start_y, ship, tech_modules=tech_modules)
            return None, 0

        with patch("src.optimization.core.apply_pattern_to_grid", side_effect=apply_only_left):
            self._assert_ml_refines_window(
                grid,
                "hyper",
                expected_window=(4, 0, 3, 3),
                sc_coord=(5, 1),
            )

    @patch("src.optimization.core.find_supercharged_opportunities", return_value=None)
    def test_pulse_4x3_sizing_prioritizes_supercharged_window_over_non_sc_pattern(
        self,
        _mock_find_opportunities,
    ):
        """
        Regression test for issue #813 (pulse 4x3 window sizing): when pulse (8 modules)
        initially matches a 4x2 non-supercharged pattern (window score 8.0) and
        _scan_grid_with_window finds a 4x3 window containing 1 open supercharged slot
        (window score 3.0 < 8.0), pulse 4x3 sizing must override the non-supercharged
        window with the supercharged 4x3 window.
        """
        grid = Grid(9, 3)
        # Separate left 4x2 non-SC region (0..3, 0..1) from right 4x3 SC region (5..8, 0..2)
        grid.set_active(4, 0, False)
        grid.set_active(4, 1, False)
        grid.set_active(4, 2, False)
        for x in range(4):
            grid.set_active(x, 2, False)
        # Deactivate (6, 0) and (6, 2) in the right 4x3 region so the rigid 4x2 photonix
        # pattern cannot fit in either rows 0..1 or rows 1..2 of (5..8), while 10 active
        # cells (>= 8 modules) remain in the 4x3 window (5..8, 0..2) with 1 open
        # supercharged slot at (6, 1) (window score 3.0 < 8.0).
        grid.set_active(6, 0, False)
        grid.set_active(6, 2, False)
        grid.set_supercharged(6, 1, True)

        self._assert_ml_refines_window(
            grid,
            "pulse",
            expected_window=(5, 0, 4, 3),
            sc_coord=(6, 1),
        )

    def test_optimize_returns_supercharged_no_fit_when_sc_window_blocked_and_not_forced(self):
        """
        When unoccupied active supercharged slots exist and the technology has >1 modules
        with at least one sc_eligible module, but surrounding modules block any (w, h) or
        (h, w) Solver Window around the supercharged slots while a non-supercharged window
        fits elsewhere, optimize_placement(..., forced=False) must return
        (None, 0.0, 0.0, "Supercharged No Fit").
        """
        grid = Grid(7, 2)
        # Left 3x2 region (0..2, 0..1) is open and non-supercharged (shield 3x2 pattern fits here).
        # Right region has an open supercharged slot at (5, 1), but other_tech modules block
        # enough cells around it that fewer than 5 empty active cells exist in any 3x2/2x3 window
        # containing (5, 1).
        for x, y in [(3, 0), (3, 1), (4, 0), (5, 0), (6, 0)]:
            grid.set_module(x, y, "OTHER")
            grid.set_tech(x, y, "other_tech")
        grid.set_supercharged(5, 1, True)

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            grid,
            "standard",
            self.modules,
            "shield",
            forced=False,
        )

        self.assertIsNone(result_grid)
        self.assertEqual(percentage, 0.0)
        self.assertEqual(solved_bonus, 0.0)
        self.assertEqual(solve_method, "Supercharged No Fit")

    def test_optimize_returns_pattern_no_fit_when_no_window_or_pattern_fits_anywhere(self):
        """
        When unoccupied active supercharged slots exist and no supercharged Solver Window fits,
        and no non-supercharged window or pattern fits anywhere on the grid either,
        optimize_placement(..., forced=False) must return (None, 0.0, 0.0, "Pattern No Fit").
        """
        grid = Grid(4, 2)
        # Occupy cells so fewer than 5 empty active cells exist anywhere on the 4x2 grid,
        # while leaving an unoccupied active supercharged slot at (3, 1).
        for x, y in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]:
            grid.set_module(x, y, "OTHER")
            grid.set_tech(x, y, "other_tech")
        grid.set_supercharged(3, 1, True)

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            grid,
            "standard",
            self.modules,
            "shield",
            forced=False,
        )

        self.assertIsNone(result_grid)
        self.assertEqual(percentage, 0.0)
        self.assertEqual(solved_bonus, 0.0)
        self.assertEqual(solve_method, "Pattern No Fit")

    def test_optimize_returns_supercharged_no_fit_when_pattern_fits_non_sc_region_without_full_window(self):
        """
        When determine_window_dimensions resolves a 4x3 Solver Window (e.g. pulse with 8 modules)
        that cannot fit on a 7x2 grid around an open supercharged slot, but the 4x2 precomputed
        pattern fits in a non-supercharged region, optimize_placement(..., forced=False) must
        halt and return (None, 0.0, 0.0, "Supercharged No Fit").
        """
        grid = Grid(7, 2)
        # Left 4x2 region (0..3, 0..1) is open and non-supercharged (photonix 4x2 pattern fits here).
        # Right region has an open supercharged slot at (6, 1) surrounded by other_tech modules.
        for x, y in [(4, 0), (4, 1), (5, 0), (5, 1), (6, 0)]:
            grid.set_module(x, y, "OTHER")
            grid.set_tech(x, y, "other_tech")
        grid.set_supercharged(6, 1, True)

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            grid,
            "standard",
            self.modules,
            "pulse",
            forced=False,
        )

        self.assertIsNone(result_grid)
        self.assertEqual(percentage, 0.0)
        self.assertEqual(solved_bonus, 0.0)
        self.assertEqual(solve_method, "Supercharged No Fit")

    def test_optimize_does_not_trigger_supercharged_no_fit_for_exclusions_or_rotated_fit(self):
        """
        Verify that:
        1. Single-module selections (len <= 1) do not trigger "Supercharged No Fit".
        2. Selections with zero sc_eligible modules do not trigger "Supercharged No Fit".
        3. Grids where all supercharged slots are occupied by other technologies do not
           trigger "Supercharged No Fit".
        4. Grids where the rotated (h, w) Solver Window fits over an open supercharged slot
           (even though original (w, h) does not) do not trigger "Supercharged No Fit".
        """
        # 1. Single-module selection (shield core module 'DS' has sc_eligible=False, or single sc_eligible module 'Xa')
        grid_single = Grid(7, 2)
        for x, y in [(3, 0), (3, 1), (4, 0), (5, 0), (6, 0)]:
            grid_single.set_module(x, y, "OTHER")
            grid_single.set_tech(x, y, "other_tech")
        grid_single.set_supercharged(5, 1, True)

        single_grid, _, _, single_method = optimize_placement(
            grid_single,
            "standard",
            self.modules,
            "shield",
            forced=False,
            available_modules=["Xa"],
        )
        self.assertIsNotNone(single_grid)
        self.assertNotEqual(single_method, "Supercharged No Fit")

        # 2. Multi-module selection with zero sc_eligible modules (trails figurines/trails are sc_eligible=False)
        non_sc_grid, _, _, non_sc_method = optimize_placement(
            grid_single,
            "standard",
            self.modules,
            "trails",
            forced=False,
            available_modules=["RT", "CT", "ET"],
        )
        self.assertIsNotNone(non_sc_grid)
        self.assertNotEqual(non_sc_method, "Supercharged No Fit")

        # 3. Supercharged slot occupied by another technology (no unoccupied active SC slot)
        grid_occupied_sc = grid_single.copy()
        grid_occupied_sc.set_module(5, 1, "OTHER_SC")
        grid_occupied_sc.set_tech(5, 1, "other_tech")

        occ_grid, _, _, occ_method = optimize_placement(
            grid_occupied_sc,
            "standard",
            self.modules,
            "shield",
            forced=False,
        )
        self.assertIsNotNone(occ_grid)
        self.assertNotEqual(occ_method, "Supercharged No Fit")

        # 4. Rotated (2x3) window fits over open supercharged slot when (3x2) does not
        grid_rotated = Grid(2, 3)
        grid_rotated.set_supercharged(1, 1, True)
        rot_grid, _, _, rot_method = optimize_placement(
            grid_rotated,
            "standard",
            self.modules,
            "shield",
            forced=False,
        )
        self.assertIsNotNone(rot_grid)
        self.assertNotEqual(rot_method, "Supercharged No Fit")

    def test_optimize_forced_evaluates_full_grid_sa_when_supercharged_window_blocked(self):
        """
        When forced=True and no supercharged Solver Window fits around an open supercharged
        slot, optimize_placement must bypass "Supercharged No Fit", evaluate full-grid
        Simulated Annealing alongside standard non-supercharged placement, and return
        whichever achieves the higher score (both for full-set pattern match and partial sets).
        """
        grid = Grid(7, 2)
        # Left 3x2 region (0..2, 0..1) is open and non-supercharged.
        # Right region has an open supercharged slot at (5, 1) with only (4, 1), (5, 1), (6, 1) open,
        # blocking any 3x2 or 2x3 supercharged Solver Window for shield (5 modules).
        for x, y in [(3, 0), (3, 1), (4, 0), (5, 0), (6, 0)]:
            grid.set_module(x, y, "OTHER")
            grid.set_tech(x, y, "other_tech")
        grid.set_supercharged(5, 1, True)

        high_sa_grid = grid.copy()
        high_sa_grid.set_module(5, 1, "Xa")
        high_sa_grid.set_tech(5, 1, "shield")

        # Case A: Full-grid SA scores higher than non-supercharged Pattern Match -> returns SA grid
        with patch(
            "src.optimization.core.simulated_annealing",
            return_value=(high_sa_grid, 99.0),
        ) as mock_sa:
            with patch(
                "src.optimization.core.calculate_grid_score",
                side_effect=lambda g, t, **kw: 99.0 if g.get_cell(5, 1)["module"] == "Xa" else 2.0,
            ):
                result_grid, _, solved_bonus, solve_method = optimize_placement(
                    grid,
                    "standard",
                    self.modules,
                    "shield",
                    forced=True,
                )

        mock_sa.assert_called_once()
        self.assertIsNotNone(result_grid)
        if result_grid is not None:
            self.assertEqual(result_grid.get_cell(5, 1)["module"], "Xa")
        self.assertEqual(solved_bonus, 99.0)
        self.assertEqual(solve_method, "Forced Full SA")

        # Case B: Non-supercharged Pattern Match scores higher than full-grid SA -> keeps Pattern Match
        with patch(
            "src.optimization.core.simulated_annealing",
            return_value=(high_sa_grid, 1.0),
        ) as mock_sa_low:
            with patch(
                "src.optimization.core.calculate_grid_score",
                side_effect=lambda g, t, **kw: 1.0 if g.get_cell(5, 1)["module"] == "Xa" else 20.0,
            ):
                result_grid_b, _, solved_bonus_b, solve_method_b = optimize_placement(
                    grid,
                    "standard",
                    self.modules,
                    "shield",
                    forced=True,
                )

        mock_sa_low.assert_called_once()
        self.assertIsNotNone(result_grid_b)
        if result_grid_b is not None:
            self.assertIsNone(result_grid_b.get_cell(5, 1)["module"])
        self.assertEqual(solved_bonus_b, 20.0)
        self.assertEqual(solve_method_b, "Pattern Match")

        # Case C: Partial module set with forced=True also compares full-grid SA against windowed SA
        with patch(
            "src.optimization.core.simulated_annealing",
            return_value=(high_sa_grid, 88.0),
        ) as mock_sa_partial:
            with patch(
                "src.optimization.core.calculate_grid_score",
                side_effect=lambda g, t, **kw: 88.0 if g.get_cell(5, 1)["module"] == "Xa" else 1.5,
            ):
                result_grid_c, _, solved_bonus_c, solve_method_c = optimize_placement(
                    grid,
                    "standard",
                    self.modules,
                    "shield",
                    forced=True,
                    available_modules=["DS", "Xa", "Xb", "Xc"],
                )

        mock_sa_partial.assert_called_once()
        self.assertIsNotNone(result_grid_c)
        if result_grid_c is not None:
            self.assertEqual(result_grid_c.get_cell(5, 1)["module"], "Xa")
        self.assertEqual(solved_bonus_c, 88.0)
        self.assertEqual(solve_method_c, "Forced Full SA")
