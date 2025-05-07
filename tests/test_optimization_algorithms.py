# test_optimization_algorithms.py
import unittest
from unittest.mock import patch, MagicMock, ANY  # <<< Added ANY
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# --- Imports from your project ---
from optimization_algorithms import (
    rotate_pattern,
    mirror_pattern_horizontally,
    mirror_pattern_vertically,
    get_all_unique_pattern_variations,
    count_adjacent_occupied,
    optimize_placement,  # <<< Ensure optimize_placement is imported
    place_all_modules_in_empty_slots,
    find_supercharged_opportunities,
    apply_localized_grid_changes,
    check_all_modules_placed,
    clear_all_modules_of_tech,
)
from grid_utils import Grid
from modules import modules as sample_modules  # <<< Use alias for clarity
from modules import solves as sample_solves  # <<< Use alias for clarity


# --- Test Class ---
class TestOptimizationAlgorithms(unittest.TestCase):

    # --- Merged setUp method ---
    def setUp(self):
        """Set up common test resources."""
        # Existing setup
        self.grid = Grid(4, 3)  # Keep this if other tests use it directly
        self.ship = "standard"
        self.tech = "pulse"  # A tech with a known solve map
        self.modules = sample_modules  # Use the imported modules
        self.player_owned_rewards = ["PC"]  # Example reward

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

    def test_count_adjacent_occupied(self):
        self.grid.set_module(0, 0, "A")
        self.grid.set_tech(0, 0, "other")  # Set tech to avoid confusion
        self.grid.set_module(1, 0, "B")
        self.grid.set_tech(1, 0, "other")
        count = count_adjacent_occupied(self.grid, 0, 1)
        self.assertEqual(count, 1)

    def test_place_all_modules_in_empty_slots(self):
        # Use the actual infra modules for standard ship
        infra_modules = [m for t in sample_modules["standard"]["types"].values() for m in t if m["key"] == "infra"][0][
            "modules"
        ]

        with patch("optimization_algorithms.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = infra_modules
            # Use self.empty_grid which is guaranteed empty
            result_grid = place_all_modules_in_empty_slots(
                self.empty_grid.copy(), self.modules, self.ship, self.tech, self.player_owned_rewards
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

    # def test_find_supercharged_opportunities_opportunity(self):
    #     # Use self.sc_grid which has SC slots
    #     sc_grid_test = self.sc_grid.copy()
    #     # Place a module of the *same* tech outside the best window to ensure it's cleared
    #     sc_grid_test.set_module(3, 2, "IK")
    #     sc_grid_test.set_tech(3, 2, self.tech)
    #     # Place a module of a *different* tech within the window
    #     sc_grid_test.set_module(1, 0, "OTHER")
    #     sc_grid_test.set_tech(1, 0, "other_tech")

    #     result = find_supercharged_opportunities(sc_grid_test, self.modules, self.ship, self.tech)
    #     # Assuming the best window starts at (0,0) for a 4x3 grid with SC at (1,1), (2,1)
    #     self.assertEqual(result, (0, 0))

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

    def test_check_all_modules_placed_all_placed(self):
        # Use a specific tech with a known small number of modules, e.g., rocket
        rocket_tech = "rocket"
        rocket_modules = [
            m for t in sample_modules["standard"]["types"].values() for m in t if m["key"] == rocket_tech
        ][0]["modules"]

        with patch("optimization_algorithms.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = rocket_modules
            grid_all_placed = Grid(2, 2)
            grid_all_placed.set_module(0, 0, "RL")
            grid_all_placed.set_tech(0, 0, rocket_tech)
            grid_all_placed.set_module(0, 1, "LR")
            grid_all_placed.set_tech(0, 1, rocket_tech)
            result = check_all_modules_placed(grid_all_placed, self.modules, self.ship, rocket_tech)
            self.assertTrue(result)

    def test_check_all_modules_placed_not_all_placed(self):
        rocket_tech = "rocket"
        rocket_modules = [
            m for t in sample_modules["standard"]["types"].values() for m in t if m["key"] == rocket_tech
        ][0]["modules"]

        with patch("optimization_algorithms.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = rocket_modules
            grid_not_all_placed = Grid(2, 2)
            grid_not_all_placed.set_module(0, 0, "RL")
            grid_not_all_placed.set_tech(0, 0, rocket_tech)
            # Missing "LR"
            result = check_all_modules_placed(grid_not_all_placed, self.modules, self.ship, rocket_tech)
            self.assertFalse(result)

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
            optimize_placement(full_grid, self.ship, sample_modules, self.tech, self.player_owned_rewards)

    @patch("optimization_algorithms.filter_solves")
    @patch("optimization_algorithms.place_all_modules_in_empty_slots")
    @patch("optimization_algorithms.calculate_grid_score")
    def test_optimize_no_solve_map_available(self, mock_calculate_score, mock_place_all, mock_filter_solves):
        """Test behavior when no solve map is found after filtering."""
        # Arrange: Mock filter_solves to return an empty dict (no solve found)
        mock_filter_solves.return_value = {}
        # Arrange: Mock place_all_modules to return a grid
        placed_grid = self.empty_grid.copy()
        # Simulate placing some modules for score calculation
        placed_grid.set_module(0, 0, "IK")
        placed_grid.set_tech(0, 0, self.tech)
        mock_place_all.return_value = placed_grid
        # Arrange: Mock calculate_grid_score
        mock_calculate_score.return_value = 5.0

        # Act
        result_grid, percentage, solved_bonus, solve_method= optimize_placement(
            self.empty_grid, self.ship, sample_modules, self.tech, self.player_owned_rewards
        )

        # Assert
        mock_filter_solves.assert_called_once_with(
            sample_solves, self.ship, sample_modules, self.tech, self.player_owned_rewards
        )
        mock_place_all.assert_called_once_with(ANY, sample_modules, self.ship, self.tech, self.player_owned_rewards)
        # Check that the grid returned by place_all is used for scoring and returned
        mock_calculate_score.assert_called_once_with(placed_grid, self.tech)
        self.assertEqual(result_grid, placed_grid)
        self.assertEqual(solved_bonus, 5.0)
        self.assertEqual(percentage, 100.0)  # Percentage is 100 if score > 0 and solve_score is 0

    @patch("optimization_algorithms.apply_pattern_to_grid")
    @patch("optimization_algorithms.simulated_annealing")
    @patch("optimization_algorithms.find_supercharged_opportunities")
    @patch("optimization_algorithms.check_all_modules_placed")
    @patch("optimization_algorithms.calculate_grid_score")
    def test_optimize_solve_map_no_pattern_fits_returns_indicator_when_not_forced(
        self, mock_calculate_score, mock_check_placed, mock_find_opp, mock_sa, mock_apply_pattern
    ):
        """Test returns 'Pattern No Fit' when solve map exists, no pattern fits, and not forced."""
        # Arrange: Assume filter_solves finds a map (using real filter_solves for 'pulse').
        # Arrange: Mock apply_pattern_to_grid to always return None (no fit)
        mock_apply_pattern.return_value = (None, 0)
        # SA and other mocks should NOT be called in this path if not forced.

        # Act: Call with forced=False (default)
        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.empty_grid, self.ship, sample_modules, self.tech, self.player_owned_rewards, forced=False
        )

        # Assert
        mock_apply_pattern.assert_called() # Pattern application was attempted
        mock_sa.assert_not_called()       # SA should NOT be called
        mock_find_opp.assert_not_called() # No refinement opportunity finding
        mock_check_placed.assert_not_called()
        mock_calculate_score.assert_not_called()

        self.assertIsNone(result_grid)
        self.assertEqual(percentage, 0.0)
        self.assertEqual(solved_bonus, 0.0)
        self.assertEqual(solve_method, "Pattern No Fit")


    @patch("optimization_algorithms.apply_pattern_to_grid")
    @patch("optimization_algorithms.simulated_annealing")  # Mock the initial SA
    @patch("optimization_algorithms.find_supercharged_opportunities")
    @patch("optimization_algorithms.check_all_modules_placed")
    @patch("optimization_algorithms.calculate_grid_score")
    def test_optimize_solve_map_no_pattern_fits_falls_back_to_sa_when_forced(
        self, mock_calculate_score, mock_check_placed, mock_find_opp, mock_sa, mock_apply_pattern
    ):
        """Test fallback to initial SA when solve map exists, no pattern fits, and forced=True."""
        # Arrange: Assume filter_solves finds a map (using real filter_solves).
        # Arrange: Mock apply_pattern_to_grid to always return None (no fit)
        mock_apply_pattern.return_value = (None, 0)
        # Arrange: Mock simulated_annealing (the initial fallback one)
        initial_sa_grid = self.empty_grid.copy()
        # Simulate SA placing modules
        initial_sa_grid.set_module(0, 1, "PE")
        initial_sa_grid.set_tech(0, 1, self.tech)
        mock_sa.return_value = (initial_sa_grid, 10.0)

        # Arrange: Mock calculate_grid_score for the two calls
        # 1. Before refinement check, 2. For final result calculation
        mock_calculate_score.side_effect = [10.0, 10.0]  # <<< Use side_effect

        # Arrange: Mock find_supercharged_opportunities to return None
        mock_find_opp.return_value = None
        # Arrange: Mock check_all_modules_placed to return True
        mock_check_placed.return_value = True

        # Act
        result_grid, percentage, solved_bonus, solve_method = optimize_placement( # Call with forced=True
            self.empty_grid, self.ship, sample_modules, self.tech, self.player_owned_rewards, forced=True
        )

        # Assert
        mock_apply_pattern.assert_called()  # Check pattern application was attempted
        mock_sa.assert_called_once()  # Check that initial SA was called
        mock_find_opp.assert_called_once()  # Refinement check still happens
        mock_check_placed.assert_called_once()
        # Expect 2 calls: before refinement check and for final result
        self.assertEqual(mock_calculate_score.call_count, 2)  # <<< Correct assertion
        self.assertEqual(result_grid, initial_sa_grid)
        self.assertEqual(solve_method, "Forced Initial SA (No Pattern Fit)")
        self.assertEqual(solved_bonus, 10.0)


# --- Run Tests ---
if __name__ == "__main__":
    # Use the standard unittest main runner
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
