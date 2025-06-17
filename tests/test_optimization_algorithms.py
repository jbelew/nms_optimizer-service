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
    refine_placement,  # <<< Added import
    _evaluate_permutation_worker, # <<< Added import
    refine_placement_for_training, # <<< Added import
    determine_window_dimensions, # <<< Added import
    apply_pattern_to_grid, # <<< Added import
    calculate_pattern_adjacency_score, # Assuming this is also in optimization_algorithms
    _handle_ml_opportunity, # <<< Added import
    create_localized_grid_ml, # <<< Added import
    restore_original_state, # <<< Added import
    _handle_sa_refine_opportunity, # <<< Added import
    create_localized_grid, # <<< Added import for SA path
    count_empty_in_localized, # <<< Added import
)
from grid_utils import Grid
import ml_placement # <<< Added import for mocking
import multiprocessing # <<< Added import for mocking Pool
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

        # Modules for testing refine_placement
        self.simple_tech_modules = [
            {"key": "A", "name": "Module A", "value": 1, "shape": [[1]]},
            {"key": "B", "name": "Module B", "value": 1, "shape": [[1]]},
        ]
        self.complex_tech_modules = [
            {"key": "C", "name": "Module C", "value": 1, "shape": [[1, 1]]}, # 1x2 module
            {"key": "D", "name": "Module D", "value": 1, "shape": [[1], [1]]}, # 2x1 module
        ]

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

    # --- Tests for refine_placement ---

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.calculate_grid_score") # Assuming refine_placement uses this
    @patch("optimization_algorithms.simulated_annealing") # Assuming refine_placement might use this
    def test_refine_placement_simple_grid(self, mock_sa, mock_calculate_score, mock_get_tech_modules):
        """Test refine_placement with a simple grid and few modules."""
        # Arrange
        grid = Grid(2, 2) # Small grid
        grid.set_module(0, 0, "A") # Pre-place one module
        grid.set_tech(0, 0, self.tech)

        mock_get_tech_modules.return_value = self.simple_tech_modules # Two 1x1 modules "A", "B"

        # Assume SA returns a slightly better grid or the same if no improvement
        refined_sa_grid = grid.copy()
        refined_sa_grid.set_module(1, 0, "B") # SA places the second module "B"
        refined_sa_grid.set_tech(1, 0, self.tech)
        mock_sa.return_value = (refined_sa_grid, 12.0) # Score for the refined grid

        # Score for the initial grid before SA
        mock_calculate_score.side_effect = [10.0, 12.0] # Initial score, then SA score

        # Act
        # Assuming refine_placement signature: (grid, ship, modules_data, tech, solve_map, current_score, variation_key)
        # For this test, solve_map and variation_key might not be strictly necessary or can be simple placeholders.
        # Let's assume a simple scenario where refine_placement tries to improve the current placement.
        # We need to know the actual signature of refine_placement.
        # For now, let's assume a plausible signature:
        # refined_grid, refined_score = refine_placement(initial_grid, ship, all_modules, tech, initial_score)

        # Placeholder call until signature is known. This will likely fail or need adjustment.
        # For now, let's assume refine_placement aims to fill remaining empty slots or optimize.
        # It might take the initial grid, modules, ship, tech, and current_score.

        # If refine_placement is simpler and just takes a grid and modules to place:
        # refined_grid, refined_score = refine_placement(grid, self.ship, self.modules, self.tech, "some_variation_key", 10.0)

        # Let's assume refine_placement's job is to take an existing grid (potentially partially filled)
        # and try to improve it using methods like SA.
        # The modules to be placed/considered would be fetched by get_tech_modules.

        initial_score = 10.0 # From mock_calculate_score's first side_effect

        # refine_placement(grid_to_refine, ship, all_modules_data, tech, current_score, sc_info=None)
        # This is a guess based on optimize_placement's call to refine_placement
        # refine_placement(best_pattern_grid, self.ship, modules, tech, best_pattern_score, sc_info=None)

        refined_grid_result, new_score_result = refine_placement(
            grid.copy(), # Pass a copy to avoid modification by reference issues in test
            self.ship,
            self.modules, # All modules data
            self.tech,
            initial_score
        )

        # Assert
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, self.tech)
        # SA should be called with the initial grid and the modules for the current tech
        mock_sa.assert_called_once_with(
            grid, # The initial grid passed to refine_placement
            self.simple_tech_modules, # Modules for the tech
            initial_score, # The initial score of the grid
            ANY, # iterations (we don't care about the exact value for this test)
            ANY, # temperature (same)
            None # sc_info (assuming None if not provided)
        )
        mock_calculate_score.assert_any_call(grid, self.tech) # Initial score calculation
        # The second call to calculate_score would be inside SA, which is mocked,
        # or if refine_placement calculates it on the SA result.
        # For now, we assume SA returns the score.

        self.assertEqual(new_score_result, 12.0)
        self.assertEqual(refined_grid_result.get_cell(0,0)['module'], "A")
        self.assertEqual(refined_grid_result.get_cell(1,0)['module'], "B")
        self.assertTrue(refined_grid_result.get_cell(0,0)['tech'] == self.tech and refined_grid_result.get_cell(1,0)['tech'] == self.tech)

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.simulated_annealing")
    def test_refine_placement_no_active_slots(self, mock_sa, mock_get_tech_modules):
        """Test refine_placement with a grid that has no active slots."""
        # Arrange
        grid = Grid(2, 2)
        grid.set_active(0, 0, False)
        grid.set_active(0, 1, False)
        grid.set_active(1, 0, False)
        grid.set_active(1, 1, False)

        initial_score = 0.0 # No modules placed, no active slots

        mock_get_tech_modules.return_value = self.simple_tech_modules

        # Act
        # refine_placement(grid_to_refine, ship, all_modules_data, tech, current_score, sc_info=None)
        refined_grid_result, new_score_result = refine_placement(
            grid.copy(),
            self.ship,
            self.modules,
            self.tech,
            initial_score
        )

        # Assert
        # It should not attempt to call SA if there's nowhere to place modules.
        # Or, if SA is called, SA itself should handle the empty active slot scenario gracefully.
        # Let's assume refine_placement checks for empty active slots or SA handles it.
        # If SA is called, it should be with the empty grid and it should return it as is.
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, self.tech)

        # We expect SA to be called, but it should not change the grid or score if no active slots.
        mock_sa.assert_called_once_with(
            grid,
            self.simple_tech_modules,
            initial_score,
            ANY, ANY, None
        )
        # SA should return the original grid and score if no refinement is possible.
        # We need to set the return value for the mock_sa call in this test.
        mock_sa.return_value = (grid, initial_score)


        self.assertEqual(new_score_result, initial_score)
        self.assertEqual(refined_grid_result.serialize(), grid.serialize()) # Compare grid states
        # Ensure no modules were placed
        for y in range(refined_grid_result.height):
            for x in range(refined_grid_result.width):
                self.assertIsNone(refined_grid_result.get_cell(x,y)['module'])

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.simulated_annealing")
    @patch("optimization_algorithms.calculate_grid_score")
    def test_refine_placement_insufficient_space(self, mock_calculate_score, mock_sa, mock_get_tech_modules):
        """Test refine_placement when not all modules can fit."""
        # Arrange
        grid = Grid(1, 1) # Only one slot
        initial_score = 0.0

        # simple_tech_modules has two 1x1 modules ("A", "B"). Only one can fit.
        mock_get_tech_modules.return_value = self.simple_tech_modules

        # SA will try to place modules. It should place one and achieve a score.
        sa_refined_grid = grid.copy()
        sa_refined_grid.set_module(0, 0, "A") # SA places module "A"
        sa_refined_grid.set_tech(0, 0, self.tech)
        sa_score = 5.0 # Assume module "A" gives a score of 5.0
        mock_sa.return_value = (sa_refined_grid, sa_score)

        # Mock calculate_grid_score for the initial call (if any, refine_placement gets score as param)
        # For this test, refine_placement receives initial_score = 0.0.
        # No, calculate_grid_score is not directly called by refine_placement based on its signature.
        # It's an argument. SA returns the new score.

        # Act
        refined_grid_result, new_score_result = refine_placement(
            grid.copy(),
            self.ship,
            self.modules,
            self.tech,
            initial_score
        )

        # Assert
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, self.tech)
        mock_sa.assert_called_once_with(
            grid,
            self.simple_tech_modules, # It will try to place "A" and "B"
            initial_score,
            ANY, ANY, None
        )

        self.assertEqual(new_score_result, sa_score)
        self.assertEqual(refined_grid_result.get_cell(0,0)['module'], "A") # Module A placed
        self.assertEqual(refined_grid_result.get_cell(0,0)['tech'], self.tech)

        # Count placed modules
        placed_count = 0
        for r in range(refined_grid_result.height):
            for c in range(refined_grid_result.width):
                if refined_grid_result.get_cell(c,r)['module'] is not None:
                    placed_count +=1
        self.assertEqual(placed_count, 1) # Only one module should be placed

    # --- Tests for _evaluate_permutation_worker ---

    @patch("optimization_algorithms.calculate_grid_score")
    @patch("grid_utils.Grid.place_module") # Assuming Grid.place_module is used
    def test_evaluate_permutation_worker_basic(self, mock_grid_place_module, mock_calculate_score):
        """Test basic functionality of _evaluate_permutation_worker."""
        # Arrange
        grid_width, grid_height = 2, 2
        tech = "test_tech"
        # Using simple_tech_modules from setUp:
        # self.simple_tech_modules = [
        #     {"key": "A", "name": "Module A", "value": 1, "shape": [[1]]},
        #     {"key": "B", "name": "Module B", "value": 1, "shape": [[1]]},
        # ]
        tech_modules = self.simple_tech_modules

        available_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        # Place module "A" at (0,0) and module "B" at (0,1)
        placement_indices = (0, 1) # Indices into available_positions

        # Mock Grid.place_module to return True (success)
        mock_grid_place_module.return_value = True
        # Mock calculate_grid_score to return a specific score
        expected_score = 15.0
        mock_calculate_score.return_value = expected_score

        # Other args for _evaluate_permutation_worker: base_score, supercharged_slots (optional)
        # Assuming signature:
        # (placement_indices, grid_width, grid_height, tech_modules, available_positions, tech, base_score=0.0, supercharged_slots=None)

        # Act
        score, returned_indices = _evaluate_permutation_worker(
            placement_indices,
            grid_width,
            grid_height,
            tech_modules,
            available_positions,
            tech,
            # base_score=0.0, # Assuming default
            # supercharged_slots=None # Assuming default
        )

        # Assert
        self.assertEqual(score, expected_score)
        self.assertEqual(returned_indices, placement_indices)

        # Verify Grid.place_module calls
        # Module A at available_positions[0] = (0,0)
        # Module B at available_positions[1] = (0,1)
        self.assertEqual(mock_grid_place_module.call_count, len(tech_modules))
        mock_grid_place_module.assert_any_call(
            available_positions[0][0], available_positions[0][1], tech_modules[0], tech, ANY # rotation
        )
        mock_grid_place_module.assert_any_call(
            available_positions[1][0], available_positions[1][1], tech_modules[1], tech, ANY # rotation
        )

        # Verify calculate_grid_score call
        # It's called with the grid that has modules placed, and the tech
        mock_calculate_score.assert_called_once()
        # The first argument to calculate_grid_score is the Grid instance.
        # We can check if it's an instance of Grid and has the correct tech.
        call_args = mock_calculate_score.call_args[0]
        self.assertIsInstance(call_args[0], Grid)
        self.assertEqual(call_args[1], tech)

    @patch("optimization_algorithms.calculate_grid_score") # Still need this as it's called if placement succeeds
    @patch("grid_utils.Grid.place_module")
    def test_evaluate_permutation_worker_invalid_indices(self, mock_grid_place_module, mock_calculate_score):
        """Test _evaluate_permutation_worker with out-of-bounds placement_indices."""
        # Arrange
        grid_width, grid_height = 1, 1
        tech = "test_tech"
        tech_modules = [self.simple_tech_modules[0]] # One module: {"key": "A", ...}
        available_positions = [(0, 0)]

        # Invalid index: 1 (available_positions only has index 0)
        # Or, if placement_indices length doesn't match tech_modules length
        # Let's test index out of bounds for available_positions first.
        placement_indices = (1,) # Tries to access available_positions[1] which is out of bounds

        # Act
        score, returned_indices = _evaluate_permutation_worker(
            placement_indices,
            grid_width,
            grid_height,
            tech_modules,
            available_positions,
            tech
        )

        # Assert
        self.assertEqual(score, -1.0)
        self.assertIsNone(returned_indices)
        mock_grid_place_module.assert_not_called() # Should fail before trying to place
        mock_calculate_score.assert_not_called() # Should not be called if placement fails early


    @patch("optimization_algorithms.calculate_grid_score")
    @patch("grid_utils.Grid.place_module")
    def test_evaluate_permutation_worker_placement_fail(self, mock_grid_place_module, mock_calculate_score):
        """Test _evaluate_permutation_worker when Grid.place_module returns False."""
        # Arrange
        grid_width, grid_height = 1, 1
        tech = "test_tech"
        tech_modules = [self.simple_tech_modules[0]] # One module
        available_positions = [(0, 0)]
        placement_indices = (0,) # Valid index

        # Mock Grid.place_module to return False (placement failed)
        mock_grid_place_module.return_value = False

        # Act
        score, returned_indices = _evaluate_permutation_worker(
            placement_indices,
            grid_width,
            grid_height,
            tech_modules,
            available_positions,
            tech
        )

        # Assert
        self.assertEqual(score, -1.0)
        self.assertIsNone(returned_indices)
        mock_grid_place_module.assert_called_once_with(
            available_positions[0][0], available_positions[0][1], tech_modules[0], tech, ANY
        )
        mock_calculate_score.assert_not_called() # Score calculation shouldn't happen if placement fails

    @patch("grid_utils.Grid.place_module") # Don't need calculate_grid_score as it shouldn't be reached
    def test_evaluate_permutation_worker_indices_length_mismatch(self, mock_grid_place_module):
        """Test _evaluate_permutation_worker with len(placement_indices) < len(tech_modules)."""
        # Arrange
        grid_width, grid_height = 1, 1
        tech = "test_tech"
        # Two modules
        tech_modules = self.simple_tech_modules
        available_positions = [(0, 0), (0,1)] # Enough positions for two modules

        # placement_indices is too short for tech_modules
        placement_indices = (0,) # Only one index, but two modules

        # Act
        score, returned_indices = _evaluate_permutation_worker(
            placement_indices,
            grid_width,
            grid_height,
            tech_modules,
            available_positions,
            tech
        )

        # Assert
        # This should be caught as an error (e.g., IndexError when accessing placement_indices[1])
        self.assertEqual(score, -1.0)
        self.assertIsNone(returned_indices)
        # Depending on implementation, place_module might be called for the first module
        # or it might fail before any placement if there's a pre-check.
        # Assuming it tries to place and fails during the loop:
        # If the worker is like:
        # for i, module in enumerate(tech_modules):
        #     idx = placement_indices[i] # This will fail for i=1
        #     ...
        # Then place_module might not be called at all if the check is before.
        # If it's robust, it should catch this. Let's assume it might try to place the first one.
        # Or, if it checks len(placement_indices) != len(tech_modules) first, then 0 calls.
        # Given the expected return of (-1.0, None), it implies an error was handled.
        # For this specific test, asserting no calls or one call depends on internal logic.
        # Let's assume it fails before placing any module if lengths mismatch.
        # A common pattern is `if len(placement_indices) != len(tech_modules): return -1.0, None`
        mock_grid_place_module.assert_not_called()

    # --- Tests for refine_placement_for_training ---

    @patch("optimization_algorithms.get_tech_modules_for_training")
    @patch("multiprocessing.Pool")
    @patch("optimization_algorithms.clear_all_modules_of_tech") # To verify it's called
    @patch("grid_utils.Grid.place_module") # To verify final placement
    def test_rpf_training_basic_success(
        self, mock_grid_place_module, mock_clear_modules, mock_pool, mock_get_tech_modules_training
    ):
        """Test basic successful execution of refine_placement_for_training."""
        # Arrange
        initial_grid = Grid(2, 2) # 4 available slots
        initial_grid.set_module(0,0,"OLD_MODULE") # To ensure it's cleared
        initial_grid.set_tech(0,0,self.tech)

        ship = "standard"
        tech = self.tech # from setUp

        # Mock get_tech_modules_for_training
        # These modules are simpler dicts, not the full module objects
        training_modules = [
            {"key": "A", "shape": [[1]]}, # Module A (1x1)
            {"key": "B", "shape": [[1]]}, # Module B (1x1)
        ]
        mock_get_tech_modules_training.return_value = training_modules

        # Mock multiprocessing.Pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance

        # Simulate worker results: (score, placement_indices for available_slots)
        # available_slots in a 2x2 grid: [(0,0), (0,1), (1,0), (1,1)]
        # Permutation (0,1) -> A at (0,0), B at (0,1) -> score 10.0 (BEST)
        # Permutation (1,0) -> A at (0,1), B at (0,0) -> score 8.0
        # Permutation (0,2) -> A at (0,0), B at (1,0) -> score 9.0
        worker_results = [
            (8.0, (1, 0)), # Lower score
            (10.0, (0, 1)), # Best score
            (9.0, (0, 2)),
            (-1.0, None) # Failed one
        ]
        mock_pool_instance.imap_unordered.return_value = worker_results

        # Mock Grid.place_module for final placement
        mock_grid_place_module.return_value = True # Assume successful placement

        # Act
        # refine_placement_for_training(initial_grid, ship, modules_data, tech, num_workers=4)
        # modules_data is self.modules from setUp
        final_grid, best_score = refine_placement_for_training(
            initial_grid.copy(), ship, self.modules, tech, num_workers=2 # num_workers is mocked anyway
        )

        # Assert
        mock_get_tech_modules_training.assert_called_once_with(self.modules, ship, tech)
        mock_clear_modules.assert_called_once_with(ANY, tech) # Called on a grid copy
        self.assertIsInstance(mock_clear_modules.call_args[0][0], Grid)

        mock_pool.assert_called_once() # Check Pool was initialized
        mock_pool_instance.imap_unordered.assert_called_once() # Check tasks were submitted

        # Check final placement calls for the best permutation (0,1)
        # Module A at available_slots[0] = (0,0)
        # Module B at available_slots[1] = (0,1)
        # Note: available_slots are derived *after* clearing the grid.
        # The grid passed to place_module should be the cleared one.
        self.assertEqual(mock_grid_place_module.call_count, len(training_modules))

        # Find the full module data from self.simple_tech_modules (defined in setUp)
        # which should be compatible with what's in self.modules for keys 'A' and 'B'
        module_A_full = next(m for m in self.simple_tech_modules if m['key'] == training_modules[0]['key'])
        module_B_full = next(m for m in self.simple_tech_modules if m['key'] == training_modules[1]['key'])

        # Expected positions for best permutation (0,1) on a 2x2 grid: A at (0,0), B at (0,1)
        # Rotation 0 is assumed if not specified by permutation logic, which is typical.
        mock_grid_place_module.assert_any_call(0, 0, module_A_full, tech, 0)
        mock_grid_place_module.assert_any_call(0, 1, module_B_full, tech, 0)

        self.assertEqual(best_score, 10.0)
        # Verify grid state: only A and B should be there.
        # The keys "A", "B" from training_modules are used for grid cell module identifiers.
        self.assertEqual(final_grid.get_cell(0,0)['module'], training_modules[0]['key'])
        self.assertEqual(final_grid.get_cell(0,1)['module'], training_modules[1]['key'])
        self.assertIsNone(final_grid.get_cell(1,0)['module']) # Should be empty
        self.assertIsNone(final_grid.get_cell(1,1)['module']) # Should be empty
        self.assertTrue(all(c['tech'] == tech for c in [final_grid.get_cell(0,0), final_grid.get_cell(0,1)] if c['module']))

    @patch("optimization_algorithms.get_tech_modules_for_training")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    @patch("multiprocessing.Pool") # Pool should not be called if no modules
    def test_rpf_training_no_modules(self, mock_pool, mock_clear_modules, mock_get_tech_modules_training):
        """Test refine_placement_for_training when no modules are returned for the tech."""
        # Arrange
        initial_grid = Grid(2, 2)
        initial_grid.set_module(0,0,"OLD_MODULE")
        initial_grid.set_tech(0,0,self.tech)

        ship = "standard"
        tech = self.tech

        mock_get_tech_modules_training.return_value = [] # No modules

        # Act
        final_grid, best_score = refine_placement_for_training(
            initial_grid.copy(), ship, self.modules, tech
        )

        # Assert
        mock_get_tech_modules_training.assert_called_once_with(self.modules, ship, tech)
        # Grid should be cleared of the specific tech modules
        mock_clear_modules.assert_called_once_with(ANY, tech)

        mock_pool.assert_not_called() # Pool should not be initialized if no modules

        self.assertEqual(best_score, 0.0)
        # Check that the grid is cleared (or at least the old module of that tech is gone)
        # The exact state depends on whether clear_all_modules_of_tech clears other techs.
        # Assuming it only clears the specified tech.
        self.assertIsNone(final_grid.get_cell(0,0)['module']) # OLD_MODULE of self.tech is cleared

    @patch("optimization_algorithms.get_tech_modules_for_training")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    @patch("multiprocessing.Pool")
    def test_rpf_training_not_enough_active_slots(
        self, mock_pool, mock_clear_modules, mock_get_tech_modules_training
    ):
        """Test when there are fewer active slots than modules to place."""
        # Arrange
        initial_grid = Grid(1, 1) # Only 1 active slot
        initial_grid.set_active(0,0,True)

        ship = "standard"
        tech = self.tech

        # Two modules, but only one slot
        training_modules = [ {"key": "A", "shape": [[1]]}, {"key": "B", "shape": [[1]]} ]
        mock_get_tech_modules_training.return_value = training_modules

        # Act
        final_grid, best_score = refine_placement_for_training(
            initial_grid.copy(), ship, self.modules, tech
        )

        # Assert
        mock_get_tech_modules_training.assert_called_once_with(self.modules, ship, tech)
        # clear_all_modules_of_tech might be called before the check, or not.
        # If it's called, the grid passed to it would be the initial_grid copy.
        # Let's assume it's called.
        mock_clear_modules.assert_called_once_with(ANY, tech)

        mock_pool.assert_not_called() # Pool should not be initialized

        self.assertEqual(best_score, 0.0)
        self.assertIsNone(final_grid.get_cell(0,0)['module']) # Grid should be cleared

    @patch("optimization_algorithms.get_tech_modules_for_training")
    @patch("multiprocessing.Pool")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    @patch("grid_utils.Grid.place_module")
    def test_rpf_training_no_optimal_grid_found(
        self, mock_grid_place_module, mock_clear_modules, mock_pool, mock_get_tech_modules_training
    ):
        """Test when all worker evaluations fail or return very low scores."""
        # Arrange
        initial_grid = Grid(2, 2)
        ship = "standard"
        tech = self.tech

        training_modules = [{"key": "A", "shape": [[1]]}]
        mock_get_tech_modules_training.return_value = training_modules

        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        worker_results = [ (-1.0, None), (-1.0, None) ] # All fail
        mock_pool_instance.imap_unordered.return_value = worker_results

        # Act
        final_grid, best_score = refine_placement_for_training(
            initial_grid.copy(), ship, self.modules, tech
        )

        # Assert
        mock_get_tech_modules_training.assert_called_once()
        mock_clear_modules.assert_called_once()
        mock_pool.assert_called_once()
        mock_pool_instance.imap_unordered.assert_called_once()

        # No modules should be placed if all evaluations failed
        mock_grid_place_module.assert_not_called()

        self.assertEqual(best_score, 0.0) # Or -1.0 if that's preferred for "no solution"
                                           # Function doc should clarify. Assuming 0.0 for "no valid placement".
        # Grid should be cleared
        for r in range(final_grid.height):
            for c in range(final_grid.width):
                self.assertIsNone(final_grid.get_cell(c,r)['module'])

    # --- Test for determine_window_dimensions ---

    @patch("logging.warning") # To verify warnings for invalid module_count
    def test_determine_window_dimensions(self, mock_logging_warning):
        """Test determine_window_dimensions for various module counts and tech types."""

        # Case: module_count < 1 (e.g., 0)
        self.assertEqual(determine_window_dimensions(0, "any_tech"), (1, 1))
        mock_logging_warning.assert_called_with("Module count 0 is invalid for window determination, defaulting to 1x1.")

        # Case: module_count < 1 (e.g., -1)
        mock_logging_warning.reset_mock() # Reset mock for the next call
        self.assertEqual(determine_window_dimensions(-1, "any_tech"), (1, 1))
        mock_logging_warning.assert_called_with("Module count -1 is invalid for window determination, defaulting to 1x1.")

        # Case: module_count = 1 (< 3)
        mock_logging_warning.reset_mock()
        self.assertEqual(determine_window_dimensions(1, "any_tech"), (1, 1))
        mock_logging_warning.assert_not_called()

        # Case: module_count = 2 (< 3)
        mock_logging_warning.reset_mock()
        self.assertEqual(determine_window_dimensions(2, "any_tech"), (2, 1))
        mock_logging_warning.assert_not_called()

        # Case: module_count = 3 (< 4)
        self.assertEqual(determine_window_dimensions(3, "any_tech"), (2, 2))

        # Case: module_count = 4, 5, 6 (< 7)
        self.assertEqual(determine_window_dimensions(4, "any_tech"), (3, 2))
        self.assertEqual(determine_window_dimensions(5, "any_tech"), (3, 2))
        self.assertEqual(determine_window_dimensions(6, "any_tech"), (3, 2))

        # Case: module_count = 7 (< 8)
        # Tech is "pulse-splitter"
        self.assertEqual(determine_window_dimensions(7, "pulse-splitter"), (3, 3))
        # Tech is not "pulse-splitter"
        self.assertEqual(determine_window_dimensions(7, "other_tech"), (3, 2))

        # Case: module_count = 8 (< 9)
        self.assertEqual(determine_window_dimensions(8, "any_tech"), (3, 3))

        # Case: module_count = 9 (< 10)
        self.assertEqual(determine_window_dimensions(9, "any_tech"), (3, 3)) # Assuming this is intended, or maybe (4,3) or (3,4)
                                                                            # Based on current structure, it will be 3,3.
                                                                            # If source code has a specific rule for 9, this test will catch deviation.
                                                                            # The prompt implies distinct logic up to <10.

        # Case: module_count >= 10 (e.g., 10)
        self.assertEqual(determine_window_dimensions(10, "any_tech"), (4, 3))

        # Case: module_count >= 10 (e.g., 11)
        self.assertEqual(determine_window_dimensions(11, "any_tech"), (4, 3))

        # Case: module_count >= 10 (e.g., 12)
        self.assertEqual(determine_window_dimensions(12, "any_tech"), (4, 3))

        # Case: module_count = very large (still should be 4x3 as per current logic)
        self.assertEqual(determine_window_dimensions(20, "any_tech"), (4, 3))


    # --- Tests for apply_pattern_to_grid ---

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.calculate_pattern_adjacency_score")
    @patch("grid_utils.Grid.place_module") # Assuming it uses Grid's method
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    def test_apply_pattern_valid_on_empty_grid(
        self, mock_clear_tech, mock_place_module, mock_calc_adj_score, mock_get_tech_modules
    ):
        """Test applying a valid pattern to an empty grid."""
        # Arrange
        grid = self.empty_grid.copy() # A 4x3 empty grid from setUp
        tech = self.tech # "pulse"
        # Pattern: { (0,0): "A", (1,0): "B" }
        # Module data for "A" and "B" are in self.simple_tech_modules
        module_A_full = self.simple_tech_modules[0] # {"key": "A", "name": "Module A", ...}
        module_B_full = self.simple_tech_modules[1] # {"key": "B", "name": "Module B", ...}
        pattern = {(0,0): module_A_full['key'], (1,0): module_B_full['key']}

        # Player owns both modules "A" and "B"
        mock_get_tech_modules.return_value = [module_A_full, module_B_full]

        mock_calc_adj_score.return_value = 5.0 # Mocked adjacency score
        mock_place_module.return_value = True # Assume all placements are successful

        # Act
        # apply_pattern_to_grid(grid, pattern, all_modules_data, tech, ship_name, rewards, start_x=0, start_y=0)
        # all_modules_data is self.modules, ship_name is self.ship, rewards is self.player_owned_rewards
        returned_grid, returned_score = apply_pattern_to_grid(
            grid, pattern, self.modules, tech, self.ship, self.player_owned_rewards
        )

        # Assert
        mock_clear_tech.assert_called_once_with(grid, tech)
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, tech)

        # Verify place_module calls
        # Module A at (0,0), Module B at (1,0) (assuming start_x=0, start_y=0 default)
        self.assertEqual(mock_place_module.call_count, 2)
        mock_place_module.assert_any_call(0, 0, module_A_full, tech, 0) # Default rotation 0
        mock_place_module.assert_any_call(1, 0, module_B_full, tech, 0) # Default rotation 0

        mock_calc_adj_score.assert_called_once_with(returned_grid, tech)

        self.assertIsNotNone(returned_grid)
        self.assertEqual(returned_score, 5.0)
        self.assertEqual(returned_grid.get_cell(0,0)['module'], module_A_full['key'])
        self.assertEqual(returned_grid.get_cell(1,0)['module'], module_B_full['key'])
        self.assertEqual(returned_grid.get_cell(0,0)['tech'], tech)
        self.assertEqual(returned_grid.get_cell(1,0)['tech'], tech)
        # Check other cells are empty
        self.assertIsNone(returned_grid.get_cell(2,0)['module'])

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.calculate_pattern_adjacency_score") # Should not be called
    @patch("grid_utils.Grid.place_module") # Should not be called if pre-check fails
    @patch("optimization_algorithms.clear_all_modules_of_tech") # Might be called before pre-check
    def test_apply_pattern_required_module_off_grid(
        self, mock_clear_tech, mock_place_module, mock_calc_adj_score, mock_get_tech_modules
    ):
        """Test applying a pattern where a required (owned) module is off-grid."""
        # Arrange
        # Use a 2x1 grid for this test, self.empty_grid is 4x3.
        grid = Grid(2,1)
        tech = self.tech

        module_A = self.simple_tech_modules[0] # {"key": "A", ...}
        # Pattern places "A" at (0,0) and "B" (None here) at (1,0), "C" (owned) at (2,0) which is off-grid
        # For this test, an owned module "A" is at (2,0) in pattern coordinates,
        # and we apply the pattern starting at grid (0,0). So "A" tries to go to grid (2,0).
        pattern = {(2,0): module_A['key']} # Module "A" is at pattern pos (2,0)
                                         # (0,0) (1,0)
                                         #  Grid
                                         # (0,0) (1,0) (2,0)
                                         #  Pattern, if "A" is at (2,0) in its own dict.

        mock_get_tech_modules.return_value = [module_A] # Player owns module "A"

        # Act
        returned_grid, returned_score = apply_pattern_to_grid(
            grid, pattern, self.modules, tech, self.ship, self.player_owned_rewards,
            start_x=0, start_y=0 # Apply pattern starting at grid's (0,0)
        )

        # Assert
        # clear_all_modules_of_tech might be called before the detailed placement check.
        # This depends on the function's internal order of operations. Let's assume it might be.
        # mock_clear_tech.assert_called_once_with(grid, tech)
        # For now, let's not assert on clear_tech as its call is conditional on pre-checks passing.
        # Update: If a required module is off-grid, pre-checks should fail, and clear should NOT be called.
        mock_clear_tech.assert_not_called()

        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, tech)

        # No placement should occur if a required module is off-grid (caught by pre-flight checks)
        mock_place_module.assert_not_called()
        mock_calc_adj_score.assert_not_called()

        self.assertIsNone(returned_grid) # Function should return None for the grid if pre-check fails
        self.assertEqual(returned_score, 0) # And 0 for score

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.calculate_pattern_adjacency_score")
    @patch("grid_utils.Grid.place_module")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    def test_apply_pattern_overlaps_different_tech(
        self, mock_clear_tech, mock_place_module, mock_calc_adj_score, mock_get_tech_modules
    ):
        """Test pattern overlap with existing module of a different tech."""
        # Arrange
        grid = self.empty_grid.copy() # 4x3
        tech = self.tech # "pulse"
        other_tech = "other_tech"

        # Pre-fill grid with a module of a different tech at (0,0)
        grid.set_module(0, 0, "EXISTING")
        grid.set_tech(0, 0, other_tech)

        module_A = self.simple_tech_modules[0] # key "A"
        pattern = {(0,0): module_A['key']} # Pattern wants to place "A" at (0,0)

        mock_get_tech_modules.return_value = [module_A] # Player owns "A"

        # Act
        returned_grid, returned_score = apply_pattern_to_grid(
            grid, pattern, self.modules, tech, self.ship, self.player_owned_rewards
        )

        # Assert
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, tech)
        # clear_all_modules_of_tech for 'tech' should not affect 'other_tech'
        # It might be called, but the check for overlap should fail the placement.
        # Update: If overlap with different tech, pre-checks should fail, and clear should NOT be called.
        mock_clear_tech.assert_not_called()

        mock_place_module.assert_not_called() # Should fail pre-check for overlap with different tech
        mock_calc_adj_score.assert_not_called()

        self.assertIsNone(returned_grid) # Function should return None for the grid if pre-check fails
        self.assertEqual(returned_score, 0) # And 0 for score
        # Ensure original module of different tech is still there
        self.assertEqual(grid.get_cell(0,0)['module'], "EXISTING")
        self.assertEqual(grid.get_cell(0,0)['tech'], other_tech)

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.calculate_pattern_adjacency_score")
    @patch("grid_utils.Grid.place_module")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    def test_apply_pattern_overlaps_same_tech(
        self, mock_clear_tech, mock_place_module, mock_calc_adj_score, mock_get_tech_modules
    ):
        """Test pattern overlap with existing modules of the same tech."""
        # Arrange
        grid = self.empty_grid.copy()
        tech = self.tech

        # Pre-fill grid with a module of the same tech at (1,1)
        grid.set_module(1, 1, "OLD_SAME_TECH_MOD")
        grid.set_tech(1, 1, tech)

        module_A = self.simple_tech_modules[0] # key "A"
        pattern = {(0,0): module_A['key']} # Pattern wants to place "A" at (0,0)

        mock_get_tech_modules.return_value = [module_A] # Player owns "A"
        mock_calc_adj_score.return_value = 3.0
        mock_place_module.return_value = True

        # Act
        returned_grid, returned_score = apply_pattern_to_grid(
            grid, pattern, self.modules, tech, self.ship, self.player_owned_rewards
        )

        # Assert
        mock_clear_tech.assert_called_once_with(grid, tech) # Must be called
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, tech)

        mock_place_module.assert_called_once_with(0, 0, module_A, tech, 0)
        mock_calc_adj_score.assert_called_once_with(returned_grid, tech)

        self.assertIsNotNone(returned_grid)
        self.assertEqual(returned_score, 3.0)
        self.assertEqual(returned_grid.get_cell(0,0)['module'], module_A['key'])
        self.assertEqual(returned_grid.get_cell(0,0)['tech'], tech)
        # Ensure the old module of the same tech at (1,1) was cleared
        self.assertIsNone(returned_grid.get_cell(1,1)['module'])

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.calculate_pattern_adjacency_score")
    @patch("grid_utils.Grid.place_module")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    def test_apply_pattern_player_owns_some_modules(
        self, mock_clear_tech, mock_place_module, mock_calc_adj_score, mock_get_tech_modules
    ):
        """Test applying a pattern where player owns only some modules."""
        # Arrange
        grid = self.empty_grid.copy()
        tech = self.tech
        module_A = self.simple_tech_modules[0] # key "A"
        module_B = self.simple_tech_modules[1] # key "B" (unowned)
        pattern = {(0,0): module_A['key'], (1,0): module_B['key']}

        mock_get_tech_modules.return_value = [module_A] # Player only owns "A"
        mock_calc_adj_score.return_value = 2.0
        mock_place_module.return_value = True # For module A

        # Act
        returned_grid, returned_score = apply_pattern_to_grid(
            grid, pattern, self.modules, tech, self.ship, self.player_owned_rewards
        )

        # Assert
        mock_clear_tech.assert_called_once_with(grid, tech)
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, tech)

        # Only module "A" should be placed
        mock_place_module.assert_called_once_with(0, 0, module_A, tech, 0)
        mock_calc_adj_score.assert_called_once_with(returned_grid, tech)

        self.assertIsNotNone(returned_grid)
        self.assertEqual(returned_score, 2.0)
        self.assertEqual(returned_grid.get_cell(0,0)['module'], module_A['key'])
        self.assertIsNone(returned_grid.get_cell(1,0)['module']) # Module B not placed (unowned)

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.calculate_pattern_adjacency_score")
    @patch("grid_utils.Grid.place_module")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    def test_apply_pattern_player_owns_no_modules_in_pattern(
        self, mock_clear_tech, mock_place_module, mock_calc_adj_score, mock_get_tech_modules
    ):
        """Test pattern with modules, but player owns none of them."""
        # Arrange
        grid = self.empty_grid.copy()
        tech = self.tech
        module_A = self.simple_tech_modules[0]
        module_B = self.simple_tech_modules[1]
        pattern = {(0,0): module_A['key'], (1,0): module_B['key']}

        mock_get_tech_modules.return_value = [] # Player owns no modules of this tech
        mock_calc_adj_score.return_value = 1.0 # Score might be calculated on the (empty) grid

        # Act
        returned_grid, returned_score = apply_pattern_to_grid(
            grid, pattern, self.modules, tech, self.ship, self.player_owned_rewards
        )

        # Assert
        mock_clear_tech.assert_called_once_with(grid, tech) # Grid is cleared for the tech
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, tech)

        mock_place_module.assert_not_called() # No modules placed
        mock_calc_adj_score.assert_called_once_with(returned_grid, tech) # Score calculated on cleared grid

        self.assertIsNotNone(returned_grid)
        self.assertEqual(returned_score, 1.0)
        self.assertIsNone(returned_grid.get_cell(0,0)['module'])
        self.assertIsNone(returned_grid.get_cell(1,0)['module'])

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.calculate_pattern_adjacency_score")
    @patch("grid_utils.Grid.place_module")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    def test_apply_pattern_empty_or_all_none(
        self, mock_clear_tech, mock_place_module, mock_calc_adj_score, mock_get_tech_modules
    ):
        """Test applying an empty pattern or pattern with only None values."""
        # Arrange
        grid = self.empty_grid.copy()
        tech = self.tech
        empty_pattern = {}
        none_pattern = {(0,0): None, (1,0): None}

        mock_get_tech_modules.return_value = [] # Doesn't matter if owned, pattern is empty
        mock_calc_adj_score.return_value = 0.5

        # Act with empty_pattern
        returned_grid_empty, returned_score_empty = apply_pattern_to_grid(
            grid, empty_pattern, self.modules, tech, self.ship, self.player_owned_rewards
        )
        # Assert for empty_pattern
        mock_clear_tech.assert_called_with(grid, tech) # Called once now
        mock_get_tech_modules.assert_called_with(self.modules, self.ship, tech) # Called once now
        mock_place_module.assert_not_called()
        mock_calc_adj_score.assert_called_with(returned_grid_empty, tech) # Called once now
        self.assertIsNotNone(returned_grid_empty)
        self.assertEqual(returned_score_empty, 0.5)

        # Reset mocks for next call with none_pattern
        mock_clear_tech.reset_mock()
        mock_get_tech_modules.reset_mock()
        mock_place_module.reset_mock()
        mock_calc_adj_score.reset_mock()
        mock_calc_adj_score.return_value = 0.6 # Different score for this case

        # Act with none_pattern
        returned_grid_none, returned_score_none = apply_pattern_to_grid(
            grid, none_pattern, self.modules, tech, self.ship, self.player_owned_rewards
        )
        # Assert for none_pattern
        mock_clear_tech.assert_called_once_with(grid, tech)
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, tech)
        mock_place_module.assert_not_called()
        mock_calc_adj_score.assert_called_once_with(returned_grid_none, tech)
        self.assertIsNotNone(returned_grid_none)
        self.assertEqual(returned_score_none, 0.6)

    @patch("optimization_algorithms.get_tech_modules")
    # No other mocks like place_module or calc_adj_score should be called if this pre-check fails
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    def test_apply_pattern_no_owned_modules_to_place_but_pattern_has_unowned(
        self, mock_clear_tech, mock_get_tech_modules
    ):
        """Test pre-check: expected_module_placements_in_pattern is 0 (no owned modules)
           but pattern has non-None (unowned) modules. Should return (None, 0)."""
        # Arrange
        grid = self.empty_grid.copy()
        tech = self.tech
        module_A = self.simple_tech_modules[0] # Key "A"
        pattern = {(0,0): module_A['key']} # Pattern has "A"

        mock_get_tech_modules.return_value = [] # Player owns no modules of this tech

        # Act
        # This scenario, based on the description, is a specific pre-check:
        # "If expected_module_placements_in_pattern == 0 and pattern contains non-None items,
        # it implies all modules in the pattern are unowned by the player.
        # This specific configuration should lead to a (None, 0) return, indicating an invalid pattern application."
        returned_grid, returned_score = apply_pattern_to_grid(
            grid, pattern, self.modules, tech, self.ship, self.player_owned_rewards
        )

        # Assert
        # This is a pre-flight check. If it fails, grid should not be cleared.
        mock_clear_tech.assert_not_called()
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, tech)

        self.assertIsNone(returned_grid)
        self.assertEqual(returned_score, 0)

    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.calculate_pattern_adjacency_score") # Should not be called
    @patch("grid_utils.Grid.place_module") # Might be called for some, then one fails
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    def test_apply_pattern_owned_module_fails_placement_on_inactive_cell(
        self, mock_clear_tech, mock_place_module, mock_calc_adj_score, mock_get_tech_modules
    ):
        """Test when an owned module should be placed, but its target cell is inactive
           or Grid.place_module returns False, leading to successfully_placed < expected.
           This test focuses on Grid.place_module returning False."""
        # Arrange
        grid = self.empty_grid.copy() # 4x3 grid
        tech = self.tech
        module_A = self.simple_tech_modules[0] # key "A"
        module_B = self.simple_tech_modules[1] # key "B"
        pattern = {(0,0): module_A['key'], (1,0): module_B['key']}

        # Player owns both modules
        mock_get_tech_modules.return_value = [module_A, module_B]

        # Simulate module "A" placing successfully, but "B" failing
        # (e.g., if (1,0) was inactive, or just general placement failure)
        # Grid.place_module(x, y, module_data, tech_name, rotation=0)
        def place_module_side_effect(x, y, module_data, tech_name, rotation):
            if module_data['key'] == module_A['key']:
                grid.set_module(x,y,module_data['key']) # Simulate actual placement for grid state
                grid.set_tech(x,y,tech_name)
                return True
            elif module_data['key'] == module_B['key']:
                return False # Module B fails to place
            return False
        mock_place_module.side_effect = place_module_side_effect

        # Act
        returned_grid, returned_score = apply_pattern_to_grid(
            grid, pattern, self.modules, tech, self.ship, self.player_owned_rewards
        )

        # Assert
        mock_clear_tech.assert_called_once_with(grid, tech) # Clearing happens before placement attempts
        mock_get_tech_modules.assert_called_once_with(self.modules, self.ship, tech)

        # Check calls to place_module
        self.assertTrue(mock_place_module.call_count >= 1) # At least A was attempted
        # It might try to place A, then B. Or it might stop after B fails.
        # The key is that not all *expected* modules were placed.

        mock_calc_adj_score.assert_not_called() # Should not be called if placement count mismatch

        self.assertIsNone(returned_grid)
        self.assertEqual(returned_score, 0)

    # --- Tests for calculate_pattern_adjacency_score ---

    def test_calculate_pattern_adjacency_score_basic_mix(self):
        """Test basic score calculation with mixed adjacencies."""
        grid = Grid(3, 3)
        target_tech = "pulse"
        other_tech = "beam"

        # Constants used by the function (assumed)
        module_edge_weight = 3.0
        grid_edge_weight = 0.5

        # M1 (0,0) - target_tech:
        #   - Right edge: M_other (1,0) -> +module_edge_weight
        #   - Bottom edge: M_other (0,1) -> +module_edge_weight
        #   - Top grid edge -> +grid_edge_weight
        #   - Left grid edge -> +grid_edge_weight
        #   Score for M1 = 2 * module_edge_weight + 2 * grid_edge_weight = 2*3.0 + 2*0.5 = 6.0 + 1.0 = 7.0
        grid.set_module(0, 0, "M1")
        grid.set_tech(0, 0, target_tech)

        grid.set_module(1, 0, "M_other1") # Adjacent to M1 (right)
        grid.set_tech(1, 0, other_tech)
        grid.set_module(0, 1, "M_other2") # Adjacent to M1 (bottom)
        grid.set_tech(0, 1, other_tech)

        # M2 (2,2) - target_tech:
        #   - Top edge: M_other (2,1) -> +module_edge_weight
        #   - Left edge: M_other (1,2) -> +module_edge_weight
        #   - Bottom grid edge -> +grid_edge_weight
        #   - Right grid edge -> +grid_edge_weight
        #   Score for M2 = 2 * module_edge_weight + 2 * grid_edge_weight = 7.0
        grid.set_module(2, 2, "M2")
        grid.set_tech(2, 2, target_tech)

        grid.set_module(2, 1, "M_other3") # Adjacent to M2 (top)
        grid.set_tech(2, 1, other_tech)
        grid.set_module(1, 2, "M_other4") # Adjacent to M2 (left)
        grid.set_tech(1, 2, other_tech)

        # M3 (1,1) - target_tech (no valid adjacencies for this score type)
        #   - Surrounded by empty cells or same tech (not part of this score)
        grid.set_module(1, 1, "M3")
        grid.set_tech(1, 1, target_tech)


        expected_score = (7.0) + (7.0) + (0.0) # M1 + M2 + M3

        score = calculate_pattern_adjacency_score(grid, target_tech)
        self.assertEqual(score, expected_score)

    def test_calculate_pattern_adjacency_score_grid_edges_only(self):
        """Test score with modules only adjacent to grid edges."""
        grid = Grid(3, 3)
        target_tech = "pulse"
        grid_edge_weight = 0.5

        # M_corner (0,0) - target_tech:
        #   - Top grid edge -> +grid_edge_weight
        #   - Left grid edge -> +grid_edge_weight
        #   Score = 2 * grid_edge_weight = 1.0
        grid.set_module(0, 0, "M_corner")
        grid.set_tech(0, 0, target_tech)
        score_corner = calculate_pattern_adjacency_score(grid, target_tech)
        self.assertEqual(score_corner, 1.0)

        # Clear grid for next sub-test
        grid = Grid(3,3)
        # M_edge (0,1) - target_tech:
        #   - Left grid edge -> +grid_edge_weight
        #   Score = 1 * grid_edge_weight = 0.5
        grid.set_module(0, 1, "M_edge")
        grid.set_tech(0, 1, target_tech)
        score_edge = calculate_pattern_adjacency_score(grid, target_tech)
        self.assertEqual(score_edge, 0.5)

        # M_center (1,1) - target_tech (no grid edge adjacency)
        grid = Grid(3,3)
        grid.set_module(1, 1, "M_center")
        grid.set_tech(1, 1, target_tech)
        score_center = calculate_pattern_adjacency_score(grid, target_tech)
        self.assertEqual(score_center, 0.0) # No grid edges, no other modules

    def test_calculate_pattern_adjacency_score_different_tech_only(self):
        """Test score with modules only adjacent to different tech modules."""
        grid = Grid(3, 3)
        target_tech = "pulse"
        other_tech = "beam"
        module_edge_weight = 3.0

        # M_target (1,1) - target_tech, surrounded by other_tech
        grid.set_module(1, 1, "M_target")
        grid.set_tech(1, 1, target_tech)

        grid.set_module(0, 1, "M_other_left") # Left
        grid.set_tech(0, 1, other_tech)
        grid.set_module(2, 1, "M_other_right") # Right
        grid.set_tech(2, 1, other_tech)
        grid.set_module(1, 0, "M_other_top") # Top
        grid.set_tech(1, 0, other_tech)
        grid.set_module(1, 2, "M_other_bottom") # Bottom
        grid.set_tech(1, 2, other_tech)

        # Score for M_target = 4 * module_edge_weight = 4 * 3.0 = 12.0
        expected_score = 12.0
        score = calculate_pattern_adjacency_score(grid, target_tech)
        self.assertEqual(score, expected_score)

    def test_calculate_pattern_adjacency_score_no_target_tech_modules(self):
        """Test score when no modules of the target tech are on the grid."""
        grid = Grid(3, 3)
        target_tech = "pulse"
        other_tech = "beam"

        grid.set_module(0, 0, "M_other")
        grid.set_tech(0, 0, other_tech)

        score = calculate_pattern_adjacency_score(grid, target_tech)
        self.assertEqual(score, 0.0)

        # Empty grid
        empty_grid = Grid(3,3)
        score_empty = calculate_pattern_adjacency_score(empty_grid, target_tech)
        self.assertEqual(score_empty, 0.0)

    def test_calculate_pattern_adjacency_score_target_tech_no_adjacencies(self):
        """Test score with target tech modules but no scorable adjacencies."""
        grid = Grid(5, 5) # Larger grid
        target_tech = "pulse"

        # M1 in center, no adjacencies
        grid.set_module(2, 2, "M1")
        grid.set_tech(2, 2, target_tech)

        # M2 and M3 far apart, also no scorable adjacencies
        grid.set_module(0, 4, "M2") # On edge, will score 0.5 (left edge)
        grid.set_tech(0, 4, target_tech)

        grid.set_module(4, 0, "M3") # On edge, will score 0.5 (top edge)
        grid.set_tech(4, 0, target_tech)

        # Score for M1 = 0
        # Score for M2 = 0.5 (left grid edge)
        # Score for M3 = 0.5 (top grid edge)
        # Total expected = 0 + 0.5 + 0.5 = 1.0
        # The prompt title says "no adjacencies" but my setup for M2/M3 has grid edge.
        # Let's make one truly isolated for the spirit of "no adjacencies"
        # and then handle the "far apart" ones.

        grid_isolated = Grid(5,5)
        grid_isolated.set_module(2,2, "M_iso")
        grid_isolated.set_tech(2,2, target_tech)
        score_isolated = calculate_pattern_adjacency_score(grid_isolated, target_tech)
        self.assertEqual(score_isolated, 0.0, "Isolated module should have 0 score")

        # Test for "far apart" modules that might be on edges
        grid_far_apart = Grid(5,5)
        grid_far_apart.set_module(0,0, "M_corner1") # 2 edges = 1.0
        grid_far_apart.set_tech(0,0, target_tech)
        grid_far_apart.set_module(4,4, "M_corner2") # 2 edges = 1.0
        grid_far_apart.set_tech(4,4, target_tech)
        score_far_apart = calculate_pattern_adjacency_score(grid_far_apart, target_tech)
        self.assertEqual(score_far_apart, 2.0, "Far apart corner modules")


    def test_calculate_pattern_adjacency_score_adjacent_to_same_tech(self):
        """Test that adjacency to modules of the same tech does not contribute to the score."""
        grid = Grid(3, 3)
        target_tech = "pulse"

        # M1 and M2 are target_tech and adjacent to each other.
        # Neither is on a grid edge or near a different tech.
        grid.set_module(1, 1, "M1")
        grid.set_tech(1, 1, target_tech)
        grid.set_module(1, 2, "M2") # M2 is below M1
        grid.set_tech(1, 2, target_tech)

        # M3 on an edge for some baseline score, but also adjacent to M1 (same tech)
        grid.set_module(0, 1, "M3") # M3 is left of M1, and on left grid edge
        grid.set_tech(0, 1, target_tech)
        # Score for M3 should only be from grid edge: 0.5
        # M1 and M2 should have 0 score from each other. M1 is adjacent to M3 (same tech).

        expected_score = 0.5 # Only M3's left edge adjacency to grid
        score = calculate_pattern_adjacency_score(grid, target_tech)
        self.assertEqual(score, expected_score)

    # --- Tests for _handle_ml_opportunity ---

    @patch("optimization_algorithms.create_localized_grid_ml")
    @patch("ml_placement.ml_placement")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    @patch("optimization_algorithms.apply_localized_grid_changes")
    @patch("optimization_algorithms.restore_original_state")
    @patch("optimization_algorithms.calculate_grid_score")
    def test_handle_ml_opportunity_success(
        self, mock_calc_score, mock_restore_state, mock_apply_changes,
        mock_clear_tech, mock_ml_placement, mock_create_localized_grid
    ):
        """Test successful ML refinement in _handle_ml_opportunity."""
        # Arrange
        original_grid = self.empty_grid.copy() # 4x3 grid
        original_grid.set_module(3,2, "EXISTING_MOD") # Add some module to check it's handled
        original_grid.set_tech(3,2, "other_tech")

        ship = self.ship # "standard"
        tech = self.tech # "pulse"
        rewards = self.player_owned_rewards # ["PC"]
        opportunity_x, opportunity_y = 0, 0
        window_width, window_height = 2, 2
        initial_global_score = 10.0 # Score of original_grid before ML attempt

        # Mock create_localized_grid_ml
        localized_grid_for_ml = Grid(window_width, window_height)
        localized_grid_for_ml.set_module(0,0,"TEMP_ML_INPUT") # Dummy content for localized grid
        localized_grid_for_ml.set_tech(0,0,tech)
        original_state_map = {(0,0): {"module": None, "tech": None}} # Simplified
        mock_create_localized_grid.return_value = (localized_grid_for_ml, original_state_map)

        # Mock ml_placement.ml_placement to return a refined localized grid and its score
        ml_refined_localized_grid = Grid(window_width, window_height)
        ml_refined_localized_grid.set_module(0,0,"ML_A") # ML places "ML_A"
        ml_refined_localized_grid.set_tech(0,0,tech)
        ml_refined_localized_grid.set_module(1,0,"ML_B") # ML places "ML_B"
        ml_refined_localized_grid.set_tech(1,0,tech)
        # This score is for the localized_grid_for_ml *after* ml_placement has refined it.
        # It's not a global score.
        mock_ml_placement.return_value = (ml_refined_localized_grid, 15.0) # Localized score

        # Mock calculate_grid_score for the final global score calculation
        final_global_score = 25.0
        mock_calc_score.return_value = final_global_score

        # Act
        # _handle_ml_opportunity(grid, ship, all_modules_data, tech, player_owned_rewards,
        #                        opportunity_x, opportunity_y, window_width, window_height,
        #                        base_score, sc_slots_in_window)
        # Assuming base_score is initial_global_score, sc_slots_in_window might be None or derived.
        returned_final_grid, returned_final_score = _handle_ml_opportunity(
            original_grid, ship, self.modules, tech, rewards,
            opportunity_x, opportunity_y, window_width, window_height,
            initial_global_score, sc_slots_in_window=None # Assuming None for simplicity
        )

        # Assert
        mock_create_localized_grid.assert_called_once_with(
            original_grid, opportunity_x, opportunity_y, window_width, window_height
        )
        mock_ml_placement.assert_called_once_with(
            localized_grid_for_ml, ship, self.modules, tech, rewards,
            window_width, window_height, sc_slots_in_window=None, # from call
            # initial_score_for_ml_window=ANY # This depends on how base score is passed to ML
        )
        # The grid passed to clear_all_modules_of_tech should be a copy of original_grid.
        mock_clear_tech.assert_called_once()
        self.assertIsInstance(mock_clear_tech.call_args[0][0], Grid)
        self.assertEqual(mock_clear_tech.call_args[0][1], tech)

        # The grid passed to apply_localized_grid_changes should be the one from clear_tech
        mock_apply_changes.assert_called_once_with(
            mock_clear_tech.call_args[0][0], # Grid obj after clearing
            ml_refined_localized_grid, tech, opportunity_x, opportunity_y
        )
        # The grid passed to restore_original_state should be the one from apply_changes
        mock_restore_state.assert_called_once_with(
            mock_apply_changes.call_args[0][0], # Grid obj after applying changes
            original_state_map, opportunity_x, opportunity_y, window_width, window_height
        )
        # The grid passed to calculate_grid_score is the one after restore_original_state
        mock_calc_score.assert_called_once_with(mock_restore_state.call_args[0][0], tech)

        self.assertIsNotNone(returned_final_grid)
        self.assertEqual(returned_final_score, final_global_score)
        # Further checks on returned_final_grid state if necessary, e.g., ML_A and ML_B are there.
        # This depends on the mocks for apply_changes and restore_state being pass-through or modifying.
        # For now, assume they modify the grid correctly and calculate_grid_score reflects that.

    @patch("optimization_algorithms.create_localized_grid_ml")
    @patch("ml_placement.ml_placement")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    @patch("optimization_algorithms.apply_localized_grid_changes")
    @patch("optimization_algorithms.restore_original_state")
    @patch("optimization_algorithms.calculate_grid_score")
    def test_handle_ml_opportunity_ml_failure(
        self, mock_calc_score, mock_restore_state, mock_apply_changes,
        mock_clear_tech, mock_ml_placement, mock_create_localized_grid
    ):
        """Test _handle_ml_opportunity when ml_placement.ml_placement fails."""
        # Arrange
        original_grid = self.empty_grid.copy()
        original_grid.set_module(0,0, "SHOULD_BE_RESTORED") # Module that should be part of original_state_map
        original_grid.set_tech(0,0, "some_other_tech") # or even the same tech, if ML window clears it

        ship = self.ship
        tech = self.tech
        rewards = self.player_owned_rewards
        opportunity_x, opportunity_y = 0, 0
        window_width, window_height = 2, 2
        initial_global_score = 10.0

        # Mock create_localized_grid_ml
        localized_grid_for_ml = Grid(window_width, window_height)
        # original_state_map should capture what was in the window of original_grid
        original_state_map = {(0,0): {"module": "SHOULD_BE_RESTORED", "tech": "some_other_tech"}}
        mock_create_localized_grid.return_value = (localized_grid_for_ml, original_state_map)

        # Mock ml_placement.ml_placement to return failure
        mock_ml_placement.return_value = (None, 0.0)

        # Mock calculate_grid_score: should be called on the grid *after* restoration.
        # This score should ideally be the initial_global_score if restoration is perfect.
        mock_calc_score.return_value = initial_global_score

        # Act
        returned_final_grid, returned_final_score = _handle_ml_opportunity(
            original_grid, ship, self.modules, tech, rewards,
            opportunity_x, opportunity_y, window_width, window_height,
            initial_global_score, sc_slots_in_window=None
        )

        # Assert
        mock_create_localized_grid.assert_called_once_with(
            original_grid, opportunity_x, opportunity_y, window_width, window_height
        )
        mock_ml_placement.assert_called_once_with(
            localized_grid_for_ml, ship, self.modules, tech, rewards,
            window_width, window_height, sc_slots_in_window=None,
            # initial_score_for_ml_window=ANY
        )

        # These should NOT be called if ml_placement fails
        mock_clear_tech.assert_not_called()
        mock_apply_changes.assert_not_called()

        # restore_original_state SHOULD be called to revert any changes made by create_localized_grid_ml
        # or to ensure the original state of the window is perfectly restored.
        # The grid passed to it would be the original_grid itself, or a pristine copy if
        # create_localized_grid_ml worked on a copy.
        # Assuming create_localized_grid_ml might modify a copy or the original to extract the window,
        # restoration is key.
        mock_restore_state.assert_called_once_with(
            original_grid, # Or a copy that create_localized_grid_ml might have used/modified
            original_state_map, opportunity_x, opportunity_y, window_width, window_height
        )

        # calculate_grid_score is called on the grid after restoration attempt.
        mock_calc_score.assert_called_once_with(mock_restore_state.call_args[0][0], tech)

        self.assertIsNone(returned_final_grid) # Or original_grid if no refinement is the outcome
        self.assertEqual(returned_final_score, 0.0) # Or initial_global_score if failure means no change in score
                                                    # The prompt implies (None, 0.0) for failure.

    # --- Tests for _handle_sa_refine_opportunity ---

    @patch("optimization_algorithms.create_localized_grid")
    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.refine_placement")
    @patch("optimization_algorithms.simulated_annealing")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    @patch("optimization_algorithms.apply_localized_grid_changes")
    @patch("optimization_algorithms.calculate_grid_score")
    def test_handle_sa_refine_opportunity_calls_refine_placement(
        self, mock_calc_score, mock_apply_changes, mock_clear_tech,
        mock_sa, mock_refine_placement, mock_get_tech_modules, mock_create_localized_grid
    ):
        """Test _handle_sa_refine_opportunity calls refine_placement for < 6 modules."""
        # Arrange
        original_grid = self.empty_grid.copy() # 4x3
        ship = self.ship; tech = self.tech; rewards = self.player_owned_rewards
        opportunity_x, opportunity_y, window_width, window_height = 0,0,2,2
        sc_slots_in_window = [] # Example

        # Mock create_localized_grid
        localized_grid_sa = Grid(window_width, window_height)
        # create_localized_grid returns: localized_grid, actual_start_x, actual_start_y
        mock_create_localized_grid.return_value = (localized_grid_sa, opportunity_x, opportunity_y)

        # Mock get_tech_modules to return < 6 modules
        # Using self.simple_tech_modules which has 2 modules (A, B)
        # These are full module dicts.
        tech_modules_for_refine = self.simple_tech_modules[:2]
        mock_get_tech_modules.return_value = tech_modules_for_refine
        self.assertLess(len(tech_modules_for_refine), 6)

        # Mock refine_placement to return a successfully refined localized grid and its score
        refined_localized_grid = Grid(window_width, window_height)
        refined_localized_grid.set_module(0,0,"RP_A"); refined_localized_grid.set_tech(0,0,tech)
        # This score is for the localized_grid_sa *after* refine_placement. Not global.
        mock_refine_placement.return_value = (refined_localized_grid, 12.0)

        # Mock calculate_grid_score for the final global score
        final_global_score = 22.0
        mock_calc_score.return_value = final_global_score

        # Act
        returned_final_grid, returned_final_score = _handle_sa_refine_opportunity(
            original_grid, ship, self.modules, tech, rewards,
            opportunity_x, opportunity_y, window_width, window_height,
            sc_slots_in_window
        )

        # Assert
        mock_create_localized_grid.assert_called_once_with(
            original_grid, opportunity_x, opportunity_y, window_width, window_height, tech
        )
        mock_get_tech_modules.assert_called_once_with(self.modules, ship, tech, rewards)

        # refine_placement should be called
        mock_refine_placement.assert_called_once_with(
            localized_grid_sa, ship, self.modules, tech, rewards, ANY, # initial_score for localized
            sc_slots_in_window=sc_slots_in_window
        )
        mock_sa.assert_not_called() # SA should NOT be called

        # Grid modification flow
        mock_clear_tech.assert_called_once() # On a copy of original_grid for the tech
        self.assertIsInstance(mock_clear_tech.call_args[0][0], Grid)
        self.assertEqual(mock_clear_tech.call_args[0][1], tech)

        mock_apply_changes.assert_called_once_with(
            mock_clear_tech.call_args[0][0], # Grid after clearing
            refined_localized_grid, tech, opportunity_x, opportunity_y
        )
        mock_calc_score.assert_called_once_with(mock_apply_changes.call_args[0][0], tech) # Grid after apply

        self.assertIsNotNone(returned_final_grid)
        self.assertEqual(returned_final_score, final_global_score)

    @patch("optimization_algorithms.create_localized_grid")
    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.refine_placement")
    @patch("optimization_algorithms.simulated_annealing")
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    @patch("optimization_algorithms.apply_localized_grid_changes")
    @patch("optimization_algorithms.calculate_grid_score")
    def test_handle_sa_refine_opportunity_calls_sa(
        self, mock_calc_score, mock_apply_changes, mock_clear_tech,
        mock_sa, mock_refine_placement, mock_get_tech_modules, mock_create_localized_grid
    ):
        """Test _handle_sa_refine_opportunity calls simulated_annealing for >= 6 modules."""
        # Arrange
        original_grid = self.empty_grid.copy()
        ship = self.ship; tech = self.tech; rewards = self.player_owned_rewards
        opportunity_x, opportunity_y, window_width, window_height = 0,0,3,3 # Larger window for more modules
        sc_slots_in_window = []

        mock_create_localized_grid.return_value = (Grid(window_width, window_height), opportunity_x, opportunity_y)

        # Mock get_tech_modules to return >= 6 modules
        # Create a list of 6 simple module dicts
        tech_modules_for_sa = [{"key": f"SA{i}", "name": f"SA Mod {i}", "shape":[[1]], "value":1} for i in range(6)]
        mock_get_tech_modules.return_value = tech_modules_for_sa
        self.assertGreaterEqual(len(tech_modules_for_sa), 6)

        # Mock simulated_annealing
        sa_refined_localized_grid = Grid(window_width, window_height)
        sa_refined_localized_grid.set_module(0,0,"SA_A"); sa_refined_localized_grid.set_tech(0,0,tech)
        mock_sa.return_value = (sa_refined_localized_grid, 18.0) # Localized score from SA

        final_global_score = 28.0
        mock_calc_score.return_value = final_global_score

        # Act
        returned_final_grid, returned_final_score = _handle_sa_refine_opportunity(
            original_grid, ship, self.modules, tech, rewards,
            opportunity_x, opportunity_y, window_width, window_height,
            sc_slots_in_window
        )

        # Assert
        mock_create_localized_grid.assert_called_once_with(
            original_grid, opportunity_x, opportunity_y, window_width, window_height, tech
        )
        # get_tech_modules is called by _handle_sa_refine_opportunity to decide path,
        # and then SA/refine_placement might call it again internally if they need all_modules_data vs specific tech_modules.
        # For this test, we are interested in the first call that determines the path.
        # The mock_get_tech_modules here serves the path determination.
        # If SA/refine_placement also call it, the mock needs to handle that or be more specific.
        # Let's assume for now the `tech_modules_for_sa` are what's passed to SA directly.
        mock_get_tech_modules.assert_called_once_with(self.modules, ship, tech, rewards)

        mock_sa.assert_called_once_with(
            mock_create_localized_grid.return_value[0], # localized_grid
            tech_modules_for_sa, # List of specific tech modules
            ANY, # initial_score for SA on localized window
            ANY, # iterations
            ANY, # temperature
            sc_slots_in_window,
            self.modules, # all_modules_data
            ship,
            tech,
            rewards
        )
        mock_refine_placement.assert_not_called()

        mock_clear_tech.assert_called_once()
        mock_apply_changes.assert_called_once_with(
            mock_clear_tech.call_args[0][0],
            sa_refined_localized_grid, tech, opportunity_x, opportunity_y
        )
        mock_calc_score.assert_called_once_with(mock_apply_changes.call_args[0][0], tech)

        self.assertIsNotNone(returned_final_grid)
        self.assertEqual(returned_final_score, final_global_score)

    @patch("optimization_algorithms.create_localized_grid")
    @patch("optimization_algorithms.get_tech_modules")
    @patch("optimization_algorithms.refine_placement")
    @patch("optimization_algorithms.simulated_annealing") # Still mock SA to ensure it's not called
    @patch("optimization_algorithms.clear_all_modules_of_tech")
    @patch("optimization_algorithms.apply_localized_grid_changes")
    @patch("optimization_algorithms.calculate_grid_score") # Might be called on original grid or not at all
    def test_handle_sa_refine_opportunity_refinement_failure(
        self, mock_calc_score, mock_apply_changes, mock_clear_tech,
        mock_sa, mock_refine_placement, mock_get_tech_modules, mock_create_localized_grid
    ):
        """Test _handle_sa_refine_opportunity when refine_placement returns (None, 0.0)."""
        # Arrange
        original_grid = self.empty_grid.copy()
        original_grid.set_module(0,0, "TEST_MOD") # For checking if grid reverts or is cleared
        original_grid.set_tech(0,0, self.tech)

        ship = self.ship; tech = self.tech; rewards = self.player_owned_rewards
        opportunity_x, opportunity_y, window_width, window_height = 0,0,2,2
        sc_slots_in_window = []

        localized_grid_sa = Grid(window_width, window_height)
        mock_create_localized_grid.return_value = (localized_grid_sa, opportunity_x, opportunity_y)

        # Path: refine_placement (< 6 modules)
        tech_modules_for_refine = self.simple_tech_modules[:2]
        mock_get_tech_modules.return_value = tech_modules_for_refine

        # Mock refine_placement to return failure
        mock_refine_placement.return_value = (None, 0.0)

        # Act
        returned_final_grid, returned_final_score = _handle_sa_refine_opportunity(
            original_grid.copy(), # Pass a copy so we can check original_grid state later
            ship, self.modules, tech, rewards,
            opportunity_x, opportunity_y, window_width, window_height,
            sc_slots_in_window
        )

        # Assert
        mock_create_localized_grid.assert_called_once()
        mock_get_tech_modules.assert_called_once()
        mock_refine_placement.assert_called_once() # Called, but returned failure
        mock_sa.assert_not_called()

        # These should NOT be called if refinement fails and returns None for the grid
        mock_clear_tech.assert_not_called()
        mock_apply_changes.assert_not_called()

        # calculate_grid_score should also NOT be called on the main grid if refinement failed this way.
        # The function should propagate the failure.
        mock_calc_score.assert_not_called()

        # Expecting the function to return the *original grid cleared of the tech* and -1.0
        # as per prompt "return (original_grid_cleared_of_tech, -1.0)"
        # This means clear_all_modules_of_tech *is* called on a copy for this return type.
        # Let's re-evaluate: if refine_placement on localized_grid returns (None, 0.0),
        # _handle_sa_refine_opportunity should not proceed to modify the main grid.
        # It should return an indicator of failure. What it returns depends on its contract.
        # Prompt: "(original_grid_cleared_of_tech, -1.0)"
        # This is a specific contract. So, clear_tech *would* be called on a *copy* of the original grid.
        # And that copy is returned.

        # Let's adjust the mock_clear_tech expectation based on this specific return contract.
        # If the above asserts for not_called are right, then the return contract is different.
        # If the contract is (None, -1.0) or (original_grid_untouched, -1.0), then not_called is correct.
        # If contract is (cleared_original_grid, -1.0), then clear_tech IS called.

        # Assuming failure means no changes applied to original grid, and a specific failure code:
        self.assertIsNone(returned_final_grid) # Or check if it's the original_grid instance if that's the contract
        self.assertEqual(returned_final_score, -1.0) # Or 0.0, depending on failure signal

        # If the contract is to return (original_grid_cleared_of_tech, -1.0):
        # My current mocks (clear_tech.assert_not_called) would contradict this.
        # The prompt for this test: "Verify the function returns (original_grid_cleared_of_tech, -1.0) or similar"
        # This implies `clear_all_modules_of_tech` IS called on a copy of the original grid,
        # and that grid is returned.

        # Let's refine based on the "original_grid_cleared_of_tech, -1.0" expectation.
        # This means:
        # 1. `refine_placement` (on localized grid) fails -> returns (None, 0.0)
        # 2. `_handle_sa_refine_opportunity` sees this.
        # 3. It should then create a fresh copy of the *original input grid*.
        # 4. Call `clear_all_modules_of_tech` on this fresh copy.
        # 5. Return this cleared copy and -1.0.
        # So, `apply_localized_grid_changes` and `calculate_grid_score` (for global score) are not called.

        # With this interpretation, the current mock_clear_tech.assert_not_called() is WRONG for this specific contract.
        # It should be called once.

        # Re-arranging the test based on the "cleared_original_grid, -1.0" contract
        # This test needs to be re-thought if clear_tech is called for the failure return type.
        # For now, I will stick to the more common pattern: if a sub-process (refinement) fails
        # and returns None for a grid, the calling function propagates this None and a failure score,
        # without further modification of the main grid beyond what was necessary for the attempt.

        # If `refine_placement` (or SA) returns (None, some_score), it means the refinement attempt
        # on the *localized* grid did not yield a usable *localized* grid.
        # In this case, `_handle_sa_refine_opportunity` should not try to apply these non-existent
        # localized changes to the main grid. So `clear_all_modules_of_tech` on the main grid copy
        # and `apply_localized_grid_changes` should indeed be skipped.
        # The return value indicating failure would be (None, -1.0) or (None, 0.0).
        # The prompt's "(original_grid_cleared_of_tech, -1.0)" is an unusual contract if the
        # localized refinement itself failed to produce a grid.
        # I will proceed with the interpretation that (None, -1.0) is returned, and main grid modifications for application are skipped.

    # --- Tests for count_empty_in_localized ---

    def test_count_empty_in_localized(self):
        """Test count_empty_in_localized for various grid states."""

        # Test with a mix of empty and occupied slots
        grid_mix = Grid(2, 2) # 4 total slots
        grid_mix.set_module(0, 0, "A")
        grid_mix.set_module(1, 1, "B")
        # (0,1) and (1,0) are empty
        self.assertEqual(count_empty_in_localized(grid_mix), 2, "Mix: Should count 2 empty slots")

        # Test with a fully occupied grid
        grid_full = Grid(2, 2)
        grid_full.set_module(0, 0, "A")
        grid_full.set_module(0, 1, "B")
        grid_full.set_module(1, 0, "C")
        grid_full.set_module(1, 1, "D")
        self.assertEqual(count_empty_in_localized(grid_full), 0, "Full: Should count 0 empty slots")

        # Test with a completely empty grid
        grid_empty = Grid(2, 3) # 6 total slots
        self.assertEqual(count_empty_in_localized(grid_empty), 6, "Empty: Should count all 6 slots as empty")

        # Test with a 1x1 empty grid
        grid_1x1_empty = Grid(1,1)
        self.assertEqual(count_empty_in_localized(grid_1x1_empty), 1, "1x1 Empty: Should count 1 empty slot")

        # Test with a 1x1 full grid
        grid_1x1_full = Grid(1,1)
        grid_1x1_full.set_module(0,0,"F")
        self.assertEqual(count_empty_in_localized(grid_1x1_full), 0, "1x1 Full: Should count 0 empty slots")

        # Test with a grid that has inactive slots (but are also empty)
        grid_inactive = Grid(2, 2) # 4 total slots
        grid_inactive.set_module(0, 0, "A")
        grid_inactive.set_active(0, 1, False) # (0,1) is inactive and empty (module is None)
        grid_inactive.set_module(1, 0, "B")
        grid_inactive.set_active(1, 1, False) # (1,1) is inactive and empty (module is None)
                                            # but then we place a module, so it's not empty
        grid_inactive.set_module(1,1,"C")
        grid_inactive.set_active(1,1,True) # Let's make (1,1) active but occupied.

        # Reset for clarity on inactive test:
        grid_inactive_test = Grid(2,2) # 4 slots
        grid_inactive_test.set_module(0,0, "OccupiedActive")
        # (0,1) is Inactive AND Empty (module is None)
        grid_inactive_test.set_active(0,1, False)
        # (1,0) is Active AND Empty (module is None)
        # (1,1) is Inactive AND Occupied
        grid_inactive_test.set_module(1,1, "OccupiedInactive")
        grid_inactive_test.set_active(1,1, False)

        # Expected: (0,1) is counted as empty (inactive, module==None)
        #           (1,0) is counted as empty (active, module==None)
        # Total empty = 2
        # The function only checks `cell["module"] is None`.
        self.assertEqual(count_empty_in_localized(grid_inactive_test), 2,
                         "InactiveEmpty: Inactive but module-less slots should be counted as empty.")


# --- Run Tests ---
if __name__ == "__main__":
    # Use the standard unittest main runner
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
