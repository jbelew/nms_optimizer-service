# test_optimization.py
import unittest
from unittest.mock import patch
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# --- Imports from your project ---
from src.optimization import (
    optimize_placement,
)
from src.optimization.helpers import (
    place_all_modules_in_empty_slots,
    check_all_modules_placed,
)
from src.optimization.windowing import (
    find_supercharged_opportunities,
)
from src.grid_utils import apply_localized_grid_changes
from src.module_placement import clear_all_modules_of_tech
from src.pattern_matching import (
    rotate_pattern,
    mirror_pattern_horizontally,
    mirror_pattern_vertically,
    get_all_unique_pattern_variations,
)
from src.grid_utils import Grid
from src.data_loader import get_all_module_data, get_all_solve_data

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

    def test_place_all_modules_in_empty_slots(self):
        # Use the actual infra modules for standard ship
        infra_modules = [m for t in sample_modules["standard"]["types"].values() for m in t if m["key"] == "infra"][0][
            "modules"
        ]

        with patch("src.optimization.helpers.get_tech_modules") as mock_get_tech_modules:
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

    def test_find_supercharged_opportunities_opportunity(self):
        # Use self.sc_grid which has SC slots
        sc_grid_test = self.sc_grid.copy()
        # Place a module of the *same* tech outside the best window to ensure it's cleared
        sc_grid_test.set_module(3, 2, "IK")
        sc_grid_test.set_tech(3, 2, self.tech)
        # Place a module of a *different* tech within the window
        sc_grid_test.set_module(1, 0, "OTHER")
        sc_grid_test.set_tech(1, 0, "other_tech")

        result = find_supercharged_opportunities(sc_grid_test, self.modules, self.ship, self.tech)
        # Assuming the best window starts at (0,0) for a 4x3 grid with SC at (1,1), (2,1)
        self.assertIsNotNone(result)
        self.assertEqual(result, (0, 0, 4, 2))

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

        with patch("src.optimization.helpers.get_tech_modules") as mock_get_tech_modules:
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

        with patch("src.optimization.helpers.get_tech_modules") as mock_get_tech_modules:
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
            optimize_placement(full_grid, self.ship, self.modules, self.tech, self.player_owned_rewards)

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.place_all_modules_in_empty_slots")
    @patch("src.optimization.core.calculate_grid_score")
    def test_optimize_no_solve_map_available(self, mock_calculate_score, mock_place_all, mock_get_solves):
        """Test behavior when no solve map is found for the ship."""
        mock_get_solves.return_value = {}
        placed_grid = self.empty_grid.copy()
        placed_grid.set_module(0, 0, "IK")
        placed_grid.set_tech(0, 0, self.tech)
        mock_place_all.return_value = placed_grid
        mock_calculate_score.return_value = 5.0

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.empty_grid, self.ship, self.modules, self.tech, self.player_owned_rewards
        )

        mock_get_solves.assert_called_once_with(self.ship, None)
        mock_place_all.assert_called_once()
        mock_calculate_score.assert_called_once_with(placed_grid, self.tech)
        self.assertEqual(result_grid, placed_grid)
        self.assertEqual(solved_bonus, 5.0)
        self.assertEqual(percentage, 100.0)

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.refinement.simulated_annealing")
    def test_optimize_solve_map_no_pattern_fits_returns_indicator_when_not_forced(
        self, mock_sa, mock_apply_pattern, mock_get_solves
    ):
        """Test returns 'Pattern No Fit' when solve map exists, no pattern fits, and not forced."""
        mock_get_solves.return_value = sample_solves[self.ship]
        mock_apply_pattern.return_value = (None, 0)

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.empty_grid, self.ship, self.modules, self.tech, self.player_owned_rewards, forced=False
        )

        mock_apply_pattern.assert_called()
        mock_sa.assert_not_called()
        self.assertIsNone(result_grid)
        self.assertEqual(percentage, 0.0)
        self.assertEqual(solved_bonus, 0.0)
        self.assertEqual(solve_method, "Pattern No Fit")

    @patch("src.optimization.core.get_solve_map")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.core.simulated_annealing")
    @patch("src.optimization.core.calculate_grid_score")
    def test_optimize_solve_map_no_pattern_fits_falls_back_to_sa_when_forced(
        self, mock_calculate_score, mock_sa, mock_apply_pattern, mock_get_solves
    ):
        """Test fallback to initial SA when solve map exists, no pattern fits, and forced=True."""
        mock_get_solves.return_value = sample_solves[self.ship]
        mock_apply_pattern.return_value = (None, 0)

        initial_sa_grid = self.empty_grid.copy()
        initial_sa_grid.set_module(0, 1, "PE")
        initial_sa_grid.set_tech(0, 1, self.tech)
        mock_sa.return_value = (initial_sa_grid, 10.0)
        mock_calculate_score.return_value = 10.0

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.empty_grid, self.ship, self.modules, self.tech, self.player_owned_rewards, forced=True
        )

        mock_apply_pattern.assert_called()
        mock_sa.assert_called_once()
        self.assertEqual(solve_method, "Forced Initial SA (No Pattern Fit)")
        self.assertEqual(solved_bonus, 10.0)

    @patch("src.optimization.core.calculate_grid_score")
    @patch("src.optimization.core.check_all_modules_placed", return_value=True)
    @patch("src.optimization.core._handle_ml_opportunity")
    @patch("src.optimization.core._handle_sa_refine_opportunity")
    @patch("src.optimization.core.find_supercharged_opportunities")
    @patch("src.optimization.core.apply_pattern_to_grid")
    @patch("src.optimization.core.get_solve_map")
    def test_optimize_ml_fallback_to_sa(
        self,
        mock_get_solves,
        mock_apply_pattern,
        mock_find_opportunities,
        mock_handle_sa,
        mock_handle_ml,
        mock_check_placed,
        mock_calculate_score,
    ):
        """Test that optimization falls back to SA when ML refinement fails."""
        # --- Setup Mocks ---
        # 1. Pattern matching succeeds and gives a base score
        pattern_grid = self.sc_grid.copy()
        pattern_grid.set_module(0, 0, "PE")
        pattern_grid.set_tech(0, 0, self.tech)
        mock_apply_pattern.return_value = (pattern_grid, 10)
        mock_get_solves.return_value = sample_solves[self.ship]

        # 2. An opportunity window is found
        mock_find_opportunities.return_value = (0, 0, 4, 3)

        # 3. ML refinement fails (returns None)
        mock_handle_ml.return_value = (None, 0.0)

        # 4. SA refinement succeeds
        sa_grid = self.sc_grid.copy()
        sa_grid.set_module(1, 1, "PE")
        sa_grid.set_tech(1, 1, self.tech)
        mock_handle_sa.return_value = (sa_grid, 25.0)

        # 5. Mock score calculations to prevent final check from overriding bonus
        def score_side_effect(grid, tech):
            # If this is the grid from the SA mock, return the SA score
            if grid.get_cell(1, 1).get("module") == "PE":
                return 25.0
            # Otherwise, it's a grid from the pattern step, return its score
            else:
                return 10.0
        mock_calculate_score.side_effect = score_side_effect


        # --- Run Optimization ---
        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.sc_grid,
            self.ship,
            self.modules,
            self.tech,
            self.player_owned_rewards,
        )

        # --- Assertions ---
        mock_handle_ml.assert_called_once()
        mock_handle_sa.assert_called_once()
        self.assertEqual(solve_method, "ML->SA/Refine Fallback")
        self.assertEqual(solved_bonus, 25.0)
        self.assertIsNotNone(result_grid)
        # Check if the final grid is the one from SA
        self.assertEqual(result_grid.get_cell(1, 1)["module"], "PE")


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

        # Call get_tech_modules with solve_type=None
        from src.modules_utils import get_tech_modules
        result_modules = get_tech_modules(
            mock_ship_modules, "test_ship", "test_tech", []
        )

        # Assert that it returns modules from the untyped definition
        self.assertIsNotNone(result_modules)
        self.assertEqual(len(result_modules), 1)
        self.assertEqual(result_modules[0]["id"], "MOD_B")

# --- Run Tests ---
if __name__ == "__main__":
    # Use the standard unittest main runner
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
