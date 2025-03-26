import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from optimization_algorithms import (
    refine_placement,
    rotate_pattern,
    mirror_pattern_horizontally,
    mirror_pattern_vertically,
    apply_pattern_to_grid,
    get_all_unique_pattern_variations,
    count_adjacent_occupied,
    calculate_pattern_adjacency_score,
    optimize_placement,
    place_all_modules_in_empty_slots,
    find_supercharged_opportunities,
    create_localized_grid,
    apply_localized_grid_changes,
    check_all_modules_placed,
    clear_all_modules_of_tech,
)
from grid_utils import Grid
from modules import modules, solves


class TestOptimizationAlgorithms(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(4, 3)
        self.ship = "Exotic"
        self.tech = "infra"
        self.modules = modules
        self.player_owned_rewards = []

    def test_refine_placement_no_modules(self):
        with patch("optimization_algorithms.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = None
            result_grid, result_bonus = refine_placement(
                self.grid, self.ship, self.modules, self.tech, self.player_owned_rewards
            )
            self.assertIsNone(result_grid)
            self.assertEqual(result_bonus, 0.0)

    def test_refine_placement_success(self):
        with patch("optimization_algorithms.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = [
                {"id": "IK", "type": "core", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "infra.png", "label": "Infraknife Accelerator"},
                {"id": "Xa", "type": "bonus", "bonus": 0.40, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png", "label": "Infraknife Accelerator Upgrade Sigma"},
            ]
            result_grid, result_bonus = refine_placement(
                self.grid, self.ship, self.modules, self.tech, self.player_owned_rewards
            )
            self.assertIsInstance(result_grid, Grid)
            self.assertGreaterEqual(result_bonus, 0.0)

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

    def test_apply_pattern_to_grid_no_overlap(self):
        pattern = {(0, 0): "IK", (1, 0): "Xa"}
        with patch("optimization_algorithms.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = [
                {"id": "IK", "type": "core", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "infra.png", "label": "Infraknife Accelerator"},
                {"id": "Xa", "type": "bonus", "bonus": 0.40, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png", "label": "Infraknife Accelerator Upgrade Sigma"},
            ]
            result, _ = apply_pattern_to_grid(self.grid, pattern, self.modules, self.tech, 0, 0, self.ship, self.player_owned_rewards)
            self.assertEqual(result, 1)
            self.assertEqual(self.grid.get_cell(0, 0)["module"], "IK")
            self.assertEqual(self.grid.get_cell(1, 0)["module"], "Xa")

    def test_apply_pattern_to_grid_with_overlap(self):
        pattern = {(0, 0): "IK", (1, 0): "Xa"}
        self.grid.set_module(0, 0, "RL")
        self.grid.set_tech(0, 0, "rocket")
        with patch("optimization_algorithms.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = [
                {"id": "IK", "type": "core", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "infra.png", "label": "Infraknife Accelerator"},
                {"id": "Xa", "type": "bonus", "bonus": 0.40, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png", "label": "Infraknife Accelerator Upgrade Sigma"},
            ]
            result, _ = apply_pattern_to_grid(self.grid, pattern, self.modules, self.tech, 0, 0, self.ship, self.player_owned_rewards)
            self.assertEqual(result, 0)

    def test_get_all_unique_pattern_variations(self):
        pattern = {(0, 0): "A", (1, 0): "B"}
        variations = get_all_unique_pattern_variations(pattern)
        self.assertGreater(len(variations), 1)

    def test_count_adjacent_occupied(self):
        self.grid.set_module(0, 0, "A")
        self.grid.set_module(1, 0, "B")
        count = count_adjacent_occupied(self.grid, 0, 1)
        self.assertEqual(count, 1)

    def test_calculate_pattern_adjacency_score(self):
        pattern = {(0, 0): "A", (1, 0): "B"}
        self.grid.set_module(2, 0, "C")
        score = calculate_pattern_adjacency_score(self.grid, pattern, 0, 0)
        self.assertEqual(score, 1)

    @patch("optimization_algorithms.filter_solves")
    def test_optimize_placement_no_solve_found(self, mock_filter_solves):
        mock_filter_solves.return_value = {}
        result_grid, result_percentage = optimize_placement(
            self.grid, self.ship, self.modules, self.tech, self.player_owned_rewards
        )
        self.assertIsInstance(result_grid, Grid)
        self.assertEqual(result_percentage, 0)

    @patch("optimization_algorithms.filter_solves")
    @patch("optimization_algorithms.get_all_unique_pattern_variations")
    @patch("optimization_algorithms.apply_pattern_to_grid")
    @patch("optimization_algorithms.calculate_grid_score")
    def test_optimize_placement_solve_found(self, mock_calculate_grid_score, mock_apply_pattern_to_grid, mock_get_all_unique_pattern_variations, mock_filter_solves):
        mock_filter_solves.return_value = {self.ship: {self.tech: {"map": {(0, 0): "IK", (1, 0): "Xa"}, "score": 100}}}
        mock_get_all_unique_pattern_variations.return_value = [{(0, 0): "IK", (1, 0): "Xa"}]
        mock_apply_pattern_to_grid.return_value = (1, 1)
        mock_calculate_grid_score.return_value = 50
        result_grid, result_percentage = optimize_placement(
            self.grid, self.ship, self.modules, self.tech, self.player_owned_rewards
        )
        self.assertIsInstance(result_grid, Grid)
        self.assertEqual(result_percentage, 50)

    def test_place_all_modules_in_empty_slots(self):
        with patch("optimization_algorithms.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = [
                {"id": "IK", "type": "core", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "infra.png", "label": "Infraknife Accelerator"},
                {"id": "Xa", "type": "bonus", "bonus": 0.40, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png", "label": "Infraknife Accelerator Upgrade Sigma"},
                {"id": "Xb", "type": "bonus", "bonus": 0.39, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png", "label": "Infraknife Accelerator Upgrade Tau"},
                {"id": "Xc", "type": "bonus", "bonus": 0.38, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png", "label": "Infraknife Accelerator Upgrade Theta"},
                {"id": "QR", "type": "bonus", "bonus": 0.04, "adjacency": True, "sc_eligible": True, "image": "q-resonator.png", "label": "Q-Resonator"},
                {"id": "FS", "type": "bonus", "bonus": 0.04, "adjacency": True, "sc_eligible": True, "image": "fragment.png", "label": "Fragment Supercharger"},
            ]
            result_grid = place_all_modules_in_empty_slots(self.grid, self.modules, self.ship, self.tech, self.player_owned_rewards)
            self.assertIsNotNone(result_grid.get_cell(0,0)["module"])
            self.assertIsNotNone(result_grid.get_cell(1,0)["module"])
            #self.assertIsNotNone(result_grid.get_cell(2,0)["module"])
            #self.assertIsNotNone(result_grid.get_cell(3,0)["module"])
            #self.assertIsNotNone(result_grid.get_cell(0,1)["module"])

    def test_find_supercharged_opportunities_no_opportunity(self):
        result = find_supercharged_opportunities(self.grid, self.modules, self.ship, self.tech)
        self.assertIsNone(result)

    def test_find_supercharged_opportunities_opportunity(self):
        self.grid.set_supercharged(0, 0, True)
        self.grid.set_module(1, 0, "IK")
        self.grid.set_tech(1, 0, "infra")
        result = find_supercharged_opportunities(self.grid, self.modules, self.ship, self.tech)
        self.assertEqual(result, (0, 0))

    def test_create_localized_grid(self):
        self.grid.set_supercharged(1, 1, True)
        localized_grid, start_x, start_y = create_localized_grid(self.grid, 1, 1)
        self.assertLessEqual(localized_grid.width, 4)
        self.assertLessEqual(localized_grid.width, 4)
        self.assertEqual(start_x, 0)
        self.assertEqual(start_y, 0)
        self.assertTrue(localized_grid.get_cell(1, 1)["supercharged"])

    def test_apply_localized_grid_changes(self):
        localized_grid = Grid(4, 4)
        localized_grid.set_module(1, 1, "IK")
        localized_grid.set_tech(1, 1, "infra")
        apply_localized_grid_changes(self.grid, localized_grid, "infra", 0, 0)
        self.assertEqual(self.grid.get_cell(1, 1)["module"], "IK")
        self.assertEqual(self.grid.get_cell(1, 1)["tech"], "infra")

    def test_check_all_modules_placed_all_placed(self):
        with patch("optimization_algorithms.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = [
                {"id": "IK", "type": "core", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "infra.png", "label": "Infraknife Accelerator"},
                {"id": "Xa", "type": "bonus", "bonus": 0.40, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png", "label": "Infraknife Accelerator Upgrade Sigma"},
            ]
            self.grid.set_module(0, 0, "IK")
            self.grid.set_tech(0, 0, "infra")
            self.grid.set_module(1, 0, "Xa")
            self.grid.set_tech(1, 0, "infra")
            result = check_all_modules_placed(self.grid, self.modules, self.ship, self.tech)
            self.assertTrue(result)

    def test_check_all_modules_placed_not_all_placed(self):
        with patch("optimization_algorithms.get_tech_modules") as mock_get_tech_modules:
            mock_get_tech_modules.return_value = [
                {"id": "IK", "type": "core", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "infra.png", "label": "Infraknife Accelerator"},
                {"id": "Xa", "type": "bonus", "bonus": 0.40, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png", "label": "Infraknife Accelerator Upgrade Sigma"},
            ]
            self.grid.set_module(0, 0, "IK")
            self.grid.set_tech(0, 0, "infra")
            result = check_all_modules_placed(self.grid, self.modules, self.ship, self.tech)
            self.assertFalse(result)

    def test_clear_all_modules_of_tech(self):
        self.grid.set_module(0, 0, "IK")
        self.grid.set_tech(0, 0, "infra")
        clear_all_modules_of_tech(self.grid, "infra")
        self.assertIsNone(self.grid.get_cell(0, 0)["module"])
        self.assertIsNone(self.grid.get_cell(0, 0)["tech"])

if __name__ == "__main__":
    unittest.main()
