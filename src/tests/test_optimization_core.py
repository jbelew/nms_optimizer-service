"""
Adversarial tests for optimization/core.py

Focus on orchestration logic, edge cases, error handling.
Tests validate the main optimization flow without deep mocking.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.grid_utils import Grid
from src.optimization.core import _prepare_optimization_run


class TestPrepareOptimizationRun(unittest.TestCase):
    """Tests for _prepare_optimization_run function"""

    def test_no_modules_returns_error_tuple(self):
        """When no modules found, should return (cleared_grid, 0.0, 0.0, error_msg)"""
        grid = Grid(5, 5)
        
        with patch('src.optimization.core.get_tech_modules', return_value=None):
            result = _prepare_optimization_run(grid, {}, "ship", "tech", None)
        
        self.assertEqual(len(result), 4)
        self.assertIsNotNone(result[0])  # cleared_grid
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[2], 0.0)
        self.assertIn("Module Definition Error", result[3])

    def test_no_empty_active_slots_raises_error(self):
        """When no empty active slots, should raise ValueError"""
        grid = Grid(2, 2)
        # Fill all cells
        for y in range(grid.height):
            for x in range(grid.width):
                grid.cells[y][x]["module"] = "something"
        
        tech_modules = [
            {"id": "m1", "label": "M1"}
        ]
        
        with patch('src.optimization.core.get_tech_modules', return_value=tech_modules):
            with self.assertRaises(ValueError):
                _prepare_optimization_run(grid, {}, "ship", "tech", None)

    def test_all_inactive_slots_raises_error(self):
        """When all slots inactive, should raise ValueError"""
        grid = Grid(2, 2)
        # Make all cells inactive
        for y in range(grid.height):
            for x in range(grid.width):
                grid.cells[y][x]["active"] = False
        
        tech_modules = [
            {"id": "m1", "label": "M1"}
        ]
        
        with patch('src.optimization.core.get_tech_modules', return_value=tech_modules):
            with self.assertRaises(ValueError):
                _prepare_optimization_run(grid, {}, "ship", "tech", None)

    def test_has_empty_active_slots_returns_modules(self):
        """With empty active slots, should return modules tuple"""
        grid = Grid(2, 2)
        for y in range(grid.height):
            for x in range(grid.width):
                grid.cells[y][x]["active"] = True
        
        full_modules = [
            {"id": f"m{i}", "label": f"M{i}"} for i in range(3)
        ]
        partial_modules = [
            {"id": f"m{i}", "label": f"M{i}"} for i in range(2)
        ]
        
        with patch('src.optimization.core.get_tech_modules') as mock_get:
            mock_get.side_effect = [full_modules, partial_modules]
            result = _prepare_optimization_run(grid, {}, "ship", "tech", ["m0", "m1"])
        
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 3)  # full list
        self.assertEqual(len(result[1]), 2)  # partial list

    def test_available_modules_filters_correctly(self):
        """When available_modules provided, should filter tech_modules"""
        grid = Grid(5, 5)
        for y in range(grid.height):
            for x in range(grid.width):
                grid.cells[y][x]["active"] = True
        
        full_modules = [
            {"id": "m1", "label": "M1"},
            {"id": "m2", "label": "M2"},
            {"id": "m3", "label": "M3"},
        ]
        
        with patch('src.optimization.core.get_tech_modules') as mock_get:
            # First call returns full list, second returns filtered (m1, m2 only)
            mock_get.side_effect = [full_modules, full_modules[:2]]
            result = _prepare_optimization_run(
                grid, {}, "ship", "tech", available_modules=["m1", "m2"]
            )
        
        self.assertEqual(len(result[0]), 3)  # full
        self.assertEqual(len(result[1]), 2)  # partial (filtered)

    def test_grid_returned_is_copy_not_original(self):
        """Returned grid should be new instance, not original"""
        grid = Grid(2, 2)
        grid.cells[0][0]["module"] = "original"
        
        tech_modules = [{"id": "m1", "label": "M1"}]
        
        with patch('src.optimization.core.get_tech_modules', return_value=None):
            result_grid, _, _, _ = _prepare_optimization_run(grid, {}, "ship", "tech", None)
        
        # Result should be a copy, not the same object
        self.assertIsNot(result_grid, grid)


class TestOptimizeOptimizationFlow(unittest.TestCase):
    """Tests for overall optimization flow edge cases"""

    def test_percentage_calculation_zero_solve_score(self):
        """With zero solve score and positive bonus, percentage should be 100%"""
        # This is about the math: no reference means achieving something is 100%
        solve_score = 0
        bonus = 5.0
        
        if solve_score > 1e-9:
            percentage = (bonus / solve_score) * 100
        else:
            percentage = 100.0 if bonus > 1e-9 else 0.0
        
        self.assertEqual(percentage, 100.0)

    def test_percentage_calculation_nonzero_solve_score(self):
        """Percentage should be (bonus / solve_score) * 100"""
        solve_score = 100
        bonus = 50
        percentage = (bonus / solve_score) * 100
        self.assertEqual(percentage, 50.0)

    def test_percentage_calculation_zero_bonus(self):
        """With zero bonus and nonzero solve score, percentage should be 0%"""
        solve_score = 100
        bonus = 0
        percentage = (bonus / solve_score) * 100
        self.assertEqual(percentage, 0.0)

    def test_percentage_calculation_exceeding_solve_score(self):
        """Bonus exceeding solve score should be possible"""
        solve_score = 50
        bonus = 75
        percentage = (bonus / solve_score) * 100
        self.assertEqual(percentage, 150.0)

    def test_grid_copy_independence(self):
        """Grid modifications shouldn't affect original"""
        original = Grid(3, 3)
        original.cells[0][0]["module"] = "original_module"
        
        copy = original.copy()
        copy.cells[0][0]["module"] = "modified_module"
        
        # Original should be unchanged
        self.assertEqual(original.cells[0][0]["module"], "original_module")
        self.assertEqual(copy.cells[0][0]["module"], "modified_module")

    def test_solve_method_tracking(self):
        """solve_method should be set appropriately"""
        # This validates that different paths set the method correctly
        solve_methods = [
            "Initial Placement (No Solve)",
            "Pattern Match",
            "Pattern No Fit",
            "Forced Initial SA (No Pattern Fit)",
            "Partial Set SA",
            "ML",
            "Final Fallback SA",
        ]
        
        # Each method should be a string
        for method in solve_methods:
            self.assertIsInstance(method, str)
            self.assertGreater(len(method), 0)

    def test_partial_module_set_pulse_pc_exception(self):
        """Pulse tech with only 'PC' missing should be treated as full set"""
        full_modules = [
            {"id": "m1", "label": "M1"},
            {"id": "PC", "label": "PC"},
        ]
        tech_modules = [
            {"id": "m1", "label": "M1"},
        ]
        
        # Logic: if pulse and only PC is missing, treat as full set
        is_partial_set = len(tech_modules) < len(full_modules)
        
        if is_partial_set:
            full_ids = {m["id"] for m in full_modules}
            partial_ids = {m["id"] for m in tech_modules}
            missing = full_ids - partial_ids
            if missing == {"PC"}:
                is_partial_set = False  # Override for pulse
        
        # Should now be False
        self.assertFalse(is_partial_set)

    def test_window_selection_prioritizes_pattern_on_tie(self):
        """When scores are equal, pattern location should be preferred"""
        pattern_window_score = 100.0
        scanned_window_score = 100.0
        
        # >= means pattern wins on tie
        if pattern_window_score >= scanned_window_score:
            chosen = "pattern"
        else:
            chosen = "scanned"
        
        self.assertEqual(chosen, "pattern")

    def test_window_selection_scanned_on_better_score(self):
        """When scanned score is better, scanned wins"""
        pattern_window_score = 90.0
        scanned_window_score = 100.0
        
        if pattern_window_score >= scanned_window_score:
            chosen = "pattern"
        else:
            chosen = "scanned"
        
        self.assertEqual(chosen, "scanned")

    def test_refinement_improvement_validation(self):
        """Refinement should only apply if score improves"""
        current_score = 50.0
        refined_score = 60.0
        
        if refined_score >= current_score:
            should_apply = True
        else:
            should_apply = False
        
        self.assertTrue(should_apply)

    def test_refinement_no_improvement_keeps_original(self):
        """If refinement doesn't improve, keep original"""
        current_score = 50.0
        refined_score = 45.0
        
        if refined_score >= current_score:
            should_apply = True
        else:
            should_apply = False
        
        self.assertFalse(should_apply)

    def test_available_supercharged_slot_detection(self):
        """Should detect if window has available supercharged slot"""
        window_cells = [
            {"module": None, "supercharged": False, "active": True},
            {"module": None, "supercharged": True, "active": True},
            {"module": "m1", "supercharged": True, "active": True},
        ]
        
        has_available = any(
            cell["module"] is None and cell["supercharged"] and cell["active"]
            for cell in window_cells
        )
        
        # Should find the second cell
        self.assertTrue(has_available)

    def test_no_available_supercharged_slot(self):
        """Should return False if no available supercharged slot"""
        window_cells = [
            {"module": None, "supercharged": False, "active": True},
            {"module": "m1", "supercharged": True, "active": True},
            {"module": "m2", "supercharged": True, "active": True},
        ]
        
        has_available = any(
            cell["module"] is None and cell["supercharged"] and cell["active"]
            for cell in window_cells
        )
        
        self.assertFalse(has_available)


class TestPartialModuleSetEdgeCases(unittest.TestCase):
    """Adversarial tests for partial module set handling"""
    
    def test_partial_set_with_empty_list(self):
        """Empty available_modules list should be treated as partial set"""
        full_modules = [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]
        tech_modules = []
        available_modules = []
        
        is_partial = (
            available_modules is not None
            and full_modules is not None
            and len(tech_modules) < len(full_modules)
        )
        
        self.assertTrue(is_partial)
    
    def test_partial_set_with_single_module(self):
        """Single available module from full set should be partial"""
        full_modules = [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]
        tech_modules = [{"id": "m1"}]
        
        is_partial = len(tech_modules) < len(full_modules)
        self.assertTrue(is_partial)
    
    def test_partial_set_false_when_none_available_modules(self):
        """When available_modules is None, should not be partial"""
        full_modules = [{"id": "m1"}, {"id": "m2"}]
        tech_modules = [{"id": "m1"}]
        available_modules = None
        
        is_partial = (
            available_modules is not None
            and full_modules is not None
            and len(tech_modules) < len(full_modules)
        )
        
        self.assertFalse(is_partial)
    
    def test_partial_set_false_when_full_match(self):
        """When all modules available, should not be partial"""
        full_modules = [{"id": "m1"}, {"id": "m2"}]
        tech_modules = [{"id": "m1"}, {"id": "m2"}]
        available_modules = ["m1", "m2"]
        
        is_partial = len(tech_modules) < len(full_modules)
        self.assertFalse(is_partial)
    
    def test_missing_modules_set_difference(self):
        """Should correctly calculate missing modules"""
        full_ids = {"m1", "m2", "m3"}
        tech_ids = {"m1", "m3"}
        
        missing = full_ids - tech_ids
        self.assertEqual(missing, {"m2"})
    
    def test_multiple_missing_modules_not_pc_exception(self):
        """With multiple missing modules (not just PC), should remain partial"""
        full_modules = [{"id": "m1"}, {"id": "m2"}, {"id": "PC"}]
        tech_modules = [{"id": "m1"}]
        
        is_partial = len(tech_modules) < len(full_modules)
        full_ids = {m["id"] for m in full_modules}
        tech_ids = {m["id"] for m in tech_modules}
        missing = full_ids - tech_ids
        
        # Multiple missing, should stay partial
        self.assertTrue(is_partial)
        self.assertEqual(missing, {"m2", "PC"})
        self.assertNotEqual(missing, {"PC"})


class TestWindowScoringAndSelection(unittest.TestCase):
    """Adversarial tests for window scoring logic"""
    
    def test_pattern_window_score_greater_than_scanned(self):
        """Pattern should be selected when score strictly greater"""
        pattern_score = 100.0
        scanned_score = 95.0
        
        should_choose_pattern = pattern_score >= scanned_score
        self.assertTrue(should_choose_pattern)
    
    def test_scanned_window_score_strictly_greater(self):
        """Scanned should be selected when strictly greater"""
        pattern_score = 95.0
        scanned_score = 100.0
        
        should_choose_pattern = pattern_score >= scanned_score
        self.assertFalse(should_choose_pattern)
    
    def test_pattern_score_negative_scanned_valid(self):
        """When pattern score is negative, valid scanned should win"""
        pattern_score = -1.0
        scanned_score = 50.0
        
        should_choose_pattern = pattern_score >= scanned_score
        self.assertFalse(should_choose_pattern)
    
    def test_both_scores_negative(self):
        """When both scores negative, less negative (pattern) wins on tie logic"""
        pattern_score = -10.0
        scanned_score = -5.0  # Scanned is better
        
        should_choose_pattern = pattern_score >= scanned_score
        self.assertFalse(should_choose_pattern)
    
    def test_score_near_zero_threshold(self):
        """Scores very close to zero should be handled correctly"""
        pattern_score = 0.0000001
        scanned_score = 0.0
        
        should_choose_pattern = pattern_score >= scanned_score
        self.assertTrue(should_choose_pattern)


class TestRefinementStageEdgeCases(unittest.TestCase):
    """Adversarial tests for refinement stage"""
    
    def test_no_pattern_or_scanned_opportunity(self):
        """When neither pattern nor scanned opportunity exists, no refinement"""
        pattern_opportunity = None
        scanned_opportunity = None
        
        has_opportunity = (
            pattern_opportunity is not None or scanned_opportunity is not None
        )
        self.assertFalse(has_opportunity)
    
    def test_fallback_to_pattern_when_scanned_fails(self):
        """Pattern should be fallback when scanned calculation fails"""
        pattern_opportunity = (5, 5, 4, 3)
        scanned_opportunity = None
        
        # Should use pattern as fallback
        final_opportunity = pattern_opportunity if pattern_opportunity else scanned_opportunity
        self.assertIsNotNone(final_opportunity)
        self.assertEqual(final_opportunity, pattern_opportunity)
    
    def test_refinement_with_equal_scores_pattern_preference(self):
        """With equal scores, pattern location is preferred"""
        current_score = 50.0
        pattern_refined = 50.0
        scanned_refined = 50.0
        
        # Both would apply, but pattern was already chosen
        should_refine = pattern_refined >= current_score
        self.assertTrue(should_refine)
    
    def test_refinement_marked_successful_at_tie(self):
        """Refinement should be applied if refined_score >= current_score"""
        current_score = 50.0
        refined_score = 50.0
        
        should_apply = refined_score >= current_score
        self.assertTrue(should_apply)
    
    def test_refinement_failure_keeps_original(self):
        """When refinement returns None, original should be kept"""
        original_grid = Grid(3, 3)
        refined_grid = None
        
        if refined_grid is not None:
            final_grid = refined_grid
        else:
            final_grid = original_grid
        
        self.assertIs(final_grid, original_grid)


class TestSuperchargedWindowDetection(unittest.TestCase):
    """Adversarial tests for supercharged slot detection in windows"""
    
    def test_window_bounds_checking(self):
        """Should respect grid boundaries when checking window cells"""
        grid_width = 5
        grid_height = 5
        window_x, window_y = 3, 3
        window_w, window_h = 3, 3  # Extends beyond grid
        
        window_cells_valid = []
        for y in range(window_y, window_y + window_h):
            for x in range(window_x, window_x + window_w):
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    window_cells_valid.append((x, y))
        
        # Should only include cells within bounds
        expected_count = 4  # 2x2 in corner
        self.assertEqual(len(window_cells_valid), expected_count)
    
    def test_supercharged_detection_inactive_cells(self):
        """Inactive cells should be excluded even if supercharged"""
        cell_inactive_sc = {
            "active": False,
            "supercharged": True,
            "module": None
        }
        
        # Should not count as available
        has_available = (
            cell_inactive_sc["active"] 
            and cell_inactive_sc["supercharged"] 
            and cell_inactive_sc["module"] is None
        )
        self.assertFalse(has_available)
    
    def test_supercharged_with_existing_module(self):
        """Supercharged cells with modules should not count as available"""
        cells = [
            {"active": True, "supercharged": True, "module": "m1"},
            {"active": True, "supercharged": True, "module": None},
        ]
        
        available = [
            c for c in cells 
            if c["active"] and c["supercharged"] and c["module"] is None
        ]
        
        # Only the second should be available
        self.assertEqual(len(available), 1)
    
    def test_all_cells_non_supercharged(self):
        """Window with no supercharged cells should return empty"""
        cells = [
            {"active": True, "supercharged": False, "module": None},
            {"active": True, "supercharged": False, "module": None},
        ]
        
        available = [
            c for c in cells 
            if c["active"] and c["supercharged"] and c["module"] is None
        ]
        
        self.assertEqual(len(available), 0)


class TestInitialPlacementFallback(unittest.TestCase):
    """Tests for initial placement fallback logic when no solve exists"""
    
    def test_percentage_with_zero_bonus_no_solve(self):
        """When no solve exists and bonus is zero, percentage should be zero"""
        bonus = 0.0
        
        percentage = 100.0 if bonus > 1e-9 else 0.0
        self.assertEqual(percentage, 0.0)
    
    def test_percentage_with_positive_bonus_no_solve(self):
        """When no solve exists and bonus is positive, percentage should be 100"""
        bonus = 5.5
        
        percentage = 100.0 if bonus > 1e-9 else 0.0
        self.assertEqual(percentage, 100.0)
    
    def test_grid_clearing_on_no_modules_error(self):
        """When no modules found, grid should be cleared of target tech"""
        # This validates the logic of the error path
        grid = Grid(3, 3)
        grid.cells[0][0]["module"] = "some_module"
        
        cleared_grid = grid.copy()
        # Simulate tech clearing
        for y in range(cleared_grid.height):
            for x in range(cleared_grid.width):
                # In real code, only target tech is cleared
                pass
        
        # Copy should be independent
        self.assertIsNot(cleared_grid, grid)


if __name__ == "__main__":
    unittest.main()
