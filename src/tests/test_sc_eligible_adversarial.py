"""
Adversarial tests for sc_eligible constraint enforcement in windowing.

These tests verify that the sc_eligible constraint is correctly enforced
in edge cases and adversarial scenarios, particularly for the fixes made to:
1. Default sc_eligible value (False instead of True)
2. Window scanning for non-sc_eligible modules (require_non_supercharge)
3. Mixed eligible/non-eligible module scenarios
"""

import unittest
from unittest.mock import patch
from src.grid_utils import Grid
from src.optimization.windowing import find_supercharged_opportunities, _scan_grid_with_window


class TestScEligibleAdversarial(unittest.TestCase):
    """Adversarial tests for sc_eligible constraint enforcement."""

    def test_non_supercharge_scan_filters_all_supercharged_windows(self):
        """
        When require_non_supercharge=True, _scan_grid_with_window should skip
        any window that contains even one supercharged cell.
        """
        # Create a 6x6 grid with supercharged cells on the right half only
        grid = Grid(6, 6)
        for y in range(6):
            for x in range(6):
                grid.cells[y][x]["active"] = True
                # Right half (x >= 3) is supercharged
                grid.cells[y][x]["supercharged"] = x >= 3

        # Scan with require_non_supercharge=True for a 2x2 window
        # Should find a window in the left half (x < 3)
        score, pos = _scan_grid_with_window(
            grid, 2, 2, 4, "test_tech", require_supercharge=False, require_non_supercharge=True
        )

        # Should find a window
        self.assertIsNotNone(pos)
        x, y = pos

        # Verify the found window contains NO supercharged cells
        for dy in range(2):
            for dx in range(2):
                cell = grid.get_cell(x + dx, y + dy)
                self.assertFalse(
                    cell["supercharged"],
                    f"Found window at ({x},{y}) contains supercharged cell at ({x+dx},{y+dy})",
                )

    def test_non_supercharge_scan_vs_supercharge_scan_different_results(self):
        """
        Scanning with require_supercharge=True should find different windows
        than scanning with require_non_supercharge=True.
        """
        # Create a grid with supercharged cells on the right side
        grid = Grid(6, 4)
        for y in range(4):
            for x in range(6):
                grid.cells[y][x]["active"] = True
                # Right side (x >= 3) is supercharged
                grid.cells[y][x]["supercharged"] = x >= 3

        # Scan for supercharged windows
        score1, pos1 = _scan_grid_with_window(
            grid, 2, 2, 4, "test_tech", require_supercharge=True, require_non_supercharge=False
        )

        # Scan for non-supercharged windows
        score2, pos2 = _scan_grid_with_window(
            grid, 2, 2, 4, "test_tech", require_supercharge=False, require_non_supercharge=True
        )

        # Both should find windows
        self.assertIsNotNone(pos1, "Should find supercharged window")
        self.assertIsNotNone(pos2, "Should find non-supercharged window")

        # Windows should be in different locations
        # Supercharged window should be on the right, non-supercharged on the left
        x1, _ = pos1
        x2, _ = pos2
        # SC window should be further right than non-SC window
        self.assertGreater(x1, x2, "Supercharged window should be to the right of non-supercharged")

    def test_find_supercharged_returns_none_when_all_non_eligible_and_all_supercharged(self):
        """
        When all modules are non-sc_eligible AND all grid cells are supercharged,
        find_supercharged_opportunities should return None (no valid placement exists).
        """
        grid = Grid(3, 3)
        # Make entire grid supercharged
        for y in range(3):
            for x in range(3):
                grid.cells[y][x]["active"] = True
                grid.cells[y][x]["supercharged"] = True

        with patch("src.optimization.windowing.get_tech_modules") as mock_modules:
            # All modules are non-sc_eligible
            mock_modules.return_value = [
                {"id": "M1", "sc_eligible": False},
                {"id": "M2", "sc_eligible": False},
            ]

            result = find_supercharged_opportunities(grid, {}, "ship", "tech")

        self.assertIsNone(result, "Should return None when no non-supercharged slots for non-eligible modules")

    def test_find_supercharged_returns_window_with_mixed_eligible_modules(self):
        """
        When modules are mixed (some eligible, some not), find_supercharged_opportunities
        should return a window with supercharged slots (because at least some modules can use them).
        """
        grid = Grid(5, 5)
        for y in range(5):
            for x in range(5):
                grid.cells[y][x]["active"] = True
                grid.cells[y][x]["supercharged"] = (x + y) % 2 == 0

        with patch("src.optimization.windowing.get_tech_modules") as mock_modules:
            # Mixed: one eligible, one non-eligible
            mock_modules.return_value = [
                {"id": "M1", "sc_eligible": True},  # Can use supercharge
                {"id": "M2", "sc_eligible": False},  # Cannot use supercharge
            ]

            result = find_supercharged_opportunities(grid, {}, "ship", "tech")

        # Should find a window (will try to find supercharged since at least one module can use it)
        self.assertIsNotNone(result, "Should find window when at least one module is sc_eligible")

    def test_sc_eligible_default_false_vs_unspecified(self):
        """
        Verify that modules without sc_eligible specified default to False (non-eligible),
        not True. This is critical for the fix.
        """
        grid = Grid(5, 5)
        for y in range(5):
            for x in range(5):
                grid.cells[y][x]["active"] = True
                grid.cells[y][x]["supercharged"] = True  # All supercharged

        # Module without sc_eligible specified
        with patch("src.optimization.windowing.get_tech_modules") as mock_modules:
            mock_modules.return_value = [{"id": "M1"}]  # No sc_eligible field

            result = find_supercharged_opportunities(grid, {}, "ship", "tech")

        # Should return None because module defaults to non-eligible
        # and there are no non-supercharged slots
        self.assertIsNone(result, "Module without sc_eligible should default to non-eligible")

    def test_require_non_supercharge_with_partial_supercharge(self):
        """
        Test require_non_supercharge with a grid that has both supercharged
        and non-supercharged cells mixed throughout.
        """
        grid = Grid(7, 3)
        for y in range(3):
            for x in range(7):
                grid.cells[y][x]["active"] = True
                # Pattern: SC cells at x=0,2,4,6
                grid.cells[y][x]["supercharged"] = x % 2 == 0

        # Scan for 2x2 windows with no supercharge
        # Valid positions: (1,0), (1,1), (3,0), (3,1), (5,0), (5,1)
        score, pos = _scan_grid_with_window(
            grid, 2, 2, 4, "tech", require_supercharge=False, require_non_supercharge=True
        )

        if pos:
            x, y = pos
            # Verify window has no supercharged cells
            for dy in range(2):
                for dx in range(2):
                    cell = grid.get_cell(x + dx, y + dy)
                    self.assertFalse(
                        cell["supercharged"],
                        f"Window at ({x},{y}) should have no SC cells",
                    )

    def test_require_both_flags_never_valid(self):
        """
        Setting both require_supercharge=True and require_non_supercharge=True
        would be contradictory. Verify the logic handles this gracefully.
        """
        grid = Grid(3, 3)
        for y in range(3):
            for x in range(3):
                grid.cells[y][x]["active"] = True
                grid.cells[y][x]["supercharged"] = x < 2  # Left side SC, right side not

        # Call with both flags true (contradictory)
        # The require_non_supercharge check comes after require_supercharge in the code
        # so it should take precedence
        score, pos = _scan_grid_with_window(
            grid, 2, 1, 2, "tech", require_supercharge=True, require_non_supercharge=True  # Want SC  # Want no SC
        )

        # Should return None (no window can satisfy both)
        self.assertIsNone(pos, "No window can have both SC and no SC cells")

    def test_window_boundary_with_sc_eligible_constraint(self):
        """
        Test windows near grid boundaries with sc_eligible constraint.
        """
        grid = Grid(4, 4)
        for y in range(4):
            for x in range(4):
                grid.cells[y][x]["active"] = True
                # Top-right corner is supercharged
                grid.cells[y][x]["supercharged"] = x >= 2 and y <= 1

        with patch("src.optimization.windowing.get_tech_modules") as mock_modules:
            # All non-eligible
            mock_modules.return_value = [
                {"id": "M1", "sc_eligible": False},
                {"id": "M2", "sc_eligible": False},
            ]

            result = find_supercharged_opportunities(grid, {}, "ship", "tech")

        # Should find a window in the non-SC region (left or bottom)
        if result:
            x, y, w, h = result
            # Verify no part of window includes top-right SC region
            for dy in range(h):
                for dx in range(w):
                    cell = grid.get_cell(x + dx, y + dy)
                    self.assertFalse(
                        cell["supercharged"],
                        "Window should not include SC cells",
                    )

    def test_explicit_false_vs_missing_sc_eligible(self):
        """
        Verify that sc_eligible: False and missing sc_eligible are treated the same.
        """
        grid = Grid(5, 5)
        for y in range(5):
            for x in range(5):
                grid.cells[y][x]["active"] = True
                grid.cells[y][x]["supercharged"] = True

        # Test 1: sc_eligible explicitly False
        with patch("src.optimization.windowing.get_tech_modules") as mock_modules:
            mock_modules.return_value = [{"id": "M1", "sc_eligible": False}]
            result1 = find_supercharged_opportunities(grid, {}, "ship", "tech")

        # Test 2: sc_eligible not specified (should default to False)
        with patch("src.optimization.windowing.get_tech_modules") as mock_modules:
            mock_modules.return_value = [{"id": "M1"}]  # No sc_eligible
            result2 = find_supercharged_opportunities(grid, {}, "ship", "tech")

        # Both should return None (same behavior)
        self.assertIsNone(result1, "Explicit sc_eligible: False should return None")
        self.assertIsNone(result2, "Missing sc_eligible should return None (defaults to False)")

    def test_scanning_with_available_modules_constraint(self):
        """
        Test that window scanning respects sc_eligible when available_modules is restricted.
        """
        grid = Grid(5, 5)
        for y in range(5):
            for x in range(5):
                grid.cells[y][x]["active"] = True
                grid.cells[y][x]["supercharged"] = True

        # Full module list has one eligible and one not
        full_modules = [
            {"id": "M1", "sc_eligible": True},
            {"id": "M2", "sc_eligible": False},
        ]

        # But available_modules only includes the non-eligible one
        with patch("src.optimization.windowing.get_tech_modules") as mock_modules:
            mock_modules.return_value = [full_modules[1]]  # Only M2 (non-eligible)

            result = find_supercharged_opportunities(grid, {}, "ship", "tech")

        # Should return None because the available module can't use supercharge
        self.assertIsNone(
            result,
            "Should return None when all available modules are non-sc_eligible",
        )


class TestScEligibleWindowSizeWithConstraint(unittest.TestCase):
    """Tests for window sizing with sc_eligible constraints."""

    def test_window_size_selection_with_sc_eligible_modules(self):
        """
        Verify that window size selection considers sc_eligible.
        With all non-eligible modules, window should not prefer supercharged regions.
        """
        grid = Grid(6, 6)
        for y in range(6):
            for x in range(6):
                grid.cells[y][x]["active"] = True
                # Right side has more supercharge
                grid.cells[y][x]["supercharged"] = x >= 3

        with patch("src.optimization.windowing.get_tech_modules") as mock_modules:
            # All non-eligible modules
            mock_modules.return_value = [{"id": f"M{i}", "sc_eligible": False} for i in range(4)]

            result = find_supercharged_opportunities(grid, {}, "ship", "tech")

        if result:
            x, y, w, h = result
            # Window should be on the left (non-SC) side or avoid SC cells
            for dy in range(h):
                for dx in range(w):
                    cell = grid.get_cell(x + dx, y + dy)
                    self.assertFalse(
                        cell["supercharged"],
                        "Window for non-eligible modules should avoid SC cells",
                    )

    def test_rotated_window_with_mixed_supercharge(self):
        """
        Test that both original and rotated window dimensions are checked
        correctly with require_non_supercharge.
        """
        grid = Grid(8, 5)
        for y in range(5):
            for x in range(8):
                grid.cells[y][x]["active"] = True
                # Vertical stripe of SC at x >= 4
                grid.cells[y][x]["supercharged"] = x >= 4

        # Original: 3x2 window
        # Rotated: 2x3 window
        # Both should be able to fit in the left side (x < 4)

        score_orig, pos_orig = _scan_grid_with_window(
            grid, 3, 2, 6, "tech", require_supercharge=False, require_non_supercharge=True
        )

        score_rot, pos_rot = _scan_grid_with_window(
            grid, 2, 3, 6, "tech", require_supercharge=False, require_non_supercharge=True
        )

        # Both should find valid windows
        self.assertIsNotNone(pos_orig, "3x2 window should fit in left side")
        self.assertIsNotNone(pos_rot, "2x3 window should fit in left side")

        # Verify neither window includes supercharged cells
        if pos_orig:
            x, y = pos_orig
            for dx in range(3):
                for dy in range(2):
                    cell = grid.get_cell(x + dx, y + dy)
                    self.assertFalse(cell["supercharged"], "3x2 window should have no SC cells")

        if pos_rot:
            x, y = pos_rot
            for dx in range(2):
                for dy in range(3):
                    cell = grid.get_cell(x + dx, y + dy)
                    self.assertFalse(cell["supercharged"], "2x3 window should have no SC cells")


class TestScEligibleFallbackPlacement(unittest.TestCase):
    """Tests for fallback placement of non-sc_eligible modules in supercharged slots."""

    def test_non_eligible_module_placed_in_sc_slot_when_no_alternatives(self):
        """
        When all available slots are supercharged and module is non-sc_eligible,
        the module should still be placed in a supercharged slot as fallback.
        """
        from src.optimization.core import optimize_placement
        from src.grid_utils import Grid

        # Create a 3x3 grid with all cells supercharged and active
        grid = Grid(3, 3)
        for y in range(3):
            for x in range(3):
                grid.cells[y][x]["active"] = True
                grid.cells[y][x]["supercharged"] = True

        # One non-sc_eligible module
        modules = {
            "types": {
                "regular": [
                    {
                        "key": "test_tech",
                        "modules": [
                            {
                                "id": "M1",
                                "label": "Test",
                                "type": "bonus",
                                "bonus": 1.0,
                                "adjacency": "no_adjacency",
                                "sc_eligible": False,
                                "image": None,
                            }
                        ],
                    }
                ]
            }
        }

        # Run optimization
        result_grid, percentage, bonus, method = optimize_placement(grid, "corvette", modules, "test_tech", forced=True)

        # Module should be placed despite supercharged constraint
        self.assertIsNotNone(result_grid)

        # Verify module was actually placed
        module_found = False
        for y in range(result_grid.height):
            for x in range(result_grid.width):
                if result_grid.get_cell(x, y)["module"] == "M1":
                    module_found = True
                    # It should be in a supercharged slot
                    self.assertTrue(
                        result_grid.get_cell(x, y)["supercharged"],
                        "Non-eligible module should be placed in SC slot as fallback",
                    )

        self.assertTrue(module_found, "Module should have been placed")

    def test_non_eligible_module_prefers_non_sc_then_falls_back(self):
        """
        Non-sc_eligible module should prefer non-supercharged slots,
        but use supercharged slots only when necessary.
        """
        from src.optimization.core import optimize_placement
        from src.grid_utils import Grid

        # Create a 5x5 grid with mixed supercharge
        grid = Grid(5, 5)
        for y in range(5):
            for x in range(5):
                grid.cells[y][x]["active"] = True
                # Left side (x < 2) is non-supercharged, right side is supercharged
                grid.cells[y][x]["supercharged"] = x >= 2

        # One non-sc_eligible module
        modules = {
            "types": {
                "regular": [
                    {
                        "key": "test_tech",
                        "modules": [
                            {
                                "id": "M1",
                                "label": "Test",
                                "type": "bonus",
                                "bonus": 1.0,
                                "adjacency": "no_adjacency",
                                "sc_eligible": False,
                                "image": None,
                            }
                        ],
                    }
                ]
            }
        }

        # Run optimization
        result_grid, _, _, _ = optimize_placement(grid, "test_ship", modules, "test_tech", forced=True)

        # Module should be placed in non-supercharged slot (preference)
        module_found = False
        module_in_sc = False
        for y in range(result_grid.height):
            for x in range(result_grid.width):
                if result_grid.get_cell(x, y)["module"] == "M1":
                    module_found = True
                    if result_grid.get_cell(x, y)["supercharged"]:
                        module_in_sc = True

        self.assertTrue(module_found, "Module should be placed")
        # With available non-SC slots, it should not use SC slots
        self.assertFalse(module_in_sc, "Module should prefer non-SC slots when available")

    def test_multiple_non_eligible_modules_fallback_to_sc(self):
        """
        Multiple non-sc_eligible modules should fallback to supercharged slots
        when all non-supercharged slots are exhausted.
        """
        from src.optimization.core import optimize_placement
        from src.grid_utils import Grid

        # Create a 5x5 grid: 1 non-supercharged slot, rest supercharged
        grid = Grid(5, 5)
        for y in range(5):
            for x in range(5):
                grid.cells[y][x]["active"] = True
                grid.cells[y][x]["supercharged"] = not (x == 0 and y == 0)

        # Three non-sc_eligible modules
        modules = {
            "types": {
                "regular": [
                    {
                        "key": "test_tech",
                        "modules": [
                            {
                                "id": f"M{i}",
                                "label": f"Test{i}",
                                "type": "bonus",
                                "bonus": 1.0,
                                "adjacency": "no_adjacency",
                                "sc_eligible": False,
                                "image": None,
                            }
                            for i in range(3)
                        ],
                    }
                ]
            }
        }

        # Run optimization
        result_grid, _, _, _ = optimize_placement(grid, "test_ship", modules, "test_tech", forced=True)

        # All three modules should be placed
        placed_count = 0
        placed_in_sc = 0
        for y in range(result_grid.height):
            for x in range(result_grid.width):
                cell = result_grid.get_cell(x, y)
                if cell["module"] and "M" in cell["module"]:
                    placed_count += 1
                    if cell["supercharged"]:
                        placed_in_sc += 1

        self.assertEqual(placed_count, 3, "All 3 modules should be placed")
        # At least 2 should be in SC slots (since only 1 non-SC slot available)
        self.assertGreaterEqual(placed_in_sc, 2, "At least 2 modules should fallback to SC slots")


if __name__ == "__main__":
    unittest.main()
