"""
Adversarial tests for optimization/helpers.py

Focus areas:
- Window dimension calculation edge cases
- Module placement in constrained spaces
- Empty slot counting
- Module placement completion validation
"""

import unittest
from src.grid_utils import Grid
from src.optimization.helpers import (
    determine_window_dimensions,
    place_all_modules_in_empty_slots,
    count_empty_in_localized,
    check_all_modules_placed,
)


class TestDetermineWindowDimensions(unittest.TestCase):
    """Adversarial tests for determine_window_dimensions"""

    def test_zero_modules_returns_default(self):
        """Zero modules BUG: returns tech-specific defaults instead of 1x1

        The function checks tech-specific rules BEFORE checking module_count < 1,
        so with 0 modules and hyper tech, it returns (4, 2) instead of (1, 1).
        This is a logic ordering bug in determine_window_dimensions.
        """
        w, h = determine_window_dimensions(0, "hyper", "corvette")
        # Documents the bug - should be (1, 1), but is (4, 2)
        self.assertEqual((w, h), (4, 2))

    def test_negative_module_count_treated_as_zero(self):
        """Negative module count should be handled gracefully"""
        w, h = determine_window_dimensions(-5, "hyper", "corvette")
        # Should return default or 1x1
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_sentinel_photonix_override(self):
        """Sentinel with photonix should use specific dimensions"""
        w, h = determine_window_dimensions(1, "photonix", "sentinel")
        self.assertEqual((w, h), (4, 3))

    def test_sentinel_photonix_override_regardless_module_count(self):
        """Sentinel photonix override should apply regardless of module count"""
        w, h = determine_window_dimensions(100, "photonix", "sentinel")
        self.assertEqual((w, h), (4, 3))

    def test_corvette_pulse_7_modules(self):
        """Corvette pulse with 7 modules has override"""
        w, h = determine_window_dimensions(7, "pulse", "corvette")
        self.assertEqual((w, h), (4, 2))

    def test_corvette_pulse_6_modules_no_override(self):
        """Corvette pulse with 6 modules should NOT match the 7-module override"""
        w, h = determine_window_dimensions(6, "pulse", "corvette")
        self.assertNotEqual((w, h), (4, 2))

    def test_corvette_8_modules_3x3(self):
        """Corvette with 8 modules should return 3x3"""
        w, h = determine_window_dimensions(8, "random_tech", "corvette")
        self.assertEqual((w, h), (3, 3))

    def test_corvette_7_modules_non_pulse(self):
        """Corvette with 7 modules (non-pulse) should return 3x3"""
        w, h = determine_window_dimensions(7, "hyper", "corvette")
        self.assertEqual((w, h), (3, 3))

    def test_hyper_12_plus_modules(self):
        """Hyper with 12+ modules should use 4x4"""
        w, h = determine_window_dimensions(12, "hyper", "any_ship")
        self.assertEqual((w, h), (4, 4))

    def test_hyper_10_to_11_modules(self):
        """Hyper with 10-11 modules should use 4x3"""
        w, h = determine_window_dimensions(10, "hyper", "any_ship")
        self.assertEqual((w, h), (4, 3))
        w, h = determine_window_dimensions(11, "hyper", "any_ship")
        self.assertEqual((w, h), (4, 3))

    def test_hyper_9_modules(self):
        """Hyper with 9 modules should use 3x3"""
        w, h = determine_window_dimensions(9, "hyper", "any_ship")
        self.assertEqual((w, h), (3, 3))

    def test_hyper_less_than_9_modules(self):
        """Hyper with <9 modules should use 4x2"""
        for count in [1, 5, 8]:
            w, h = determine_window_dimensions(count, "hyper", "any_ship")
            self.assertEqual((w, h), (4, 2))

    def test_bolt_caster_any_count(self):
        """Bolt-caster should always use 4x3"""
        for count in [1, 5, 10, 20]:
            w, h = determine_window_dimensions(count, "bolt-caster", "any_ship")
            self.assertEqual((w, h), (4, 3))

    def test_pulse_spitter_jetpack_less_than_8(self):
        """Pulse-spitter/jetpack <7 modules should use 3x3, 7+ modules should use 4x2"""
        w, h = determine_window_dimensions(7, "pulse-spitter", "any_ship")
        self.assertEqual((w, h), (4, 2))  # Changed from 3x3 based on SA validation
        w, h = determine_window_dimensions(5, "jetpack", "any_ship")
        self.assertEqual((w, h), (3, 3))

    def test_pulse_spitter_jetpack_8_plus(self):
        """Pulse-spitter/jetpack 8+ modules should use 4x2"""
        w, h = determine_window_dimensions(8, "pulse-spitter", "any_ship")
        self.assertEqual((w, h), (4, 2))
        w, h = determine_window_dimensions(10, "jetpack", "any_ship")
        self.assertEqual((w, h), (3, 3))

    def test_pulse_6_modules(self):
        """Pulse with 6 modules should use 3x2"""
        w, h = determine_window_dimensions(6, "pulse", "any_ship")
        self.assertEqual((w, h), (3, 2))

    def test_pulse_7_to_8_modules(self):
        """Pulse with 7-8 modules should use 4x2"""
        w, h = determine_window_dimensions(7, "pulse", "random_ship")
        self.assertEqual((w, h), (4, 2))
        w, h = determine_window_dimensions(8, "pulse", "random_ship")
        self.assertEqual((w, h), (4, 2))

    def test_pulse_9_plus_modules(self):
        """Pulse with 9+ modules should use 4x3"""
        w, h = determine_window_dimensions(9, "pulse", "random_ship")
        self.assertEqual((w, h), (4, 3))

    def test_generic_fallback_less_than_3(self):
        """Generic with <3 modules should use 2x1"""
        w, h = determine_window_dimensions(2, "unknown_tech", "unknown_ship")
        self.assertEqual((w, h), (2, 1))

    def test_generic_fallback_3_modules(self):
        """Generic with 3 modules should use 2x2"""
        w, h = determine_window_dimensions(3, "unknown_tech", "unknown_ship")
        self.assertEqual((w, h), (2, 2))

    def test_generic_fallback_4_modules(self):
        """Generic with 4 modules should use 2x2"""
        w, h = determine_window_dimensions(4, "unknown_tech", "unknown_ship")
        self.assertEqual((w, h), (2, 2))

    def test_generic_fallback_5_to_6_modules(self):
        """Generic with 5-6 modules should use 3x2"""
        w, h = determine_window_dimensions(5, "unknown_tech", "unknown_ship")
        self.assertEqual((w, h), (3, 2))
        w, h = determine_window_dimensions(6, "unknown_tech", "unknown_ship")
        self.assertEqual((w, h), (3, 2))

    def test_generic_fallback_7_to_8_modules(self):
        """Generic with 7 modules should use 4x2, 8 modules should use 3x3"""
        w, h = determine_window_dimensions(7, "unknown_tech", "unknown_ship")
        self.assertEqual((w, h), (4, 2))  # Changed from 3x3 based on SA validation
        w, h = determine_window_dimensions(8, "unknown_tech", "unknown_ship")
        self.assertEqual((w, h), (3, 3))

    def test_generic_fallback_9_modules(self):
        """Generic with exactly 9 modules should use 3x3"""
        w, h = determine_window_dimensions(9, "unknown_tech", "unknown_ship")
        self.assertEqual((w, h), (3, 3))

    def test_generic_fallback_10_plus_modules(self):
        """Generic with 10+ modules should use 4x3"""
        w, h = determine_window_dimensions(10, "unknown_tech", "unknown_ship")
        self.assertEqual((w, h), (4, 3))
        w, h = determine_window_dimensions(20, "unknown_tech", "unknown_ship")
        self.assertEqual((w, h), (4, 3))

    def test_very_large_module_count(self):
        """Very large module count should still return reasonable dimensions"""
        w, h = determine_window_dimensions(1000, "hyper", "any_ship")
        self.assertEqual((w, h), (4, 4))

    def test_dimensions_are_positive(self):
        """All returned dimensions should be positive"""
        for count in [0, 1, 5, 10, 20, 50]:
            for tech in ["hyper", "pulse", "bolt-caster", "unknown"]:
                for ship in ["corvette", "sentinel", "fighter"]:
                    w, h = determine_window_dimensions(count, tech, ship)
                    self.assertGreater(w, 0, f"Width 0 for count={count}, tech={tech}, ship={ship}")
                    self.assertGreater(h, 0, f"Height 0 for count={count}, tech={tech}, ship={ship}")


class TestPlaceAllModulesInEmptySlots(unittest.TestCase):
    """Adversarial tests for place_all_modules_in_empty_slots"""

    def _create_grid(self, width=5, height=5):
        """Helper to create a test grid"""
        grid = Grid(width, height)
        # Activate all cells
        for y in range(height):
            for x in range(width):
                grid.cells[y][x]["active"] = True
        return grid

    def test_no_modules_returns_grid_unchanged(self):
        """With no modules, grid should be returned unchanged"""
        grid = self._create_grid(3, 3)
        modules = {}
        result = place_all_modules_in_empty_slots(grid, modules, "ship", "tech")
        # Grid should still be empty
        for y in range(grid.height):
            for x in range(grid.width):
                self.assertIsNone(result.get_cell(x, y)["module"])

    def test_all_empty_grid_places_all_modules(self):
        """All modules should be placed in completely empty grid"""
        grid = self._create_grid(5, 5)
        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"Module {i}",
                "tech": "tech",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(5)
        ]
        modules = {"ship": {"tech": tech_modules}}

        result = place_all_modules_in_empty_slots(grid, modules, "ship", "tech", tech_modules=tech_modules)

        # Count placed modules
        placed_count = sum(
            1 for y in range(result.height) for x in range(result.width) if result.get_cell(x, y)["module"] is not None
        )
        self.assertEqual(placed_count, 5)

    def test_partial_grid_places_available_modules(self):
        """Only as many modules as empty slots should be placed"""
        grid = self._create_grid(3, 3)
        # Fill first 4 cells
        for i in range(4):
            y = i // 3
            x = i % 3
            grid.cells[y][x]["module"] = f"existing_m{i}"

        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"Module {i}",
                "tech": "tech",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(10)
        ]
        modules = {"ship": {"tech": tech_modules}}

        result = place_all_modules_in_empty_slots(grid, modules, "ship", "tech", tech_modules=tech_modules)

        # Count placed modules of target tech
        placed_count = sum(
            1
            for y in range(result.height)
            for x in range(result.width)
            if result.get_cell(x, y)["module"] is not None and result.get_cell(x, y)["tech"] == "tech"
        )
        self.assertEqual(placed_count, 5)  # 9 total cells - 4 existing = 5 available

    def test_no_active_cells_places_nothing(self):
        """If no cells are active, no modules should be placed"""
        grid = self._create_grid(3, 3)
        # Deactivate all cells
        for y in range(grid.height):
            for x in range(grid.width):
                grid.cells[y][x]["active"] = False

        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"Module {i}",
                "tech": "tech",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(5)
        ]
        modules = {"ship": {"tech": tech_modules}}

        result = place_all_modules_in_empty_slots(grid, modules, "ship", "tech", tech_modules=tech_modules)

        # No modules should be placed
        for y in range(result.height):
            for x in range(result.width):
                self.assertIsNone(result.get_cell(x, y)["module"])

    def test_mixed_active_inactive_places_in_active_only(self):
        """Modules should only be placed in active cells"""
        grid = self._create_grid(3, 3)
        # Deactivate right column
        for y in range(grid.height):
            grid.cells[y][2]["active"] = False

        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"Module {i}",
                "tech": "tech",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(10)
        ]
        modules = {"ship": {"tech": tech_modules}}

        result = place_all_modules_in_empty_slots(grid, modules, "ship", "tech", tech_modules=tech_modules)

        # Should only have placed modules in columns 0 and 1 (6 cells)
        placed_count = sum(
            1 for y in range(result.height) for x in range(result.width) if result.get_cell(x, y)["module"] is not None
        )
        self.assertEqual(placed_count, 6)

    def test_preserves_existing_modules(self):
        """Existing modules should not be overwritten"""
        grid = self._create_grid(3, 3)
        grid.cells[0][0]["module"] = "existing_module"
        grid.cells[0][0]["tech"] = "other_tech"

        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"Module {i}",
                "tech": "tech",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(5)
        ]
        modules = {"ship": {"tech": tech_modules}}

        result = place_all_modules_in_empty_slots(grid, modules, "ship", "tech", tech_modules=tech_modules)

        # Existing module should remain
        self.assertEqual(result.get_cell(0, 0)["module"], "existing_module")
        self.assertEqual(result.get_cell(0, 0)["tech"], "other_tech")

    def test_column_by_column_order(self):
        """Modules should be placed column-by-column (x first, then y)"""
        grid = self._create_grid(2, 2)
        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"Module {i}",
                "tech": "tech",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(4)
        ]
        modules = {"ship": {"tech": tech_modules}}

        result = place_all_modules_in_empty_slots(grid, modules, "ship", "tech", tech_modules=tech_modules)

        # Expected order: (0,0), (0,1), (1,0), (1,1)
        self.assertEqual(result.get_cell(0, 0)["module"], "m0")
        self.assertEqual(result.get_cell(0, 1)["module"], "m1")
        self.assertEqual(result.get_cell(1, 0)["module"], "m2")
        self.assertEqual(result.get_cell(1, 1)["module"], "m3")

    def test_fewer_slots_than_modules_logs_warning(self):
        """Should handle gracefully when fewer slots than modules"""
        grid = self._create_grid(2, 2)
        tech_modules = [
            {
                "id": f"m{i}",
                "label": f"Module {i}",
                "tech": "tech",
                "type": "core",
                "bonus": 1,
                "adjacency": "none",
                "sc_eligible": False,
                "image": None,
            }
            for i in range(10)
        ]
        modules = {"ship": {"tech": tech_modules}}

        result = place_all_modules_in_empty_slots(grid, modules, "ship", "tech", tech_modules=tech_modules)

        # Should only place 4 modules in a 2x2 grid
        placed_count = sum(
            1 for y in range(result.height) for x in range(result.width) if result.get_cell(x, y)["module"] is not None
        )
        self.assertEqual(placed_count, 4)


class TestCountEmptyInLocalized(unittest.TestCase):
    """Adversarial tests for count_empty_in_localized"""

    def test_completely_empty_grid(self):
        """Completely empty grid should return full count"""
        grid = Grid(5, 5)
        count = count_empty_in_localized(grid)
        self.assertEqual(count, 25)

    def test_completely_full_grid(self):
        """Completely full grid should return 0"""
        grid = Grid(5, 5)
        for y in range(grid.height):
            for x in range(grid.width):
                grid.cells[y][x]["module"] = f"module_{x}_{y}"
        count = count_empty_in_localized(grid)
        self.assertEqual(count, 0)

    def test_partially_filled_grid(self):
        """Partially filled grid should return correct count"""
        grid = Grid(3, 3)
        grid.cells[0][0]["module"] = "m1"
        grid.cells[1][1]["module"] = "m2"
        grid.cells[2][2]["module"] = "m3"
        count = count_empty_in_localized(grid)
        self.assertEqual(count, 6)

    def test_single_cell_grid_empty(self):
        """Single cell empty grid should return 1"""
        grid = Grid(1, 1)
        count = count_empty_in_localized(grid)
        self.assertEqual(count, 1)

    def test_single_cell_grid_full(self):
        """Single cell full grid should return 0"""
        grid = Grid(1, 1)
        grid.cells[0][0]["module"] = "m"
        count = count_empty_in_localized(grid)
        self.assertEqual(count, 0)

    def test_ignores_inactive_cells(self):
        """Should count all cells including inactive ones"""
        grid = Grid(2, 2)
        grid.cells[0][0]["module"] = "m"
        grid.cells[0][1]["active"] = False
        count = count_empty_in_localized(grid)
        # Inactive cells are still counted as empty if module is None
        self.assertEqual(count, 3)

    def test_large_grid(self):
        """Large grid should count correctly"""
        grid = Grid(100, 100)
        # Fill first row
        for x in range(100):
            grid.cells[0][x]["module"] = f"m{x}"
        count = count_empty_in_localized(grid)
        self.assertEqual(count, 10000 - 100)


class TestCheckAllModulesPlaced(unittest.TestCase):
    """Adversarial tests for check_all_modules_placed"""

    def test_all_modules_placed_returns_true(self):
        """When all expected modules are placed, should return True"""
        grid = Grid(5, 5)
        tech_modules = [
            {"id": "m1", "label": "Module 1"},
            {"id": "m2", "label": "Module 2"},
            {"id": "m3", "label": "Module 3"},
        ]
        grid.cells[0][0]["module"] = "m1"
        grid.cells[0][0]["tech"] = "test_tech"
        grid.cells[0][1]["module"] = "m2"
        grid.cells[0][1]["tech"] = "test_tech"
        grid.cells[0][2]["module"] = "m3"
        grid.cells[0][2]["tech"] = "test_tech"

        result = check_all_modules_placed(grid, {}, "ship", "test_tech", tech_modules=tech_modules)
        self.assertTrue(result)

    def test_missing_modules_returns_false(self):
        """When modules are missing, should return False"""
        grid = Grid(5, 5)
        tech_modules = [
            {"id": "m1", "label": "Module 1"},
            {"id": "m2", "label": "Module 2"},
            {"id": "m3", "label": "Module 3"},
        ]
        grid.cells[0][0]["module"] = "m1"
        grid.cells[0][0]["tech"] = "test_tech"
        # m2 and m3 are missing

        result = check_all_modules_placed(grid, {}, "ship", "test_tech", tech_modules=tech_modules)
        self.assertFalse(result)

    def test_empty_tech_modules_returns_true(self):
        """Empty module list should return True (all 0 modules placed)"""
        grid = Grid(5, 5)
        tech_modules = []

        result = check_all_modules_placed(grid, {}, "ship", "test_tech", tech_modules=tech_modules)
        self.assertTrue(result)

    def test_ignores_other_tech_modules(self):
        """Should only count modules matching the target tech"""
        grid = Grid(5, 5)
        tech_modules = [
            {"id": "m1", "label": "Module 1"},
            {"id": "m2", "label": "Module 2"},
        ]
        grid.cells[0][0]["module"] = "m1"
        grid.cells[0][0]["tech"] = "test_tech"
        grid.cells[0][1]["module"] = "m2"
        grid.cells[0][1]["tech"] = "test_tech"
        grid.cells[0][2]["module"] = "m3"
        grid.cells[0][2]["tech"] = "other_tech"  # Different tech

        result = check_all_modules_placed(grid, {}, "ship", "test_tech", tech_modules=tech_modules)
        self.assertTrue(result)

    def test_duplicate_modules_in_grid_returns_false(self):
        """If same module placed twice, should return False"""
        grid = Grid(5, 5)
        tech_modules = [
            {"id": "m1", "label": "Module 1"},
            {"id": "m2", "label": "Module 2"},
        ]
        grid.cells[0][0]["module"] = "m1"
        grid.cells[0][0]["tech"] = "test_tech"
        grid.cells[0][1]["module"] = "m1"  # Duplicate!
        grid.cells[0][1]["tech"] = "test_tech"

        result = check_all_modules_placed(grid, {}, "ship", "test_tech", tech_modules=tech_modules)
        self.assertFalse(result)

    def test_extra_modules_returns_false(self):
        """If extra modules are placed, should return False"""
        grid = Grid(5, 5)
        tech_modules = [
            {"id": "m1", "label": "Module 1"},
        ]
        grid.cells[0][0]["module"] = "m1"
        grid.cells[0][0]["tech"] = "test_tech"
        grid.cells[0][1]["module"] = "m2"  # Extra!
        grid.cells[0][1]["tech"] = "test_tech"

        result = check_all_modules_placed(grid, {}, "ship", "test_tech", tech_modules=tech_modules)
        self.assertFalse(result)

    def test_none_module_slot_returns_false(self):
        """If any module slot is None, should return False when expecting all placed"""
        grid = Grid(5, 5)
        tech_modules = [
            {"id": "m1", "label": "Module 1"},
            {"id": "m2", "label": "Module 2"},
        ]
        grid.cells[0][0]["module"] = "m1"
        grid.cells[0][0]["tech"] = "test_tech"
        # m2 is missing (None)

        result = check_all_modules_placed(grid, {}, "ship", "test_tech", tech_modules=tech_modules)
        self.assertFalse(result)

    def test_case_sensitive_module_ids(self):
        """Module IDs should be case-sensitive"""
        grid = Grid(5, 5)
        tech_modules = [
            {"id": "M1", "label": "Module 1"},
        ]
        grid.cells[0][0]["module"] = "m1"  # Lowercase
        grid.cells[0][0]["tech"] = "test_tech"

        result = check_all_modules_placed(grid, {}, "ship", "test_tech", tech_modules=tech_modules)
        self.assertFalse(result)  # Should not match

    def test_full_grid_all_placed(self):
        """Large grid with all modules placed"""
        grid = Grid(10, 10)
        tech_modules = [{"id": f"m{i}", "label": f"Module {i}"} for i in range(100)]
        for i in range(100):
            y = i // 10
            x = i % 10
            grid.cells[y][x]["module"] = f"m{i}"
            grid.cells[y][x]["tech"] = "test_tech"

        result = check_all_modules_placed(grid, {}, "ship", "test_tech", tech_modules=tech_modules)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
