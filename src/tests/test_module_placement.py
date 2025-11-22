"""
Comprehensive test suite for module_placement.py

This test suite focuses on finding bugs in grid cell state management,
module placement, and tech clearing logic.
"""

import unittest
from src.grid_utils import Grid
from src.module_placement import place_module, clear_all_modules_of_tech


class TestPlaceModule(unittest.TestCase):
    """Test module placement on grid cells."""

    def setUp(self):
        """Set up a clean grid for each test."""
        self.grid = Grid(5, 5)

    def test_place_module_sets_all_properties(self):
        """Placing a module should set all required properties."""
        place_module(
            self.grid, 0, 0,
            module_id="pulse_a",
            label="Pulse Engine A",
            tech="pulse",
            module_type="bonus",
            bonus=10.5,
            adjacency="greater",
            sc_eligible=True,
            image="pulse_a.png"
        )
        
        cell = self.grid.get_cell(0, 0)
        self.assertEqual(cell["module"], "pulse_a")
        self.assertEqual(cell["label"], "Pulse Engine A")
        self.assertEqual(cell["tech"], "pulse")
        self.assertEqual(cell["type"], "bonus")
        self.assertEqual(cell["bonus"], 10.5)
        self.assertEqual(cell["adjacency"], "greater")
        self.assertEqual(cell["sc_eligible"], True)
        self.assertEqual(cell["image"], "pulse_a.png")

    def test_place_module_sets_module_position(self):
        """Module position should be set correctly."""
        place_module(
            self.grid, 2, 3,
            module_id="test_mod",
            label="Test",
            tech="test",
            module_type="bonus",
            bonus=5.0,
            adjacency="lesser",
            sc_eligible=False,
            image=None
        )
        
        cell = self.grid.get_cell(2, 3)
        self.assertEqual(cell["module_position"], (2, 3))

    def test_place_module_overwrites_previous(self):
        """Placing a module should overwrite the previous one."""
        # Place first module
        place_module(self.grid, 0, 0, "mod1", "Mod 1", "pulse", "bonus",
                    10.0, "greater", True, None)
        
        # Place second module at same location
        place_module(self.grid, 0, 0, "mod2", "Mod 2", "engineering", "bonus",
                    15.0, "lesser", False, None)
        
        cell = self.grid.get_cell(0, 0)
        self.assertEqual(cell["module"], "mod2")
        self.assertEqual(cell["tech"], "engineering")
        self.assertEqual(cell["bonus"], 15.0)

    def test_place_module_at_different_coordinates(self):
        """Modules at different coordinates should be independent."""
        place_module(self.grid, 0, 0, "mod1", "Mod 1", "pulse", "bonus",
                    10.0, "greater", True, None)
        place_module(self.grid, 1, 1, "mod2", "Mod 2", "engineering", "bonus",
                    15.0, "lesser", False, None)
        
        cell1 = self.grid.get_cell(0, 0)
        cell2 = self.grid.get_cell(1, 1)
        
        self.assertEqual(cell1["module"], "mod1")
        self.assertEqual(cell2["module"], "mod2")
        self.assertEqual(cell1["tech"], "pulse")
        self.assertEqual(cell2["tech"], "engineering")

    def test_place_module_at_corner(self):
        """Should be able to place at grid corners."""
        corners = [(0, 0), (4, 0), (0, 4), (4, 4)]
        
        for i, (x, y) in enumerate(corners):
            place_module(self.grid, x, y, f"mod_{i}", f"Mod {i}", "test",
                        "bonus", 5.0 + i, "greater", True, None)
            
            cell = self.grid.get_cell(x, y)
            self.assertEqual(cell["module"], f"mod_{i}")
            self.assertEqual(cell["module_position"], (x, y))

    def test_place_module_with_various_bonus_values(self):
        """Should handle various bonus values correctly."""
        test_values = [0.0, 1.5, 10.0, 100.5, 999.99]
        
        for i, bonus in enumerate(test_values):
            place_module(self.grid, i, 0, f"mod_{i}", "Mod", "test",
                        "bonus", bonus, "greater", True, None)
            
            cell = self.grid.get_cell(i, 0)
            self.assertEqual(cell["bonus"], bonus)

    def test_place_module_with_none_image(self):
        """Should handle None image gracefully."""
        place_module(self.grid, 0, 0, "mod", "Mod", "test", "bonus",
                    5.0, "greater", False, None)
        
        cell = self.grid.get_cell(0, 0)
        self.assertIsNone(cell["image"])

    def test_place_module_with_string_image(self):
        """Should handle string image paths."""
        place_module(self.grid, 0, 0, "mod", "Mod", "test", "bonus",
                    5.0, "greater", False, "path/to/image.png")
        
        cell = self.grid.get_cell(0, 0)
        self.assertEqual(cell["image"], "path/to/image.png")


class TestClearAllModulesOfTech(unittest.TestCase):
    """Test clearing all modules of a specific technology."""

    def setUp(self):
        """Set up a grid with mixed tech modules."""
        self.grid = Grid(5, 5)
        
        # Place pulse modules
        place_module(self.grid, 0, 0, "pulse_a", "Pulse A", "pulse", "bonus",
                    10.0, "greater", True, None)
        place_module(self.grid, 1, 0, "pulse_b", "Pulse B", "pulse", "bonus",
                    8.0, "lesser", False, None)
        
        # Place engineering modules
        place_module(self.grid, 0, 1, "eng_a", "Eng A", "engineering", "bonus",
                    12.0, "greater", True, None)
        place_module(self.grid, 1, 1, "eng_b", "Eng B", "engineering", "bonus",
                    11.0, "lesser", False, None)
        
        # Place weapons modules
        place_module(self.grid, 0, 2, "weap_a", "Weap A", "weapons", "bonus",
                    15.0, "greater", False, None)

    def test_clear_pulse_modules(self):
        """Clearing pulse should remove only pulse modules."""
        clear_all_modules_of_tech(self.grid, "pulse")
        
        # Pulse modules should be cleared
        self.assertIsNone(self.grid.get_cell(0, 0)["module"])
        self.assertIsNone(self.grid.get_cell(1, 0)["module"])
        
        # Other tech modules should remain
        self.assertEqual(self.grid.get_cell(0, 1)["module"], "eng_a")
        self.assertEqual(self.grid.get_cell(1, 1)["module"], "eng_b")
        self.assertEqual(self.grid.get_cell(0, 2)["module"], "weap_a")

    def test_clear_engineering_modules(self):
        """Clearing engineering should remove only engineering modules."""
        clear_all_modules_of_tech(self.grid, "engineering")
        
        # Engineering modules should be cleared
        self.assertIsNone(self.grid.get_cell(0, 1)["module"])
        self.assertIsNone(self.grid.get_cell(1, 1)["module"])
        
        # Other tech modules should remain
        self.assertEqual(self.grid.get_cell(0, 0)["module"], "pulse_a")
        self.assertEqual(self.grid.get_cell(1, 0)["module"], "pulse_b")
        self.assertEqual(self.grid.get_cell(0, 2)["module"], "weap_a")

    def test_clear_clears_all_properties(self):
        """Clearing should reset all module-related properties."""
        clear_all_modules_of_tech(self.grid, "pulse")
        
        cell = self.grid.get_cell(0, 0)
        self.assertIsNone(cell["module"])
        self.assertEqual(cell["label"], "")
        self.assertIsNone(cell["tech"])
        self.assertEqual(cell["type"], "")
        self.assertEqual(cell["bonus"], 0)
        self.assertEqual(cell["total"], 0)
        self.assertEqual(cell["adjacency_bonus"], 0)
        self.assertEqual(cell["sc_eligible"], False)
        self.assertIsNone(cell["image"])
        self.assertIsNone(cell["module_position"])

    def test_clear_preserves_grid_structure(self):
        """Clearing should not affect grid structure."""
        original_width = self.grid.width
        original_height = self.grid.height
        
        clear_all_modules_of_tech(self.grid, "pulse")
        
        self.assertEqual(self.grid.width, original_width)
        self.assertEqual(self.grid.height, original_height)

    def test_clear_nonexistent_tech(self):
        """Clearing nonexistent tech should not crash."""
        # Should not raise an exception
        clear_all_modules_of_tech(self.grid, "nonexistent_tech")
        
        # All modules should still be there
        self.assertEqual(self.grid.get_cell(0, 0)["module"], "pulse_a")

    def test_clear_empty_grid(self):
        """Clearing from empty grid should not crash."""
        empty_grid = Grid(5, 5)
        
        # Should not raise an exception
        clear_all_modules_of_tech(empty_grid, "pulse")
        
        # Grid should still be empty
        for y in range(empty_grid.height):
            for x in range(empty_grid.width):
                self.assertIsNone(empty_grid.get_cell(x, y)["module"])

    def test_clear_multiple_techs_sequentially(self):
        """Clearing multiple techs should work correctly."""
        clear_all_modules_of_tech(self.grid, "pulse")
        clear_all_modules_of_tech(self.grid, "engineering")
        
        # Both should be cleared
        self.assertIsNone(self.grid.get_cell(0, 0)["module"])
        self.assertIsNone(self.grid.get_cell(1, 0)["module"])
        self.assertIsNone(self.grid.get_cell(0, 1)["module"])
        self.assertIsNone(self.grid.get_cell(1, 1)["module"])
        
        # Weapons should remain
        self.assertEqual(self.grid.get_cell(0, 2)["module"], "weap_a")

    def test_clear_twice_is_idempotent(self):
        """Clearing twice should result in same state."""
        clear_all_modules_of_tech(self.grid, "pulse")
        state_after_first = {
            "0_0_module": self.grid.get_cell(0, 0)["module"],
            "1_0_module": self.grid.get_cell(1, 0)["module"],
        }
        
        # Clear again
        clear_all_modules_of_tech(self.grid, "pulse")
        state_after_second = {
            "0_0_module": self.grid.get_cell(0, 0)["module"],
            "1_0_module": self.grid.get_cell(1, 0)["module"],
        }
        
        self.assertEqual(state_after_first, state_after_second)

    def test_clear_preserves_active_and_supercharge_state(self):
        """Clearing should preserve active/supercharge cell properties."""
        # Set custom active/supercharge state
        self.grid.get_cell(0, 0)["active"] = False
        self.grid.get_cell(1, 0)["supercharged"] = True
        
        original_active = self.grid.get_cell(0, 0)["active"]
        original_supercharge = self.grid.get_cell(1, 0)["supercharged"]
        
        clear_all_modules_of_tech(self.grid, "pulse")
        
        # These should be preserved
        self.assertEqual(self.grid.get_cell(0, 0)["active"], original_active)
        self.assertEqual(self.grid.get_cell(1, 0)["supercharged"], original_supercharge)


class TestClearAndReplaceWorkflow(unittest.TestCase):
    """Test clearing and replacing module workflow."""

    def test_clear_then_place_new_module(self):
        """Should be able to clear and place new module in same location."""
        grid = Grid(3, 3)
        
        # Place initial module
        place_module(grid, 0, 0, "old_mod", "Old", "pulse", "bonus",
                    10.0, "greater", True, None)
        
        # Clear and place new
        clear_all_modules_of_tech(grid, "pulse")
        place_module(grid, 0, 0, "new_mod", "New", "pulse", "bonus",
                    20.0, "lesser", False, None)
        
        cell = grid.get_cell(0, 0)
        self.assertEqual(cell["module"], "new_mod")
        self.assertEqual(cell["bonus"], 20.0)

    def test_clear_tech_then_place_different_tech(self):
        """Should be able to place different tech in same location after clear."""
        grid = Grid(3, 3)
        
        # Place pulse
        place_module(grid, 0, 0, "pulse_mod", "Pulse", "pulse", "bonus",
                    10.0, "greater", True, None)
        
        # Clear pulse and place engineering
        clear_all_modules_of_tech(grid, "pulse")
        place_module(grid, 0, 0, "eng_mod", "Eng", "engineering", "bonus",
                    12.0, "lesser", False, None)
        
        cell = grid.get_cell(0, 0)
        self.assertEqual(cell["module"], "eng_mod")
        self.assertEqual(cell["tech"], "engineering")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_place_on_entire_grid(self):
        """Should be able to place modules on all grid cells."""
        grid = Grid(3, 3)
        
        for y in range(grid.height):
            for x in range(grid.width):
                place_module(grid, x, y, f"mod_{x}_{y}", f"Mod {x},{y}",
                           "test", "bonus", 5.0 + x + y, "greater", True, None)
        
        # Verify all cells have modules
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                self.assertIsNotNone(cell["module"])
                self.assertEqual(cell["module"], f"mod_{x}_{y}")

    def test_clear_entire_grid(self):
        """Should be able to clear all modules from grid."""
        grid = Grid(3, 3)
        
        # Fill entire grid
        for y in range(grid.height):
            for x in range(grid.width):
                place_module(grid, x, y, f"mod_{x}_{y}", f"Mod {x},{y}",
                           "pulse", "bonus", 5.0, "greater", True, None)
        
        # Clear all
        clear_all_modules_of_tech(grid, "pulse")
        
        # Verify all cells are cleared
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                self.assertIsNone(cell["module"])

    def test_place_with_empty_label(self):
        """Should handle empty label."""
        grid = Grid(3, 3)
        place_module(grid, 0, 0, "mod", "", "test", "bonus",
                    5.0, "greater", False, None)
        
        cell = grid.get_cell(0, 0)
        self.assertEqual(cell["label"], "")

    def test_place_with_special_characters_in_id(self):
        """Should handle special characters in module ID."""
        grid = Grid(3, 3)
        special_id = "mod_@#$%^&*()"
        place_module(grid, 0, 0, special_id, "Special", "test", "bonus",
                    5.0, "greater", False, None)
        
        cell = grid.get_cell(0, 0)
        self.assertEqual(cell["module"], special_id)

    def test_clear_case_sensitive(self):
        """Tech clearing should be case-sensitive."""
        grid = Grid(3, 3)
        place_module(grid, 0, 0, "mod", "Mod", "Pulse", "bonus",
                    5.0, "greater", True, None)
        
        # Clear with different case
        clear_all_modules_of_tech(grid, "pulse")
        
        # Should not be cleared (case mismatch)
        self.assertEqual(grid.get_cell(0, 0)["module"], "mod")


if __name__ == "__main__":
    unittest.main()
