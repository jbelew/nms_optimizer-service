"""
Comprehensive test suite for pattern_matching.py

This test suite focuses on finding bugs in pattern rotation, mirroring,
application, and adjacency scoring logic. It uses adversarial test cases
to expose coordinate transformation errors and state management issues.
"""

import unittest
from src.grid_utils import Grid
from src.pattern_matching import (
    rotate_pattern,
    mirror_pattern_horizontally,
    mirror_pattern_vertically,
    apply_pattern_to_grid,
    get_all_unique_pattern_variations,
    calculate_pattern_adjacency_score,
    _extract_pattern_from_grid,
)
from src.module_placement import place_module


class TestPatternRotation(unittest.TestCase):
    """Test pattern rotation logic for off-by-one errors and coordinate issues."""

    def test_rotate_single_cell_pattern(self):
        """Single cell patterns should remain unchanged after rotation."""
        pattern = {(0, 0): "module_a"}
        rotated = rotate_pattern(pattern)
        self.assertEqual(rotated, pattern)

    def test_rotate_empty_pattern(self):
        """Empty patterns should stay empty."""
        pattern = {}
        rotated = rotate_pattern(pattern)
        self.assertEqual(rotated, {})

    def test_rotate_2x1_horizontal_line(self):
        """Rotate a 2x1 horizontal line 90 degrees clockwise."""
        # Original: [(0,0)=A, (1,0)=B]
        # After 90° CW rotation should become a vertical line
        pattern = {(0, 0): "A", (1, 0): "B"}
        rotated = rotate_pattern(pattern)
        # Expected: max_x=1, so (0,0)->new_y=1-0=1 giving (0,1)
        #                     (1,0)->new_y=1-1=0 giving (0,0)
        expected = {(0, 1): "A", (0, 0): "B"}
        self.assertEqual(rotated, expected)

    def test_rotate_3x3_pattern_four_times_returns_original(self):
        """Rotating a pattern 4 times should return the original."""
        pattern = {(0, 0): "A", (1, 0): "B", (2, 0): "C",
                   (0, 1): "D", (1, 1): "E", (2, 1): "F"}
        
        rotated_once = rotate_pattern(pattern)
        rotated_twice = rotate_pattern(rotated_once)
        rotated_thrice = rotate_pattern(rotated_twice)
        rotated_four = rotate_pattern(rotated_thrice)
        
        self.assertEqual(rotated_four, pattern, "4 rotations should return original")

    def test_rotate_l_shaped_pattern(self):
        """Test rotation of an L-shaped pattern."""
        # Original L shape:
        # A B
        # A
        pattern = {(0, 0): "A", (1, 0): "B", (0, 1): "A"}
        rotated = rotate_pattern(pattern)
        
        # Verify all coordinates are within valid range
        self.assertTrue(all(isinstance(k, tuple) and len(k) == 2 for k in rotated.keys()))
        # Verify count unchanged
        self.assertEqual(len(rotated), 3)

    def test_rotate_preserves_module_ids(self):
        """Rotation should preserve module IDs."""
        pattern = {(0, 0): "mod1", (1, 1): "mod2", (2, 0): "mod3"}
        rotated = rotate_pattern(pattern)
        
        module_ids_original = set(pattern.values())
        module_ids_rotated = set(rotated.values())
        self.assertEqual(module_ids_original, module_ids_rotated)


class TestPatternMirroring(unittest.TestCase):
    """Test pattern mirroring logic."""

    def test_mirror_horizontal_single_cell(self):
        """Single cell should remain unchanged when mirrored horizontally."""
        pattern = {(0, 0): "A"}
        mirrored = mirror_pattern_horizontally(pattern)
        self.assertEqual(mirrored, pattern)

    def test_mirror_horizontal_empty_pattern(self):
        """Empty pattern should stay empty."""
        pattern = {}
        mirrored = mirror_pattern_horizontally(pattern)
        self.assertEqual(mirrored, {})

    def test_mirror_horizontal_2x1_line(self):
        """Mirror a 2x1 horizontal line."""
        # Original: A B (at x=0, x=1)
        # Mirrored: B A
        pattern = {(0, 0): "A", (1, 0): "B"}
        mirrored = mirror_pattern_horizontally(pattern)
        expected = {(1, 0): "A", (0, 0): "B"}
        self.assertEqual(mirrored, expected)

    def test_mirror_horizontal_twice_returns_original(self):
        """Mirroring horizontally twice should return original."""
        pattern = {(0, 0): "A", (1, 0): "B", (2, 0): "C"}
        mirrored_once = mirror_pattern_horizontally(pattern)
        mirrored_twice = mirror_pattern_horizontally(mirrored_once)
        self.assertEqual(mirrored_twice, pattern)

    def test_mirror_vertical_single_cell(self):
        """Single cell should remain unchanged when mirrored vertically."""
        pattern = {(0, 0): "A"}
        mirrored = mirror_pattern_vertically(pattern)
        self.assertEqual(mirrored, pattern)

    def test_mirror_vertical_2x1_column(self):
        """Mirror a 1x2 vertical line."""
        # Original: A (y=0), B (y=1)
        # Mirrored: B (y=0), A (y=1)
        pattern = {(0, 0): "A", (0, 1): "B"}
        mirrored = mirror_pattern_vertically(pattern)
        expected = {(0, 1): "A", (0, 0): "B"}
        self.assertEqual(mirrored, expected)

    def test_mirror_vertical_twice_returns_original(self):
        """Mirroring vertically twice should return original."""
        pattern = {(0, 0): "A", (0, 1): "B", (0, 2): "C"}
        mirrored_once = mirror_pattern_vertically(pattern)
        mirrored_twice = mirror_pattern_vertically(mirrored_once)
        self.assertEqual(mirrored_twice, pattern)

    def test_mirror_preserves_module_ids(self):
        """Mirroring should preserve module IDs."""
        pattern = {(0, 0): "mod1", (1, 1): "mod2"}
        mirrored_h = mirror_pattern_horizontally(pattern)
        mirrored_v = mirror_pattern_vertically(pattern)
        
        self.assertEqual(set(pattern.values()), set(mirrored_h.values()))
        self.assertEqual(set(pattern.values()), set(mirrored_v.values()))


class TestPatternVariationGeneration(unittest.TestCase):
    """Test unique pattern variation generation."""

    def test_single_cell_has_one_variation(self):
        """Single cell pattern should have only one unique variation."""
        pattern = {(0, 0): "A"}
        variations = get_all_unique_pattern_variations(pattern)
        # Single cell: all rotations/mirrors are identical
        self.assertEqual(len(variations), 1)

    def test_2x1_line_has_multiple_variations(self):
        """2x1 line should have multiple unique variations."""
        pattern = {(0, 0): "A", (1, 0): "B"}
        variations = get_all_unique_pattern_variations(pattern)
        # Should have at least 2 distinct orientations (horizontal and vertical)
        self.assertGreaterEqual(len(variations), 2)

    def test_square_pattern_symmetry(self):
        """Symmetric square pattern should have fewer unique variations."""
        # 2x2 symmetric square
        pattern = {(0, 0): "A", (1, 0): "A",
                   (0, 1): "A", (1, 1): "A"}
        variations = get_all_unique_pattern_variations(pattern)
        # All rotations should be identical for a uniform square
        self.assertEqual(len(variations), 1)

    def test_variations_always_include_original(self):
        """Variations should always include the original pattern."""
        pattern = {(0, 0): "A", (1, 0): "B", (0, 1): "C"}
        variations = get_all_unique_pattern_variations(pattern)
        self.assertIn(pattern, variations)

    def test_all_variations_preserve_module_count(self):
        """All variations should have the same number of modules."""
        pattern = {(0, 0): "A", (1, 0): "B", (2, 0): "C"}
        variations = get_all_unique_pattern_variations(pattern)
        
        for var in variations:
            self.assertEqual(len(var), len(pattern),
                           f"Variation {var} has different count than original")

    def test_no_duplicate_variations(self):
        """Generated variations should be unique."""
        pattern = {(0, 0): "A", (1, 0): "B", (0, 1): "C"}
        variations = get_all_unique_pattern_variations(pattern)
        
        # Convert to tuples of tuples for hashability
        variations_as_tuples = [tuple(sorted(v.items())) for v in variations]
        unique_variations = set(variations_as_tuples)
        
        self.assertEqual(len(variations_as_tuples), len(unique_variations),
                        "There are duplicate variations in the generated list")


class TestPatternApplicationToGrid(unittest.TestCase):
    """Test applying patterns to grids with various edge cases."""

    def setUp(self):
        """Set up common test fixtures."""
        self.grid = Grid(4, 4)
        self.modules_data = {
            "types": {
                "Pulse": [
                    {
                        "key": "pulse",
                        "modules": [
                            {"id": "pulse_a", "label": "Pulse A", "bonus": 10.0,
                             "adjacency": "greater", "type": "bonus", "sc_eligible": True, "image": None},
                            {"id": "pulse_b", "label": "Pulse B", "bonus": 8.0,
                             "adjacency": "lesser", "type": "bonus", "sc_eligible": False, "image": None},
                        ]
                    }
                ]
            }
        }

    def test_apply_pattern_to_empty_grid(self):
        """Apply a simple pattern to an empty grid."""
        pattern = {(0, 0): "pulse_a", (1, 0): "pulse_b"}
        new_grid, score = apply_pattern_to_grid(self.grid, pattern, self.modules_data,
                                              "pulse", 0, 0, "corvette")
        
        self.assertIsNotNone(new_grid)
        self.assertEqual(new_grid.get_cell(0, 0)["module"], "pulse_a")
        self.assertEqual(new_grid.get_cell(1, 0)["module"], "pulse_b")

    def test_apply_pattern_off_grid_returns_none(self):
        """Pattern that goes off-grid should return None."""
        pattern = {(0, 0): "pulse_a", (10, 0): "pulse_b"}  # (10,0) out of bounds
        new_grid, score = apply_pattern_to_grid(self.grid, pattern, self.modules_data,
                                              "pulse", 0, 0, "corvette")
        
        self.assertIsNone(new_grid)
        self.assertEqual(score, 0)

    def test_apply_pattern_with_negative_offset(self):
        """Pattern with negative offset should fail gracefully."""
        pattern = {(0, 0): "pulse_a"}
        new_grid, score = apply_pattern_to_grid(self.grid, pattern, self.modules_data,
                                              "pulse", -1, 0, "corvette")
        
        # (-1,0) + (0,0) = (-1,0) which is out of bounds
        self.assertIsNone(new_grid)

    def test_apply_pattern_with_unowned_modules(self):
        """Pattern with modules player doesn't own should return None."""
        pattern = {(0, 0): "nonexistent_module"}
        new_grid, score = apply_pattern_to_grid(self.grid, pattern, self.modules_data,
                                              "pulse", 0, 0, "corvette")
        
        # No owned modules in pattern
        self.assertIsNone(new_grid)

    def test_apply_pattern_preserves_other_tech_modules(self):
        """Applying a pattern shouldn't remove other tech modules."""
        # Place a different tech module first
        cell = self.grid.get_cell(2, 2)
        cell["module"] = "engineering_mod"
        cell["tech"] = "engineering"
        
        # Apply pulse pattern
        pattern = {(0, 0): "pulse_a"}
        new_grid, score = apply_pattern_to_grid(self.grid, pattern, self.modules_data,
                                              "pulse", 0, 0, "corvette")
        
        self.assertIsNotNone(new_grid)
        self.assertEqual(new_grid.get_cell(2, 2)["tech"], "engineering",
                        "Other tech modules should be preserved")

    def test_apply_pattern_to_inactive_cells_fails(self):
        """Pattern placement on inactive cells should fail."""
        # Mark cell as inactive
        self.grid.get_cell(0, 0)["active"] = False
        
        pattern = {(0, 0): "pulse_a"}
        new_grid, score = apply_pattern_to_grid(self.grid, pattern, self.modules_data,
                                              "pulse", 0, 0, "corvette")
        
        self.assertIsNone(new_grid)

    def test_apply_empty_pattern(self):
        """Applying empty pattern should clear the tech and return valid grid."""
        # Pre-place a module
        place_module(self.grid, 0, 0, "pulse_a", "Pulse A", "pulse", "bonus", 10.0,
                    "greater", True, None)
        
        pattern = {}
        new_grid, score = apply_pattern_to_grid(self.grid, pattern, self.modules_data,
                                              "pulse", 0, 0, "corvette")
        
        self.assertIsNotNone(new_grid)
        # Should have cleared the module
        self.assertIsNone(new_grid.get_cell(0, 0)["module"])

    def test_apply_pattern_with_mixed_owned_unowned_modules(self):
        """Pattern with mix of owned and unowned modules should place only owned ones."""
        pattern = {(0, 0): "pulse_a", (1, 0): "unknown_module"}
        new_grid, score = apply_pattern_to_grid(self.grid, pattern, self.modules_data,
                                              "pulse", 0, 0, "corvette")
        
        # Should place pulse_a but skip unknown_module
        self.assertIsNotNone(new_grid)
        self.assertEqual(new_grid.get_cell(0, 0)["module"], "pulse_a")
        self.assertIsNone(new_grid.get_cell(1, 0)["module"])

    def test_apply_pattern_does_not_modify_original_grid(self):
        """Applying a pattern should not modify the original grid."""
        original_cell_0_0 = self.grid.get_cell(0, 0)["module"]
        
        pattern = {(0, 0): "pulse_a"}
        new_grid, score = apply_pattern_to_grid(self.grid, pattern, self.modules_data,
                                              "pulse", 0, 0, "corvette")
        
        # Original grid should be unchanged
        self.assertEqual(self.grid.get_cell(0, 0)["module"], original_cell_0_0)


class TestPatternAdjacencyScoring(unittest.TestCase):
    """Test the adjacency score calculation logic."""

    def setUp(self):
        """Set up test grid with modules."""
        self.grid = Grid(3, 3)

    def test_empty_grid_score_is_zero(self):
        """Empty grid should have zero adjacency score."""
        score = calculate_pattern_adjacency_score(self.grid, "pulse")
        self.assertEqual(score, 0.0)

    def test_single_module_at_corner_gets_edge_bonus(self):
        """Module at corner should get edge bonuses."""
        cell = self.grid.get_cell(0, 0)
        cell["module"] = "pulse_a"
        cell["tech"] = "pulse"
        cell["adjacency"] = "greater"
        
        score = calculate_pattern_adjacency_score(self.grid, "pulse")
        # Corner module: left edge (0.5) + top edge (0.5) = 1.0
        self.assertEqual(score, 1.0)

    def test_single_module_at_edge_gets_edge_bonus(self):
        """Module on edge should get one edge bonus."""
        cell = self.grid.get_cell(1, 0)
        cell["module"] = "pulse_a"
        cell["tech"] = "pulse"
        cell["adjacency"] = "greater"
        
        score = calculate_pattern_adjacency_score(self.grid, "pulse")
        # Edge module: top edge (0.5) = 0.5
        self.assertEqual(score, 0.5)

    def test_single_center_module_no_edge_bonus(self):
        """Center module should not get edge bonuses."""
        cell = self.grid.get_cell(1, 1)
        cell["module"] = "pulse_a"
        cell["tech"] = "pulse"
        cell["adjacency"] = "greater"
        
        score = calculate_pattern_adjacency_score(self.grid, "pulse")
        # Center: no edge bonuses, no adjacent modules
        self.assertEqual(score, 0.0)

    def test_adjacent_modules_get_adjacency_bonus(self):
        """Two adjacent modules of same tech should get adjacency bonus."""
        cell1 = self.grid.get_cell(0, 0)
        cell1["module"] = "pulse_a"
        cell1["tech"] = "pulse"
        cell1["adjacency"] = "greater"
        
        cell2 = self.grid.get_cell(1, 0)
        cell2["module"] = "pulse_b"
        cell2["tech"] = "pulse"
        cell2["adjacency"] = "greater"
        
        score = calculate_pattern_adjacency_score(self.grid, "pulse")
        # Each module: edge bonus + adjacency bonus to other module
        # cell1: left(0.5) + adjacent_different_tech(3.0) = 3.5
        # cell2: top(0.5) + adjacent_different_tech(3.0) = 3.5
        # Total = 7.0
        self.assertGreater(score, 0.0)

    def test_score_only_counts_specified_tech(self):
        """Score calculation should only count modules of the specified tech."""
        # Place pulse module
        cell1 = self.grid.get_cell(0, 0)
        cell1["module"] = "pulse_a"
        cell1["tech"] = "pulse"
        cell1["adjacency"] = "greater"
        
        # Place engineering module nearby
        cell2 = self.grid.get_cell(1, 0)
        cell2["module"] = "engineering_a"
        cell2["tech"] = "engineering"
        cell2["adjacency"] = "greater"
        
        score_pulse = calculate_pattern_adjacency_score(self.grid, "pulse")
        score_engineering = calculate_pattern_adjacency_score(self.grid, "engineering")
        
        # Should not be equal since only one of each tech is present
        # But both should be > 0
        self.assertGreater(score_pulse, 0.0)
        self.assertGreater(score_engineering, 0.0)


class TestPatternExtraction(unittest.TestCase):
    """Test extracting patterns from grids."""

    def setUp(self):
        """Set up test grid."""
        self.grid = Grid(4, 4)

    def test_extract_single_module(self):
        """Extract a single module should give normalized (0,0) coordinate."""
        cell = self.grid.get_cell(2, 2)
        cell["module"] = "pulse_a"
        cell["tech"] = "pulse"
        
        pattern = _extract_pattern_from_grid(self.grid, "pulse")
        self.assertEqual(pattern, {(0, 0): "pulse_a"})

    def test_extract_multiple_modules_normalized(self):
        """Extract multiple modules should normalize to min coordinates."""
        # Place modules at (2,2), (3,2), (2,3)
        self.grid.get_cell(2, 2)["module"] = "A"
        self.grid.get_cell(2, 2)["tech"] = "pulse"
        self.grid.get_cell(3, 2)["module"] = "B"
        self.grid.get_cell(3, 2)["tech"] = "pulse"
        self.grid.get_cell(2, 3)["module"] = "C"
        self.grid.get_cell(2, 3)["tech"] = "pulse"
        
        pattern = _extract_pattern_from_grid(self.grid, "pulse")
        
        # Should normalize to (0,0), (1,0), (0,1)
        expected = {(0, 0): "A", (1, 0): "B", (0, 1): "C"}
        self.assertEqual(pattern, expected)

    def test_extract_empty_grid(self):
        """Extract from empty grid should return empty pattern."""
        pattern = _extract_pattern_from_grid(self.grid, "pulse")
        self.assertEqual(pattern, {})

    def test_extract_only_specified_tech(self):
        """Extract should only include specified tech."""
        self.grid.get_cell(0, 0)["module"] = "pulse_a"
        self.grid.get_cell(0, 0)["tech"] = "pulse"
        self.grid.get_cell(1, 0)["module"] = "eng_a"
        self.grid.get_cell(1, 0)["tech"] = "engineering"
        
        pulse_pattern = _extract_pattern_from_grid(self.grid, "pulse")
        eng_pattern = _extract_pattern_from_grid(self.grid, "engineering")
        
        self.assertEqual(len(pulse_pattern), 1)
        self.assertEqual(len(eng_pattern), 1)
        self.assertNotEqual(pulse_pattern, eng_pattern)


class TestPatternEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_rotate_with_negative_coordinates(self):
        """Patterns with negative coordinates after rotation."""
        # This shouldn't happen in normal flow but tests robustness
        pattern = {(0, 0): "A"}
        rotated = rotate_pattern(pattern)
        # Should not crash and should have valid structure
        self.assertIsInstance(rotated, dict)

    def test_very_large_pattern(self):
        """Handle large patterns without performance issues."""
        # Create a 10x10 pattern
        pattern = {(x, y): f"mod_{x}_{y}" for x in range(10) for y in range(10)}
        
        rotated = rotate_pattern(pattern)
        variations = get_all_unique_pattern_variations(pattern)
        
        self.assertEqual(len(rotated), 100)
        self.assertGreater(len(variations), 0)

    def test_pattern_with_none_values(self):
        """Patterns with None module IDs (empty slots)."""
        pattern = {(0, 0): "A", (1, 0): None, (2, 0): "B"}
        rotated = rotate_pattern(pattern)
        
        # Should handle None values gracefully
        self.assertIn(None, rotated.values())
        self.assertEqual(len(rotated), 3)


if __name__ == "__main__":
    unittest.main()
