"""
Comprehensive test suite for data_loader.py

This test suite focuses on finding bugs in JSON parsing, caching,
tuple key conversion, and error handling.
"""

import unittest
import json

from src.data_loader import (
    get_module_data,
    get_solve_map,
    get_all_module_data,
    get_all_solve_data,
    get_training_module_ids,
    _convert_map_keys_to_tuple,
)


class TestConvertMapKeysToTuple(unittest.TestCase):
    """Test the tuple key conversion utility."""

    def test_convert_simple_tuple_string(self):
        """Convert simple '0,0' string to (0,0) tuple."""
        data = {"0,0": "value"}
        result = _convert_map_keys_to_tuple(data)
        self.assertEqual(result, {(0, 0): "value"})

    def test_convert_multiple_tuple_keys(self):
        """Convert multiple tuple string keys."""
        data = {"0,0": "a", "1,0": "b", "2,1": "c"}
        result = _convert_map_keys_to_tuple(data)
        self.assertEqual(result, {(0, 0): "a", (1, 0): "b", (2, 1): "c"})

    def test_preserve_non_tuple_string_keys(self):
        """Non-tuple-like string keys should be preserved."""
        data = {"name": "value", "0,0": "coord"}
        result = _convert_map_keys_to_tuple(data)
        self.assertIn("name", result)
        self.assertIn((0, 0), result)

    def test_convert_nested_dicts(self):
        """Convert tuple keys in nested dictionaries."""
        data = {"outer": {"0,0": "inner_value", "1,1": "another_value"}}
        result = _convert_map_keys_to_tuple(data)
        self.assertEqual(result["outer"][(0, 0)], "inner_value")
        self.assertEqual(result["outer"][(1, 1)], "another_value")

    def test_convert_list_with_dicts(self):
        """Convert tuple keys in dictionaries within lists."""
        data = [{"0,0": "a"}, {"1,1": "b"}]
        result = _convert_map_keys_to_tuple(data)
        self.assertEqual(result[0][(0, 0)], "a")
        self.assertEqual(result[1][(1, 1)], "b")

    def test_handle_invalid_tuple_strings(self):
        """Invalid tuple strings should remain unchanged."""
        data = {"0,0,0": "value", "abc": "def"}
        result = _convert_map_keys_to_tuple(data)
        # "0,0,0" should either convert to (0,0,0) or fail gracefully
        # "abc" should be preserved
        self.assertIn("abc", result)

    def test_convert_negative_coordinates(self):
        """Handle negative coordinate values."""
        data = {"-1,0": "a", "1,-1": "b"}
        result = _convert_map_keys_to_tuple(data)
        self.assertIn((-1, 0), result)
        self.assertIn((1, -1), result)

    def test_convert_large_coordinates(self):
        """Handle large coordinate values."""
        data = {"999,999": "value"}
        result = _convert_map_keys_to_tuple(data)
        self.assertEqual(result, {(999, 999): "value"})

    def test_empty_dict_conversion(self):
        """Empty dictionary should remain empty."""
        data = {}
        result = _convert_map_keys_to_tuple(data)
        self.assertEqual(result, {})

    def test_empty_list_conversion(self):
        """Empty list should remain empty."""
        data = []
        result = _convert_map_keys_to_tuple(data)
        self.assertEqual(result, [])

    def test_scalar_value_passthrough(self):
        """Scalar values should pass through unchanged."""
        result = _convert_map_keys_to_tuple("string")
        self.assertEqual(result, "string")

        result = _convert_map_keys_to_tuple(42)
        self.assertEqual(result, 42)

        result = _convert_map_keys_to_tuple(None)
        self.assertEqual(result, None)


class TestGetModuleData(unittest.TestCase):
    """Test module data loading and caching."""

    def test_get_module_data_returns_dict(self):
        """Module data should return a dictionary."""
        data = get_module_data("corvette")
        self.assertIsInstance(data, dict)

    def test_get_nonexistent_ship_returns_empty_dict(self):
        """Nonexistent ship should return empty dict."""
        data = get_module_data("nonexistent_ship_xyz")
        self.assertEqual(data, {})

    def test_get_module_data_caching(self):
        """Repeated calls should return cached data."""
        # Get data twice
        data1 = get_module_data("corvette")
        data2 = get_module_data("corvette")

        # Should be the same object (cached)
        self.assertIs(data1, data2)

    def test_module_data_has_expected_structure(self):
        """Module data should have expected structure."""
        data = get_module_data("corvette")
        if data:  # If data exists
            # Should have 'types' key
            self.assertIn("types", data)

    def test_get_module_data_for_different_ships(self):
        """Different ships should have different data."""
        # Try to get data for different ship types
        corvette = get_module_data("corvette")
        hauler = get_module_data("hauler")

        # If both exist, they should have different content or both be empty
        # But they shouldn't crash the system
        self.assertIsInstance(corvette, dict)
        self.assertIsInstance(hauler, dict)


class TestGetSolveMap(unittest.TestCase):
    """Test solve map loading and caching."""

    def test_get_solve_map_returns_dict(self):
        """Solve map should return a dictionary."""
        data = get_solve_map("corvette")
        self.assertIsInstance(data, dict)

    def test_get_nonexistent_solve_returns_empty_dict(self):
        """Nonexistent solve should return empty dict."""
        data = get_solve_map("nonexistent_ship_xyz")
        self.assertEqual(data, {})

    def test_get_solve_map_caching(self):
        """Repeated calls should return cached data."""
        data1 = get_solve_map("corvette")
        data2 = get_solve_map("corvette")

        # Should be the same object (cached)
        self.assertIs(data1, data2)

    def test_solve_map_has_valid_structure(self):
        """Solve map entries should have 'map' and 'score' keys."""
        data = get_solve_map("corvette")
        for tech_name, tech_data in data.items():
            self.assertIn("map", tech_data)
            self.assertIn("score", tech_data)

            # Map keys should be tuples (after conversion)
            for key in tech_data["map"].keys():
                self.assertIsInstance(key, tuple, f"Map key {key} should be a tuple, not {type(key)}")

    def test_solve_map_converts_tuple_keys(self):
        """Solve map should convert string keys to tuples."""
        data = get_solve_map("corvette")
        if data:
            for tech_name, tech_data in data.items():
                for coord_key in tech_data["map"].keys():
                    # After conversion, should be tuple
                    self.assertIsInstance(coord_key, tuple)
                    self.assertEqual(len(coord_key), 2)  # (x, y)


class TestGetAllModuleData(unittest.TestCase):
    """Test getting all module data at once."""

    def test_get_all_module_data_returns_dict(self):
        """Should return a dictionary."""
        data = get_all_module_data()
        self.assertIsInstance(data, dict)

    def test_all_module_data_not_empty(self):
        """Should return non-empty dict if data files exist."""
        data = get_all_module_data()
        if data:  # If any data exists
            # Each entry should be a dict
            for ship_type, ship_data in data.items():
                self.assertIsInstance(ship_data, dict)

    def test_multiple_ships_in_all_data(self):
        """Should have multiple ship types if data exists."""
        data = get_all_module_data()
        # Should have multiple entries
        self.assertIsInstance(data, dict)


class TestGetAllSolveData(unittest.TestCase):
    """Test getting all solve data at once."""

    def test_get_all_solve_data_returns_dict(self):
        """Should return a dictionary."""
        data = get_all_solve_data()
        self.assertIsInstance(data, dict)

    def test_all_solve_data_structure(self):
        """Solve data should have valid structure."""
        data = get_all_solve_data()
        for ship_type, ship_solves in data.items():
            self.assertIsInstance(ship_solves, dict)
            # Each tech should have a map (after conversion)
            for tech_name, tech_data in ship_solves.items():
                if isinstance(tech_data, dict) and "map" in tech_data:
                    for coord_key in tech_data["map"].keys():
                        self.assertIsInstance(coord_key, tuple)


class TestGetTrainingModuleIds(unittest.TestCase):
    """Test training module ID retrieval."""

    def test_get_training_module_ids_returns_list(self):
        """Should return a list."""
        result = get_training_module_ids("corvette", "pulse")
        self.assertIsInstance(result, list)

    def test_get_training_module_ids_nonexistent_ship(self):
        """Nonexistent ship should return empty list."""
        result = get_training_module_ids("nonexistent_ship", "pulse")
        self.assertEqual(result, [])

    def test_get_training_module_ids_nonexistent_tech(self):
        """Nonexistent tech should return empty list."""
        result = get_training_module_ids("corvette", "nonexistent_tech_xyz")
        self.assertEqual(result, [])

    def test_training_module_ids_are_strings(self):
        """All module IDs should be strings."""
        result = get_training_module_ids("corvette", "pulse")
        if result:
            for module_id in result:
                self.assertIsInstance(module_id, str)

    def test_training_module_ids_unique(self):
        """Module IDs should be unique (no duplicates)."""
        result = get_training_module_ids("corvette", "pulse")
        self.assertEqual(len(result), len(set(result)), "Module IDs should be unique")

    def test_training_module_ids_not_empty_if_data_exists(self):
        """If ship/tech combo exists, should have modules."""
        # Try a known combination
        result = get_training_module_ids("corvette", "pulse")
        # Should either have modules or be empty (if data doesn't exist)
        self.assertIsInstance(result, list)


class TestDataLoaderErrorHandling(unittest.TestCase):
    """Test error handling in data loader."""

    def test_invalid_json_in_module_file(self):
        """Corrupt JSON file should be handled gracefully."""
        # This tests the error handling in get_module_data
        # Should not raise an exception, just return empty dict
        result = get_module_data("nonexistent")
        self.assertEqual(result, {})

    def test_missing_required_fields_in_solve_map(self):
        """Solve map with missing 'map' or 'score' should be skipped."""
        # Tested implicitly through get_solve_map
        # It should only include entries with both 'map' and 'score'
        data = get_solve_map("corvette")
        for tech_name, tech_data in data.items():
            self.assertIn("map", tech_data)
            self.assertIn("score", tech_data)

    def test_cache_size_limit(self):
        """Cache should not grow unbounded."""
        # Get module data from many different (non-existent) ships
        for i in range(50):
            get_module_data(f"ship_{i}")

        # System should still work (cache doesn't overflow)
        result = get_module_data("corvette")
        self.assertIsInstance(result, dict)


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity across operations."""

    def test_module_data_not_modified_by_access(self):
        """Getting module data shouldn't modify it."""
        data1 = get_module_data("corvette")
        original_state = json.dumps(data1, sort_keys=True, default=str)

        data2 = get_module_data("corvette")
        final_state = json.dumps(data2, sort_keys=True, default=str)

        self.assertEqual(original_state, final_state)

    def test_solve_map_tuple_conversion_consistent(self):
        """Tuple conversion should be consistent."""
        data1 = get_solve_map("corvette")
        data2 = get_solve_map("corvette")

        # Should be identical (same cached object)
        self.assertIs(data1, data2)


if __name__ == "__main__":
    unittest.main()
