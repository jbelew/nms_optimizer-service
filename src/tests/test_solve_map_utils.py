"""
Comprehensive test suite for solve_map_utils.py

This test suite focuses on finding bugs in solve map filtering logic,
module ownership checking, and edge cases.
"""

import unittest
from unittest.mock import patch
from src.solve_map_utils import filter_solves


class TestFilterSolves(unittest.TestCase):
    """Test the filter_solves function."""

    def setUp(self):
        """Set up common test fixtures."""
        self.modules_data = {
            "types": {
                "Pulse": [
                    {
                        "key": "pulse",
                        "modules": [
                            {
                                "id": "pulse_a",
                                "label": "Pulse A",
                                "bonus": 10.0,
                                "adjacency": "greater",
                                "type": "bonus",
                                "sc_eligible": True,
                                "image": None,
                            },
                            {
                                "id": "pulse_b",
                                "label": "Pulse B",
                                "bonus": 8.0,
                                "adjacency": "lesser",
                                "type": "bonus",
                                "sc_eligible": False,
                                "image": None,
                            },
                        ],
                    }
                ],
                "Engineering": [
                    {
                        "key": "engineering",
                        "modules": [
                            {
                                "id": "eng_a",
                                "label": "Eng A",
                                "bonus": 12.0,
                                "adjacency": "greater",
                                "type": "bonus",
                                "sc_eligible": True,
                                "image": None,
                            },
                        ],
                    }
                ],
            }
        }

        self.solves = {
            "corvette": {
                "pulse": {
                    "map": {
                        (0, 0): "pulse_a",
                        (1, 0): "pulse_b",
                        (2, 0): None,
                    },
                    "score": 25.5,
                },
                "engineering": {
                    "map": {
                        (0, 1): "eng_a",
                    },
                    "score": 12.0,
                },
            }
        }

    def test_filter_solves_basic(self):
        """Basic filtering should include owned modules and None slots."""
        result = filter_solves(self.solves, "corvette", self.modules_data, "pulse")

        self.assertIn("corvette", result)
        self.assertIn("pulse", result["corvette"])
        self.assertIn("map", result["corvette"]["pulse"])
        self.assertIn("score", result["corvette"]["pulse"])

    def test_filter_solves_preserves_score(self):
        """Filtered solve should preserve original score."""
        result = filter_solves(self.solves, "corvette", self.modules_data, "pulse")

        self.assertEqual(result["corvette"]["pulse"]["score"], 25.5)

    def test_filter_solves_includes_owned_modules(self):
        """Filtered solve should include modules the player owns."""
        result = filter_solves(self.solves, "corvette", self.modules_data, "pulse")

        filtered_map = result["corvette"]["pulse"]["map"]
        self.assertIn((0, 0), filtered_map)
        self.assertEqual(filtered_map[(0, 0)], "pulse_a")
        self.assertIn((1, 0), filtered_map)
        self.assertEqual(filtered_map[(1, 0)], "pulse_b")

    def test_filter_solves_includes_none_slots(self):
        """Filtered solve should include None (empty) slots."""
        result = filter_solves(self.solves, "corvette", self.modules_data, "pulse")

        filtered_map = result["corvette"]["pulse"]["map"]
        self.assertIn((2, 0), filtered_map)
        self.assertIsNone(filtered_map[(2, 0)])

    def test_filter_solves_nonexistent_ship(self):
        """Filtering for nonexistent ship should return empty dict."""
        result = filter_solves(self.solves, "nonexistent", self.modules_data, "pulse")

        self.assertEqual(result, {})

    def test_filter_solves_nonexistent_tech(self):
        """Filtering for nonexistent tech should return empty dict."""
        result = filter_solves(self.solves, "corvette", self.modules_data, "nonexistent")

        self.assertEqual(result, {})

    def test_filter_solves_no_modules_for_tech(self):
        """If no modules found for tech, should still return solve with all modules included."""
        # Create modules_data with no pulse modules
        modules_data = {"types": {"Engineering": [{"key": "engineering", "modules": []}]}}

        result = filter_solves(self.solves, "corvette", modules_data, "pulse")

        # Should still return the solve data even if modules can't be retrieved
        # This allows solves to proceed with fallback logic
        self.assertIn("corvette", result)
        self.assertIn("pulse", result["corvette"])
        # All modules from the original solve should be included
        self.assertEqual(len(result["corvette"]["pulse"]["map"]), 3)

    def test_filter_solves_excludes_unowned_modules(self):
        """Filtered solve should exclude modules the player doesn't own."""
        # Add a module to the solve that player doesn't own
        solves = {
            "corvette": {
                "pulse": {
                    "map": {
                        (0, 0): "pulse_a",
                        (1, 0): "unowned_module",
                    },
                    "score": 20.0,
                }
            }
        }

        result = filter_solves(solves, "corvette", self.modules_data, "pulse")

        filtered_map = result["corvette"]["pulse"]["map"]
        self.assertIn((0, 0), filtered_map)
        # Unowned module should be excluded
        self.assertNotIn((1, 0), filtered_map)

    def test_filter_solves_empty_solve_map(self):
        """Empty solve map should be handled gracefully."""
        solves = {"corvette": {"pulse": {"map": {}, "score": 0.0}}}

        result = filter_solves(solves, "corvette", self.modules_data, "pulse")

        self.assertEqual(result["corvette"]["pulse"]["map"], {})

    def test_filter_solves_multiple_techs(self):
        """Filtering different techs from same ship should work correctly."""
        result_pulse = filter_solves(self.solves, "corvette", self.modules_data, "pulse")
        result_eng = filter_solves(self.solves, "corvette", self.modules_data, "engineering")

        self.assertEqual(result_pulse["corvette"]["pulse"]["score"], 25.5)
        self.assertEqual(result_eng["corvette"]["engineering"]["score"], 12.0)

    def test_filter_solves_none_string_handling(self):
        """Should handle both None and "None" string values."""
        solves = {
            "corvette": {
                "pulse": {
                    "map": {
                        (0, 0): "pulse_a",
                        (1, 0): "None",  # String "None" instead of None
                        (2, 0): None,  # Actual None
                    },
                    "score": 20.0,
                }
            }
        }

        result = filter_solves(solves, "corvette", self.modules_data, "pulse")

        filtered_map = result["corvette"]["pulse"]["map"]
        # Both should be included
        self.assertIn((1, 0), filtered_map)
        self.assertIn((2, 0), filtered_map)

    def test_filter_solves_returns_new_dict(self):
        """Filtering should return a new dict, not modify original."""
        original_map = self.solves["corvette"]["pulse"]["map"].copy()

        filter_solves(self.solves, "corvette", self.modules_data, "pulse")

        # Original should be unchanged
        self.assertEqual(self.solves["corvette"]["pulse"]["map"], original_map)

    def test_filter_solves_with_no_available_modules(self):
        """Should handle available_modules parameter when None."""
        result = filter_solves(self.solves, "corvette", self.modules_data, "pulse", available_modules=None)

        self.assertIn("corvette", result)

    def test_filter_solves_preserves_map_keys(self):
        """Map position keys should be preserved (tuples)."""
        result = filter_solves(self.solves, "corvette", self.modules_data, "pulse")

        filtered_map = result["corvette"]["pulse"]["map"]
        for key in filtered_map.keys():
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)

    def test_filter_solves_missing_score_defaults_to_zero(self):
        """Missing score should default to 0."""
        solves = {
            "corvette": {
                "pulse": {
                    "map": {(0, 0): "pulse_a"},
                    # No score key
                }
            }
        }

        result = filter_solves(solves, "corvette", self.modules_data, "pulse")

        self.assertEqual(result["corvette"]["pulse"]["score"], 0)

    def test_filter_solves_large_pattern(self):
        """Should handle large solve patterns."""
        large_map = {(x, y): "pulse_a" if (x + y) % 2 == 0 else "pulse_b" for x in range(10) for y in range(10)}
        solves = {"corvette": {"pulse": {"map": large_map, "score": 100.0}}}

        result = filter_solves(solves, "corvette", self.modules_data, "pulse")

        # Should have many entries after filtering
        self.assertGreater(len(result["corvette"]["pulse"]["map"]), 0)


class TestFilterSolvesPhotonixOverride(unittest.TestCase):
    """Test the photonix override behavior for PC platform."""

    def setUp(self):
        """Set up test fixtures with photonix data."""
        self.modules_data = {
            "types": {
                "Pulse": [
                    {
                        "key": "pulse",
                        "modules": [
                            {
                                "id": "pulse_a",
                                "label": "Pulse A",
                                "bonus": 10.0,
                                "adjacency": "greater",
                                "type": "bonus",
                                "sc_eligible": True,
                                "image": None,
                            },
                        ],
                    }
                ]
            }
        }

        self.solves = {
            "corvette": {
                "pulse": {"map": {(0, 0): "pulse_a"}, "score": 10.0},
                "photonix": {"map": {(0, 0): "pulse_a"}, "score": 15.0},
            }
        }

    @patch("builtins.print")
    def test_filter_solves_photonix_override_with_pc(self, mock_print):
        """When tech is pulse and PC in available_modules, should use photonix."""
        result = filter_solves(self.solves, "corvette", self.modules_data, "pulse", available_modules=["PC"])

        # Should return photonix data instead
        self.assertEqual(result["corvette"]["pulse"]["score"], 15.0)
        # Should print info message
        mock_print.assert_called()

    def test_filter_solves_no_photonix_override_without_pc(self):
        """Without PC in available_modules, should not override."""
        result = filter_solves(
            self.solves, "corvette", self.modules_data, "pulse", available_modules=["other_platform"]
        )

        # Should return regular pulse data
        self.assertEqual(result["corvette"]["pulse"]["score"], 10.0)

    def test_filter_solves_no_photonix_override_for_other_techs(self):
        """Photonix override should only apply to pulse tech."""
        solves = {"corvette": {"engineering": {"map": {(0, 0): "eng_a"}, "score": 12.0}}}

        modules_data = {
            "types": {
                "Engineering": [
                    {
                        "key": "engineering",
                        "modules": [
                            {
                                "id": "eng_a",
                                "label": "Eng A",
                                "bonus": 12.0,
                                "adjacency": "greater",
                                "type": "bonus",
                                "sc_eligible": True,
                                "image": None,
                            }
                        ],
                    }
                ]
            }
        }

        result = filter_solves(solves, "corvette", modules_data, "engineering", available_modules=["PC"])

        # Should not override for non-pulse tech
        self.assertEqual(result["corvette"]["engineering"]["score"], 12.0)


class TestFilterSolvesEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_filter_solves_empty_solves_dict(self):
        """Empty solves dict should return empty result."""
        result = filter_solves({}, "corvette", {}, "pulse")
        self.assertEqual(result, {})

    def test_filter_solves_solve_data_none(self):
        """If solve_data is None or falsy, should return empty dict."""
        solves = {"corvette": {"pulse": None}}
        modules_data = {"types": {}}

        result = filter_solves(solves, "corvette", modules_data, "pulse")
        self.assertEqual(result, {})

    def test_filter_solves_with_duplicate_modules_in_pattern(self):
        """Pattern with repeated module IDs should be preserved."""
        solves = {
            "corvette": {
                "pulse": {
                    "map": {
                        (0, 0): "pulse_a",
                        (1, 0): "pulse_a",
                        (2, 0): "pulse_a",
                    },
                    "score": 30.0,
                }
            }
        }
        modules_data = {
            "types": {
                "Pulse": [
                    {
                        "key": "pulse",
                        "modules": [
                            {
                                "id": "pulse_a",
                                "label": "Pulse A",
                                "bonus": 10.0,
                                "adjacency": "greater",
                                "type": "bonus",
                                "sc_eligible": True,
                                "image": None,
                            }
                        ],
                    }
                ]
            }
        }

        result = filter_solves(solves, "corvette", modules_data, "pulse")

        filtered_map = result["corvette"]["pulse"]["map"]
        # All three positions should be included
        self.assertEqual(len(filtered_map), 3)

    def test_filter_solves_with_missing_map_key(self):
        """Solve data without 'map' key should handle gracefully."""
        solves = {
            "corvette": {
                "pulse": {
                    "score": 10.0
                    # No 'map' key
                }
            }
        }
        modules_data = {
            "types": {
                "Pulse": [
                    {
                        "key": "pulse",
                        "modules": [
                            {
                                "id": "pulse_a",
                                "label": "Pulse A",
                                "bonus": 10.0,
                                "adjacency": "greater",
                                "type": "bonus",
                                "sc_eligible": True,
                                "image": None,
                            }
                        ],
                    }
                ]
            }
        }

        result = filter_solves(solves, "corvette", modules_data, "pulse")

        # Should create empty map
        self.assertEqual(result["corvette"]["pulse"]["map"], {})


if __name__ == "__main__":
    unittest.main()
