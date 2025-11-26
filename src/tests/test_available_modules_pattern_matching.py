"""
Adversarial tests for available_modules vs solve pattern matching logic in optimize_placement.

Tests validate the decision to use pattern matching vs windowed SA based on whether
available_modules exactly match the solve pattern modules.
"""

import unittest
from unittest.mock import patch, MagicMock
from src.grid_utils import Grid
from src.optimization.core import optimize_placement


class TestAvailableModulesPatternMatching(unittest.TestCase):
    """Tests for the logic that detects available modules vs pattern modules"""

    def setUp(self):
        """Set up test fixtures"""
        self.grid = Grid(10, 6)
        self.ship = "corvette"
        self.tech = "trails"
        self.modules = {
            "corvette": {
                "trails": [
                    {"id": "AB", "label": "AB", "bonus": 5},
                    {"id": "PB", "label": "PB", "bonus": 5},
                    {"id": "SB", "label": "SB", "bonus": 5},
                    {"id": "RT", "label": "RT", "bonus": 5},
                    {"id": "CT", "label": "CT", "bonus": 5},
                    {"id": "ET", "label": "ET", "bonus": 5},
                    {"id": "GT", "label": "GT", "bonus": 5},
                    {"id": "ST", "label": "ST", "bonus": 5},
                    {"id": "SP", "label": "SP", "bonus": 5},
                    {"id": "TT", "label": "TT", "bonus": 5},
                    {"id": "PT", "label": "PT", "bonus": 5},
                    {"id": "AT", "label": "AT", "bonus": 5},
                ]
            }
        }
        # Pattern has only 7 modules
        self.pattern_solve = {
            self.ship: {
                self.tech: {
                    "map": {
                        (0, 0): "RT",
                        (1, 0): "None",
                        (2, 0): "ET",
                        (0, 1): "CT",
                        (1, 1): "AB",
                        (2, 1): "GT",
                        (0, 2): "ST",
                        (1, 2): "PB",
                        (2, 2): "None",
                    },
                    "score": 0.1334,
                }
            }
        }

    def test_exact_match_with_pattern_uses_pattern_matching(self):
        """When available_modules exactly match pattern modules, use pattern matching"""
        # These 7 modules exactly match the pattern
        available_modules = ["AB", "PB", "RT", "CT", "ET", "GT", "ST"]

        with patch("src.optimization.core.get_tech_modules") as mock_get_tech:
            with patch("src.optimization.core.get_solve_map") as mock_get_solve:
                with patch(
                    "src.optimization.core.filter_solves",
                    return_value=self.pattern_solve,
                ):
                    with patch(
                        "src.optimization.core.get_all_unique_pattern_variations"
                    ) as mock_get_patterns:
                        with patch(
                            "src.optimization.core.apply_pattern_to_grid"
                        ) as mock_apply:
                            with patch(
                                "src.optimization.core.calculate_grid_score",
                                return_value=0.1334,
                            ):
                                with patch(
                                    "src.optimization.core.clear_all_modules_of_tech"
                                ):
                                    with patch(
                                        "src.optimization.core.print_grid_compact"
                                    ):
                                        mock_get_tech.return_value = [
                                            m
                                            for m in self.modules["corvette"]["trails"]
                                            if m["id"] in available_modules
                                        ]
                                        mock_get_solve.return_value = (
                                            self.pattern_solve[self.ship]
                                        )
                                        mock_get_patterns.return_value = [
                                            {(0, 0): "RT"}
                                        ]
                                        test_grid = self.grid.copy()
                                        mock_apply.return_value = (
                                            test_grid,
                                            1.0,
                                        )

                                        # Call optimize_placement
                                        result_grid, percentage, bonus, method = (
                                            optimize_placement(
                                                self.grid,
                                                self.ship,
                                                self.modules,
                                                self.tech,
                                                available_modules=available_modules,
                                            )
                                        )

                                        # Should use "Pattern Match" method, not windowed SA
                                        self.assertIn(
                                            "Pattern Match",
                                            method,
                                            f"Expected Pattern Match, got {method}",
                                        )

    def test_more_available_than_pattern_uses_windowed_sa(self):
        """When available_modules has more than pattern, use windowed SA"""
        # 12 available modules, but pattern only has 7
        available_modules = [
            "AB",
            "PB",
            "SB",
            "RT",
            "CT",
            "ET",
            "GT",
            "ST",
            "SP",
            "TT",
            "PT",
            "AT",
        ]

        with patch("src.optimization.core.get_tech_modules") as mock_get_tech:
            with patch("src.optimization.core.get_solve_map") as mock_get_solve:
                with patch(
                    "src.optimization.core.filter_solves",
                    return_value=self.pattern_solve,
                ):
                    with patch(
                        "src.optimization.core.find_supercharged_opportunities",
                        return_value=None,
                    ):
                        with patch(
                            "src.optimization.core.determine_window_dimensions",
                            return_value=(3, 3),
                        ):
                            with patch(
                                "src.optimization.core._scan_grid_with_window"
                            ) as mock_scan:
                                with patch(
                                    "src.optimization.core._handle_sa_refine_opportunity"
                                ) as mock_sa:
                                    with patch(
                                        "src.optimization.core.clear_all_modules_of_tech"
                                    ):
                                        with patch(
                                            "src.optimization.core.print_grid_compact"
                                        ):
                                            with patch(
                                                "src.optimization.core.calculate_grid_score",
                                                return_value=0.2,
                                            ):
                                                mock_get_tech.return_value = [
                                                    m
                                                    for m in self.modules["corvette"][
                                                        "trails"
                                                    ]
                                                    if m["id"] in available_modules
                                                ]
                                                mock_get_solve.return_value = (
                                                    self.pattern_solve[self.ship]
                                                )
                                                mock_scan.return_value = (5.0, (0, 0))
                                                test_grid = self.grid.copy()
                                                mock_sa.return_value = (
                                                    test_grid,
                                                    0.2,
                                                )

                                                result_grid, percentage, bonus, method = (
                                                    optimize_placement(
                                                        self.grid,
                                                        self.ship,
                                                        self.modules,
                                                        self.tech,
                                                        available_modules=available_modules,
                                                    )
                                                )

                                                # Should use windowed SA, not pattern match
                                                self.assertIn(
                                                    "SA",
                                                    method,
                                                    f"Expected SA method, got {method}",
                                                )

    def test_different_modules_than_pattern_uses_windowed_sa(self):
        """When available_modules differ (not superset), use windowed SA"""
        # Different 7 modules than the pattern
        available_modules = ["SB", "SP", "TT", "PT", "AT", "XA", "XB"]

        with patch("src.optimization.core.get_tech_modules") as mock_get_tech:
            with patch("src.optimization.core.get_solve_map") as mock_get_solve:
                with patch(
                    "src.optimization.core.filter_solves",
                    return_value=self.pattern_solve,
                ):
                    with patch(
                        "src.optimization.core.find_supercharged_opportunities",
                        return_value=None,
                    ):
                        with patch(
                            "src.optimization.core.determine_window_dimensions",
                            return_value=(3, 3),
                        ):
                            with patch(
                                "src.optimization.core._scan_grid_with_window"
                            ) as mock_scan:
                                with patch(
                                    "src.optimization.core._handle_sa_refine_opportunity"
                                ) as mock_sa:
                                    with patch(
                                        "src.optimization.core.clear_all_modules_of_tech"
                                    ):
                                        with patch(
                                            "src.optimization.core.print_grid_compact"
                                        ):
                                            with patch(
                                                "src.optimization.core.calculate_grid_score",
                                                return_value=0.2,
                                            ):
                                                mock_get_tech.return_value = [
                                                    {
                                                        "id": m,
                                                        "label": m,
                                                        "bonus": 5,
                                                    }
                                                    for m in available_modules
                                                ]
                                                mock_get_solve.return_value = (
                                                    self.pattern_solve[self.ship]
                                                )
                                                mock_scan.return_value = (5.0, (0, 0))
                                                test_grid = self.grid.copy()
                                                mock_sa.return_value = (
                                                    test_grid,
                                                    0.2,
                                                )

                                                result_grid, percentage, bonus, method = (
                                                    optimize_placement(
                                                        self.grid,
                                                        self.ship,
                                                        self.modules,
                                                        self.tech,
                                                        available_modules=available_modules,
                                                    )
                                                )

                                                # Should use windowed SA
                                                self.assertIn(
                                                    "SA",
                                                    method,
                                                    f"Expected SA method, got {method}",
                                                )

    def test_no_available_modules_uses_tech_modules(self):
        """When available_modules is None, use all tech_modules from filter"""
        with patch("src.optimization.core.get_tech_modules") as mock_get_tech:
            with patch("src.optimization.core.get_solve_map") as mock_get_solve:
                with patch(
                    "src.optimization.core.filter_solves",
                    return_value=self.pattern_solve,
                ):
                    with patch(
                        "src.optimization.core.get_all_unique_pattern_variations"
                    ) as mock_get_patterns:
                        with patch(
                            "src.optimization.core.apply_pattern_to_grid"
                        ) as mock_apply:
                            with patch(
                                "src.optimization.core.calculate_grid_score",
                                return_value=0.1334,
                            ):
                                with patch(
                                    "src.optimization.core.clear_all_modules_of_tech"
                                ):
                                    with patch(
                                        "src.optimization.core.print_grid_compact"
                                    ):
                                        # All 12 modules available
                                        all_modules = [
                                            m
                                            for m in self.modules["corvette"]["trails"]
                                        ]
                                        mock_get_tech.return_value = all_modules
                                        mock_get_solve.return_value = (
                                            self.pattern_solve[self.ship]
                                        )
                                        mock_get_patterns.return_value = [
                                            {(0, 0): "RT"}
                                        ]
                                        test_grid = self.grid.copy()
                                        mock_apply.return_value = (
                                            test_grid,
                                            1.0,
                                        )

                                        # Call with available_modules=None
                                        result_grid, percentage, bonus, method = (
                                            optimize_placement(
                                                self.grid,
                                                self.ship,
                                                self.modules,
                                                self.tech,
                                                available_modules=None,
                                            )
                                        )

                                        # Since 12 modules != 7 in pattern, should use SA
                                        # (or at least not pure pattern match)
                                        self.assertIsNotNone(result_grid)

    def test_pattern_with_none_values_excluded_from_comparison(self):
        """Pattern with 'None' values should not be included in module count"""
        # Pattern is 3x3 with 2 None values, so 7 actual modules
        available_modules = ["AB", "PB", "RT", "CT", "ET", "GT", "ST"]

        # Create a pattern map where some cells are "None"
        pattern_with_nones = {
            self.ship: {
                self.tech: {
                    "map": {
                        (0, 0): "RT",
                        (1, 0): "None",  # Should be ignored
                        (2, 0): "ET",
                        (0, 1): "CT",
                        (1, 1): "AB",
                        (2, 1): "GT",
                        (0, 2): "ST",
                        (1, 2): "PB",
                        (2, 2): "None",  # Should be ignored
                    },
                    "score": 0.1334,
                }
            }
        }

        with patch("src.optimization.core.get_tech_modules") as mock_get_tech:
            with patch("src.optimization.core.get_solve_map") as mock_get_solve:
                with patch(
                    "src.optimization.core.filter_solves",
                    return_value=pattern_with_nones,
                ):
                    with patch(
                        "src.optimization.core.get_all_unique_pattern_variations"
                    ) as mock_get_patterns:
                        with patch(
                            "src.optimization.core.apply_pattern_to_grid"
                        ) as mock_apply:
                            with patch(
                                "src.optimization.core.calculate_grid_score",
                                return_value=0.1334,
                            ):
                                with patch(
                                    "src.optimization.core.clear_all_modules_of_tech"
                                ):
                                    with patch(
                                        "src.optimization.core.print_grid_compact"
                                    ):
                                        mock_get_tech.return_value = [
                                            m
                                            for m in self.modules["corvette"]["trails"]
                                            if m["id"] in available_modules
                                        ]
                                        mock_get_solve.return_value = (
                                            pattern_with_nones[self.ship]
                                        )
                                        mock_get_patterns.return_value = [
                                            {(0, 0): "RT"}
                                        ]
                                        test_grid = self.grid.copy()
                                        mock_apply.return_value = (
                                            test_grid,
                                            1.0,
                                        )

                                        result_grid, percentage, bonus, method = (
                                            optimize_placement(
                                                self.grid,
                                                self.ship,
                                                self.modules,
                                                self.tech,
                                                available_modules=available_modules,
                                            )
                                        )

                                        # Should match pattern (7 modules match exactly)
                                        self.assertIn("Pattern Match", method)

    def test_pattern_with_dict_format_modules(self):
        """Pattern with dict format modules (with 'id' key) should be parsed correctly"""
        available_modules = ["AB", "PB", "RT", "CT", "ET", "GT", "ST"]

        # Pattern with dict format
        pattern_dict_format = {
            self.ship: {
                self.tech: {
                    "map": {
                        (0, 0): {"id": "RT", "bonus": 5},
                        (1, 0): "None",
                        (2, 0): {"id": "ET", "bonus": 5},
                        (0, 1): {"id": "CT", "bonus": 5},
                        (1, 1): {"id": "AB", "bonus": 5},
                        (2, 1): {"id": "GT", "bonus": 5},
                        (0, 2): {"id": "ST", "bonus": 5},
                        (1, 2): {"id": "PB", "bonus": 5},
                        (2, 2): "None",
                    },
                    "score": 0.1334,
                }
            }
        }

        with patch("src.optimization.core.get_tech_modules") as mock_get_tech:
            with patch("src.optimization.core.get_solve_map") as mock_get_solve:
                with patch(
                    "src.optimization.core.filter_solves",
                    return_value=pattern_dict_format,
                ):
                    with patch(
                        "src.optimization.core.get_all_unique_pattern_variations"
                    ) as mock_get_patterns:
                        with patch(
                            "src.optimization.core.apply_pattern_to_grid"
                        ) as mock_apply:
                            with patch(
                                "src.optimization.core.calculate_grid_score",
                                return_value=0.1334,
                            ):
                                with patch(
                                    "src.optimization.core.clear_all_modules_of_tech"
                                ):
                                    with patch(
                                        "src.optimization.core.print_grid_compact"
                                    ):
                                        mock_get_tech.return_value = [
                                            m
                                            for m in self.modules["corvette"]["trails"]
                                            if m["id"] in available_modules
                                        ]
                                        mock_get_solve.return_value = (
                                            pattern_dict_format[self.ship]
                                        )
                                        mock_get_patterns.return_value = [
                                            {(0, 0): "RT"}
                                        ]
                                        test_grid = self.grid.copy()
                                        mock_apply.return_value = (
                                            test_grid,
                                            1.0,
                                        )

                                        result_grid, percentage, bonus, method = (
                                            optimize_placement(
                                                self.grid,
                                                self.ship,
                                                self.modules,
                                                self.tech,
                                                available_modules=available_modules,
                                            )
                                        )

                                        # Should match pattern (7 modules match exactly)
                                        self.assertIn("Pattern Match", method)

    def test_empty_available_modules_list(self):
        """When available_modules is empty list, should return error tuple"""
        available_modules = []

        with patch("src.optimization.core.get_tech_modules") as mock_get_tech:
            with patch("src.optimization.core.get_solve_map") as mock_get_solve:
                with patch(
                    "src.optimization.core.filter_solves",
                    return_value=self.pattern_solve,
                ):
                    with patch(
                        "src.optimization.core.clear_all_modules_of_tech"
                    ):
                        with patch(
                            "src.optimization.core.print_grid_compact"
                        ):
                            mock_get_tech.return_value = []
                            mock_get_solve.return_value = (
                                self.pattern_solve[self.ship]
                            )

                            result_grid, percentage, bonus, method = (
                                optimize_placement(
                                    self.grid,
                                    self.ship,
                                    self.modules,
                                    self.tech,
                                    available_modules=available_modules,
                                )
                            )

                            # Should return error tuple with cleared grid and error message
                            self.assertIsNotNone(result_grid)
                            self.assertEqual(percentage, 0.0)
                            self.assertEqual(bonus, 0.0)
                            self.assertIn("Module Definition Error", method)

    def test_subset_of_pattern_modules(self):
        """When available is subset of pattern, use windowed SA"""
        # Only 4 of the 7 pattern modules available
        available_modules = ["AB", "PB", "RT", "CT"]

        with patch("src.optimization.core.get_tech_modules") as mock_get_tech:
            with patch("src.optimization.core.get_solve_map") as mock_get_solve:
                with patch(
                    "src.optimization.core.filter_solves",
                    return_value=self.pattern_solve,
                ):
                    with patch(
                        "src.optimization.core.find_supercharged_opportunities",
                        return_value=None,
                    ):
                        with patch(
                            "src.optimization.core.determine_window_dimensions",
                            return_value=(3, 3),
                        ):
                            with patch(
                                "src.optimization.core._scan_grid_with_window"
                            ) as mock_scan:
                                with patch(
                                    "src.optimization.core._handle_sa_refine_opportunity"
                                ) as mock_sa:
                                    with patch(
                                        "src.optimization.core.clear_all_modules_of_tech"
                                    ):
                                        with patch(
                                            "src.optimization.core.print_grid_compact"
                                        ):
                                            with patch(
                                                "src.optimization.core.calculate_grid_score",
                                                return_value=0.1,
                                            ):
                                                mock_get_tech.return_value = [
                                                    m
                                                    for m in self.modules["corvette"][
                                                        "trails"
                                                    ]
                                                    if m["id"] in available_modules
                                                ]
                                                mock_get_solve.return_value = (
                                                    self.pattern_solve[self.ship]
                                                )
                                                mock_scan.return_value = (5.0, (0, 0))
                                                test_grid = self.grid.copy()
                                                mock_sa.return_value = (
                                                    test_grid,
                                                    0.1,
                                                )

                                                result_grid, percentage, bonus, method = (
                                                    optimize_placement(
                                                        self.grid,
                                                        self.ship,
                                                        self.modules,
                                                        self.tech,
                                                        available_modules=available_modules,
                                                    )
                                                )

                                                # Should use windowed SA
                                                self.assertIn(
                                                    "SA",
                                                    method,
                                                    f"Expected SA method, got {method}",
                                                )


if __name__ == "__main__":
    unittest.main()
