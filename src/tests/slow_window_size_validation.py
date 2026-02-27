"""
Empirical validation tests for determine_window_dimensions using optimized score comparison.

This test suite validates that the window dimensions returned by
determine_window_dimensions are optimal by:
1. Generating random supercharged placements
2. Running simulated annealing to place modules in different window sizes
3. Comparing the actual optimized scores to ensure chosen size performs best
"""

import unittest
import pytest
import random
from typing import List, Tuple
from src.optimization.helpers import determine_window_dimensions
from src.data_loader import get_all_module_data, get_training_module_ids
from src.modules_utils import get_tech_modules
from src.grid_utils import Grid
from src.optimization.refinement import simulated_annealing


class TestWindowSizeValidation(unittest.TestCase):
    """Empirically validate window dimensions via optimized score comparison"""

    @classmethod
    def setUpClass(cls):
        """Load all module data once for all tests"""
        cls.all_modules = get_all_module_data()
        if not cls.all_modules:
            raise RuntimeError("Failed to load module data for validation tests")

    def _generate_random_sc_placements(
        self, width: int, height: int, num_samples: int = 20, max_sc: int = 4
    ) -> List[List[Tuple[int, int]]]:
        """
        Generate random supercharged placements for testing.

        Args:
            width: Grid width
            height: Grid height
            num_samples: Number of random configurations to generate
            max_sc: Maximum supercharged cells per configuration

        Returns:
            List of SC placement lists, where each placement is a list of (x, y) tuples
        """
        all_positions = [(x, y) for y in range(height) for x in range(width)]
        sc_placements = []

        for _ in range(num_samples):
            num_sc = random.randint(0, min(max_sc, len(all_positions)))
            if num_sc > 0:
                sc_cells = random.sample(all_positions, num_sc)
            else:
                sc_cells = []
            sc_placements.append(sc_cells)

        return sc_placements

    def _optimize_grid(
        self,
        width: int,
        height: int,
        sc_cells: List[Tuple[int, int]],
        ship: str,
        tech: str,
        tech_modules: list,
    ) -> float:
        """
        Run SA optimization on a grid and return the final score.

        Args:
            width: Grid width
            height: Grid height
            sc_cells: List of (x, y) supercharged cell positions
            ship: Ship identifier
            tech: Tech identifier
            tech_modules: List of module definitions

        Returns:
            Final optimized score
        """
        grid = Grid(width, height)

        # Activate all cells
        for y in range(height):
            for x in range(width):
                grid.set_active(x, y, True)

        # Set supercharged cells
        for x, y in sc_cells:
            grid.set_supercharged(x, y, True)

        # Run simulated annealing
        optimized_grid, final_score = simulated_annealing(
            grid=grid,
            ship=ship,
            modules=self.all_modules,
            tech=tech,
            full_grid=grid,
            tech_modules=tech_modules,
        )

        return final_score

    def _compare_window_scores_with_optimization(
        self,
        ship: str,
        tech: str,
        tech_modules: list,
        window_sizes: List[Tuple[int, int]],
        num_samples: int = 20,
    ) -> dict:
        """
        Compare average optimized scores across different window sizes.

        Args:
            ship: Ship identifier
            tech: Technology key
            tech_modules: List of module definitions
            window_sizes: List of (width, height) tuples to compare
            num_samples: Number of random SC configurations to test

        Returns:
            Dict mapping (w, h) to average optimized score across all samples
        """
        scores_by_size = {size: [] for size in window_sizes}

        for size in window_sizes:
            w, h = size
            sc_placements = self._generate_random_sc_placements(w, h, num_samples)

            for sc_cells in sc_placements:
                score = self._optimize_grid(w, h, sc_cells, ship, tech, tech_modules)
                scores_by_size[size].append(score)

        # Return average scores
        return {size: sum(scores) / len(scores) if scores else 0.0 for size, scores in scores_by_size.items()}

    @pytest.mark.slow
    def test_chosen_window_size_is_optimal_with_sa(self):
        """
        Test that for key module counts, the chosen window size produces
        higher optimized scores on average after SA optimization.

        This validates critical decision points like:
        - pulse with 7: should 4x2 beat 3x3?
        - hyper with 9: should 3x3 beat 4x2?
        - trails with 12: should 4x4 beat 4x3?

        NOTE: This test is marked as @pytest.mark.slow because it takes ~15 minutes.
        It will be excluded from pre-commit hooks.
        """
        test_cases = [
            # Corvette tests
            ("pulse", "corvette", [(4, 3), (4, 2)], "corvette pulse (10 modules): 4x3 vs 4x2"),
            ("hyper", "corvette", [(4, 4), (4, 3)], "corvette hyper (12 modules): 4x4 vs 4x3"),
            ("trails", "corvette", [(4, 3), (4, 4)], "corvette trails (12 modules): 4x3 vs 4x4"),
            # Atlantid tests
            ("bolt-caster", "atlantid", [(4, 3), (3, 3)], "atlantid bolt-caster (9 modules): 4x3 vs 3x3"),
            ("pulse-spitter", "atlantid", [(3, 3), (4, 2)], "atlantid pulse-spitter (7 modules): 3x3 vs 4x2"),
            ("mining", "atlantid", [(3, 3), (4, 2)], "atlantid mining (7 modules): 3x3 vs 4x2"),
            # Standard ship tests
            ("hyper", "standard", [(3, 3), (4, 2)], "standard hyper (9 modules): 3x3 vs 4x2"),
            ("trails", "standard", [(4, 3), (4, 4)], "standard trails (12 modules): 4x3 vs 4x4"),
        ]

        num_samples = 15  # Balance between confidence and test speed

        for tech, ship, window_sizes, description in test_cases:
            with self.subTest(description=description):
                # Get modules for this tech
                # get_tech_modules expects ship data directly, not wrapped
                tech_modules = get_tech_modules(self.all_modules[ship], ship, tech)
                if not tech_modules:
                    self.skipTest(f"No modules found for {ship}/{tech}")
                    continue

                module_count = len(tech_modules)
                chosen_w, chosen_h = determine_window_dimensions(module_count, tech, ship, self.all_modules[ship])
                chosen_size = (chosen_w, chosen_h)

                # Only test if the chosen size is in our alternatives
                if chosen_size not in window_sizes:
                    print(f"\n  Skipping {description}: chosen {chosen_size} not in alternatives {window_sizes}")
                    self.skipTest(f"Chosen size {chosen_size} not in test alternatives")
                    continue

                print(f"\n  Testing: {description}")
                print(f"    Module count: {module_count}, Chosen size: {chosen_size}")
                print(f"    Running SA optimization with {num_samples} random SC configs per size...")

                avg_scores = self._compare_window_scores_with_optimization(
                    ship, tech, tech_modules, window_sizes, num_samples
                )

                chosen_score = avg_scores[chosen_size]
                max_score = max(avg_scores.values())

                # Allow tolerance for statistical noise (within 10%)
                tolerance = max_score * 0.10

                print(f"    Chosen: {chosen_size}, Avg Score: {chosen_score:.4f}")
                print(f"    All avg scores: {avg_scores}")

                self.assertGreaterEqual(
                    chosen_score,
                    max_score - tolerance,
                    f"{description}: Chosen {chosen_size} scored {chosen_score:.4f}, "
                    f"but max was {max_score:.4f}. Scores: {avg_scores}",
                )

    def test_basic_structural_validity(self):
        """
        Quick structural validation that window sizes can hold modules.
        This is a sanity check before running expensive SA tests.
        """
        validations_passed = 0

        for ship, ship_data in self.all_modules.items():
            if "types" not in ship_data:
                continue

            for category_data in ship_data["types"].values():
                if not isinstance(category_data, list):
                    continue

                for tech_info in category_data:
                    if not isinstance(tech_info, dict):
                        continue

                    tech = tech_info.get("key")
                    if not tech:
                        continue

                    training_module_ids = get_training_module_ids(ship, tech)
                    if not training_module_ids:
                        continue

                    module_count = len(training_module_ids)

                    with self.subTest(ship=ship, tech=tech, module_count=module_count):
                        w, h = determine_window_dimensions(module_count, tech, ship, ship_data)

                        self.assertGreater(w, 0)
                        self.assertGreater(h, 0)
                        self.assertGreaterEqual(w * h, module_count)
                        self.assertLessEqual(w, 4)
                        self.assertLessEqual(h, 4)

                        validations_passed += 1

        print(f"\n✓ Validated {validations_passed} ship/tech combinations")
        self.assertGreater(validations_passed, 0)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
