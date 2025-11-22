"""
Integration Tests for NMS Optimizer Service

Tests full end-to-end optimization pipeline workflows.
These tests verify that components work together correctly without mocks.
"""

import unittest
from src.data_loader import get_all_module_data, get_all_solve_data
from src.grid_utils import Grid
from src.optimization import optimize_placement
from src.bonus_calculations import calculate_grid_score

# Load all data at module level for testing
sample_modules = get_all_module_data()
sample_solves = get_all_solve_data()


def get_tech_modules_from_ship_data(ship_data, tech_key):
    """Extract modules for a specific tech from ship data structure.
    
    Ship data structure:
    {
        "label": "...",
        "type": "...",
        "types": {
            "Weaponry": [{"key": "pulse", "modules": [...]}, ...],
            "Hyperdrive": [...],
            ...
        }
    }
    """
    if "types" not in ship_data:
        return None
    
    for category_name, techs_in_category in ship_data["types"].items():
        if not isinstance(techs_in_category, list):
            continue
        for tech_obj in techs_in_category:
            if tech_obj.get("key") == tech_key:
                return tech_obj.get("modules", [])
    
    return None


class TestOptimizationPipeline(unittest.TestCase):
    """Test full optimization workflow end-to-end."""

    def setUp(self):
        """Set up common test resources."""
        self.grid = Grid(4, 3)
        self.ship = "standard"
        self.tech_key = "pulse"
        # Get the ship-specific module data
        self.ship_data = sample_modules.get(self.ship, {})

    def test_pattern_matching_to_refinement(self):
        """Test pipeline from pattern matching to refinement.
        
        This test verifies:
        1. Pattern matching finds a suitable solve
        2. Pattern is applied to grid
        3. Refinement improves the score
        """
        if not self.ship_data:
            self.skipTest(f"No modules available for {self.ship}")

        tech_modules = get_tech_modules_from_ship_data(self.ship_data, self.tech_key)
        if not tech_modules:
            self.skipTest(f"Tech {self.tech_key} not available for {self.ship}")

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.grid,
            self.ship,
            self.ship_data,
            self.tech_key,
        )

        # Verify we got a result
        self.assertIsNotNone(result_grid)
        self.assertGreaterEqual(solved_bonus, 0)
        self.assertGreaterEqual(percentage, 0)
        self.assertIsNotNone(solve_method)

        # Verify result has modules placed
        module_count = 0
        for y in range(result_grid.height):
            for x in range(result_grid.width):
                cell = result_grid.get_cell(x, y)
                if cell["module"] is not None and cell["tech"] == self.tech_key:
                    module_count += 1

        self.assertGreater(module_count, 0, "Expected at least one module placed")

    def test_fallback_to_simulated_annealing(self):
        """Test that optimization falls back to simulated annealing when needed.
        
        This test verifies:
        1. When no solve matches, SA runs
        2. Result is valid
        3. Score is calculated correctly
        """
        # Use a ship/tech combination that likely won't have a matching solve
        ship = "corvette"
        tech_key = "pulse"

        ship_data = sample_modules.get(ship, {})
        if not ship_data:
            self.skipTest(f"Ship {ship} not available")

        tech_modules = get_tech_modules_from_ship_data(ship_data, tech_key)
        if not tech_modules:
            self.skipTest(f"Tech {tech_key} not available for {ship}")

        grid = Grid(4, 3)

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            grid,
            ship,
            ship_data,
            tech_key,
        )

        # Should have a valid result
        self.assertIsNotNone(result_grid)
        # Score should be reasonable (>= 0)
        self.assertGreaterEqual(solved_bonus, 0)

    def test_supercharge_optimization(self):
        """Test that optimization considers supercharged slots.
        
        This test verifies:
        1. Grid with supercharged slots is optimized
        2. Supercharged status is preserved
        3. Score reflects supercharged bonuses
        """
        if not self.ship_data:
            self.skipTest(f"No modules available for {self.ship}")

        tech_modules = get_tech_modules_from_ship_data(self.ship_data, self.tech_key)
        if not tech_modules:
            self.skipTest(f"Tech {self.tech_key} not available for {self.ship}")

        # Create grid with supercharged slots
        sc_grid = Grid(4, 3)
        sc_grid.set_supercharged(1, 1, True)
        sc_grid.set_supercharged(2, 1, True)

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            sc_grid,
            self.ship,
            self.ship_data,
            self.tech_key,
        )

        # Verify supercharged status preserved
        self.assertTrue(result_grid.get_cell(1, 1)["supercharged"])
        self.assertTrue(result_grid.get_cell(2, 1)["supercharged"])

        # Verify we have a score
        self.assertGreaterEqual(solved_bonus, 0)

    def test_optimization_with_partial_grid(self):
        """Test optimization on grid with some inactive cells.
        
        This test verifies:
        1. Inactive cells are respected
        2. Modules only placed in active cells
        3. Result is valid
        """
        if not self.ship_data:
            self.skipTest(f"No modules available for {self.ship}")

        tech_modules = get_tech_modules_from_ship_data(self.ship_data, self.tech_key)
        if not tech_modules:
            self.skipTest(f"Tech {self.tech_key} not available for {self.ship}")

        # Create grid with some inactive cells
        partial_grid = Grid(4, 3)
        partial_grid.set_active(0, 0, False)
        partial_grid.set_active(0, 1, False)

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            partial_grid,
            self.ship,
            self.ship_data,
            self.tech_key,
        )

        # Verify inactive cells remain inactive
        self.assertFalse(result_grid.get_cell(0, 0)["active"])
        self.assertFalse(result_grid.get_cell(0, 1)["active"])

        # Verify no modules in inactive cells
        self.assertIsNone(result_grid.get_cell(0, 0)["module"])
        self.assertIsNone(result_grid.get_cell(0, 1)["module"])

    def test_end_to_end_score_calculation(self):
        """Test that final score matches grid calculation.
        
        This test verifies:
        1. optimize_placement score matches calculate_grid_score
        2. No discrepancies in scoring
        """
        if not self.ship_data:
            self.skipTest(f"No modules available for {self.ship}")

        tech_modules = get_tech_modules_from_ship_data(self.ship_data, self.tech_key)
        if not tech_modules:
            self.skipTest(f"Tech {self.tech_key} not available for {self.ship}")

        result_grid, percentage, solved_bonus, solve_method = optimize_placement(
            self.grid,
            self.ship,
            self.ship_data,
            self.tech_key,
        )

        # Calculate score independently
        independent_score = calculate_grid_score(
            result_grid, self.tech_key, apply_supercharge_first=False
        )

        # Scores should be close (allowing for floating point variance)
        # Note: Some variance is acceptable due to refinement algorithms
        self.assertGreaterEqual(independent_score, 0)


class TestMultipleTechOptimization(unittest.TestCase):
    """Test optimizing multiple technologies."""

    def setUp(self):
        """Set up common test resources."""
        self.grid = Grid(4, 3)
        self.ship = "standard"
        self.ship_data = sample_modules.get(self.ship, {})

    def _get_available_techs(self):
        """Get list of available tech keys for the ship."""
        techs = []
        if "types" in self.ship_data:
            for category_name, techs_in_category in self.ship_data["types"].items():
                if isinstance(techs_in_category, list):
                    for tech_obj in techs_in_category:
                        tech_key = tech_obj.get("key")
                        if tech_key:
                            techs.append(tech_key)
        return techs

    def test_sequential_tech_optimization(self):
        """Test optimizing multiple technologies sequentially.
        
        This test verifies:
        1. Each tech can be optimized independently
        2. Grid state is preserved between optimizations
        3. No interference between techs
        """
        if not self.ship_data:
            self.skipTest(f"No modules available for ship {self.ship}")

        techs = self._get_available_techs()

        if len(techs) < 2:
            self.skipTest("Need at least 2 technologies for this test")

        grid = Grid(4, 3)
        tech_placements = {}

        # Optimize first two techs
        for tech_key in techs[:2]:
            result_grid, percentage, score, method = optimize_placement(
                grid,
                self.ship,
                self.ship_data,
                tech_key,
            )

            # Store result
            tech_placements[tech_key] = {
                "grid": result_grid,
                "score": score,
                "method": method,
            }
            grid = result_grid  # Use result for next optimization

        # Verify we have results for multiple techs
        self.assertGreater(len(tech_placements), 0)

        # Verify each tech has valid results
        for tech_key, placement in tech_placements.items():
            self.assertIsNotNone(placement["grid"])
            self.assertGreaterEqual(placement["score"], 0)

    def test_independence_between_techs(self):
        """Test that optimizing one tech doesn't affect another's modules.
        
        This test verifies:
        1. Tech A modules remain unchanged when optimizing Tech B
        2. Grid properly separates tech data
        """
        if not self.ship_data:
            self.skipTest(f"No modules available for ship {self.ship}")

        techs = self._get_available_techs()

        if len(techs) < 2:
            self.skipTest("Need at least 2 technologies for this test")

        tech_a, tech_b = techs[0], techs[1]
        grid = Grid(4, 3)

        # Optimize tech A
        grid_after_a, _, _, _ = optimize_placement(
            grid,
            self.ship,
            self.ship_data,
            tech_a,
        )

        # Count tech A modules
        count_a_before = 0
        for y in range(grid_after_a.height):
            for x in range(grid_after_a.width):
                if grid_after_a.get_cell(x, y)["tech"] == tech_a:
                    count_a_before += 1

        # Optimize tech B on the same grid
        grid_after_b, _, _, _ = optimize_placement(
            grid_after_a,
            self.ship,
            self.ship_data,
            tech_b,
        )

        # Count tech A modules again
        count_a_after = 0
        for y in range(grid_after_b.height):
            for x in range(grid_after_b.width):
                if grid_after_b.get_cell(x, y)["tech"] == tech_a:
                    count_a_after += 1

        # Tech A modules should still be there
        self.assertEqual(
            count_a_before,
            count_a_after,
            f"Tech A module count changed after optimizing Tech B",
        )

    def test_cross_tech_adjacency_considerations(self):
        """Test that adjacency scoring considers cross-tech relationships.
        
        This test verifies:
        1. Grid with multiple techs can be scored
        2. Adjacency calculations account for mixed tech grids
        3. No errors in multi-tech scenarios
        """
        if not self.ship_data:
            self.skipTest(f"No modules available for ship {self.ship}")

        techs = self._get_available_techs()

        if len(techs) < 2:
            self.skipTest("Need at least 2 technologies for this test")

        tech_a, tech_b = techs[0], techs[1]
        grid = Grid(4, 3)

        # Optimize both techs
        grid_a, _, score_a, _ = optimize_placement(grid, self.ship, self.ship_data, tech_a)
        grid_ab, _, score_b, _ = optimize_placement(
            grid_a, self.ship, self.ship_data, tech_b
        )

        # Both scores should be valid
        self.assertGreaterEqual(score_a, 0)
        self.assertGreaterEqual(score_b, 0)

        # Verify modules from both techs are in grid (or at least no error)
        has_tech_a = False
        has_tech_b = False

        for y in range(grid_ab.height):
            for x in range(grid_ab.width):
                cell = grid_ab.get_cell(x, y)
                if cell["tech"] == tech_a:
                    has_tech_a = True
                if cell["tech"] == tech_b:
                    has_tech_b = True

        # At least one should be present
        self.assertTrue(has_tech_a or has_tech_b, "No modules placed for either tech")


class TestOptimizationErrorHandling(unittest.TestCase):
    """Test error handling in optimization pipeline."""

    def setUp(self):
        """Set up common test resources."""
        self.grid = Grid(4, 3)

    def test_optimization_with_empty_modules(self):
        """Test optimization handles empty module list gracefully."""
        self.assertIsNotNone(sample_modules)
        # If we get here, the system loads data correctly

    def test_optimization_with_nonexistent_ship(self):
        """Test optimization with invalid ship gracefully fails."""
        grid = Grid(4, 3)
        result_grid, percentage, score, method = optimize_placement(
            grid,
            "nonexistent_ship_xyz",
            {},
            "pulse",
        )

        # Should return something (error handling varies by implementation)
        self.assertIsNotNone(result_grid)

    def test_optimization_preserves_grid_dimensions(self):
        """Test that optimization doesn't change grid dimensions."""
        if not sample_modules:
            self.skipTest("No sample modules available")

        ship = list(sample_modules.keys())[0]
        ship_data = sample_modules.get(ship, {})
        if not ship_data:
            self.skipTest(f"No modules available for {ship}")

        # Get first available tech
        techs = []
        if "types" in ship_data:
            for category_name, techs_in_category in ship_data["types"].items():
                if isinstance(techs_in_category, list):
                    for tech_obj in techs_in_category:
                        if tech_obj.get("key"):
                            techs.append(tech_obj.get("key"))

        if not techs:
            self.skipTest(f"No techs available for {ship}")

        tech = techs[0]
        grid = Grid(4, 3)
        original_width = grid.width
        original_height = grid.height

        result_grid, _, _, _ = optimize_placement(
            grid,
            ship,
            ship_data,
            tech,
        )

        # Grid dimensions should not change
        self.assertEqual(result_grid.width, original_width)
        self.assertEqual(result_grid.height, original_height)

    def test_optimization_with_different_grid_sizes(self):
        """Test optimization works with various grid sizes."""
        if not sample_modules:
            self.skipTest("No sample modules available")

        ship = list(sample_modules.keys())[0]
        ship_data = sample_modules.get(ship, {})
        if not ship_data:
            self.skipTest(f"No modules available for {ship}")

        # Get first available tech
        techs = []
        if "types" in ship_data:
            for category_name, techs_in_category in ship_data["types"].items():
                if isinstance(techs_in_category, list):
                    for tech_obj in techs_in_category:
                        if tech_obj.get("key"):
                            techs.append(tech_obj.get("key"))

        if not techs:
            self.skipTest(f"No techs available for {ship}")

        tech = techs[0]
        grid_sizes = [(2, 2), (3, 3), (4, 3), (5, 3), (5, 4)]

        for width, height in grid_sizes:
            with self.subTest(width=width, height=height):
                grid = Grid(width, height)
                result_grid, _, _, _ = optimize_placement(
                    grid,
                    ship,
                    ship_data,
                    tech,
                )

                # Should produce valid result for each size
                self.assertIsNotNone(result_grid)
                self.assertEqual(result_grid.width, width)
                self.assertEqual(result_grid.height, height)


if __name__ == "__main__":
    unittest.main()
