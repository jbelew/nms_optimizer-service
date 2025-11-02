import unittest
from src.grid_utils import Grid as PythonGrid
from rust_scorer import calculate_grid_score, Grid, Cell, AdjacencyType, ModuleType


class TestRustIntegration(unittest.TestCase):
    def test_rust_scoring(self):
        # Create a Python Grid object to hold the initial data
        python_grid = PythonGrid(3, 3)
        pulse_module_data = {
            "id": "PULSE",
            "label": "Pulse Engine",
            "tech": "pulse",
            "type": "bonus",
            "bonus": 10.0,
            "adjacency": "greater",
            "sc_eligible": True,
            "image": None,
        }
        python_grid.get_cell(1, 1).update(pulse_module_data)
        python_grid.get_cell(1, 1)["module"] = pulse_module_data["id"]

        # Convert the Python grid data to rust_scorer types
        rust_cells = []
        for y in range(python_grid.height):
            row = []
            for x in range(python_grid.width):
                py_cell = python_grid.get_cell(x, y)

                # Convert string representations of enums to the actual enum types from the rust_scorer module
                adjacency = None
                if py_cell["adjacency"] == "greater":
                    adjacency = AdjacencyType.Greater
                elif py_cell["adjacency"] == "lesser":
                    adjacency = AdjacencyType.Lesser

                module_type = None
                if py_cell["type"] == "bonus":
                    module_type = ModuleType.Bonus
                elif py_cell["type"] == "core":
                    module_type = ModuleType.Core

                # Create a rust_scorer.Cell object
                rust_cell = Cell(
                    py_cell["value"],
                    py_cell["total"],
                    py_cell["adjacency_bonus"],
                    py_cell["bonus"],
                    py_cell["active"],
                    py_cell["supercharged"],
                    py_cell["sc_eligible"],
                    module=py_cell["module"],
                    label=py_cell["label"],
                    module_type=module_type,
                    adjacency=adjacency,
                    tech=py_cell["tech"],
                    image=py_cell["image"],
                )
                row.append(rust_cell)
            rust_cells.append(row)

        # Create the rust_scorer.Grid object
        rust_grid = Grid(
            width=python_grid.width,
            height=python_grid.height,
            cells=rust_cells,
        )

        # Call the Rust scoring function
        score = calculate_grid_score(rust_grid, "pulse", False)

        # Verify the results by accessing the properties of the returned rust_scorer.Grid object
        self.assertAlmostEqual(score, 10.0)
