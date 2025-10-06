import unittest
from src.grid_utils import Grid

class TestGridUtils(unittest.TestCase):
    """
    Test suite for the Grid class and related utility functions in grid_utils.py.
    """

    def setUp(self):
        """
        Set up a common grid for testing.
        """
        self.width = 4
        self.height = 3
        self.grid = Grid(self.width, self.height)

    def test_grid_initialization(self):
        """
        Test that the Grid is initialized with the correct dimensions and empty cells.
        """
        self.assertEqual(self.grid.width, self.width)
        self.assertEqual(self.grid.height, self.height)
        self.assertEqual(len(self.grid.cells), self.height)
        self.assertEqual(len(self.grid.cells[0]), self.width)
        # Check that a cell has the default 'module' value of None
        self.assertIsNone(self.grid.get_cell(0, 0)['module'])

    def test_to_dict_and_from_dict(self):
        """
        Test that the Grid can be correctly serialized to and deserialized from a dictionary.
        """
        # Modify a cell to test serialization
        self.grid.set_module(1, 1, "TEST_MOD")
        self.grid.set_supercharged(1, 1, True)

        grid_dict = self.grid.to_dict()

        # Check basic structure of the dictionary
        self.assertIn("width", grid_dict)
        self.assertIn("height", grid_dict)
        self.assertIn("cells", grid_dict)
        self.assertEqual(grid_dict['width'], self.width)

        # Create a new grid from the dictionary
        new_grid = Grid.from_dict(grid_dict)

        # Check that the new grid has the same properties
        self.assertEqual(new_grid.width, self.grid.width)
        self.assertEqual(new_grid.height, self.grid.height)
        self.assertEqual(new_grid.get_cell(1, 1)['module'], "TEST_MOD")
        self.assertTrue(new_grid.get_cell(1, 1)['supercharged'])
        self.assertIsNone(new_grid.get_cell(0, 0)['module'])

    def test_grid_copy(self):
        """
        Test that the copy method creates a deep copy of the grid.
        """
        # Place a module in the original grid
        self.grid.set_module(0, 0, "ORIGINAL")

        # Create a copy
        copied_grid = self.grid.copy()

        # Verify the copy has the module
        self.assertEqual(copied_grid.get_cell(0, 0)['module'], "ORIGINAL")

        # Modify the copied grid
        copied_grid.set_module(0, 0, "COPIED")

        # Verify the original grid is unchanged
        self.assertEqual(self.grid.get_cell(0, 0)['module'], "ORIGINAL")
        self.assertEqual(copied_grid.get_cell(0, 0)['module'], "COPIED")


if __name__ == '__main__':
    unittest.main()