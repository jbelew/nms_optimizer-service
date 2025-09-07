# grid_utils.py
import logging
from copy import deepcopy
from .module_placement import place_module, clear_all_modules_of_tech

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [
            [
                {
                    "module": None,
                    "label": None,
                    "value": 0,
                    "type": "",
                    "total": 0.0,
                    "adjacency_bonus": 0.0,
                    "bonus": 0.0,
                    "active": True,
                    "adjacency": False,
                    "tech": None,
                    "supercharged": False,
                    "sc_eligible": False,
                    "image": None,
                }
                for _ in range(width)
            ]
            for _ in range(height)
        ]

    def get_cell(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cells[y][x]
        else:
            raise IndexError("Cell out of bounds")

    def set_cell(self, x, y, value):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["value"] = value
        else:
            raise IndexError("Cell out of bounds")
        
    def copy(self):
        new_grid = Grid(self.width, self.height)  # Pass width and height
        new_grid.cells = [deepcopy(row) for row in self.cells]  # Use deepcopy for nested lists
        return new_grid
        
    def set_label(self, x, y, label):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["label"] = label
        else:
            raise IndexError("Cell out of bounds")

    def set_supercharged(self, x, y, value):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["supercharged"] = value
        else:
            raise IndexError("Cell out of bounds")

    def set_active(self, x, y, value):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["active"] = value
        else:
            raise IndexError("Cell out of bounds")

    def is_supercharged(self, x, y):
        return self.get_cell(x, y)["supercharged"]

    def set_adjacency_bonus(self, x, y, bonus):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["adjacency_bonus"] = bonus
        else:
            raise IndexError("Cell out of bounds")

    def set_bonus(self, x, y, bonus):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["bonus"] = bonus
        else:
            raise IndexError("Cell out of bounds")

    def set_type(self, x, y, type):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["type"] = type
        else:
            raise IndexError("Cell out of bounds")

    def set_value(self, x, y, value):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["value"] = value
        else:
            raise IndexError("Cell out of bounds")

    def set_total(self, x, y, total):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["total"] = total
        else:
            raise IndexError("Cell out of bounds")

    def set_module(self, x, y, module):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["module"] = module
        else:
            raise IndexError("Cell out of bounds")

    def set_adjacency(self, x, y, adjacency):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["adjacency"] = adjacency
        else:
            raise IndexError("Cell out of bounds")

    def set_tech(self, x, y, tech):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["tech"] = tech
        else:
            raise IndexError("Cell out of bounds")

    def set_sc_eligible(self, x, y, sc_eligible):
        """Set whether the cell at (x, y) is eligible for supercharging."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["sc_eligible"] = sc_eligible
        else:
            raise IndexError("Cell out of bounds")

    def set_image(self, x, y, image):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x]["image"] = image
        else:
            raise IndexError("Cell out of bounds")

    def to_dict(self):
        """Convert the grid into a JSON-serializable dictionary."""
        return {"width": self.width, "height": self.height, "cells": self.cells}

    @classmethod
    def from_dict(cls, data: dict) -> "Grid":
        """Create a Grid instance from a dictionary."""
        grid = cls(data["width"], data["height"])
        for y, row in enumerate(data["cells"]):
            for x, cell_data in enumerate(row):
                cell = grid.get_cell(x, y)
                cell.update(
                    {
                        "module": cell_data["module"],
                        "label": cell_data["label"],
                        "value": cell_data["value"],
                        "type": cell_data["type"],
                        "total": cell_data["total"],
                        "active": cell_data.get("active", False),
                        "adjacency_bonus": cell_data["adjacency_bonus"],
                        "bonus": cell_data["bonus"],
                        "adjacency": cell_data["adjacency"],
                        "tech": cell_data["tech"],
                        "supercharged": cell_data.get("supercharged", False),
                        "sc_eligible": cell_data.get("sc_eligible", False),
                        "image": cell_data["image"],
                    }
                )
        return grid

    def __str__(self):
        """Generate a string representation of the grid."""
        return "\n".join(
            " ".join(
                "."
                if cell["value"] == 0
                else str(cell["total"])
                for cell in row
            )
            for row in self.cells
        )

def restore_original_state(grid, original_state_map):
    """
    Restores the original state of cells in the main grid that were temporarily
    modified (other tech modules removed and marked inactive) during localization.

    Args:
        grid (Grid): The main grid to restore.
        original_state_map (dict): The dictionary mapping main grid coordinates (x, y)
                                   to their original cell data, as returned by
                                   create_localized_grid_ml.
    """
    if not original_state_map:
        return  # Nothing to restore

    logging.info(f"Restoring original state for {len(original_state_map)} cells.")
    for (x, y), original_cell_data in original_state_map.items():
        if 0 <= x < grid.width and 0 <= y < grid.height:
            grid.cells[y][x].update(deepcopy(original_cell_data))
        else:
            logging.warning(
                f"Coordinate ({x},{y}) from original_state_map is out of bounds for the main grid."
            )

def apply_localized_grid_changes(main_grid, localized_grid, tech, start_x, start_y):
    """
    Applies changes from a localized grid back to the main grid.
    Only modules of the specified tech are transferred.
    """
    clear_all_modules_of_tech(main_grid, tech)

    for y_local in range(localized_grid.height):
        for x_local in range(localized_grid.width):
            local_cell = localized_grid.get_cell(x_local, y_local)
            if local_cell['module'] is not None and local_cell['tech'] == tech:
                main_x = start_x + x_local
                main_y = start_y + y_local
                if (
                    0 <= main_x < main_grid.width
                    and 0 <= main_y < main_grid.height
                    and main_grid.get_cell(main_x, main_y)['active']
                ):
                    place_module(
                        main_grid,
                        main_x,
                        main_y,
                        local_cell['module'],
                        local_cell['label'],
                        local_cell['tech'],
                        local_cell['type'],
                        local_cell['bonus'],
                        local_cell['adjacency'],
                        local_cell['sc_eligible'],
                        local_cell['image'],
                    )
                else:
                    logging.warning(
                        f"Attempted to place module {local_cell['label']} from localized grid at ({main_x},{main_y}) on main grid, but cell is inactive or out of bounds."
                    )

__all__ = ["Grid", "restore_original_state", "apply_localized_grid_changes"]