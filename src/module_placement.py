# module_placement.py
from grid_utils import Grid

def place_module(
    grid: Grid,
    x: int,
    y: int,
    module: str | None,
    label: str | None,
    tech: str | None,
    type: str,
    bonus: float,
    adjacency: bool,
    sc_eligible: bool,
    image: str | None,
) -> None:
    """Places a module in the grid at the specified position."""
    grid.set_module(x, y, module)
    grid.set_label(x, y, label)
    grid.set_tech(x, y, tech)
    grid.set_type(x, y, type)
    grid.set_bonus(x, y, bonus)
    grid.set_adjacency(x, y, adjacency)
    grid.set_sc_eligible(x, y, sc_eligible)
    grid.set_image(x, y, image)
