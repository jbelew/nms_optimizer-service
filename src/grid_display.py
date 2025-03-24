# grid_display.py
import copy
from grid_utils import Grid

def print_grid(grid: Grid) -> None:
    """Displays the grid with module info, total value, and active state (+/-)."""
    for y, row in enumerate(grid.cells):
        formatted_row = []

        for x, cell in enumerate(row):
            cell_data = copy.deepcopy(
                grid.get_cell(x, y)
            )  # make a copy of the data to ensure we don't modify it
            is_supercharged = cell_data["supercharged"]
            is_shield = cell_data["tech"] == "shield"
            is_infra = cell_data["tech"] == "infra"
            active_state = (
                " +" if cell_data["active"] else " -"
            )  # what to show if there is no module

            formatted_row.append(
                f"\033[93m{active_state if cell_data['module'] is None else cell_data['module']} (T: {cell_data['total']:.2f}) (B: {cell_data['bonus']:.2f}) (A: {cell_data['adjacency_bonus']:.2f})\033[0m"
                if is_supercharged
                else (
                    f"\033[96m{active_state if cell_data['module'] is None else cell_data['module']} (T: {cell_data['total']:.2f}) (B: {cell_data['bonus']:.2f}) (A: {cell_data['adjacency_bonus']:.2f})\033[0m"
                    if is_shield
                    else (
                        f"\033[91m{active_state if cell_data['module'] is None else cell_data['module']} (T: {cell_data['total']:.2f}) (B: {cell_data['bonus']:.2f}) (A: {cell_data['adjacency_bonus']:.2f})\033[0m"
                        if is_infra
                        else f"{active_state if cell_data['module'] is None else cell_data['module']} (T: {cell_data['total']:.2f}) (B: {cell_data['bonus']:.2f}) (A: {cell_data['adjacency_bonus']:.2f})"
                    )
                )
            )
        print(" | ".join(formatted_row))
    print()


def print_grid_compact(grid: Grid) -> None:
    """Displays a compact version of the grid with module info."""
    for y, row in enumerate(grid.cells):
        formatted_row = []

        for x, cell in enumerate(row):
            cell_data = copy.deepcopy(
                grid.get_cell(x, y)
            )  # make a copy of the data to ensure we don't modify it
            is_supercharged = cell_data["supercharged"]
            is_shield = cell_data["tech"] == "shield"
            is_infra = cell_data["tech"] == "infra"
            active_state = (
                " +" if cell_data["active"] else " -"
            )  # what to show if there is no module

            formatted_row.append(
                f"\033[93m{active_state if cell_data['module'] is None else cell_data['module']} \033[0m"
                if is_supercharged
                else (
                    f"\033[96m{active_state if cell_data['module'] is None else cell_data['module']} \033[0m"
                    if is_shield
                    else (
                        f"\033[91m{active_state if cell_data['module'] is None else cell_data['module']} \033[0m"
                        if is_infra
                        else f"{active_state if cell_data['module'] is None else cell_data['module']} "
                    )
                )
            )
        print(" | ".join(formatted_row))
    print()
