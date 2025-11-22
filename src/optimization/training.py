# optimization/training.py
import logging
import math
from itertools import permutations
from typing import Optional

from src.bonus_calculations import calculate_grid_score
from src.grid_utils import Grid, apply_localized_grid_changes
from src.module_placement import clear_all_modules_of_tech, place_module
from src.modules_utils import get_tech_modules
from src.optimization.windowing import find_supercharged_opportunities, create_localized_grid


def refine_placement_for_training(
    grid: Grid,
    ship: str,
    modules: dict,
    tech: str,
    progress_callback=None,
    run_id=None,
    send_grid_updates=False,
    available_modules: Optional[list[dict]] = None,
) -> tuple[Grid, float]:
    """
    Refines module placement for training purposes, focusing on a single tech.
    This function is a simplified version of optimize_placement, designed to
    generate training data by exploring various module arrangements within
    supercharged opportunity windows.

    Args:
        grid (Grid): The initial grid state.
        ship (str): The ship type key.
        modules (dict): The main modules dictionary.
        tech (str): The technology key.
        progress_callback (callable, optional): Callback for progress updates. Defaults to None.
        run_id (str, optional): Identifier for the current run. Defaults to None.
        send_grid_updates (bool): Whether to send grid updates via callback. Defaults to False.
        available_modules (list[dict], optional): List of modules available for placement.
                                                  If None, all modules for the tech are considered.

    Returns:
        tuple[Grid, float]: The optimized grid and its highest bonus score.
    """
    tech_modules = get_tech_modules(
        modules,
        ship,
        tech,
        available_modules=available_modules,
    )

    if not tech_modules:
        logging.warning(f"No modules retrieved for ship '{ship}', tech '{tech}'. Returning original grid.")
        return grid.copy(), 0.0

    # Initialize with a cleared grid for the target tech
    working_grid = grid.copy()
    clear_all_modules_of_tech(working_grid, tech)

    # Find supercharged opportunities
    opportunity_result = find_supercharged_opportunities(
        working_grid,
        modules,
        ship,
        tech,
        tech_modules=tech_modules,
    )

    optimal_grid = working_grid.copy()
    highest_bonus = 0.0

    if opportunity_result:
        opp_x, opp_y, opp_w, opp_h = opportunity_result
        logging.debug(f"Found supercharged opportunity window: {opp_w}x{opp_h} at ({opp_x}, {opp_y})")

        # Create a localized grid for the window
        localized_grid, window_offset_x, window_offset_y = create_localized_grid(
            working_grid,
            opp_x,
            opp_y,
            tech,
            opp_w,
            opp_h,
        )

        # Generate all permutations of modules that fit the window
        # Filter tech_modules to only include those that are sc_eligible if the window is supercharged
        sc_eligible_modules_in_window = [
            m for m in tech_modules if m.get("sc_eligible", False) and m.get("type") != "core"
        ]
        core_modules = [m for m in tech_modules if m.get("type") == "core"]

        # For simplicity in training, we'll only consider permutations of bonus/upgrade modules
        # and place core modules separately if they exist.
        modules_for_permutation = sc_eligible_modules_in_window

        if not modules_for_permutation and not core_modules:
            logging.warning("No eligible modules for permutation in supercharged window.")
            return grid.copy(), 0.0

        # Determine available slots in the localized grid
        available_slots = []
        for y in range(localized_grid.height):
            for x in range(localized_grid.width):
                if localized_grid.get_cell(x, y)["active"] and localized_grid.get_cell(x, y)["module"] is None:
                    available_slots.append((x, y))

        if not available_slots:
            logging.warning("No available slots in the localized grid for permutation.")
            return grid.copy(), 0.0

        # Limit permutations to avoid excessive computation
        max_permutations = 10000  # Adjust as needed
        if math.factorial(len(modules_for_permutation)) > max_permutations:
            logging.warning(
                f"Too many permutations ({math.factorial(len(modules_for_permutation))}) for {tech}. Skipping."
            )
            return grid.copy(), 0.0

        # Try all permutations of placing modules into available slots
        for p_modules in permutations(modules_for_permutation, min(len(modules_for_permutation), len(available_slots))):
            temp_localized_grid = localized_grid.copy()
            temp_tech_modules_on_grid = []

            # Place core modules first if any
            core_placed = False
            for core_mod in core_modules:
                for y_slot, x_slot in available_slots:
                    if temp_localized_grid.get_cell(x_slot, y_slot)["supercharged"]:
                        place_module(
                            temp_localized_grid,
                            x_slot,
                            y_slot,
                            core_mod["id"],
                            core_mod["label"],
                            core_mod["tech"],
                            core_mod["type"],
                            core_mod["bonus"],
                            core_mod["adjacency"],
                            core_mod["sc_eligible"],
                            core_mod["image"],
                        )
                        temp_tech_modules_on_grid.append(core_mod)
                        core_placed = True
                        break  # Place only one core module in a supercharged slot
                if core_placed:
                    break

            # Place other modules from the permutation
            slot_idx = 0
            for mod_data in p_modules:
                while slot_idx < len(available_slots):
                    x_slot, y_slot = available_slots[slot_idx]
                    if temp_localized_grid.get_cell(x_slot, y_slot)["module"] is None:
                        place_module(
                            temp_localized_grid,
                            x_slot,
                            y_slot,
                            mod_data["id"],
                            mod_data["label"],
                            mod_data["tech"],
                            mod_data["type"],
                            mod_data["bonus"],
                            mod_data["adjacency"],
                            mod_data["sc_eligible"],
                            mod_data["image"],
                        )
                        temp_tech_modules_on_grid.append(mod_data)
                        slot_idx += 1
                        break
                    slot_idx += 1

            current_bonus = calculate_grid_score(temp_localized_grid, tech, apply_supercharge_first=False)

            if current_bonus > highest_bonus:
                highest_bonus = current_bonus
                optimal_grid = temp_localized_grid.copy()

        # Apply the best localized grid back to the main grid
        apply_localized_grid_changes(
            grid,
            optimal_grid,
            tech,
            window_offset_x,
            window_offset_y,
        )

        final_score = calculate_grid_score(optimal_grid, tech, apply_supercharge_first=False)
        if abs(final_score - highest_bonus) > 1e-6:
            logging.warning(
                f"Final score ({final_score:.4f}) differs from tracked best ({highest_bonus:.4f}). Using final score."
            )
            highest_bonus = final_score

    # --- Handle No Valid Placement Found ---
    elif len(tech_modules) > 0:  # Check if modules existed but no solution found
        logging.warning(f"No optimal grid found for {tech}. Returning cleared grid.")
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)
        highest_bonus = 0.0  # Score is 0 for a cleared grid
    else:  # No modules to place initially
        logging.info(f"No modules to place for {tech}. Returning cleared grid.")
        optimal_grid = grid.copy()
        clear_all_modules_of_tech(optimal_grid, tech)
        highest_bonus = 0.0
    # --- End Handling ---

    return optimal_grid, highest_bonus
