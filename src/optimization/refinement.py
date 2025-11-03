# optimization/refinement.py
import random
import math
import gevent
import logging
from copy import deepcopy
from itertools import permutations
from typing import Optional

import rust_scorer

from src.grid_utils import Grid, restore_original_state, apply_localized_grid_changes
from src.modules_utils import get_tech_modules
from src.bonus_calculations import calculate_grid_score
from src.module_placement import place_module, clear_all_modules_of_tech
from .windowing import create_localized_grid, create_localized_grid_ml


def _handle_ml_opportunity(
    grid,
    modules,
    ship,
    tech,
    player_owned_rewards,
    opportunity_x,
    opportunity_y,
    window_width,
    window_height,
    progress_callback=None,
    run_id=None,
    stage=None,
    send_grid_updates=False,
    solve_type: Optional[str] = None,
    tech_modules=None,
    available_modules=None,
):
    """Handles the ML-based refinement within an opportunity window."""
    from src.ml_placement import ml_placement  # Keep import local if possible

    if player_owned_rewards is None:
        player_owned_rewards = []

    logging.info(
        f"Using ML for opportunity refinement at ({opportunity_x}, {opportunity_y}) with window {window_width}x{window_height}"
    )
    if progress_callback:
        progress_callback(
            {
                "tech": tech,
                "run_id": run_id,
                "stage": stage,
                "status": "Optimized with Rust. Obviously.",
                "progress_percent": 0,
            }
        )
        gevent.sleep(0)
    # 1. Create localized grid specifically for ML
    # <<< Pass window dimensions to create_localized_grid_ml >>>
    localized_grid_ml, start_x, start_y, original_state_map = create_localized_grid_ml(
        grid,
        opportunity_x,
        opportunity_y,
        tech,
        window_width,
        window_height,  # <<< Pass dimensions
    )

    # 2. Run ML placement on the localized grid
    #    Make sure player_owned_rewards is passed correctly!
    # <<< Pass localized grid dimensions to ml_placement >>>
    ml_refined_grid, ml_refined_score_local = ml_placement(
        localized_grid_ml,
        ship,
        tech,
        full_grid_original=grid,  # Pass the original full grid
        start_x_original=opportunity_x,  # Pass the original start_x
        start_y_original=opportunity_y,  # Pass the original start_y
        player_owned_rewards=player_owned_rewards,
        model_grid_width=localized_grid_ml.width,  # <<< Use actual localized width
        model_grid_height=localized_grid_ml.height,  # <<< Use actual localized height
        polish_result=True,  # Usually don't polish within the main polish step
        progress_callback=progress_callback,
        run_id=run_id,
        stage=stage,
        send_grid_updates=send_grid_updates,
        original_state_map=original_state_map,
        solve_type=solve_type,
        tech_modules=tech_modules,  # type: ignore
        available_modules=available_modules,
    )

    # 3. Process ML result (logic remains the same)
    if ml_refined_grid is not None:
        # logging.info(f"ML refinement produced a grid. Applying changes...")
        grid_copy = grid.copy()
        clear_all_modules_of_tech(grid_copy, tech)
        apply_localized_grid_changes(grid_copy, ml_refined_grid, tech, start_x, start_y)
        restore_original_state(grid_copy, original_state_map)
        new_score_global = calculate_grid_score(grid_copy, tech)
        # logging.info(f"Score after ML refinement and restoration: {new_score_global:.4f}")
        return grid_copy, new_score_global
    else:
        # Handle ML failure (logic remains the same)
        logging.info("ML refinement failed or returned None. No changes applied.")
        grid_copy = grid.copy()
        restore_original_state(grid_copy, original_state_map)
        original_score = calculate_grid_score(grid_copy, tech)
        logging.info(f"Returning grid with original score after failed ML: {original_score:.4f}")
        return None, 0.0


def _handle_sa_refine_opportunity(
    grid,
    modules,
    ship,
    tech,
    player_owned_rewards,
    opportunity_x,
    opportunity_y,
    window_width,
    window_height,
    progress_callback=None,
    run_id=None,
    stage=None,
    send_grid_updates=False,
    solve_type: Optional[str] = None,
    tech_modules=None,
):
    """Handles the SA/Refine-based refinement within an opportunity window."""
    logging.info(
        f"Using SA/Refine for opportunity refinement at ({opportunity_x}, {opportunity_y}) with window {window_width}x{window_height}"
    )

    # --- *** Modification Start *** ---
    # Create a copy of the grid *before* clearing, to preserve the original state for localization
    grid_copy_for_localization = grid.copy()

    # Clear the target tech from the original grid passed in (or a copy if preferred, but this modifies the copy passed from optimize_placement)
    clear_all_modules_of_tech(grid_copy_for_localization, tech)
    # --- *** Modification End *** ---

    # Create a localized grid (preserves other tech modules)
    # <<< Pass window dimensions to create_localized_grid >>>
    # <<< Use the copy made *before* clearing for localization info >>>
    localized_grid, start_x, start_y = create_localized_grid(
        grid_copy_for_localization,
        opportunity_x,
        opportunity_y,
        tech,
        window_width,
        window_height,  # <<< Pass dimensions
    )
    # print_grid(localized_grid) # Optional debug

    # Get the number of modules for the given tech (no change)
    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards, solve_type=solve_type)
    num_modules = len(tech_modules) if tech_modules else 0

    # Refine the localized grid (no change in logic here)
    if num_modules < 6:
        logging.info(f"{tech} has less than 7 modules, running refine_placement")
        temp_refined_grid, temp_refined_bonus_local = refine_placement(
            localized_grid,
            ship,
            modules,
            tech,
            player_owned_rewards,
            solve_type=solve_type,
            tech_modules=tech_modules,
            progress_callback=progress_callback,
            run_id=run_id,
            stage=stage,
        )
    else:
        logging.info(f"{tech} has 6 or more modules, running simulated_annealing")
        temp_refined_grid, temp_refined_bonus_local = simulated_annealing(
            localized_grid,
            ship,
            modules,
            tech,
            grid,  # full_grid
            player_owned_rewards,
            start_x=start_x,
            start_y=start_y,
            progress_callback=progress_callback,
            run_id=run_id,
            stage=stage,
            send_grid_updates=send_grid_updates,
            solve_type=solve_type,
            tech_modules=tech_modules or [],
        )

    # Process SA/Refine result (logic remains the same)
    if temp_refined_grid is not None:
        # Apply changes back to the main grid copy (grid - which was cleared earlier)
        clear_all_modules_of_tech(grid, tech)
        apply_localized_grid_changes(grid, temp_refined_grid, tech, start_x, start_y)
        new_score_global = calculate_grid_score(grid, tech)
        logging.info(f"Score after SA/Refine refinement: {new_score_global:.4f}")
        # print_grid(grid) # Print the modified grid (grid)
        return grid, new_score_global
    else:
        logging.info("SA/Refine refinement failed. No changes applied.")
        # Return the grid (which was cleared of the tech) and indicate failure
        return grid, -1.0


def refine_placement(
    grid,
    ship,
    modules,
    tech,
    player_owned_rewards=None,
    solve_type: Optional[str] = None,
    tech_modules=None,
    progress_callback=None,
    run_id=None,
    stage=None,
):
    optimal_grid = None
    highest_bonus = 0.0
    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards, solve_type=solve_type)

    if tech_modules is None:
        logging.error(f"No modules found for ship '{ship}' and tech '{tech}'.")
        return None, 0.0

    # Precompute available positions for fast access
    available_positions = [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.get_cell(x, y)["module"] is None and grid.get_cell(x, y)["active"]
    ]

    # Check if there are enough available positions for all modules
    if len(available_positions) < len(tech_modules):
        return None, 0.0

    # Initialize the iteration counter
    iteration_count = 0
    # --- Progress Reporting Setup ---
    try:
        total_iterations = math.perm(len(available_positions), len(tech_modules))
    except (ValueError, ZeroDivisionError):
        total_iterations = 0
    # --- End Progress Reporting Setup ---

    # Generate all permutations of module placements
    for placement in permutations(available_positions, len(tech_modules)):
        # Shuffle the tech_modules list for each permutation
        shuffled_tech_modules = tech_modules[:]  # Create a copy to avoid modifying the original list
        random.shuffle(shuffled_tech_modules)

        # Increment the iteration counter
        iteration_count += 1

        # Clear all modules of the selected technology - MOVED BEFORE PLACEMENT
        clear_all_modules_of_tech(grid, tech)

        # Place all modules in the current permutation
        for index, (x, y) in enumerate(placement):
            module = shuffled_tech_modules[index]
            place_module(
                grid,
                x,
                y,
                module["id"],
                module["label"],
                tech,
                module["type"],
                module["bonus"],
                module["adjacency"],
                module["sc_eligible"],
                module["image"],
            )

        # Calculate the score for the current arrangement - MOVED OUTSIDE THE INNER LOOP
        grid_bonus = calculate_grid_score(grid, tech)

        # Update the best grid if a better score is found - MOVED OUTSIDE THE INNER LOOP
        if grid_bonus > highest_bonus:
            highest_bonus = grid_bonus
            optimal_grid = deepcopy(grid)
            # print(highest_bonus)
            # print_grid(optimal_grid)

        # --- Progress Reporting ---
        if progress_callback and (iteration_count % 5000 == 0 or iteration_count == total_iterations):
            progress_percent = (iteration_count / total_iterations) * 100 if total_iterations > 0 else 0
            progress_data = {
                "tech": tech,
                "run_id": run_id,
                "stage": stage,
                "progress_percent": progress_percent,
                "best_score": highest_bonus,
                "status": "in_progress",
            }
            progress_callback(progress_data)
            gevent.sleep(0)

    # Print the total number of iterations
    logging.info(f"refine_placement completed {iteration_count} iterations for ship: '{ship}' -- tech: '{tech}'")

    return optimal_grid, highest_bonus


def simulated_annealing(
    grid,
    ship,
    modules,
    tech,
    full_grid,
    player_owned_rewards=None,
    initial_temperature=9000,
    cooling_rate=0.99,
    stopping_temperature=0.1,
    iterations_per_temp=75,
    initial_swap_probability=0.5,
    final_swap_probability=0.1,
    start_from_current_grid: bool = False,
    max_processing_time: float = 600.0,
    progress_callback=None,
    run_id=None,
    stage=None,
    progress_offset=0,
    progress_scale=100,
    send_grid_updates=False,
    start_x: int = 0,
    start_y: int = 0,
    solve_type: Optional[str] = None,
    tech_modules: Optional[list] = None,
    max_steps_without_improvement=250,
    reheat_factor=0.6,
    max_reheats=10,
    num_sa_runs: int = 6,
):
    # --- Define max_reheats early ---
    if start_from_current_grid:  # Polishing
        max_reheats = 4
    else:
        max_reheats = 10
    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards, solve_type=solve_type)

    if tech_modules is None:
        logging.error(f"SA: No modules found for ship '{ship}' and tech '{tech}'.")
        return grid.copy(), 0.0

    # Convert tech_modules to Module objects
    module_type_map = {
        "core": rust_scorer.ModuleType.Core,  # type: ignore
        "bonus": rust_scorer.ModuleType.Bonus,  # type: ignore
        "upgrade": rust_scorer.ModuleType.Upgrade,  # type: ignore
        "cosmetic": rust_scorer.ModuleType.Cosmetic,  # type: ignore
        "reactor": rust_scorer.ModuleType.Reactor,  # type: ignore
        "atlantid": rust_scorer.ModuleType.Atlantid,  # type: ignore
    }
    adjacency_map = {
        "greater": rust_scorer.AdjacencyType.Greater,  # type: ignore
        "lesser": rust_scorer.AdjacencyType.Lesser,  # type: ignore
        "none": rust_scorer.AdjacencyType.NoAdjacency,  # type: ignore
    }

    tech_modules_rs = [
        rust_scorer.Module(  # type: ignore
            id=m["id"],
            label=m["label"],
            tech=tech,
            module_type=module_type_map[m["type"]],
            bonus=m["bonus"],
            adjacency=adjacency_map[m["adjacency"]],
            sc_eligible=m["sc_eligible"],
            image=m.get("image"),
        )
        for m in tech_modules
    ]

    grid_json = grid.to_json()

    overall_best_grid_json = ""
    overall_best_score = -float("inf")

    for run_idx in range(num_sa_runs):
        # Adjust progress_offset for each run
        current_progress_offset = progress_offset + (run_idx / num_sa_runs) * progress_scale

        # Pass a wrapper progress_callback that adjusts progress_percent
        # Ensure progress_callback is not None before wrapping
        wrapped_progress_callback = None
        if progress_callback:

            def _wrapped_callback(pd):
                # Create a copy to avoid modifying the original dict if it's reused
                pd_copy = pd.copy()
                # Only adjust progress_percent for 'in_progress' status
                if pd_copy.get("status") == "in_progress" and "progress_percent" in pd_copy:
                    adjusted_progress_percent = current_progress_offset + (pd_copy["progress_percent"] / 100) * (
                        progress_scale / num_sa_runs
                    )
                    pd_copy["progress_percent"] = adjusted_progress_percent
                # For 'start' and 'finish' messages, set progress_percent explicitly
                elif pd_copy.get("status") == "start":
                    pd_copy["progress_percent"] = current_progress_offset
                elif pd_copy.get("status") == "finish":
                    pd_copy["progress_percent"] = current_progress_offset + (
                        progress_scale / num_sa_runs
                    )  # This represents 100% of this run
                progress_callback(pd_copy)  # Pass the modified copy

            wrapped_progress_callback = _wrapped_callback

        current_run_best_grid_json, current_run_best_score = rust_scorer.simulated_annealing(  # type: ignore
            grid_json,
            tech_modules_rs,
            tech,
            initial_temperature,
            cooling_rate,
            stopping_temperature,
            iterations_per_temp,
            wrapped_progress_callback,
            max_steps_without_improvement,
            reheat_factor,
            max_reheats,
            initial_swap_probability,
            final_swap_probability,
        )

        if current_run_best_score > overall_best_score:
            overall_best_score = current_run_best_score
            overall_best_grid_json = current_run_best_grid_json

    best_grid = Grid.from_json(overall_best_grid_json)

    return best_grid, overall_best_score
