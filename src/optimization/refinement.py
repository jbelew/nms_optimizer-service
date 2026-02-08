"""
Refinement module for NMS grid module placement optimization.

This module provides refinement strategies for improving module placement scores:
- ML-based refinement: Uses machine learning models to optimize placement
- SA-based refinement: Uses simulated annealing for exploration and optimization
- Permutation refinement: Exhaustive placement optimization for small module sets

The module handles both full-grid and windowed (localized) optimization scenarios,
with support for progress tracking and multi-run strategies.

Key Functions:
- _handle_ml_opportunity(): ML-based window refinement
- _handle_sa_refine_opportunity(): SA/permutation-based window refinement
- simulated_annealing(): Multi-run SA optimization with cooling schedule
- refine_placement(): Exhaustive permutation-based optimization
"""

import random
import math
import gevent
import logging
from copy import deepcopy
from itertools import permutations
from typing import Optional, Tuple, Callable

import rust_scorer

from src.grid_utils import Grid, restore_original_state, apply_localized_grid_changes
from src.modules_utils import get_tech_modules
from src.bonus_calculations import calculate_grid_score
from src.module_placement import place_module, clear_all_modules_of_tech
from .windowing import create_localized_grid, create_localized_grid_ml


def _handle_ml_opportunity(
    grid: Grid,
    modules: dict,
    ship: str,
    tech: str,
    opportunity_x: int,
    opportunity_y: int,
    window_width: int,
    window_height: int,
    progress_callback: Optional[Callable] = None,
    run_id: Optional[str] = None,
    stage: Optional[str] = None,
    send_grid_updates: bool = False,
    tech_modules: Optional[list] = None,
    available_modules: Optional[list[str]] = None,
) -> Tuple[Optional[Grid], float]:
    """
    Applies ML-based refinement within a specified opportunity window.

    Creates a localized grid around the opportunity, runs the ML placement model
    on it, and applies the results back to the full grid while preserving other
    tech modules and restoring original grid state.

    Args:
        grid (Grid): The full grid state (input remains unmodified, changes applied to internal copy).
        modules (dict): Module definitions indexed by ship and tech.
        ship (str): Ship identifier (e.g., "corvette", "freighter").
        tech (str): Technology identifier (e.g., "trails", "photon", "pulse").
        opportunity_x (int): X-coordinate (column) of the window's top-left corner.
        opportunity_y (int): Y-coordinate (row) of the window's top-left corner.
        window_width (int): Width of the opportunity window in cells.
        window_height (int): Height of the opportunity window in cells.
        progress_callback (Optional[Callable], optional): Callback for progress updates.
                                                        Signature: fn(progress_dict).
                                                        Defaults to None.
        run_id (Optional[str], optional): Run identifier for tracking. Defaults to None.
        stage (Optional[str], optional): Stage identifier for logging context. Defaults to None.
        send_grid_updates (bool, optional): Whether to emit intermediate grid updates.
                                           Defaults to False.
        tech_modules (Optional[list], optional): List of module definitions for the tech.
                                                If None, fetched from modules dict.
                                                Defaults to None.
        available_modules (Optional[list[str]], optional): List of module IDs available to player.
                                                         Passed to ML placement function.
                                                         Defaults to None.

    Returns:
        Tuple[Optional[Grid], float]: A tuple containing:
            - best_grid (Grid | None): The refined grid with ML optimizations applied,
                                      or None if ML refinement failed
            - best_score (float): The score of the refined grid, or 0.0 if refinement failed

    Notes:
        - The ML refinement works on a localized window, preserving modules outside the window
        - Original grid state is restored after ML processing to maintain consistency
        - If ML processing fails, returns None with score 0.0
    """
    from src.ml_placement import ml_placement  # Keep import local if possible

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
        model_grid_width=localized_grid_ml.width,  # <<< Use actual localized width
        model_grid_height=localized_grid_ml.height,  # <<< Use actual localized height
        polish_result=True,  # Usually don't polish within the main polish step
        progress_callback=progress_callback,
        run_id=run_id,
        stage=stage,
        send_grid_updates=send_grid_updates,
        original_state_map=original_state_map,
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
    grid: Grid,
    modules: dict,
    ship: str,
    tech: str,
    opportunity_x: int,
    opportunity_y: int,
    window_width: int,
    window_height: int,
    progress_callback: Optional[Callable] = None,
    run_id: Optional[str] = None,
    stage: Optional[str] = None,
    send_grid_updates: bool = False,
    tech_modules: Optional[list] = None,
) -> Tuple[Optional[Grid], float]:
    """
    Applies SA/permutation-based refinement within a specified opportunity window.

    Creates a localized window and applies either simulated annealing (for 6+ modules)
    or exhaustive permutation optimization (for <6 modules). Results are applied back
    to the full grid while preserving other tech modules.

    Args:
        grid (Grid): The full grid state (modified copy is returned on success).
        modules (dict): Module definitions indexed by ship and tech.
        ship (str): Ship identifier (e.g., "corvette", "freighter").
        tech (str): Technology identifier (e.g., "trails", "photon", "pulse").
        opportunity_x (int): X-coordinate (column) of the window's top-left corner.
        opportunity_y (int): Y-coordinate (row) of the window's top-left corner.
        window_width (int): Width of the opportunity window in cells.
        window_height (int): Height of the opportunity window in cells.
        progress_callback (Optional[Callable], optional): Callback for progress updates.
                                                        Signature: fn(progress_dict).
                                                        Defaults to None.
        run_id (Optional[str], optional): Run identifier for tracking. Defaults to None.
        stage (Optional[str], optional): Stage identifier for logging context. Defaults to None.
        send_grid_updates (bool, optional): Whether to emit intermediate grid updates.
                                           Defaults to False.
        tech_modules (Optional[list], optional): List of module definitions for the tech.
                                                If None, fetched from modules dict.
                                                Defaults to None.

    Returns:
        Tuple[Optional[Grid], float]: A tuple containing:
            - refined_grid (Grid | None): The refined grid with changes applied, or None on failure
            - refined_score (float): The score of the refined grid, or -1.0 on failure

    Strategy Selection:
        - For <6 modules: Uses refine_placement() for exhaustive permutation optimization
        - For 6+ modules: Uses simulated_annealing() for efficient exploration

    Notes:
        - The input grid is modified in-place by this function
        - Modules of other tech types outside the window are preserved
        - If refinement fails, returns (grid, -1.0) indicating failure
    """
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
        tech_modules = get_tech_modules(modules, ship, tech)
    num_modules = len(tech_modules) if tech_modules else 0

    # Refine the localized grid (no change in logic here)
    if num_modules < 6:
        logging.info(f"{tech} has less than 7 modules, running refine_placement")
        temp_refined_grid, temp_refined_bonus_local = refine_placement(
            localized_grid,
            ship,
            modules,
            tech,
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
            start_x=start_x,
            start_y=start_y,
            progress_callback=progress_callback,
            run_id=run_id,
            stage=stage,
            send_grid_updates=send_grid_updates,
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
    grid: Grid,
    ship: str,
    modules: dict,
    tech: str,
    tech_modules: Optional[list] = None,
    progress_callback: Optional[Callable] = None,
    run_id: Optional[str] = None,
    stage: Optional[str] = None,
) -> Tuple[Optional[Grid], float]:
    """
    Exhaustively optimizes module placement using all permutations.

    Generates all possible permutations of placement positions and module assignments,
    evaluating each to find the configuration with the highest bonus score. Suitable
    for small module sets (typically <6 modules) where brute-force is feasible.

    Args:
        grid (Grid): The grid to optimize (modified in-place during processing).
        ship (str): Ship identifier (e.g., "corvette", "freighter").
        modules (dict): Module definitions indexed by ship and tech.
        tech (str): Technology identifier (e.g., "trails", "photon", "pulse").
        tech_modules (Optional[list], optional): Pre-fetched list of module definitions.
                                                If None, fetched from modules dict.
                                                Defaults to None.
        progress_callback (Optional[Callable], optional): Callback for progress updates.
                                                        Signature: fn(progress_dict).
                                                        Defaults to None.
        run_id (Optional[str], optional): Run identifier for tracking. Defaults to None.
        stage (Optional[str], optional): Stage identifier for logging context. Defaults to None.

    Returns:
        Tuple[Optional[Grid], float]: A tuple containing:
            - optimal_grid (Grid | None): The best grid found, or None if no valid placement exists
            - highest_bonus (float): The score of the optimal grid, or 0.0 if no valid placement

    Algorithm:
        1. Find all available empty, active cells in the grid
        2. Generate all permutations of selecting module_count positions from available cells
        3. For each permutation, shuffle modules and place them at those positions
        4. Score each configuration and track the best
        5. Report progress at regular intervals

    Notes:
        - Time complexity is O(P(n, k) * k) where n = available cells, k = module count
        - Only practical for small module sets (k < 6) due to permutation explosion
        - The grid passed in will be modified during evaluation but only the best is returned
        - Requires sufficient empty cells for all modules or returns None
    """
    optimal_grid_valid = None
    highest_bonus_valid = -float("inf")

    optimal_grid_fallback = None
    highest_bonus_fallback = -float("inf")

    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech)

    if tech_modules is None or len(tech_modules) == 0:
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

        # Check if the placement is strictly valid (honors sc_eligible)
        is_strictly_valid = True
        for index, (x, y) in enumerate(placement):
            module = shuffled_tech_modules[index]
            cell = grid.get_cell(x, y)
            if cell["supercharged"] and not module.get("sc_eligible", False):
                is_strictly_valid = False
                break

        # We process ALL placements now, tracking valid and fallback separately

        # Clear all modules of the selected technology before placing this permutation
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

        # Calculate the score for the current arrangement
        grid_bonus = calculate_grid_score(grid, tech)

        # Update best valid grid if this placement is valid
        if is_strictly_valid:
            if grid_bonus > highest_bonus_valid:
                highest_bonus_valid = grid_bonus
                optimal_grid_valid = deepcopy(grid)

        # Always update fallback (best overall) - used if no valid placement exists
        if grid_bonus > highest_bonus_fallback:
            highest_bonus_fallback = grid_bonus
            optimal_grid_fallback = deepcopy(grid)

        # --- Progress Reporting ---
        if progress_callback and (iteration_count % 5000 == 0 or iteration_count == total_iterations):
            # Best score to report depends on what we've found so far
            current_best = highest_bonus_valid if highest_bonus_valid > -float("inf") else highest_bonus_fallback

            progress_percent = (iteration_count / total_iterations) * 100 if total_iterations > 0 else 0
            progress_data = {
                "tech": tech,
                "run_id": run_id,
                "stage": stage,
                "progress_percent": progress_percent,
                "best_score": current_best,
                "status": "Optimized with Rust. Obviously.",
            }
            progress_callback(progress_data)
            gevent.sleep(0)

    # Print the total number of iterations
    logging.info(f"refine_placement completed {iteration_count} iterations for ship: '{ship}' -- tech: '{tech}'")

    # Return valid grid if found, otherwise fallback
    if optimal_grid_valid is not None:
        logging.info(f"Found valid placement (honoring sc_eligible) with score {highest_bonus_valid:.4f}")
        return optimal_grid_valid, highest_bonus_valid
    elif optimal_grid_fallback is not None:
        logging.warning(
            f"No strictly valid placement found. Using best fallback (ignoring sc_eligible) with score {highest_bonus_fallback:.4f}"
        )
        return optimal_grid_fallback, highest_bonus_fallback
    else:
        return None, 0.0


def simulated_annealing(
    grid: Grid,
    ship: str,
    modules: dict,
    tech: str,
    full_grid: Grid,
    initial_temperature: float = 7000,
    cooling_rate: float = 0.99,
    stopping_temperature: float = 0.1,
    iterations_per_temp: int = 75,
    initial_swap_probability: float = 0.75,
    final_swap_probability: float = 0.25,
    start_from_current_grid: bool = False,
    max_processing_time: float = 600.0,
    progress_callback: Optional[Callable] = None,
    run_id: Optional[str] = None,
    stage: Optional[str] = None,
    progress_offset: int = 0,
    progress_scale: int = 100,
    send_grid_updates: bool = False,
    start_x: int = 0,
    start_y: int = 0,
    tech_modules: Optional[list] = None,
    max_steps_without_improvement: int = 200,
    reheat_factor: float = 0.6,
    max_reheats: int = 10,
    num_sa_runs: int = 6,
    seed: int = 161616,
) -> Tuple[Grid, float]:
    """
    Multi-run simulated annealing optimization for module placement.

    Performs multiple independent SA runs with a cooling schedule, reheat strategy,
    and adaptive swap probabilities. Delegates to Rust-based SA for computational
    efficiency. Tracks progress across all runs and returns the overall best result.

    Args:
        grid (Grid): The grid to optimize (typically localized/windowed).
        ship (str): Ship identifier (e.g., "corvette", "freighter").
        modules (dict): Module definitions indexed by ship and tech.
        tech (str): Technology identifier (e.g., "trails", "photon", "pulse").
        full_grid (Grid): The complete grid for adjacency calculations and context.
        initial_temperature (float, optional): Starting temperature for SA.
                                              Higher = more exploration. Defaults to 7000.
        cooling_rate (float, optional): Temperature reduction rate per iteration.
                                       Range (0, 1), closer to 1 = slower cooling.
                                       Defaults to 0.99.
        stopping_temperature (float, optional): Temperature threshold to stop cooling.
                                               Defaults to 0.1.
        iterations_per_temp (int, optional): Swaps evaluated at each temperature.
                                            Defaults to 75.
        initial_swap_probability (float, optional): Probability of module swap at start.
                                                   Range (0, 1). Defaults to 0.75.
        final_swap_probability (float, optional): Probability of module swap at end.
                                                 Range (0, 1). Defaults to 0.25.
        start_from_current_grid (bool, optional): If True, use grid as initial state
                                                 (polishing mode). Otherwise random init.
                                                 Defaults to False.
        max_processing_time (float, optional): Maximum time limit in seconds.
                                              Defaults to 600.0.
        progress_callback (Optional[Callable], optional): Callback for progress updates.
                                                        Signature: fn(progress_dict).
                                                        Defaults to None.
        run_id (Optional[str], optional): Run identifier for tracking. Defaults to None.
        stage (Optional[str], optional): Stage identifier for logging context. Defaults to None.
        progress_offset (int, optional): Starting progress percentage. Defaults to 0.
        progress_scale (int, optional): Scale of progress range (usually 100). Defaults to 100.
        send_grid_updates (bool, optional): Whether to emit intermediate grid updates.
                                           Defaults to False.
        start_x (int, optional): X offset for localized grid context. Defaults to 0.
        start_y (int, optional): Y offset for localized grid context. Defaults to 0.
        tech_modules (Optional[list], optional): Pre-fetched module definitions.
                                                If None, fetched from modules dict.
                                                Defaults to None.
        max_steps_without_improvement (int, optional): Steps before reheat trigger.
                                                      Defaults to 200.
        reheat_factor (float, optional): Temperature multiplier for reheats.
                                        Range (0, 1]. Defaults to 0.6.
        max_reheats (int, optional): Maximum reheat cycles per run (4 for polishing,
                                    10 otherwise). Defaults to 10.
        num_sa_runs (int, optional): Number of independent SA runs to perform.
                                    Each run uses a different seed. Defaults to 6.
        seed (int, optional): Base seed for deterministic randomization. Actual seeds
                             are derived from this to ensure reproducibility.
                             Defaults to 161616.

    Returns:
        Tuple[Grid, float]: A tuple containing:
            - best_grid (Grid): The best grid found across all runs
            - best_score (float): The score of the best grid

    Algorithm:
        1. Convert tech_modules to Rust Module objects
        2. Serialize grid to JSON for Rust processing
        3. For each of num_sa_runs:
           a. Generate deterministic seed
           b. Wrap progress callback to adjust percentages
           c. Call Rust SA implementation
           d. Track overall best result
        4. Return grid with overall best score

    Cooling Schedule:
        - Linear temperature reduction: T(n) = initial_T * cooling_rate^n
        - Stops when T reaches stopping_temperature
        - Reheat triggered after max_steps_without_improvement without improvement

    Progress Reporting:
        - Adjusted across runs to reflect overall progress
        - Status messages: "start", "in_progress", "finish"
        - Percentage scaled relative to progress_offset and progress_scale

    Notes:
        - This function delegates actual optimization to Rust via rust_scorer module
        - Deterministic with same seed produces same results
        - Multi-run strategy balances exploration with computational cost
        - Adaptive probabilities transition from exploration to exploitation
        - Polishing mode (start_from_current_grid=True) uses fewer reheats
    """
    if start_from_current_grid:  # Polishing
        max_reheats = 4
    else:
        max_reheats = 10
    if tech_modules is None:
        tech_modules = get_tech_modules(modules, ship, tech)

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

    def get_rust_adjacency(adj_str):
        if not adj_str:
            return rust_scorer.AdjacencyType.NoAdjacency
        if adj_str == "greater":
            return rust_scorer.AdjacencyType.Greater
        if adj_str == "lesser":
            return rust_scorer.AdjacencyType.Lesser
        adj_lower = adj_str.lower()
        if "greater" in adj_lower:
            return rust_scorer.AdjacencyType.Greater
        if "lesser" in adj_lower:
            return rust_scorer.AdjacencyType.Lesser
        return rust_scorer.AdjacencyType.NoAdjacency

    tech_modules_rs = [
        rust_scorer.Module(  # type: ignore
            id=m["id"],
            label=m["label"],
            tech=tech,
            module_type=module_type_map[m["type"]],
            bonus=m["bonus"],
            adjacency=get_rust_adjacency(m["adjacency"]),
            sc_eligible=m["sc_eligible"],
            image=m.get("image"),
        )
        for m in tech_modules
    ]

    grid_json = grid.to_json()

    overall_best_grid_json = ""
    overall_best_score = -float("inf")

    # Create a local random generator for deterministic seed generation
    rng = random.Random(seed)

    for run_idx in range(num_sa_runs):
        # Adjust progress_offset for each run
        current_progress_offset = progress_offset + (run_idx / num_sa_runs) * progress_scale

        # Generate a deterministic seed for the current Rust SA run
        current_run_seed = rng.randint(0, 2**64 - 1)  # Generate a new seed for each run

        # Pass a wrapper progress_callback that adjusts progress_percent
        # Ensure progress_callback is not None before wrapping
        wrapped_progress_callback = None
        if progress_callback:

            def _wrapped_callback(pd):
                # Create a copy to avoid modifying the original dict if it's reused
                pd_copy = pd.copy()

                # Handle grid updates if present
                if "best_grid_json" in pd_copy and send_grid_updates:
                    try:
                        # Deserialize the grid from Rust
                        localized_best_grid = Grid.from_json(pd_copy["best_grid_json"])

                        # Reconstitute with full grid context
                        reconstituted_grid = full_grid.copy()
                        clear_all_modules_of_tech(reconstituted_grid, tech)
                        apply_localized_grid_changes(reconstituted_grid, localized_best_grid, tech, start_x, start_y)

                        # Convert to dict for transmission
                        pd_copy["best_grid"] = reconstituted_grid.to_dict()
                        del pd_copy["best_grid_json"]  # Remove the JSON version
                    except Exception as e:
                        logging.warning(f"Failed to reconstitute grid for progress update: {e}")
                        # Remove the failed grid data
                        if "best_grid_json" in pd_copy:
                            del pd_copy["best_grid_json"]

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

        current_run_best_grid_json, current_run_best_score = rust_scorer.simulated_annealing(  # type: ignore[call-arg,misc,arg-type]
            grid_json,
            tech_modules_rs,
            tech,
            initial_temperature,
            cooling_rate,
            stopping_temperature,
            iterations_per_temp,
            wrapped_progress_callback,
            max_steps_without_improvement,  # type: ignore[arg-type]
            reheat_factor,  # type: ignore[arg-type]
            max_reheats,  # type: ignore[arg-type]
            initial_swap_probability,  # type: ignore[arg-type]
            final_swap_probability,  # type: ignore[arg-type]
            run_idx,  # type: ignore[arg-type]
            num_sa_runs,  # type: ignore[arg-type]
            current_run_seed,  # Pass the seed to the Rust function  # type: ignore[arg-type]
            send_grid_updates,  # type: ignore[arg-type]
        )

        if current_run_best_score > overall_best_score:
            overall_best_score = current_run_best_score
            overall_best_grid_json = current_run_best_grid_json

    best_grid = Grid.from_json(overall_best_grid_json)

    return best_grid, overall_best_score
