# optimization/core.py
import logging

from src.grid_utils import Grid
from src.modules_utils import get_tech_modules
from src.grid_display import print_grid_compact
from src.bonus_calculations import calculate_grid_score

from src.module_placement import clear_all_modules_of_tech
from .refinement import simulated_annealing, _handle_ml_opportunity, _handle_sa_refine_opportunity
from src.data_loader import get_solve_map
from src.solve_map_utils import filter_solves
from src.pattern_matching import (
    apply_pattern_to_grid,
    get_all_unique_pattern_variations,
    _extract_pattern_from_grid,
)
from .helpers import (
    determine_window_dimensions,
    place_all_modules_in_empty_slots,
)
from .windowing import (
    create_localized_grid,
    find_supercharged_opportunities,
    calculate_window_score,
    _scan_grid_with_window,
)


def _prepare_optimization_run(grid, modules, ship, tech, available_modules):
    """
    Handles the initial setup for the optimization process, including fetching
    modules and performing pre-checks on the grid.

    Returns:
        tuple: (full_tech_modules_list, tech_modules)
    Raises:
        ValueError: If no empty, active slots are available.
    """
    full_tech_modules_list = get_tech_modules(modules, ship, tech, available_modules=None)
    tech_modules = get_tech_modules(modules, ship, tech, available_modules=available_modules)

    if not tech_modules:
        logging.warning(f"No modules retrieved for ship '{ship}', tech '{tech}'. Cannot proceed with optimization.")
        cleared_grid_on_fail = grid.copy()
        clear_all_modules_of_tech(cleared_grid_on_fail, tech)
        return cleared_grid_on_fail, 0.0, 0.0, "Module Definition Error"

    has_empty_active_slots = any(
        grid.get_cell(x, y)["active"] and grid.get_cell(x, y)["module"] is None
        for y in range(grid.height)
        for x in range(grid.width)
    )

    if not has_empty_active_slots:
        raise ValueError(f"No empty, active slots available on the grid for ship: '{ship}' -- tech: '{tech}'.")

    return full_tech_modules_list, tech_modules


def optimize_placement(
    grid,
    ship,
    modules,
    tech,
    forced=False,
    progress_callback=None,
    run_id=None,
    send_grid_updates=False,
    available_modules=None,
):
    """
    Optimizes the placement of modules in a grid for a specific ship and technology.
    Uses pre-defined solve maps first, then refines based on supercharged opportunities,
    prioritizing the pattern's location if its opportunity score is high and it contains
    an available supercharged slot.
    Includes an experimental path using ML placement with SA fallback.

    Args:
        grid (Grid): The initial grid state.
        ship (str): The ship type key.
        modules (dict): The main modules dictionary.
        tech (str): The technology key.
        forced (bool): If True and no pattern fits a solve map, forces SA.
                       If False, returns "Pattern No Fit" to allow UI intervention.

    Returns:
        tuple: (best_grid, percentage, best_bonus, solve_method)
               - best_grid (Grid): The optimized grid.
               - percentage (float): The percentage of the official solve score achieved.
               - best_bonus (float): The actual score achieved by the best grid.
               - solve_method (str): The name of the method used to generate the final grid.
    Raises:
        ValueError: If no empty, active slots are available or if critical steps fail.
    """
    logging.info(f"Attempting solve for ship: '{ship}' -- tech: '{tech}'")
    logging.debug(f"send_grid_updates: {send_grid_updates}")

    prep_result = _prepare_optimization_run(grid, modules, ship, tech, available_modules)
    if len(prep_result) == 4:
        return prep_result
    full_tech_modules_list, tech_modules = prep_result

    # Initialize variables for tracking best results
    solved_grid = grid.copy()  # Grid state after initial pattern/SA fallback
    solved_bonus = -float("inf")
    best_pattern_grid = grid.copy()  # Best grid found using pattern matching
    highest_pattern_bonus = -float("inf")
    best_pattern_adjacency_score = 0
    best_pattern_start_x = -1
    best_pattern_start_y = -1
    best_pattern_width = -1
    best_pattern_height = -1
    solve_score = 0
    solve_method = "Unknown"  # <<< Initialize solve_method >>>

    # --- Load Solves On-Demand ---
    all_solves_for_ship = get_solve_map(ship)
    # Create the structure that filter_solves expects
    solves_for_filtering = {ship: all_solves_for_ship} if all_solves_for_ship else {}
    filtered_solves = filter_solves(
        solves_for_filtering,
        ship,
        modules,
        tech,
        available_modules=available_modules,
    )
    # --- End On-Demand Loading ---

    # --- Initial Placement Strategy ---
    if ship not in filtered_solves or (ship in filtered_solves and tech not in filtered_solves[ship]):
        # --- Special Case: No Solve Available ---
        logging.info(
            f"No solve found for ship: '{ship}' -- tech: '{tech}'. Placing modules in empty slots."
        )  # <<< KEEP: Important outcome >>>
        solve_method = "Initial Placement (No Solve)"  # <<< Set method >>>
        # Assuming place_all_modules_in_empty_slots is defined elsewhere
        solved_grid = place_all_modules_in_empty_slots(
            grid,
            modules,
            ship,
            tech,
            tech_modules=tech_modules,
        )
        solved_bonus = calculate_grid_score(solved_grid, tech, apply_supercharge_first=False)
        percentage = 100.0 if solved_bonus > 1e-9 else 0.0
        # <<< KEEP: Final result for this path >>>
        logging.info(
            f"Final Score (No Solve Map): {solved_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) using method '{solve_method}' for ship: '{ship}' -- tech: '{tech}'"
        )
        # print_grid_compact(solved_grid) # Optional: Can be noisy for many calls
        return solved_grid, round(percentage, 4), solved_bonus, solve_method
    else:
        # --- Partial Module Set Path ---
        # If the number of available modules for this tech is less than the full set,
        # skip pattern matching and go straight to a windowed SA solve.
        # --- Partial Module Set Path ---
        is_partial_set = (
            available_modules is not None
            and full_tech_modules_list is not None
            and len(tech_modules) < len(full_tech_modules_list)
        )

        # Special case for 'pulse' tech: if only 'PC' is missing, it's not a partial set
        if is_partial_set and tech == "pulse" and full_tech_modules_list:
            full_tech_module_ids = {m["id"] for m in full_tech_modules_list}
            tech_module_ids = {m["id"] for m in tech_modules}
            missing_modules = full_tech_module_ids - tech_module_ids
            if missing_modules == {"PC"}:
                logging.info(
                    "Pulse tech with only 'PC' module missing is considered a full set. Proceeding with normal optimization."
                )
                is_partial_set = False

        if is_partial_set and tech == "trails":
            logging.info("Trails are always considered a full set. Proceeding with normal optimization.")
            is_partial_set = False

        if is_partial_set:
            if full_tech_modules_list:
                logging.info(
                    f"Partial module set ({len(tech_modules)}/{len(full_tech_modules_list)}) for {ship}/{tech}. Skipping patterns, running windowed SA."
                )
            else:
                # This path should not be taken given the is_partial_set check, but it satisfies pyright
                logging.info(
                    f"Partial module set ({len(tech_modules)}/Unknown) for {ship}/{tech}. Skipping patterns, running windowed SA."
                )

            # We need solve_score for percentage calculation. Let's get it now.
            solve_score = 0
            all_solves_for_ship = get_solve_map(ship)
            if all_solves_for_ship:
                solves_for_filtering = {ship: all_solves_for_ship}
                # Filter with full module list to find the official solve score
                temp_filtered_solves = filter_solves(
                    solves_for_filtering,
                    ship,
                    modules,
                    tech,
                )
                if ship in temp_filtered_solves and tech in temp_filtered_solves[ship]:
                    solve_score = temp_filtered_solves[ship][tech].get("score", 0)

            grid_for_sa = grid.copy()
            clear_all_modules_of_tech(grid_for_sa, tech)

            # Find the best window for the available modules.
            opportunity_result = find_supercharged_opportunities(
                grid_for_sa,
                modules,
                ship,
                tech,
                tech_modules=tech_modules,
            )

            solved_grid = None
            solved_bonus = -float("inf")
            solve_method = "Unknown"

            if opportunity_result:
                opp_x, opp_y, opp_w, opp_h = opportunity_result
                logging.info(
                    f"Found supercharged opportunity window for partial solve: {opp_w}x{opp_h} at ({opp_x}, {opp_y})"
                )
                solved_grid, solved_bonus = _handle_sa_refine_opportunity(
                    grid_for_sa,
                    modules,
                    ship,
                    tech,
                    opp_x,
                    opp_y,
                    opp_w,
                    opp_h,
                    progress_callback=progress_callback,
                    run_id=run_id,
                    stage="partial_set_sa",
                    send_grid_updates=send_grid_updates,
                    tech_modules=tech_modules,
                )
                solve_method = "Partial Set SA"

                # Recalculate score just in case to ensure accuracy
                final_check_score = calculate_grid_score(solved_grid, tech, apply_supercharge_first=False)
                if abs(final_check_score - solved_bonus) > 1e-6:
                    logging.warning(
                        f"Final check score {final_check_score:.4f} differs from tracked solved_bonus {solved_bonus:.4f}. Using check score."
                    )
                    solved_bonus = final_check_score

                if solve_score > 1e-9:
                    percentage = (solved_bonus / solve_score) * 100
                else:
                    percentage = 100.0 if solved_bonus > 1e-9 else 0.0

                logging.info(
                    f"SUCCESS -- Final Score (Partial Set): {solved_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) using method '{solve_method}' for ship: '{ship}' -- tech: '{tech}'"
                )
                print_grid_compact(solved_grid)
                return solved_grid, round(percentage, 2), solved_bonus, solve_method
            else:  # No supercharged opportunity window found.
                logging.warning("No supercharged opportunity window found. Generating pattern from SA solve.")
                num_modules = len(tech_modules)
                w, h = determine_window_dimensions(num_modules, tech, ship)

                # First, perform an initial SA to get a base arrangement
                solved_grid_sa = None
                solved_bonus_sa = -float("inf")
                sa_method_for_pattern_gen = "Unknown"

                best_score_scan, best_pos_scan = _scan_grid_with_window(
                    grid_for_sa.copy(),
                    w,
                    h,
                    num_modules,
                    tech,
                    require_supercharge=False,
                )

                if best_pos_scan:
                    opp_x_scan, opp_y_scan = best_pos_scan
                    logging.info(
                        f"Found best available window via scan: {w}x{h} at ({opp_x_scan}, {opp_y_scan}) with score {best_score_scan:.4f} for initial SA."
                    )
                    solved_grid_sa, solved_bonus_sa = _handle_sa_refine_opportunity(
                        grid_for_sa,
                        modules,
                        ship,
                        tech,
                        opp_x_scan,
                        opp_y_scan,
                        w,
                        h,
                        progress_callback=progress_callback,
                        run_id=run_id,
                        stage="partial_set_sa_pattern_gen_scanned",
                        send_grid_updates=send_grid_updates,
                        tech_modules=tech_modules,
                    )
                    sa_method_for_pattern_gen = "Windowed SA (Scanned Fit)"

                else:
                    if not forced:
                        logging.info(
                            f"Partial module set for {ship}/{tech}, but no suitable window found. Returning 'Pattern No Fit'. UI can prompt to force SA."
                        )
                        return None, 0.0, 0.0, "Pattern No Fit"
                    else:
                        logging.warning(
                            "Could not find any suitable window for partial module set. FORCING full Simulated Annealing for pattern generation."
                        )
                        solved_grid_sa, solved_bonus_sa = simulated_annealing(
                            grid_for_sa,
                            ship,
                            modules,
                            tech,
                            grid,  # full_grid
                            progress_callback=progress_callback,
                            run_id=run_id,
                            stage="partial_set_sa_pattern_gen_full",
                            send_grid_updates=send_grid_updates,
                            tech_modules=tech_modules,
                        )
                        sa_method_for_pattern_gen = "Full SA"

                if solved_grid_sa is None:
                    logging.error(
                        f"Initial SA for pattern generation failed for {ship}/{tech}. Returning 'Partial SA Failed'."
                    )
                    cleared_grid_on_fail = grid.copy()
                    clear_all_modules_of_tech(cleared_grid_on_fail, tech)
                    return cleared_grid_on_fail, 0.0, 0.0, "Partial SA Failed"

                logging.info(
                    f"Initial SA for pattern generation ({sa_method_for_pattern_gen}) completed with score: {solved_bonus_sa:.4f}"
                )

                # Extract pattern from the SA-generated grid
                original_pattern_from_sa = _extract_pattern_from_grid(solved_grid_sa, tech)
                if not original_pattern_from_sa:
                    logging.error(
                        f"Could not extract pattern from SA-generated grid for {ship}/{tech}. Returning 'Partial SA Failed'."
                    )
                    cleared_grid_on_fail = grid.copy()
                    clear_all_modules_of_tech(cleared_grid_on_fail, tech)
                    return cleared_grid_on_fail, 0.0, 0.0, "Partial SA Failed"

                # Apply pattern matching with rotations
                patterns_to_try_from_sa = get_all_unique_pattern_variations(original_pattern_from_sa)
                grid_dict = grid.to_dict()  # Use the original grid for placement attempts

                highest_pattern_bonus = -float("inf")
                best_pattern_grid = None
                best_pattern_adjacency_score = 0
                best_pattern_start_x = -1
                best_pattern_start_y = -1
                best_pattern_width = -1
                best_pattern_height = -1

                for pattern_from_sa in patterns_to_try_from_sa:
                    x_coords = [coord[0] for coord in pattern_from_sa.keys()]
                    y_coords = [coord[1] for coord in pattern_from_sa.keys()]
                    if not x_coords or not y_coords:
                        continue
                    pattern_width_current = max(x_coords) + 1
                    pattern_height_current = max(y_coords) + 1

                    for start_x_pattern in range(grid.width - pattern_width_current + 1):
                        for start_y_pattern in range(grid.height - pattern_height_current + 1):
                            temp_grid_pattern = Grid.from_dict(grid_dict)
                            temp_result_grid, adjacency_score = apply_pattern_to_grid(
                                temp_grid_pattern,
                                pattern_from_sa,
                                modules,
                                tech,
                                start_x_pattern,
                                start_y_pattern,
                                ship,
                                tech_modules=tech_modules,
                            )
                            if temp_result_grid is not None:
                                current_pattern_bonus = calculate_grid_score(
                                    temp_result_grid, tech, apply_supercharge_first=False
                                )
                                if current_pattern_bonus > highest_pattern_bonus:
                                    highest_pattern_bonus = current_pattern_bonus
                                    best_pattern_grid = temp_result_grid.copy()
                                    best_pattern_adjacency_score = adjacency_score
                                    best_pattern_start_x = start_x_pattern
                                    best_pattern_start_y = start_y_pattern
                                    best_pattern_width = pattern_width_current
                                    best_pattern_height = pattern_height_current
                                elif (
                                    current_pattern_bonus == highest_pattern_bonus
                                    and adjacency_score > best_pattern_adjacency_score
                                ):
                                    best_pattern_grid = temp_result_grid.copy()
                                    best_pattern_adjacency_score = adjacency_score
                                    best_pattern_start_x = start_x_pattern
                                    best_pattern_start_y = start_y_pattern
                                    best_pattern_width = pattern_width_current
                                    best_pattern_height = pattern_height_current

                if best_pattern_grid is not None:
                    solved_grid = best_pattern_grid
                    solved_bonus = highest_pattern_bonus
                    solve_method = "Partial Set SA Pattern Match"
                    logging.info(
                        f"Best SA-generated pattern score: {solved_bonus:.4f} (Adjacency: {best_pattern_adjacency_score:.2f}) found at ({best_pattern_start_x},{best_pattern_start_y}) with size {best_pattern_width}x{best_pattern_height} for ship: '{ship}' -- tech: '{tech}'."
                    )
                else:
                    if not forced:
                        logging.info(
                            f"SA-generated pattern matching failed for {ship}/{tech}. Returning 'Pattern No Fit'. UI can prompt to force SA."
                        )
                        return None, 0.0, 0.0, "Pattern No Fit"
                    else:
                        logging.warning(
                            f"SA-generated pattern matching failed for {ship}/{tech}. FORCING full Simulated Annealing."
                        )
                        # Fallback to full SA if pattern matching fails even with forced=True
                        # This is similar to the original "no pattern fits, force SA" logic
                        solved_grid, solved_bonus = simulated_annealing(
                            grid_for_sa,
                            ship,
                            modules,
                            tech,
                            grid,  # full_grid
                            progress_callback=progress_callback,
                            run_id=run_id,
                            stage="final_fallback_sa_no_pattern_fit",
                            send_grid_updates=send_grid_updates,
                            tech_modules=tech_modules,
                        )
                        if solved_grid is None:
                            raise ValueError(
                                f"Fallback simulated_annealing failed after SA-generated pattern matching failed on {ship}/{tech}."
                            )
                        logging.info(f"Forced fallback SA score (no SA-generated pattern fit): {solved_bonus:.4f}")
                        solve_method = "Forced Initial SA (No SA-generated Pattern Fit)"

            if solve_score > 1e-9:
                percentage = (solved_bonus / solve_score) * 100
            else:
                percentage = 100.0 if solved_bonus > 1e-9 else 0.0

            logging.info(
                f"SUCCESS -- Final Score (Partial Set): {solved_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) using method '{solve_method}' for ship: '{ship}' -- tech: '{tech}'"
            )
            if solved_grid:
                print_grid_compact(solved_grid)
            return solved_grid, round(percentage, 2), solved_bonus, solve_method

        # --- Case 2: Solve Map Exists ---
        solve_data = filtered_solves[ship][tech]
        original_pattern = solve_data.get("map")
        solve_score = solve_data.get("score", 0.0)

        # print(f"INFO -- Found solve map for {ship}/{tech}. Score: {solve_score:.4f}. Attempting pattern matching.") # <<< KEEP: Important outcome >>>
        # Assuming get_all_unique_pattern_variations is defined elsewhere
        patterns_to_try = get_all_unique_pattern_variations(original_pattern)
        grid_dict = grid.to_dict()

        for pattern in patterns_to_try:
            x_coords = [coord[0] for coord in pattern.keys()]
            y_coords = [coord[1] for coord in pattern.keys()]
            if not x_coords or not y_coords:
                continue
            pattern_width = max(x_coords) + 1
            pattern_height = max(y_coords) + 1

            for start_x in range(grid.width - pattern_width + 1):
                for start_y in range(grid.height - pattern_height + 1):
                    temp_grid_pattern = Grid.from_dict(grid_dict)
                    # Assuming apply_pattern_to_grid is defined elsewhere
                    temp_result_grid, adjacency_score = apply_pattern_to_grid(
                        temp_grid_pattern,
                        pattern,
                        modules,
                        tech,
                        start_x,
                        start_y,
                        ship,
                        tech_modules=tech_modules,
                    )
                    if temp_result_grid is not None:
                        current_pattern_bonus = calculate_grid_score(
                            temp_result_grid, tech, apply_supercharge_first=False
                        )
                        if current_pattern_bonus > highest_pattern_bonus:
                            highest_pattern_bonus = current_pattern_bonus
                            best_pattern_grid = temp_result_grid.copy()
                            best_pattern_adjacency_score = adjacency_score
                            best_pattern_start_x = start_x
                            best_pattern_start_y = start_y
                            best_pattern_width = pattern_width
                            best_pattern_height = pattern_height
                        elif (
                            current_pattern_bonus == highest_pattern_bonus
                            and adjacency_score
                            > best_pattern_adjacency_score  # Change >= to > to prefer earlier (top-left) positions on tie
                        ):
                            best_pattern_grid = temp_result_grid.copy()
                            best_pattern_adjacency_score = adjacency_score
                            best_pattern_start_x = start_x
                            best_pattern_start_y = start_y
                            best_pattern_width = pattern_width
                            best_pattern_height = pattern_height

        if highest_pattern_bonus > -float("inf"):
            solved_grid = best_pattern_grid
            solved_bonus = highest_pattern_bonus
            solve_method = "Pattern Match"  # <<< Set method >>>
            # <<< KEEP: Best pattern result >>>
            logging.info(
                f"Best pattern score: {solved_bonus:.4f} (Adjacency: {best_pattern_adjacency_score:.2f}) found at ({best_pattern_start_x},{best_pattern_start_y}) with size {best_pattern_width}x{best_pattern_height} for ship: '{ship}' -- tech: '{tech}' that fits."
            )
            # print_grid_compact(solved_grid) # Optional: Can be noisy for many calls
        else:
            # --- Case 2b: No Pattern Fits ---
            if not forced:
                logging.info(
                    f"Solve map exists for {ship}/{tech}, but no pattern variation fits. Returning 'Pattern No Fit'. UI can prompt to force SA."
                )  # <<< KEEP: Important outcome >>>
                return None, 0.0, 0.0, "Pattern No Fit"
            else:
                # <<< KEEP: Important fallback >>>
                logging.warning(
                    f"Solve map exists for {ship}/{tech}, but no pattern variation fits the grid. FORCING initial Simulated Annealing."
                )
                initial_sa_grid = grid.copy()
                clear_all_modules_of_tech(initial_sa_grid, tech)
                solved_grid, solved_bonus = simulated_annealing(
                    initial_sa_grid,
                    ship,
                    modules,
                    tech,
                    grid,  # full_grid
                    progress_callback=progress_callback,
                    run_id=run_id,
                    stage="initial_sa_no_window",
                    send_grid_updates=send_grid_updates,
                    tech_modules=tech_modules,
                )
                if solved_grid is None:
                    raise ValueError(
                        f"Fallback simulated_annealing failed for partial set with no window on {ship}/{tech}."
                    )
                logging.info(
                    f"Forced fallback SA score (no pattern fit): {solved_bonus:.4f}"
                )  # <<< KEEP: Result of fallback >>>
                solve_method = "Forced Initial SA (No Pattern Fit)"  # <<< Set method >>>

    # --- Opportunity Refinement Stage ---
    # Ensure solved_grid is not None before proceeding (it could be if "Pattern No Fit" was returned and this logic is ever reached without a prior assignment)
    if solved_grid is None:
        # This case should ideally not be hit if "Pattern No Fit" returns early.
        # However, as a safeguard:
        raise ValueError(
            "optimize_placement: solved_grid is None before opportunity refinement, indicating an unexpected state."
        )
    grid_after_initial_placement = solved_grid.copy()  # Grid state before refinement starts
    current_best_score = (
        solved_bonus
        if solved_bonus > -float("inf")
        else calculate_grid_score(grid_after_initial_placement, tech, apply_supercharge_first=False)
    )

    # Prepare grid for opportunity scanning (clear target tech)
    grid_for_opportunity_scan = grid_after_initial_placement.copy()
    clear_all_modules_of_tech(grid_for_opportunity_scan, tech)

    # --- Calculate Pattern Window Score (if applicable) ---
    pattern_window_score = 0.0
    pattern_opportunity_result = None
    if highest_pattern_bonus > -float("inf") and best_pattern_start_x != -1 and best_pattern_width > 0:
        try:
            # Assuming create_localized_grid and calculate_window_score are defined elsewhere
            pattern_window_grid, _, _ = create_localized_grid(
                grid_for_opportunity_scan,  # Use the grid with tech cleared
                best_pattern_start_x,
                best_pattern_start_y,
                tech,
                best_pattern_width,
                best_pattern_height,
            )
            pattern_window_score = calculate_window_score(pattern_window_grid, tech)
            pattern_opportunity_result = (
                best_pattern_start_x,
                best_pattern_start_y,
                best_pattern_width,
                best_pattern_height,
            )
            # logging.debug(f"Calculated opportunity score for pattern area ({best_pattern_width}x{best_pattern_height} at {best_pattern_start_x},{best_pattern_start_y}): {pattern_window_score:.2f}") # <<< COMMENT OUT/DEBUG >>>
        except Exception as e:
            logging.warning(f"Error calculating pattern window score: {e}")
            pattern_window_score = -1.0
            pattern_opportunity_result = None

    # --- Find Scanned Supercharged Opportunities ---
    # Assuming find_supercharged_opportunities is defined elsewhere
    scanned_opportunity_result = find_supercharged_opportunities(
        grid_for_opportunity_scan,
        modules,
        ship,
        tech,
        tech_modules=tech_modules,
    )

    # --- Calculate Scanned Window Score (if applicable) ---
    scanned_window_score = -1.0
    if scanned_opportunity_result:
        scan_x, scan_y, scan_w, scan_h = scanned_opportunity_result
        try:
            # Assuming create_localized_grid and calculate_window_score are defined elsewhere
            scanned_window_grid, _, _ = create_localized_grid(
                grid_for_opportunity_scan,
                scan_x,
                scan_y,
                tech,
                scan_w,
                scan_h,  # Use the grid with tech cleared
            )
            scanned_window_score = calculate_window_score(scanned_window_grid, tech)
            # logging.debug(f"Calculated opportunity score for best scanned area ({scan_w}x{scan_h} at {scan_x},{scan_y}): {scanned_window_score:.2f}") # <<< COMMENT OUT/DEBUG >>>
        except Exception as e:
            logging.warning(f"Error calculating scanned window score: {e}")
            scanned_window_score = -1.0
            scanned_opportunity_result = None  # Invalidate if score calculation failed

    # --- Compare Scores and Select Final Opportunity ---
    final_opportunity_result = None
    opportunity_source = "None"  # For logging
    if pattern_window_score >= scanned_window_score and pattern_opportunity_result is not None:
        logging.info(
            "Using pattern location as the refinement opportunity window (score >= scanned)."
        )  # <<< COMMENT OUT >>>
        final_opportunity_result = pattern_opportunity_result
        opportunity_source = "Pattern"
    elif scanned_opportunity_result is not None:
        logging.info(
            "Using scanned location as the refinement opportunity window (score > pattern or pattern invalid)."
        )  # <<< COMMENT OUT >>>
        final_opportunity_result = scanned_opportunity_result
        opportunity_source = "Scan"
    elif pattern_opportunity_result is not None:  # Fallback if scanning failed but pattern exists
        logging.info(
            "Using pattern location as the refinement opportunity window (scanning failed)."
        )  # <<< COMMENT OUT >>>
        final_opportunity_result = pattern_opportunity_result
        opportunity_source = "Pattern (Fallback)"
        # else: # <<< COMMENT OUT >>>
        logging.info("No suitable opportunity window found from pattern or scanning.")

    # --- Perform Refinement using the Selected Opportunity ---
    # --- Window Sizing for 'pulse' tech ---
    if tech == "pulse" and final_opportunity_result and len(tech_modules) >= 8:
        logging.info("Window sizing active for 'pulse' tech.")
        opp_x_anchor, opp_y_anchor, current_opp_w, current_opp_h = final_opportunity_result
        # Determine the score of the current best opportunity (before considering 4x3 override)
        score_of_current_best_opportunity = -1.0
        if opportunity_source in ["Pattern", "Pattern (Fallback)"]:
            score_of_current_best_opportunity = pattern_window_score
        elif opportunity_source == "Scan":
            score_of_current_best_opportunity = scanned_window_score
        else:
            # Fallback: if opportunity_source is unknown but final_opportunity_result exists,
            # calculate its score based on its current dimensions.
            # This ensures we have a baseline for comparison.
            logging.warning(
                f"Unknown opportunity_source '{opportunity_source}' for experimental sizing. Recalculating score for current best."
            )
            temp_loc_grid, _, _ = create_localized_grid(
                grid_for_opportunity_scan.copy(),
                opp_x_anchor,
                opp_y_anchor,
                tech,
                current_opp_w,
                current_opp_h,
            )
            score_of_current_best_opportunity = calculate_window_score(temp_loc_grid, tech)

        logging.info(
            f"Pulse Sizing: Current best opportunity ({current_opp_w}x{current_opp_h} from {opportunity_source}) score: {score_of_current_best_opportunity:.4f}"
        )

        # Scan the entire grid for the best 4x3 window using _scan_grid_with_window
        # grid_for_opportunity_scan is the grid with the target tech cleared
        # tech_modules is available from the top of optimize_placement
        best_4x3_score_from_scan, best_4x3_pos_from_scan = _scan_grid_with_window(
            grid_for_opportunity_scan.copy(),  # Scan on the grid with tech cleared
            4,
            3,
            len(tech_modules),
            tech,  # Pass fixed dimensions, module count, tech
        )

        if best_4x3_pos_from_scan:
            # The score returned by _scan_grid_with_window is already the window score
            logging.info(
                f"Pulse Sizing: Best 4x3 window found by scan: score {best_4x3_score_from_scan:.4f} at ({best_4x3_pos_from_scan[0]},{best_4x3_pos_from_scan[1]})."
            )

            # Compare the best 4x3 score (from scan) with the score of the current best opportunity
            if best_4x3_score_from_scan > score_of_current_best_opportunity:
                logging.info(
                    f"Pulse Sizing: Scanned 4x3 window (score {best_4x3_score_from_scan:.4f}) is better than current best ({score_of_current_best_opportunity:.4f}). Selecting 4x3."
                )
                # Override final_opportunity_result with the 4x3 window's location and dimensions
                final_opportunity_result = (
                    best_4x3_pos_from_scan[0],
                    best_4x3_pos_from_scan[1],
                    4,
                    3,
                )
            else:
                logging.info(
                    f"Pulse Sizing: Current best opportunity (score {score_of_current_best_opportunity:.4f}) is better or equal to scanned 4x3 ({best_4x3_score_from_scan:.4f}). Keeping original dimensions ({current_opp_w}x{current_opp_h})."
                )
                # final_opportunity_result remains unchanged
        else:
            logging.info("Pulse Sizing: No suitable 4x3 window found by full scan. Keeping original dimensions.")
            # final_opportunity_result remains unchanged
    # --- End Window Sizing ---
    if final_opportunity_result:
        opportunity_x, opportunity_y, window_width, window_height = final_opportunity_result
        # <<< KEEP: Selected window info >>>
        # print(f"INFO -- Selected opportunity window ({opportunity_source}): Start ({opportunity_x}, {opportunity_y}), Size {window_width}x{window_height}")

        # <<< --- Add Check for Available Supercharged Slot in Window --- >>>
        window_has_available_sc = False
        # Use grid_for_opportunity_scan as it has the tech cleared
        for y_win in range(opportunity_y, opportunity_y + window_height):
            for x_win in range(opportunity_x, opportunity_x + window_width):
                # Bounds check for safety
                if 0 <= x_win < grid_for_opportunity_scan.width and 0 <= y_win < grid_for_opportunity_scan.height:
                    cell = grid_for_opportunity_scan.get_cell(x_win, y_win)
                    if cell["active"] and cell["supercharged"] and cell["module"] is None:
                        window_has_available_sc = True
                        break
            if window_has_available_sc:
                break
        # <<< --- End Check --- >>>

        # <<< --- Only proceed if the check passes --- >>>
        if window_has_available_sc:
            # print("INFO -- Opportunity window contains at least one available supercharged slot. Proceeding with refinement.") # <<< COMMENT OUT >>>

            refined_grid_candidate = None
            refined_score_global = -1.0

            refinement_method = ""  # For logging

            # --- Try ML refinement first ---
            refinement_method = "ML"
            # Assuming _handle_ml_opportunity is defined elsewhere
            refined_grid_candidate, refined_score_global = _handle_ml_opportunity(
                grid_after_initial_placement.copy(),  # Pass a copy of the pre-cleared state
                modules,
                ship,
                tech,
                opportunity_x,
                opportunity_y,
                window_width,
                window_height,
                progress_callback=progress_callback,
                run_id=run_id,
                stage="refinement_ml",
                send_grid_updates=send_grid_updates,
                tech_modules=tech_modules,
                available_modules=available_modules,
            )
            if refined_grid_candidate is None:
                logging.info(
                    "ML refinement failed or model not found. Falling back to SA/Refine refinement."
                )  # <<< KEEP: Important fallback >>>

                refinement_method = "ML->SA/Refine Fallback"
                # Assuming _handle_sa_refine_opportunity is defined elsewhere - THIS WAS THE BUG
                refined_grid_candidate, refined_score_global = _handle_sa_refine_opportunity(
                    grid_after_initial_placement.copy(),  # Pass a fresh copy
                    modules,
                    ship,
                    tech,
                    opportunity_x,
                    opportunity_y,
                    window_width,
                    window_height,
                    progress_callback=progress_callback,
                    run_id=run_id,
                    stage="refinement_sa_fallback",
                    send_grid_updates=send_grid_updates,
                    tech_modules=tech_modules,
                )

            # --- Compare and Update based on Refinement Result ---
            if refined_grid_candidate is not None and refined_score_global >= current_best_score:
                # <<< KEEP: Score improvement >>>
                logging.info(
                    f"Opportunity refinement ({refinement_method}) improved score from {current_best_score:.4f} to {refined_score_global:.4f}"
                )
                solved_grid = refined_grid_candidate
                solved_bonus = refined_score_global
                solve_method = refinement_method  # <<< Update method based on successful refinement >>>
            else:  # Refinement didn't improve or failed, keep grid_after_initial_placement
                if refined_grid_candidate is not None:
                    # <<< KEEP: Score did not improve >>>
                    logging.info(
                        f"Opportunity refinement ({refinement_method}) did not improve score ({refined_score_global:.4f} vs {current_best_score:.4f}). Keeping previous best."
                    )
                else:
                    # <<< KEEP: Refinement failed >>>
                    logging.info(
                        f"Opportunity refinement ({refinement_method}) failed completely. Keeping previous best."
                    )
                # solved_grid remains grid_after_initial_placement
                # solved_bonus remains what it was before refinement
                # solve_method remains what it was before refinement

                # --- Final Fallback SA Logic ---
                # <<< KEEP: Important fallback >>>
                logging.info("Refinement didn't improve/failed. Attempting final fallback Simulated Annealing.")
                grid_for_sa_fallback = grid_after_initial_placement.copy()
                # Assuming _handle_sa_refine_opportunity is defined elsewhere
                sa_fallback_grid, sa_fallback_bonus = _handle_sa_refine_opportunity(
                    grid_for_sa_fallback.copy(),  # Pass a copy
                    modules,
                    ship,
                    tech,
                    opportunity_x,
                    opportunity_y,
                    window_width,
                    window_height,
                    progress_callback=progress_callback,
                    run_id=run_id,
                    stage="final_fallback_sa",
                    send_grid_updates=send_grid_updates,
                    tech_modules=tech_modules,
                )
                if sa_fallback_grid is not None and sa_fallback_bonus > current_best_score:
                    # <<< KEEP: Score improvement >>>
                    logging.info(
                        f"Final fallback SA improved score from {current_best_score:.4f} to {sa_fallback_bonus:.4f}"
                    )
                    solved_grid = sa_fallback_grid
                    solved_bonus = sa_fallback_bonus
                    solve_method = "Final Fallback SA"  # <<< Update method >>>

                elif sa_fallback_grid is not None:
                    # <<< KEEP: Score did not improve >>>
                    logging.info(
                        f"Final fallback SA (due to unplaced modules) did not improve score ({sa_fallback_bonus:.4f} vs {current_best_score:.4f}). Keeping previous best."
                    )
                else:
                    logging.error("Final fallback Simulated Annealing failed. Keeping previous best.")

        # <<< --- Else block for the supercharged check --- >>>
        else:
            # <<< KEEP: Reason for skipping refinement >>>
            logging.info(
                "Skipping refinement: Selected opportunity window does not contain any available supercharged slots."
            )
            # solved_grid remains grid_after_initial_placement
            solved_bonus = current_best_score
            # solve_method remains what it was before refinement
        # <<< --- End supercharged check block --- >>>

    else:  # No final_opportunity_result found
        # <<< KEEP: Reason for skipping refinement >>>
        print("INFO -- No refinement performed as no suitable opportunity window was selected.")
        # solved_grid remains grid_after_initial_placement
        solved_bonus = current_best_score
        # solve_method remains what it was before refinement

    # --- Final Result Calculation ---
    best_grid = solved_grid
    # Use the final solved_bonus calculated through the process
    best_bonus = solved_bonus  # This holds the score after refinement/fallbacks

    # Recalculate just in case, but should match solved_bonus
    final_check_score = calculate_grid_score(solved_grid, tech, apply_supercharge_first=False)
    if abs(final_check_score - best_bonus) > 1e-6:
        logging.warning(
            f"Final check score {final_check_score:.4f} differs from tracked best_bonus {best_bonus:.4f}. Using check score."
        )
        best_bonus = final_check_score

    if solve_score > 1e-9:
        percentage = (best_bonus / solve_score) * 100
    else:
        percentage = 100.0 if best_bonus > 1e-9 else 0.0

    # <<< KEEP: Final result >>>
    logging.info(
        f"SUCCESS -- Final Score: {best_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) using method '{solve_method}' for ship: '{ship}' -- tech: '{tech}'"
    )
    print_grid_compact(best_grid)

    return best_grid, round(percentage, 2), best_bonus, solve_method
