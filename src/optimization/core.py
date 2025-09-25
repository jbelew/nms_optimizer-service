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
)
from .helpers import (
    check_all_modules_placed,
    place_all_modules_in_empty_slots,
)
from .windowing import (
    create_localized_grid,
    find_supercharged_opportunities,
    calculate_window_score,
    _scan_grid_with_window,
)


def optimize_placement(
    grid,
    ship,
    modules,
    tech,
    player_owned_rewards=None,
    forced=False,
    experimental_window_sizing=False,
    progress_callback=None,
    run_id=None,
    send_grid_updates=False,
    solve_type=None,
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
        player_owned_rewards (list, optional): List of reward module IDs owned. Defaults to None.
        forced (bool): If True and no pattern fits a solve map, forces SA.
                       If False, returns "Pattern No Fit" to allow UI intervention.
        experimental_window_sizing (bool): If True and tech is 'pulse', dynamically chooses
                                           between a 4x3 and 4x2 window for refinement.

    Returns:
        tuple: (best_grid, percentage, best_bonus, solve_method)
               - best_grid (Grid): The optimized grid.
               - percentage (float): The percentage of the official solve score achieved.
               - best_bonus (float): The actual score achieved by the best grid.
               - solve_method (str): The name of the method used to generate the final grid.
    Raises:
        ValueError: If no empty, active slots are available or if critical steps fail.
    """
    logging.info(
        f"Attempting solve for ship: '{ship}' -- tech: '{tech}' -- Exp. Window: {experimental_window_sizing}"
    )
    logging.debug(f"send_grid_updates: {send_grid_updates}")

    if player_owned_rewards is None:
        player_owned_rewards = []

    # --- Get modules for the current tech ---
    # This list is used to determine module_count for experimental window sizing
    # and for the check_all_modules_placed function.
    full_tech_modules_list = get_tech_modules(
        modules,
        ship,
        tech,
        player_owned_rewards,
        solve_type=solve_type,
        available_modules=None,
    )
    tech_modules = get_tech_modules(
        modules,
        ship,
        tech,
        player_owned_rewards,
        solve_type=solve_type,
        available_modules=available_modules,
    )
    if not tech_modules:
        # This case should ideally be caught by has_empty_active_slots or other checks,
        # but as a safeguard if get_tech_modules returns None or empty for a valid tech key.
        logging.warning(
            f"No modules retrieved for ship '{ship}', tech '{tech}'. Cannot proceed with optimization."
        )
        # Return a grid that's essentially empty for this tech, with 0 score.
        cleared_grid_on_fail = grid.copy()
        clear_all_modules_of_tech(cleared_grid_on_fail, tech)
        return cleared_grid_on_fail, 0.0, 0.0, "Module Definition Error"

    # --- Early Check: Any Empty, Active Slots? ---
    has_empty_active_slots = False
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.get_cell(x, y)["active"] and grid.get_cell(x, y)["module"] is None:
                has_empty_active_slots = True
                break
        if has_empty_active_slots:
            break
    if not has_empty_active_slots:
        raise ValueError(
            f"No empty, active slots available on the grid for ship: '{ship}' -- tech: '{tech}'."
        )

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
    sa_was_initial_placement = False
    solve_method = "Unknown"  # <<< Initialize solve_method >>>

    # --- Load Solves On-Demand ---
    all_solves_for_ship = get_solve_map(ship, solve_type)
    # Create the structure that filter_solves expects
    solves_for_filtering = {ship: all_solves_for_ship} if all_solves_for_ship else {}
    filtered_solves = filter_solves(
        solves_for_filtering,
        ship,
        modules,
        tech,
        player_owned_rewards,
        solve_type=solve_type,
        available_modules=available_modules,
    )
    # --- End On-Demand Loading ---

    # --- Initial Placement Strategy ---
    if ship not in filtered_solves or (
        ship in filtered_solves and tech not in filtered_solves[ship]
    ):
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
            player_owned_rewards,
            solve_type=solve_type,
            tech_modules=tech_modules,
        )
        solved_bonus = calculate_grid_score(solved_grid, tech)
        solve_score = 0
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
            and len(tech_modules) < len(full_tech_modules_list)
        )

        # Special case for 'pulse' tech: if only 'PC' is missing, it's not a partial set
        if is_partial_set and tech == "pulse":
            full_tech_module_ids = {m["id"] for m in full_tech_modules_list}
            tech_module_ids = {m["id"] for m in tech_modules}
            missing_modules = full_tech_module_ids - tech_module_ids
            if missing_modules == {"PC"}:
                logging.info(
                    "Pulse tech with only 'PC' module missing is considered a full set. Proceeding with normal optimization."
                )
                is_partial_set = False

        if is_partial_set:
            logging.info(
                f"Partial module set ({len(tech_modules)}/{len(full_tech_modules_list)}) for {ship}/{tech}. Skipping patterns, running windowed SA."
            )

            # We need solve_score for percentage calculation. Let's get it now.
            solve_score = 0
            all_solves_for_ship = get_solve_map(ship, solve_type)
            if all_solves_for_ship:
                solves_for_filtering = {ship: all_solves_for_ship}
                # Filter with full module list to find the official solve score
                temp_filtered_solves = filter_solves(
                    solves_for_filtering,
                    ship,
                    modules,
                    tech,
                    player_owned_rewards,
                    solve_type=solve_type,
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
                player_owned_rewards,
                solve_type=solve_type,
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
                    player_owned_rewards,
                    opp_x,
                    opp_y,
                    opp_w,
                    opp_h,
                    progress_callback=progress_callback,
                    run_id=run_id,
                    stage="partial_set_sa",
                    send_grid_updates=send_grid_updates,
                    solve_type=solve_type,
                    tech_modules=tech_modules,
                )
                solve_method = "Partial Set SA"
            else:
                logging.warning(
                    "No supercharged opportunity window found. Finding first available window for SA."
                )
                num_modules = len(tech_modules)
                w, h = 0, 0
                if num_modules <= 2:
                    w, h = 2, 1
                elif num_modules <= 4:
                    w, h = 2, 2
                elif num_modules <= 6:
                    w, h = 3, 2
                elif num_modules <= 8:
                    w, h = 4, 2
                elif num_modules == 9:
                    w, h = 3, 3
                elif num_modules <= 12:
                    w, h = 4, 3
                else:
                    w, h = 5, 3  # Fallback

                best_pos = None
                # Find the first top-left position that can fit the window and is composed of active, empty slots
                for y in range(grid_for_sa.height - h + 1):
                    for x in range(grid_for_sa.width - w + 1):
                        is_valid = True
                        for j in range(h):
                            for i in range(w):
                                cell = grid_for_sa.get_cell(x + i, y + j)
                                if not cell["active"] or cell["module"] is not None:
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if is_valid:
                            best_pos = (x, y)
                            break
                    if best_pos:
                        break

                if best_pos:
                    opp_x, opp_y = best_pos
                    logging.info(f"Found first available window: {w}x{h} at ({opp_x}, {opp_y})")
                    solved_grid, solved_bonus = _handle_sa_refine_opportunity(
                        grid_for_sa,
                        modules,
                        ship,
                        tech,
                        player_owned_rewards,
                        opp_x,
                        opp_y,
                        w,
                        h,
                        progress_callback=progress_callback,
                        run_id=run_id,
                        stage="partial_set_sa_scanned",
                        send_grid_updates=send_grid_updates,
                        solve_type=solve_type,
                        tech_modules=tech_modules,
                    )
                    solve_method = "Partial Set SA (First Fit)"
                else:
                    logging.error(
                        "Could not find any suitable window for partial module set. Placing modules directly."
                    )
                    solved_grid = place_all_modules_in_empty_slots(
                        grid_for_sa,
                        modules,
                        ship,
                        tech,
                        player_owned_rewards,
                        solve_type=solve_type,
                        tech_modules=tech_modules,
                    )
                    solved_bonus = calculate_grid_score(solved_grid, tech)
                    solve_method = "Partial Set Placement (No Window)"

            if solved_grid is None:
                logging.error(f"Partial module set SA failed for {ship}/{tech}.")
                cleared_grid_on_fail = grid.copy()
                clear_all_modules_of_tech(cleared_grid_on_fail, tech)
                return cleared_grid_on_fail, 0.0, 0.0, "Partial SA Failed"

            if solve_score > 1e-9:
                percentage = (solved_bonus / solve_score) * 100
            else:
                percentage = 100.0 if solved_bonus > 1e-9 else 0.0

            logging.info(
                f"SUCCESS -- Final Score (Partial Set): {solved_bonus:.4f} ({percentage:.2f}% of potential {solve_score:.4f}) using method '{solve_method}' for ship: '{ship}' -- tech: '{tech}'"
            )
            print_grid_compact(solved_grid)
            return solved_grid, round(percentage, 2), solved_bonus, solve_method

        # --- Case 2: Solve Map Exists ---
        solve_data = filtered_solves[ship][tech]
        original_pattern = solve_data.get("map")
        solve_score = solve_data.get("score")

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
                        player_owned_rewards,
                        solve_type=solve_type,
                        tech_modules=tech_modules,
                    )
                    if temp_result_grid is not None:
                        current_pattern_bonus = calculate_grid_score(
                            temp_result_grid, tech
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
            # print_grid_compact(solved_grid)
            sa_was_initial_placement = False
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
                    player_owned_rewards,
                    cooling_rate=0.999,
                    iterations_per_temp=35,
                    initial_swap_probability=0.55,
                    max_processing_time=20.0,
                    progress_callback=progress_callback,
                    run_id=run_id,
                    stage="initial_placement",
                    send_grid_updates=send_grid_updates,
                    solve_type=solve_type if solve_type is not None else "",
                    tech_modules=tech_modules,
                )
                if solved_grid is None:
                    raise ValueError(
                        f"Forced fallback simulated_annealing failed for {ship}/{tech} when no pattern fit."
                    )
                logging.info(
                    f"Forced fallback SA score (no pattern fit): {solved_bonus:.4f}"
                )  # <<< KEEP: Result of fallback >>>
                solve_method = (
                    "Forced Initial SA (No Pattern Fit)"  # <<< Set method >>>
                )
                sa_was_initial_placement = True

    # --- Opportunity Refinement Stage ---
    # Ensure solved_grid is not None before proceeding (it could be if "Pattern No Fit" was returned and this logic is ever reached without a prior assignment)
    if solved_grid is None:
        # This case should ideally not be hit if "Pattern No Fit" returns early.
        # However, as a safeguard:
        raise ValueError(
            "optimize_placement: solved_grid is None before opportunity refinement, indicating an unexpected state."
        )
    grid_after_initial_placement = (
        solved_grid.copy()
    )  # Grid state before refinement starts
    current_best_score = calculate_grid_score(grid_after_initial_placement, tech)

    # Prepare grid for opportunity scanning (clear target tech)
    grid_for_opportunity_scan = grid_after_initial_placement.copy()
    clear_all_modules_of_tech(grid_for_opportunity_scan, tech)

    # --- Calculate Pattern Window Score (if applicable) ---
    pattern_window_score = 0.0
    pattern_opportunity_result = None
    if (
        highest_pattern_bonus > -float("inf")
        and best_pattern_start_x != -1
        and best_pattern_width > 0
    ):
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
        player_owned_rewards,
        solve_type=solve_type,
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
    if (
        pattern_window_score >= scanned_window_score
        and pattern_opportunity_result is not None
    ):
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
    elif (
        pattern_opportunity_result is not None
    ):  # Fallback if scanning failed but pattern exists
        logging.info(
            "Using pattern location as the refinement opportunity window (scanning failed)."
        )  # <<< COMMENT OUT >>>
        final_opportunity_result = pattern_opportunity_result
        opportunity_source = "Pattern (Fallback)"
        # else: # <<< COMMENT OUT >>>
        logging.info("No suitable opportunity window found from pattern or scanning.")

    # --- Perform Refinement using the Selected Opportunity ---
    # --- Experimental Window Sizing for 'pulse' tech ---
    if (
        experimental_window_sizing
        and tech == "pulse"
        and final_opportunity_result
        and len(tech_modules) >= 8
    ):
        logging.info("Experimental window sizing active for 'pulse' tech.")
        opp_x_anchor, opp_y_anchor, current_opp_w, current_opp_h = (
            final_opportunity_result
        )

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
            score_of_current_best_opportunity = calculate_window_score(
                temp_loc_grid, tech
            )

        logging.info(
            f"Experimental: Current best opportunity ({current_opp_w}x{current_opp_h} from {opportunity_source}) score: {score_of_current_best_opportunity:.4f}"
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
                f"Experimental: Best 4x3 window found by scan: score {best_4x3_score_from_scan:.4f} at ({best_4x3_pos_from_scan[0]},{best_4x3_pos_from_scan[1]})."
            )

            # Compare the best 4x3 score (from scan) with the score of the current best opportunity
            if best_4x3_score_from_scan > score_of_current_best_opportunity:
                logging.info(
                    f"Experimental: Scanned 4x3 window (score {best_4x3_score_from_scan:.4f}) is better than current best ({score_of_current_best_opportunity:.4f}). Selecting 4x3."
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
                    f"Experimental: Current best opportunity (score {score_of_current_best_opportunity:.4f}) is better or equal to scanned 4x3 ({best_4x3_score_from_scan:.4f}). Keeping original dimensions ({current_opp_w}x{current_opp_h})."
                )
                # final_opportunity_result remains unchanged
        else:
            logging.info(
                "Experimental: No suitable 4x3 window found by full scan. Keeping original dimensions."
            )
            # final_opportunity_result remains unchanged
    # --- End Experimental Window Sizing ---
    if final_opportunity_result:
        opportunity_x, opportunity_y, window_width, window_height = (
            final_opportunity_result
        )
        # <<< KEEP: Selected window info >>>
        # print(f"INFO -- Selected opportunity window ({opportunity_source}): Start ({opportunity_x}, {opportunity_y}), Size {window_width}x{window_height}")

        # <<< --- Add Check for Available Supercharged Slot in Window --- >>>
        window_has_available_sc = False
        # Use grid_for_opportunity_scan as it has the tech cleared
        for y_win in range(opportunity_y, opportunity_y + window_height):
            for x_win in range(opportunity_x, opportunity_x + window_width):
                # Bounds check for safety
                if (
                    0 <= x_win < grid_for_opportunity_scan.width
                    and 0 <= y_win < grid_for_opportunity_scan.height
                ):
                    cell = grid_for_opportunity_scan.get_cell(x_win, y_win)
                    if (
                        cell["active"]
                        and cell["supercharged"]
                        and cell["module"] is None
                    ):
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
            sa_was_ml_fallback = False
            refinement_method = ""  # For logging

            # --- Branch based on experimental flag ---
            # Default path: Try ML refinement first
            refinement_method = "ML"
            # Assuming _handle_ml_opportunity is defined elsewhere
            refined_grid_candidate, refined_score_global = _handle_ml_opportunity(
                grid_after_initial_placement.copy(),  # Pass a copy of the pre-cleared state
                modules,
                ship,
                tech,
                player_owned_rewards,
                opportunity_x,
                opportunity_y,
                window_width,
                window_height,
                progress_callback=progress_callback,
                run_id=run_id,
                stage="refinement_ml",
                send_grid_updates=send_grid_updates,
                solve_type=solve_type,
                tech_modules=tech_modules,
                available_modules=available_modules,
            )
            if refined_grid_candidate is None:
                logging.info(
                    "ML refinement failed or model not found. Falling back to SA/Refine refinement."
                )  # <<< KEEP: Important fallback >>>
                sa_was_ml_fallback = True
                refinement_method = "ML->SA/Refine Fallback"
                # Assuming _handle_sa_refine_opportunity is defined elsewhere - THIS WAS THE BUG
                refined_grid_candidate, refined_score_global = (
                    _handle_sa_refine_opportunity(
                        grid_after_initial_placement.copy(),  # Pass a fresh copy
                        modules,
                        ship,
                        tech,
                        player_owned_rewards,
                        opportunity_x,
                        opportunity_y,
                        window_width,
                        window_height,
                        progress_callback=progress_callback,
                        run_id=run_id,
                        stage="refinement_sa_fallback",
                        send_grid_updates=send_grid_updates,
                        solve_type=solve_type,
                        tech_modules=tech_modules,
                    )
                )

            # --- Compare and Update based on Refinement Result ---
            if (
                refined_grid_candidate is not None
                and refined_score_global >= current_best_score
            ):
                # <<< KEEP: Score improvement >>>
                logging.info(
                    f"Opportunity refinement ({refinement_method}) improved score from {current_best_score:.4f} to {refined_score_global:.4f}"
                )
                solved_grid = refined_grid_candidate
                solved_bonus = refined_score_global
                solve_method = refinement_method  # <<< Update method based on successful refinement >>>
                sa_was_initial_placement = False
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
                solved_bonus = current_best_score
                # solve_method remains what it was before refinement

                # --- Final Fallback SA Logic ---
                if (
                    not sa_was_ml_fallback
                ):  # Only run if the previous SA wasn't already a fallback from ML
                    # <<< KEEP: Important fallback >>>
                    logging.info(
                        "Refinement didn't improve/failed (and was not ML->SA fallback). Attempting final fallback Simulated Annealing."
                    )
                    grid_for_sa_fallback = grid_after_initial_placement.copy()
                    # Assuming _handle_sa_refine_opportunity is defined elsewhere
                    sa_fallback_grid, sa_fallback_bonus = _handle_sa_refine_opportunity(
                        grid_for_sa_fallback.copy(),  # Pass a copy
                        modules,
                        ship,
                        tech,
                        player_owned_rewards,
                        opportunity_x,
                        opportunity_y,
                        window_width,
                        window_height,
                        progress_callback=progress_callback,
                        run_id=run_id,
                        stage="final_fallback_sa",
                        send_grid_updates=send_grid_updates,
                        solve_type=solve_type,
                        tech_modules=tech_modules,
                    )
                    if (
                        sa_fallback_grid is not None
                        and sa_fallback_bonus > current_best_score
                    ):
                        # <<< KEEP: Score improvement >>>
                        logging.info(
                            f"Final fallback SA improved score from {current_best_score:.4f} to {sa_fallback_bonus:.4f}"
                        )
                        solved_grid = sa_fallback_grid
                        solved_bonus = sa_fallback_bonus
                        solve_method = "Final Fallback SA"  # <<< Update method >>>
                        sa_was_initial_placement = False
                    elif sa_fallback_grid is not None:
                        # <<< KEEP: Score did not improve >>>
                        logging.info(
                            f"Final fallback SA did not improve score ({sa_fallback_bonus:.4f} vs {current_best_score:.4f}). Keeping previous best."
                        )
                    else:
                        logging.error(
                            "Final fallback Simulated Annealing failed. Keeping previous best."
                        )

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
        print(
            "INFO -- No refinement performed as no suitable opportunity window was selected."
        )
        # solved_grid remains grid_after_initial_placement
        solved_bonus = current_best_score
        # solve_method remains what it was before refinement

    # --- Final Checks and Fallbacks (Simulated Annealing if modules not placed) ---
    # Assuming check_all_modules_placed is defined elsewhere (or imported)
    all_modules_placed = check_all_modules_placed(
        solved_grid,
        modules,
        ship,
        tech,
        player_owned_rewards,
        tech_modules=tech_modules,
        solve_type=solve_type,
    )
    if not all_modules_placed and not sa_was_initial_placement:
        # <<< KEEP: Important fallback >>>
        logging.warning(
            "Not all modules placed AND initial placement wasn't SA. Running final SA."
        )
        grid_for_final_sa = solved_grid.copy()
        clear_all_modules_of_tech(grid_for_final_sa, tech)
        temp_solved_grid, temp_solved_bonus_sa = simulated_annealing(
            grid_for_final_sa,
            ship,
            modules,
            tech,
            grid,  # full_grid
            player_owned_rewards,
            iterations_per_temp=25,
            initial_swap_probability=0.70,
            final_swap_probability=0.25,
            max_processing_time=20.0,
            progress_callback=progress_callback,
            run_id=run_id,
            stage="final_sa_unplaced_modules",
            send_grid_updates=send_grid_updates,
            solve_type=solve_type if solve_type is not None else "",
            tech_modules=tech_modules,
        )
        if temp_solved_grid is not None:
            final_sa_score = calculate_grid_score(temp_solved_grid, tech)
            # Use solved_bonus which holds the score *after* potential refinement
            if final_sa_score > solved_bonus:
                # <<< KEEP: Score improvement >>>
                logging.info(
                    f"Final SA (due to unplaced modules) improved score from {solved_bonus:.4f} to {final_sa_score:.4f}"
                )
                solved_grid = temp_solved_grid
                solved_bonus = final_sa_score
                solve_method = "Final SA (Unplaced Modules)"  # <<< Update method >>>
            else:
                # <<< KEEP: Score did not improve >>>
                logging.info(
                    f"Final SA (due to unplaced modules) did not improve score ({final_sa_score:.4f} vs {solved_bonus:.4f}). Keeping previous best."
                )
        else:
            logging.error(
                f"Final simulated_annealing solver (due to unplaced modules) failed for ship: '{ship}' -- tech: '{tech}'. Returning previous best grid."
            )
    elif not all_modules_placed and sa_was_initial_placement:
        # <<< KEEP: Important warning >>>
        logging.warning(
            "Not all modules placed, but initial placement WAS SA. Skipping final SA check."
        )

    # --- Final Result Calculation ---
    best_grid = solved_grid
    # Use the final solved_bonus calculated through the process
    best_bonus = solved_bonus  # This holds the score after refinement/fallbacks

    # Recalculate just in case, but should match solved_bonus
    final_check_score = calculate_grid_score(solved_grid, tech)
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
