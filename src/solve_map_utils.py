# solve_map_utils.py
"""
This module provides utilities for handling "solve maps," which are pre-computed
optimal layouts for specific technologies on specific ships.

The primary function, `filter_solves`, is used to adapt a generic solve map
to a player's specific context by removing modules that the player does not own.
"""
from .modules_utils import get_tech_modules


def filter_solves(
    solves,
    ship,
    modules,
    tech,
    available_modules=None,
):
    """Filters a solve map to only include modules the player owns.

    This function takes a generic solve map and customizes it based on the
    player's available modules. It checks which modules are owned (including
    rewards) and creates a new solve map containing only those modules.

    Args:
        solves (dict): The dictionary containing all solve maps.
        ship (str): The ship type key.
        modules (dict): The complete module data for the ship.
        tech (str): The technology key.
        available_modules (list, optional): A specific list of available
            module IDs to use. Defaults to None.

    Returns:
        dict: A new solves dictionary containing only the filtered solve map
              for the specified ship and tech. Returns an empty dictionary if
              no solve is found or if no modules are available.
    """

    filtered_solves = {}
    if ship in solves and tech in solves[ship]:
        solve_data = solves[ship][tech]

        if tech == "pulse" and available_modules is not None and "PC" in available_modules:
            solve_data = solves[ship]["photonix"]
            print("INFO -- Forcing tech to 'photonix' for PC")

        if not solve_data:
            return {}

        filtered_solves[ship] = {tech: {}}
        tech_modules = get_tech_modules(modules, ship, tech)
        
        # Initialize the 'map' and 'score' for the filtered solve
        filtered_solves[ship][tech]["map"] = {}
        filtered_solves[ship][tech]["score"] = solve_data.get("score", 0)
        
        if tech_modules is None:
            # If tech_modules retrieval fails, include all modules from the solve map
            # This allows the solve to proceed even if module data structure is unusual
            import logging
            logging.warning(f"Could not retrieve module data for ship '{ship}' and tech '{tech}'. Including all modules from solve map.")
            for position, module_id in solve_data.get("map", {}).items():
                filtered_solves[ship][tech]["map"][position] = module_id
        else:
            owned_module_ids = {module["id"] for module in tech_modules}
            for position, module_id in solve_data.get("map", {}).items():  # Access the nested 'map'
                if module_id is None or module_id == "None" or module_id in owned_module_ids:
                    filtered_solves[ship][tech]["map"][position] = module_id
    return filtered_solves
