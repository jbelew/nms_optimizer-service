# solve_map_utils.py
from modules_data import get_tech_modules

def filter_solves(solves, ship, modules, tech, player_owned_rewards=None):
    """
    Filters the solves dictionary to remove modules that the player does not own.

    Args:
        solves (dict): The solves dictionary.
        ship (str): The ship type.
        modules (dict): The modules data.
        tech (str): The technology key.
        player_owned_rewards (list, optional): A list of reward module IDs owned by the player. Defaults to None.

    Returns:
        dict: A new solves dictionary with unowned modules removed from the solve map.  Returns an empty dictionary if no solves are found for the given ship and tech.
    """
    filtered_solves = {}
    if ship in solves and tech in solves[ship]:
        solve_data = solves[ship][tech]
        filtered_solves[ship] = {tech: {}}
        tech_modules = get_tech_modules(modules, ship, tech, player_owned_rewards)
        if tech_modules is None:
            print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
            return {}  # Return empty dict if no modules are found

        owned_module_ids = {module["id"] for module in tech_modules}
        for position, module_id in solve_data["map"].items(): # Access the nested 'map'
            if module_id is None or module_id in owned_module_ids:
                filtered_solves[ship][tech]["map"] = filtered_solves[ship][tech].get("map", {})
                filtered_solves[ship][tech]["map"][position] = module_id
        filtered_solves[ship][tech]["score"] = solve_data["score"] # Copy the score
    return filtered_solves