# solve_map_utils.py
from .modules_utils import get_tech_modules


def filter_solves(
    solves,
    ship,
    modules,
    tech,
    player_owned_rewards=None,
    solve_type=None,
    available_modules=None,
):
    """
    Filters the solves dictionary to remove modules that the player does not own.

    Args:
        solves (dict): The solves dictionary.
        ship (str): The ship type.
        modules (dict): The modules data.
        tech (str): The technology key.
        player_owned_rewards (list, optional): A list of reward module IDs owned by the player. Defaults to None.
        solve_type (str, optional): The type of solve, e.g., "normal" or "max". Defaults to None.
        available_modules (list, optional): A list of available module IDs. Defaults to None.

    Returns:
        dict: A new solves dictionary with unowned modules removed from the solve map.  Returns an empty dictionary if no solves are found for the given ship and tech.
    """

    filtered_solves = {}
    if ship in solves and tech in solves[ship]:
        solve_data = solves[ship][tech]

        if (
            tech == "pulse"
            and available_modules is not None
            and "PC" in available_modules
        ):
            solve_data = solves[ship]["photonix"]
            print("INFO -- Forcing tech to 'photonix' for PC")

        if not solve_data:
            return {}

        filtered_solves[ship] = {tech: {}}
        tech_modules = get_tech_modules(
            modules, ship, tech, player_owned_rewards, solve_type=solve_type
        )
        if tech_modules is None:
            print(f"Error: No modules found for ship '{ship}' and tech '{tech}'.")
            return {}  # Return empty dict if no modules are found

        owned_module_ids = {module["id"] for module in tech_modules}
        # Initialize the 'map' and 'score' for the filtered solve
        filtered_solves[ship][tech]["map"] = {}
        filtered_solves[ship][tech]["score"] = solve_data.get("score", 0)

        for position, module_id in solve_data.get(
            "map", {}
        ).items():  # Access the nested 'map'
            if module_id is None or module_id == "None" or module_id in owned_module_ids:
                filtered_solves[ship][tech]["map"][position] = module_id
    return filtered_solves
