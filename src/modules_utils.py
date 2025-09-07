import json
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def get_tech_modules(modules, ship, tech_key, player_owned_rewards=None, solve_type=None):
    logging.debug(f"get_tech_modules called with solve_type: {solve_type}")
    """
    Retrieves modules for a specified ship and technology key, considering player-owned rewards.
    Selects the module list based on the provided solve_type ("normal" or "max").

    Args:
        modules (dict): The modules data dictionary.
        ship (str): The ship type.
        tech_key (str): The technology key.
        player_owned_rewards (list, optional): A list of reward module IDs owned by the player. Defaults to None.
        solve_type (str, optional): The type of solve, e.g., "normal" or "max". Defaults to None.

    Returns:
        list: A list of module dictionaries, or None if an error occurs.
    """
    if player_owned_rewards is None:
        player_owned_rewards = []

    ship_data = modules
    if ship_data is None:
        logging.error(f"Ship '{ship}' not found in modules data.")
        return None

    types_data = ship_data.get("types")
    if types_data is None:
        logging.error(f"'types' key not found for ship '{ship}'.")
        return None

    found_modules_list = None
    for tech_list in types_data.values():
        for technology_info in tech_list:
            if technology_info.get("key") == tech_key:
                # If solve_type is provided, try to match the 'type' field
                if solve_type is not None:
                    if technology_info.get("type") == solve_type:
                        found_modules_list = technology_info.get("modules")
                        break
                elif solve_type is None and technology_info.get("type") is None:
                    # If solve_type is None, and the technology_info also has no 'type' key,
                    # then this is the default set of modules.
                    found_modules_list = technology_info.get("modules")
                    break
        if found_modules_list:
            break

    if found_modules_list is None:
        logging.error(f"Technology '{tech_key}' with solve_type '{solve_type}' not found or has no modules for ship '{ship}'.")
        return None

    modules_list = found_modules_list
    if modules_list is None:
        logging.error(
            f"'modules' key not found for technology '{tech_key}' (type: {solve_type}) on ship '{ship}'."
        )
        return None

    filtered_modules = []
    for module in modules_list:
        if module["type"] == "reward":
            if module["id"] in player_owned_rewards:
                modified_module = module.copy()
                modified_module["type"] = "bonus"
                filtered_modules.append(modified_module)
        else:
            filtered_modules.append(module)

    return filtered_modules


def get_tech_modules_for_training(modules_dict, ship, tech_key):
    """
    Retrieves modules for training from a provided modules dictionary,
    returning the modules as they are defined.

    Args:
        modules_dict (dict): The modules data dictionary to use (e.g., from modules_for_training.py).
        ship (str): The ship type.
        tech_key (str): The technology key.

    Returns:
        list: A list of module dictionaries, or an empty list if not found.
    """
    # The 'modules_dict' parameter is now the ship-specific data, so we don't need to do a lookup.
    ship_data = modules_dict
    if ship_data is None:
        logging.error(f"Ship '{ship}' not found in modules data.")
        return []

    types_data = ship_data.get("types")
    if types_data is None:
        logging.error(f"'types' key not found for ship '{ship}'.")
        return []

    for tech_list in types_data.values():
        for tech_data in tech_list:
            if tech_data.get("key") == tech_key:
                return tech_data.get("modules", [])
    return []


def get_tech_tree_json(ship, module_data):
    """
    Generates a technology tree for a given ship and returns it as JSON.
    """
    try:
        tech_tree = get_tech_tree(ship, module_data)  # Call your existing function
        if "error" in tech_tree:
            return json.dumps({"error": tech_tree["error"]})  # Return error as JSON
        else:
            return json.dumps(
                tech_tree, indent=2
            )  # Return tree as JSON with indentation for readability
    except Exception as e:
        return json.dumps({"error": str(e)})  # Catch any errors during tree generation


def get_tech_tree(ship, module_data):
    """
    Generates a technology tree for a given ship.
    """
    if not module_data:
        return {"error": f"Ship '{ship}' not found."}

    types_data = module_data.get("types")
    if types_data is None:
        return {"error": f"'types' key not found for ship '{ship}'."}

    tech_tree = {}
    for tech_type, tech_list in types_data.items():
        tech_tree[tech_type] = []
        for tech in tech_list:
            tech_info = {
                "label": tech["label"],
                "key": tech["key"],
                "modules": tech["modules"],
                "image": tech.get("image"),
                "color": tech.get("color"),
                "module_count": len(
                    [m for m in tech["modules"] if m.get("type") != "reward"]
                ),
            }
            if "type" in tech:
                tech_info["type"] = tech["type"]
            tech_tree[tech_type].append(tech_info)

    return tech_tree


__all__ = [
    "get_tech_modules",
    "get_tech_modules_for_training",
    "get_tech_tree",
    "get_tech_tree_json",
]