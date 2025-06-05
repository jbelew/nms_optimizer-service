import json
from modules import modules


def get_tech_modules(modules, ship, tech_key, player_owned_rewards=None):
    """
    Retrieves modules for a specified ship and technology key, considering player-owned rewards.

    Args:
        modules (dict): The modules data dictionary.
        ship (str): The ship type.
        tech_key (str): The technology key.
        player_owned_rewards (list, optional): A list of reward module IDs owned by the player. Defaults to None.

    Returns:
        list: A list of module dictionaries, or None if an error occurs.
    """
    if player_owned_rewards is None:
        player_owned_rewards = []

    ship_data = modules.get(ship)
    if ship_data is None:
        print(f"Error: Ship '{ship}' not found in modules data.")
        return None

    types_data = ship_data.get("types")
    if types_data is None:
        print(f"Error: 'types' key not found for ship '{ship}'.")
        return None

    for tech_type in types_data:
        tech_category = types_data.get(tech_type)
        if tech_category is None:
            print(f"Error: Technology type '{tech_type}' not found for ship '{ship}'.")
            continue  # skip this type and check the next

        for technology in tech_category:
            if technology.get("key") == tech_key:
                modules_list = technology.get("modules")
                if modules_list is None:
                    print(
                        f"Error: 'modules' key not found for technology '{tech_key}' within type '{tech_type}' on ship '{ship}'."
                    )
                    return None

                filtered_modules = []
                for module in modules_list:
                    if module["type"] == "reward":
                        if module["id"] in player_owned_rewards:
                            # Create a copy of the module before modifying it
                            modified_module = module.copy()
                            modified_module["type"] = "bonus"  # Convert type to bonus
                            filtered_modules.append(modified_module)
                    else:
                        filtered_modules.append(
                            module
                        )  # No need to copy non-reward modules

                return filtered_modules

    print(f"Error: Technology '{tech_key}' not found for ship '{ship}'.")
    return None


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
    ship_data = modules_dict.get(ship)
    if ship_data is None:
        print(f"Error: Ship '{ship}' not found in modules data.")
        return []

    types_data = ship_data.get("types")
    if types_data is None:
        print(f"Error: 'types' key not found for ship '{ship}'.")
        return []

    for tech_list in types_data.values():
        for tech_data in tech_list:
            if tech_data.get("key") == tech_key:
                return tech_data.get("modules", [])
    return []


def get_tech_tree_json(ship):
    """Generates a technology tree for a given ship and returns it as JSON."""
    try:
        tech_tree = get_tech_tree(ship)  # Call your existing function
        if "error" in tech_tree:
            return json.dumps({"error": tech_tree["error"]})  # Return error as JSON
        else:
            return json.dumps(
                tech_tree, indent=2
            )  # Return tree as JSON with indentation for readability
    except Exception as e:
        return json.dumps({"error": str(e)})  # Catch any errors during tree generation


def get_tech_tree(ship):
    """Generates a technology tree for a given ship."""
    ship_data = modules.get(ship)
    if ship_data is None:
        return {"error": f"Ship '{ship}' not found."}

    types_data = ship_data.get("types")
    if types_data is None:
        return {"error": f"'types' key not found for ship '{ship}'."}

    tech_tree = {}
    for tech_type, tech_list in types_data.items():
        tech_tree[tech_type] = []
        for tech in tech_list:
            tech_tree[tech_type].append(
                {
                    "label": tech["label"],
                    "key": tech["key"],
                    "modules": tech["modules"],
                    "image": tech.get("image"),  # Get the image value
                    "color": tech.get("color"),  # Get the color value
                }
            )

    return tech_tree


__all__ = [
    "get_tech_modules",
    "get_tech_modules_for_training",
    "get_tech_tree",
    "get_tech_tree_json",
]
