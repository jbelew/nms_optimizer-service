import json
import logging
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def get_tech_modules(modules, ship, tech_key, available_modules=None):
    """Retrieves and filters module definitions for a specific technology.

    This function performs several levels of filtering:
    1.  Finds the technology definition that matches the `tech_key`.
    2.  Selects the standard module variant.
    3.  Filters the final list against a specific list of `available_modules`
        if provided.

    Args:
        modules (dict): The complete module data for the ship.
        ship (str): The ship/platform key (e.g., "hauler").
        tech_key (str): The technology key (e.g., "pulse").
        available_modules (list, optional): A specific list of module IDs to
            filter the final results against. Defaults to None.

    Returns:
        list: A filtered list of module dictionaries for the optimizer to use,
              or None if the technology is not found.
    """
    ship_data = modules
    if ship_data is None:
        logging.error(f"Ship '{ship}' not found in modules data.")
        return None

    types_data = ship_data.get("types")
    if types_data is None:
        logging.error(f"'types' key not found for ship '{ship}'.")
        return None

    # Find all technology definitions that match the tech_key
    candidates_for_tech = []
    for tech_list in types_data.values():
        for technology_info in tech_list:
            if technology_info.get("key") == tech_key:
                candidates_for_tech.append(technology_info)

    # Select the correct technology data (implicitly selecting the default where 'type' is None)
    selected_tech_data = None
    for candidate in candidates_for_tech:
        if candidate.get("type") is None:
            selected_tech_data = candidate
            break

    if selected_tech_data is None:
        logging.warning(f"Technology '{tech_key}' not found for ship '{ship}'. Trying fallback.")
        # Fallback for old data structure (if tech_key is a direct key in the modules dict)
        # This part of the logic assumes the old structure doesn't have solve_type variants.
        if tech_key in modules:
            tech_modules = modules[tech_key].get("modules", [])
        else:
            logging.error(f"Technology '{tech_key}' not found or has no modules for ship '{ship}'.")
            return None
    else:
        tech_modules = selected_tech_data.get("modules", [])

    filtered_modules = tech_modules

    # If a list of available modules is provided, filter the list against it
    if available_modules is not None:
        logging.debug(f"Filtering modules based on available_modules: {available_modules}")
        filtered_modules = [m for m in filtered_modules if m.get("id") in available_modules]

    return filtered_modules


def get_tech_modules_for_training(modules_dict, ship, tech_key):
    """Retrieves all modules for a technology without filtering.

    This function is used for training purposes where all possible modules
    for a technology are required, without considering ownership or solve type.

    Args:
        modules_dict (dict): The modules data dictionary to use, typically
            from a dedicated training data file.
        ship (str): The ship type (e.g., "hauler").
        tech_key (str): The technology key (e.g., "pulse").

    Returns:
        list: A list of all module dictionaries for the specified tech,
              or an empty list if not found.
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
    """Generates a technology tree and returns it as a JSON string.

    This function serves as a wrapper around `get_tech_tree` to provide
    the output in a JSON format suitable for API responses.

    Args:
        ship (str): The ship type (e.g., "hauler").
        module_data (dict): The complete module data for the ship.

    Returns:
        str: A JSON string representing the technology tree, or a JSON
             object with an "error" key if an issue occurs.
    """
    try:
        tech_tree = get_tech_tree(ship, module_data)  # Call your existing function
        if "error" in tech_tree:
            return json.dumps({"error": tech_tree["error"]})  # Return error as JSON
        else:
            return json.dumps(tech_tree, indent=2)  # Return tree as JSON with indentation for readability
    except Exception as e:
        return json.dumps({"error": str(e)})  # Catch any errors during tree generation


def get_tech_tree(ship, module_data):
    """Generates a structured technology tree for a given ship.

    The tree is organized by technology type (e.g., "ship", "weapon") and
    contains detailed information for each technology, including its modules.

    Args:
        ship (str): The ship type (e.g., "hauler").
        module_data (dict): The complete module data for the ship.

    Returns:
        dict: A dictionary representing the technology tree. The keys are
              technology types, and the values are lists of technology
              information dictionaries. Returns a dictionary with an "error"
              key if the ship or its data is not found.
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
                "module_count": len([m for m in tech["modules"] if not m.get("reward")]),
            }
            if "type" in tech:
                tech_info["type"] = tech["type"]
            tech_tree[tech_type].append(tech_info)

    return tech_tree


# Cache for window profiles
_WINDOW_PROFILES = None


def _get_window_profiles():
    global _WINDOW_PROFILES
    if _WINDOW_PROFILES is None:
        profiles_path = os.path.join(os.path.dirname(__file__), "data_definitions", "window_profiles.json")
        try:
            with open(profiles_path, "r", encoding="utf-8") as f:
                _WINDOW_PROFILES = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load window_profiles.json: {e}")
            _WINDOW_PROFILES = {}
    return _WINDOW_PROFILES


def get_tech_window_rules(modules, ship, tech_key):
    """Retrieves the merged window rules dictionary for a specific technology.

    Loads the base rules from window_profiles.json and applies any specific
    overrides defined in the tech's 'window_overrides'.

    Args:
        modules (dict): The complete module data for the ship.
        ship (str): The ship type (e.g., "corvette").
        tech_key (str): The technology key (e.g., "pulse").

    Returns:
        dict: The merged window rules for the technology, or an empty dict if not found.
    """
    ship_data = modules
    if ship_data is None:
        return {}

    types_data = ship_data.get("types")
    if types_data is None:
        return {}

    candidates_for_tech = []
    for tech_list in types_data.values():
        for technology_info in tech_list:
            if technology_info.get("key") == tech_key:
                candidates_for_tech.append(technology_info)

    selected_tech_data = None
    for candidate in candidates_for_tech:
        if candidate.get("type") is None:
            selected_tech_data = candidate
            break

    if selected_tech_data is None and candidates_for_tech:
        selected_tech_data = candidates_for_tech[0]

    if not selected_tech_data:
        return {}

    overrides = selected_tech_data.get("window_overrides", {})
    rules = {}

    profiles = _get_window_profiles()

    # 1. ALWAYS start with the 'standard' profile as the base foundation
    # This solves scaling issues where users select 1-2 modules for things like jetpacks
    if "standard" in profiles:
        rules = json.loads(json.dumps(profiles["standard"]))

    def deep_merge(target, source):
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                deep_merge(target[k], v)
            else:
                target[k] = v

    # 2. Deep merge any specific explicit overrides from this very JSON block
    deep_merge(rules, overrides)

    # Fallback to older window_rules if present (legacy format)
    if not overrides and "window_rules" in selected_tech_data:
        # If legacy, we just return it raw
        return selected_tech_data["window_rules"]

    return rules


__all__ = [
    "get_tech_modules",
    "get_tech_modules_for_training",
    "get_tech_tree",
    "get_tech_tree_json",
    "get_tech_window_rules",
]
