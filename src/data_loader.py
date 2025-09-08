import json
import logging
import os
from functools import lru_cache
from typing import List, Optional

from .data_definitions.modules_for_training import MODULES_FOR_TRAINING

# --- Constants ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data_definitions")


def _convert_map_keys_to_tuple(data):
    """
    Recursively converts string keys in a 'map' dictionary back to tuples.
    This is the reverse of the conversion done when saving to JSON.
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            # Check if the key is a string representation of a tuple (e.g., "0,0")
            if isinstance(k, str) and "," in k:
                try:
                    # Attempt to convert "x,y" string back to a tuple of integers
                    new_key = tuple(map(int, k.split(",")))
                    new_dict[new_key] = _convert_map_keys_to_tuple(v)
                except ValueError:
                    # If conversion fails, keep the original key
                    new_dict[k] = _convert_map_keys_to_tuple(v)
            else:
                new_dict[k] = _convert_map_keys_to_tuple(v)
        return new_dict
    elif isinstance(data, list):
        return [_convert_map_keys_to_tuple(item) for item in data]
    else:
        return data


@lru_cache(maxsize=64)
def get_solve_map(ship_type: str, solve_type: Optional[str] = None):
    """
    Loads the solve map for a specific ship type from its JSON file.
    - If solve_type is specified, it looks for a solve with a matching key.
    - If solve_type is None, it looks for a solve that is not nested under a type key.
    """
    file_path = os.path.join(DATA_DIR, "solves", f"{ship_type}.json")

    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        processed_data = {}
        for tech_name, tech_info in data.items():
            solve_data_to_use = None

            # Case 1: The tech_info itself is the solve (flat structure, e.g., "infra" in corvette.json)
            # This is the default solve when no solve_type is requested.
            if "map" in tech_info and "score" in tech_info:
                if solve_type is None:
                    solve_data_to_use = tech_info

            # Case 2: The tech_info contains multiple solves keyed by type (e.g., "cyclotron" in corvette.json)
            elif solve_type in tech_info:
                solve_data_to_use = tech_info[solve_type]

            if solve_data_to_use and 'map' in solve_data_to_use:
                processed_data[tech_name] = {
                    'map': _convert_map_keys_to_tuple(solve_data_to_use['map']),
                    'score': solve_data_to_use.get('score', 0)
                }

        return processed_data
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Error loading or parsing solve map for {ship_type}: {e}")
        return {}


@lru_cache(maxsize=32)
def get_module_data(ship_type: str):
    """
    Loads the module data for a specific ship type from its JSON file.
    Results are cached to avoid repeated file I/O.
    """
    file_path = os.path.join(DATA_DIR, "modules_data", f"{ship_type}.json")

    if not os.path.exists(file_path):
        # Return an empty dict if module data for the ship type doesn't exist
        return {}

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Error loading or parsing module data for {ship_type}: {e}")
        # Return an empty dict on error to prevent crashes
        return {}


def get_all_solve_data():
    """
    Loads all solve data from all JSON files in the solves directory.
    This is used for testing purposes.
    """
    all_solves = {}
    solves_dir = os.path.join(DATA_DIR, "solves")

    if not os.path.exists(solves_dir):
        return {}

    for filename in os.listdir(solves_dir):
        if filename.endswith(".json"):
            ship_type = filename.replace(".json", "")
            file_path = os.path.join(solves_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # We need to process the loaded data to convert map keys back to tuples
                    processed_data = {}
                    for tech_name, tech_info in data.items():
                        if 'map' in tech_info:
                            tech_info['map'] = _convert_map_keys_to_tuple(tech_info['map'])
                        processed_data[tech_name] = tech_info
                    all_solves[ship_type] = processed_data
            except (IOError, json.JSONDecodeError) as e:
                logging.error(f"Error loading or parsing solve data from {filename}: {e}")
                continue

    return all_solves


def get_all_module_data():
    """
    Loads all module data from all JSON files in the modules_data directory.
    This is used for endpoints that need to list all available platforms.
    """
    all_modules = {}
    modules_dir = os.path.join(DATA_DIR, "modules_data")

    if not os.path.exists(modules_dir):
        return {}

    for filename in os.listdir(modules_dir):
        if filename.endswith(".json"):
            ship_type = filename.replace(".json", "")
            file_path = os.path.join(modules_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    all_modules[ship_type] = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logging.error(f"Error loading or parsing module data from {filename}: {e}")
                continue

    return all_modules


def get_training_module_ids(ship_key: str, tech_key: str, solve_type: Optional[str] = None) -> List[str]:
    """
    Gets the list of module IDs for a given technology.

    This function defines the modules that a model should be trained on.
    It first checks for a specific override in `modules_for_training.py`.
    If no override exists, it falls back to the main module data, matching based on `solve_type`.

    Args:
        ship_key: The key for the ship/platform (e.g., "standard").
        tech_key: The key for the technology (e.g., "pulse").
        solve_type: The specific solve type (e.g., "max") or None for the default.

    Returns:
        A list of module ID strings for training.
    """
    # 1. Check for a specific override in modules_for_training.py
    if ship_key in MODULES_FOR_TRAINING and tech_key in MODULES_FOR_TRAINING[ship_key]:
        override_modules = MODULES_FOR_TRAINING[ship_key][tech_key]
        logging.debug(f"get_training_module_ids: Found override for {ship_key}/{tech_key}, returning {len(override_modules)} modules.")
        return override_modules

    # 2. If no override, load from the main JSON data
    module_data = get_module_data(ship_key)
    if not module_data:
        logging.warning(f"get_training_module_ids: No module data found for ship_key '{ship_key}'.")
        return []

    # 3. Find all candidate technology definitions for the given tech_key
    candidates_for_tech = []
    for tech_list in module_data.get("types", {}).values():
        for tech_data in tech_list:
            if tech_data.get("key") == tech_key:
                candidates_for_tech.append(tech_data)

    if not candidates_for_tech:
        logging.warning(f"get_training_module_ids: No technology found with key '{tech_key}' for ship '{ship_key}'.")
        return []

    # 4. Select the correct candidate based on solve_type
    selected_tech_data = None
    for candidate in candidates_for_tech:
        if candidate.get("type") == solve_type:
            selected_tech_data = candidate
            break

    # If no exact match was found, and solve_type is None, this implies we are looking for the "default"
    # untyped technology. This maintains the principle that `None` should not arbitrarily match "normal".
    if selected_tech_data is None and solve_type is None:
        for candidate in candidates_for_tech:
            if candidate.get("type") is None:
                selected_tech_data = candidate
                break

    # 5. Extract module IDs from the selected technology data
    if selected_tech_data:
        modules = selected_tech_data.get("modules", [])
        module_ids = [m['id'] for m in modules]
        logging.debug(f"get_training_module_ids: Returning {len(module_ids)} modules from main JSON for {ship_key}/{tech_key} (solve_type: {solve_type})")
        return module_ids

    logging.warning(f"get_training_module_ids: No matching module list found for {ship_key}/{tech_key} with solve_type '{solve_type}'.")
    return []
