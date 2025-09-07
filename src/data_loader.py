import json
import logging
import os
from functools import lru_cache
from typing import List, Union

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
def get_solve_map(ship_type: str, solve_type: Union[str, None] = None):
    """
    Loads the solve map for a specific ship type from its JSON file.
    If solve_type is specified, it will be used. Otherwise, it defaults to "normal".
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
            # Determine the key to look for in the tech_info dictionary
            key_to_find = solve_type if solve_type else "normal"

            # Check if the tech_info itself contains the map and score (flat structure)
            if "map" in tech_info and "score" in tech_info:
                solve_data_to_use = tech_info
            elif key_to_find in tech_info:
                solve_data_to_use = tech_info[key_to_find]

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


def get_training_module_ids(ship_key: str, tech_key: str) -> List[str]:
    """
    Gets the list of module IDs for a given technology, using the specific
    training definition if available, otherwise falling back to the main
    module data. This is used to define the ML model architecture.

    Args:
        ship_key: The key for the ship/platform (e.g., "standard").
        tech_key: The key for the technology (e.g., "pulse").

    Returns:
        A list of module ID strings.
    """
    # 1. Check for a specific override in modules_for_training.py
    if ship_key in MODULES_FOR_TRAINING and tech_key in MODULES_FOR_TRAINING[ship_key]:
        return MODULES_FOR_TRAINING[ship_key][tech_key]

    # 2. If no override, load from the main JSON data and extract the IDs
    module_data = get_module_data(ship_key)
    if not module_data:
        return []

    # This logic is adapted from modules_utils.get_tech_modules_for_training
    types_data = module_data.get("types", {})
    for tech_list in types_data.values():
        for tech_data in tech_list:
            if tech_data.get("key") == tech_key:
                modules = tech_data.get("modules", [])
                # Return a list of just the IDs. For technologies not explicitly
                # defined in MODULES_FOR_TRAINING, we assume the training set
                # included all modules listed in the main data file.
                return [m['id'] for m in modules]

    return []  # Return empty list if tech_key not found
