"""
This script provides utility functions to process and reformat module definition
JSON files. It is designed to be run as a standalone script to perform bulk
updates on the module data.

The main operations include:
- Setting a 'checked' status based on the module's original type.
- Re-categorizing modules based on their labels (e.g., 'Theta' -> 'upgrade').
- Persisting the changes back to the JSON files with custom formatting.
"""

import json
import os


def process_module(module):
    """Applies formatting and type changes to a single module dictionary.

    This function modifies the module dictionary in-place.

    Args:
        module (dict): The module dictionary to process.
    """
    # First, determine the 'checked' status based on the original type.
    is_reward = module.get("type") == "reward"

    if is_reward:
        module["checked"] = False
    else:
        module["checked"] = True

    # Second, update the type.
    label = module.get("label", "")
    if "Theta" in label or "Sigma" in label or "Tau" in label:
        module["type"] = "upgrade"
    elif is_reward:  # Use the stored boolean here to correctly identify original rewards
        module["type"] = "bonus"


def find_and_process_modules(data):
    """Recursively finds and processes 'modules' lists within a JSON structure.

    This function traverses the data (which can be a dict or a list) and
    applies the `process_module` function to each item in any list found
    under the key "modules".

    Args:
        data (dict or list): The JSON data to traverse.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "modules" and isinstance(value, list):
                for module in value:
                    process_module(module)
            else:
                find_and_process_modules(value)
    elif isinstance(data, list):
        for item in data:
            find_and_process_modules(item)


def custom_json_dump(data, f, level=0):
    """Writes a dictionary to a file with custom JSON formatting.

    This function formats the JSON to have one module object per line for
    better readability and easier diffing in version control.

    Args:
        data (dict): The dictionary to write.
        f (file object): The file to write to.
        level (int, optional): The current indentation level for recursion.
            Defaults to 0.
    """
    indent = "    "
    f.write("{\n")
    items = list(data.items())
    for i, (key, value) in enumerate(items):
        f.write(indent * (level + 1))
        f.write(f'"{key}": ')
        if key == "modules" and isinstance(value, list):
            f.write("[\n")
            for mod_idx, module in enumerate(value):
                f.write(indent * (level + 2))
                f.write(json.dumps(module, separators=(", ", ": ")))
                if mod_idx < len(value) - 1:
                    f.write(",")
                f.write("\n")
            f.write(indent * (level + 1))
            f.write("]")
        elif isinstance(value, dict):
            custom_json_dump(value, f, level + 1)
        elif isinstance(value, list):
            f.write("[\n")
            for list_idx, item in enumerate(value):
                f.write(indent * (level + 2))
                if isinstance(item, dict):
                    custom_json_dump(item, f, level + 2)
                else:
                    f.write(json.dumps(item))

                if list_idx < len(value) - 1:
                    f.write(",")
                f.write("\n")
            f.write(indent * (level + 1))
            f.write("]")
        else:
            f.write(json.dumps(value))

        if i < len(items) - 1:
            f.write(",")
        f.write("\n")
    f.write(indent * level)
    f.write("}")


def update_module_files():
    """Main function to read, process, and rewrite all module JSON files.

    This function iterates through all `.json` files in the
    `src/data_definitions/modules_data/` directory, applies the processing
    logic, and saves the files with the new formatting.
    """
    directory = "src/data_definitions/modules_data/"
    files = os.listdir(directory)
    files.sort()

    for filename in files:
        if filename == "corvette.json":
            continue

        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            data = None
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {filepath}: {e}")
                    continue

            if data:
                find_and_process_modules(data)

                with open(filepath, "w", encoding="utf-8") as f:
                    custom_json_dump(data, f)


if __name__ == "__main__":
    update_module_files()
    print("Bulk update of module JSON files complete.")
