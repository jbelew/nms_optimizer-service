import os
import re

DATA_DIR = "src/data_definitions/modules_data"


def fix_json_quotes(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # The broken format looks like "window_overrides": {'exact': {'7': [3, 3], '8': [3, 3]}},
    # We want to replace all single quotes with double quotes within the window_overrides value.
    # A simple regex to find the dictionary string and replace ' with "

    def replacer(match):
        dict_str = match.group(1)
        fixed_dict_str = dict_str.replace("'", '"')
        return f'"window_overrides": {fixed_dict_str},'

    # Match "window_overrides": {...},
    fixed_content = re.sub(r'"window_overrides":\s*(\{.*?\})\s*,', replacer, content)

    # We also have \n instead of actual newlines injected by the script because I wrote `\\n`
    # Let's fix that too
    fixed_content = fixed_content.replace("\\n                ", "\n                ")

    if content != fixed_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(fixed_content)
        print(f"Fixed {filepath}")


for filename in os.listdir(DATA_DIR):
    if filename.endswith(".json"):
        fix_json_quotes(os.path.join(DATA_DIR, filename))
