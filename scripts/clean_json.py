import os
import re

DATA_DIR = "src/data_definitions/modules_data"


def clean_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # We want to remove lines that match exactly:
    # "window_profile": "standard",
    # "window_overrides": {"exact": {"8": [3, 3]}},
    # "window_overrides": {},
    # including the preceding indentation and the newline.

    original_content = content

    # Regex to remove "window_profile": "standard",\n
    content = re.sub(r'^[ \t]*"window_profile": "standard",\n', "", content, flags=re.MULTILINE)

    # Regex to remove "window_overrides": {"exact": {"8": [3, 3]}},\n
    content = re.sub(r'^[ \t]*"window_overrides": \{"exact": \{"8": \[3, 3\]\}\},\n', "", content, flags=re.MULTILINE)

    # Regex to remove "window_overrides": {},\n
    content = re.sub(r'^[ \t]*"window_overrides": \{\},\n', "", content, flags=re.MULTILINE)

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Cleaned {filepath}")


for filename in os.listdir(DATA_DIR):
    if filename.endswith(".json"):
        clean_json_file(os.path.join(DATA_DIR, filename))
