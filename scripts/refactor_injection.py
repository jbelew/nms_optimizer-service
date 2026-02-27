import os
import re
import json

DATA_DIR = "src/data_definitions/modules_data"

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove window_profile lines
    content = re.sub(r'^[ \t]*"window_profile"\s*:\s*".*?"\s*,?\n', '', content, flags=re.MULTILINE)
    # Remove window_overrides lines
    content = re.sub(r'^[ \t]*"window_overrides"\s*:\s*\{.*?\}\s*,?\n', '', content, flags=re.MULTILINE)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# Flat mapping logic: if not exact, uses the next largest key. So we define the threshold maxes.
TECH_OVERRIDES = {
    "hyper": {
        "11": [4, 3],
        "default": [4, 4]
    },
    "pulse-spitter": {
        "6": [3, 3],
        "9": None,
        "default": [4, 2]
    },
    "pulse": {
        "9": None
    }
}

SHIP_TECH_OVERRIDES = {
    "sentinel": {
        "photonix": {"default": [4, 3]},
        "pulse": {
            "7": [4, 2],
            "8": [4, 2]
        }
    },
    "corvette": {
        "pulse": {
            "9": None
        }
    }
}

def inject():
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"): continue
        filepath = os.path.join(DATA_DIR, filename)
        ship_id = filename.replace(".json", "")

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        current_tech = None
        
        for line in lines:
            if '"key":' in line:
                parts = line.split('"key":')
                if len(parts) > 1:
                    val_part = parts[1].strip()
                    if val_part.startswith('"'):
                        current_tech = val_part.split('"')[1]
                        
            if '"modules": [' in line and current_tech is not None:
                overrides = None
                if ship_id in SHIP_TECH_OVERRIDES and current_tech in SHIP_TECH_OVERRIDES[ship_id]:
                    overrides = SHIP_TECH_OVERRIDES[ship_id][current_tech]
                elif current_tech in TECH_OVERRIDES:
                    overrides = TECH_OVERRIDES[current_tech]
                
                if overrides:
                    indent_match = len(line) - len(line.lstrip())
                    base_indent = " " * indent_match
                    overrides_json = json.dumps(overrides)
                    new_lines.append(f'{base_indent}"window_overrides": {overrides_json},\n')
                    
                current_tech = None
                
            new_lines.append(line)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

if __name__ == "__main__":
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            clean_file(os.path.join(DATA_DIR, filename))
    inject()
    print("Done cleaning and injecting flat overrides.")
