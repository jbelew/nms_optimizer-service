import json
import os

DATA_DIR = "src/data_definitions/modules_data"

def get_window_profile_and_overrides(ship: str, tech: str):
    profile = tech
    # Fallback any unlisted tech to "standard"
    if profile not in ["hyper", "daedalus", "bolt-caster", "trails", "jetpack", "neutron", "pulse-spitter", "pulse"]:
        profile = "standard"
        
    overrides = None
    
    # 1. Ship + tech specific overrides
    if ship == "sentinel" and tech == "photonix":
        # Usually photonix falls under standard
        profile = "standard"
        overrides = {"default": [4, 3]}
        
    # 2. Ship-level overrides
    if ship == "corvette":
        if tech == "pulse":
            overrides = {
                "exact": { "7": [4, 2], "6": [3, 2] },
                "max": { "7": [4, 2] },
                "default": [4, 3]
            }
        elif tech not in ("pulse", "photonix", "hyper", "pulse-spitter"):
            overrides = {
                "exact": {"7": [3, 3], "8": [3, 3]}
            }
            
    # Generic module count overrides
    if not overrides and tech not in ("pulse", "photonix", "hyper", "pulse-spitter"):
        overrides = {
            "exact": {"8": [3, 3]}
        }
        
    return profile, overrides

def process_file(filepath: str):
    filename = os.path.basename(filepath)
    ship = filename.replace(".json", "")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # check if already processed
    for line in lines:
        if '"window_profile":' in line:
            print(f"Skipping {ship}, already processed.")
            return

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
            profile, overrides = get_window_profile_and_overrides(ship, current_tech)
            
            indent_match = len(line) - len(line.lstrip())
            base_indent = " " * indent_match
            
            profile_line = f'{base_indent}"window_profile": "{profile}",\\n'
            new_lines.append(profile_line)
            
            if overrides:
                overrides_json = json.dumps(overrides)
                overrides_line = f'{base_indent}"window_overrides": {overrides_json},\\n'
                new_lines.append(overrides_line)
                
            current_tech = None
            
        new_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Updated {ship}")

if __name__ == "__main__":
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            process_file(os.path.join(DATA_DIR, filename))
