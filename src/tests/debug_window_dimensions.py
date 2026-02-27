
import json
import os
import logging
from typing import Optional, Tuple

# Set up basic logging to capture warnings if any
logging.basicConfig(level=logging.INFO)

# --- Mocking environment similar to actual application ---
# We need _get_window_profiles and get_tech_window_rules to accurately simulate.

# Mimic src/modules_utils.py's _get_window_profiles
def _get_window_profiles_diagnostic():
    current_dir = os.path.dirname(__file__)
    profiles_path = os.path.join(current_dir, "../data_definitions", "window_profiles.json")
    if not os.path.exists(profiles_path):
        logging.error(f"Diagnostic: window_profiles.json not found at {profiles_path}")
        return {}
    with open(profiles_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Mimic src/modules_utils.py's get_tech_window_rules
def get_tech_window_rules_diagnostic(modules: dict, ship: str, tech_key: str) -> dict:
    profiles = _get_window_profiles_diagnostic()
    rules = {}

    # 1. ALWAYS start with the 'standard' profile as the base foundation
    if "standard" in profiles:
        rules = json.loads(json.dumps(profiles["standard"]))

    overrides = {}
    if modules and "types" in modules:
        for category in modules["types"].values():
            if isinstance(category, list):
                for tech_info in category:
                    if tech_info.get("key") == tech_key and "window_overrides" in tech_info:
                        overrides = tech_info["window_overrides"]
                        break
            if overrides:
                break
    
    # Simple deep_merge for demonstration
    def deep_merge(target, source):
        for k, v in source.items():
            target[k] = v
    
    # 2. Deep merge any specific explicit overrides from this very JSON block
    deep_merge(rules, overrides)

    return rules


# --- Mimic determine_window_dimensions from src/optimization/helpers.py ---
def determine_window_dimensions_diagnostic(module_count: int, tech: str, ship: str, modules: Optional[dict] = None) -> Tuple[int, int]:
    print(f"--- Diagnosing: Ship={ship}, Tech={tech}, ModuleCount={module_count} ---")

    rules = {}
    if module_count < 1:
        print(f"DIAG: Module count is {module_count}. Returning default 1x1 window.")
        return 1, 1

    if modules is not None:
        rules = get_tech_window_rules_diagnostic(modules, ship, tech)
    
    print(f"DIAG: Initial rules after get_tech_window_rules_diagnostic: {rules}")

    if not rules:
        print(f"DIAG: Rules are empty, falling back to standard from window_profiles.json")
        profiles = _get_window_profiles_diagnostic()
        rules = json.loads(json.dumps(profiles.get("standard", {})))
        print(f"DIAG: Rules after fallback: {rules}")

    count_str = str(module_count)
    print(f"DIAG: Looking for count_str '{count_str}' in rules.")

    if count_str in rules and rules[count_str] is not None:
        print(f"DIAG: Exact match found for '{count_str}': {rules[count_str]}. Returning this.")
        return rules[count_str][0], rules[count_str][1]
        
    int_keys = [int(k) for k in rules.keys() if k.isdigit() and rules[k] is not None]
    print(f"DIAG: Numeric keys in rules (non-None values): {sorted(int_keys)}")

    larger_keys = [k for k in int_keys if k > module_count]
    print(f"DIAG: Larger keys than {module_count}: {sorted(larger_keys)}")
    
    if larger_keys:
        best_key = str(min(larger_keys))
        print(f"DIAG: Smallest larger key found: '{best_key}', value: {rules[best_key]}. Returning this.")
        return rules[best_key][0], rules[best_key][1]
        
    if "default" in rules:
        print(f"DIAG: No larger key found. Returning default: {rules['default']}.")
        return rules["default"][0], rules["default"][1]
        
    print(f"DIAG: Final safety fallback to 1x1.")
    return 1, 1


# --- Simulate the failing 'trails' test case ---
# Ship: standard, Tech: trails, ModuleCount: 12
ship_to_test = "standard"
tech_to_test = "trails"
module_count_to_test = 12

# Load actual module data for 'standard' ship
current_dir = os.path.dirname(__file__)
standard_modules_path = os.path.join(current_dir, "../data_definitions/modules_data", f"{ship_to_test}.json")
standard_modules_data = {}
if os.path.exists(standard_modules_path):
    with open(standard_modules_path, "r", encoding="utf-8") as f:
        standard_modules_data = json.load(f)
else:
    print(f"DIAG ERROR: Module data for '{ship_to_test}' not found at {standard_modules_path}")

print(f"DIAG: Loaded standard_modules_data: {json.dumps(standard_modules_data, indent=2)}")


actual_w, actual_h = determine_window_dimensions_diagnostic(
    module_count_to_test,
    tech_to_test,
    ship_to_test,
    modules=standard_modules_data
)

print(f"DIAG: Final result for {ship_to_test}/{tech_to_test} ({module_count_to_test} modules): ({actual_w}, {actual_h})")

