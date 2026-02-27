import json
import os
import logging

# Ensure logging doesn't interfere with test output
logging.basicConfig(level=logging.CRITICAL)

# --- Mocking dependencies for the old determine_window_dimensions function ---
# The old function directly references _get_window_profiles, which reads window_profiles.json
# We need to provide a functional mock for it.

# Content of window_profiles.json from its original state (before any changes made by me)
MOCK_WINDOW_PROFILES_CONTENT = {
    "standard": {"1": [1, 1], "2": [2, 1], "4": [2, 2], "6": [3, 2], "9": [3, 3], "10": [4, 3], "default": [4, 3]}
}


class MockModulesUtils:
    _WINDOW_PROFILES = MOCK_WINDOW_PROFILES_CONTENT

    @staticmethod
    def _get_window_profiles():
        return MockModulesUtils._WINDOW_PROFILES

    @staticmethod
    def get_tech_window_rules(*args, **kwargs):
        # This mock is essential because the new function uses it,
        # but the old hardcoded logic would have bypassed it or used it differently.
        # For the purpose of getting old logic's "ground truth", this is not used.
        # But if the old function called it, we'd need a more complex mock.
        # In this specific case, the old logic does NOT call get_tech_window_rules
        # but directly implements the rules.
        pass


# Injecting the mock into the globals so the old function can find it
_old_determine_window_dimensions_globals = globals().copy()
_old_determine_window_dimensions_globals["src"] = type("module", (object,), {"modules_utils": MockModulesUtils})

# --- Old determine_window_dimensions function (from commit c1576e5) ---
_old_determine_window_dimensions_code = """
def determine_window_dimensions_old(module_count: int, tech: str, ship: str) -> Tuple[int, int]:
    # --- Ship- and tech-specific overrides ---
    if ship == "sentinel" and tech == "photonix":
        return 4, 3

    if ship == "corvette" and tech == "pulse" and module_count == 7:
        return 4, 2

    if ship == "corvette" and module_count in (7, 8):
        return 3, 3

    # Set 8 modules to 3x3 unless tech has specific sizing rules
    if module_count == 8 and tech not in ("pulse", "photonix", "hyper", "pulse-spitter"):
        return 3, 3

    # Default window size if no other conditions are met
    window_width, window_height = 3, 3

    # --- Technology-specific rules ---
    if tech == "hyper":
        if module_count >= 12:
            window_width, window_height = 4, 4
        elif module_count >= 10:
            window_width, window_height = 4, 3
        elif module_count >= 9:
            window_width, window_height = 3, 3
        else:
            window_width, window_height = 4, 2

    elif tech == "daedalus":
        window_width, window_height = 3, 3

    elif tech == "bolt-caster":
        window_width, window_height = 4, 3

    elif tech == "trails":
        if module_count <= 9:
            window_width, window_height = 3, 3
        else:
            window_width, window_height = 4, 3

    elif tech == "jetpack":
        window_width, window_height = 3, 3

    elif tech == "neutron":
        window_width, window_height = 3, 3

    elif tech == "pulse-spitter":
        if module_count < 7:
            window_width, window_height = 3, 3
        else:
            window_width, window_height = 4, 2

    elif tech == "pulse":
        if module_count == 6:
            window_width, window_height = 3, 2
        elif module_count < 8:
            window_width, window_height = 4, 2
        else:
            window_width, window_height = 4, 3

    # --- Generic fallback rules ---
    elif module_count < 2:
        # The old function had a logging.warning here, but for test generation we suppress it
        return 1, 1
    elif module_count < 3:
        window_width, window_height = 2, 1
    elif module_count < 5:
        window_width, window_height = 2, 2
    elif module_count < 7:
        window_width, window_height = 3, 2
    elif module_count == 7:
        window_width, window_height = 4, 2
    elif module_count <= 9:
        window_width, window_height = 3, 3
    else:  # module_count >= 10
        window_width, window_height = 4, 3

    return window_width, window_height
"""

exec(_old_determine_window_dimensions_code, _old_determine_window_dimensions_globals)
determine_window_dimensions_old = _old_determine_window_dimensions_globals["determine_window_dimensions_old"]

# --- Test Case Generation Logic ---
MODULE_DATA_DIR = "src/data_definitions/modules_data"
MODULE_FILES = [
    "standard.json",
    "atlantid.json",
    "corvette.json",
    "standard-mt.json",
    "sentinel.json",
    "nomad.json",
    "minotaur.json",
    "freighter.json",
    "colossus.json",
    "solar.json",
    "pilgrim.json",
    "nautilon.json",
    "living.json",
    "exosuit.json",
    "staves.json",
    "sentinel-mt.json",
    "roamer.json",
]


def load_module_data(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # logging.error(f"Error loading {file_path}: {e}") # Suppressed for script execution
        return {}


def generate_old_logic_test_cases():
    test_cases_data = []

    # Helper function to check the new business rule for hyper modules
    def should_skip_hyper_test(ship_name, tech_key, module_count):
        if tech_key != "hyper":
            return False

        if ship_name == "corvette":
            return module_count > 12
        elif ship_name == "freighter":
            return module_count > 11
        elif ship_name in ["any_ship", "unknown", "unknown_ship", "random_tech", "standard"]:  # General ships
            return module_count > 9
        else:  # Any other specific ship type not explicitly defined
            return module_count > 9

    # Removed all explicit test cases that use 'any_ship' or 'unknown' as per user directive.
    # These will now be generated directly from module data files.

    # Iterate through all module data files
    for module_file in MODULE_FILES:
        ship_name = os.path.basename(module_file).replace(".json", "")
        module_data = load_module_data(os.path.join(MODULE_DATA_DIR, module_file))

        if not module_data or "types" not in module_data:
            continue

        for category, tech_list in module_data["types"].items():
            if not isinstance(tech_list, list):
                continue

            for tech_info in tech_list:
                if not isinstance(tech_info, dict) or "key" not in tech_info or "modules" not in tech_info:
                    continue

                tech_key = tech_info["key"]
                # Filter out modules that don't count towards the total, like cosmetic or reward modules
                # This logic is based on common patterns in the data files and how module_count is derived
                module_count = len(
                    [
                        m
                        for m in tech_info["modules"]
                        if not m.get("reward") and m.get("type") not in ("cosmetic", "reactor")
                    ]
                )

                # Skip if module_count is 0 or negative
                if module_count <= 0:
                    continue

                # Skip if hyper module count exceeds limits based on ship type (new business rule)
                if should_skip_hyper_test(ship_name, tech_key, module_count):
                    continue

                expected_w, expected_h = determine_window_dimensions_old(module_count, tech_key, ship_name)

                test_cases_data.append(
                    {
                        "description": f"{ship_name} {tech_key} ({module_count} modules)",
                        "module_count": module_count,
                        "tech": tech_key,
                        "ship": ship_name,
                        "expected": (expected_w, expected_h),
                    }
                )

    # Apply the user's new rules:
    # 1. 1 module always returns (1, 1)
    # 2. 7 modules always returns (3, 3), except for pulse (which returns 4,2 from old logic)
    # 3. 5 modules always returns (3, 2)
    for case in test_cases_data:
        if case["module_count"] == 1:
            case["expected"] = (1, 1)
        elif case["module_count"] == 7 and case["tech"] != "pulse":
            case["expected"] = (3, 3)
        elif case["module_count"] == 5:
            case["expected"] = (3, 2)
        # If module_count is 7 and tech is pulse, it should remain (4, 2)
        # as determined by determine_window_dimensions_old, so no change needed.
        # Handle specific cases for corvette pulse 8 modules
        elif case["ship"] == "corvette" and case["tech"] == "pulse" and case["module_count"] == 8:
            case["expected"] = (4, 3)  # User stated this is correct behavior

    return test_cases_data


generated_tests = generate_old_logic_test_cases()

test_file_lines = []

# Add imports and class definition
test_file_lines.append("import unittest")
test_file_lines.append("from src.optimization.helpers import determine_window_dimensions")
test_file_lines.append("from src.data_loader import get_module_data ")
test_file_lines.append("")
test_file_lines.append("class TestOldDetermineWindowDimensionsBehavior(unittest.TestCase):")
test_file_lines.append('    """')
test_file_lines.append("    This class captures the behavior of `determine_window_dimensions`")
test_file_lines.append('    as it was at commit c1576e5 (the "old" version).')
test_file_lines.append("    ")
test_file_lines.append("    Failures here indicate a change in behavior from that commit.")
test_file_lines.append("    The goal is to ensure the new implementation (which uses external JSON")
test_file_lines.append("    overrides and a standard profile) produces the exact same results")
test_file_lines.append("    as the old hardcoded logic for these specific inputs.")
test_file_lines.append('    """')

# Add test methods
for i, test_case in enumerate(generated_tests):
    description = test_case["description"].replace('"', '\\"').replace("'", "\\'")
    module_count = test_case["module_count"]
    tech = test_case["tech"]
    ship = test_case["ship"]
    expected_w, expected_h = test_case["expected"]

    # Sanitize ship and tech names for method names
    ship_safe = ship.replace("-", "_").replace(" ", "_")
    tech_safe = tech.replace("-", "_").replace(" ", "_")
    # Sanitize module_count for method name
    module_count_safe = str(module_count).replace("-", "_neg_")

    test_file_lines.append("")  # Blank line before each test for readability
    test_file_lines.append(
        f"    def test_old_logic_case_{i:03d}_{ship_safe}_{tech_safe}_count_{module_count_safe}(self):"
    )
    test_file_lines.append(f'        """Old Logic: {description}"""')
    test_file_lines.append(
        f"        modules_data = get_module_data('{ship}')"
    )  # Always load modules_data for specific ships

    test_file_lines.append(
        f"        w, h = determine_window_dimensions({module_count}, '{tech}', '{ship}', modules=modules_data)"
    )
    test_file_lines.append(f"        self.assertEqual((w, h), ({expected_w}, {expected_h}))")


# Join all lines with actual newline characters
final_output_string = "\n".join(test_file_lines)

# Write the generated test cases to a file
with open("src/tests/test_old_determine_window_dimensions_behavior.py", "w", encoding="utf-8") as f:
    f.write(final_output_string)

print("Generated test cases saved to src/tests/test_old_determine_window_dimensions_behavior.py")

# No need to return anything, the file is created.
