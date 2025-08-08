# model_mapping.py
from typing import (
    Optional,
    List,
    Dict,
)  # Removed Tuple as it's no longer used for return type

"""
Maps the user-facing platform key (ship) and technology key (tech)
to the corresponding keys used internally to identify the correct
trained model file, potentially using different tech keys based on reward ownership.
...
"""

# --- Mapping Dictionary (Base mappings) ---
PLATFORM_TECH_TO_MODEL_KEYS = {
    # --- Starships ---
    "standard": {
        "cyclotron": ("standard", "cyclotron"),
        "infra": ("standard", "infra"),
        "phase": ("standard", "phase"),
        "positron": ("standard", "positron"),
        "rocket": ("standard", "rocket"),
        "photon": ("standard", "photon"),
        "shield": ("standard", "shield"),
        "launch": ("standard", "launch"),
        "hyper": ("standard", "hyper"),
        "pulse": ("standard", "pulse"),
        "photonix": ("standard", "photonix"),
        "trails": ("standard", "trails"),
        "aqua": ("standard", "aqua"),
        "bobble": ("standard", "bobble"),
        "scanners": ("standard", "scanners"),
        "teleporter": ("standard", "teleporter"),
    },
    "sentinel": {
        "cyclotron": ("standard", "cyclotron"),
        "infra": ("standard", "infra"),
        "phase": ("standard", "phase"),
        "positron": ("standard", "positron"),
        "rocket": ("standard", "rocket"),
        "photon": ("standard", "photon"),
        "shield": ("standard", "shield"),
        "launch": ("standard", "launch"),
        "hyper": ("standard", "hyper"),
        "pulse": ("standard", "pulse"),
        "photonix": ("standard", "photonix"),
        "trails": ("standard", "trails"),
        "aqua": ("standard", "aqua"),
        "bobble": ("standard", "bobble"),
        "scanners": ("standard", "scanners"),
        "teleporter": ("standard", "teleporter"),
        "pilot": ("sentinel", "pilot"),
    },
    "solar": {
        "cyclotron": ("standard", "cyclotron"),
        "infra": ("standard", "infra"),
        "phase": ("standard", "phase"),
        "positron": ("standard", "positron"),
        "rocket": ("standard", "rocket"),
        "photon": ("standard", "photon"),
        "shield": ("standard", "shield"),
        "launch": ("standard", "launch"),
        "hyper": ("standard", "hyper"),
        "trails": ("standard", "trails"),
        "aqua": ("standard", "aqua"),
        "bobble": ("standard", "bobble"),
        "scanners": ("standard", "scanners"),
        "teleporter": ("standard", "teleporter"),
        "pilot": ("sentinel", "pilot"),
    },
    # --- Multi-Tools ---
    "sentinel-mt": {
        "mining": ("atlantid", "mining"),
        "analysis": ("standard-mt", "analysis"),
        "scanner": ("standard-mt", "scanner"),
        "survey": ("standard-mt", "survey"),
        "blaze-javelin": ("standard-mt", "blaze-javelin"),
        "bolt-caster": ("standard-mt", "bolt-caster"),
        "neutron": ("standard-mt", "neutron"),
        "plasma-launcher": ("standard-mt", "plasma-launcher"),
        "pulse-spitter": ("standard-mt", "pulse-spitter"),
        "scatter": ("standard-mt", "scatter"),
        "cloaking": ("standard-mt", "cloaking"),
        "combat": ("standard-mt", "combat"),
        "voltaic-amplifier": ("standard-mt", "voltaic-amplifier"),
        "paralysis": ("standard-mt", "paralysis"),
        "personal": ("standard-mt", "personal"),
        "fishing": ("standard-mt", "fishing"),
        "forbidden": ("standard-mt", "forbidden"),
        "terrian": ("standard-mt", "terrian"),
    },
    "staves": {
        "mining": ("standard-mt", "mining"),
        "analysis": ("standard-mt", "analysis"),
        "scanner": ("standard-mt", "scanner"),
        "survey": ("standard-mt", "survey"),
        "blaze-javelin": ("standard-mt", "blaze-javelin"),
        "bolt-caster": ("standard-mt", "bolt-caster"),
        "neutron": ("standard-mt", "neutron"),
        "plasma-launcher": ("standard-mt", "plasma-launcher"),
        "pulse-spitter": ("standard-mt", "pulse-spitter"),
        "scatter": ("standard-mt", "scatter"),
        "cloaking": ("standard-mt", "cloaking"),
        "combat": ("standard-mt", "combat"),
        "voltaic-amplifier": ("standard-mt", "voltaic-amplifier"),
        "paralysis": ("standard-mt", "paralysis"),
        "personal": ("standard-mt", "personal"),
        "fishing": ("standard-mt", "fishing"),
        "forbidden": ("standard-mt", "forbidden"),
        "terrian": ("standard-mt", "terrian"),
    },
    "atlantid": {
        "mining": ("atlantid", "mining"),
        "analysis": ("standard-mt", "analysis"),
        "scanner": ("standard-mt", "scanner"),
        "survey": ("standard-mt", "survey"),
        "blaze-javelin": ("standard-mt", "blaze-javelin"),
        "bolt-caster": ("standard-mt", "bolt-caster"),
        "neutron": ("standard-mt", "neutron"),
        "plasma-launcher": ("standard-mt", "plasma-launcher"),
        "pulse-spitter": ("standard-mt", "pulse-spitter"),
        "scatter": ("standard-mt", "scatter"),
        "cloaking": ("standard-mt", "cloaking"),
        "combat": ("standard-mt", "combat"),
        "voltaic-amplifier": ("standard-mt", "voltaic-amplifier"),
        "paralysis": ("standard-mt", "paralysis"),
        "personal": ("standard-mt", "personal"),
        "fishing": ("standard-mt", "fishing"),
        "forbidden": ("standard-mt", "forbidden"),
        "terrian": ("standard-mt", "terrian"),
    },
    # --- Living Ship (No mapping needed, uses its own keys) ---
    "living": {},
    # --- Standard Multi-Tool (No mapping needed) ---
    "standard-mt": {},
}

# <<< Remove _find_reward_module_id and suffix logic >>>


def get_model_keys(
    ui_ship_key: str,
    ui_tech_key: str,
    grid_width: int,
    grid_height: int,
    player_owned_rewards: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Determines keys for model filename and module definitions.
    1. Establishes initial base keys from UI keys and PLATFORM_TECH_TO_MODEL_KEYS.
       These serve as the primary keys for module definitions.
    2. Applies reward-based overrides (e.g., "pulse" + "PC" -> "photonix"),
       updating both module definition tech key and filename tech key.
    3. Applies grid-size specific overrides, which primarily modify the
       filename keys, while the module definition keys remain from step 2.

    Args:
        ui_ship_key: The ship key provided by the user/UI.
        ui_tech_key: The tech key provided by the user/UI.
        grid_width: The expected grid width for this tech (from determine_window_dimensions).
        grid_height: The expected grid height for this tech (from determine_window_dimensions).
        player_owned_rewards: List of reward module IDs owned by the player.

    Returns:
        A dictionary with keys:
            "filename_ship_key": Key for the ship part of the model filename.
            "filename_tech_key": Key for the tech part of the model filename.
            "module_def_ship_key": Key for ship lookup in module definitions.
            "module_def_tech_key": Key for tech lookup in module definitions.
    """
    player_rewards_set = set(player_owned_rewards) if player_owned_rewards else set()

    # --- Step 1: Determine initial base keys (primarily for module definitions) ---
    # These will also be the starting point for filename keys.
    if ui_ship_key in PLATFORM_TECH_TO_MODEL_KEYS:
        initial_model_ship_key, initial_model_tech_key = PLATFORM_TECH_TO_MODEL_KEYS[
            ui_ship_key
        ].get(ui_tech_key, (ui_ship_key, ui_tech_key))
    else:
        initial_model_ship_key, initial_model_tech_key = ui_ship_key, ui_tech_key

    # Initialize all key types from these base keys
    module_def_ship_key = initial_model_ship_key
    module_def_tech_key = initial_model_tech_key
    filename_ship_key = initial_model_ship_key
    filename_tech_key = initial_model_tech_key

    # --- Step 2: Apply reward-based overrides ---
    # This affects both module definition tech key and filename tech key.
    if ui_tech_key == "pulse" and "PC" in player_rewards_set:
        # If Photonix Core is owned, "pulse" tech effectively becomes "photonix".
        # The ship key (e.g., "standard") remains as determined by initial_model_ship_key.
        module_def_tech_key = "photonix"
        filename_tech_key = "photonix"

    # --- Step 3: Apply grid-size specific overrides ---
    # These conditions use the module_def_ship_key and module_def_tech_key (which may have been
    # updated by reward logic) to decide if FILENAME keys need further specialization.
    # The module_def keys themselves are NOT changed by this step.
    if (
        module_def_ship_key == "standard"
        and module_def_tech_key == "photonix"
        and grid_width == 4
        and grid_height == 3
    ):
        # This specific model filename is "model_sentinel_photonix_4x3.pth"
        # but it's trained on "standard" "photonix" modules.
        filename_ship_key = "standard"
        filename_tech_key = "photonix_4x3"

    if (
        module_def_ship_key == "solar"
        and module_def_tech_key == "pulse"
        and grid_width == 4
        and grid_height == 3
    ):
        # This specific model filename is "model_sentinel_photonix_4x3.pth"
        # but it's trained on "standard" "photonix" modules.
        filename_ship_key = "solar"
        filename_tech_key = "pulse_4x3"

    if (
        module_def_ship_key == "solar"
        and module_def_tech_key == "photonix"
        and grid_width == 4
        and grid_height == 3
    ):
        # This specific model filename is "model_sentinel_photonix_4x3.pth"
        # but it's trained on "standard" "photonix" modules.
        filename_ship_key = "solar"
        filename_tech_key = "photonix_4x3"

    # Add other grid-size specific conditions here, for example:
    # if model_ship_key == "some_ship" and model_tech_key == "some_tech" and grid_width == X and grid_height == Y:
    #     return "specific_model_ship_for_size", "specific_model_tech_for_size"

    # --- Step 4: Return the (potentially modified) model keys ---
    return {
        "filename_ship_key": filename_ship_key,
        "filename_tech_key": filename_tech_key,
        "module_def_ship_key": module_def_ship_key,
        "module_def_tech_key": module_def_tech_key,
    }
