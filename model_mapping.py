# model_mapping.py
from typing import Optional, List, Tuple, Dict # <<< Added Dict, List, Optional

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
    # --- Multi-Tools ---
    "sentinel-mt": {
        "mining": ("standard-mt", "mining"),
        "analysis": ("standard-mt", "analysis"),
        "scanner": ("standard-mt", "scanner"),
        "survey": ("standard-mt", "survey"),
        "blaze-javelin": ("standard-mt", "blaze-javelin"),
        "bolt-caster": ("standard-mt", "bolt-caster"),
        "neutron": ("standard-mt", "neutron"),
        "plasma-launcher": ("standard-mt", "plasma-launcher"),
        "pulse-splitter": ("standard-mt", "pulse-splitter"),
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
    "atlantid-mt": {
        "mining": ("atlantid-mt", "mining"),
        "analysis": ("standard-mt", "analysis"),
        "scanner": ("standard-mt", "scanner"),
        "survey": ("standard-mt", "survey"),
        "blaze-javelin": ("standard-mt", "blaze-javelin"),
        "bolt-caster": ("standard-mt", "bolt-caster"),
        "neutron": ("standard-mt", "neutron"),
        "plasma-launcher": ("standard-mt", "plasma-launcher"),
        "pulse-splitter": ("standard-mt", "pulse-splitter"),
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
    "standard-mt": {}
}

# <<< Remove _find_reward_module_id and suffix logic >>>

def get_model_keys(
    ui_ship_key: str,
    ui_tech_key: str,
    # <<< Add player_owned_rewards >>>
    player_owned_rewards: Optional[List[str]] = None
) -> Tuple[str, str]:
    """
    Looks up the mapping for a given UI ship and tech key.
    Includes specific logic to map to different tech keys based on reward ownership
    (e.g., 'pulse' maps to 'photonix' if 'PC' reward is owned).
    Defaults to returning the original keys if no specific mapping is found.

    Args:
        ui_ship_key: The ship key provided by the user/UI.
        ui_tech_key: The tech key provided by the user/UI.
        player_owned_rewards: List of reward module IDs owned by the player.

    Returns:
        A tuple containing (model_ship_key, model_tech_key) to use for
        loading the model file.
    """
    player_rewards_set = set(player_owned_rewards) if player_owned_rewards else set()

    # --- 1. Specific Reward-Based Mapping Logic ---
    # Example: Pulse Engine / Photonix Core ('PC')
    if ui_tech_key == "pulse" and "PC" in player_rewards_set:
        # If the player has the Photonix Core, map to the 'photonix' model key.
        # We still need the base ship key mapping first.
        base_ship_map = PLATFORM_TECH_TO_MODEL_KEYS.get(ui_ship_key, {})
        base_model_ship_key, _ = base_ship_map.get(ui_tech_key, (ui_ship_key, ui_tech_key)) # Get base ship key
        # Return the base ship key and the specific 'photonix' tech key
        return base_model_ship_key, "photonix"

    # Add other specific reward mappings here if needed following the same pattern
    # Example (if you had a model for mining with plasma resonator):
    # if ui_tech_key == "mining" and "PR" in player_rewards_set:
    #     base_ship_map = PLATFORM_TECH_TO_MODEL_KEYS.get(ui_ship_key, {})
    #     base_model_ship_key, _ = base_ship_map.get(ui_tech_key, (ui_ship_key, ui_tech_key))
    #     # Check if the base ship is 'atlantid-mt', which has its own mining model
    #     if base_model_ship_key == "atlantid-mt":
    #          return "atlantid-mt", "mining_with_resonator" # Hypothetical name
    #     else:
    #          return "standard-mt", "mining_with_resonator" # Hypothetical name

    # --- 2. General Dictionary-Based Mapping (Fallback) ---
    # If no specific reward logic matched, use the standard dictionary lookup.
    if ui_ship_key in PLATFORM_TECH_TO_MODEL_KEYS:
        return PLATFORM_TECH_TO_MODEL_KEYS[ui_ship_key].get(ui_tech_key, (ui_ship_key, ui_tech_key))
    else:
        # If the ship key itself isn't in the mapping, default to original UI keys
        return (ui_ship_key, ui_tech_key)

