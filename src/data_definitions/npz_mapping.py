# npz_mapping.py
from typing import (
    Optional,
    Dict,
    List,
)  # Removed Tuple as it's no longer used for return type

"""
Maps the user-facing platform key (ship) and technology key (tech)
to the corresponding keys used internally to identify the correct
.npz file path.
"""

# --- Mapping Dictionary (Base mappings) ---
PLATFORM_TECH_TO_NPZ_KEYS = {
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
        "pulse": ("solar", "pulse"),
        "photonix": ("solar", "photonix"),
    },
    "corvette": {
        "cyclotron": ("corvette", "cyclotron"),
        "hyper": ("corvette", "hyper"),
        "launch": ("corvette", "launch"),
        "phase": ("corvette", "phase"),
        "photon": ("corvette", "photon"),
        "photonix": ("corvette", "photonix"),
        "positron": ("corvette", "positron"),
        "pulse": ("corvette", "pulse"),
        "shield": ("corvette", "shield"),
    },
    # --- Multi-Tools ---
    "standard-mt": {
        "mining": ("standard-mt", "mining"),
        "scanner": ("standard-mt", "scanner"),
        "bolt-caster": ("standard-mt", "bolt-caster"),
        "blaze-javelin": ("standard-mt", "blaze-javelin"),
        "pulse-spitter": ("standard-mt", "pulse-spitter"),
        "plasma-launcher": ("standard-mt", "plasma-launcher"),
        "neutron": ("standard-mt", "neutron"),
        "scatter": ("standard-mt", "scatter"),
        "geology": ("standard-mt", "geology"),
    },
    "atlantid": {
        "mining": ("atlantid", "mining"),
    },
    "living": {
        "grafted": ("living", "grafted"),
        "pulsing": ("living", "pulsing"),
        "scream": ("living", "scream"),
        "spewing": ("living", "spewing"),
        "assembly": ("living", "assembly"),
        "singularity": ("living", "singularity"),
    },
    "freighter": {
        "hyper": ("freighter", "hyper"),
    },
}

def get_npz_keys(
    ui_ship_key: str,
    ui_tech_key: str,
    solve_type: Optional[str] = None,
    player_owned_rewards: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Determines keys for .npz file path.

    Args:
        ui_ship_key: The ship key provided by the user/UI.
        ui_tech_key: The tech key provided by the user/UI.
        solve_type: The solve type, e.g., "max" or "normal".

    Returns:
        A dictionary with keys:
            "npz_ship_key": Key for the ship part of the .npz path.
            "npz_tech_key": Key for the tech part of the .npz path.
    """
    player_rewards_set = set(player_owned_rewards) if player_owned_rewards else set()

    if ui_ship_key in PLATFORM_TECH_TO_NPZ_KEYS:
        npz_ship_key, npz_tech_key = PLATFORM_TECH_TO_NPZ_KEYS[ui_ship_key].get(
            ui_tech_key, (ui_ship_key, ui_tech_key)
        )
    else:
        npz_ship_key, npz_tech_key = ui_ship_key, ui_tech_key

    # Apply reward-based overrides (e.g., "pulse" + "PC" -> "photonix")
    if ui_tech_key == "pulse" and "PC" in player_rewards_set:
        npz_tech_key = "photonix"

    return {
        "npz_ship_key": npz_ship_key,
        "npz_tech_key": npz_tech_key,
    }
