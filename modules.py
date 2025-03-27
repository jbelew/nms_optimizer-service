# These value are highly generalized and do not reepresent the actual values in the game. They are ratios derrived from in-game experimentation and used for properly seeding the solving allgorithm.

modules = {
    "Exotic": {
        "types": {
            "weapons": [
                {
                    "label": "Cyclotron Ballista",
                    "key": "cyclotron",
                    "modules": [
                        {"id": "CB", "type": "core", "label": "Cyclotron Ballista", "bonus": 1.0, "adjacency": False, "sc_eligible": True, "image": "cyclotron.png"},
                        {"id": "QR", "type": "bonus", "label": "Dyson Pump", "bonus": 0.04, "adjacency": True, "sc_eligible": True, "image": "dyson.png"},
                        {"id": "Xa", "type": "bonus", "label": "Cyclotron Ballista Upgrade Sigma", "bonus": 0.40, "adjacency": True, "sc_eligible": True, "image": "cyclotron-upgrade.png"},
                        {"id": "Xb", "type": "bonus", "label": "Cyclotron Ballista Upgrade Tau", "bonus": 0.39, "adjacency": True, "sc_eligible": True, "image": "cyclotron-upgrade.png"},
                        {"id": "Xc", "type": "bonus", "label": "Cyclotron Ballista Upgrade Theta", "bonus": 0.38, "adjacency": True, "sc_eligible": True, "image": "cyclotron-upgrade.png"},
                    ],
                },                
                {
                    "label": "Infraknife Accelerator",
                    "key": "infra",
                    "modules": [
                        {"id": "IK", "type": "core", "label": "Infraknife Accelerator", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "infra.png"},
                        {"id": "QR", "type": "bonus", "label": "Q-Resonator", "bonus": 0.04, "adjacency": True, "sc_eligible": True, "image": "q-resonator.png"},
                        {"id": "Xa", "type": "bonus", "label": "Infraknife Accelerator Upgrade Sigma", "bonus": 0.40, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png"},
                        {"id": "Xb", "type": "bonus", "label": "Infraknife Accelerator Upgrade Tau", "bonus": 0.39, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png"},
                        {"id": "Xc", "type": "bonus", "label": "Infraknife Accelerator Upgrade Theta", "bonus": 0.38, "adjacency": True, "sc_eligible": True, "image": "infra-upgrade.png"},
                    ],
                },
                                {
                    "label": "Phase Beam",
                    "key": "phase",
                    "modules": [
                        {"id": "PB", "type": "core", "label": "Phase Beam", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "phase-beam.png"},
                        {"id": "FD", "type": "bonus", "label": "Fourier De-Limiter", "bonus": 0.04, "adjacency": True, "sc_eligible": True, "image": "fourier.png"},
                        {"id": "Xa", "type": "bonus", "label": "Phase Beam Upgrade Sigma", "bonus": 0.40, "adjacency": True, "sc_eligible": True, "image": "phase-upgrade.png"},
                        {"id": "Xb", "type": "bonus", "label": "Phase Beam Upgrade Tau", "bonus": 0.39, "adjacency": True, "sc_eligible": True, "image": "phase-upgrade.png"},
                        {"id": "Xc", "type": "bonus", "label": "Phase Beam Upgrade Theta", "bonus": 0.38, "adjacency": True, "sc_eligible": True, "image": "phase-upgrade.png"},
                    ],
                },
                {
                    "label": "Photon Cannon",
                    "key": "photon",
                    "modules": [
                        {"id": "PC", "type": "core", "label": "Photon Cannon", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "photon.png"},
                        {"id": "NO", "type": "bonus", "label": "Nonlinear Optics", "bonus": 0.04, "adjacency": True, "sc_eligible": True, "image": "nonlinear.png"},
                        {"id": "Xa", "type": "bonus", "label": "Photon Cannon Upgrade Sigma", "bonus": 0.40, "adjacency": True, "sc_eligible": True, "image": "photon-upgrade.png"},
                        {"id": "Xb", "type": "bonus", "label": "Photon Cannon Upgrade Tau", "bonus": 0.39, "adjacency": True, "sc_eligible": True, "image": "photon-upgrade.png"},
                        {"id": "Xc", "type": "bonus", "label": "Photon Cannon Upgrade Theta", "bonus": 0.38, "adjacency": True, "sc_eligible": True, "image": "photon-upgrade.png"},
                    ],
                },
                {
                    "label": "Positron Ejector",
                    "key": "positron",
                    "modules": [
                        {"id": "PE", "type": "core", "label": "Positron Ejector", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "positron.png"},
                        {"id": "FS", "type": "bonus", "label": "Fragment Supercharger", "bonus": 0.04, "adjacency": True, "sc_eligible": True, "image": "fragment.png"},
                        {"id": "Xa", "type": "bonus", "label": "Positron Ejector Upgrade Sigma", "bonus": 0.4, "adjacency": True, "sc_eligible": True, "image": "positron-upgrade.png"},
                        {"id": "Xb", "type": "bonus", "label": "Positron Ejector Upgrade Tau", "bonus": 0.39, "adjacency": True, "sc_eligible": True, "image": "positron-upgrade.png"},
                        {"id": "Xc", "type": "bonus", "label": "Positron Ejector Upgrade Theta", "bonus": 0.38, "adjacency": True, "sc_eligible": True, "image": "positron-upgrade.png"},
                    ],
                },
                {
                    "label": "Rocket Launcher",
                    "key": "rocket",
                    "modules": [
                        {"id": "RL", "type": "core", "label": "Rocket Launger", "bonus": 1.0, "adjacency": True, "sc_eligible": True, "image": "rocket.png"},
                        {"id": "LR", "type": "bonus", "label": "Large Rocket Tubes", "bonus": 0.056, "adjacency": True, "sc_eligible": True, "image": "tubes.png"},
                   ],
                },
            ],
             "mobility": [
                 {
                    "label": "Hyperdrive",
                    "key": "hyper",
                    "modules": [
                        {"id": "HD", "type": "core", "label": "Hyperdrive", "bonus": 0.0, "adjacency": True, "sc_eligible": False, "image": "hyperdrive.png"},
                        {"id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.02, "adjacency": True, "sc_eligible": False, "image": "atlantid.png"},
                        {"id": "CD", "type": "bonus", "label": "Cadmium Drive", "bonus": 0.01, "adjacency": True, "sc_eligible": False, "image": "cadmium.png"},
                        {"id": "ED", "type": "bonus", "label": "Emeril Drive", "bonus": 0.01, "adjacency": True, "sc_eligible": False, "image": "emeril.png"},
                        {"id": "ID", "type": "bonus", "label": "Indium Drive", "bonus": 0.01, "adjacency": True, "sc_eligible": False, "image": "indium.png"},              
                        {"id": "EW", "type": "bonus", "label": "Emergency Warp Unit", "bonus": 0.00, "adjacency": False, "sc_eligible": False, "image": "emergency.png"},
                        {"id": "Xa", "type": "bonus", "label": "Hyperdrive Upgrade Tau", "bonus": .320, "adjacency": True, "sc_eligible": True, "image": "hyper-upgrade.png"},
                        {"id": "Xb", "type": "bonus", "label": "Hyperdrive Upgrade Tau", "bonus": .310, "adjacency": True, "sc_eligible": True, "image": "hyper-upgrade.png"},
                        {"id": "Xc", "type": "bonus", "label": "Hyperdrive Upgrade Theta", "bonus": .300, "adjacency": True, "sc_eligible": True, "image": "hyper-upgrade.png"},
                    ],
                 },
                 {
                    "label": "Launch Thruster",
                    "key": "launch",
                    "modules": [
                        {"id": "LT", "type": "core", "label": "Launch Thruster", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "launch.png"},
                        {"id": "EF", "type": "bonus", "label": "Efficient Thrusters", "bonus": 0.20, "adjacency": False, "sc_eligible": False, "image": "efficient.png"},
                        {"id": "RC", "type": "bonus", "label": "Launch System Recharger", "bonus": 0.00, "adjacency": False, "sc_eligible": False, "image": "recharger.png"},
                        {"id": "Xa", "type": "bonus", "label": "Launch Thruster Upgrade Sigma", "bonus": 0.30, "adjacency": True, "sc_eligible": True, "image": "launch-upgrade.png"},
                        {"id": "Xb", "type": "bonus", "label": "Launch Thruster Upgrade Tau", "bonus": 0.29, "adjacency": True, "sc_eligible": True, "image": "launch-upgrade.png"},
                        {"id": "Xc", "type": "bonus", "label": "Launch Thruster Upgrade Theta", "bonus": 0.28, "adjacency": True, "sc_eligible": True, "image": "launch-upgrade.png"},
                    ],
                 },
                 {
                    "label": "Pulse Engine",
                    "key": "pulse",
                    "modules": [
                        {"id": "PE", "type": "core", "label": "Pulse Engine", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "pulse.png"},
                        {"id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 0.08, "adjacency": True, "sc_eligible": False, "image": "flight-assist.png"},
                        {"id": "PC", "type": "reward", "label": "Photonix Core", "bonus": 0.067, "adjacency": True, "sc_eligible": False, "image": "photonix.png"},
                        {"id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.00, "adjacency": True, "sc_eligible": False, "image": "sublight.png"},
                        {"id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.00, "adjacency": True, "sc_eligible": False, "image": "instability.png"},
                        {"id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 0.12, "adjacency": True, "sc_eligible": True, "image": "pulse-upgrade.png"},
                        {"id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 0.11, "adjacency": True, "sc_eligible": True, "image": "pulse-upgrade.png"},
                        {"id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 0.10, "adjacency": True, "sc_eligible": True, "image": "pulse-upgrade.png"},
                    ],
                 },
                 {
                    "label": "Starship Trails",
                    "key": "trails",
                    "modules": [
                        {"id": "AB", "type": "bonus", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": True, "sc_eligible": True, "image": "artemis.png"},
                        {"id": "PB", "type": "bonus", "label": "Polo Figurine", "bonus": 0.05, "adjacency": True, "sc_eligible": True, "image": "polo.png"},
                        {"id": "SB", "type": "core", "label": "Tenticled Figurine", "bonus": 0.01, "adjacency": True, "sc_eligible": True, "image": "squid.png"},
                        {"id": "SP", "type": "bonus", "label": "Sputtering Starship Trail", "bonus": 0.0, "adjacency": True, "sc_eligible": False, "image": "sputtering-trail.png"},
                        {"id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.0, "adjacency": True, "sc_eligible": False, "image": "cadmium-trail.png"},
                        {"id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.0, "adjacency": True, "sc_eligible": True, "image": "emeril-trail.png"},
                        {"id": "TT", "type": "bonus", "label": "Temporal Starship Trail", "bonus": 0.0, "adjacency": True, "sc_eligible": True, "image": "temporal-trail.png"},
                        {"id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.0, "adjacency": True, "sc_eligible": True, "image": "stealth-trail.png"},
                        {"id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.0, "adjacency": True, "sc_eligible": True, "image": "golden-trail.png"},
                        {"id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.0, "adjacency": True, "sc_eligible": True, "image": "chromatic-trail.png"},
  
                    ],
                 },
            ],
             "other": [
                 {
                    "label": "Bobbleheads",
                    "key": "bobble",
                    "modules": [
                        {"id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "apollo.png"},
                        {"id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "atlas.png"},
                        {"id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "nada.png"},          
                        {"id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "null.png"},
                    ],
                 },
                 {
                "label": "Scanners, Teleporter, Etc ...",
                    "key": "scanners",
                    "modules": [
                        {"id": "CD", "type": "core", "label": "Cargo Scan Deflector", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "cargo.png"},
                        {"id": "ES", "type": "core", "label": "Economy Scanner", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "economy.png"},
                        {"id": "CS", "type": "core", "label": "Conflict Scanner", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "conflict.png"},
                        {"id": "AJ", "type": "core", "label": "Aqua-Jets", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "aquajets.png"},
                        {"id": "TP", "type": "core", "label": "Teleport Receiver", "bonus": 0.0, "adjacency": False, "sc_eligible": False, "image": "teleport.png"},
                    ],
                 },                 
                 {
                    "label": "Starship Shields",
                    "key": "shield",
                    "modules": [
                        {"id": "DS", "type": "core", "label": "Defensive Shields", "bonus": 1.0, "adjacency": False, "sc_eligible": False, "image": "shield.png"},
                        {"id": "AA", "type": "bonus", "label": "Ablative Armor", "bonus": 0.07, "adjacency": False, "sc_eligible": False, "image": "ablative.png"},
                        {"id": "Xa", "type": "bonus", "label": "Shield Upgrade Sigma", "bonus": 0.3, "adjacency": True, "sc_eligible": True, "image": "shield-upgrade.png"},
                        {"id": "Xb", "type": "bonus", "label": "Shield Upgrade Tau", "bonus": 0.29, "adjacency": True, "sc_eligible": True, "image": "shield-upgrade.png"},
                        {"id": "Xc", "type": "bonus", "label": "Shield Upgrade Theta", "bonus": 0.28, "adjacency": True, "sc_eligible": True, "image": "shield-upgrade.png"},
                    ],
                 },

            ]
        },
    },
    # Add other ships here if needed (e.g., "Fighter", "Hauler")
}

solves = {
    "Exotic": {  # Ship type
        "cyclotron": {
            "map": {
                (0, 0): "CB",
                (1, 0): "Xa",
                (2, 0): "QR",
                (0, 1): "None",
                (1, 1): "Xb",
                (2, 1): "Xc",
            },
            "score": 5.03
        },
        "infra": {
            "map": {
                (0, 0): "IK",
                (1, 0): "Xa",
                (2, 0): "QR",
                (0, 1): "None",
                (1, 1): "Xb",
                (2, 1): "Xc",
            },
            "score": 5.03
        },
        "phase": {
            "map": {
                (0, 0): "PB",
                (1, 0): "Xa",
                (2, 0): "FD",
                (0, 1): "None",
                (1, 1): "Xb",
                (2, 1): "Xc",
            },
            "score": 5.03
        },
        "photon": {
            "map": {
                (0, 0): "PC",
                (1, 0): "Xa",
                (2, 0): "NO",
                (0, 1): "None",
                (1, 1): "Xb",
                (2, 1): "Xc",
            },
            "score": 5.03
        },
        "positron": {
            "map": {
                (0, 0): "PE",
                (1, 0): "Xa",
                (2, 0): "FS",
                (0, 1): "None",
                (1, 1): "Xb",
                (2, 1): "Xc",
            },
            "score": 5.03
        },
        "rocket": {
            "map": {
                (0, 0): "RL",
                (1, 0): "LR",
            },
            "score": 1.112
        },
        "hyper": {
            "map": {
                (0, 0): "HD",
                (1, 0): "AD",
                (2, 0): "CD",
                (0, 1): "ED",
                (1, 1): "Xa",
                (2, 1): "Xb",
                (0, 2): "EW",
                (1, 2): "Xc",
                (2, 2): "ID",
            },
            "score": 4.22
        },
        "launch": {
            "map": {
                (0, 0): "LT",
                (1, 0): "Xa",
                (2, 0): "EF",
                (0, 1): "Xc",
                (1, 1): "Xb",
                (2, 1): "RC",
            },
            "score": 3.40
        },
        "pulse": {
            "map": {
                (0, 0): "PE",
                (1, 0): "Xb",
                (2, 0): "FA",
                (0, 1): "Xc",
                (1, 1): "Xa",
                (2, 1): "PC",
                (0, 2): "SL",
                (1, 2): "ID",
                (2, 2): "None",
            },
            "score": 1.8810000000000002
        },
        "shield": {
            "map": {
                (0, 0): "DS",
                (1, 0): "Xa",
                (2, 0): "AA",
                (0, 1): "Xc",
                (1, 1): "Xb",
                (2, 1): "None",
            },
            "score": 3.98
        },
        "trails": {
            "map": {
                (0, 0): "RT",
                (1, 0): "ET",
                (2, 0): "ST",
                (3, 0): "TT",
                (0, 1): "SB",
                (1, 1): "AB",
                (2, 1): "PB",
                (3, 1): "None",
                (0, 2): "GT",
                (1, 2): "CT",
                (2, 2): "SP",
                (3, 2): "None",
            },
            "score": 0.46
        },
    },
}