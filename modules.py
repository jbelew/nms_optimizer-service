# These value are highly generalized and do not represent the actual values in the game. They are ratios derived from in-game experimentation and used for properly seeding the solving algorithm.

# fmt: off
modules = {
    "standard": {
        "label": "Standard / Exotic Starships",
        "type": "Starship",
        "types": {
            "Weaponry": [
                {
                    "label": "Cyclotron Ballista",
                    "key": "cyclotron",
                    "image": "cyclotron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "CB", "type": "core", "label": "Cyclotron Ballista", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron.webp", },
                        { "id": "QR", "type": "bonus", "label": "Dyson Pump", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "dyson.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Cyclotron Ballista Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Cyclotron Ballista Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Cyclotron Ballista Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Infraknife Accelerator",
                    "key": "infra",
                    "image": "infra.webp",
                    "color": "red",
                    "modules": [
                        { "id": "IK", "type": "core", "label": "Infraknife Accelerator", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "infra.webp", },
                        { "id": "QR", "type": "bonus", "label": "Q-Resonator", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "q-resonator.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Infraknife Accelerator Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Infraknife Accelerator Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Infraknife Accelerator Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.webp", },
                    ],
                },
                {
                    "label": "Phase Beam",
                    "key": "phase",
                    "image": "phase.webp",
                    "color": "green",
                    "modules": [
                        { "id": "PB", "type": "core", "label": "Phase Beam", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "phase-beam.webp", },
                        { "id": "FD", "type": "bonus", "label": "Fourier De-Limiter", "bonus": 0.07, "adjacency": "lesser", "sc_eligible": True, "image": "fourier.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Phase Beam Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Phase Beam Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Phase Beam Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.webp", },
                    ],
                },
                {
                    "label": "Photon Cannon",
                    "key": "photon",
                    "image": "photon.webp",
                    "color": "blue",
                    "modules": [
                        { "id": "PC", "type": "core", "label": "Photon Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "photon.webp", },
                        { "id": "NO", "type": "bonus", "label": "Nonlinear Optics", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "nonlinear.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Photon Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Photon Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Photon Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.webp", },
                    ],
                },
                {
                    "label": "Positron Ejector",
                    "key": "positron",
                    "image": "positron.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Positron Ejector", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "positron.webp", },
                        { "id": "FS", "type": "bonus", "label": "Fragment Supercharger", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "fragment.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Positron Ejector Upgrade Theta", "bonus": 0.4, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Positron Ejector Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Positron Ejector Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Rocket Launcher",
                    "key": "rocket",
                    "image": "rocket.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "RL", "type": "core", "label": "Rocket Launger", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "rocket.webp", },
                        { "id": "LR", "type": "bonus", "label": "Large Rocket Tubes", "bonus": 0.056, "adjacency": "greater", "sc_eligible": True, "image": "tubes.webp", },
                    ],
                },
            ],
            "Defensive Systems": [
                {
                    "label": "Starship Shields",
                    "key": "shield",
                    "image": "shield.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "DS", "type": "core", "label": "Defensive Shields", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "shield.webp", },
                        { "id": "AA", "type": "bonus", "label": "Ablative Armor", "bonus": 0.07, "adjacency": "greater", "sc_eligible": True, "image": "ablative.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Shield Upgrade Theta", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Shield Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Shield Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.webp", },
                    ],
                },
            ],
            "Hyperdrive": [
                {
                    "label": "Hyperdrive",
                    "key": "hyper",
                    "image": "hyper.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "HD", "type": "core", "label": "Hyperdrive", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "hyperdrive.webp", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "atlantid.webp", },
                        { "id": "CD", "type": "bonus", "label": "Cadmium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "cadmium.webp", },
                        { "id": "ED", "type": "bonus", "label": "Emeril Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "emeril.webp", },
                        { "id": "ID", "type": "bonus", "label": "Indium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "indium.webp", },
                        { "id": "EW", "type": "bonus", "label": "Emergency Warp Unit", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "emergency.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Hyperdrive Upgrade Theta", "bonus": .320, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Hyperdrive Upgrade Tau", "bonus": .310, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Hyperdrive Upgrade Sigma", "bonus": .300, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.webp", },
                    ],
                },
                {
                    "label": "Launch Thruster",
                    "key": "launch",
                    "image": "launch.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "LT", "type": "core", "label": "Launch Thruster", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": False, "image": "launch.webp", },
                        { "id": "EF", "type": "bonus", "label": "Efficient Thrusters", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "efficient.webp", },
                        { "id": "RC", "type": "bonus", "label": "Launch Auto-Charger", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "recharger.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Launch Thruster Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Launch Thruster Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Launch Thruster Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Engine",
                    "key": "pulse",
                    "image": "pulse.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Pulse Engine", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "pulse.webp", },
                        { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1.07, "adjacency": "greater", "sc_eligible": False, "image": "flight-assist.webp", },
                        { "id": "PC", "type": "reward", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "photonix.webp", },
                        { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "sublight.webp", },
                        { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "instability.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                    ],
                },
                # {
                #     "label": "Pulse Engine",
                #     "key": "photonix",
                #     "image": "pulse.webp",
                #     "color": "orange",
                #     "modules": [
                #         { "id": "PE", "type": "core", "label": "Pulse Engine", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "pulse.webp", },
                #         { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1.07, "adjacency": "greater", "sc_eligible": False, "image": "flight-assist.webp", },
                #         { "id": "PC", "type": "bonus", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "photonix.webp", },
                #         { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "sublight.webp", },
                #         { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "instability.webp", },
                #         { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                #         { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                #         { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                #     ],
                # },
            ],
            "Utilities": [
                {
                    "label": "Aqua-Jets",
                    "key": "aqua",
                    "image": "aquajet.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AJ", "type": "core", "label": "Aqua-Jets", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "aquajets.webp", },
                    ],
                },
                {
                    "label": "Figurines",
                    "key": "bobble",
                    "image": "bobble.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "apollo.webp", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "atlas.webp", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "nada.webp", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "null.webp", },
                    ],
                },
                {
                    "label": "Scanners",
                    "key": "scanners",
                    "image": "scanner.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "ES", "type": "core", "label": "Economy Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "economy.webp", },
                        { "id": "CS", "type": "core", "label": "Conflict Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "conflict.webp", },
                        { "id": "CD", "type": "core", "label": "Cargo Scan Deflector", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "cargo.webp", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "trails.webp",
                    "color": "white", 
                    "modules": [
                        { "id": "AB", "type": "bonus", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "artemis.webp", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.06, "adjacency": "greater", "sc_eligible": True, "image": "polo.webp", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "squid.webp", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "sputtering-trail.webp", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "cadmium-trail.webp", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "emeril-trail.webp", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "temporal-trail.webp", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "stealth-trail.webp", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "golden-trail.webp", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "chromatic-trail.webp", },
                    ],
                },
                {
                    "label": "Teleport Receiver",
                    "key": "teleporter",
                    "image": "teleport.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TP", "type": "core", "label": "Teleport Receiver", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "teleport.webp", },
                    ],
                },
            ],
        },
    },
    "sentinel": {
        "label": "Sentinel Interceptors",
        "type": "Starship",
        "types": {
            "Weaponry": [
                {
                    "label": "Cyclotron Ballista",
                    "key": "cyclotron",
                    "image": "cyclotron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "CB", "type": "core", "label": "Cyclotron Ballista", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron.webp", },
                        { "id": "QR", "type": "bonus", "label": "Dyson Pump", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "dyson.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Cyclotron Ballista Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Cyclotron Ballista Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Cyclotron Ballista Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Infraknife Accelerator",
                    "key": "infra",
                    "image": "infra.webp",
                    "color": "red",
                    "modules": [
                        { "id": "IK", "type": "core", "label": "Infraknife Accelerator", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "infra.webp", },
                        { "id": "QR", "type": "bonus", "label": "Q-Resonator", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "q-resonator.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Infraknife Accelerator Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Infraknife Accelerator Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Infraknife Accelerator Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.webp", },
                    ],
                },
                {
                    "label": "Phase Beam",
                    "key": "phase",
                    "image": "phase.webp",
                    "color": "green",
                    "modules": [
                        { "id": "PB", "type": "core", "label": "Phase Beam", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "phase-beam.webp", },
                        { "id": "FD", "type": "bonus", "label": "Fourier De-Limiter", "bonus": 0.07, "adjacency": "lesser", "sc_eligible": True, "image": "fourier.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Phase Beam Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Phase Beam Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Phase Beam Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.webp", },
                    ],
                },
                {
                    "label": "Positron Ejector",
                    "key": "positron",
                    "image": "positron.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Positron Ejector", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "positron.webp", },
                        { "id": "FS", "type": "bonus", "label": "Fragment Supercharger", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "fragment.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Positron Ejector Upgrade Theta", "bonus": 0.4, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Positron Ejector Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Positron Ejector Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Rocket Launcher",
                    "key": "rocket",
                    "image": "rocket.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "RL", "type": "core", "label": "Rocket Launger", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "rocket.webp", },
                        { "id": "LR", "type": "bonus", "label": "Large Rocket Tubes", "bonus": 0.056, "adjacency": "greater", "sc_eligible": True, "image": "tubes.webp", },
                    ],
                },
                {
                    "label": "Sentinel Cannon",
                    "key": "photon",
                    "image": "cannon.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "PC", "type": "core", "label": "Sentinel Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "cannon.webp", },
                        { "id": "NO", "type": "bonus", "label": "Nonlinear Optics", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "nonlinear.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Photon Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Photon Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Photon Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.webp", },
                    ],
                },
            ],
            "Defensive Systems": [
                {
                    "label": "Aeron Shields",
                    "key": "shield",
                    "image": "aeron.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "DS", "type": "core", "label": "Aeron Shields", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "aeron.webp", },
                        { "id": "AA", "type": "bonus", "label": "Ablative Armor", "bonus": 0.07, "adjacency": "greater", "sc_eligible": True, "image": "ablative.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Shield Upgrade Theta", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Shield Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Shield Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.webp", },
                    ],
                },
            ],
            "Hyperdrive": [
                {
                    "label": "Anti-Gravity Well",
                    "key": "launch",
                    "image": "anti-gravity.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "LT", "type": "core", "label": "Anti-Gravity Well", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "anti-gravity.webp", },
                        { "id": "EF", "type": "bonus", "label": "Efficient Thrusters", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "efficient.webp", },
                        { "id": "RC", "type": "bonus", "label": "Launch Atuo-Charger", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "recharger.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Launch Thruster Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Launch Thruster Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Launch Thruster Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.webp", },
                    ],
                },
                {
                    "label": "Crimson Core",
                    "key": "hyper",
                    "image": "crimson.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "HD", "type": "core", "label": "Crimson Core", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "crimson.webp", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "atlantid.webp", },
                        { "id": "CD", "type": "bonus", "label": "Cadmium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "cadmium.webp", },
                        { "id": "ED", "type": "bonus", "label": "Emeril Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "emeril.webp", },
                        { "id": "ID", "type": "bonus", "label": "Indium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "indium.webp", },
                        { "id": "EW", "type": "bonus", "label": "Emergency Warp Unit", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "emergency.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Crimson Core Upgrade Theta", "bonus": 0.320, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Crimson Core Upgrade Tau", "bonus": 0.310, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Crimson Core Upgrade Sigma", "bonus": 0.300, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.webp", },
                  ],
                },
                {
                    "label": "Luminance Drive",
                    "key": "pulse",
                    "image": "luminance.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Luminance Drive", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "luminance.webp", },
                        { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1.07, "adjacency": "greater", "sc_eligible": False, "image": "flight-assist.webp", },
                        { "id": "PC", "type": "reward", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "photonix.webp", },
                        { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "sublight.webp", },
                        { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "instability.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                    ],
                },
                # {
                #     "label": "Luminance Drive",
                #     "key": "photonix",
                #     "image": "luminance.webp",
                #     "color": "orange",
                #     "modules": [
                #         { "id": "PE", "type": "core", "label": "Luminance Drive", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "luminance.webp", },
                #         { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1.07, "adjacency": "greater", "sc_eligible": False, "image": "flight-assist.webp", },
                #         { "id": "PC", "type": "bonus", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "photonix.webp", },
                #         { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "sublight.webp", },
                #         { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "instability.webp", },
                #         { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                #         { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                #         { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                #     ],
                # },
            ],
            "Utilities": [
                {
                    "label": "Aqua-Jets",
                    "key": "aqua",
                    "image": "aquajet.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AJ", "type": "core", "label": "Aqua-Jets", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "aquajets.webp", },
                    ],
                },
                {
                    "label": "Figurines",
                    "key": "bobble",
                    "image": "bobble.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "apollo.webp", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "atlas.webp", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "nada.webp", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "null.webp", },
                    ],
                },
                {
                    "label": "Pilot Interface",
                    "key": "pilot",
                    "image": "pilot.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PI", "type": "bonus", "label": "Pilot Interface", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "pilot.webp", },
                    ],
                },
                {
                    "label": "Scanners",
                    "key": "scanners",
                    "image": "scanner.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "ES", "type": "core", "label": "Economy Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "economy.webp", },
                        { "id": "CS", "type": "core", "label": "Conflict Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "conflict.webp", },
                        { "id": "CD", "type": "core", "label": "Cargo Scan Deflector", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "cargo.webp", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "trails.webp",
                    "color": "white",
                    "modules": [
                        { "id": "AB", "type": "bonus", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "artemis.webp", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.06, "adjacency": "greater", "sc_eligible": True, "image": "polo.webp", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "squid.webp", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "sputtering-trail.webp", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "cadmium-trail.webp", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "emeril-trail.webp", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "temporal-trail.webp", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "stealth-trail.webp", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "golden-trail.webp", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "chromatic-trail.webp", },
                     ],
                },
                {
                    "label": "Teleport Receiver",
                    "key": "teleporter",
                    "image": "teleport.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TP", "type": "core", "label": "Teleport Receiver", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "teleport.webp", },
                    ],
                },
            ],
        },
    },
    "solar": {
        "label": "Solar Starships",
        "type": "Starship",
        "types": {
            "Weaponry": [
                {
                    "label": "Cyclotron Ballista",
                    "key": "cyclotron",
                    "image": "cyclotron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "CB", "type": "core", "label": "Cyclotron Ballista", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron.webp", },
                        { "id": "QR", "type": "bonus", "label": "Dyson Pump", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "dyson.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Cyclotron Ballista Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Cyclotron Ballista Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Cyclotron Ballista Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Infraknife Accelerator",
                    "key": "infra",
                    "image": "infra.webp",
                    "color": "red",
                    "modules": [
                        { "id": "IK", "type": "core", "label": "Infraknife Accelerator", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "infra.webp", },
                        { "id": "QR", "type": "bonus", "label": "Q-Resonator", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "q-resonator.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Infraknife Accelerator Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Infraknife Accelerator Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Infraknife Accelerator Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.webp", },
                    ],
                },
                {
                    "label": "Phase Beam",
                    "key": "phase",
                    "image": "phase.webp",
                    "color": "green",
                    "modules": [
                        { "id": "PB", "type": "core", "label": "Phase Beam", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "phase-beam.webp", },
                        { "id": "FD", "type": "bonus", "label": "Fourier De-Limiter", "bonus": 0.07, "adjacency": "lesser", "sc_eligible": True, "image": "fourier.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Phase Beam Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Phase Beam Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Phase Beam Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.webp", },
                    ],
                },
                {
                    "label": "Photon Cannon",
                    "key": "photon",
                    "image": "photon.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "PC", "type": "core", "label": "Photon Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "photon.webp", },
                        { "id": "NO", "type": "bonus", "label": "Nonlinear Optics", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "nonlinear.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Photon Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Photon Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Photon Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.webp", },
                    ],
                },
                {
                    "label": "Positron Ejector",
                    "key": "positron",
                    "image": "positron.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Positron Ejector", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "positron.webp", },
                        { "id": "FS", "type": "bonus", "label": "Fragment Supercharger", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "fragment.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Positron Ejector Upgrade Theta", "bonus": 0.4, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Positron Ejector Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Positron Ejector Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Rocket Launcher",
                    "key": "rocket",
                    "image": "rocket.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "RL", "type": "core", "label": "Rocket Launger", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "rocket.webp", },
                        { "id": "LR", "type": "bonus", "label": "Large Rocket Tubes", "bonus": 0.056, "adjacency": "greater", "sc_eligible": True, "image": "tubes.webp", },
                    ],
                },
            ],
            "Defensive Systems": [
                {
                    "label": "Starship Shields",
                    "key": "shield",
                    "image": "shield.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "DS", "type": "core", "label": "Defensive Shields", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "shield.webp", },
                        { "id": "AA", "type": "bonus", "label": "Ablative Armor", "bonus": 0.07, "adjacency": "greater", "sc_eligible": True, "image": "ablative.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Shield Upgrade Theta", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Shield Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Shield Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.webp", },
                    ],
                },
            ],
            "Hyperdrive": [
                {
                    "label": "Hyperdrive",
                    "key": "hyper",
                    "image": "hyper.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "HD", "type": "core", "label": "Hyperdrive", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "hyperdrive.webp", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "atlantid.webp", },
                        { "id": "CD", "type": "bonus", "label": "Cadmium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "cadmium.webp", },
                        { "id": "ED", "type": "bonus", "label": "Emeril Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "emeril.webp", },
                        { "id": "ID", "type": "bonus", "label": "Indium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "indium.webp", },
                        { "id": "EW", "type": "bonus", "label": "Emergency Warp Unit", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "emergency.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Hyperdrive Upgrade Theta", "bonus": .320, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Hyperdrive Upgrade Tau", "bonus": .310, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Hyperdrive Upgrade Sigma", "bonus": .300, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.webp", },
                    ],
                },
                {
                    "label": "Launch Thruster",
                    "key": "launch",
                    "image": "launch.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "LT", "type": "core", "label": "Launch Thruster", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": False, "image": "launch.webp", },
                        { "id": "EF", "type": "bonus", "label": "Efficient Thrusters", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "efficient.webp", },
                        { "id": "RC", "type": "bonus", "label": "Launch Auto-Charger", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "recharger.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Launch Thruster Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Launch Thruster Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Launch Thruster Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Engine",
                    "key": "pulse",
                    "image": "pulse.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Pulse Engine", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "pulse.webp", },
                        { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1.07, "adjacency": "greater", "sc_eligible": False, "image": "flight-assist.webp", },
                        { "id": "VS", "type": "bonus", "label": "Vesper Sail", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "vesper.webp", },
                        { "id": "PC", "type": "reward", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "photonix.webp", },
                        { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "sublight.webp", },
                        { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "instability.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                    ],
                },
                # {
                #     "label": "Pulse Engine",
                #     "key": "photonix",
                #     "image": "pulse.webp",
                #     "color": "orange",
                #     "modules": [
                #         { "id": "PE", "type": "core", "label": "Pulse Engine", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "pulse.webp", },
                #         { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1., "adjacency": "greater", "sc_eligible": False, "image": "flight-assist.webp", },
                #         { "id": "VS", "type": "bonus", "label": "Vesper Sail", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "vesper.webp", },
                #         { "id": "PC", "type": "bonus", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "photonix.webp", },
                #         { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "sublight.webp", },
                #         { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "instability.webp", },
                #         { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                #         { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                #         { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.webp", },
                #     ],
                # },
            ],
            "Utilities": [
                {
                    "label": "Aqua-Jets",
                    "key": "aqua",
                    "image": "aquajet.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AJ", "type": "core", "label": "Aqua-Jets", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "aquajets.webp", },
                    ],
                },
                {
                    "label": "Figurines",
                    "key": "bobble",
                    "image": "bobble.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "apollo.webp", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "atlas.webp", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "nada.webp", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "null.webp", },
                    ],
                },
                {
                    "label": "Scanners",
                    "key": "scanners",
                    "image": "scanner.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "ES", "type": "core", "label": "Economy Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "economy.webp", },
                        { "id": "CS", "type": "core", "label": "Conflict Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "conflict.webp", },
                        { "id": "CD", "type": "core", "label": "Cargo Scan Deflector", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "cargo.webp", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "trails.webp",
                    "color": "white", 
                    "modules": [
                        { "id": "AB", "type": "bonus", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "artemis.webp", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.06, "adjacency": "greater", "sc_eligible": True, "image": "polo.webp", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "squid.webp", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "sputtering-trail.webp", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "cadmium-trail.webp", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "emeril-trail.webp", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "temporal-trail.webp", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "stealth-trail.webp", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "golden-trail.webp", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "chromatic-trail.webp", },
                    ],
                },
                {
                    "label": "Teleport Receiver",
                    "key": "teleporter",
                    "image": "teleport.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TP", "type": "core", "label": "Teleport Receiver", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "teleport.webp", },
                    ],
                },
            ],
        },
    },    
    "living": {
        "label": "Living Starships",
        "type": "Starship",
        "types": {
            "Weaponry": [
                {
                    "label": "Grafted Eyes",
                    "key": "grafted",
                    "image": "grafted.webp",
                    "color": "green",
                    "modules": [
                        { "id": "GE", "type": "core", "label": "Grafted Eyes", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "grafted.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Grafted Eyes Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "grafted-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Grafted Eyes Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "grafted-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Grafted Eyes Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "grafted-upgrade.webp", },
                    ],
                },
                {
                    "label": "Spewing Vents",
                    "key": "spewing",
                    "image": "spewing.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "SV", "type": "core", "label": "Spewing Vents", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "spewing.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Spewing Vents Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "spewing-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Spewing Vents Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "spewing-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Spewing Vents Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "spewing-upgrade.webp", },
                    ],
                },
            ],
            "Defensive Systems": [
                {
                    "label": "Scream Supressor",
                    "key": "scream",
                    "image": "scream.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "SS", "type": "core", "label": "Scream Supressor", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "scream.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scream Supressor Upgrade Theta", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "scream-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scream Supressor Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "scream-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scream Supressor Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "scream-upgrade.webp", },
                    ],
                },
            ],
            "Hyperdrive": [
                {
                    "label": "Neural Assembly",
                    "key": "assembly",
                    "image": "assembly.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "NA", "type": "core", "label": "Neural Assembly", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "assembly.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Neural Assembly Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "assembly-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Neural Assembly Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "assembly-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Neural Assembly Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "assembly-upgrade.webp", },
                        { "id": "CM", "type": "bonus", "label": "Chroloplast Membrane", "bonus": 0.0001, "adjacency": "greater", "sc_eligible": True, "image": "chloroplast.webp", },
                    ],
                },
                {
                    "label": "Singularity Cortex",
                    "key": "singularity",
                    "image": "singularity.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Singularity Cortex", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "singularity.webp", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "atlantid.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Singularity Cortex Upgrade Theta", "bonus": 0.320, "adjacency": "greater", "sc_eligible": True, "image": "singularity-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Singularity Cortex Upgrade Tau", "bonus": 0.310, "adjacency": "greater", "sc_eligible": True, "image": "singularity-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Singularity Cortex Upgrade Sigma", "bonus": 0.300, "adjacency": "greater", "sc_eligible": True, "image": "singularity-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulsing Heart",
                    "key": "pulsing",
                    "image": "pulsing.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PH", "type": "core", "label": "Pulsing Heart", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "pulsing.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulsing Heart Upgrade Theta", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "pulsing-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulsing Heart Upgrade Tau", "bonus": 0.49, "adjacency": "greater", "sc_eligible": True, "image": "pulsing-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulsing Heart Upgrade Sigma", "bonus": 0.48, "adjacency": "greater", "sc_eligible": True, "image": "pulsing-upgrade.webp", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Saline Carapace",
                    "key": "saline",
                    "image": "saline.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Saline Catapace", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "saline.webp", },
                    ],
                },
                {
                    "label": "Figurines",
                    "key": "bobble",
                    "image": "bobble.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "apollo.webp", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "atlas.webp", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "nada.webp", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "null.webp", },
                    ],
                },
                {
                    "label": "Scanners",
                    "key": "scanners",
                    "image": "wormhole.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "WB", "type": "core", "label": "Wormhole Brain", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "wormhole.webp", },
                        { "id": "NS", "type": "core", "label": "Neural Shielding", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "neural.webp", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "trails.webp",
                    "color": "white", 
                    "modules": [
                        { "id": "AB", "type": "bonus", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "artemis.webp", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.06, "adjacency": "greater", "sc_eligible": True, "image": "polo.webp", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "squid.webp", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "sputtering-trail.webp", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "cadmium-trail.webp", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "emeril-trail.webp", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "temporal-trail.webp", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "stealth-trail.webp", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "golden-trail.webp", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "chromatic-trail.webp", },
                    ],
                },
            ],
        },
    },
    "standard-mt": {
        "label": "Standard / Royal Multi-Tools",
        "type": "Multi-Tool",
        "types": {
            "Mining": [
                {
                    "label": "Mining Beam",
                    "key": "mining",
                    "image": "mining-beam.webp",
                    "color": "green",
                    "modules": [
                        { "id": "MB", "type": "core", "label": "Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-laser.webp", },
                        { "id": "AM", "type": "bonus", "label": "Advanced Mining Laser", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "advanced-mining.webp", },
                        { "id": "OD", "type": "bonus", "label": "Optical Drill", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "optical.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mining Laser Upgrade Theta", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mining Laser Upgrade Tau", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mining Laser Upgrade Sigma", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                    ],
                },
            ],
            "Scanner": [
                {
                    "label": "Analysis Visor",
                    "key": "analysis",
                    "image": "analysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AV", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "analysis.webp", },
                    ],
                },
                {
                    "label": "Fishing Rig",
                    "key": "fishing",
                    "image": "fishing.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "FR", "type": "core", "label": "F", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "fishing.webp", },
                    ],
                },
                {
                    "label": "Scanner",
                    "key": "scanner",
                    "image": "mt-scanner.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Scanner", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "mt-scanner.webp", },
                        { "id": "WR", "type": "bonus", "label": "Waveform Recycler", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "waveform.webp", },
                        { "id": "SH", "type": "bonus", "label": "Scan Harmonizer", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "harmonizer.webp", },
                        { "id": "PC", "type": "bonus", "label": "Polyphonic Core", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "polyphonic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scanner Upgrade Theta", "bonus": 1.50, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scanner Upgrade Tau", "bonus": 1.45, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scanner Upgrade Sigma", "bonus": 1.40, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                    ],
                },
                {
                    "label": "Survey Device",
                    "key": "survey",
                    "image": "survey.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "SD", "type": "core", "label": "Survey Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "survey.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Blaze Javelin",
                    "key": "blaze-javelin",
                    "image": "blaze-javelin.webp",
                    "color": "red",
                    "modules": [
                        { "id": "BJ", "type": "core", "label": "Blaze Javelin", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "blaze-javelin.webp", },
                        { "id": "MA", "type": "bonus", "label": "Mass Accelerator", "bonus": .03, "adjacency": "greater", "sc_eligible": True, "image": "mass-accelerator.webp", },
                        { "id": "WO", "type": "bonus", "label": "Waveform Oscillator", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "waveform-osc.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Blaze Javelin Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Blaze Javelin Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Blaze Javelin Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                    ],
                },
                {
                    "label": "Boltcaster",
                    "key": "bolt-caster",
                    "image": "boltcaster.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "BC", "type": "core", "label": "Bolt Caster", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "boltcaster.webp", },
                        { "id": "RM", "type": "bonus", "label": "Boltcaster Ricochet Module", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-rm.webp", },
                        { "id": "BI", "type": "bonus", "label": "Barrel Ionizer", "bonus": 0.1, "adjacency": "greater", "sc_eligible": True, "image": "barrel-ionizer.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Boltcaster Upgrade Theta", "bonus": 0.15, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Boltcaster Upgrade Tau", "bonus": 0.14, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Boltcaster Upgrade Sigma", "bonus": 0.13, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Fa", "type": "bonus", "label": "Forbidden Module Theta", "bonus": 0.22, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                        { "id": "Fb", "type": "bonus", "label": "Forbidden Module Tau", "bonus": 0.21, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                        { "id": "Fc", "type": "bonus", "label": "Forbidden Module Sigma", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                    ],
                },
                {
                    "label": "Geology Cannon",
                    "key": "geology",
                    "image": "geology.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "GC", "type": "core", "label": "Geology Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "geology.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Geology Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Geology Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Geology Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                    ],
                },
                {
                    "label": "Neutron Cannon",
                    "key": "neutron",
                    "image": "neutron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "NC", "type": "core", "label": "Neutron Cannon", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "neutron.webp", },
                        { "id": "PF", "type": "bonus", "label": "P-Field Compressor", "bonus": 1.62, "adjacency": "lesser", "sc_eligible": True, "image": "p-field.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Neutron Cannon Upgrade Theta", "bonus": 3.15, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Neutron Cannon Upgrade Tau", "bonus": 3.14, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Neutron Cannon Upgrade Sigma", "bonus": 3.13, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Plasma Launcher",
                    "key": "plasma-launcher",
                    "image": "plasma-launcher.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "PL", "type": "core", "label": "Plasma Launcher", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "plasma-launcher.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Plasma Launcher Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Plasma Launcher Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Plasma Launcher Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Spitter",
                    "key": "pulse-splitter",
                    "image": "pulse-splitter.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "PS", "type": "core", "label": "Pulse Spitter", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter.webp", },
                        { "id": "AC", "type": "bonus", "label": "Amplified Cartridges", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "amplified.webp", },
                        { "id": "RM", "type": "bonus", "label": "Richochet Module", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-rm.webp", },
                        { "id": "II", "type": "bonus", "label": "Impact Ignitor", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "impact-ignitor.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Spitter Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Spitter Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Spitter Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                    ],
                },
                {
                    "label": "Scatter Blaster",
                    "key": "scatter",
                    "image": "scatter.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "SB", "type": "core", "label": "Scatter Blaster", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "scatter.webp", },
                        { "id": "SG", "type": "bonus", "label": "Shell Greaser", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "shell-greaser.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scatter Blaster Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scatter Blaster Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scatter Blaster Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                    ],
                },
            ],
            "Secondary Weapons": [
                {
                    "label": "Cloaking Device",
                    "key": "cloaking",
                    "image": "cloaking.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cloaking Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "cloaking.webp", },
                    ],
                },
                {
                    "label": "Combat Scope",
                    "key": "combat",
                    "image": "combat.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Combat Scope", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "combat.webp", },
                    ],
                },
                {
                    "label": "Voltaic Amplifier",
                    "key": "voltaic-amplifier",
                    "image": "voltaic-amplifier.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "VA", "type": "core", "label": "Voltaic Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "voltaic-amplifier.webp", },
                    ],
                },
                {
                    "label": "Paralysis Mortar",
                    "key": "paralysis",
                    "image": "paralysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PM", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "paralysis.webp", },
                    ],
                },
                {
                    "label": "Personal Forcefield",
                    "key": "personal",
                    "image": "personal.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PF", "type": "core", "label": "Personal Forcefield", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "personal.webp", },
                    ],
                },
                {
                    "label": "Terrain Manipulator",
                    "key": "terrian",
                    "image": "terrian.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TM", "type": "core", "label": "Fishing Rig", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "terrian.webp", },
                    ],
                },
            ],
        },
    },
    "atlantid": {
        "label": "Atlantid Multi-Tools",
        "type": "Multi-Tool",
        "types": {
            "Mining": [
                {
                    "label": "Mining Beam",
                    "key": "mining",
                    "image": "mining-beam.webp",
                    "color": "green",
                    "modules": [
                        { "id": "MB", "type": "core", "label": "Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-laser.webp", },
                        { "id": "AM", "type": "bonus", "label": "Advanced Mining Laser", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "advanced-mining.webp", },
                        { "id": "OD", "type": "bonus", "label": "Optical Drill", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "optical.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mining Laser Upgrade Theta", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mining Laser Upgrade Tau", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mining Laser Upgrade Sigma", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                        { "id": "RL", "type": "bonus", "label": "Runic Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "runic-laser.webp", },
                    ],
                },
            ],
            "Scanner": [
                {
                    "label": "Analysis Visor",
                    "key": "analysis",
                    "image": "analysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AV", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "analysis.webp", },
                    ],
                },
                {
                    "label": "Fishing Rig",
                    "key": "fishing",
                    "image": "fishing.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "FR", "type": "core", "label": "F", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "fishing.webp", },
                    ],
                },
                {
                    "label": "Scanner",
                    "key": "scanner",
                    "image": "mt-scanner.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Scanner", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "mt-scanner.webp", },
                        { "id": "WR", "type": "bonus", "label": "Waveform Recycler", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "waveform.webp", },
                        { "id": "SH", "type": "bonus", "label": "Scan Harmonizer", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "harmonizer.webp", },
                        { "id": "PC", "type": "bonus", "label": "Polyphonic Core", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "polyphonic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scanner Upgrade Theta", "bonus": 1.50, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scanner Upgrade Tau", "bonus": 1.45, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scanner Upgrade Sigma", "bonus": 1.40, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                    ],
                },
                {
                    "label": "Survey Device",
                    "key": "survey",
                    "image": "survey.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "SD", "type": "core", "label": "Survey Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "survey.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Blaze Javelin",
                    "key": "blaze-javelin",
                    "image": "blaze-javelin.webp",
                    "color": "red",
                    "modules": [
                        { "id": "BJ", "type": "core", "label": "Blaze Javelin", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "blaze-javelin.webp", },
                        { "id": "MA", "type": "bonus", "label": "Mass Accelerator", "bonus": .03, "adjacency": "greater", "sc_eligible": True, "image": "mass-accelerator.webp", },
                        { "id": "WO", "type": "bonus", "label": "Waveform Oscillator", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "waveform-osc.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Blaze Javelin Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Blaze Javelin Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Blaze Javelin Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                    ],
                },
                {
                    "label": "Boltcaster",
                    "key": "bolt-caster",
                    "image": "boltcaster.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "BC", "type": "core", "label": "Bolt Caster", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "boltcaster.webp", },
                        { "id": "RM", "type": "bonus", "label": "Boltcaster Ricochet Module", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-rm.webp", },
                        { "id": "BI", "type": "bonus", "label": "Barrel Ionizer", "bonus": 0.1, "adjacency": "greater", "sc_eligible": True, "image": "barrel-ionizer.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Boltcaster Upgrade Theta", "bonus": 0.15, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Boltcaster Upgrade Tau", "bonus": 0.14, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Boltcaster Upgrade Sigma", "bonus": 0.13, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Fa", "type": "bonus", "label": "Forbidden Module Theta", "bonus": 0.22, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                        { "id": "Fb", "type": "bonus", "label": "Forbidden Module Tau", "bonus": 0.21, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                        { "id": "Fc", "type": "bonus", "label": "Forbidden Module Sigma", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                    ],
                },
                {
                    "label": "Geology Cannon",
                    "key": "geology",
                    "image": "geology.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "GC", "type": "core", "label": "Geology Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "geology.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Geology Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Geology Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Geology Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                    ],
                },
                {
                    "label": "Neutron Cannon",
                    "key": "neutron",
                    "image": "neutron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "NC", "type": "core", "label": "Neutron Cannon", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "neutron.webp", },
                        { "id": "PF", "type": "bonus", "label": "P-Field Compressor", "bonus": 1.62, "adjacency": "lesser", "sc_eligible": True, "image": "p-field.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Neutron Cannon Upgrade Theta", "bonus": 3.15, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Neutron Cannon Upgrade Tau", "bonus": 3.14, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Neutron Cannon Upgrade Sigma", "bonus": 3.13, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Plasma Launcher",
                    "key": "plasma-launcher",
                    "image": "plasma-launcher.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "PL", "type": "core", "label": "Plasma Launcher", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "plasma-launcher.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Plasma Launcher Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Plasma Launcher Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Plasma Launcher Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Spitter",
                    "key": "pulse-splitter",
                    "image": "pulse-splitter.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "PS", "type": "core", "label": "Pulse Spitter", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter.webp", },
                        { "id": "AC", "type": "bonus", "label": "Amplified Cartridges", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "amplified.webp", },
                        { "id": "RM", "type": "bonus", "label": "Richochet Module", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-rm.webp", },
                        { "id": "II", "type": "bonus", "label": "Impact Ignitor", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "impact-ignitor.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Spitter Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Spitter Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Spitter Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                    ],
                },
                {
                    "label": "Scatter Blaster",
                    "key": "scatter",
                    "image": "scatter.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "SB", "type": "core", "label": "Scatter Blaster", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "scatter.webp", },
                        { "id": "SG", "type": "bonus", "label": "Shell Greaser", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "shell-greaser.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scatter Blaster Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scatter Blaster Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scatter Blaster Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                    ],
                },
            ],
            "Secondary Weapons": [
                {
                    "label": "Cloaking Device",
                    "key": "cloaking",
                    "image": "cloaking.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cloaking Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "cloaking.webp", },
                    ],
                },
                {
                    "label": "Combat Scope",
                    "key": "combat",
                    "image": "combat.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Combat Scope", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "combat.webp", },
                    ],
                },
                {
                    "label": "Voltaic Amplifier",
                    "key": "voltaic-amplifier",
                    "image": "voltaic-amplifier.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "VA", "type": "core", "label": "Voltaic Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "voltaic-amplifier.webp", },
                    ],
                },
                {
                    "label": "Paralysis Mortar",
                    "key": "paralysis",
                    "image": "paralysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PM", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "paralysis.webp", },
                    ],
                },
                {
                    "label": "Personal Forcefield",
                    "key": "personal",
                    "image": "personal.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PF", "type": "core", "label": "Personal Forcefield", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "personal.webp", },
                    ],
                },
                {
                    "label": "Terrain Manipulator",
                    "key": "terrian",
                    "image": "terrian.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TM", "type": "core", "label": "Fishing Rig", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "terrian.webp", },
                    ],
                },
            ],
        },
    }, 
    "sentinel-mt": {
        "label": "Sentinel Multi-Tools",
        "type": "Multi-Tool",
        "types": {
            "Mining": [
                {
                    "label": "Hijacked Laser",
                    "key": "mining",
                    "image": "hijacked.webp",
                    "color": "green",
                    "modules": [
                        { "id": "MB", "type": "core", "label": "Hijacked Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "hijacked.webp", },
                        { "id": "AM", "type": "bonus", "label": "Advanced Mining Laser", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "advanced-mining.webp", },
                        { "id": "OD", "type": "bonus", "label": "Optical Drill", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "optical.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mining Laser Upgrade Theta", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mining Laser Upgrade Tau", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mining Laser Upgrade Sigma", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                    ],
                },
            ],
            "Scanner": [
                {
                    "label": "Analysis Visor",
                    "key": "analysis",
                    "image": "analysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AV", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "analysis.webp", },
                    ],
                },
                {
                    "label": "Fishing Rig",
                    "key": "fishing",
                    "image": "fishing.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "FR", "type": "core", "label": "F", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "fishing.webp", },
                    ],
                },
                {
                    "label": "Scanner",
                    "key": "scanner",
                    "image": "mt-scanner.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Scanner", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "mt-scanner.webp", },
                        { "id": "WR", "type": "bonus", "label": "Waveform Recycler", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "waveform.webp", },
                        { "id": "SH", "type": "bonus", "label": "Scan Harmonizer", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "harmonizer.webp", },
                        { "id": "PC", "type": "bonus", "label": "Polyphonic Core", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "polyphonic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scanner Upgrade Theta", "bonus": 1.50, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scanner Upgrade Tau", "bonus": 1.45, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scanner Upgrade Sigma", "bonus": 1.40, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                    ],
                },
                {
                    "label": "Survey Device",
                    "key": "survey",
                    "image": "survey.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "SD", "type": "core", "label": "Survey Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "survey.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Blaze Javelin",
                    "key": "blaze-javelin",
                    "image": "blaze-javelin.webp",
                    "color": "red",
                    "modules": [
                        { "id": "BJ", "type": "core", "label": "Blaze Javelin", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "blaze-javelin.webp", },
                        { "id": "MA", "type": "bonus", "label": "Mass Accelerator", "bonus": .03, "adjacency": "greater", "sc_eligible": True, "image": "mass-accelerator.webp", },
                        { "id": "WO", "type": "bonus", "label": "Waveform Oscillator", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "waveform-osc.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Blaze Javelin Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Blaze Javelin Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Blaze Javelin Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                    ],
                },
                {
                    "label": "Boltcaster",
                    "key": "bolt-caster",
                    "image": "boltcaster.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "BC", "type": "core", "label": "Bolt Caster", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "boltcaster.webp", },
                        { "id": "RM", "type": "bonus", "label": "Boltcaster Ricochet Module", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-rm.webp", },
                        { "id": "BI", "type": "bonus", "label": "Barrel Ionizer", "bonus": 0.1, "adjacency": "greater", "sc_eligible": True, "image": "barrel-ionizer.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Boltcaster Upgrade Theta", "bonus": 0.15, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Boltcaster Upgrade Tau", "bonus": 0.14, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Boltcaster Upgrade Sigma", "bonus": 0.13, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Fa", "type": "bonus", "label": "Forbidden Module Theta", "bonus": 0.22, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                        { "id": "Fb", "type": "bonus", "label": "Forbidden Module Tau", "bonus": 0.21, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                        { "id": "Fc", "type": "bonus", "label": "Forbidden Module Sigma", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                    ],
                },
                {
                    "label": "Geology Cannon",
                    "key": "geology",
                    "image": "geology.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "GC", "type": "core", "label": "Geology Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "geology.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Geology Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Geology Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Geology Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                    ],
                },
                {
                    "label": "Neutron Cannon",
                    "key": "neutron",
                    "image": "neutron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "NC", "type": "core", "label": "Neutron Cannon", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "neutron.webp", },
                        { "id": "PF", "type": "bonus", "label": "P-Field Compressor", "bonus": 1.62, "adjacency": "lesser", "sc_eligible": True, "image": "p-field.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Neutron Cannon Upgrade Theta", "bonus": 3.15, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Neutron Cannon Upgrade Tau", "bonus": 3.14, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Neutron Cannon Upgrade Sigma", "bonus": 3.13, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Plasma Launcher",
                    "key": "plasma-launcher",
                    "image": "plasma-launcher.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "PL", "type": "core", "label": "Plasma Launcher", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "plasma-launcher.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Plasma Launcher Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Plasma Launcher Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Plasma Launcher Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Spitter",
                    "key": "pulse-splitter",
                    "image": "pulse-splitter.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "PS", "type": "core", "label": "Pulse Spitter", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter.webp", },
                        { "id": "AC", "type": "bonus", "label": "Amplified Cartridges", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "amplified.webp", },
                        { "id": "RM", "type": "bonus", "label": "Richochet Module", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-rm.webp", },
                        { "id": "II", "type": "bonus", "label": "Impact Ignitor", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "impact-ignitor.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Spitter Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Spitter Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Spitter Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                    ],
                },
                {
                    "label": "Scatter Blaster",
                    "key": "scatter",
                    "image": "scatter.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "SB", "type": "core", "label": "Scatter Blaster", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "scatter.webp", },
                        { "id": "SG", "type": "bonus", "label": "Shell Greaser", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "shell-greaser.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scatter Blaster Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scatter Blaster Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scatter Blaster Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                    ],
                },
            ],
            "Secondary Weapons": [
                {
                    "label": "Cloaking Device",
                    "key": "cloaking",
                    "image": "cloaking.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cloaking Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "cloaking.webp", },
                    ],
                },
                {
                    "label": "Combat Scope",
                    "key": "combat",
                    "image": "combat.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Combat Scope", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "combat.webp", },
                    ],
                },
                {
                    "label": "Voltaic Amplifier",
                    "key": "voltaic-amplifier",
                    "image": "voltaic-amplifier.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "VA", "type": "core", "label": "Voltaic Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "voltaic-amplifier.webp", },
                    ],
                },
                {
                    "label": "Paralysis Mortar",
                    "key": "paralysis",
                    "image": "paralysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PM", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "paralysis.webp", },
                    ],
                },
                {
                    "label": "Personal Forcefield",
                    "key": "personal",
                    "image": "personal.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PF", "type": "core", "label": "Personal Forcefield", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "personal.webp", },
                    ],
                },
                {
                    "label": "Terrain Manipulator",
                    "key": "terrian",
                    "image": "terrian.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TM", "type": "core", "label": "Fishing Rig", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "terrian.webp", },
                    ],
                },
            ],
        }
    },
    "staves": {
        "label": "Voltaic / Expedition Staves",
        "type": "Multi-Tool",
        "types": {
            "Mining": [
                {
                    "label": "Mining Beam",
                    "key": "mining",
                    "image": "mining-beam.webp",
                    "color": "green",
                    "modules": [
                        { "id": "MB", "type": "core", "label": "Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-laser.webp", },
                        { "id": "AM", "type": "bonus", "label": "Advanced Mining Laser", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "advanced-mining.webp", },
                        { "id": "OD", "type": "bonus", "label": "Optical Drill", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "optical.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mining Laser Upgrade Theta", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mining Laser Upgrade Tau", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mining Laser Upgrade Sigma", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "mining-upgrade.webp", },
                    ],
                },
            ],
            "Scanner": [
                {
                    "label": "Analysis Visor",
                    "key": "analysis",
                    "image": "analysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AV", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "analysis.webp", },
                    ],
                },
                {
                    "label": "Fishing Rig",
                    "key": "fishing",
                    "image": "fishing.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "FR", "type": "core", "label": "F", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "fishing.webp", },
                    ],
                },
                {
                    "label": "Scanner",
                    "key": "scanner",
                    "image": "mt-scanner.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Scanner", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "mt-scanner.webp", },
                        { "id": "WR", "type": "bonus", "label": "Waveform Recycler", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "waveform.webp", },
                        { "id": "SH", "type": "bonus", "label": "Scan Harmonizer", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "harmonizer.webp", },
                        { "id": "PC", "type": "bonus", "label": "Polyphonic Core", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "polyphonic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scanner Upgrade Theta", "bonus": 1.50, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scanner Upgrade Tau", "bonus": 1.45, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scanner Upgrade Sigma", "bonus": 1.40, "adjacency": "greater", "sc_eligible": True, "image": "scanner-upgrade.webp", },
                    ],
                },
                {
                    "label": "Survey Device",
                    "key": "survey",
                    "image": "survey.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "SD", "type": "core", "label": "Survey Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "survey.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Blaze Javelin",
                    "key": "blaze-javelin",
                    "image": "blaze-javelin.webp",
                    "color": "red",
                    "modules": [
                        { "id": "BJ", "type": "core", "label": "Blaze Javelin", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "blaze-javelin.webp", },
                        { "id": "MA", "type": "bonus", "label": "Mass Accelerator", "bonus": .03, "adjacency": "greater", "sc_eligible": True, "image": "mass-accelerator.webp", },
                        { "id": "WO", "type": "bonus", "label": "Waveform Oscillator", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "waveform-osc.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Blaze Javelin Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Blaze Javelin Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Blaze Javelin Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "blaze-upgrade.webp", },
                    ],
                },
                {
                    "label": "Boltcaster",
                    "key": "bolt-caster",
                    "image": "boltcaster.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "BC", "type": "core", "label": "Bolt Caster", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "boltcaster.webp", },
                        { "id": "RM", "type": "bonus", "label": "Boltcaster Ricochet Module", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-rm.webp", },
                        { "id": "BI", "type": "bonus", "label": "Barrel Ionizer", "bonus": 0.1, "adjacency": "greater", "sc_eligible": True, "image": "barrel-ionizer.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Boltcaster Upgrade Theta", "bonus": 0.15, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Boltcaster Upgrade Tau", "bonus": 0.14, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Boltcaster Upgrade Sigma", "bonus": 0.13, "adjacency": "greater", "sc_eligible": True, "image": "boltcaster-upgrade.webp", },
                        { "id": "Fa", "type": "bonus", "label": "Forbidden Module Theta", "bonus": 0.22, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                        { "id": "Fb", "type": "bonus", "label": "Forbidden Module Tau", "bonus": 0.21, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                        { "id": "Fc", "type": "bonus", "label": "Forbidden Module Sigma", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "forbidden.webp", },
                    ],
                },
                {
                    "label": "Geology Cannon",
                    "key": "geology",
                    "image": "geology.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "GC", "type": "core", "label": "Geology Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "geology.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Geology Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Geology Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Geology Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "geology-upgrade.webp", },
                    ],
                },
                 {
                    "label": "Neutron Cannon",
                    "key": "neutron",
                    "image": "neutron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "NC", "type": "core", "label": "Neutron Cannon", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "neutron.webp", },
                        { "id": "PF", "type": "bonus", "label": "P-Field Compressor", "bonus": 1.62, "adjacency": "lesser", "sc_eligible": True, "image": "p-field.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Neutron Cannon Upgrade Theta", "bonus": 3.15, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Neutron Cannon Upgrade Tau", "bonus": 3.14, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Neutron Cannon Upgrade Sigma", "bonus": 3.13, "adjacency": "greater", "sc_eligible": True, "image": "neutron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Plasma Launcher",
                    "key": "plasma-launcher",
                    "image": "plasma-launcher.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "PL", "type": "core", "label": "Plasma Launcher", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "plasma-launcher.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Plasma Launcher Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Plasma Launcher Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Plasma Launcher Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "plasma-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Spitter",
                    "key": "pulse-splitter",
                    "image": "pulse-splitter.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "PS", "type": "core", "label": "Pulse Spitter", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter.webp", },
                        { "id": "AC", "type": "bonus", "label": "Amplified Cartridges", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "amplified.webp", },
                        { "id": "RM", "type": "bonus", "label": "Richochet Module", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-rm.webp", },
                        { "id": "II", "type": "bonus", "label": "Impact Ignitor", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "impact-ignitor.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Spitter Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Spitter Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Spitter Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "pulse-splitter-upgrade.webp", },
                    ],
                },
                {
                    "label": "Scatter Blaster",
                    "key": "scatter",
                    "image": "scatter.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "SB", "type": "core", "label": "Scatter Blaster", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "scatter.webp", },
                        { "id": "SG", "type": "bonus", "label": "Shell Greaser", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "shell-greaser.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scatter Blaster Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scatter Blaster Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scatter Blaster Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "scatter-upgrade.webp", },
                    ],
                },
            ],
            "Secondary Weapons": [
                {
                    "label": "Cloaking Device",
                    "key": "cloaking",
                    "image": "cloaking.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cloaking Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "cloaking.webp", },
                    ],
                },
                {
                    "label": "Combat Scope",
                    "key": "combat",
                    "image": "combat.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Combat Scope", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "combat.webp", },
                    ],
                },
                {
                    "label": "Voltaic Amplifier",
                    "key": "voltaic-amplifier",
                    "image": "voltaic-amplifier.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "VA", "type": "core", "label": "Voltaic Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "voltaic-amplifier.webp", },
                    ],
                },
                {
                    "label": "Paralysis Mortar",
                    "key": "paralysis",
                    "image": "paralysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PM", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "paralysis.webp", },
                    ],
                },
                {
                    "label": "Personal Forcefield",
                    "key": "personal",
                    "image": "personal.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PF", "type": "core", "label": "Personal Forcefield", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "personal.webp", },
                    ],
                },
                {
                    "label": "Terrain Manipulator",
                    "key": "terrian",
                    "image": "terrian.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TM", "type": "core", "label": "Fishing Rig", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "terrian.webp", },
                    ],
                },
            ],
        },
    },
    "freighter": {
        "label": "Freighters",
        "type": "Starship",
        "types": {
            "Hyperdrive": [
                {
                    "label": "Freighter Hyperdrive",
                    "key": "hyper",
                    "image": "freighter-hyper.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "HD", "type": "core", "label": "Hyperdrive", "bonus": 0.10, "adjacency": "lesser", "sc_eligible": True, "image": "freighter-hyperdrive.webp", },
                        { "id": "TW", "type": "bonus", "label": "Temporal Warp Computer", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "freighter-temporal.webp", },
                        { "id": "RD", "type": "bonus", "label": "Reality De-Threader", "bonus": 0.80, "adjacency": "greater", "sc_eligible": True, "image": "freighter-reality.webp", },
                        { "id": "RM", "type": "bonus", "label": "Resonance Matrix", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "freighter-resonance.webp", },
                        { "id": "PW", "type": "bonus", "label": "Plasmatic Warp Injector", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "freighter-plasmatic.webp", },
                        { "id": "AW", "type": "bonus", "label": "Amplified Warp Shielding", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "freighter-amplified.webp", },
                        { "id": "WC", "type": "bonus", "label": "Warp Core Resonator", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "freighter-warpcore.webp", },
                        { "id": "CW", "type": "bonus", "label": "Chromatic Warp Shielding", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "freighter-chromatic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Hyperdrive Upgrade Theta", "bonus": .25, "adjacency": "greater", "sc_eligible": True, "image": "freighter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Hyperdrive Upgrade Tau", "bonus": .24, "adjacency": "greater", "sc_eligible": True, "image": "freighter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Hyperdrive Upgrade Sigma", "bonus": .23, "adjacency": "greater", "sc_eligible": True, "image": "freighter-upgrade.webp", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Interstellar Scanner",
                    "key": "interstellar",
                    "image": "freighter-scanner.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "IS", "type": "core", "label": "Interstellar Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "freighter-scanner.webp", },
                    ],
                },
                {
                    "label": "Matter Beam",
                    "key": "matterbeam",
                    "image": "freighter-matter.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "MB", "type": "core", "label": "Matter Beam", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "freighter-matter.webp", },
                    ],
                },
            ],
            "Fleet Upgrades": [
                {
                    "label": "Fuel Efficiency",
                    "key": "fleet-fuel",
                    "image": "fleet-fuel.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Fuel Efficiency Upgrade Theta", "bonus": .300, "adjacency": "greater", "sc_eligible": True, "image": "fleet-fuel.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Fuel Efficiency Upgrade Tau", "bonus": .290, "adjacency": "greater", "sc_eligible": True, "image": "fleet-fuel.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Fuel EfficiencyUpgrade Sigma", "bonus": .280, "adjacency": "greater", "sc_eligible": True, "image": "fleet-fuel.webp", },
                      ],
                },
                {
                    "label": "Expedition Speed",
                    "key": "fleet-speed",
                    "image": "fleet-speed.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Expedition Speed Upgrade Theta", "bonus": .300, "adjacency": "greater", "sc_eligible": True, "image": "fleet-speed.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Expedition Speed Upgrade Tau", "bonus": .290, "adjacency": "greater", "sc_eligible": True, "image": "fleet-speed.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Expedition Speed Upgrade Sigma", "bonus": .280, "adjacency": "greater", "sc_eligible": True, "image": "fleet-speed.webp", },
                      ],
                },
                {
                    "label": "Combat and Defense",
                    "key": "fleet-combat",
                    "image": "fleet-combat.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Combat and Defense Upgrade Theta", "bonus": .300, "adjacency": "greater", "sc_eligible": True, "image": "fleet-combat.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Combat and Defense Upgrade Tau", "bonus": .290, "adjacency": "greater", "sc_eligible": True, "image": "fleet-combat.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Combat and Defense Upgrade Sigma", "bonus": .280, "adjacency": "greater", "sc_eligible": True, "image": "fleet-combat.webp", },
                      ],
                },
                {
                    "label": "Exploration and Science",
                    "key": "fleet-exploration",
                    "image": "fleet-exploration.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Exploration and Science Upgrade Theta", "bonus": .300, "adjacency": "greater", "sc_eligible": True, "image": "fleet-exploration.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Exploration and Science Upgrade Tau", "bonus": .290, "adjacency": "greater", "sc_eligible": True, "image": "fleet-exploration.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Exploration and Science Upgrade Sigma", "bonus": .280, "adjacency": "greater", "sc_eligible": True, "image": "fleet-exploration.webp", },
                      ],
                },
                {
                    "label": "Mining and Industrial",
                    "key": "fleet-mining",
                    "image": "fleet-mining.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Theta", "bonus": .300, "adjacency": "greater", "sc_eligible": True, "image": "fleet-mining.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Tau", "bonus": .290, "adjacency": "greater", "sc_eligible": True, "image": "fleet-mining.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Sigma", "bonus": .280, "adjacency": "greater", "sc_eligible": True, "image": "fleet-mining.webp", },
                      ],
                },
                {
                    "label": "Trade",
                    "key": "fleet-trade",
                    "image": "fleet-trade.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Theta", "bonus": .300, "adjacency": "greater", "sc_eligible": True, "image": "fleet-trade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Tau", "bonus": .290, "adjacency": "greater", "sc_eligible": True, "image": "fleet-trade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Sigma", "bonus": .280, "adjacency": "greater", "sc_eligible": True, "image": "fleet-trade.webp", },
                      ],
                },
            ],
        },
    },
}

solves = {
    "standard": {
        "cyclotron": {
            "map": {
                (0, 0): "QR",
                (1, 0): "None",
                (0, 1): "CB",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5411
        },
        "infra": {
            "map": {
                (0, 0): "QR",
                (1, 0): "None",
                (0, 1): "IK",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5304
        },
        "phase": {
            "map": {
                (0, 0): "FD",
                (1, 0): "None",
                (0, 1): "PB",
                (1, 1): "Xb",
                (0, 2): "Xc",
                (1, 2): "Xa"
            },
            "score": 2.398
        },
        "photon": {
            "map": {
                (0, 0): "NO",
                (1, 0): "None",
                (0, 1): "PC",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5411
        },
        "positron": {
            "map": {
                (0, 0): "FS",
                (1, 0): "None",
                (0, 1): "PE",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5411
        },
        "rocket": {
            "map": {
                (0, 0): "RL",
                (0, 1): "LR",
                (0, 2): "None"
            },
            "score": 1.1199
        },
        "shield": {
            "map": {
                (0, 0): "DS",
                (1, 0): "None",
                (0, 1): "Xa",
                (1, 1): "AA",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 1.0749
        },
        "hyper": {
            "map": {
                (0, 0): "AD",
                (1, 0): "HD",
                (2, 0): "CD",
                (0, 1): "ED",
                (1, 1): "Xa",
                (2, 1): "Xb",
                (0, 2): "ID",
                (1, 2): "Xc",
                (2, 2): "EW"
            },
            "score": 1.1524
        },
        "launch": {
            "map": {
                (0, 0): "LT",
                (1, 0): "EF",
                (0, 1): "Xb",
                (1, 1): "Xa",
                (0, 2): "RC",
                (1, 2): "Xc"
            },
            "score": 1.0142
        },
        "pulse": {
            "map": {
                (0, 0): "PE",
                (1, 0): "Xc",
                (2, 0): "Xa",
                (3, 0): "SL",
                (0, 1): "ID",
                (1, 1): "Xb",
                (2, 1): "FA",
                (3, 1): "None",
            },
            "score": 6.1562
        },
        "photonix": {
            "map": {
                (0, 0): "PE",
                (1, 0): "PC",
                (2, 0): "FA",
                (0, 1): "None",
                (1, 1): "Xa",
                (2, 1): "Xb",
                (0, 2): "SL",
                (1, 2): "Xc",
                (2, 2): "ID"
            },
            "score": 7.4546
        },
        "trails": {
            "map": {
                (0, 0): "TT",
                (1, 0): "RT",
                (2, 0): "CT",
                (3, 0): "None",
                (0, 1): "SB",
                (1, 1): "AB",
                (2, 1): "PB",
                (3, 1): "ET",
                (0, 2): "SP",
                (1, 2): "GT",
                (2, 2): "ST",
                (3, 2): "None"
            },
            "score": 0.3595
        }
    },
    "sentinel": {
        "cyclotron": {
            "map": {
                (0, 0): "QR",
                (1, 0): "None",
                (0, 1): "CB",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5411
        },
        "infra": {
            "map": {
                (0, 0): "QR",
                (1, 0): "None",
                (0, 1): "IK",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5411
        },
        "phase": {
            "map": {
                (0, 0): "FD",
                (1, 0): "None",
                (0, 1): "PB",
                (1, 1): "Xb",
                (0, 2): "Xc",
                (1, 2): "Xa"
            },
            "score": 2.398
        },
        "positron": {
            "map": {
                (0, 0): "FS",
                (1, 0): "None",
                (0, 1): "PE",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5411
        },
        "rocket": {
            "map": {
                (0, 0): "RL",
                (0, 1): "LR",
                (0, 2): "None"
            },
            "score": 1.1199
        },
        "photon": {
            "map": {
                (0, 0): "NO",
                (1, 0): "None",
                (0, 1): "PC",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5411
        },
        "shield": {
            "map": {
                (0, 0): "DS",
                (1, 0): "None",
                (0, 1): "Xa",
                (1, 1): "AA",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 1.0749
        },
        "launch": {
            "map": {
                (0, 0): "LT",
                (1, 0): "EF",
                (0, 1): "Xb",
                (1, 1): "Xa",
                (0, 2): "RC",
                (1, 2): "Xc"
            },
            "score": 1.0142
        },
        "hyper": {
            "map": {
                (0, 0): "AD",
                (1, 0): "HD",
                (2, 0): "CD",
                (0, 1): "ED",
                (1, 1): "Xa",
                (2, 1): "Xb",
                (0, 2): "ID",
                (1, 2): "Xc",
                (2, 2): "EW"
            },
            "score": 1.1524
        },
        "pulse": {
            "map": {
                (0, 0): "PE",
                (1, 0): "Xc",
                (2, 0): "Xa",
                (3, 0): "SL",
                (0, 1): "ID",
                (1, 1): "Xb",
                (2, 1): "FA",
                (3, 1): "None",
            },
            "score": 6.1562
        },
        "photonix": {
            "map": {
                (0, 0): "PE",
                (1, 0): "PC",
                (2, 0): "Xa",
                (3, 0): "FA",
                (0, 1): "SL",
                (1, 1): "Xb",
                (2, 1): "Xc",
                (3, 1): "ID"
            },
            "score": 7.4546
        },
        "trails": {
            "map": {
                (0, 0): "TT",
                (1, 0): "RT",
                (2, 0): "CT",
                (3, 0): "None",
                (0, 1): "SB",
                (1, 1): "AB",
                (2, 1): "PB",
                (3, 1): "ET",
                (0, 2): "SP",
                (1, 2): "GT",
                (2, 2): "ST",
                (3, 2): "None"
            },
            "score": 0.3595
        }
    },
    "solar": {
        "cyclotron": {
            "map": {
                (0, 0): "QR",
                (1, 0): "None",
                (0, 1): "CB",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5411
        },
        "infra": {
            "map": {
                (0, 0): "QR",
                (1, 0): "None",
                (0, 1): "IK",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5304
        },
        "phase": {
            "map": {
                (0, 0): "FD",
                (1, 0): "None",
                (0, 1): "PB",
                (1, 1): "Xb",
                (0, 2): "Xc",
                (1, 2): "Xa"
            },
            "score": 2.398
        },
        "photon": {
            "map": {
                (0, 0): "NO",
                (1, 0): "None",
                (0, 1): "PC",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5411
        },
        "positron": {
            "map": {
                (0, 0): "FS",
                (1, 0): "None",
                (0, 1): "PE",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5411
        },
        "rocket": {
            "map": {
                (0, 0): "RL",
                (0, 1): "LR",
                (0, 2): "None"
            },
            "score": 1.1199
        },
        "shield": {
            "map": {
                (0, 0): "DS",
                (1, 0): "None",
                (0, 1): "Xa",
                (1, 1): "AA",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 1.0749
        },
        "hyper": {
            "map": {
                (0, 0): "AD",
                (1, 0): "HD",
                (2, 0): "ED",
                (0, 1): "Xb",
                (1, 1): "Xa",
                (2, 1): "CD",
                (0, 2): "EW",
                (1, 2): "Xc",
                (2, 2): "ID"
            },
            "score": 1.1524
        },
        "launch": {
            "map": {
                (0, 0): "LT",
                (1, 0): "EF",
                (0, 1): "Xb",
                (1, 1): "Xa",
                (0, 2): "RC",
                (1, 2): "Xc"
            },
            "score": 1.0142
        },
        "pulse": {
            "map": {
                (0, 0): "PE",
                (1, 0): "VS",
                (2, 0): "Xa",
                (3, 0): "FA",
                (0, 1): "SL",
                (1, 1): "Xb",
                (2, 1): "Xc",
                (3, 1): "ID"
            },
            "score": 7.4546
        },
        "photonix": {
            "map": {
                (0, 0): "PE",
                (1, 0): "VS",
                (2, 0): "FA",
                (0, 1): "PC",
                (1, 1): "Xa",
                (2, 1): "Xb",
                (0, 2): "SL",
                (1, 2): "Xc",
                (2, 2): "ID"
            },
            "score": 8.7990
        },
        "trails": {
            "map": {
                (0, 0): "TT",
                (1, 0): "RT",
                (2, 0): "CT",
                (3, 0): "None",
                (0, 1): "SB",
                (1, 1): "AB",
                (2, 1): "PB",
                (3, 1): "ET",
                (0, 2): "SP",
                (1, 2): "GT",
                (2, 2): "ST",
                (3, 2): "None"
            },
            "score": 0.3595
        }
    },
    "living": {
        "grafted": {
            "map": {
                (0, 0): "GE",
                (1, 0): "Xb",
                (0, 1): "Xc",
                (1, 1): "Xa"
            },
            "score": 2.2952
        },
        "spewing": {
            "map": {
                (0, 0): "SV",
                (1, 0): "Xa",
                (0, 1): "Xb",
                (1, 1): "Xc"
            },
            "score": 2.4383
        },
        "scream": {
            "map": {
                (0, 0): "SS",
                (1, 0): "Xb",
                (0, 1): "Xc",
                (1, 1): "Xa"
            },
            "score": 0.9732
        },
        "assembly": {
            "map": {
                (0, 0): "NA",
                (1, 0): "None",
                (0, 1): "Xa",
                (1, 1): "Xb",
                (0, 2): "Xc",
                (1, 2): "CM"
            },
            "score": 0.9966
        },
        "singularity": {
            "map": {
                (0, 0): "SC",
                (1, 0): "None",
                (0, 1): "Xa",
                (1, 1): "AD",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 1.0645
        },
        "pulsing": {
            "map": {
                (0, 0): "PH",
                (1, 0): "Xb",
                (0, 1): "Xc",
                (1, 1): "Xa"
            },
            "score": 1.6372
        },
        "trails": {
            "map": {
                (0, 0): "TT",
                (1, 0): "RT",
                (2, 0): "CT",
                (3, 0): "None",
                (0, 1): "SB",
                (1, 1): "AB",
                (2, 1): "PB",
                (3, 1): "ET",
                (0, 2): "SP",
                (1, 2): "GT",
                (2, 2): "ST",
                (3, 2): "None"
            },
            "score": 0.3595
        }
    },
    "standard-mt": {
        "mining": {
            "map": {
                (0, 0): "OD",
                (1, 0): "Xa",
                (0, 1): "MB",
                (1, 1): "AM",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 1.775
        },
        "scanner": {
            "map": {
                (0, 0): "SC",
                (1, 0): "Xc",
                (2, 0): "Xa",
                (3, 0): "WR",
                (0, 1): "None",
                (1, 1): "SH",
                (2, 1): "Xb",
                (3, 1): "PC"
            },
            "score": 5.3615
        },
        "blaze-javelin": {
            "map": {
                (0, 0): "MA",
                (1, 0): "WO",
                (0, 1): "BJ",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5304
        },
        "bolt-caster": {
            "map": {
                (0, 0): "BC",
                (1, 0): "Xa",
                (2, 0): "RM",
                (0, 1): "Xb",
                (1, 1): "Fa",
                (2, 1): "Fb",
                (0, 2): "BI",
                (1, 2): "Fc",
                (2, 2): "Xc"
            },
            "score": 2.3508
        },
        "geology": {
            "map": {
                (0, 0): "GC",
                (1, 0): "Xa",
                (0, 1): "Xb",
                (1, 1): "Xc"
            },
            "score": 2.4383
        },
        "neutron": {
            "map": {
                (0, 0): "PF",
                (1, 0): "None",
                (0, 1): "NC",
                (1, 1): "Xb",
                (0, 2): "Xc",
                (1, 2): "Xa"
            },
            "score": 13.14
        },
        "plasma-launcher": {
            "map": {
                (0, 0): "PL",
                (1, 0): "Xa",
                (0, 1): "Xb",
                (1, 1): "Xc"
            },
            "score": 2.4383
        },
        "pulse-splitter": {
            "map": {
                (0, 0): "II",
                (1, 0): "AC",
                (2, 0): "None",
                (0, 1): "RM",
                (1, 1): "PS",
                (2, 1): "Xa",
                (0, 2): "None",
                (1, 2): "Xb",
                (2, 2): "Xc"
            },
            "score": 2.6225
        },
        "scatter": {
            "map": {
                (0, 0): "SG",
                (1, 0): "None",
                (0, 1): "SB",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.4983
        }
    },
    "atlantid": {
        "mining": {
            "map": {
                (0, 0): "AM",
                (1, 0): "MB",
                (2, 0): "OD",
                (3, 0): "Xa",
                (0, 1): "Xb",
                (1, 1): "RL",
                (2, 1): "Xc",
                (3, 1): "None"
            },
            "score": 3.417
        },
        "scanner": {
            "map": {
                (0, 0): "SC",
                (1, 0): "Xc",
                (2, 0): "Xa",
                (3, 0): "WR",
                (0, 1): "None",
                (1, 1): "SH",
                (2, 1): "Xb",
                (3, 1): "PC"
            },
            "score": 5.3615
        },
        "blaze-javelin": {
            "map": {
                (0, 0): "MA",
                (1, 0): "WO",
                (0, 1): "BJ",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5304
        },
        "bolt-caster": {
            "map": {
                (0, 0): "BC",
                (1, 0): "Xa",
                (2, 0): "RM",
                (0, 1): "Xb",
                (1, 1): "Fa",
                (2, 1): "Fb",
                (0, 2): "BI",
                (1, 2): "Fc",
                (2, 2): "Xc"
            },
            "score": 2.3508
        },
        "geology": {
            "map": {
                (0, 0): "GC",
                (1, 0): "Xa",
                (0, 1): "Xb",
                (1, 1): "Xc"
            },
            "score": 2.4383
        },
        "neutron": {
            "map": {
                (0, 0): "PF",
                (1, 0): "None",
                (0, 1): "NC",
                (1, 1): "Xb",
                (0, 2): "Xc",
                (1, 2): "Xa"
            },
            "score": 13.14
        },
        "plasma-launcher": {
            "map": {
                (0, 0): "PL",
                (1, 0): "Xa",
                (0, 1): "Xb",
                (1, 1): "Xc"
            },
            "score": 2.4383
        },
        "pulse-splitter": {
            "map": {
                (0, 0): "None",
                (1, 0): "AC",
                (2, 0): "None",
                (0, 1): "RM",
                (1, 1): "PS",
                (2, 1): "Xa",
                (0, 2): "II",
                (1, 2): "Xb",
                (2, 2): "Xc"
            },
            "score": 2.6225
        },
        "scatter": {
            "map": {
                (0, 0): "SG",
                (1, 0): "None",
                (0, 1): "SB",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.4983
        }
    },
    "sentinel-mt": {
        "mining": {
            "map": {
                (0, 0): "AM",
                (1, 0): "Xa",
                (0, 1): "MB",
                (1, 1): "OD",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.227
        },
        "scanner": {
            "map": {
                (0, 0): "SC",
                (1, 0): "Xc",
                (2, 0): "Xa",
                (3, 0): "WR",
                (0, 1): "None",
                (1, 1): "PC",
                (2, 1): "Xb",
                (3, 1): "SH"
            },
            "score": 5.3615
        },
        "blaze-javelin": {
            "map": {
                (0, 0): "MA",
                (1, 0): "WO",
                (0, 1): "BJ",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5304
        },
        "bolt-caster": {
            "map": {
                (0, 0): "BC",
                (1, 0): "Xa",
                (2, 0): "RM",
                (0, 1): "Xb",
                (1, 1): "Fa",
                (2, 1): "Fb",
                (0, 2): "BI",
                (1, 2): "Fc",
                (2, 2): "Xc"
            },
            "score": 2.3508
        },
        "geology": {
            "map": {
                (0, 0): "GC",
                (1, 0): "Xa",
                (0, 1): "Xb",
                (1, 1): "Xc"
            },
            "score": 2.4383
        },
        "neutron": {
            "map": {
                (0, 0): "PF",
                (1, 0): "None",
                (0, 1): "NC",
                (1, 1): "Xb",
                (0, 2): "Xc",
                (1, 2): "Xa"
            },
            "score": 13.14
        },
        "plasma-launcher": {
            "map": {
                (0, 0): "PL",
                (1, 0): "Xa",
                (0, 1): "Xb",
                (1, 1): "Xc"
            },
            "score": 2.4383
        },
        "pulse-splitter": {
            "map": {
                (0, 0): "II",
                (1, 0): "AC",
                (2, 0): "None",
                (0, 1): "RM",
                (1, 1): "PS",
                (2, 1): "Xa",
                (0, 2): "None",
                (1, 2): "Xb",
                (2, 2): "Xc"
            },
            "score": 2.6225
        },
        "scatter": {
            "map": {
                (0, 0): "SG",
                (1, 0): "None",
                (0, 1): "SB",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.4983
        }
    },
    "staves": {
        "mining": {
            "map": {
                (0, 0): "AM",
                (1, 0): "Xa",
                (0, 1): "MB",
                (1, 1): "OD",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.227
        },
        "scanner": {
            "map": {
                (0, 0): "SC",
                (1, 0): "Xc",
                (2, 0): "Xa",
                (3, 0): "SH",
                (0, 1): "None",
                (1, 1): "PC",
                (2, 1): "Xb",
                (3, 1): "WR"
            },
            "score": 5.3615
        },
        "blaze-javelin": {
            "map": {
                (0, 0): "MA",
                (1, 0): "WO",
                (0, 1): "BJ",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.5304
        },
        "bolt-caster": {
            "map": {
                (0, 0): "BC",
                (1, 0): "Xa",
                (2, 0): "RM",
                (0, 1): "Xb",
                (1, 1): "Fa",
                (2, 1): "Fb",
                (0, 2): "BI",
                (1, 2): "Fc",
                (2, 2): "Xc"
            },
            "score": 2.3508
        },
        "geology": {
            "map": {
                (0, 0): "GC",
                (1, 0): "Xa",
                (0, 1): "Xb",
                (1, 1): "Xc"
            },
            "score": 2.4383
        },
        "neutron": {
            "map": {
                (0, 0): "PF",
                (1, 0): "None",
                (0, 1): "NC",
                (1, 1): "Xb",
                (0, 2): "Xc",
                (1, 2): "Xa"
            },
            "score": 13.14
        },
        "plasma-launcher": {
            "map": {
                (0, 0): "PL",
                (1, 0): "Xa",
                (0, 1): "Xb",
                (1, 1): "Xc"
            },
            "score": 2.4383
        },
        "pulse-splitter": {
            "map": {
                (0, 0): "II",
                (1, 0): "AC",
                (2, 0): "None",
                (0, 1): "RM",
                (1, 1): "PS",
                (2, 1): "Xa",
                (0, 2): "None",
                (1, 2): "Xb",
                (2, 2): "Xc"
            },
            "score": 2.6225
        },
        "scatter": {
            "map": {
                (0, 0): "SG",
                (1, 0): "None",
                (0, 1): "SB",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "Xc"
            },
            "score": 2.4983
        }
    },
    "freighter": {
        "hyper": {
            "map": {
                (0, 0): "HD",
                (1, 0): "WC",
                (2, 0): "Xc",
                (3, 0): "AW",
                (0, 1): "RM",
                (1, 1): "PW",
                (2, 1): "RD",
                (3, 1): "Xb",
                (0, 2): "None",
                (1, 2): "TW",
                (2, 2): "Xa",
                (3, 2): "CW",
            },
            "score": 2.7688
        },
        "fleet-fuel": {
            "map": {
                (0, 0): "Xb",
                (0, 1): "Xa",
                (0, 2): "Xc"
            },
            "score": 0.9402
        },
        "fleet-speed": {
            "map": {
                (0, 0): "Xb",
                (0, 1): "Xa",
                (0, 2): "Xc"
            },
            "score": 0.9402
        },
        "fleet-combat": {
            "map": {
                (0, 0): "Xb",
                (0, 1): "Xa",
                (0, 2): "Xc"
            },
            "score": 0.9402
        },
        "fleet-exploration": {
            "map": {
                (0, 0): "Xb",
                (0, 1): "Xa",
                (0, 2): "Xc"
            },
            "score": 0.9402
        },
        "fleet-mining": {
            "map": {
                (0, 0): "Xb",
                (0, 1): "Xa",
                (0, 2): "Xc"
            },
            "score": 0.9402
        },
        "fleet-trade": {
            "map": {
                (0, 0): "Xb",
                (0, 1): "Xa",
                (0, 2): "Xc"
            },
            "score": 0.9402
        }
    }
}
# fmt:on
