# These value are highly generalized and do not reepresent the actual values in the game. They are ratios derrived from in-game experimentation and used for properly seeding the solving allgorithm.

# fmt: off
modules = {
    "standard": {
        "label": "Standard / Exotic Starships",
        "types": {
            "Weaponry": [
                {
                    "label": "Cyclotron Ballista",
                    "key": "cyclotron",
                    "image": "cyclotron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "CB", "type": "core", "label": "Cyclotron Ballista", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron.png", },
                        { "id": "QR", "type": "bonus", "label": "Dyson Pump", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "dyson.png", },
                        { "id": "Xa", "type": "bonus", "label": "Cyclotron Ballista Upgrade Sigma", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Cyclotron Ballista Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Cyclotron Ballista Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.png", },
                    ],
                },
                {
                    "label": "Infraknife Accelerator",
                    "key": "infra",
                    "image": "infra.webp",
                    "color": "red",
                    "modules": [
                        { "id": "IK", "type": "core", "label": "Infraknife Accelerator", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "infra.png", },
                        { "id": "QR", "type": "bonus", "label": "Q-Resonator", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "q-resonator.png", },
                        { "id": "Xa", "type": "bonus", "label": "Infraknife Accelerator Upgrade Sigma", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Infraknife Accelerator Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Infraknife Accelerator Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.png", },
                    ],
                },
                {
                    "label": "Phase Beam",
                    "key": "phase",
                    "image": "phase.webp",
                    "color": "green",
                    "modules": [
                        { "id": "PB", "type": "core", "label": "Phase Beam", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "phase-beam.png", },
                        { "id": "FD", "type": "bonus", "label": "Fourier De-Limiter", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "fourier.png", },
                        { "id": "Xa", "type": "bonus", "label": "Phase Beam Upgrade Sigma", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Phase Beam Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Phase Beam Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.png", },
                    ],
                },
                {
                    "label": "Photon Cannon",
                    "key": "photon",
                    "image": "photon.webp",
                    "color": "cyan",
                    "modules": [
                        { "id": "PC", "type": "core", "label": "Photon Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "photon.png", },
                        { "id": "NO", "type": "bonus", "label": "Nonlinear Optics", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "nonlinear.png", },
                        { "id": "Xa", "type": "bonus", "label": "Photon Cannon Upgrade Sigma", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Photon Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Photon Cannon Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.png", },
                    ],
                },
                {
                    "label": "Positron Ejector",
                    "key": "positron",
                    "image": "positron.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Positron Ejector", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "positron.png", },
                        { "id": "FS", "type": "bonus", "label": "Fragment Supercharger", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "fragment.png", },
                        { "id": "Xa", "type": "bonus", "label": "Positron Ejector Upgrade Sigma", "bonus": 0.4, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Positron Ejector Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Positron Ejector Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.png", },
                    ],
                },
                {
                    "label": "Rocket Launcher",
                    "key": "rocket",
                    "image": "rocket.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "RL", "type": "core", "label": "Rocket Launger", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "rocket.png", },
                        { "id": "LR", "type": "bonus", "label": "Large Rocket Tubes", "bonus": 0.056, "adjacency": "greater", "sc_eligible": True, "image": "tubes.png", },
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
                        { "id": "DS", "type": "core", "label": "Defensive Shields", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "shield.png", },
                        { "id": "AA", "type": "bonus", "label": "Ablative Armor", "bonus": 0.07, "adjacency": "greater", "sc_eligible": True, "image": "ablative.png", },
                        { "id": "Xa", "type": "bonus", "label": "Shield Upgrade Sigma", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Shield Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Shield Upgrade Theta", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.png", },
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
                        { "id": "HD", "type": "core", "label": "Hyperdrive", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "hyperdrive.png", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.02, "adjacency": "lesser", "sc_eligible": False, "image": "atlantid.png", },
                        { "id": "CD", "type": "bonus", "label": "Cadmium Drive", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": False, "image": "cadmium.png", },
                        { "id": "ED", "type": "bonus", "label": "Emeril Drive", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": False, "image": "emeril.png", },
                        { "id": "ID", "type": "bonus", "label": "Indium Drive", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": False, "image": "indium.png", },
                        { "id": "EW", "type": "bonus", "label": "Emergency Warp Unit", "bonus": 0.01, "adjacency": "greater", "sc_eligible": False, "image": "emergency.png", },
                        { "id": "Xa", "type": "bonus", "label": "Hyperdrive Upgrade Sigma", "bonus": 0.320, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Hyperdrive Upgrade Tau", "bonus": 0.310, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Hyperdrive Upgrade Theta", "bonus": 0.300, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.png", },
                    ],
                },
                {
                    "label": "Launch Thruster",
                    "key": "launch",
                    "image": "launch.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "LT", "type": "core", "label": "Launch Thruster", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "launch.png", },
                        { "id": "EF", "type": "bonus", "label": "Efficient Thrusters", "bonus": 0.20, "adjacency": "greater", "sc_eligible": False, "image": "efficient.png", },
                        { "id": "RC", "type": "bonus", "label": "Launch System Recharger", "bonus": 0.00, "adjacency": "greater", "sc_eligible": False, "image": "recharger.png", },
                        { "id": "Xa", "type": "bonus", "label": "Launch Thruster Upgrade Sigma", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Launch Thruster Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Launch Thruster Upgrade Theta", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.png", },
                    ],
                },
                {
                    "label": "Pulse Engine",
                    "key": "pulse",
                    "image": "pulse.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Pulse Engine", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": False, "image": "pulse.png", },
                        { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "flight-assist.png", },
                        { "id": "PC", "type": "reward", "label": "Photonix Core", "bonus": 0.27, "adjacency": "greater", "sc_eligible": True, "image": "photonix.png", },
                        { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.00, "adjacency": "greater", "sc_eligible": True, "image": "sublight.png", },
                        { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.01, "adjacency": "greater", "sc_eligible": True, "image": "instability.png", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 0.49, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 0.48, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.png", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Aqua-Jets",
                    "key": "aqua",
                    "image": "aquajet.webp",
                    "color": "black",
                    "modules": [
                        { "id": "AJ", "type": "core", "label": "Aqua-Jets", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "aquajets.png", },
                    ],
                },
                {
                    "label": "Bobbleheads",
                    "key": "bobble",
                    "image": "bobble.webp",
                    "color": "black",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "apollo.png", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "atlas.png", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "nada.png", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "null.png", },
                    ],
                },
                {
                    "label": "Scanners",
                    "key": "scanners",
                    "image": "scanner.webp",
                    "color": "black",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cargo Scan Deflector", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "cargo.png", },
                        { "id": "ES", "type": "core", "label": "Economy Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "economy.png", },
                        { "id": "CS", "type": "core", "label": "Conflict Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "conflict.png", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "trails.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AB", "type": "core", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "artemis.png", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "polo.png", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.01, "adjacency": "greater", "sc_eligible": True, "image": "squid.png", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": False, "image": "sputtering-trail.png", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": False, "image": "cadmium-trail.png", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "emeril-trail.png", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "temporal-trail.png", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "stealth-trail.png", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "golden-trail.png", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "chromatic-trail.png", },
                    ],
                },
                {
                    "label": "Teleporter",
                    "key": "teleporter",
                    "image": "teleport.webp",
                    "color": "black",
                    "modules": [
                        { "id": "TP", "type": "core", "label": "Teleport Receiver", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "teleport.png", },
                    ],
                },
            ],
        },
    },
    "sentinel": {
        "label": "Sentinel Interceptors",
        "types": {
            "Weaponry": [
                {
                    "label": "Cyclotron Ballista",
                    "key": "cyclotron",
                    "image": "cyclotron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "CB", "type": "core", "label": "Cyclotron Ballista", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron.png", },
                        { "id": "QR", "type": "bonus", "label": "Dyson Pump", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "dyson.png", },
                        { "id": "Xa", "type": "bonus", "label": "Cyclotron Ballista Upgrade Sigma", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Cyclotron Ballista Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Cyclotron Ballista Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "cyclotron-upgrade.png", },
                    ],
                },
                {
                    "label": "Infraknife Accelerator",
                    "key": "infra",
                    "image": "infra.webp",
                    "color": "red",
                    "modules": [
                        { "id": "IK", "type": "core", "label": "Infraknife Accelerator", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "infra.png", },
                        { "id": "QR", "type": "bonus", "label": "Q-Resonator", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "q-resonator.png", },
                        { "id": "Xa", "type": "bonus", "label": "Infraknife Accelerator Upgrade Sigma", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Infraknife Accelerator Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Infraknife Accelerator Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "infra-upgrade.png", },
                    ],
                },
                {
                    "label": "Phase Beam",
                    "key": "phase",
                    "image": "phase.webp",
                    "color": "green",
                    "modules": [
                        { "id": "PB", "type": "core", "label": "Phase Beam", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "phase-beam.png", },
                        { "id": "FD", "type": "bonus", "label": "Fourier De-Limiter", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "fourier.png", },
                        { "id": "Xa", "type": "bonus", "label": "Phase Beam Upgrade Sigma", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Phase Beam Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Phase Beam Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "phase-upgrade.png", },
                    ],
                },
                {
                    "label": "Positron Ejector",
                    "key": "positron",
                    "image": "positron.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Positron Ejector", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "positron.png", },
                        { "id": "FS", "type": "bonus", "label": "Fragment Supercharger", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "fragment.png", },
                        { "id": "Xa", "type": "bonus", "label": "Positron Ejector Upgrade Sigma", "bonus": 0.4, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Positron Ejector Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Positron Ejector Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "positron-upgrade.png", },
                    ],
                },
                {
                    "label": "Rocket Launcher",
                    "key": "rocket",
                    "image": "rocket.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "RL", "type": "core", "label": "Rocket Launger", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "rocket.png", },
                        { "id": "LR", "type": "bonus", "label": "Large Rocket Tubes", "bonus": 0.056, "adjacency": "greater", "sc_eligible": True, "image": "tubes.png", },
                    ],
                },
                {
                    "label": "Sentinel Cannon",
                    "key": "photon",
                    "image": "cannon.webp",
                    "color": "cyan",
                    "modules": [
                        { "id": "PC", "type": "core", "label": "Sentinel Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "cannon.png", },
                        { "id": "NO", "type": "bonus", "label": "Nonlinear Optics", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "nonlinear.png", },
                        { "id": "Xa", "type": "bonus", "label": "Photon Cannon Upgrade Sigma", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Photon Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Photon Cannon Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "photon-upgrade.png", },
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
                        { "id": "DS", "type": "core", "label": "Aeron Shields", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "aeron.png", },
                        { "id": "AA", "type": "bonus", "label": "Ablative Armor", "bonus": 0.07, "adjacency": "greater", "sc_eligible": False, "image": "ablative.png", },
                        { "id": "Xa", "type": "bonus", "label": "Shield Upgrade Sigma", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Shield Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Shield Upgrade Theta", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "shield-upgrade.png", },
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
                        { "id": "LT", "type": "core", "label": "Anti-Gravity Well", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "anti-gravity.png", },
                        { "id": "EF", "type": "bonus", "label": "Efficient Thrusters", "bonus": 0.20, "adjacency": "greater", "sc_eligible": False, "image": "efficient.png", },
                        { "id": "RC", "type": "bonus", "label": "Launch System Recharger", "bonus": 0.00, "adjacency": "greater", "sc_eligible": False, "image": "recharger.png", },
                        { "id": "Xa", "type": "bonus", "label": "Launch Thruster Upgrade Sigma", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Launch Thruster Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Launch Thruster Upgrade Theta", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "launch-upgrade.png", },
                    ],
                },
                {
                    "label": "Hyperdrive",
                    "key": "hyper",
                    "image": "hyper.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "HD", "type": "core", "label": "Hyperdrive", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "crimson.png", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.02, "adjacency": "lesser", "sc_eligible": False, "image": "atlantid.png", },
                        { "id": "CD", "type": "bonus", "label": "Cadmium Drive", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": False, "image": "cadmium.png", },
                        { "id": "ED", "type": "bonus", "label": "Emeril Drive", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": False, "image": "emeril.png", },
                        { "id": "ID", "type": "bonus", "label": "Indium Drive", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": False, "image": "indium.png", },
                        { "id": "EW", "type": "bonus", "label": "Emergency Warp Unit", "bonus": 0.00, "adjacency": "greater", "sc_eligible": False, "image": "emergency.png", },
                        { "id": "Xa", "type": "bonus", "label": "Hyperdrive Upgrade Sigma", "bonus": 0.320, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Hyperdrive Upgrade Tau", "bonus": 0.310, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Hyperdrive Upgrade Theta", "bonus": 0.300, "adjacency": "greater", "sc_eligible": True, "image": "hyper-upgrade.png", },
                    ],
                },
                {
                    "label": "Luminance Drive",
                    "key": "pulse",
                    "image": "luminance.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Luminance Drive", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": False, "image": "luminance.png", },
                        { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "flight-assist.png", },
                        { "id": "PC", "type": "reward", "label": "Photonix Core", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "photonix.png", },
                        { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.01, "adjacency": "greater", "sc_eligible": True, "image": "sublight.png", },
                        { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.17, "adjacency": "greater", "sc_eligible": True, "image": "instability.png", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 0.49, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 0.48, "adjacency": "greater", "sc_eligible": True, "image": "pulse-upgrade.png", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Aqua-Jets",
                    "key": "aqua",
                    "image": "aquajet.webp",
                    "color": "black",
                    "modules": [
                        { "id": "AJ", "type": "core", "label": "Aqua-Jets", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "aquajets.png", },
                    ],
                },
                {
                    "label": "Bobbleheads",
                    "key": "bobble",
                    "image": "bobble.webp",
                    "color": "black",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "apollo.png", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "atlas.png", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "nada.png", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "null.png", },
                    ],
                },
                {
                    "label": "Pilot Interface",
                    "key": "pilot",
                    "image": "pilot.webp",
                    "color": "black",
                    "modules": [
                        { "id": "PI", "type": "bonus", "label": "Pilot Interface", "bonus": 0.30, "adjacency": "none", "sc_eligible": True, "image": "pilot.png", },
                    ],
                },
                {
                    "label": "Scanners",
                    "key": "scanners",
                    "image": "scanner.webp",
                    "color": "black",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cargo Scan Deflector", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "cargo.png", },
                        { "id": "ES", "type": "core", "label": "Economy Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "economy.png", },
                        { "id": "CS", "type": "core", "label": "Conflict Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "conflict.png", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "trails.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AB", "type": "core", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "artemis.png", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "polo.png", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.01, "adjacency": "greater", "sc_eligible": True, "image": "squid.png", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": False, "image": "sputtering-trail.png", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": False, "image": "cadmium-trail.png", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "emeril-trail.png", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "temporal-trail.png", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "stealth-trail.png", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "golden-trail.png", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "chromatic-trail.png", },
                     ],
                },
                {
                    "label": "Teleporter",
                    "key": "teleporter",
                    "image": "teleport.webp",
                    "color": "black",
                    "modules": [
                        { "id": "TP", "type": "core", "label": "Teleport Receiver", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "teleport.png", },
                    ],
                },
            ],
        },
    },
    "living": {
        "label": "Living Starships",
        "types": {
            "Weaponry": [
                {
                    "label": "Grafted Eyes",
                    "key": "grafted",
                    "image": "grafted.webp",
                    "color": "green",
                    "modules": [
                        { "id": "GE", "type": "core", "label": "Grafted Eyes", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "grafted.png", },
                        { "id": "Xa", "type": "bonus", "label": "Grafted Eyes Upgrade Sigma", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "grafted-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Grafted Eyes Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "grafted-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Grafted Eyes Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "grafted-upgrade.png", },
                    ],
                },
                {
                    "label": "Spewing Vents",
                    "key": "spewing",
                    "image": "spewing.webp",
                    "color": "cyan",
                    "modules": [
                        { "id": "SV", "type": "core", "label": "Spewing Vents", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "spewing.png", },
                        { "id": "Xa", "type": "bonus", "label": "Spewing Vents Upgrade Sigma", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "spewing-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Spewing Vents Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "spewing-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Spewing Vents Upgrade Theta", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "spewing-upgrade.png", },
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
                        { "id": "SS", "type": "core", "label": "Scream Supressor", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "scream.png", },
                        { "id": "Xa", "type": "bonus", "label": "Scream Supressor Sigma", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "scream-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Scream Supressor Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "scream-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Scream Supressor Theta", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "scream-upgrade.png", },
                    ],
                },
            ],
            "Hyperdrive": [
                {
                    "label": "Neural Assembly",
                    "key": "assembly",
                    "image": "assembly.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "NA", "type": "core", "label": "Neural Assembly", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "assembly.png", },
                        { "id": "Xa", "type": "bonus", "label": "Neural Assembly Upgrade Sigma", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "assembly-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Neural Assembly Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "assembly-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Neural Assembly Upgrade Theta", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "assembly-upgrade.png", },
                        { "id": "CM", "type": "bonus", "label": "Chroloplast Membrane", "bonus": 0.00, "adjacency": "greater", "sc_eligible": False, "image": "chloroplast.png", },
                    ],
                },
                {
                    "label": "Singularity Cortex",
                    "key": "singularity",
                    "image": "singularity.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Singularity Cortex", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "singularity.png", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.02, "adjacency": "greater", "sc_eligible": False, "image": "atlantid.png", },
                        { "id": "Xa", "type": "bonus", "label": "Singularity Cortex Upgrade Sigma", "bonus": 0.320, "adjacency": "greater", "sc_eligible": True, "image": "singularity-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Singularity Cortex Upgrade Tau", "bonus": 0.310, "adjacency": "greater", "sc_eligible": True, "image": "singularity-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Singularity Cortex Upgrade Theta", "bonus": 0.300, "adjacency": "greater", "sc_eligible": True, "image": "singularity-upgrade.png", },
                    ],
                },
                {
                    "label": "Pulsing Heart",
                    "key": "pulsing",
                    "image": "pulsing.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PH", "type": "core", "label": "Pulsing Heart", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "pulsing.png", },
                        { "id": "Xa", "type": "bonus", "label": "Pulsing Heart Upgrade Sigma", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "pulsing-upgrade.png", },
                        { "id": "Xb", "type": "bonus", "label": "Pulsing Heart Upgrade Tau", "bonus": 0.49, "adjacency": "greater", "sc_eligible": True, "image": "pulsing-upgrade.png", },
                        { "id": "Xc", "type": "bonus", "label": "Pulsing Heart Upgrade Theta", "bonus": 0.48, "adjacency": "greater", "sc_eligible": True, "image": "pulsing-upgrade.png", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Saline Carapace",
                    "key": "saline",
                    "image": "saline.webp",
                    "color": "black",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Saline Catapace", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "saline.png", },
                    ],
                },
                {
                    "label": "Bobbleheads",
                    "key": "bobble",
                    "image": "bobble.webp",
                    "color": "black",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "apollo.png", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "atlas.png", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "nada.png", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "null.png", },
                    ],
                },
                {
                    "label": "Scanners",
                    "key": "scanners",
                    "image": "wormhole.webp",
                    "color": "black",
                    "modules": [
                        { "id": "WB", "type": "core", "label": "Wormhole Brain", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "wormhole.png", },
                        { "id": "NS", "type": "core", "label": "Neural Shielding", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "neural.png", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "trails.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AB", "type": "core", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "artemis.png", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "polo.png", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.01, "adjacency": "greater", "sc_eligible": True, "image": "squid.png", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": False, "image": "sputtering-trail.png", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": False, "image": "cadmium-trail.png", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "emeril-trail.png", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "temporal-trail.png", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "stealth-trail.png", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "golden-trail.png", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.001, "adjacency": "greater", "sc_eligible": True, "image": "chromatic-trail.png", },
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
                (0, 0): "None",
                (1, 0): "QR",
                (0, 1): "Xc",
                (1, 1): "CB",
                (0, 2): "Xa",
                (1, 2): "Xb"
            },
            "score": 16.533104257280005
        },
        "infra": {
            "map": {
                (0, 0): "None",
                (1, 0): "QR",
                (0, 1): "Xc",
                (1, 1): "IK",
                (0, 2): "Xa",
                (1, 2): "Xb"
            },
            "score": 16.533104257280005
        },
        "phase": {
            "map": {
                (0, 0): "Xc",
                (1, 0): "Xb",
                (0, 1): "FD",
                (1, 1): "Xa",
                (0, 2): "None",
                (1, 2): "PB"
            },
            "score": 15.770084195840003
        },
        "photon": {
            "map": {
                (0, 0): "None",
                (1, 0): "NO",
                (0, 1): "Xb",
                (1, 1): "PC",
                (0, 2): "Xa",
                (1, 2): "Xc"
            },
            "score": 16.53310425728
        },
        "positron": {
            "map": {
                (0, 0): "Xc",
                (1, 0): "Xa",
                (0, 1): "PE",
                (1, 1): "Xb",
                (0, 2): "FS",
                (1, 2): "None"
            },
            "score": 16.533104257279998
        },
        "rocket": {
            "map": {
                (0, 0): "RL",
                (0, 1): "LR",
                (0, 2): "None"
            },
            "score": 3.42008832
        },
        "shield": {
            "map": {
                (0, 0): "None",
                (1, 0): "DS",
                (0, 1): "Xc",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "AA"
            },
            "score": 11.625227202560005
        },
        "hyper": {
            "map": {
                (0, 0): "HD",
                (1, 0): "Xa",
                (2, 0): "EW",
                (0, 1): "ED",
                (1, 1): "Xb",
                (2, 1): "Xc",
                (0, 2): "CD",
                (1, 2): "AD",
                (2, 2): "ID"
            },
            "score": 12.331969050187706
        },
        "launch": {
            "map": {
                (0, 0): "Xc",
                (1, 0): "EF",
                (0, 1): "Xa",
                (1, 1): "Xb",
                (0, 2): "RC",
                (1, 2): "LT"
            },
            "score": 13.829370854400004
        },
        "pulse": {
            "map": {
                (0, 0): "PE",
                (1, 0): "None",
                (2, 0): "SL",
                (0, 1): "Xc",
                (1, 1): "Xa",
                (2, 1): "FA",
                (0, 2): "ID",
                (1, 2): "Xb",
                (2, 2): "PC"
            },
            "score": 33.99634364958593
        },
        "trails": {
            "map": {
                (0, 0): "None",
                (1, 0): "RT",
                (2, 0): "CT",
                (3, 0): "TT",
                (0, 1): "SB",
                (1, 1): "AB",
                (2, 1): "PB",
                (3, 1): "ET",
                (0, 2): "None",
                (1, 2): "GT",
                (2, 2): "ST",
                (3, 2): "SP"
            },
            "score": 0.162
        }
    },
    "sentinel": {
        "cyclotron": {
            "map": {
                (0, 0): "None",
                (1, 0): "QR",
                (0, 1): "Xc",
                (1, 1): "CB",
                (0, 2): "Xa",
                (1, 2): "Xb"
            },
            "score": 16.533104257280005
        },
        "infra": {
            "map": {
                (0, 0): "None",
                (1, 0): "QR",
                (0, 1): "Xb",
                (1, 1): "IK",
                (0, 2): "Xa",
                (1, 2): "Xc"
            },
            "score": 16.53310425728
        },
        "phase": {
            "map": {
                (0, 0): "Xb",
                (1, 0): "Xc",
                (0, 1): "Xa",
                (1, 1): "FD",
                (0, 2): "PB",
                (1, 2): "None"
            },
            "score": 15.770084195840003
        },
        "positron": {
            "map": {
                (0, 0): "None",
                (1, 0): "FS",
                (0, 1): "Xc",
                (1, 1): "PE",
                (0, 2): "Xa",
                (1, 2): "Xb"
            },
            "score": 16.533104257280005
        },
        "rocket": {
            "map": {
                (0, 0): "RL",
                (0, 1): "LR",
                (0, 2): "None"
            },
            "score": 3.42008832
        },
        "photon": {
            "map": {
                (0, 0): "None",
                (1, 0): "NO",
                (0, 1): "Xc",
                (1, 1): "PC",
                (0, 2): "Xa",
                (1, 2): "Xb"
            },
            "score": 16.533104257280005
        },
        "shield": {
            "map": {
                (0, 0): "None",
                (1, 0): "DS",
                (0, 1): "Xc",
                (1, 1): "Xa",
                (0, 2): "Xb",
                (1, 2): "AA"
            },
            "score": 11.625227202560005
        },
        "launch": {
            "map": {
                (0, 0): "EF",
                (1, 0): "Xc",
                (0, 1): "Xb",
                (1, 1): "Xa",
                (0, 2): "LT",
                (1, 2): "RC"
            },
            "score": 13.829370854400004
        },
        "hyper": {
            "map": {
                (0, 0): "Xb",
                (1, 0): "Xc",
                (2, 0): "ID",
                (0, 1): "Xa",
                (1, 1): "EW",
                (2, 1): "AD",
                (0, 2): "HD",
                (1, 2): "ED",
                (2, 2): "CD"
            },
            "score": 12.164765707900973
        },
        "pulse": {
            "map": {
                (0, 0): "SL",
                (1, 0): "None",
                (2, 0): "PE",
                (0, 1): "PC",
                (1, 1): "Xa",
                (2, 1): "Xc",
                (0, 2): "ID",
                (1, 2): "Xb",
                (2, 2): "FA"
            },
            "score": 42.964968895926276
        },
        "trails": {
            "map": {
                (0, 0): "None",
                (1, 0): "RT",
                (2, 0): "CT",
                (3, 0): "TT",
                (0, 1): "SB",
                (1, 1): "AB",
                (2, 1): "PB",
                (3, 1): "ET",
                (0, 2): "None",
                (1, 2): "GT",
                (2, 2): "ST",
                (3, 2): "SP"
            },
            "score": 0.162
        }
    },
    "living": {
        "grafted": {
            "map": {
                (0, 0): "GE",
                (1, 0): "Xc",
                (0, 1): "Xb",
                (1, 1): "Xa",
                (0, 2): "None",
                (1, 2): "None"
            },
            "score": 14.747714128
        },
        "spewing": {
            "map": {
                (0, 0): "SV",
                (1, 0): "Xc",
                (0, 1): "Xb",
                (1, 1): "Xa",
                (0, 2): "None",
                (1, 2): "None"
            },
            "score": 15.392229328
        },
        "scream": {
            "map": {
                (0, 0): "Xc",
                (1, 0): "SS",
                (0, 1): "Xa",
                (1, 1): "Xb",
                (0, 2): "None",
                (1, 2): "None"
            },
            "score": 10.446795136000002
        },
        "assembly": {
            "map": {
                (0, 0): "NA",
                (1, 0): "None",
                (0, 1): "Xa",
                (1, 1): "CM",
                (0, 2): "Xc",
                (1, 2): "Xb"
            },
            "score": 3.074946432000001
        },
        "singularity": {
            "map": {
                (0, 0): "Xb",
                (1, 0): "AD",
                (0, 1): "Xa",
                (1, 1): "Xc",
                (0, 2): "SC",
                (1, 2): "None"
            },
            "score": 11.677480414080001
        },
        "pulsing": {
            "map": {
                (0, 0): "Xa",
                (1, 0): "Xb",
                (0, 1): "Xc",
                (1, 1): "PH",
                (0, 2): "None",
                (1, 2): "None"
            },
            "score": 20.238178480000002
        },
        "trails": {
            "map": {
                (0, 0): "None",
                (1, 0): "RT",
                (2, 0): "CT",
                (3, 0): "TT",
                (0, 1): "SB",
                (1, 1): "AB",
                (2, 1): "PB",
                (3, 1): "ET",
                (0, 2): "None",
                (1, 2): "GT",
                (2, 2): "ST",
                (3, 2): "SP"
            },
            "score": 0.162
        }
    }
}
# fmt:on