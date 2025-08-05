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
                    "image": "starship/cyclotron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "CB", "type": "core", "label": "Cyclotron Ballista", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron.webp", },
                        { "id": "QR", "type": "bonus", "label": "Dyson Pump", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "starship/dyson.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Cyclotron Ballista Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Cyclotron Ballista Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Cyclotron Ballista Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Infraknife Accelerator",
                    "key": "infra",
                    "image": "starship/infra.webp",
                    "color": "red",
                    "modules": [
                        { "id": "IK", "type": "core", "label": "Infraknife Accelerator", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra.webp", },
                        { "id": "QR", "type": "bonus", "label": "Q-Resonator", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "starship/q-resonator.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Infraknife Accelerator Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Infraknife Accelerator Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Infraknife Accelerator Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra-upgrade.webp", },
                    ],
                },
                {
                    "label": "Phase Beam",
                    "key": "phase",
                    "image": "starship/phase.webp",
                    "color": "green",
                    "modules": [
                        { "id": "PB", "type": "core", "label": "Phase Beam", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "starship/phase-beam.webp", },
                        { "id": "FD", "type": "bonus", "label": "Fourier De-Limiter", "bonus": 0.07, "adjacency": "lesser", "sc_eligible": True, "image": "starship/fourier.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Phase Beam Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/phase-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Phase Beam Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/phase-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Phase Beam Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/phase-upgrade.webp", },
                    ],
                },
                {
                    "label": "Photon Cannon",
                    "key": "photon",
                    "image": "starship/photon.webp",
                    "color": "blue",
                    "modules": [
                        { "id": "PC", "type": "core", "label": "Photon Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon.webp", },
                        { "id": "NO", "type": "bonus", "label": "Nonlinear Optics", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "starship/nonlinear.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Photon Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Photon Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Photon Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon-upgrade.webp", },
                    ],
                },
                {
                    "label": "Positron Ejector",
                    "key": "positron",
                    "image": "starship/positron.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Positron Ejector", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron.webp", },
                        { "id": "FS", "type": "bonus", "label": "Fragment Supercharger", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "starship/fragment.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Positron Ejector Upgrade Theta", "bonus": 0.4, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Positron Ejector Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Positron Ejector Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Rocket Launcher",
                    "key": "rocket",
                    "image": "starship/rocket.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "RL", "type": "core", "label": "Rocket Launger", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/rocket.webp", },
                        { "id": "LR", "type": "bonus", "label": "Large Rocket Tubes", "bonus": 0.056, "adjacency": "greater", "sc_eligible": True, "image": "starship/tubes.webp", },
                    ],
                },
            ],
            "Defensive Systems": [
                {
                    "label": "Starship Shields",
                    "key": "shield",
                    "image": "starship/shield.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "DS", "type": "core", "label": "Defensive Shields", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/shield.webp", },
                        { "id": "AA", "type": "bonus", "label": "Ablative Armor", "bonus": 0.07, "adjacency": "greater", "sc_eligible": True, "image": "starship/ablative.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Shield Upgrade Theta", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "starship/shield-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Shield Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "starship/shield-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Shield Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "starship/shield-upgrade.webp", },
                    ],
                },
            ],
            "Hyperdrive": [
                {
                    "label": "Hyperdrive",
                    "key": "hyper",
                    "image": "starship/hyper.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "HD", "type": "core", "label": "Hyperdrive", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/hyperdrive.webp", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/atlantid.webp", },
                        { "id": "CD", "type": "bonus", "label": "Cadmium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/cadmium.webp", },
                        { "id": "ED", "type": "bonus", "label": "Emeril Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/emeril.webp", },
                        { "id": "ID", "type": "bonus", "label": "Indium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/indium.webp", },
                        { "id": "EW", "type": "bonus", "label": "Emergency Warp Unit", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/emergency.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Hyperdrive Upgrade Theta", "bonus": .320, "adjacency": "greater", "sc_eligible": True, "image": "starship/hyper-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Hyperdrive Upgrade Tau", "bonus": .310, "adjacency": "greater", "sc_eligible": True, "image": "starship/hyper-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Hyperdrive Upgrade Sigma", "bonus": .300, "adjacency": "greater", "sc_eligible": True, "image": "starship/hyper-upgrade.webp", },
                    ],
                },
                {
                    "label": "Launch Thruster",
                    "key": "launch",
                    "image": "starship/launch.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "LT", "type": "core", "label": "Launch Thruster", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": False, "image": "starship/launch.webp", },
                        { "id": "EF", "type": "bonus", "label": "Efficient Thrusters", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/efficient.webp", },
                        { "id": "RC", "type": "bonus", "label": "Launch Auto-Charger", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/recharger.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Launch Thruster Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "starship/launch-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Launch Thruster Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "starship/launch-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Launch Thruster Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "starship/launch-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Engine",
                    "key": "pulse",
                    "image": "starship/pulse.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Pulse Engine", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "starship/pulse.webp", },
                        { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1.07, "adjacency": "greater", "sc_eligible": False, "image": "starship/flight-assist.webp", },
                        { "id": "PC", "type": "reward", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "starship/photonix.webp", },
                        { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/sublight.webp", },
                        { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/instability.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                    ],
                },
                # {
                #     "label": "Pulse Engine",
                #     "key": "photonix",
                #     "image": "starship/pulse.webp",
                #     "color": "orange",
                #     "modules": [
                #         { "id": "PE", "type": "core", "label": "Pulse Engine", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "starship/pulse.webp", },
                #         { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1.07, "adjacency": "greater", "sc_eligible": False, "image": "starship/flight-assist.webp", },
                #         { "id": "PC", "type": "bonus", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "starship/photonix.webp", },
                #         { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/sublight.webp", },
                #         { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/instability.webp", },
                #         { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                #         { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                #         { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                #     ],
                # },
            ],
            "Utilities": [
                {
                    "label": "Aqua-Jets",
                    "key": "aqua",
                    "image": "starship/aquajet.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AJ", "type": "core", "label": "Aqua-Jets", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/aquajets.webp", },
                    ],
                },
                {
                    "label": "Figurines",
                    "key": "bobble",
                    "image": "starship/bobble.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/apollo.webp", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/atlas.webp", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/nada.webp", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/null.webp", },
                    ],
                },
                {
                    "label": "Conflict Scanner",
                    "key": "conflict_scanner",
                    "image": "starship/conflict.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Conflict Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/conflict.webp", }
                    ],
                },
                {
                    "label": "Economy Scanner",
                    "key": "economy_scanner",
                    "image": "starship/economy.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "ES", "type": "core", "label": "Economy Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/economy.webp", }
                    ],
                },
                {
                    "label": "Cargo Scan Deflector",
                    "key": "cargo_scanner",
                    "image": "starship/cargo.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cargo Scan Deflector", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/cargo.webp", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "starship/trails.webp",
                    "color": "white", 
                    "modules": [
                        { "id": "AB", "type": "bonus", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "starship/artemis.webp", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.06, "adjacency": "greater", "sc_eligible": True, "image": "starship/polo.webp", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "starship/squid.webp", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/sputtering-trail.webp", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/cadmium-trail.webp", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/emeril-trail.webp", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/temporal-trail.webp", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/stealth-trail.webp", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/golden-trail.webp", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/chromatic-trail.webp", },
                    ],
                },
                {
                    "label": "Teleport Receiver",
                    "key": "teleporter",
                    "image": "starship/teleport.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TP", "type": "core", "label": "Teleport Receiver", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/teleport.webp", },
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
                    "image": "starship/cyclotron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "CB", "type": "core", "label": "Cyclotron Ballista", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron.webp", },
                        { "id": "QR", "type": "bonus", "label": "Dyson Pump", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "starship/dyson.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Cyclotron Ballista Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Cyclotron Ballista Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Cyclotron Ballista Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Infraknife Accelerator",
                    "key": "infra",
                    "image": "starship/infra.webp",
                    "color": "red",
                    "modules": [
                        { "id": "IK", "type": "core", "label": "Infraknife Accelerator", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra.webp", },
                        { "id": "QR", "type": "bonus", "label": "Q-Resonator", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "starship/q-resonator.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Infraknife Accelerator Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Infraknife Accelerator Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Infraknife Accelerator Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra-upgrade.webp", },
                    ],
                },
                {
                    "label": "Phase Beam",
                    "key": "phase",
                    "image": "starship/phase.webp",
                    "color": "green",
                    "modules": [
                        { "id": "PB", "type": "core", "label": "Phase Beam", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "starship/phase-beam.webp", },
                        { "id": "FD", "type": "bonus", "label": "Fourier De-Limiter", "bonus": 0.07, "adjacency": "lesser", "sc_eligible": True, "image": "starship/fourier.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Phase Beam Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/phase-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Phase Beam Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/phase-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Phase Beam Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/phase-upgrade.webp", },
                    ],
                },
                {
                    "label": "Positron Ejector",
                    "key": "positron",
                    "image": "starship/positron.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Positron Ejector", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron.webp", },
                        { "id": "FS", "type": "bonus", "label": "Fragment Supercharger", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "starship/fragment.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Positron Ejector Upgrade Theta", "bonus": 0.4, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Positron Ejector Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Positron Ejector Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Rocket Launcher",
                    "key": "rocket",
                    "image": "starship/rocket.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "RL", "type": "core", "label": "Rocket Launger", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/rocket.webp", },
                        { "id": "LR", "type": "bonus", "label": "Large Rocket Tubes", "bonus": 0.056, "adjacency": "greater", "sc_eligible": True, "image": "starship/tubes.webp", },
                    ],
                },
                {
                    "label": "Sentinel Cannon",
                    "key": "photon",
                    "image": "starship/cannon.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "PC", "type": "core", "label": "Sentinel Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/cannon.webp", },
                        { "id": "NO", "type": "bonus", "label": "Nonlinear Optics", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "starship/nonlinear.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Photon Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Photon Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Photon Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon-upgrade.webp", },
                    ],
                },
            ],
            "Defensive Systems": [
                {
                    "label": "Aeron Shields",
                    "key": "shield",
                    "image": "starship/aeron.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "DS", "type": "core", "label": "Aeron Shields", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/aeron.webp", },
                        { "id": "AA", "type": "bonus", "label": "Ablative Armor", "bonus": 0.07, "adjacency": "greater", "sc_eligible": True, "image": "starship/ablative.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Shield Upgrade Theta", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "starship/shield-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Shield Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "starship/shield-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Shield Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "starship/shield-upgrade.webp", },
                    ],
                },
            ],
            "Hyperdrive": [
                {
                    "label": "Anti-Gravity Well",
                    "key": "launch",
                    "image": "starship/anti-gravity.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "LT", "type": "core", "label": "Anti-Gravity Well", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/anti-gravity.webp", },
                        { "id": "EF", "type": "bonus", "label": "Efficient Thrusters", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/efficient.webp", },
                        { "id": "RC", "type": "bonus", "label": "Launch Atuo-Charger", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/recharger.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Launch Thruster Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "starship/launch-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Launch Thruster Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "starship/launch-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Launch Thruster Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "starship/launch-upgrade.webp", },
                    ],
                },
                {
                    "label": "Crimson Core",
                    "key": "hyper",
                    "image": "starship/crimson.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "HD", "type": "core", "label": "Crimson Core", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/crimson.webp", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/atlantid.webp", },
                        { "id": "CD", "type": "bonus", "label": "Cadmium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/cadmium.webp", },
                        { "id": "ED", "type": "bonus", "label": "Emeril Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/emeril.webp", },
                        { "id": "ID", "type": "bonus", "label": "Indium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/indium.webp", },
                        { "id": "EW", "type": "bonus", "label": "Emergency Warp Unit", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/emergency.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Crimson Core Upgrade Theta", "bonus": 0.320, "adjacency": "greater", "sc_eligible": True, "image": "starship/hyper-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Crimson Core Upgrade Tau", "bonus": 0.310, "adjacency": "greater", "sc_eligible": True, "image": "starship/hyper-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Crimson Core Upgrade Sigma", "bonus": 0.300, "adjacency": "greater", "sc_eligible": True, "image": "starship/hyper-upgrade.webp", },
                  ],
                },
                {
                    "label": "Luminance Drive",
                    "key": "pulse",
                    "image": "starship/luminance.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Luminance Drive", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "starship/luminance.webp", },
                        { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1.07, "adjacency": "greater", "sc_eligible": False, "image": "starship/flight-assist.webp", },
                        { "id": "PC", "type": "reward", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "starship/photonix.webp", },
                        { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/sublight.webp", },
                        { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/instability.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                    ],
                },
                # {
                #     "label": "Luminance Drive",
                #     "key": "photonix",
                #     "image": "starship/luminance.webp",
                #     "color": "orange",
                #     "modules": [
                #         { "id": "PE", "type": "core", "label": "Luminance Drive", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "starship/luminance.webp", },
                #         { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1.07, "adjacency": "greater", "sc_eligible": False, "image": "starship/flight-assist.webp", },
                #         { "id": "PC", "type": "bonus", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "starship/photonix.webp", },
                #         { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/sublight.webp", },
                #         { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/instability.webp", },
                #         { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                #         { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                #         { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                #     ],
                # },
            ],
            "Utilities": [
                {
                    "label": "Aqua-Jets",
                    "key": "aqua",
                    "image": "starship/aquajet.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AJ", "type": "core", "label": "Aqua-Jets", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/aquajets.webp", },
                    ],
                },
                {
                    "label": "Figurines",
                    "key": "bobble",
                    "image": "starship/bobble.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/apollo.webp", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/atlas.webp", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/nada.webp", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/null.webp", },
                    ],
                },
                {
                    "label": "Pilot Interface",
                    "key": "pilot",
                    "image": "starship/pilot.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PI", "type": "bonus", "label": "Pilot Interface", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/pilot.webp", },
                    ],
                },
                {
                    "label": "Conflict Scanner",
                    "key": "conflict_scanner",
                    "image": "starship/conflict.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Conflict Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/conflict.webp", }
                    ],
                },
                {
                    "label": "Economy Scanner",
                    "key": "economy_scanner",
                    "image": "starship/economy.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "ES", "type": "core", "label": "Economy Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/economy.webp", }
                    ],
                },
                {
                    "label": "Cargo Scan Deflector",
                    "key": "cargo_scanner",
                    "image": "starship/cargo.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cargo Scan Deflector", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/cargo.webp", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "starship/trails.webp",
                    "color": "white",
                    "modules": [
                        { "id": "AB", "type": "bonus", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "starship/artemis.webp", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.06, "adjacency": "greater", "sc_eligible": True, "image": "starship/polo.webp", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "starship/squid.webp", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/sputtering-trail.webp", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/cadmium-trail.webp", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/emeril-trail.webp", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/temporal-trail.webp", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/stealth-trail.webp", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/golden-trail.webp", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/chromatic-trail.webp", },
                     ],
                },
                {
                    "label": "Teleport Receiver",
                    "key": "teleporter",
                    "image": "starship/teleport.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TP", "type": "core", "label": "Teleport Receiver", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/teleport.webp", },
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
                    "image": "starship/cyclotron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "CB", "type": "core", "label": "Cyclotron Ballista", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron.webp", },
                        { "id": "QR", "type": "bonus", "label": "Dyson Pump", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "starship/dyson.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Cyclotron Ballista Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Cyclotron Ballista Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Cyclotron Ballista Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/cyclotron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Infraknife Accelerator",
                    "key": "infra",
                    "image": "starship/infra.webp",
                    "color": "red",
                    "modules": [
                        { "id": "IK", "type": "core", "label": "Infraknife Accelerator", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra.webp", },
                        { "id": "QR", "type": "bonus", "label": "Q-Resonator", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "starship/q-resonator.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Infraknife Accelerator Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Infraknife Accelerator Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Infraknife Accelerator Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/infra-upgrade.webp", },
                    ],
                },
                {
                    "label": "Phase Beam",
                    "key": "phase",
                    "image": "starship/phase.webp",
                    "color": "green",
                    "modules": [
                        { "id": "PB", "type": "core", "label": "Phase Beam", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "starship/phase-beam.webp", },
                        { "id": "FD", "type": "bonus", "label": "Fourier De-Limiter", "bonus": 0.07, "adjacency": "lesser", "sc_eligible": True, "image": "starship/fourier.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Phase Beam Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/phase-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Phase Beam Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/phase-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Phase Beam Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/phase-upgrade.webp", },
                    ],
                },
                {
                    "label": "Photon Cannon",
                    "key": "photon",
                    "image": "starship/photon.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "PC", "type": "core", "label": "Photon Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon.webp", },
                        { "id": "NO", "type": "bonus", "label": "Nonlinear Optics", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "starship/nonlinear.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Photon Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Photon Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Photon Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/photon-upgrade.webp", },
                    ],
                },
                {
                    "label": "Positron Ejector",
                    "key": "positron",
                    "image": "starship/positron.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Positron Ejector", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron.webp", },
                        { "id": "FS", "type": "bonus", "label": "Fragment Supercharger", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "starship/fragment.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Positron Ejector Upgrade Theta", "bonus": 0.4, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Positron Ejector Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Positron Ejector Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/positron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Rocket Launcher",
                    "key": "rocket",
                    "image": "starship/rocket.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "RL", "type": "core", "label": "Rocket Launger", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/rocket.webp", },
                        { "id": "LR", "type": "bonus", "label": "Large Rocket Tubes", "bonus": 0.056, "adjacency": "greater", "sc_eligible": True, "image": "starship/tubes.webp", },
                    ],
                },
            ],
            "Defensive Systems": [
                {
                    "label": "Starship Shields",
                    "key": "shield",
                    "image": "starship/shield.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "DS", "type": "core", "label": "Defensive Shields", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/shield.webp", },
                        { "id": "AA", "type": "bonus", "label": "Ablative Armor", "bonus": 0.07, "adjacency": "greater", "sc_eligible": True, "image": "starship/ablative.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Shield Upgrade Theta", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "starship/shield-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Shield Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "starship/shield-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Shield Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "starship/shield-upgrade.webp", },
                    ],
                },
            ],
            "Hyperdrive": [
                {
                    "label": "Hyperdrive",
                    "key": "hyper",
                    "image": "starship/hyper.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "HD", "type": "core", "label": "Hyperdrive", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/hyperdrive.webp", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/atlantid.webp", },
                        { "id": "CD", "type": "bonus", "label": "Cadmium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/cadmium.webp", },
                        { "id": "ED", "type": "bonus", "label": "Emeril Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/emeril.webp", },
                        { "id": "ID", "type": "bonus", "label": "Indium Drive", "bonus": 0.0, "adjacency": "lesser", "sc_eligible": False, "image": "starship/indium.webp", },
                        { "id": "EW", "type": "bonus", "label": "Emergency Warp Unit", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/emergency.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Hyperdrive Upgrade Theta", "bonus": .320, "adjacency": "greater", "sc_eligible": True, "image": "starship/hyper-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Hyperdrive Upgrade Tau", "bonus": .310, "adjacency": "greater", "sc_eligible": True, "image": "starship/hyper-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Hyperdrive Upgrade Sigma", "bonus": .300, "adjacency": "greater", "sc_eligible": True, "image": "starship/hyper-upgrade.webp", },
                    ],
                },
                {
                    "label": "Launch Thruster",
                    "key": "launch",
                    "image": "starship/launch.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "LT", "type": "core", "label": "Launch Thruster", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": False, "image": "starship/launch.webp", },
                        { "id": "EF", "type": "bonus", "label": "Efficient Thrusters", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/efficient.webp", },
                        { "id": "RC", "type": "bonus", "label": "Launch Auto-Charger", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/recharger.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Launch Thruster Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "starship/launch-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Launch Thruster Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "starship/launch-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Launch Thruster Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "starship/launch-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Engine",
                    "key": "pulse",
                    "image": "starship/pulse.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PE", "type": "core", "label": "Pulse Engine", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "starship/pulse.webp", },
                        { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1.07, "adjacency": "greater", "sc_eligible": False, "image": "starship/flight-assist.webp", },
                        { "id": "VS", "type": "bonus", "label": "Vesper Sail", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "starship/vesper.webp", },
                        { "id": "PC", "type": "reward", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "starship/photonix.webp", },
                        { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/sublight.webp", },
                        { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/instability.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                    ],
                },
                # {
                #     "label": "Pulse Engine",
                #     "key": "photonix",
                #     "image": "starship/pulse.webp",
                #     "color": "orange",
                #     "modules": [
                #         { "id": "PE", "type": "core", "label": "Pulse Engine", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "starship/pulse.webp", },
                #         { "id": "FA", "type": "bonus", "label": "Flight Assist Override", "bonus": 1., "adjacency": "greater", "sc_eligible": False, "image": "starship/flight-assist.webp", },
                #         { "id": "VS", "type": "bonus", "label": "Vesper Sail", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "starship/vesper.webp", },
                #         { "id": "PC", "type": "bonus", "label": "Photonix Core", "bonus": 1.10, "adjacency": "greater", "sc_eligible": False, "image": "starship/photonix.webp", },
                #         { "id": "SL", "type": "bonus", "label": "Sub-Light Amplifier", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/sublight.webp", },
                #         { "id": "ID", "type": "bonus", "label": "Instability Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/instability.webp", },
                #         { "id": "Xa", "type": "bonus", "label": "Pulse Engine Upgrade Theta", "bonus": 1.14, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                #         { "id": "Xb", "type": "bonus", "label": "Pulse Engine Upgrade Tau", "bonus": 1.13, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                #         { "id": "Xc", "type": "bonus", "label": "Pulse Engine Upgrade Sigma", "bonus": 1.12, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulse-upgrade.webp", },
                #     ],
                # },
            ],
            "Utilities": [
                {
                    "label": "Aqua-Jets",
                    "key": "aqua",
                    "image": "starship/aquajet.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AJ", "type": "core", "label": "Aqua-Jets", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/aquajets.webp", },
                    ],
                },
                {
                    "label": "Figurines",
                    "key": "bobble",
                    "image": "starship/bobble.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/apollo.webp", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/atlas.webp", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/nada.webp", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/null.webp", },
                    ],
                },
                 {
                    "label": "Conflict Scanner",
                    "key": "conflict_scanner",
                    "image": "starship/conflict.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Conflict Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/conflict.webp", }
                    ],
                },
                {
                    "label": "Economy Scanner",
                    "key": "economy_scanner",
                    "image": "starship/economy.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "ES", "type": "core", "label": "Economy Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/economy.webp", }
                    ],
                },
                {
                    "label": "Cargo Scan Deflector",
                    "key": "cargo_scanner",
                    "image": "starship/cargo.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cargo Scan Deflector", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/cargo.webp", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "starship/trails.webp",
                    "color": "white", 
                    "modules": [
                        { "id": "AB", "type": "bonus", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "starship/artemis.webp", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.06, "adjacency": "greater", "sc_eligible": True, "image": "starship/polo.webp", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "starship/squid.webp", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/sputtering-trail.webp", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/cadmium-trail.webp", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/emeril-trail.webp", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/temporal-trail.webp", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/stealth-trail.webp", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/golden-trail.webp", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/chromatic-trail.webp", },
                    ],
                },
                {
                    "label": "Teleport Receiver",
                    "key": "teleporter",
                    "image": "starship/teleport.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TP", "type": "core", "label": "Teleport Receiver", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/teleport.webp", },
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
                    "image": "starship/grafted.webp",
                    "color": "green",
                    "modules": [
                        { "id": "GE", "type": "core", "label": "Grafted Eyes", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "starship/grafted.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Grafted Eyes Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/grafted-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Grafted Eyes Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/grafted-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Grafted Eyes Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/grafted-upgrade.webp", },
                    ],
                },
                {
                    "label": "Spewing Vents",
                    "key": "spewing",
                    "image": "starship/spewing.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "SV", "type": "core", "label": "Spewing Vents", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/spewing.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Spewing Vents Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "starship/spewing-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Spewing Vents Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "starship/spewing-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Spewing Vents Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "starship/spewing-upgrade.webp", },
                    ],
                },
            ],
            "Defensive Systems": [
                {
                    "label": "Scream Supressor",
                    "key": "scream",
                    "image": "starship/scream.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "SS", "type": "core", "label": "Scream Supressor", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/scream.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scream Supressor Upgrade Theta", "bonus": 0.3, "adjacency": "greater", "sc_eligible": True, "image": "starship/scream-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scream Supressor Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "starship/scream-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scream Supressor Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "starship/scream-upgrade.webp", },
                    ],
                },
            ],
            "Hyperdrive": [
                {
                    "label": "Neural Assembly",
                    "key": "assembly",
                    "image": "starship/assembly.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "NA", "type": "core", "label": "Neural Assembly", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/assembly.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Neural Assembly Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "starship/assembly-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Neural Assembly Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "starship/assembly-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Neural Assembly Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "starship/assembly-upgrade.webp", },
                        { "id": "CM", "type": "bonus", "label": "Chroloplast Membrane", "bonus": 0.0001, "adjacency": "greater", "sc_eligible": True, "image": "starship/chloroplast.webp", },
                    ],
                },
                {
                    "label": "Singularity Cortex",
                    "key": "singularity",
                    "image": "starship/singularity.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Singularity Cortex", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/singularity.webp", },
                        { "id": "AD", "type": "bonus", "label": "Atlantid Drive", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "starship/atlantid.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Singularity Cortex Upgrade Theta", "bonus": 0.320, "adjacency": "greater", "sc_eligible": True, "image": "starship/singularity-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Singularity Cortex Upgrade Tau", "bonus": 0.310, "adjacency": "greater", "sc_eligible": True, "image": "starship/singularity-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Singularity Cortex Upgrade Sigma", "bonus": 0.300, "adjacency": "greater", "sc_eligible": True, "image": "starship/singularity-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulsing Heart",
                    "key": "pulsing",
                    "image": "starship/pulsing.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "PH", "type": "core", "label": "Pulsing Heart", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "starship/pulsing.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulsing Heart Upgrade Theta", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulsing-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulsing Heart Upgrade Tau", "bonus": 0.49, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulsing-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulsing Heart Upgrade Sigma", "bonus": 0.48, "adjacency": "greater", "sc_eligible": True, "image": "starship/pulsing-upgrade.webp", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Saline Carapace",
                    "key": "saline",
                    "image": "starship/saline.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Saline Catapace", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/saline.webp", },
                    ],
                },
                {
                    "label": "Figurines",
                    "key": "bobble",
                    "image": "starship/bobble.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AP", "type": "core", "label": "Apollo Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/apollo.webp", },
                        { "id": "AT", "type": "core", "label": "Atlas Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/atlas.webp", },
                        { "id": "NA", "type": "bonus", "label": "Nada Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/nada.webp", },
                        { "id": "NB", "type": "bonus", "label": "-null- Figurine", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/null.webp", },
                    ],
                },
                {
                    "label": "Scanners",
                    "key": "scanners",
                    "image": "starship/wormhole.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "WB", "type": "core", "label": "Wormhole Brain", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/wormhole.webp", },
                        { "id": "NS", "type": "core", "label": "Neural Shielding", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "starship/neural.webp", },
                    ],
                },
                {
                    "label": "Starship Trails",
                    "key": "trails",
                    "image": "starship/trails.webp",
                    "color": "white", 
                    "modules": [
                        { "id": "AB", "type": "bonus", "label": "Artemis Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "starship/artemis.webp", },
                        { "id": "PB", "type": "core", "label": "Polo Figurine", "bonus": 0.06, "adjacency": "greater", "sc_eligible": True, "image": "starship/polo.webp", },
                        { "id": "SB", "type": "reward", "label": "Tentacled Figurine", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "starship/squid.webp", },
                        { "id": "SP", "type": "reward", "label": "Sputtering Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/sputtering-trail.webp", },
                        { "id": "CT", "type": "bonus", "label": "Cadmium Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/cadmium-trail.webp", },
                        { "id": "ET", "type": "bonus", "label": "Emeril Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/emeril-trail.webp", },
                        { "id": "TT", "type": "reward", "label": "Temporal Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/temporal-trail.webp", },
                        { "id": "ST", "type": "bonus", "label": "Stealth Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/stealth-trail.webp", },
                        { "id": "GT", "type": "bonus", "label": "Golden Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/golden-trail.webp", },
                        { "id": "RT", "type": "bonus", "label": "Chromatic Starship Trail", "bonus": 0.0, "adjacency": "greater", "sc_eligible": False, "image": "starship/chromatic-trail.webp", },
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
                    "image": "multi-tool/mining-beam.webp",
                    "color": "green",
                    "modules": [
                        { "id": "MB", "type": "core", "label": "Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-laser.webp", },
                        { "id": "AM", "type": "bonus", "label": "Advanced Mining Laser", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/advanced-mining.webp", },
                        { "id": "OD", "type": "bonus", "label": "Optical Drill", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/optical.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mining Laser Upgrade Theta", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mining Laser Upgrade Tau", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mining Laser Upgrade Sigma", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                    ],
                },
            ],
            "Scanners": [
                {
                    "label": "Analysis Visor",
                    "key": "analysis",
                    "image": "multi-tool/analysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AV", "type": "core", "label": "Analysis Visor", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/analysis.webp", },
                    ],
                },
                {
                    "label": "Fishing Rig",
                    "key": "fishing",
                    "image": "multi-tool/fishing.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "FR", "type": "core", "label": "Fishing Rig", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/fishing.webp", },
                    ],
                },
                {
                    "label": "Scanner",
                    "key": "scanner",
                    "image": "multi-tool/mt-scanner.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Scanner", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/mt-scanner.webp", },
                        { "id": "WR", "type": "bonus", "label": "Waveform Recycler", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/waveform.webp", },
                        { "id": "SH", "type": "bonus", "label": "Scan Harmonizer", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/harmonizer.webp", },
                        { "id": "PC", "type": "bonus", "label": "Polyphonic Core", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/polyphonic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scanner Upgrade Theta", "bonus": 1.50, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scanner Upgrade Tau", "bonus": 1.45, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scanner Upgrade Sigma", "bonus": 1.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                    ],
                },
                {
                    "label": "Survey Device",
                    "key": "survey",
                    "image": "multi-tool/survey.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "SD", "type": "core", "label": "Survey Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/survey.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Blaze Javelin",
                    "key": "blaze-javelin",
                    "image": "multi-tool/blaze-javelin.webp",
                    "color": "red",
                    "modules": [
                        { "id": "BJ", "type": "core", "label": "Blaze Javelin", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-javelin.webp", },
                        { "id": "MA", "type": "bonus", "label": "Mass Accelerator", "bonus": .03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mass-accelerator.webp", },
                        { "id": "WO", "type": "bonus", "label": "Waveform Oscillator", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "multi-tool/waveform-osc.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Blaze Javelin Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Blaze Javelin Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Blaze Javelin Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                    ],
                },
                {
                    "label": "Boltcaster",
                    "key": "bolt-caster",
                    "image": "multi-tool/boltcaster.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "BC", "type": "core", "label": "Bolt Caster", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/boltcaster.webp", },
                        { "id": "RM", "type": "bonus", "label": "Boltcaster Ricochet Module", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-rm.webp", },
                        { "id": "BI", "type": "bonus", "label": "Barrel Ionizer", "bonus": 0.1, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/barrel-ionizer.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Boltcaster Upgrade Theta", "bonus": 0.15, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Boltcaster Upgrade Tau", "bonus": 0.14, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Boltcaster Upgrade Sigma", "bonus": 0.13, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Fa", "type": "bonus", "label": "Forbidden Module Theta", "bonus": 0.22, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                        { "id": "Fb", "type": "bonus", "label": "Forbidden Module Tau", "bonus": 0.21, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                        { "id": "Fc", "type": "bonus", "label": "Forbidden Module Sigma", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                    ],
                },
                {
                    "label": "Geology Cannon",
                    "key": "geology",
                    "image": "multi-tool/geology.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "GC", "type": "core", "label": "Geology Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Geology Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Geology Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Geology Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                    ],
                },
                {
                    "label": "Neutron Cannon",
                    "key": "neutron",
                    "image": "multi-tool/neutron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "NC", "type": "core", "label": "Neutron Cannon", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/neutron.webp", },
                        { "id": "PF", "type": "bonus", "label": "P-Field Compressor", "bonus": 1.62, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/p-field.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Neutron Cannon Upgrade Theta", "bonus": 3.15, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Neutron Cannon Upgrade Tau", "bonus": 3.14, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Neutron Cannon Upgrade Sigma", "bonus": 3.13, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Plasma Launcher",
                    "key": "plasma-launcher",
                    "image": "multi-tool/plasma-launcher.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "PL", "type": "core", "label": "Plasma Launcher", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-launcher.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Plasma Launcher Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Plasma Launcher Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Plasma Launcher Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Spitter",
                    "key": "pulse-splitter",
                    "image": "multi-tool/pulse-splitter.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "PS", "type": "core", "label": "Pulse Spitter", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter.webp", },
                        { "id": "AC", "type": "bonus", "label": "Amplified Cartridges", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/amplified.webp", },
                        { "id": "RM", "type": "bonus", "label": "Richochet Module", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-rm.webp", },
                        { "id": "II", "type": "bonus", "label": "Impact Ignitor", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "multi-tool/impact-ignitor.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Spitter Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Spitter Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Spitter Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                    ],
                },
                {
                    "label": "Scatter Blaster",
                    "key": "scatter",
                    "image": "multi-tool/scatter.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "SB", "type": "core", "label": "Scatter Blaster", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter.webp", },
                        { "id": "SG", "type": "bonus", "label": "Shell Greaser", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/shell-greaser.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scatter Blaster Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scatter Blaster Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scatter Blaster Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                    ],
                },
            ],
            "Secondary Weapons": [
                {
                    "label": "Cloaking Device",
                    "key": "cloaking",
                    "image": "multi-tool/cloaking.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cloaking Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/cloaking.webp", },
                    ],
                },
                {
                    "label": "Combat Scope",
                    "key": "combat",
                    "image": "multi-tool/combat.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Combat Scope", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/combat.webp", },
                    ],
                },
                {
                    "label": "Voltaic Amplifier",
                    "key": "voltaic-amplifier",
                    "image": "multi-tool/voltaic-amplifier.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "VA", "type": "core", "label": "Voltaic Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/voltaic-amplifier.webp", },
                    ],
                },
                {
                    "label": "Paralysis Mortar",
                    "key": "paralysis",
                    "image": "multi-tool/paralysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PM", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/paralysis.webp", },
                    ],
                },
                {
                    "label": "Personal Forcefield",
                    "key": "personal",
                    "image": "multi-tool/personal.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PF", "type": "core", "label": "Personal Forcefield", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/personal.webp", },
                    ],
                },
                {
                    "label": "Terrain Manipulator",
                    "key": "terrian",
                    "image": "multi-tool/terrian.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TM", "type": "core", "label": "Terrain Manipulator", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/terrian.webp", },
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
                    "label": "Runic Lens",
                    "key": "mining",
                    "image": "multi-tool/runic.webp",
                    "color": "green",
                    "modules": [
                        { "id": "MB", "type": "core", "label": "Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-laser.webp", },
                        { "id": "AM", "type": "bonus", "label": "Advanced Mining Laser", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/advanced-mining.webp", },
                        { "id": "OD", "type": "bonus", "label": "Optical Drill", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/optical.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mining Laser Upgrade Theta", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mining Laser Upgrade Tau", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mining Laser Upgrade Sigma", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                        { "id": "RL", "type": "bonus", "label": "Runic Lens", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/runic-lens.webp", },
                    ],
                },
            ],
            "Scanners": [
                {
                    "label": "Analysis Visor",
                    "key": "analysis",
                    "image": "multi-tool/analysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AV", "type": "core", "label": "Analysis Visor", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/analysis.webp", },
                    ],
                },
                {
                    "label": "Fishing Rig",
                    "key": "fishing",
                    "image": "multi-tool/fishing.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "FR", "type": "core", "label": "Fishing Rig", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/fishing.webp", },
                    ],
                },
                {
                    "label": "Scanners",
                    "key": "scanner",
                    "image": "multi-tool/mt-scanner.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Scanner", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/mt-scanner.webp", },
                        { "id": "WR", "type": "bonus", "label": "Waveform Recycler", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/waveform.webp", },
                        { "id": "SH", "type": "bonus", "label": "Scan Harmonizer", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/harmonizer.webp", },
                        { "id": "PC", "type": "bonus", "label": "Polyphonic Core", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/polyphonic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scanner Upgrade Theta", "bonus": 1.50, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scanner Upgrade Tau", "bonus": 1.45, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scanner Upgrade Sigma", "bonus": 1.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                    ],
                },
                {
                    "label": "Survey Device",
                    "key": "survey",
                    "image": "multi-tool/survey.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "SD", "type": "core", "label": "Survey Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/survey.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Blaze Javelin",
                    "key": "blaze-javelin",
                    "image": "multi-tool/blaze-javelin.webp",
                    "color": "red",
                    "modules": [
                        { "id": "BJ", "type": "core", "label": "Blaze Javelin", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-javelin.webp", },
                        { "id": "MA", "type": "bonus", "label": "Mass Accelerator", "bonus": .03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mass-accelerator.webp", },
                        { "id": "WO", "type": "bonus", "label": "Waveform Oscillator", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "multi-tool/waveform-osc.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Blaze Javelin Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Blaze Javelin Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Blaze Javelin Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                    ],
                },
                {
                    "label": "Boltcaster",
                    "key": "bolt-caster",
                    "image": "multi-tool/boltcaster.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "BC", "type": "core", "label": "Bolt Caster", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/boltcaster.webp", },
                        { "id": "RM", "type": "bonus", "label": "Boltcaster Ricochet Module", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-rm.webp", },
                        { "id": "BI", "type": "bonus", "label": "Barrel Ionizer", "bonus": 0.1, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/barrel-ionizer.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Boltcaster Upgrade Theta", "bonus": 0.15, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Boltcaster Upgrade Tau", "bonus": 0.14, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Boltcaster Upgrade Sigma", "bonus": 0.13, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Fa", "type": "bonus", "label": "Forbidden Module Theta", "bonus": 0.22, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                        { "id": "Fb", "type": "bonus", "label": "Forbidden Module Tau", "bonus": 0.21, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                        { "id": "Fc", "type": "bonus", "label": "Forbidden Module Sigma", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                    ],
                },
                {
                    "label": "Geology Cannon",
                    "key": "geology",
                    "image": "multi-tool/geology.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "GC", "type": "core", "label": "Geology Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Geology Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Geology Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Geology Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                    ],
                },
                {
                    "label": "Neutron Cannon",
                    "key": "neutron",
                    "image": "multi-tool/neutron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "NC", "type": "core", "label": "Neutron Cannon", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/neutron.webp", },
                        { "id": "PF", "type": "bonus", "label": "P-Field Compressor", "bonus": 1.62, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/p-field.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Neutron Cannon Upgrade Theta", "bonus": 3.15, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Neutron Cannon Upgrade Tau", "bonus": 3.14, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Neutron Cannon Upgrade Sigma", "bonus": 3.13, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Plasma Launcher",
                    "key": "plasma-launcher",
                    "image": "multi-tool/plasma-launcher.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "PL", "type": "core", "label": "Plasma Launcher", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-launcher.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Plasma Launcher Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Plasma Launcher Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Plasma Launcher Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Spitter",
                    "key": "pulse-splitter",
                    "image": "multi-tool/pulse-splitter.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "PS", "type": "core", "label": "Pulse Spitter", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter.webp", },
                        { "id": "AC", "type": "bonus", "label": "Amplified Cartridges", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/amplified.webp", },
                        { "id": "RM", "type": "bonus", "label": "Richochet Module", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-rm.webp", },
                        { "id": "II", "type": "bonus", "label": "Impact Ignitor", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "multi-tool/impact-ignitor.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Spitter Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Spitter Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Spitter Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                    ],
                },
                {
                    "label": "Scatter Blaster",
                    "key": "scatter",
                    "image": "multi-tool/scatter.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "SB", "type": "core", "label": "Scatter Blaster", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter.webp", },
                        { "id": "SG", "type": "bonus", "label": "Shell Greaser", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/shell-greaser.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scatter Blaster Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scatter Blaster Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scatter Blaster Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                    ],
                },
            ],
            "Secondary Weapons": [
                {
                    "label": "Cloaking Device",
                    "key": "cloaking",
                    "image": "multi-tool/cloaking.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cloaking Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/cloaking.webp", },
                    ],
                },
                {
                    "label": "Combat Scope",
                    "key": "combat",
                    "image": "multi-tool/combat.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Combat Scope", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/combat.webp", },
                    ],
                },
                {
                    "label": "Voltaic Amplifier",
                    "key": "voltaic-amplifier",
                    "image": "multi-tool/voltaic-amplifier.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "VA", "type": "core", "label": "Voltaic Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/voltaic-amplifier.webp", },
                    ],
                },
                {
                    "label": "Paralysis Mortar",
                    "key": "paralysis",
                    "image": "multi-tool/paralysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PM", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/paralysis.webp", },
                    ],
                },
                {
                    "label": "Personal Forcefield",
                    "key": "personal",
                    "image": "multi-tool/personal.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PF", "type": "core", "label": "Personal Forcefield", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/personal.webp", },
                    ],
                },
                {
                    "label": "Terrain Manipulator",
                    "key": "terrian",
                    "image": "multi-tool/terrian.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TM", "type": "core", "label": "Terrain Manipulator", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/terrian.webp", },
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
                    "image": "multi-tool/hijacked.webp",
                    "color": "green",
                    "modules": [
                        { "id": "RL", "type": "core", "label": "Hijacked Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/hijacked.webp", },
                        { "id": "AM", "type": "bonus", "label": "Advanced Mining Laser", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/advanced-mining.webp", },
                        { "id": "OD", "type": "bonus", "label": "Optical Drill", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/optical.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mining Laser Upgrade Theta", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mining Laser Upgrade Tau", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mining Laser Upgrade Sigma", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                        { "id": "MB", "type": "core", "label": "Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-laser.webp", },
                    ],
                },
            ],     
            "Scanners": [
                {
                    "label": "Analysis Visor",
                    "key": "analysis",
                    "image": "multi-tool/analysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AV", "type": "core", "label": "Analysis Visor", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/analysis.webp", },
                    ],
                },
                {
                    "label": "Fishing Rig",
                    "key": "fishing",
                    "image": "multi-tool/fishing.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "FR", "type": "core", "label": "Fishing Rig", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/fishing.webp", },
                    ],
                },
                {
                    "label": "Scanners",
                    "key": "scanner",
                    "image": "multi-tool/mt-scanner.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Scanner", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/mt-scanner.webp", },
                        { "id": "WR", "type": "bonus", "label": "Waveform Recycler", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/waveform.webp", },
                        { "id": "SH", "type": "bonus", "label": "Scan Harmonizer", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/harmonizer.webp", },
                        { "id": "PC", "type": "bonus", "label": "Polyphonic Core", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/polyphonic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scanner Upgrade Theta", "bonus": 1.50, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scanner Upgrade Tau", "bonus": 1.45, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scanner Upgrade Sigma", "bonus": 1.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                    ],
                },
                {
                    "label": "Survey Device",
                    "key": "survey",
                    "image": "multi-tool/survey.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "SD", "type": "core", "label": "Survey Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/survey.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Blaze Javelin",
                    "key": "blaze-javelin",
                    "image": "multi-tool/blaze-javelin.webp",
                    "color": "red",
                    "modules": [
                        { "id": "BJ", "type": "core", "label": "Blaze Javelin", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-javelin.webp", },
                        { "id": "MA", "type": "bonus", "label": "Mass Accelerator", "bonus": .03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mass-accelerator.webp", },
                        { "id": "WO", "type": "bonus", "label": "Waveform Oscillator", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "multi-tool/waveform-osc.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Blaze Javelin Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Blaze Javelin Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Blaze Javelin Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                    ],
                },
                {
                    "label": "Boltcaster",
                    "key": "bolt-caster",
                    "image": "multi-tool/boltcaster.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "BC", "type": "core", "label": "Bolt Caster", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/boltcaster.webp", },
                        { "id": "RM", "type": "bonus", "label": "Boltcaster Ricochet Module", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-rm.webp", },
                        { "id": "BI", "type": "bonus", "label": "Barrel Ionizer", "bonus": 0.1, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/barrel-ionizer.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Boltcaster Upgrade Theta", "bonus": 0.15, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Boltcaster Upgrade Tau", "bonus": 0.14, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Boltcaster Upgrade Sigma", "bonus": 0.13, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Fa", "type": "bonus", "label": "Forbidden Module Theta", "bonus": 0.22, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                        { "id": "Fb", "type": "bonus", "label": "Forbidden Module Tau", "bonus": 0.21, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                        { "id": "Fc", "type": "bonus", "label": "Forbidden Module Sigma", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                    ],
                },
                {
                    "label": "Geology Cannon",
                    "key": "geology",
                    "image": "multi-tool/geology.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "GC", "type": "core", "label": "Geology Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Geology Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Geology Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Geology Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                    ],
                },
                {
                    "label": "Neutron Cannon",
                    "key": "neutron",
                    "image": "multi-tool/neutron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "NC", "type": "core", "label": "Neutron Cannon", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/neutron.webp", },
                        { "id": "PF", "type": "bonus", "label": "P-Field Compressor", "bonus": 1.62, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/p-field.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Neutron Cannon Upgrade Theta", "bonus": 3.15, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Neutron Cannon Upgrade Tau", "bonus": 3.14, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Neutron Cannon Upgrade Sigma", "bonus": 3.13, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Plasma Launcher",
                    "key": "plasma-launcher",
                    "image": "multi-tool/plasma-launcher.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "PL", "type": "core", "label": "Plasma Launcher", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-launcher.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Plasma Launcher Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Plasma Launcher Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Plasma Launcher Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Spitter",
                    "key": "pulse-splitter",
                    "image": "multi-tool/pulse-splitter.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "PS", "type": "core", "label": "Pulse Spitter", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter.webp", },
                        { "id": "AC", "type": "bonus", "label": "Amplified Cartridges", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/amplified.webp", },
                        { "id": "RM", "type": "bonus", "label": "Richochet Module", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-rm.webp", },
                        { "id": "II", "type": "bonus", "label": "Impact Ignitor", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "multi-tool/impact-ignitor.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Spitter Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Spitter Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Spitter Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                    ],
                },
                {
                    "label": "Scatter Blaster",
                    "key": "scatter",
                    "image": "multi-tool/scatter.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "SB", "type": "core", "label": "Scatter Blaster", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter.webp", },
                        { "id": "SG", "type": "bonus", "label": "Shell Greaser", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/shell-greaser.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scatter Blaster Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scatter Blaster Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scatter Blaster Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                    ],
                },
            ],
            "Secondary Weapons": [
                {
                    "label": "Cloaking Device",
                    "key": "cloaking",
                    "image": "multi-tool/cloaking.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cloaking Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/cloaking.webp", },
                    ],
                },
                {
                    "label": "Combat Scope",
                    "key": "combat",
                    "image": "multi-tool/combat.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Combat Scope", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/combat.webp", },
                    ],
                },
                {
                    "label": "Voltaic Amplifier",
                    "key": "voltaic-amplifier",
                    "image": "multi-tool/voltaic-amplifier.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "VA", "type": "core", "label": "Voltaic Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/voltaic-amplifier.webp", },
                    ],
                },
                {
                    "label": "Paralysis Mortar",
                    "key": "paralysis",
                    "image": "multi-tool/paralysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PM", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/paralysis.webp", },
                    ],
                },
                {
                    "label": "Personal Forcefield",
                    "key": "personal",
                    "image": "multi-tool/personal.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PF", "type": "core", "label": "Personal Forcefield", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/personal.webp", },
                    ],
                },
                {
                    "label": "Terrain Manipulator",
                    "key": "terrian",
                    "image": "multi-tool/terrian.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TM", "type": "core", "label": "Terrain Manipulator", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/terrian.webp", },
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
                    "image": "multi-tool/mining-beam.webp",
                    "color": "green",
                    "modules": [
                        { "id": "MB", "type": "core", "label": "Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-laser.webp", },
                        { "id": "AM", "type": "bonus", "label": "Advanced Mining Laser", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/advanced-mining.webp", },
                        { "id": "OD", "type": "bonus", "label": "Optical Drill", "bonus": 0.50, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/optical.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mining Laser Upgrade Theta", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mining Laser Upgrade Tau", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mining Laser Upgrade Sigma", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mining-upgrade.webp", },
                    ],
                },
            ],
            "Scanners": [
                {
                    "label": "Analysis Visor",
                    "key": "analysis",
                    "image": "multi-tool/analysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "AV", "type": "core", "label": "Analysis Visor", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/analysis.webp", },
                    ],
                },
                {
                    "label": "Fishing Rig",
                    "key": "fishing",
                    "image": "multi-tool/fishing.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "FR", "type": "core", "label": "Fishing Rig", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/fishing.webp", },
                    ],
                },
                {
                    "label": "Scanner",
                    "key": "scanner",
                    "image": "multi-tool/mt-scanner.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "SC", "type": "core", "label": "Scanner", "bonus": 0.01, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/mt-scanner.webp", },
                        { "id": "WR", "type": "bonus", "label": "Waveform Recycler", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/waveform.webp", },
                        { "id": "SH", "type": "bonus", "label": "Scan Harmonizer", "bonus": 0.11, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/harmonizer.webp", },
                        { "id": "PC", "type": "bonus", "label": "Polyphonic Core", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/polyphonic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scanner Upgrade Theta", "bonus": 1.50, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scanner Upgrade Tau", "bonus": 1.45, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scanner Upgrade Sigma", "bonus": 1.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scanner-upgrade.webp", },
                    ],
                },
                {
                    "label": "Survey Device",
                    "key": "survey",
                    "image": "multi-tool/survey.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "SD", "type": "core", "label": "Survey Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/survey.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Blaze Javelin",
                    "key": "blaze-javelin",
                    "image": "multi-tool/blaze-javelin.webp",
                    "color": "red",
                    "modules": [
                        { "id": "BJ", "type": "core", "label": "Blaze Javelin", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-javelin.webp", },
                        { "id": "MA", "type": "bonus", "label": "Mass Accelerator", "bonus": .03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/mass-accelerator.webp", },
                        { "id": "WO", "type": "bonus", "label": "Waveform Oscillator", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "multi-tool/waveform-osc.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Blaze Javelin Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Blaze Javelin Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Blaze Javelin Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/blaze-upgrade.webp", },
                    ],
                },
                {
                    "label": "Boltcaster",
                    "key": "bolt-caster",
                    "image": "multi-tool/boltcaster.webp",
                    "color": "teal",
                    "modules": [
                        { "id": "BC", "type": "core", "label": "Bolt Caster", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/boltcaster.webp", },
                        { "id": "RM", "type": "bonus", "label": "Boltcaster Ricochet Module", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-rm.webp", },
                        { "id": "BI", "type": "bonus", "label": "Barrel Ionizer", "bonus": 0.1, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/barrel-ionizer.webp", },               
                        { "id": "Xa", "type": "bonus", "label": "Boltcaster Upgrade Theta", "bonus": 0.15, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Boltcaster Upgrade Tau", "bonus": 0.14, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Boltcaster Upgrade Sigma", "bonus": 0.13, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/boltcaster-upgrade.webp", },
                        { "id": "Fa", "type": "bonus", "label": "Forbidden Module Theta", "bonus": 0.22, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                        { "id": "Fb", "type": "bonus", "label": "Forbidden Module Tau", "bonus": 0.21, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                        { "id": "Fc", "type": "bonus", "label": "Forbidden Module Sigma", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/forbidden.webp", },
                    ],
                },
                {
                    "label": "Geology Cannon",
                    "key": "geology",
                    "image": "multi-tool/geology.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "GC", "type": "core", "label": "Geology Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Geology Cannon Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Geology Cannon Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Geology Cannon Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/geology-upgrade.webp", },
                    ],
                },
                 {
                    "label": "Neutron Cannon",
                    "key": "neutron",
                    "image": "multi-tool/neutron.webp",
                    "color": "purple",
                    "modules": [
                        { "id": "NC", "type": "core", "label": "Neutron Cannon", "bonus": 1.00, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/neutron.webp", },
                        { "id": "PF", "type": "bonus", "label": "P-Field Compressor", "bonus": 1.62, "adjacency": "lesser", "sc_eligible": True, "image": "multi-tool/p-field.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Neutron Cannon Upgrade Theta", "bonus": 3.15, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Neutron Cannon Upgrade Tau", "bonus": 3.14, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Neutron Cannon Upgrade Sigma", "bonus": 3.13, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/neutron-upgrade.webp", },
                    ],
                },
                {
                    "label": "Plasma Launcher",
                    "key": "plasma-launcher",
                    "image": "multi-tool/plasma-launcher.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "PL", "type": "core", "label": "Plasma Launcher", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-launcher.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Plasma Launcher Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Plasma Launcher Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Plasma Launcher Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/plasma-upgrade.webp", },
                    ],
                },
                {
                    "label": "Pulse Spitter",
                    "key": "pulse-splitter",
                    "image": "multi-tool/pulse-splitter.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "PS", "type": "core", "label": "Pulse Spitter", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter.webp", },
                        { "id": "AC", "type": "bonus", "label": "Amplified Cartridges", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/amplified.webp", },
                        { "id": "RM", "type": "bonus", "label": "Richochet Module", "bonus": 0.03, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-rm.webp", },
                        { "id": "II", "type": "bonus", "label": "Impact Ignitor", "bonus": 0.0, "adjacency": "none", "sc_eligible": True, "image": "multi-tool/impact-ignitor.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Pulse Spitter Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Pulse Spitter Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Pulse Spitter Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/pulse-splitter-upgrade.webp", },
                    ],
                },
                {
                    "label": "Scatter Blaster",
                    "key": "scatter",
                    "image": "multi-tool/scatter.webp",
                    "color": "amber",
                    "modules": [
                        { "id": "SB", "type": "core", "label": "Scatter Blaster", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter.webp", },
                        { "id": "SG", "type": "bonus", "label": "Shell Greaser", "bonus": 0.0, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/shell-greaser.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Scatter Blaster Upgrade Theta", "bonus": 0.40, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Scatter Blaster Upgrade Tau", "bonus": 0.39, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Scatter Blaster Upgrade Sigma", "bonus": 0.38, "adjacency": "greater", "sc_eligible": True, "image": "multi-tool/scatter-upgrade.webp", },
                    ],
                },
            ],
            "Secondary Weapons": [
                {
                    "label": "Cloaking Device",
                    "key": "cloaking",
                    "image": "multi-tool/cloaking.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CD", "type": "core", "label": "Cloaking Device", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/cloaking.webp", },
                    ],
                },
                {
                    "label": "Combat Scope",
                    "key": "combat",
                    "image": "multi-tool/combat.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "CS", "type": "core", "label": "Combat Scope", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/combat.webp", },
                    ],
                },
                {
                    "label": "Voltaic Amplifier",
                    "key": "voltaic-amplifier",
                    "image": "multi-tool/voltaic-amplifier.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "VA", "type": "core", "label": "Voltaic Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/voltaic-amplifier.webp", },
                    ],
                },
                {
                    "label": "Paralysis Mortar",
                    "key": "paralysis",
                    "image": "multi-tool/paralysis.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PM", "type": "core", "label": "Paralysis Mortar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/paralysis.webp", },
                    ],
                },
                {
                    "label": "Personal Forcefield",
                    "key": "personal",
                    "image": "multi-tool/personal.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "PF", "type": "core", "label": "Personal Forcefield", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/personal.webp", },
                    ],
                },
                {
                    "label": "Terrain Manipulator",
                    "key": "terrian",
                    "image": "multi-tool/terrian.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "TM", "type": "core", "label": "Terrain Manipulator", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "multi-tool/terrian.webp", },
                    ],
                },
            ],
        },
    },
    "freighter": {
        "label": "Freighters",
        "type": "Other",
        "types": {
            "Hyperdrive": [
                {
                    "label": "Freighter Hyperdrive",
                    "key": "hyper",
                    "image": "other/freighter-hyper.webp",
                    "color": "iris",
                    "modules": [
                        { "id": "HD", "type": "core", "label": "Hyperdrive", "bonus": 0.10, "adjacency": "lesser", "sc_eligible": True, "image": "other/freighter-hyperdrive.webp", },
                        { "id": "TW", "type": "bonus", "label": "Temporal Warp Computer", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "other/freighter-temporal.webp", },
                        { "id": "RD", "type": "bonus", "label": "Reality De-Threader", "bonus": 0.80, "adjacency": "greater", "sc_eligible": True, "image": "other/freighter-reality.webp", },
                        { "id": "RM", "type": "bonus", "label": "Resonance Matrix", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "other/freighter-resonance.webp", },
                        { "id": "PW", "type": "bonus", "label": "Plasmatic Warp Injector", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "other/freighter-plasmatic.webp", },
                        { "id": "AW", "type": "bonus", "label": "Amplified Warp Shielding", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "other/freighter-amplified.webp", },
                        { "id": "WC", "type": "bonus", "label": "Warp Core Resonator", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "other/freighter-warpcore.webp", },
                        { "id": "CW", "type": "bonus", "label": "Chromatic Warp Shielding", "bonus": 0.05, "adjacency": "greater", "sc_eligible": True, "image": "other/freighter-chromatic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Hyperdrive Upgrade Theta", "bonus": .25, "adjacency": "greater", "sc_eligible": True, "image": "other/freighter-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Hyperdrive Upgrade Tau", "bonus": .24, "adjacency": "greater", "sc_eligible": True, "image": "other/freighter-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Hyperdrive Upgrade Sigma", "bonus": .23, "adjacency": "greater", "sc_eligible": True, "image": "other/freighter-upgrade.webp", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Interstellar Scanner",
                    "key": "interstellar",
                    "image": "other/freighter-scanner.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "IS", "type": "core", "label": "Interstellar Scanner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "other/freighter-scanner.webp", },
                    ],
                },
                {
                    "label": "Matter Beam",
                    "key": "matterbeam",
                    "image": "other/freighter-matter.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "MB", "type": "core", "label": "Matter Beam", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "other/freighter-matter.webp", },
                    ],
                },
            ],
            "Fleet Upgrades": [
                {
                    "label": "Fuel Efficiency",
                    "key": "fleet-fuel",
                    "image": "other/fleet-fuel.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Fuel Efficiency Upgrade Theta", "bonus": .300, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-fuel.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Fuel Efficiency Upgrade Tau", "bonus": .290, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-fuel.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Fuel Efficiency Upgrade Sigma", "bonus": .280, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-fuel.webp", },
                      ],
                },
                {
                    "label": "Expedition Speed",
                    "key": "fleet-speed",
                    "image": "other/fleet-speed.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Expedition Speed Upgrade Theta", "bonus": .300, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-speed.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Expedition Speed Upgrade Tau", "bonus": .290, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-speed.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Expedition Speed Upgrade Sigma", "bonus": .280, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-speed.webp", },
                      ],
                },
                {
                    "label": "Combat and Defense",
                    "key": "fleet-combat",
                    "image": "other/fleet-combat.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Combat and Defense Upgrade Theta", "bonus": .300, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-combat.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Combat and Defense Upgrade Tau", "bonus": .290, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-combat.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Combat and Defense Upgrade Sigma", "bonus": .280, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-combat.webp", },
                      ],
                },
                {
                    "label": "Exploration and Science",
                    "key": "fleet-exploration",
                    "image": "other/fleet-exploration.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Exploration and Science Upgrade Theta", "bonus": .300, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-exploration.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Exploration and Science Upgrade Tau", "bonus": .290, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-exploration.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Exploration and Science Upgrade Sigma", "bonus": .280, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-exploration.webp", },
                      ],
                },
                {
                    "label": "Mining and Industrial",
                    "key": "fleet-mining",
                    "image": "other/fleet-mining.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Theta", "bonus": .300, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-mining.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Tau", "bonus": .290, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-mining.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Sigma", "bonus": .280, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-mining.webp", },
                      ],
                },
                {
                    "label": "Trade",
                    "key": "fleet-trade",
                    "image": "other/fleet-trade.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Theta", "bonus": .300, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-trade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Tau", "bonus": .290, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-trade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fleet Mining and Industrial Upgrade Sigma", "bonus": .280, "adjacency": "greater_1", "sc_eligible": True, "image": "other/fleet-trade.webp", },
                      ],
                },
            ],
        },
    },
    "nomad": {
        "label": "Nomad",
        "type": "Exocraft",
        "types": {
            "Propulsion": [
                {
                    "label": "Fusion Engine",
                    "key": "fusion",
                    "image": "exocraft/fusion.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "FE", "type": "core", "label": "Fusion Engine", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exocraft/fusion.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Fusion Engine Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fusion Engine Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fusion Engine Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                    ],
                },
                {
                    "label": "Icarus Fuel System",
                    "key": "icarus",
                    "image": "exocraft/icarus.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "FS", "type": "core", "label": "Icarus Fuel System", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/icarus.webp", },
                    ],
                },
                {
                    "label": "Exocraft Boosters",
                    "key": "boost",
                    "image": "exocraft/boost.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "EB", "type": "core", "label": "Exocraft Boost Module", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exocraft/boost.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Exocraft Boost Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Exocraft Boost Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Exocraft Boost Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                    ],
                },
                {
                    "label": "Hi-Slide Suspension",
                    "key": "slide",
                    "image": "exocraft/slide.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "HS", "type": "core", "label": "Hi-Slide Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/slide.webp", },
                    ],
                },
                {
                    "label": "Grip Boost Suspension",
                    "key": "grip",
                    "image": "exocraft/grip.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "GB", "type": "core", "label": "Grip Boost Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/grip.webp", },
                    ],
                },
                {
                    "label": "Drift Suspension",
                    "key": "drift",
                    "image": "exocraft/drift.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "DS", "type": "core", "label": "Drift Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/drift.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Exocraft Mining Laser",
                    "key": "mining",
                    "image": "exocraft/mining.webp",
                    "color": "green",
                    "modules": [
                        { "id": "ML", "type": "core", "label": "Exocraft Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining.webp", },
                        { "id": "AL", "type": "bonus", "label": "Advanced Exocraft Laser", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/advanced-mining.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", }
                    ],
                },
                {
                    "label": "Mounted Cannon",
                    "key": "mounted",
                    "image": "exocraft/mounted.webp",
                    "color": "blue",
                    "modules": [
                        { "id": "MC", "type": "core", "label": "Mounted Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mounted Cannon Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mounted Cannon Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mounted Cannon Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", }
                    ],
                },
                {
                    "label": "Mounted Flamethrower",
                    "key": "flamethrower",
                    "image": "exocraft/flamethrower.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "MF", "type": "core", "label": "Mounted Flamethrower", "bonus": 1.0, "adjacency": "none", "sc_eligible": True, "image": "exocraft/flamethrower.webp", }
                    ],
                },
            ],    
            "Defensive Systems": [
                {
                    "label": "Thermal Buffer",
                    "key": "thermal",
                    "image": "exocraft/thermal.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "TB", "type": "core", "label": "Thermal Buffer", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/thermal.webp", }
                    ],
                },
                {
                    "label": "Megawatt Heater",
                    "key": "cold",
                    "image": "exocraft/cold.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "MH", "type": "core", "label": "Megawatt Heater", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/cold.webp", }
                    ],
                },
                {
                    "label": "Neutron Shielding",
                    "key": "radiation",
                    "image": "exocraft/radiation.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "NS", "type": "core", "label": "Neutron Shielding", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/radiation.webp", }
                    ],
                },
                {
                    "label": "Air Filtration Unit",
                    "key": "toxic",
                    "image": "exocraft/toxic.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "AF", "type": "core", "label": "Air Filtration Unit", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/toxic.webp", }
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Exocraft Radar",
                    "key": "radar",
                    "image": "exocraft/radar.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "ER", "type": "core", "label": "Exocraft Radar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/radar.webp", }
                    ],
                },
                {
                    "label": "Radar Amplifier",
                    "key": "amplifier",
                    "image": "exocraft/amplifier.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "RA", "type": "core", "label": "Radar Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/amplifier.webp", }
 
                    ],
                },
                {
                    "label": "Power Resonator",
                    "key": "power",
                    "image": "exocraft/power.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "PR", "type": "core", "label": "Power Resonator", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/power.webp", }
 
                    ],
                },
            ],
        },
    },
    "pilgrim": {
        "label": "Pilgrim",
        "type": "Exocraft",
        "types": {
            "Propulsion": [
                {
                    "label": "Fusion Engine",
                    "key": "fusion",
                    "image": "exocraft/fusion.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "FE", "type": "core", "label": "Fusion Engine", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exocraft/fusion.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Fusion Engine Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fusion Engine Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fusion Engine Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                    ],
                },
                {
                    "label": "Icarus Fuel System",
                    "key": "icarus",
                    "image": "exocraft/icarus.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "FS", "type": "core", "label": "Icarus Fuel System", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/icarus.webp", },
                    ],
                },
                {
                    "label": "Exocraft Boosters",
                    "key": "boost",
                    "image": "exocraft/boost.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "EB", "type": "core", "label": "Exocraft Boost Module", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exocraft/boost.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Exocraft Boost Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Exocraft Boost Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Exocraft Boost Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                    ],
                },
                {
                    "label": "Hi-Slide Suspension",
                    "key": "slide",
                    "image": "exocraft/slide.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "HS", "type": "core", "label": "Hi-Slide Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/slide.webp", },
                    ],
                },
                {
                    "label": "Grip Boost Suspension",
                    "key": "grip",
                    "image": "exocraft/grip.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "GB", "type": "core", "label": "Grip Boost Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/grip.webp", },
                    ],
                },
                {
                    "label": "Drift Suspension",
                    "key": "drift",
                    "image": "exocraft/drift.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "DS", "type": "core", "label": "Drift Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/drift.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Exocraft Mining Laser",
                    "key": "mining",
                    "image": "exocraft/mining.webp",
                    "color": "green",
                    "modules": [
                        { "id": "ML", "type": "core", "label": "Exocraft Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining.webp", },
                        { "id": "AL", "type": "bonus", "label": "Advanced Exocraft Laser", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/advanced-mining.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                    ],
                },
                {
                    "label": "Mounted Cannon",
                    "key": "mounted",
                    "image": "exocraft/mounted.webp",
                    "color": "blue",
                    "modules": [
                        { "id": "MC", "type": "core", "label": "Mounted Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mounted Cannon Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mounted Cannon Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mounted Cannon Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                    ],
                },
                {
                    "label": "Mounted Flamethrower",
                    "key": "flamethrower",
                    "image": "exocraft/flamethrower.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "MF", "type": "core", "label": "Mounted Flamethrower", "bonus": 1.0, "adjacency": "none", "sc_eligible": True, "image": "exocraft/flamethrower.webp", },
                    ],
                },
            ],    
            "Defensive Systems": [
                {
                    "label": "Thermal Buffer",
                    "key": "thermal",
                    "image": "exocraft/thermal.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "TB", "type": "core", "label": "Thermal Buffer", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/thermal.webp", },
                    ],
                },
                {
                    "label": "Megawatt Heater",
                    "key": "cold",
                    "image": "exocraft/cold.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "MH", "type": "core", "label": "Megawatt Heater", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/cold.webp", },
                    ],
                },
                {
                    "label": "Neutron Shielding",
                    "key": "radiation",
                    "image": "exocraft/radiation.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "NS", "type": "core", "label": "Neutron Shielding", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/radiation.webp", },
                    ],
                },
                {
                    "label": "Air Filtration Unit",
                    "key": "toxic",
                    "image": "exocraft/toxic.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "AF", "type": "core", "label": "Air Filtration Unit", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/toxic.webp", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Exocraft Radar",
                    "key": "radar",
                    "image": "exocraft/radar.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "ER", "type": "core", "label": "Exocraft Radar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/radar.webp", }, 
                    ],
                },
                {
                    "label": "Radar Amplifier",
                    "key": "amplifier",
                    "image": "exocraft/amplifier.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "RA", "type": "core", "label": "Radar Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/amplifier.webp", },
 
                    ],
                },
                {
                    "label": "Power Resonator",
                    "key": "power",
                    "image": "exocraft/power.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "PR", "type": "core", "label": "Power Resonator", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/power.webp", },
 
                    ],
                },
            ],
        },
    },
    "roamer": {
        "label": "Roamer",
        "type": "Exocraft",
        "types": {
            "Propulsion": [
                {
                    "label": "Fusion Engine",
                    "key": "fusion",
                    "image": "exocraft/fusion.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "FE", "type": "core", "label": "Fusion Engine", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exocraft/fusion.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Fusion Engine Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fusion Engine Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fusion Engine Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                    ],
                },
                {
                    "label": "Icarus Fuel System",
                    "key": "icarus",
                    "image": "exocraft/icarus.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "FS", "type": "core", "label": "Icarus Fuel System", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/icarus.webp", },
                    ],
                },
                {
                    "label": "Exocraft Boosters",
                    "key": "boost",
                    "image": "exocraft/boost.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "EB", "type": "core", "label": "Exocraft Boost Module", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exocraft/boost.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Exocraft Boost Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Exocraft Boost Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Exocraft Boost Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                    ],
                },
                {
                    "label": "Hi-Slide Suspension",
                    "key": "slide",
                    "image": "exocraft/slide.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "HS", "type": "core", "label": "Hi-Slide Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/slide.webp", },
                    ],
                },
                {
                    "label": "Grip Boost Suspension",
                    "key": "grip",
                    "image": "exocraft/grip.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "GB", "type": "core", "label": "Grip Boost Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/grip.webp", },
                    ],
                },
                {
                    "label": "Drift Suspension",
                    "key": "drift",
                    "image": "exocraft/drift.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "DS", "type": "core", "label": "Drift Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/drift.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Exocraft Mining Laser",
                    "key": "mining",
                    "image": "exocraft/mining.webp",
                    "color": "green",
                    "modules": [
                        { "id": "ML", "type": "core", "label": "Exocraft Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining.webp", },
                        { "id": "AL", "type": "bonus", "label": "Advanced Exocraft Laser", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/advanced-mining.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                    ],
                },
                {
                    "label": "Mounted Cannon",
                    "key": "mounted",
                    "image": "exocraft/mounted.webp",
                    "color": "blue",
                    "modules": [
                        { "id": "MC", "type": "core", "label": "Mounted Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mounted Cannon Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mounted Cannon Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mounted Cannon Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                    ],
                },
                {
                    "label": "Mounted Flamethrower",
                    "key": "flamethrower",
                    "image": "exocraft/flamethrower.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "MF", "type": "core", "label": "Mounted Flamethrower", "bonus": 1.0, "adjacency": "none", "sc_eligible": True, "image": "exocraft/flamethrower.webp", },
                    ],
                },
            ],    
            "Defensive Systems": [
                {
                    "label": "Thermal Buffer",
                    "key": "thermal",
                    "image": "exocraft/thermal.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "TB", "type": "core", "label": "Thermal Buffer", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/thermal.webp", },
                    ],
                },
                {
                    "label": "Megawatt Heater",
                    "key": "cold",
                    "image": "exocraft/cold.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "MH", "type": "core", "label": "Megawatt Heater", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/cold.webp", },
                    ],
                },
                {
                    "label": "Neutron Shielding",
                    "key": "radiation",
                    "image": "exocraft/radiation.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "NS", "type": "core", "label": "Neutron Shielding", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/radiation.webp", },
                    ],
                },
                {
                    "label": "Air Filtration Unit",
                    "key": "toxic",
                    "image": "exocraft/toxic.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "AF", "type": "core", "label": "Air Filtration Unit", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/toxic.webp", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Exocraft Radar",
                    "key": "radar",
                    "image": "exocraft/radar.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "ER", "type": "core", "label": "Exocraft Radar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/radar.webp", }, 
                    ],
                },
                {
                    "label": "Radar Amplifier",
                    "key": "amplifier",
                    "image": "exocraft/amplifier.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "RA", "type": "core", "label": "Radar Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/amplifier.webp", },
 
                    ],
                },
                {
                    "label": "Power Resonator",
                    "key": "power",
                    "image": "exocraft/power.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "PR", "type": "core", "label": "Power Resonator", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/power.webp", },
 
                    ],
                },
            ],
        },
    },
    "nautilon": {
        "label": "Nautilon",
        "type": "Exocraft",
        "types": {
            "Propulsion": [
                {
                    "label": "Humboldt Drive",
                    "key": "humboldt",
                    "image": "exocraft/humboldt.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "HD", "type": "core", "label": "Humboldt Drive", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exocraft/humboldt.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Humboldt Drive Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/humboldt-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Humboldt Drive Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/humboldt-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Humboldt Drive Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/humboldt-upgrade.webp", },
						{ "id": "OG", "type": "bonus", "label": "Osmatic Generator", "bonus": 0.20, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/osmatic.webp", },
                    ],
                },
                {
                    "label": "Icarus Fuel System",
                    "key": "icarus",
                    "image": "exocraft/icarus.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "FS", "type": "core", "label": "Icarus Fuel System", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/icarus.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Dredging Laser",
                    "key": "dredging",
                    "image": "exocraft/dredging.webp",
                    "color": "green",
                    "modules": [
                        { "id": "DL", "type": "core", "label": "Dredging Laser", "bonus": 1.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/dredging.webp", },
                    ],
                },
                {
                    "label": "Tethys Beam",
                    "key": "tethys",
                    "image": "exocraft/tethys.webp",
                    "color": "blue",
                    "modules": [
                        { "id": "TB", "type": "core", "label": "Tethys Beam,", "bonus": 1.0, "adjacency": "none", "sc_eligible": True, "image": "exocraft/tethys.webp", },
                    ],
                },
                {
                    "label": "Nautilon Cannon",
                    "key": "nautilon",
                    "image": "exocraft/nautilon.webp",
                    "color": "blue",
                    "modules": [
                        { "id": "NC", "type": "core", "label": "Nautilon Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/nautilon.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Nautilon Cannon Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/nautilon-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Nautilon Cannon Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/nautilon-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Nautilon Cannon Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/nautilon-upgrade.webp", },
                    ],
                },
            ],    
            "Utilities": [
                {
                    "label": "Hi-Powered Sonar",
                    "key": "sonar",
                    "image": "exocraft/sonar.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "HS", "type": "core", "label": "Exocraft Radar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/sonar.webp", }, 
                    ],
                },
            ],
        },
    },
    "colossus": {
        "label": "Colossus",
        "type": "Exocraft",
        "types": {
            "Propulsion": [
                {
                    "label": "Fusion Engine",
                    "key": "fusion",
                    "image": "exocraft/fusion.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "FE", "type": "core", "label": "Fusion Engine", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exocraft/fusion.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Fusion Engine Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Fusion Engine Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Fusion Engine Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/fusion-upgrade.webp", },
                    ],
                },
                {
                    "label": "Icarus Fuel System",
                    "key": "icarus",
                    "image": "exocraft/icarus.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "FS", "type": "core", "label": "Icarus Fuel System", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/icarus.webp", },
                    ],
                },
                {
                    "label": "Exocraft Boosters",
                    "key": "boost",
                    "image": "exocraft/boost.webp",
                    "color": "jade",
                    "modules": [
                        { "id": "EB", "type": "core", "label": "Exocraft Boost Module", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exocraft/boost.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Exocraft Boost Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Exocraft Boost Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Exocraft Boost Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/boost-upgrade.webp", },
                    ],
                },
                {
                    "label": "Hi-Slide Suspension",
                    "key": "slide",
                    "image": "exocraft/slide.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "HS", "type": "core", "label": "Hi-Slide Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/slide.webp", },
                    ],
                },
                {
                    "label": "Grip Boost Suspension",
                    "key": "grip",
                    "image": "exocraft/grip.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "GB", "type": "core", "label": "Grip Boost Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/grip.webp", },
                    ],
                },
                {
                    "label": "Drift Suspension",
                    "key": "drift",
                    "image": "exocraft/drift.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "DS", "type": "core", "label": "Drift Suspension", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/drift.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Exocraft Mining Laser",
                    "key": "mining",
                    "image": "exocraft/mining.webp",
                    "color": "green",
                    "modules": [
                        { "id": "ML", "type": "core", "label": "Exocraft Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining.webp", },
                        { "id": "AL", "type": "bonus", "label": "Advanced Exocraft Laser", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/advanced-mining.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                    ],
                },
                {
                    "label": "Mounted Cannon",
                    "key": "mounted",
                    "image": "exocraft/mounted.webp",
                    "color": "blue",
                    "modules": [
                        { "id": "MC", "type": "core", "label": "Mounted Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Mounted Cannon Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Mounted Cannon Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Mounted Cannon Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mounted-upgrade.webp", },
                    ],
                },
                {
                    "label": "Mounted Flamethrower",
                    "key": "flamethrower",
                    "image": "exocraft/flamethrower.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "MF", "type": "core", "label": "Mounted Flamethrower", "bonus": 1.0, "adjacency": "none", "sc_eligible": True, "image": "exocraft/flamethrower.webp", },
                    ],
                },
            ],    
            "Defensive Systems": [
                {
                    "label": "Thermal Buffer",
                    "key": "thermal",
                    "image": "exocraft/thermal.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "TB", "type": "core", "label": "Thermal Buffer", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/thermal.webp", },
                    ],
                },
                {
                    "label": "Megawatt Heater",
                    "key": "cold",
                    "image": "exocraft/cold.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "MH", "type": "core", "label": "Megawatt Heater", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/cold.webp", },
                    ],
                },
                {
                    "label": "Neutron Shielding",
                    "key": "radiation",
                    "image": "exocraft/radiation.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "NS", "type": "core", "label": "Neutron Shielding", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/radiation.webp", },
                    ],
                },
                {
                    "label": "Air Filtration Unit",
                    "key": "toxic",
                    "image": "exocraft/toxic.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "AF", "type": "core", "label": "Air Filtration Unit", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/toxic.webp", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Exocraft Radar",
                    "key": "radar",
                    "image": "exocraft/radar.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "ER", "type": "core", "label": "Exocraft Radar", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/radar.webp", }, 
                    ],
                },
                {
                    "label": "Radar Amplifier",
                    "key": "amplifier",
                    "image": "exocraft/amplifier.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "RA", "type": "core", "label": "Radar Amplifier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/amplifier.webp", },
 
                    ],
                },
                {
                    "label": "Power Resonator",
                    "key": "power",
                    "image": "exocraft/power.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "PR", "type": "core", "label": "Power Resonator", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/power.webp", },
 
                    ],
                },
                {
                    "label": "Mineral Processing Rig",
                    "key": "mineral",
                    "image": "exocraft/mineral.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "MP", "type": "core", "label": "Mineral Processing Rig", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/mineral.webp", },
 
                    ],
                },
            ],
        },
    },
    "minotaur": {
        "label": "Minotaur",
        "type": "Exocraft",
        "types": {
            "Propulsion": [
                {
                    "label": "Daedalus Engine",
                    "key": "daedalus",
                    "image": "exocraft/daedalus.webp",
                    "color": "orange",
                    "modules": [
                        { "id": "DE", "type": "core", "label": "Daedalus Engine", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exocraft/daedalus.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Daedalus Engine Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/daedalus-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Daedalus Engine Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/daedalus-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Daedalus Engine Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/daedalus-upgrade.webp", },
						{ "id": "SG", "type": "bonus", "label": "Self-Greasing Servos", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/servos.webp", },
                    ],
                },
                {
                    "label": "Ariadne's Flame",
                    "key": "ariadnes",
                    "image": "exocraft/ariadnes.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "AF", "type": "core", "label": "Ariadne's Flame", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/ariadnes.webp", },
                    ],
                },
                {
                    "label": "Icarus Fuel System",
                    "key": "icarus",
                    "image": "exocraft/icarus.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "FS", "type": "core", "label": "Icarus Fuel System", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/icarus.webp", },
                    ],
                },
                {
                    "label": "Hardframe Legs",
                    "key": "hardframe-legs",
                    "image": "exocraft/hardframe-legs.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "HL", "type": "core", "label": "Hardframe Legs", "bonus": 0.36, "adjacency": "none", "sc_eligible": False, "image": "exocraft/hardframe-legs.webp", },
                    ],
                },
                {
                    "label": "Liquidator Legs",
                    "key": "liquidator-legs",
                    "image": "exocraft/liquidator-legs.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "LL", "type": "core", "label": "Liqiquidator Legs", "bonus": 0.36, "adjacency": "none", "sc_eligible": False, "image": "exocraft/liquidator-legs.webp", },
                    ],
                },
            ],
            "Weaponry": [
                {
                    "label": "Exocraft Mining Laser",
                    "key": "mining",
                    "image": "exocraft/mining.webp",
                    "color": "green",
                    "modules": [
                        { "id": "ML", "type": "core", "label": "Exocraft Mining Laser", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining.webp", },
                        { "id": "AL", "type": "bonus", "label": "Precision Laser", "bonus": 0.04, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/precision.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Exocraft Mining Laser Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/mining-upgrade.webp", },
                    ],
                },
                {
                    "label": "Minotaur Cannon",
                    "key": "minotaur",
                    "image": "exocraft/minotaur.webp",
                    "color": "blue",
                    "modules": [
                        { "id": "MC", "type": "core", "label": "Minotaur Cannon", "bonus": 1.0, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/minotaur.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Minotaur Cannon Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/minotaur-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Minotaur Cannon Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/minotaur-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Minotaur Cannon Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/minotaur-upgrade.webp", },
                    ],
                },
                {
                    "label": "Hardframe Right Arm",
                    "key": "hardframe-right",
                    "image": "exocraft/hardframe-right.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "HR", "type": "core", "label": "Hardframe Right Arm", "bonus": 1.0, "adjacency": "none", "sc_eligible": True, "image": "exocraft/hardframe-right.webp", },
                    ],
                },
                {
                    "label": "Hardframe Left Arm",
                    "key": "hardframe-left",
                    "image": "exocraft/hardframe-left.webp",
                    "color": "gray",
                    "modules": [
                        { "id": "HL", "type": "core", "label": "Hardframe Left Arm", "bonus": 1.0, "adjacency": "none", "sc_eligible": True, "image": "exocraft/hardframe-left.webp", },
                    ],
                },
                {
                    "label": "Liquidator Right Arm",
                    "key": "liquidator-right",
                    "image": "exocraft/liquidator-right.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "LR", "type": "core", "label": "Liquidator Right Arm", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": True, "image": "exocraft/liquidator-right.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Flamethrower Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/liquidator-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Flamethrower Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/liquidator-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Flamethrower Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exocraft/liquidator-upgrade.webp", },
                    ],
                },
                {
                    "label": "Liquidator Left Arm",
                    "key": "liquidator-left",
                    "image": "exocraft/liquidator-left.webp",
                    "color": "sky",
                    "modules": [
                        { "id": "LL", "type": "core", "label": "Liquidator Left Arm", "bonus": 1.0, "adjacency": "none", "sc_eligible": True, "image": "exocraft/liquidator-left.webp", },
                    ],
                },
            ],    
            "Defensive Systems": [
                {
                    "label": "Environment Control Unit",
                    "key": "environment",
                    "image": "exocraft/environment.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "EC", "type": "core", "label": "Environment Control Unit", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/environment.webp", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Radar Array",
                    "key": "array",
                    "image": "exocraft/array.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "RA", "type": "core", "label": "Radar Array", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/array.webp", }, 
                    ],
                },
                {
                    "label": "AI Pilot",
                    "key": "ai",
                    "image": "exocraft/ai.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "AI", "type": "core", "label": "AI Pilot", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/ai.webp", },
                    ],
                },
                {
                    "label": "Minotaur Bore",
                    "key": "bore",
                    "image": "exocraft/bore.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "MB", "type": "core", "label": "Minotaur Bore", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/bore.webp", },
                    ],
                },
                {
                    "label": "Hardframe Body",
                    "key": "hardframe-body",
                    "image": "exocraft/hardframe-body.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "HB", "type": "core", "label": "Hardframe Body", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/hardframe-body.webp", },
                    ],
                },
                {
                    "label": "Liquidator Body",
                    "key": "liquidator-body",
                    "image": "exocraft/liquidator-body.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "LB", "type": "core", "label": "Liquidator Body", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exocraft/liquidator-body.webp", },
                    ],
                },
            ],
        },
    },
    "exosuit": {
        "label": "Exosuit",
        "type": "Other",
        "types": {
            "Movement Systems": [
                {
                    "label": "Jetpack",
                    "key": "jetpack",
                    "image": "exosuit/jetpack.webp",
                    "color": "yellow",
                    "modules": [
                        { "id": "JP", "type": "core", "label": "Jetpack", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exosuit/jetpack.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Jetpack Upgrade Theta", "bonus": 0.30, "adjacency": "greater", "sc_eligible": True, "image": "exosuit/jetpack-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Jetpack Upgrade Tau", "bonus": 0.29, "adjacency": "greater", "sc_eligible": True, "image": "exosuit/jetpack-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Jetpack Upgrade Sigma", "bonus": 0.28, "adjacency": "greater", "sc_eligible": True, "image": "exosuit/jetpack-upgrade.webp", },
                        { "id": "WJ", "type": "bonus", "label": "Efficient Water Jets", "bonus": 0.00, "adjacency": "greater", "sc_eligible": True, "image": "exosuit/waterjets.webp", },
                        { "id": "NS", "type": "bonus", "label": "Neural Stimulator", "bonus": 0.10, "adjacency": "greater", "sc_eligible": True, "image": "exosuit/neural.webp", },
                        { "id": "AB", "type": "bonus", "label": "Airburst Engine", "bonus": 0.00, "adjacency": "greater", "sc_eligible": True, "image": "exosuit/airburst.webp", },
                        { "id": "RB", "type": "reward", "label": "Rocket Boots", "bonus": 0.00, "adjacency": "greater", "sc_eligible": True, "image": "exosuit/boots.webp", },
                   ],
                },
                {
                    "label": "Personal Refiner",
                    "key": "refiner",
                    "image": "exosuit/refiner.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "PR", "type": "core", "label": "Personal Refiner", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/refiner.webp", },
                    ],
                },
            ],
            "Life Support": [
                {
                    "label": "Life Support",
                    "key": "life",
                    "image": "exosuit/life.webp",
                    "color": "blue",
                    "modules": [
                        { "id": "LS", "type": "core", "label": "Life Support", "bonus": 1.0, "adjacency": "lesser", "sc_eligible": False, "image": "exosuit/life.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Life Support Upgrade Theta", "bonus": 0.30, "adjacency": "lesser", "sc_eligible": True, "image": "exosuit/life-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Life Support Upgrade Tau", "bonus": 0.29, "adjacency": "lesser", "sc_eligible": True, "image": "exosuit/life-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Life Support Upgrade Sigma", "bonus": 0.28, "adjacency": "lesser", "sc_eligible": True, "image": "exosuit/life-upgrade.webp", },
 					    { "id": "RR", "type": "bonus", "label": "Oxygen Rereouter", "bonus": 0.00, "adjacency": "none", "sc_eligible": True, "image": "exosuit/rerouter.webp", },
 					    { "id": "OR", "type": "bonus", "label": "Oxygen Recycler", "bonus": 0.10, "adjacency": "lesser", "sc_eligible": True, "image": "exosuit/recycler.webp", },
                   ],
                },
            ],
            "Hazard Protection": [
                {
                    "label": "Hazard Protection",
                    "key": "hazard",
                    "image": "exosuit/hazard.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "HP", "type": "core", "label": "Hazard Protection", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/hazard.webp", },
                    ],
                },
                {
                    "label": "Pressure Membrane",
                    "key": "pressure",
                    "image": "exosuit/pressure.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "PM", "type": "core", "label": "Pressure Membrane", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/pressure.webp", },
                    ],
                },
                {
                    "label": "Coolant Network",
                    "key": "coolant",
                    "image": "exosuit/coolant.webp",
                    "color": "white",
                    "modules": [
                        { "id": "CN", "type": "bonus", "label": "Coolant Networt", "bonus": 1.00, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/coolant.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Coolant Network Upgrade Theta", "bonus": 0.30, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/coolant-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Coolant Network Upgrade Tau", "bonus": 0.29, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/coolant-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Coolant Network Upgrade Sigma", "bonus": 0.28, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/coolant-upgrade.webp", },
                   ],
                },
                {
                    "label": "Radiation Deflector",
                    "key": "radiation",
                    "image": "exosuit/radiation.webp",
                    "color": "white",
                    "modules": [
                        { "id": "RD", "type": "bonus", "label": "Radiation Deflector", "bonus": 1.00, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/radiation.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Radiation Deflector Upgrade Theta", "bonus": 0.30, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/radiation-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Radiation Deflector Upgrade Tau", "bonus": 0.29, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/radiation-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Radiation Deflector Upgrade Sigma", "bonus": 0.28, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/radiation-upgrade.webp", },
                   ],
                },
               {
                    "label": "Aeration Membrane",
                    "key": "aeration",
                    "image": "exosuit/aeration.webp",
                    "color": "white",
                    "modules": [
                        { "id": "AM", "type": "bonus", "label": "Aeration Membrane", "bonus": 1.00, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/aeration.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Aeration Membrane Upgrade Theta", "bonus": 0.30, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/aeration-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Aeration Membrane Upgrade Tau", "bonus": 0.29, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/aeration-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Aeration Membrane Upgrade Sigma", "bonus": 0.28, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/aeration-upgrade.webp", },
                   ],
                },
               {
                    "label": "Thermic Layer",
                    "key": "thermic",
                    "image": "exosuit/thermic.webp",
                    "color": "white",
                    "modules": [
                        { "id": "TL", "type": "bonus", "label": "Thermic Layer", "bonus": 1.00, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/thermic.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Thermic Layer Upgrade Theta", "bonus": 0.30, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/thermic-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Thermic Layer Upgrade Tau", "bonus": 0.29, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/thermic-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Thermic Layer Upgrade Sigma", "bonus": 0.28, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/thermic-upgrade.webp", },
                   ],
                },
               {
                    "label": "Toxin Suppressor",
                    "key": "toxin",
                    "image": "exosuit/toxin.webp",
                    "color": "white",
                    "modules": [
                        { "id": "TS", "type": "bonus", "label": "Toxin Suppressor", "bonus": 1.00, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/toxin.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Toxin Suppressor Upgrade Theta", "bonus": 0.30, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/toxin-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Toxin Suppressor Upgrade Tau", "bonus": 0.29, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/toxin-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Toxin Suppressor Upgrade Sigma", "bonus": 0.28, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/toxin-upgrade.webp", },
                   ],
                },
                {
                    "label": "Sheild Lattice",
                    "key": "protection",
                    "image": "exosuit/protection.webp",
                    "color": "white",
                    "modules": [
                        { "id": "SL", "type": "bonus", "label": "Sheild Lattice", "bonus": 1.00, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/protection.webp", },
                        { "id": "Xa", "type": "bonus", "label": "Sheild Lattice Upgrade Theta", "bonus": 0.30, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/protection-upgrade.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Sheild Lattice Upgrade Tau", "bonus": 0.29, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/protection-upgrade.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Sheild Lattice Upgrade Sigma", "bonus": 0.28, "adjacency": "greater_2", "sc_eligible": True, "image": "exosuit/protection-upgrade.webp", },
                   ],
                },
            ],
            "Upgrade Modules": [
                {
                    "label": "Defense Systems",
                    "key": "defense",
                    "image": "exosuit/defense.webp",
                    "color": "crimson",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Defense Systems Theta", "bonus": 0.30, "adjacency": "greater_1", "sc_eligible": True, "image": "exosuit/defense.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Defense Systems Tau", "bonus": 0.29, "adjacency": "greater_1", "sc_eligible": True, "image": "exosuit/defense.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Defense Systems Sigma", "bonus": 0.28, "adjacency": "greater_1", "sc_eligible": True, "image": "exosuit/defense.webp", },
                   ],
                },
                {
                    "label": "Rebuilt Exosuit Module",
                    "key": "rebuilt",
                    "image": "exosuit/rebuilt.webp",
                    "color": "crimson",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Rebuilt Exosuit Module Theta", "bonus": 0.30, "adjacency": "greater_1", "sc_eligible": True, "image": "exosuit/rebuilt.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Rebuilt Exosuit Module Tau", "bonus": 0.29, "adjacency": "greater_1", "sc_eligible": True, "image": "exosuit/rebuilt.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Rebuilt Exosuit Module Sigma", "bonus": 0.28, "adjacency": "greater_1", "sc_eligible": True, "image": "exosuit/rebuilt.webp", },
                   ],
                },
                {
                    "label": "Forbidden Exosuit Module",
                    "key": "forbidden",
                    "image": "exosuit/forbidden.webp",
                    "color": "crimson",
                    "modules": [
                        { "id": "Xa", "type": "bonus", "label": "Forbidden Exosuit Module Theta", "bonus": 0.30, "adjacency": "greater_1", "sc_eligible": True, "image": "exosuit/forbidden.webp", },
                        { "id": "Xb", "type": "bonus", "label": "Forbidden Exosuit Module Tau", "bonus": 0.29, "adjacency": "greater_1", "sc_eligible": True, "image": "exosuit/forbidden.webp", },
                        { "id": "Xc", "type": "bonus", "label": "Forbidden Exosuit Module Sigma", "bonus": 0.28, "adjacency": "greater_1", "sc_eligible": True, "image": "exosuit/forbidden.webp", },
                   ],
                },                
            ],
            "Quest Rewards": [
                {
                    "label": "Anomaly Suppressor",
                    "key": "anomaly",
                    "image": "exosuit/anomaly.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "AS", "type": "core", "label": "Anomaly Suppressor", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/anomaly.webp", },
                    ],
                },
                {
                    "label": "Mark of the Denier",
                    "key": "denier",
                    "image": "exosuit/denier.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "MD", "type": "core", "label": "Mark of the Denier", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/denier.webp", },
                    ],
                },
                {
                    "label": "Remembrance",
                    "key": "remembrance",
                    "image": "exosuit/remembrance.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "RM", "type": "core", "label": "Remembrance", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/remembrance.webp", },
                    ],
                },
                {
                    "label": "Star Seed",
                    "key": "starseed",
                    "image": "exosuit/starseed.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "SS", "type": "core", "label": "Star Seed", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/starseed.webp", },
                    ],
                },
            ],
            "Utilities": [
                {
                    "label": "Tranlators",
                    "key": "translators",
                    "image": "exosuit/translator.webp",
                    "color": "white",
                    "modules": [
                        { "id": "Ta", "type": "bonus", "label": "Artemis' Translator", "bonus": 0.00, "adjacency": "lesser", "sc_eligible": True, "image": "exosuit/artemis-translator.webp", },
                        { "id": "Tb", "type": "bonus", "label": "Simple Translator", "bonus": 0.00, "adjacency": "lesser", "sc_eligible": True, "image": "exosuit/simple.webp", },
                        { "id": "Tc", "type": "bonus", "label": "Superior Translator", "bonus": 0.00, "adjacency": "lesser", "sc_eligible": True, "image": "exosuit/superior.webp", },
                        { "id": "Td", "type": "bonus", "label": "Advanced Translator", "bonus": 0.00, "adjacency": "lesser", "sc_eligible": True, "image": "exosuit/advanced.webp", },
                   ],
                },
                {
                    "label": "Haz-Mat Gauntlets",
                    "key": "hazmat",
                    "image": "exosuit/hazmat.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "HG", "type": "core", "label": "Haz-Mat Gauntlets", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/hazmat.webp", },
                    ],
                },
                {
                    "label": "Nutrient Ingestor",
                    "key": "nutrient",
                    "image": "exosuit/nutrient.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "NI", "type": "core", "label": "Nutrient Ingestor", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/nutrient.webp", },
                    ],
                },
                {
                    "label": "Exo Skiff",
                    "key": "skiff",
                    "image": "exosuit/skiff.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "ES", "type": "core", "label": "Exo Skiff", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/skiff.webp", },
                    ],
                },
                {
                    "label": "Trade Rocket",
                    "key": "trade",
                    "image": "exosuit/trade.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "TR", "type": "core", "label": "Trade Rocket", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/trade.webp", },
                    ],
                },
                {
                    "label": "Exocraft Summoning Unit",
                    "key": "exocraft",
                    "image": "exosuit/exocraft.webp",
                    "color": "gray",
                    "modules": [
                         { "id": "ES", "type": "core", "label": "Exosuit Summoning Unit", "bonus": 0.0, "adjacency": "none", "sc_eligible": False, "image": "exosuit/exocraft.webp", },
                    ],
                },
            ],
        },
    },
}
