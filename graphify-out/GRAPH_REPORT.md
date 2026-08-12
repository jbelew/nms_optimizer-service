# Graph Report - nms_optimizer-service  (2026-08-12)

## Corpus Check
- 141 files · ~118,489 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2273 nodes · 4101 edges · 333 communities (106 shown, 227 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 77 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `54966b94`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- place_module
- Grid
- TestOptimizationPipeline
- clear_all_modules_of_tech
- lib.rs
- TestScEligibleAdversarial
- package.json
- data_loader.py
- TestOptimization
- TestScEligibleEdgeCases
- TestOptimizeOptimizationFlow
- check_all_modules_placed
- _convert_map_keys_to_tuple
- mirror_pattern_horizontally
- _scan_grid_with_window
- calculate_grid_score
- ModulePlacementCNN
- TestRefinePlacement
- calculate_window_score
- TestClearAllModulesOfTech
- find_supercharged_opportunities
- TestPlaceModule
- calculate_pattern_adjacency_score
- ._create_grid
- create_localized_grid_ml
- TestPatternApplicationToGrid
- test_analytics.py
- patch
- TestAvailableModulesPatternMatching
- TestPlaceAllModulesInEmptySlots
- rotate_pattern
- TestGA4Client
- Implementation Plan - Improve Group Adjacency Weights
- app.py
- GA4Client
- TestPatternAdjacencyScoring
- TestOptimizationEndpoint
- TestPlacementAlgorithmLogic
- AnalyticsEvent
- TestGetTrainingModuleIds
- TestPartialModuleSetEdgeCases
- get_all_unique_pattern_variations
- TestFilterSolves
- TestWindowSizeValidation
- TestRequestValidation
- TestGetModuleData
- TestGetSolveMap
- TestMLPlacementOutputValidation
- TestWindowScoringAndSelection
- TestRefinementStageEdgeCases
- TestFilterSolvesEdgeCases
- TestErrorHandling
- TestMLPlacementModelLoading
- test_pattern_matching.py
- TestFilterSolvesPhotonixOverride
- format_moduleDefs.py
- filter_solves
- TestAppInitialization
- TestGridUtils
- TestMLPlacementModuleAssignment
- TestMLPlacementPolishing
- TestMLPlacementGridHandling
- TestMLPlacementErrorHandling
- TestOldDetermineWindowDimensionsBehavior
- TestSuperchargedWindowDetection
- TestScEligibleWindowSizeWithConstraint
- determine_window_dimensions
- TestTechTreeEndpoint
- .test_old_logic_case_238_staves_cloaking_count_1
- TestGetAllSolveData
- TestGetAllModuleData
- TestDataLoaderErrorHandling
- TestMLPlacementTensorPreparation
- TestMLPlacementProgressCallback
- TestMLPlacementIntegration
- TestInitialPlacementFallback
- TestDetermineWindowDimensions
- .test_old_logic_case_176_pilgrim_mounted_count_4
- analyze_benchmarks.py
- .test_old_logic_case_226_staves_analysis_count_1
- test_app_endpoints.py
- TestAnalyticsEndpoint
- TestResponseFormat
- TestCORSHeaders
- TestContentNegotiation
- TestEndpointIntegration
- .test_old_logic_case_094_nomad_boost_count_4
- generate_old_logic_tests.py
- TestEventNameValidation
- TestPerformanceAnalyticsEndpoint
- TestDataIntegrity
- optimize_placement
- apply_labels.py
- .get_cell
- logger.py
- optimizer.py
- .test_old_logic_case_095_nomad_slide_count_1
- debug_window_dimensions.py
- NMS Optimizer Service README
- inject_window_rules.py
- Gemini Agent: Core Directives
- generate_solves.sh
- CI/CD Workflow
- run_solvers_tmux.sh
- run_solvers_tmux_weapons.sh
- run_standard.sh
- run_standard-master.sh
- .test_old_logic_case_164_solar_conflict_scanner_count_1
- .test_old_logic_case_165_solar_economy_scanner_count_1
- .test_old_logic_case_105_nomad_cyclops_count_1
- .test_filter_solves_nonexistent_ship
- .test_old_logic_case_167_solar_trails_count_12
- .test_old_logic_case_168_solar_teleporter_count_1
- .test_filter_solves_excludes_unowned_modules
- .test_old_logic_case_170_pilgrim_icarus_count_1
- .test_old_logic_case_172_pilgrim_slide_count_1
- .test_old_logic_case_173_pilgrim_grip_count_1
- .test_old_logic_case_174_pilgrim_drift_count_1
- .test_old_logic_case_175_pilgrim_mining_count_5
- Changes Made
- .test_old_logic_case_015_standard_trails_count_12
- .test_old_logic_case_177_pilgrim_flamethrower_count_1
- .test_filter_solves_missing_score_defaults_to_zero
- .test_filter_solves_preserves_score
- .test_old_logic_case_184_pilgrim_amplifier_count_1
- .test_old_logic_case_185_pilgrim_power_count_1
- .test_old_logic_case_016_standard_teleporter_count_1
- .test_old_logic_case_187_nautilon_icarus_count_1
- .test_old_logic_case_188_nautilon_dredging_count_1
- get_module_data
- .test_old_logic_case_192_nautilon_sonar_count_1
- .test_old_logic_case_193_living_grafted_count_4
- .test_old_logic_case_194_living_spewing_count_4
- .test_old_logic_case_195_living_scream_count_4
- .test_old_logic_case_196_living_assembly_count_5
- .test_old_logic_case_198_living_pulsing_count_4
- .test_old_logic_case_200_living_bobble_count_4
- .test_old_logic_case_201_living_scanners_count_2
- .test_old_logic_case_204_exosuit_refiner_count_1
- .test_old_logic_case_205_exosuit_life_count_6
- .test_old_logic_case_018_atlantid_analysis_count_1
- .test_old_logic_case_207_exosuit_anomaly_count_1
- .test_old_logic_case_208_exosuit_hazard_count_1
- .test_old_logic_case_209_exosuit_pressure_count_1
- .test_old_logic_case_211_exosuit_radiation_count_4
- .test_old_logic_case_213_exosuit_thermic_count_4
- .test_old_logic_case_214_exosuit_toxin_count_4
- .test_old_logic_case_216_exosuit_defense_count_3
- .test_old_logic_case_217_exosuit_rebuilt_count_3
- .test_old_logic_case_218_exosuit_forbidden_count_3
- .test_old_logic_case_220_exosuit_hazmat_count_1
- .test_old_logic_case_221_exosuit_nutrient_count_1
- .test_old_logic_case_222_exosuit_skiff_count_1
- .test_old_logic_case_223_exosuit_trade_count_1
- .test_old_logic_case_224_exosuit_exocraft_count_1
- .test_old_logic_case_225_staves_mining_count_6
- .test_old_logic_case_227_staves_fishing_count_1
- .test_old_logic_case_228_staves_gravatino_count_1
- .test_old_logic_case_229_staves_scanner_count_7
- .test_old_logic_case_230_staves_survey_count_1
- .test_old_logic_case_232_staves_bolt_caster_count_9
- .test_old_logic_case_234_staves_neutron_count_5
- .test_old_logic_case_235_staves_plasma_launcher_count_4
- .test_old_logic_case_021_atlantid_scanner_count_7
- .test_old_logic_case_237_staves_scatter_count_5
- .test_old_logic_case_171_pilgrim_boost_count_4
- .test_old_logic_case_239_staves_combat_count_1
- .test_old_logic_case_240_staves_voltaic_amplifier_count_1
- .test_old_logic_case_242_staves_personal_count_1
- .test_old_logic_case_243_staves_terrian_count_1
- .test_old_logic_case_246_sentinel_mt_fishing_count_1
- .test_old_logic_case_247_sentinel_mt_gravatino_count_1
- .test_old_logic_case_249_sentinel_mt_survey_count_1
- .test_old_logic_case_253_sentinel_mt_neutron_count_5
- .test_old_logic_case_254_sentinel_mt_plasma_launcher_count_4
- .test_old_logic_case_023_atlantid_blaze_javelin_count_6
- .test_old_logic_case_258_sentinel_mt_combat_count_1
- .test_old_logic_case_259_sentinel_mt_voltaic_amplifier_count_1
- .test_old_logic_case_260_sentinel_mt_paralysis_count_1
- .test_old_logic_case_262_sentinel_mt_terrian_count_1
- .test_old_logic_case_265_roamer_boost_count_4
- .test_old_logic_case_266_roamer_slide_count_1
- .test_old_logic_case_024_atlantid_bolt_caster_count_9
- .test_old_logic_case_268_roamer_drift_count_1
- .test_old_logic_case_270_roamer_mounted_count_4
- .test_old_logic_case_271_roamer_flamethrower_count_1
- .test_old_logic_case_275_roamer_toxic_count_1
- .test_old_logic_case_276_roamer_cyclops_count_1
- .test_old_logic_case_025_atlantid_geology_count_4
- .test_old_logic_case_000_standard_cyclotron_count_5
- .test_old_logic_case_027_atlantid_plasma_launcher_count_4
- .test_old_logic_case_028_atlantid_pulse_spitter_count_7
- .test_old_logic_case_251_sentinel_mt_bolt_caster_count_9
- .test_old_logic_case_030_atlantid_cloaking_count_1
- .test_old_logic_case_034_atlantid_personal_count_1
- .test_old_logic_case_035_atlantid_terrian_count_1
- .test_old_logic_case_036_corvette_cyclotron_count_5
- .test_old_logic_case_001_standard_infra_count_5
- .test_old_logic_case_038_corvette_phase_count_5
- .test_old_logic_case_039_corvette_photon_count_5
- .test_old_logic_case_043_corvette_hyper_count_9
- .test_old_logic_case_045_corvette_launch_count_6
- .test_old_logic_case_046_corvette_pulse_count_8
- .test_old_logic_case_002_standard_phase_count_5
- .test_old_logic_case_047_corvette_habitation_count_3
- .test_old_logic_case_049_corvette_bobble_count_4
- .test_old_logic_case_050_corvette_conflict_scanner_count_1
- .test_old_logic_case_052_corvette_cargo_scanner_count_1
- .test_old_logic_case_055_standard_mt_mining_count_6
- .test_old_logic_case_056_standard_mt_analysis_count_1
- .test_old_logic_case_003_standard_photon_count_5
- .test_old_logic_case_057_standard_mt_fishing_count_1
- .test_old_logic_case_059_standard_mt_scanner_count_7
- .test_old_logic_case_060_standard_mt_survey_count_1
- .test_old_logic_case_061_standard_mt_blaze_javelin_count_6
- .test_old_logic_case_062_standard_mt_bolt_caster_count_9
- .test_old_logic_case_064_standard_mt_neutron_count_5
- .test_old_logic_case_065_standard_mt_plasma_launcher_count_4
- .test_old_logic_case_066_standard_mt_pulse_spitter_count_7
- .test_old_logic_case_004_standard_positron_count_5
- .test_old_logic_case_071_standard_mt_paralysis_count_1
- .test_old_logic_case_072_standard_mt_personal_count_1
- .test_old_logic_case_074_sentinel_cyclotron_count_5
- .test_old_logic_case_075_sentinel_infra_count_5
- .test_old_logic_case_079_sentinel_photon_count_5
- .test_old_logic_case_080_sentinel_shield_count_5
- .test_old_logic_case_081_sentinel_launch_count_6
- .test_old_logic_case_083_sentinel_pulse_count_8
- .test_old_logic_case_086_sentinel_pilot_count_1
- .test_old_logic_case_006_standard_shield_count_5
- .test_old_logic_case_087_sentinel_conflict_scanner_count_1
- .test_old_logic_case_088_sentinel_economy_scanner_count_1
- .test_old_logic_case_089_sentinel_cargo_scanner_count_1
- .test_old_logic_case_090_sentinel_trails_count_12
- .test_old_logic_case_092_nomad_fusion_count_4
- .test_old_logic_case_040_corvette_positron_count_5
- .test_old_logic_case_096_nomad_grip_count_1
- .test_old_logic_case_098_nomad_mining_count_5
- .test_old_logic_case_100_nomad_flamethrower_count_1
- .test_old_logic_case_101_nomad_thermal_count_1
- .test_old_logic_case_102_nomad_cold_count_1
- .test_old_logic_case_103_nomad_radiation_count_1
- .test_old_logic_case_106_nomad_radar_count_1
- .test_old_logic_case_107_nomad_amplifier_count_1
- .test_old_logic_case_108_nomad_power_count_1
- .test_old_logic_case_111_minotaur_icarus_count_1
- .test_old_logic_case_112_minotaur_minotaur_laser_count_6
- .test_old_logic_case_113_minotaur_minotaur_count_4
- .test_old_logic_case_114_minotaur_hardframe_right_count_1
- .test_old_logic_case_115_minotaur_liquidator_right_count_4
- .test_old_logic_case_117_minotaur_environment_count_1
- .test_old_logic_case_118_minotaur_cyclops_count_1
- .test_old_logic_case_119_minotaur_array_count_1
- .test_old_logic_case_120_minotaur_ai_count_1
- .test_old_logic_case_121_minotaur_bore_count_1
- .test_old_logic_case_123_minotaur_liquidator_body_count_1
- .test_old_logic_case_124_freighter_hyper_count_11
- .test_old_logic_case_125_freighter_interstellar_count_1
- .test_old_logic_case_010_standard_aqua_count_1
- .test_old_logic_case_127_freighter_fleet_fuel_count_3
- .test_old_logic_case_128_freighter_fleet_speed_count_3
- .test_old_logic_case_129_freighter_fleet_combat_count_3
- .test_old_logic_case_130_freighter_fleet_exploration_count_3
- .test_old_logic_case_131_freighter_fleet_mining_count_3
- .test_old_logic_case_134_colossus_icarus_count_1
- .test_old_logic_case_138_colossus_drift_count_1
- .test_old_logic_case_139_colossus_mining_count_5
- .test_old_logic_case_141_colossus_flamethrower_count_1
- .test_old_logic_case_142_colossus_thermal_count_1
- .test_old_logic_case_146_colossus_radar_count_1
- .test_old_logic_case_012_standard_conflict_scanner_count_1
- .test_old_logic_case_148_colossus_excavation_count_1
- .test_old_logic_case_149_colossus_amplifier_count_1
- .test_old_logic_case_150_colossus_power_count_1
- .test_old_logic_case_151_colossus_mineral_count_1
- .test_old_logic_case_152_solar_cyclotron_count_5
- .test_old_logic_case_153_solar_infra_count_5
- .test_old_logic_case_154_solar_phase_count_5
- .test_old_logic_case_155_solar_photon_count_5
- .test_old_logic_case_156_solar_positron_count_5
- .test_old_logic_case_013_standard_economy_scanner_count_1
- .test_old_logic_case_157_solar_rocket_count_2
- .test_old_logic_case_159_solar_hyper_count_9
- .test_old_logic_case_161_solar_pulse_count_9
- .test_old_logic_case_162_solar_aqua_count_1
- .test_old_logic_case_163_solar_bobble_count_4
- .test_pulse_spitter_jetpack_less_than_8
- .test_pulse_spitter_jetpack_8_plus
- .test_pulse_6_modules
- .test_pulse_7_to_8_modules
- .test_pulse_9_plus_modules
- .test_generic_fallback_less_than_3
- .test_generic_fallback_3_modules
- .test_generic_fallback_4_modules
- .test_generic_fallback_5_to_6_modules
- .test_generic_fallback_7_modules
- .test_generic_fallback_8_modules
- .test_generic_fallback_10_plus_modules
- .test_very_large_module_count
- .test_dimensions_are_positive
- .test_zero_modules_returns_default
- .test_negative_module_count_treated_as_zero
- .test_sentinel_photonix_override
- .test_sentinel_photonix_override_regardless_module_count
- .test_corvette_pulse_7_modules
- .test_corvette_7_modules_non_pulse
- .test_hyper_12_plus_modules
- .test_hyper_10_to_11_modules
- .test_hyper_9_modules
- .test_hyper_less_than_9_modules
- .test_old_logic_case_085_sentinel_bobble_count_4
- .test_filter_solves_preserves_map_keys
- update_wheel.sh
- Architecture Design Document
- Project Backlog
- Product Requirements Document
- Software Requirements Specification
- Technical Design Document
- Google Analytics 4 API
- nms-optimizer-service
- .test_filter_solves_nonexistent_tech
- .test_filter_solves_none_string_handling

## God Nodes (most connected - your core abstractions)
1. `determine_window_dimensions()` - 325 edges
2. `get_module_data()` - 303 edges
3. `TestOldDetermineWindowDimensionsBehavior` - 282 edges
4. `Grid` - 226 edges
5. `optimize_placement()` - 66 edges
6. `place_module()` - 55 edges
7. `clear_all_modules_of_tech()` - 39 edges
8. `calculate_grid_score()` - 35 edges
9. `AnalyticsEvent` - 34 edges
10. `find_supercharged_opportunities()` - 34 edges

## Surprising Connections (you probably didn't know these)
- `generate_solve_map()` --calls--> `Grid`  [EXTRACTED]
  scripts/debugging_utils/generate_solves.py → src/grid_utils.py
- `generate_solve_map()` --calls--> `Grid`  [EXTRACTED]
  scripts/debugging_utils/solve_map_generator.py → src/grid_utils.py
- `generate_solve_map()` --calls--> `get_tech_modules()`  [EXTRACTED]
  scripts/debugging_utils/solve_map_generator.py → src/modules_utils.py
- `generate_solve_map()` --calls--> `get_tech_modules_for_training()`  [EXTRACTED]
  scripts/debugging_utils/solve_map_generator.py → src/modules_utils.py
- `generate_solve_map()` --calls--> `simulated_annealing()`  [EXTRACTED]
  scripts/debugging_utils/solve_map_generator.py → src/optimization/refinement.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Core Project Documentation** — docs_architecture_design_document_md, docs_product_requirements_document_md, docs_software_requirements_specification_md, docs_technical_design_document_md [EXTRACTED 1.00]

## Communities (333 total, 227 thin omitted)

### Community 0 - "place_module"
Cohesion: 0.12
Nodes (21): place_module(), Populates a grid cell with the data for a specific module. This function sets…, get_tech_modules(), Retrieves and filters module definitions for a specific technology. This…, Core optimization module for NMS grid module placement. This module…, place_all_modules_in_empty_slots(), Helper utilities for the optimization module. Provides helper functions for: -…, Places all modules of a given tech into the remaining empty slots. Iterates… (+13 more)

### Community 1 - "Grid"
Cohesion: 0.03
Nodes (40): Grid, Sets the label for a module in a specific cell. Args: x (int): The x-coordinate…, Sets the supercharged status of a specific cell. Args: x (int): The…, Sets the active status of a specific cell. Args: x (int): The x-coordinate of…, Sets the adjacency bonus for a module in a specific cell. Args: x (int): The…, Sets the base bonus for a module in a specific cell. Args: x (int): The…, Sets the type for a module in a specific cell. Args: x (int): The x-coordinate…, Sets the 'value' property for a module in a specific cell. Args: x (int): The… (+32 more)

### Community 2 - "TestOptimizationPipeline"
Cohesion: 0.16
Nodes (10): get_tech_modules_from_ship_data(), Test that optimization considers supercharged slots. This test verifies: 1.…, Test optimization on grid with some inactive cells. This test verifies: 1.…, Extract modules for a specific tech from ship data structure. Ship data…, Test that final score matches grid calculation. This test verifies: 1.…, Test full optimization workflow end-to-end., Set up common test resources., Test pipeline from pattern matching to refinement. This test verifies: 1.… (+2 more)

### Community 3 - "clear_all_modules_of_tech"
Cohesion: 0.07
Nodes (41): generate_all_solves(), generate_solve_map(), generate_solve_map_template(), load_trails_stub(), Loads the trails stub data from a JSON file., Saves the solves data to a JSON file., Generates a single solve map for a given technology and ship type. Args:…, Generates a solve map template from a Grid object. (+33 more)

### Community 4 - "lib.rs"
Cohesion: 0.15
Nodes (31): AdjacencyType, Bound, D, Error, ModuleType, Option, Py, PyAny (+23 more)

### Community 5 - "TestScEligibleAdversarial"
Cohesion: 0.09
Nodes (12): When modules are mixed (some eligible, some not),…, Verify that modules without sc_eligible specified default to False (non-…, Test require_non_supercharge with a grid that has both supercharged and non-…, Adversarial tests for sc_eligible constraint enforcement., Setting both require_supercharge=True and require_non_supercharge=True would be…, Test windows near grid boundaries with sc_eligible constraint., When require_non_supercharge=True, _scan_grid_with_window should skip any…, Verify that sc_eligible: False and missing sc_eligible are treated the same. (+4 more)

### Community 6 - "package.json"
Cohesion: 0.06
Nodes (33): commitizen, @commitlint/cli, @commitlint/config-conventional, @elsikora/commitizen-plugin-commitlint-ai, husky, author, bugs, url (+25 more)

### Community 7 - "data_loader.py"
Cohesion: 0.07
Nodes (38): generate_solve_map(), generate_solve_map_template(), Generates a solve map template from a Grid object., Generates a single solve map for a given technology and ship type. Args:…, Runs a benchmark on the simulated annealing algorithm., run_benchmark(), generate_all_input_grids(), _generate_random_input_grid() (+30 more)

### Community 8 - "TestOptimization"
Cohesion: 0.08
Nodes (16): mirror_pattern_vertically(), Mirrors a pattern vertically. Args: pattern (dict): A dictionary where keys are…, patch, Test behavior when no solve map is found for the ship (uses adjacency scoring)., Test returns 'Pattern No Fit' when solve map exists, no pattern fits, and not…, Test fallback to initial SA when solve map exists, no pattern fits, and…, Set up common test resources., Test returns 'Pattern No Fit' when a partial set has no window and is not… (+8 more)

### Community 9 - "TestScEligibleEdgeCases"
Cohesion: 0.08
Nodes (18): create_grid(), Test that place_all_modules_in_empty_slots prefers supercharged slots for…, Test that refine_placement skips permutations placing non-eligible in…, Test that find_supercharged_opportunities returns None when all modules are…, Test that find_supercharged_opportunities works when all modules are…, Test that core.py fallback placement skips supercharged for non-eligible., Helper to create a test grid., Test placement with mix of eligible and non-eligible modules. (+10 more)

### Community 10 - "TestOptimizeOptimizationFlow"
Cohesion: 0.07
Nodes (15): Tests for overall optimization flow edge cases, With zero solve score and positive bonus, percentage should be 100%, Percentage should be (bonus / solve_score) * 100, With zero bonus and nonzero solve score, percentage should be 0%, Bonus exceeding solve score should be possible, Grid modifications shouldn't affect original, solve_method should be set appropriately, Pulse tech with only 'PC' missing should be treated as full set (+7 more)

### Community 11 - "check_all_modules_placed"
Cohesion: 0.07
Nodes (25): check_all_modules_placed(), count_empty_in_localized(), Grid, Counts unoccupied module slots in a localized grid. Iterates through all cells…, Verifies that all expected modules for a tech are placed in the grid. Compares…, Adversarial tests for count_empty_in_localized, Completely empty grid should return full count, Completely full grid should return 0 (+17 more)

### Community 12 - "_convert_map_keys_to_tuple"
Cohesion: 0.11
Nodes (15): _convert_map_keys_to_tuple(), Recursively converts string keys in a dictionary back to tuples. This is an…, Test the tuple key conversion utility., Convert simple '0,0' string to (0,0) tuple., Convert multiple tuple string keys., Non-tuple-like string keys should be preserved., Convert tuple keys in nested dictionaries., Convert tuple keys in dictionaries within lists. (+7 more)

### Community 13 - "mirror_pattern_horizontally"
Cohesion: 0.11
Nodes (12): mirror_pattern_horizontally(), Mirrors a pattern horizontally. Args: pattern (dict): A dictionary where keys…, Mirroring horizontally twice should return original., Single cell should remain unchanged when mirrored vertically., Mirror a 1x2 vertical line., Mirroring vertically twice should return original., Mirroring should preserve module IDs., Test pattern mirroring logic. (+4 more)

### Community 14 - "_scan_grid_with_window"
Cohesion: 0.09
Nodes (17): Scans a grid with a fixed window size to find the best placement opportunity.…, _scan_grid_with_window(), Should check all valid window positions, Returned position should be (start_x, start_y) of window, Inactive cells should count as unavailable for placement, Windows without enough empty slots for module_count should be skipped, Adversarial tests for _scan_grid_with_window, Helper to create a test grid (+9 more)

### Community 15 - "calculate_grid_score"
Cohesion: 0.07
Nodes (29): Enum, RustGrid, AdjacencyType, calculate_grid_score(), calculate_score_delta(), clear_scores(), _get_orthogonal_neighbors(), ModuleType (+21 more)

### Community 16 - "ModulePlacementCNN"
Cohesion: 0.12
Nodes (13): PlacementDataset, Initializes and trains a ModulePlacementCNN model with validation and early…, train_model(), get_model(), _load_model_from_disk(), This module provides a caching mechanism for loading PyTorch models. It uses an…, Internal function to load a model from disk. This function is decorated with…, Retrieves a model, utilizing an LRU cache to avoid reloading from disk. (+5 more)

### Community 17 - "TestRefinePlacement"
Cohesion: 0.12
Nodes (13): Exact fit: modules == empty positions, More empty positions than modules, Adversarial tests for refine_placement function, Inactive cells should not be used for placement, Helper to create a test grid, Progress callback should be called during permutation, Function should complete and log iteration count, Should clear tech modules before trying each permutation (+5 more)

### Community 18 - "calculate_window_score"
Cohesion: 0.15
Nodes (13): calculate_window_score(), Calculates a quality score for a placement window. Evaluates window…, Adversarial tests for calculate_window_score, Helper to create a window grid, Window with only supercharged empty cells, Window with empty cells but no supercharged, Window with no empty cells, Window with mix of empty and occupied cells (+5 more)

### Community 19 - "TestClearAllModulesOfTech"
Cohesion: 0.09
Nodes (12): Test clearing all modules of a specific technology., Set up a grid with mixed tech modules., Clearing pulse should remove only pulse modules., Clearing engineering should remove only engineering modules., Clearing should reset all module-related properties., Clearing should not affect grid structure., Clearing nonexistent tech should not crash., Clearing from empty grid should not crash. (+4 more)

### Community 20 - "find_supercharged_opportunities"
Cohesion: 0.12
Nodes (12): find_supercharged_opportunities(), Finds the highest-scoring window with available supercharged slots. Dynamically…, Adversarial tests for find_supercharged_opportunities, No supercharged slots should return None, All supercharged slots occupied should return None, Available supercharged slots should return a window, Should return (x, y, width, height), No modules defined should return None (+4 more)

### Community 21 - "TestPlaceModule"
Cohesion: 0.10
Nodes (11): Should handle various bonus values correctly., Should handle None image gracefully., Should handle string image paths., Test module placement on grid cells., Set up a clean grid for each test., Placing a module should set all required properties., Module position should be set correctly., Placing a module should overwrite the previous one. (+3 more)

### Community 22 - "calculate_pattern_adjacency_score"
Cohesion: 0.11
Nodes (14): calculate_pattern_adjacency_score(), Calculates a heuristic adjacency score for a placed pattern. This score is not…, Test that lesser_2 module score scales correctly with number of matching…, Tests for adjacency-based module placement logic, Test that different rule values in the same family (e.g. greater_3 and…, Grid with no modules of tech should score 0, Modules of different tech should be ignored in scoring, Center location with no neighbors should score 0 adjacency (+6 more)

### Community 23 - "._create_grid"
Cohesion: 0.12
Nodes (12): Adversarial tests for create_localized_grid, Helper to create a test grid, Localized grid at (0, 0) should extract top-left corner, Localized grid at offset should extract correct region, Localized grid extending beyond grid boundary should be clamped, Localized grid at bottom-right corner, Localized grid should copy module data correctly, Localized grid should preserve supercharged status (+4 more)

### Community 24 - "create_localized_grid_ml"
Cohesion: 0.11
Nodes (15): create_localized_grid_ml(), Grid, Creates a localized grid for ML-based refinement. Extracts a rectangular…, Adversarial tests for optimization/windowing.py Focus areas: - Window scanning…, Adversarial tests for create_localized_grid_ml, Helper to create a test grid, Modules of target tech should be preserved, Modules of other tech should be removed (+7 more)

### Community 25 - "TestPatternApplicationToGrid"
Cohesion: 0.10
Nodes (11): Test applying patterns to grids with various edge cases., Set up common test fixtures., Apply a simple pattern to an empty grid., Pattern that goes off-grid should return None., Pattern with negative offset should fail gracefully., Pattern with modules player doesn't own should return None., Applying a pattern shouldn't remove other tech modules., Pattern placement on inactive cells should fail. (+3 more)

### Community 26 - "test_analytics.py"
Cohesion: 0.13
Nodes (13): Any, GA4 server-side event tracking via Google Analytics Measurement Protocol. This…, Send an analytics event to GA4. Args: event_name: Name of the event (e.g.,…, Send multiple analytics events in a batch to GA4. Args: events: List of…, Convert event to Measurement Protocol format. Note: Does not mutate self.…, send_analytics_batch(), send_analytics_event(), Tests for GA4 analytics module. (+5 more)

### Community 27 - "patch"
Cohesion: 0.11
Nodes (10): patch, Test successful event send., Test send_event returns False when disabled., Test send_event returns False for invalid event name., Test send_event includes user_id in payload., Test send_event generates client_id if not provided., Test send_event handles timeout exception., Test successful batch event send. (+2 more)

### Community 28 - "TestAvailableModulesPatternMatching"
Cohesion: 0.11
Nodes (10): When available_modules has more than pattern, use windowed SA, Tests for the logic that detects available modules vs pattern modules, When available_modules differ (not superset), use windowed SA, When available_modules is None, use all tech_modules from filter, Pattern with 'None' values should not be included in module count, Pattern with dict format modules (with 'id' key) should be parsed correctly, When available_modules is empty list, should return error tuple, When available is subset of pattern, use windowed SA (+2 more)

### Community 29 - "TestPlaceAllModulesInEmptySlots"
Cohesion: 0.14
Nodes (11): Adversarial tests for place_all_modules_in_empty_slots, Helper to create a test grid, With no modules, grid should be returned unchanged, All modules should be placed in completely empty grid, Only as many modules as empty slots should be placed, If no cells are active, no modules should be placed, Modules should only be placed in active cells, Existing modules should not be overwritten (+3 more)

### Community 30 - "rotate_pattern"
Cohesion: 0.10
Nodes (15): Rotates a pattern 90 degrees clockwise. Args: pattern (dict): A dictionary…, rotate_pattern(), Test pattern rotation logic for off-by-one errors and coordinate issues., Single cell patterns should remain unchanged after rotation., Empty patterns should stay empty., Rotate a 2x1 horizontal line 90 degrees clockwise., Test edge cases and error conditions., Patterns with negative coordinates after rotation. (+7 more)

### Community 31 - "TestGA4Client"
Cohesion: 0.11
Nodes (10): Test send_event returns False when params is not a dict., Test _validate_event rejects invalid event name format., Test _validate_event rejects empty event name., Test _validate_event rejects non-string event name., Test _validate_event rejects non-dict params., Test GA4Client class., Test client initialization with environment variables., Test client is disabled when measurement_id is missing. (+2 more)

### Community 32 - "Implementation Plan - Improve Group Adjacency Weights"
Cohesion: 0.15
Nodes (12): Automated Tests, Core Logic, For `greater_N` rules:, For `lesser_N` rules:, Goal Description, Implementation Plan - Improve Group Adjacency Weights, Manual Verification, [MODIFY] `src/pattern_matching.py` (+4 more)

### Community 33 - "app.py"
Cohesion: 0.05
Nodes (35): after_request, date, on, initialize_clients(), Initializes the Google Analytics and BigQuery clients. It attempts to load…, add_cache_headers(), get_ship_types(), get_technology_tree() (+27 more)

### Community 34 - "GA4Client"
Cohesion: 0.14
Nodes (9): GA4Client, Send multiple events in batch to GA4. Args: events: List of AnalyticsEvent…, Validate event before sending. Args: event: Event to validate Returns: True if…, Client for sending events to GA4 via the Measurement Protocol., Initialize GA4 client. Args: measurement_id: GA4 Measurement ID (usually starts…, Send a single event to GA4. Args: event: AnalyticsEvent to send Returns: True…, Test send_event returns False on HTTP error., Test _validate_event with valid event. (+1 more)

### Community 35 - "TestPatternAdjacencyScoring"
Cohesion: 0.12
Nodes (9): Test the adjacency score calculation logic., Set up test grid with modules., Empty grid should have zero adjacency score., Module at corner should get edge bonuses., Module on edge should get one edge bonus., Center module should not get edge bonuses., Two adjacent modules of same tech should get adjacency bonus., Score calculation should only count modules of the specified tech. (+1 more)

### Community 36 - "TestOptimizationEndpoint"
Cohesion: 0.13
Nodes (8): Successful response should contain grid data., Test optimization endpoint., Optimization should require POST method., Request without ship should fail., Request without tech should fail., Invalid JSON should be rejected., Response should be valid JSON., TestOptimizationEndpoint

### Community 37 - "TestPlacementAlgorithmLogic"
Cohesion: 0.22
Nodes (5): Tests for the logic of the placement algorithm itself, Simulating the placement algorithm to verify best position selection, Multi-module placement should consider already-placed modules, Algorithm should skip inactive cells, TestPlacementAlgorithmLogic

### Community 38 - "AnalyticsEvent"
Cohesion: 0.20
Nodes (9): AnalyticsEvent, Represents a GA4 event to be sent to the Measurement Protocol., Test AnalyticsEvent dataclass., Test creating event with minimal parameters., Test creating event with all parameters., Test conversion to measurement protocol format., Test conversion includes timestamp when provided., Test that conversion doesn't mutate the original event. (+1 more)

### Community 39 - "TestGetTrainingModuleIds"
Cohesion: 0.14
Nodes (8): Test training module ID retrieval., Should return a list., Nonexistent ship should return empty list., Nonexistent tech should return empty list., All module IDs should be strings., Module IDs should be unique (no duplicates)., If ship/tech combo exists, should have modules., TestGetTrainingModuleIds

### Community 40 - "TestPartialModuleSetEdgeCases"
Cohesion: 0.14
Nodes (8): Adversarial tests for partial module set handling, Empty available_modules list should be treated as partial set, Single available module from full set should be partial, When available_modules is None, should not be partial, When all modules available, should not be partial, Should correctly calculate missing modules, With multiple missing modules (not just PC), should remain partial, TestPartialModuleSetEdgeCases

### Community 41 - "get_all_unique_pattern_variations"
Cohesion: 0.15
Nodes (10): get_all_unique_pattern_variations(), Generates all unique variations of a pattern (rotations and mirrors). This…, Test unique pattern variation generation., Single cell pattern should have only one unique variation., 2x1 line should have multiple unique variations., Symmetric square pattern should have fewer unique variations., Variations should always include the original pattern., All variations should have the same number of modules. (+2 more)

### Community 42 - "TestFilterSolves"
Cohesion: 0.12
Nodes (9): If no modules found for tech, should still return solve with all modules…, Test the filter_solves function., Empty solve map should be handled gracefully., Set up common test fixtures., Filtering should return a new dict, not modify original., Should handle available_modules parameter when None., Should handle large solve patterns., Filtered solve should include modules the player owns. (+1 more)

### Community 43 - "TestWindowSizeValidation"
Cohesion: 0.16
Nodes (9): slow, Compare average optimized scores across different window sizes. Args: ship:…, Test that for key module counts, the chosen window size produces higher…, Quick structural validation that window sizes can hold modules. This is a…, Empirically validate window dimensions via optimized score comparison, Load all module data once for all tests, Generate random supercharged placements for testing. Args: width: Grid width…, Run SA optimization on a grid and return the final score. Args: width: Grid… (+1 more)

### Community 44 - "TestRequestValidation"
Cohesion: 0.15
Nodes (7): Test request parameter validation., Invalid ship type should be rejected or handled., Invalid tech type should be rejected or handled., Empty rewards list should be accepted., Invalid reward format should be handled., Negative seed should be handled., TestRequestValidation

### Community 45 - "TestGetModuleData"
Cohesion: 0.17
Nodes (7): Test module data loading and caching., Module data should return a dictionary., Nonexistent ship should return empty dict., Repeated calls should return cached data., Module data should have expected structure., Different ships should have different data., TestGetModuleData

### Community 46 - "TestGetSolveMap"
Cohesion: 0.17
Nodes (7): Test solve map loading and caching., Solve map should return a dictionary., Nonexistent solve should return empty dict., Repeated calls should return cached data., Solve map entries should have 'map' and 'score' keys., Solve map should convert string keys to tuples., TestGetSolveMap

### Community 47 - "TestMLPlacementOutputValidation"
Cohesion: 0.17
Nodes (7): Test output validation and correctness., Set up test fixtures., Output should be a tuple of (grid, score)., Output grid should be a Grid instance or None., Output score should be a float., Score should never be negative., TestMLPlacementOutputValidation

### Community 48 - "TestWindowScoringAndSelection"
Cohesion: 0.17
Nodes (7): Adversarial tests for window scoring logic, Pattern should be selected when score strictly greater, Scanned should be selected when strictly greater, When pattern score is negative, valid scanned should win, When both scores negative, less negative (pattern) wins on tie logic, Scores very close to zero should be handled correctly, TestWindowScoringAndSelection

### Community 49 - "TestRefinementStageEdgeCases"
Cohesion: 0.17
Nodes (7): Adversarial tests for refinement stage, When neither pattern nor scanned opportunity exists, no refinement, Pattern should be fallback when scanned calculation fails, With equal scores, pattern location is preferred, Refinement should be applied if refined_score >= current_score, When refinement returns None, original should be kept, TestRefinementStageEdgeCases

### Community 50 - "TestFilterSolvesEdgeCases"
Cohesion: 0.20
Nodes (6): Test edge cases and error conditions., Empty solves dict should return empty result., If solve_data is None or falsy, should return empty dict., Pattern with repeated module IDs should be preserved., Solve data without 'map' key should handle gracefully., TestFilterSolvesEdgeCases

### Community 51 - "TestErrorHandling"
Cohesion: 0.18
Nodes (6): Test error handling in endpoints., Nonexistent endpoint should return 404., Malformed request body should be rejected., Empty request body should be rejected., Null request body should be rejected., TestErrorHandling

### Community 52 - "TestMLPlacementModelLoading"
Cohesion: 0.14
Nodes (10): patch, Test model loading and validation., Set up test fixtures., Test handling of empty or minimal results., Set up test fixtures., When no placeable modules found, should return cleared grid., Nonexistent model file should return None., When no training module IDs found, should return None. (+2 more)

### Community 53 - "test_pattern_matching.py"
Cohesion: 0.17
Nodes (9): _extract_pattern_from_grid(), Extracts a normalized pattern (relative coordinates) from a grid for a specific…, Comprehensive test suite for pattern_matching.py This test suite focuses on…, Test extracting patterns from grids., Extract a single module should give normalized (0,0) coordinate., Extract multiple modules should normalize to min coordinates., Extract from empty grid should return empty pattern., Extract should only include specified tech. (+1 more)

### Community 54 - "TestFilterSolvesPhotonixOverride"
Cohesion: 0.18
Nodes (7): patch, Test the photonix override behavior for PC platform., Set up test fixtures with photonix data., When tech is pulse and PC in available_modules, should use photonix., Without PC in available_modules, should not override., Photonix override should only apply to pulse tech., TestFilterSolvesPhotonixOverride

### Community 55 - "format_moduleDefs.py"
Cohesion: 0.27
Nodes (9): custom_json_dump(), find_and_process_modules(), process_module(), This script provides utility functions to process and reformat module…, Main function to read, process, and rewrite all module JSON files. This…, Applies formatting and type changes to a single module dictionary. This…, Recursively finds and processes 'modules' lists within a JSON structure. This…, Writes a dictionary to a file with custom JSON formatting. This function… (+1 more)

### Community 56 - "filter_solves"
Cohesion: 0.20
Nodes (6): filter_solves(), Filters a solve map to only include modules the player owns. This function…, Comprehensive test suite for solve_map_utils.py This test suite focuses on…, Filtered solve should include None (empty) slots., Filtering different techs from same ship should work correctly., Basic filtering should include owned modules and None slots.

### Community 57 - "TestAppInitialization"
Cohesion: 0.20
Nodes (6): Test Flask app initialization and basic setup., Flask app should be created., App should be a Flask instance., CORS should be enabled., Compression should be enabled., TestAppInitialization

### Community 58 - "TestGridUtils"
Cohesion: 0.20
Nodes (6): Set up a common grid for testing., Test that the Grid is initialized with the correct dimensions and empty cells., Test that the Grid can be correctly serialized to and deserialized from a…, Test that the copy method creates a deep copy of the grid., Test suite for the Grid class and related utility functions in grid_utils.py., TestGridUtils

### Community 59 - "TestMLPlacementModuleAssignment"
Cohesion: 0.20
Nodes (6): Test module assignment from ML predictions., Set up test fixtures., Modules should only be placed on active cells., Should place exactly the required number of modules., Shouldn't place multiple modules in same cell., TestMLPlacementModuleAssignment

### Community 60 - "TestMLPlacementPolishing"
Cohesion: 0.20
Nodes (6): Test SA polishing behavior., Set up test fixtures., With polish_result=True, should attempt SA polishing., With polish_result=False, should skip SA polishing., Polishing should not decrease the score., TestMLPlacementPolishing

### Community 61 - "TestMLPlacementGridHandling"
Cohesion: 0.20
Nodes (6): Test grid handling and state management., Set up test fixtures., ML placement should not modify the input grid., Should handle localized grid coordinates correctly., Output grid should preserve supercharge state from input., TestMLPlacementGridHandling

### Community 62 - "TestMLPlacementErrorHandling"
Cohesion: 0.20
Nodes (6): Test error handling and edge cases., Set up test fixtures., Should handle empty grid with no active cells., Should handle grid where all cells are supercharged., Should handle model prediction errors gracefully., TestMLPlacementErrorHandling

### Community 63 - "TestOldDetermineWindowDimensionsBehavior"
Cohesion: 0.03
Nodes (36): Old Logic: standard cargo_scanner (1 modules), Old Logic: pilgrim thermal (1 modules), Old Logic: pilgrim toxic (1 modules), Old Logic: pilgrim cyclops (1 modules), Old Logic: pilgrim radar (1 modules), Old Logic: nautilon humboldt (5 modules), Old Logic: exosuit jetpack (8 modules), Old Logic: atlantid fishing (1 modules) (+28 more)

### Community 64 - "TestSuperchargedWindowDetection"
Cohesion: 0.20
Nodes (6): Adversarial tests for supercharged slot detection in windows, Should respect grid boundaries when checking window cells, Inactive cells should be excluded even if supercharged, Supercharged cells with modules should not count as available, Window with no supercharged cells should return empty, TestSuperchargedWindowDetection

### Community 65 - "TestScEligibleWindowSizeWithConstraint"
Cohesion: 0.33
Nodes (4): Tests for window sizing with sc_eligible constraints., Verify that window size selection considers sc_eligible. With all non-eligible…, Test that both original and rotated window dimensions are checked correctly…, TestScEligibleWindowSizeWithConstraint

### Community 66 - "determine_window_dimensions"
Cohesion: 0.03
Nodes (37): get_tech_window_rules(), _get_window_profiles(), Retrieves the merged window rules dictionary for a specific technology. Loads…, determine_window_dimensions(), Old Logic: solar cargo_scanner (1 modules), Old Logic: pilgrim fusion (4 modules), Old Logic: pilgrim cold (1 modules), Old Logic: pilgrim radiation (1 modules) (+29 more)

### Community 67 - "TestTechTreeEndpoint"
Cohesion: 0.22
Nodes (5): Test technology tree endpoint., Tech tree should require POST method., Request without ship should fail., Response should be valid JSON., TestTechTreeEndpoint

### Community 69 - "TestGetAllSolveData"
Cohesion: 0.33
Nodes (4): Test getting all solve data at once., Should return a dictionary., Solve data should have valid structure., TestGetAllSolveData

### Community 70 - "TestGetAllModuleData"
Cohesion: 0.25
Nodes (5): Test getting all module data at once., Should return a dictionary., Should return non-empty dict if data files exist., Should have multiple ship types if data exists., TestGetAllModuleData

### Community 71 - "TestDataLoaderErrorHandling"
Cohesion: 0.25
Nodes (5): Test error handling in data loader., Corrupt JSON file should be handled gracefully., Solve map with missing 'map' or 'score' should be skipped., Cache should not grow unbounded., TestDataLoaderErrorHandling

### Community 72 - "TestMLPlacementTensorPreparation"
Cohesion: 0.25
Nodes (5): Input tensor should have correct shape., Supercharge flags should be correctly set in tensor., Test input tensor preparation., Set up test fixtures., TestMLPlacementTensorPreparation

### Community 73 - "TestMLPlacementProgressCallback"
Cohesion: 0.25
Nodes (5): Test progress callback functionality., Set up test fixtures., Should call progress callback when provided and send_grid_updates=True., Should not crash when progress_callback is None., TestMLPlacementProgressCallback

### Community 74 - "TestMLPlacementIntegration"
Cohesion: 0.25
Nodes (5): Integration tests for ml_placement., Set up test fixtures., Should handle all optional parameters., Calling twice with same input should give same output., TestMLPlacementIntegration

### Community 75 - "TestInitialPlacementFallback"
Cohesion: 0.25
Nodes (5): Tests for initial placement fallback logic when no solve exists, When no solve exists and bonus is zero, percentage should be zero, When no solve exists and bonus is positive, percentage should be 100, When no modules found, grid should be cleared of target tech, TestInitialPlacementFallback

### Community 76 - "TestDetermineWindowDimensions"
Cohesion: 0.25
Nodes (5): Adversarial tests for determine_window_dimensions, Corvette pulse with 6 modules should NOT match the 7-module override, Corvette with 8 modules should return 3x3, Bolt-caster falls back to standard now so it scales with count, TestDetermineWindowDimensions

### Community 78 - "analyze_benchmarks.py"
Cohesion: 0.33
Nodes (6): analyze_benchmarks(), parse_benchmark_log(), print_results(), Analyzes all benchmark logs in a directory., Prints the analysis results in a formatted table., Parses a single benchmark log file.

### Community 80 - "test_app_endpoints.py"
Cohesion: 0.29
Nodes (4): Comprehensive test suite for Flask app endpoints This test suite focuses on…, Test health/status endpoint., Health endpoint should exist and return 200., TestHealthEndpoint

### Community 81 - "TestAnalyticsEndpoint"
Cohesion: 0.29
Nodes (4): Test analytics/popular data endpoint., Analytics endpoint should exist., Analytics response should be valid JSON if successful., TestAnalyticsEndpoint

### Community 82 - "TestResponseFormat"
Cohesion: 0.29
Nodes (4): Test response format consistency., Error responses should contain an error message., Response headers should be valid., TestResponseFormat

### Community 83 - "TestCORSHeaders"
Cohesion: 0.29
Nodes (4): Test CORS header handling., Successful responses should have CORS headers., OPTIONS preflight requests should be handled., TestCORSHeaders

### Community 84 - "TestContentNegotiation"
Cohesion: 0.29
Nodes (4): Test content type handling., application/json content type should be accepted., Wrong content type might be rejected., TestContentNegotiation

### Community 85 - "TestEndpointIntegration"
Cohesion: 0.29
Nodes (4): Integration tests for endpoints., Should handle multiple requests in sequence., Requests should not interfere with each other., TestEndpointIntegration

### Community 87 - "generate_old_logic_tests.py"
Cohesion: 0.40
Nodes (3): generate_old_logic_test_cases(), load_module_data(), MockModulesUtils

### Community 88 - "TestEventNameValidation"
Cohesion: 0.33
Nodes (4): Test event name pattern validation., Test that valid event names match pattern., Test that invalid event names don't match pattern., TestEventNameValidation

### Community 89 - "TestPerformanceAnalyticsEndpoint"
Cohesion: 0.33
Nodes (4): patch, Test performance analytics data endpoint., Test performance analytics response includes p50, p75, and p90., TestPerformanceAnalyticsEndpoint

### Community 90 - "TestDataIntegrity"
Cohesion: 0.33
Nodes (4): Test data integrity across operations., Getting module data shouldn't modify it., Tuple conversion should be consistent., TestDataIntegrity

### Community 91 - "optimize_placement"
Cohesion: 0.05
Nodes (30): optimize_placement(), _prepare_optimization_run(), Grid, Optimizes module placement for a specific technology on a ship's grid. This is…, Prepares the optimization run by validating the grid and retrieving available…, Test optimizing multiple technologies., Set up common test resources., Get list of available tech keys for the ship. (+22 more)

### Community 92 - "apply_labels.py"
Cohesion: 0.40
Nodes (4): process_file(), Replace label if it doesn't already have a bracket prefix, Process a single JSON file using regex, replace_label()

### Community 94 - "logger.py"
Cohesion: 0.50
Nodes (3): This module provides a centralized logging setup for the application. It…, Sets up the root logger for the application, but ONLY if no handlers are…, setup_logger()

### Community 95 - "optimizer.py"
Cohesion: 0.33
Nodes (6): get_tech_modules_for_training(), get_tech_tree(), get_tech_tree_json(), Generates a technology tree and returns it as a JSON string. This function…, Generates a structured technology tree for a given ship. The tree is organized…, Retrieves all modules for a technology without filtering. This function is used…

### Community 97 - "debug_window_dimensions.py"
Cohesion: 1.00
Nodes (3): determine_window_dimensions_diagnostic(), get_tech_window_rules_diagnostic(), _get_window_profiles_diagnostic()

### Community 98 - "NMS Optimizer Service README"
Cohesion: 0.67
Nodes (3): NMS Optimizer Service README, Service Dependencies, Training Dependencies

### Community 124 - "Changes Made"
Cohesion: 0.25
Nodes (7): Automated Tests, Changes Made, Core Logic, Project Configuration, Validation Results, Verification & Testing, Walkthrough - Improve Group Adjacency Weights

### Community 139 - "get_module_data"
Cohesion: 0.03
Nodes (36): get_module_data(), Loads the module data for a specific ship type from its JSON file. Results are…, Old Logic: nautilon tethys (1 modules), Old Logic: nautilon nautilon (4 modules), Old Logic: nautilon cyclops (1 modules), Old Logic: living saline (1 modules), Old Logic: living trails (12 modules), Old Logic: exosuit aeration (4 modules) (+28 more)

## Knowledge Gaps
- **50 isolated node(s):** `generate_solves.sh script`, `name`, `version`, `description`, `main` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **227 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Grid` connect `Grid` to `place_module`, `TestOptimizationPipeline`, `clear_all_modules_of_tech`, `TestScEligibleAdversarial`, `data_loader.py`, `TestOptimization`, `TestScEligibleEdgeCases`, `TestOptimizeOptimizationFlow`, `check_all_modules_placed`, `mirror_pattern_horizontally`, `_scan_grid_with_window`, `calculate_grid_score`, `TestRefinePlacement`, `calculate_window_score`, `TestClearAllModulesOfTech`, `find_supercharged_opportunities`, `TestPlaceModule`, `calculate_pattern_adjacency_score`, `._create_grid`, `create_localized_grid_ml`, `TestPatternApplicationToGrid`, `TestAvailableModulesPatternMatching`, `TestPlaceAllModulesInEmptySlots`, `rotate_pattern`, `app.py`, `TestPatternAdjacencyScoring`, `TestPlacementAlgorithmLogic`, `TestPartialModuleSetEdgeCases`, `get_all_unique_pattern_variations`, `TestWindowSizeValidation`, `TestMLPlacementOutputValidation`, `TestWindowScoringAndSelection`, `TestRefinementStageEdgeCases`, `TestMLPlacementModelLoading`, `test_pattern_matching.py`, `TestGridUtils`, `TestMLPlacementModuleAssignment`, `TestMLPlacementPolishing`, `TestMLPlacementGridHandling`, `TestMLPlacementErrorHandling`, `TestSuperchargedWindowDetection`, `TestScEligibleWindowSizeWithConstraint`, `TestMLPlacementTensorPreparation`, `TestMLPlacementProgressCallback`, `TestMLPlacementIntegration`, `TestInitialPlacementFallback`, `TestDetermineWindowDimensions`, `optimize_placement`, `.get_cell`, `optimizer.py`?**
  _High betweenness centrality (0.362) - this node is a cross-community bridge._
- **Why does `determine_window_dimensions()` connect `determine_window_dimensions` to `place_module`, `data_loader.py`, `find_supercharged_opportunities`, `TestWindowSizeValidation`, `TestOldDetermineWindowDimensionsBehavior`, `.test_old_logic_case_238_staves_cloaking_count_1`, `TestDetermineWindowDimensions`, `.test_old_logic_case_176_pilgrim_mounted_count_4`, `.test_old_logic_case_226_staves_analysis_count_1`, `.test_old_logic_case_094_nomad_boost_count_4`, `optimize_placement`, `.test_old_logic_case_095_nomad_slide_count_1`, `.test_old_logic_case_164_solar_conflict_scanner_count_1`, `.test_old_logic_case_165_solar_economy_scanner_count_1`, `.test_old_logic_case_105_nomad_cyclops_count_1`, `.test_old_logic_case_167_solar_trails_count_12`, `.test_old_logic_case_168_solar_teleporter_count_1`, `.test_old_logic_case_170_pilgrim_icarus_count_1`, `.test_old_logic_case_172_pilgrim_slide_count_1`, `.test_old_logic_case_173_pilgrim_grip_count_1`, `.test_old_logic_case_174_pilgrim_drift_count_1`, `.test_old_logic_case_175_pilgrim_mining_count_5`, `.test_old_logic_case_015_standard_trails_count_12`, `.test_old_logic_case_177_pilgrim_flamethrower_count_1`, `.test_old_logic_case_184_pilgrim_amplifier_count_1`, `.test_old_logic_case_185_pilgrim_power_count_1`, `.test_old_logic_case_016_standard_teleporter_count_1`, `.test_old_logic_case_187_nautilon_icarus_count_1`, `.test_old_logic_case_188_nautilon_dredging_count_1`, `get_module_data`, `.test_old_logic_case_192_nautilon_sonar_count_1`, `.test_old_logic_case_193_living_grafted_count_4`, `.test_old_logic_case_194_living_spewing_count_4`, `.test_old_logic_case_195_living_scream_count_4`, `.test_old_logic_case_196_living_assembly_count_5`, `.test_old_logic_case_198_living_pulsing_count_4`, `.test_old_logic_case_200_living_bobble_count_4`, `.test_old_logic_case_201_living_scanners_count_2`, `.test_old_logic_case_204_exosuit_refiner_count_1`, `.test_old_logic_case_205_exosuit_life_count_6`, `.test_old_logic_case_018_atlantid_analysis_count_1`, `.test_old_logic_case_207_exosuit_anomaly_count_1`, `.test_old_logic_case_208_exosuit_hazard_count_1`, `.test_old_logic_case_209_exosuit_pressure_count_1`, `.test_old_logic_case_211_exosuit_radiation_count_4`, `.test_old_logic_case_213_exosuit_thermic_count_4`, `.test_old_logic_case_214_exosuit_toxin_count_4`, `.test_old_logic_case_216_exosuit_defense_count_3`, `.test_old_logic_case_217_exosuit_rebuilt_count_3`, `.test_old_logic_case_218_exosuit_forbidden_count_3`, `.test_old_logic_case_220_exosuit_hazmat_count_1`, `.test_old_logic_case_221_exosuit_nutrient_count_1`, `.test_old_logic_case_222_exosuit_skiff_count_1`, `.test_old_logic_case_223_exosuit_trade_count_1`, `.test_old_logic_case_224_exosuit_exocraft_count_1`, `.test_old_logic_case_225_staves_mining_count_6`, `.test_old_logic_case_227_staves_fishing_count_1`, `.test_old_logic_case_228_staves_gravatino_count_1`, `.test_old_logic_case_229_staves_scanner_count_7`, `.test_old_logic_case_230_staves_survey_count_1`, `.test_old_logic_case_232_staves_bolt_caster_count_9`, `.test_old_logic_case_234_staves_neutron_count_5`, `.test_old_logic_case_235_staves_plasma_launcher_count_4`, `.test_old_logic_case_021_atlantid_scanner_count_7`, `.test_old_logic_case_237_staves_scatter_count_5`, `.test_old_logic_case_171_pilgrim_boost_count_4`, `.test_old_logic_case_239_staves_combat_count_1`, `.test_old_logic_case_240_staves_voltaic_amplifier_count_1`, `.test_old_logic_case_242_staves_personal_count_1`, `.test_old_logic_case_243_staves_terrian_count_1`, `.test_old_logic_case_246_sentinel_mt_fishing_count_1`, `.test_old_logic_case_247_sentinel_mt_gravatino_count_1`, `.test_old_logic_case_249_sentinel_mt_survey_count_1`, `.test_old_logic_case_253_sentinel_mt_neutron_count_5`, `.test_old_logic_case_254_sentinel_mt_plasma_launcher_count_4`, `.test_old_logic_case_023_atlantid_blaze_javelin_count_6`, `.test_old_logic_case_258_sentinel_mt_combat_count_1`, `.test_old_logic_case_259_sentinel_mt_voltaic_amplifier_count_1`, `.test_old_logic_case_260_sentinel_mt_paralysis_count_1`, `.test_old_logic_case_262_sentinel_mt_terrian_count_1`, `.test_old_logic_case_265_roamer_boost_count_4`, `.test_old_logic_case_266_roamer_slide_count_1`, `.test_old_logic_case_024_atlantid_bolt_caster_count_9`, `.test_old_logic_case_268_roamer_drift_count_1`, `.test_old_logic_case_270_roamer_mounted_count_4`, `.test_old_logic_case_271_roamer_flamethrower_count_1`, `.test_old_logic_case_275_roamer_toxic_count_1`, `.test_old_logic_case_276_roamer_cyclops_count_1`, `.test_old_logic_case_025_atlantid_geology_count_4`, `.test_old_logic_case_000_standard_cyclotron_count_5`, `.test_old_logic_case_027_atlantid_plasma_launcher_count_4`, `.test_old_logic_case_028_atlantid_pulse_spitter_count_7`, `.test_old_logic_case_251_sentinel_mt_bolt_caster_count_9`, `.test_old_logic_case_030_atlantid_cloaking_count_1`, `.test_old_logic_case_034_atlantid_personal_count_1`, `.test_old_logic_case_035_atlantid_terrian_count_1`, `.test_old_logic_case_036_corvette_cyclotron_count_5`, `.test_old_logic_case_001_standard_infra_count_5`, `.test_old_logic_case_038_corvette_phase_count_5`, `.test_old_logic_case_039_corvette_photon_count_5`, `.test_old_logic_case_043_corvette_hyper_count_9`, `.test_old_logic_case_045_corvette_launch_count_6`, `.test_old_logic_case_046_corvette_pulse_count_8`, `.test_old_logic_case_002_standard_phase_count_5`, `.test_old_logic_case_047_corvette_habitation_count_3`, `.test_old_logic_case_049_corvette_bobble_count_4`, `.test_old_logic_case_050_corvette_conflict_scanner_count_1`, `.test_old_logic_case_052_corvette_cargo_scanner_count_1`, `.test_old_logic_case_055_standard_mt_mining_count_6`, `.test_old_logic_case_056_standard_mt_analysis_count_1`, `.test_old_logic_case_003_standard_photon_count_5`, `.test_old_logic_case_057_standard_mt_fishing_count_1`, `.test_old_logic_case_059_standard_mt_scanner_count_7`, `.test_old_logic_case_060_standard_mt_survey_count_1`, `.test_old_logic_case_061_standard_mt_blaze_javelin_count_6`, `.test_old_logic_case_062_standard_mt_bolt_caster_count_9`, `.test_old_logic_case_064_standard_mt_neutron_count_5`, `.test_old_logic_case_065_standard_mt_plasma_launcher_count_4`, `.test_old_logic_case_066_standard_mt_pulse_spitter_count_7`, `.test_old_logic_case_004_standard_positron_count_5`, `.test_old_logic_case_071_standard_mt_paralysis_count_1`, `.test_old_logic_case_072_standard_mt_personal_count_1`, `.test_old_logic_case_074_sentinel_cyclotron_count_5`, `.test_old_logic_case_075_sentinel_infra_count_5`, `.test_old_logic_case_079_sentinel_photon_count_5`, `.test_old_logic_case_080_sentinel_shield_count_5`, `.test_old_logic_case_081_sentinel_launch_count_6`, `.test_old_logic_case_083_sentinel_pulse_count_8`, `.test_old_logic_case_086_sentinel_pilot_count_1`, `.test_old_logic_case_006_standard_shield_count_5`, `.test_old_logic_case_087_sentinel_conflict_scanner_count_1`, `.test_old_logic_case_088_sentinel_economy_scanner_count_1`, `.test_old_logic_case_089_sentinel_cargo_scanner_count_1`, `.test_old_logic_case_090_sentinel_trails_count_12`, `.test_old_logic_case_092_nomad_fusion_count_4`, `.test_old_logic_case_040_corvette_positron_count_5`, `.test_old_logic_case_096_nomad_grip_count_1`, `.test_old_logic_case_098_nomad_mining_count_5`, `.test_old_logic_case_100_nomad_flamethrower_count_1`, `.test_old_logic_case_101_nomad_thermal_count_1`, `.test_old_logic_case_102_nomad_cold_count_1`, `.test_old_logic_case_103_nomad_radiation_count_1`, `.test_old_logic_case_106_nomad_radar_count_1`, `.test_old_logic_case_107_nomad_amplifier_count_1`, `.test_old_logic_case_108_nomad_power_count_1`, `.test_old_logic_case_111_minotaur_icarus_count_1`, `.test_old_logic_case_112_minotaur_minotaur_laser_count_6`, `.test_old_logic_case_113_minotaur_minotaur_count_4`, `.test_old_logic_case_114_minotaur_hardframe_right_count_1`, `.test_old_logic_case_115_minotaur_liquidator_right_count_4`, `.test_old_logic_case_117_minotaur_environment_count_1`, `.test_old_logic_case_118_minotaur_cyclops_count_1`, `.test_old_logic_case_119_minotaur_array_count_1`, `.test_old_logic_case_120_minotaur_ai_count_1`, `.test_old_logic_case_121_minotaur_bore_count_1`, `.test_old_logic_case_123_minotaur_liquidator_body_count_1`, `.test_old_logic_case_124_freighter_hyper_count_11`, `.test_old_logic_case_125_freighter_interstellar_count_1`, `.test_old_logic_case_010_standard_aqua_count_1`, `.test_old_logic_case_127_freighter_fleet_fuel_count_3`, `.test_old_logic_case_128_freighter_fleet_speed_count_3`, `.test_old_logic_case_129_freighter_fleet_combat_count_3`, `.test_old_logic_case_130_freighter_fleet_exploration_count_3`, `.test_old_logic_case_131_freighter_fleet_mining_count_3`, `.test_old_logic_case_134_colossus_icarus_count_1`, `.test_old_logic_case_138_colossus_drift_count_1`, `.test_old_logic_case_139_colossus_mining_count_5`, `.test_old_logic_case_141_colossus_flamethrower_count_1`, `.test_old_logic_case_142_colossus_thermal_count_1`, `.test_old_logic_case_146_colossus_radar_count_1`, `.test_old_logic_case_012_standard_conflict_scanner_count_1`, `.test_old_logic_case_148_colossus_excavation_count_1`, `.test_old_logic_case_149_colossus_amplifier_count_1`, `.test_old_logic_case_150_colossus_power_count_1`, `.test_old_logic_case_151_colossus_mineral_count_1`, `.test_old_logic_case_152_solar_cyclotron_count_5`, `.test_old_logic_case_153_solar_infra_count_5`, `.test_old_logic_case_154_solar_phase_count_5`, `.test_old_logic_case_155_solar_photon_count_5`, `.test_old_logic_case_156_solar_positron_count_5`, `.test_old_logic_case_013_standard_economy_scanner_count_1`, `.test_old_logic_case_157_solar_rocket_count_2`, `.test_old_logic_case_159_solar_hyper_count_9`, `.test_old_logic_case_161_solar_pulse_count_9`, `.test_old_logic_case_162_solar_aqua_count_1`, `.test_old_logic_case_163_solar_bobble_count_4`, `.test_pulse_spitter_jetpack_less_than_8`, `.test_pulse_spitter_jetpack_8_plus`, `.test_pulse_6_modules`, `.test_pulse_7_to_8_modules`, `.test_pulse_9_plus_modules`, `.test_generic_fallback_less_than_3`, `.test_generic_fallback_3_modules`, `.test_generic_fallback_4_modules`, `.test_generic_fallback_5_to_6_modules`, `.test_generic_fallback_7_modules`, `.test_generic_fallback_8_modules`, `.test_generic_fallback_10_plus_modules`, `.test_very_large_module_count`, `.test_dimensions_are_positive`, `.test_zero_modules_returns_default`, `.test_negative_module_count_treated_as_zero`, `.test_sentinel_photonix_override`, `.test_corvette_pulse_7_modules`, `.test_corvette_7_modules_non_pulse`, `.test_hyper_12_plus_modules`, `.test_hyper_10_to_11_modules`, `.test_hyper_9_modules`, `.test_hyper_less_than_9_modules`, `.test_old_logic_case_085_sentinel_bobble_count_4`?**
  _High betweenness centrality (0.229) - this node is a cross-community bridge._
- **Why does `get_module_data()` connect `get_module_data` to `place_module`, `clear_all_modules_of_tech`, `data_loader.py`, `app.py`, `TestGetModuleData`, `TestOldDetermineWindowDimensionsBehavior`, `determine_window_dimensions`, `.test_old_logic_case_238_staves_cloaking_count_1`, `TestDataLoaderErrorHandling`, `.test_old_logic_case_176_pilgrim_mounted_count_4`, `.test_old_logic_case_226_staves_analysis_count_1`, `.test_old_logic_case_094_nomad_boost_count_4`, `TestDataIntegrity`, `.test_old_logic_case_095_nomad_slide_count_1`, `.test_old_logic_case_164_solar_conflict_scanner_count_1`, `.test_old_logic_case_165_solar_economy_scanner_count_1`, `.test_old_logic_case_105_nomad_cyclops_count_1`, `.test_old_logic_case_167_solar_trails_count_12`, `.test_old_logic_case_168_solar_teleporter_count_1`, `.test_old_logic_case_170_pilgrim_icarus_count_1`, `.test_old_logic_case_172_pilgrim_slide_count_1`, `.test_old_logic_case_173_pilgrim_grip_count_1`, `.test_old_logic_case_174_pilgrim_drift_count_1`, `.test_old_logic_case_175_pilgrim_mining_count_5`, `.test_old_logic_case_015_standard_trails_count_12`, `.test_old_logic_case_177_pilgrim_flamethrower_count_1`, `.test_old_logic_case_184_pilgrim_amplifier_count_1`, `.test_old_logic_case_185_pilgrim_power_count_1`, `.test_old_logic_case_016_standard_teleporter_count_1`, `.test_old_logic_case_187_nautilon_icarus_count_1`, `.test_old_logic_case_188_nautilon_dredging_count_1`, `.test_old_logic_case_192_nautilon_sonar_count_1`, `.test_old_logic_case_193_living_grafted_count_4`, `.test_old_logic_case_194_living_spewing_count_4`, `.test_old_logic_case_195_living_scream_count_4`, `.test_old_logic_case_196_living_assembly_count_5`, `.test_old_logic_case_198_living_pulsing_count_4`, `.test_old_logic_case_200_living_bobble_count_4`, `.test_old_logic_case_201_living_scanners_count_2`, `.test_old_logic_case_204_exosuit_refiner_count_1`, `.test_old_logic_case_205_exosuit_life_count_6`, `.test_old_logic_case_018_atlantid_analysis_count_1`, `.test_old_logic_case_207_exosuit_anomaly_count_1`, `.test_old_logic_case_208_exosuit_hazard_count_1`, `.test_old_logic_case_209_exosuit_pressure_count_1`, `.test_old_logic_case_211_exosuit_radiation_count_4`, `.test_old_logic_case_213_exosuit_thermic_count_4`, `.test_old_logic_case_214_exosuit_toxin_count_4`, `.test_old_logic_case_216_exosuit_defense_count_3`, `.test_old_logic_case_217_exosuit_rebuilt_count_3`, `.test_old_logic_case_218_exosuit_forbidden_count_3`, `.test_old_logic_case_220_exosuit_hazmat_count_1`, `.test_old_logic_case_221_exosuit_nutrient_count_1`, `.test_old_logic_case_222_exosuit_skiff_count_1`, `.test_old_logic_case_223_exosuit_trade_count_1`, `.test_old_logic_case_224_exosuit_exocraft_count_1`, `.test_old_logic_case_225_staves_mining_count_6`, `.test_old_logic_case_227_staves_fishing_count_1`, `.test_old_logic_case_228_staves_gravatino_count_1`, `.test_old_logic_case_229_staves_scanner_count_7`, `.test_old_logic_case_230_staves_survey_count_1`, `.test_old_logic_case_232_staves_bolt_caster_count_9`, `.test_old_logic_case_234_staves_neutron_count_5`, `.test_old_logic_case_235_staves_plasma_launcher_count_4`, `.test_old_logic_case_021_atlantid_scanner_count_7`, `.test_old_logic_case_237_staves_scatter_count_5`, `.test_old_logic_case_171_pilgrim_boost_count_4`, `.test_old_logic_case_239_staves_combat_count_1`, `.test_old_logic_case_240_staves_voltaic_amplifier_count_1`, `.test_old_logic_case_242_staves_personal_count_1`, `.test_old_logic_case_243_staves_terrian_count_1`, `.test_old_logic_case_246_sentinel_mt_fishing_count_1`, `.test_old_logic_case_247_sentinel_mt_gravatino_count_1`, `.test_old_logic_case_249_sentinel_mt_survey_count_1`, `.test_old_logic_case_253_sentinel_mt_neutron_count_5`, `.test_old_logic_case_254_sentinel_mt_plasma_launcher_count_4`, `.test_old_logic_case_023_atlantid_blaze_javelin_count_6`, `.test_old_logic_case_258_sentinel_mt_combat_count_1`, `.test_old_logic_case_259_sentinel_mt_voltaic_amplifier_count_1`, `.test_old_logic_case_260_sentinel_mt_paralysis_count_1`, `.test_old_logic_case_262_sentinel_mt_terrian_count_1`, `.test_old_logic_case_265_roamer_boost_count_4`, `.test_old_logic_case_266_roamer_slide_count_1`, `.test_old_logic_case_024_atlantid_bolt_caster_count_9`, `.test_old_logic_case_268_roamer_drift_count_1`, `.test_old_logic_case_270_roamer_mounted_count_4`, `.test_old_logic_case_271_roamer_flamethrower_count_1`, `.test_old_logic_case_275_roamer_toxic_count_1`, `.test_old_logic_case_276_roamer_cyclops_count_1`, `.test_old_logic_case_025_atlantid_geology_count_4`, `.test_old_logic_case_000_standard_cyclotron_count_5`, `.test_old_logic_case_027_atlantid_plasma_launcher_count_4`, `.test_old_logic_case_028_atlantid_pulse_spitter_count_7`, `.test_old_logic_case_251_sentinel_mt_bolt_caster_count_9`, `.test_old_logic_case_030_atlantid_cloaking_count_1`, `.test_old_logic_case_034_atlantid_personal_count_1`, `.test_old_logic_case_035_atlantid_terrian_count_1`, `.test_old_logic_case_036_corvette_cyclotron_count_5`, `.test_old_logic_case_001_standard_infra_count_5`, `.test_old_logic_case_038_corvette_phase_count_5`, `.test_old_logic_case_039_corvette_photon_count_5`, `.test_old_logic_case_043_corvette_hyper_count_9`, `.test_old_logic_case_045_corvette_launch_count_6`, `.test_old_logic_case_046_corvette_pulse_count_8`, `.test_old_logic_case_002_standard_phase_count_5`, `.test_old_logic_case_047_corvette_habitation_count_3`, `.test_old_logic_case_049_corvette_bobble_count_4`, `.test_old_logic_case_050_corvette_conflict_scanner_count_1`, `.test_old_logic_case_052_corvette_cargo_scanner_count_1`, `.test_old_logic_case_055_standard_mt_mining_count_6`, `.test_old_logic_case_056_standard_mt_analysis_count_1`, `.test_old_logic_case_003_standard_photon_count_5`, `.test_old_logic_case_057_standard_mt_fishing_count_1`, `.test_old_logic_case_059_standard_mt_scanner_count_7`, `.test_old_logic_case_060_standard_mt_survey_count_1`, `.test_old_logic_case_061_standard_mt_blaze_javelin_count_6`, `.test_old_logic_case_062_standard_mt_bolt_caster_count_9`, `.test_old_logic_case_064_standard_mt_neutron_count_5`, `.test_old_logic_case_065_standard_mt_plasma_launcher_count_4`, `.test_old_logic_case_066_standard_mt_pulse_spitter_count_7`, `.test_old_logic_case_004_standard_positron_count_5`, `.test_old_logic_case_071_standard_mt_paralysis_count_1`, `.test_old_logic_case_072_standard_mt_personal_count_1`, `.test_old_logic_case_074_sentinel_cyclotron_count_5`, `.test_old_logic_case_075_sentinel_infra_count_5`, `.test_old_logic_case_079_sentinel_photon_count_5`, `.test_old_logic_case_080_sentinel_shield_count_5`, `.test_old_logic_case_081_sentinel_launch_count_6`, `.test_old_logic_case_083_sentinel_pulse_count_8`, `.test_old_logic_case_086_sentinel_pilot_count_1`, `.test_old_logic_case_006_standard_shield_count_5`, `.test_old_logic_case_087_sentinel_conflict_scanner_count_1`, `.test_old_logic_case_088_sentinel_economy_scanner_count_1`, `.test_old_logic_case_089_sentinel_cargo_scanner_count_1`, `.test_old_logic_case_090_sentinel_trails_count_12`, `.test_old_logic_case_092_nomad_fusion_count_4`, `.test_old_logic_case_040_corvette_positron_count_5`, `.test_old_logic_case_096_nomad_grip_count_1`, `.test_old_logic_case_098_nomad_mining_count_5`, `.test_old_logic_case_100_nomad_flamethrower_count_1`, `.test_old_logic_case_101_nomad_thermal_count_1`, `.test_old_logic_case_102_nomad_cold_count_1`, `.test_old_logic_case_103_nomad_radiation_count_1`, `.test_old_logic_case_106_nomad_radar_count_1`, `.test_old_logic_case_107_nomad_amplifier_count_1`, `.test_old_logic_case_108_nomad_power_count_1`, `.test_old_logic_case_111_minotaur_icarus_count_1`, `.test_old_logic_case_112_minotaur_minotaur_laser_count_6`, `.test_old_logic_case_113_minotaur_minotaur_count_4`, `.test_old_logic_case_114_minotaur_hardframe_right_count_1`, `.test_old_logic_case_115_minotaur_liquidator_right_count_4`, `.test_old_logic_case_117_minotaur_environment_count_1`, `.test_old_logic_case_118_minotaur_cyclops_count_1`, `.test_old_logic_case_119_minotaur_array_count_1`, `.test_old_logic_case_120_minotaur_ai_count_1`, `.test_old_logic_case_121_minotaur_bore_count_1`, `.test_old_logic_case_123_minotaur_liquidator_body_count_1`, `.test_old_logic_case_124_freighter_hyper_count_11`, `.test_old_logic_case_125_freighter_interstellar_count_1`, `.test_old_logic_case_010_standard_aqua_count_1`, `.test_old_logic_case_127_freighter_fleet_fuel_count_3`, `.test_old_logic_case_128_freighter_fleet_speed_count_3`, `.test_old_logic_case_129_freighter_fleet_combat_count_3`, `.test_old_logic_case_130_freighter_fleet_exploration_count_3`, `.test_old_logic_case_131_freighter_fleet_mining_count_3`, `.test_old_logic_case_134_colossus_icarus_count_1`, `.test_old_logic_case_138_colossus_drift_count_1`, `.test_old_logic_case_139_colossus_mining_count_5`, `.test_old_logic_case_141_colossus_flamethrower_count_1`, `.test_old_logic_case_142_colossus_thermal_count_1`, `.test_old_logic_case_146_colossus_radar_count_1`, `.test_old_logic_case_012_standard_conflict_scanner_count_1`, `.test_old_logic_case_148_colossus_excavation_count_1`, `.test_old_logic_case_149_colossus_amplifier_count_1`, `.test_old_logic_case_150_colossus_power_count_1`, `.test_old_logic_case_151_colossus_mineral_count_1`, `.test_old_logic_case_152_solar_cyclotron_count_5`, `.test_old_logic_case_153_solar_infra_count_5`, `.test_old_logic_case_154_solar_phase_count_5`, `.test_old_logic_case_155_solar_photon_count_5`, `.test_old_logic_case_156_solar_positron_count_5`, `.test_old_logic_case_013_standard_economy_scanner_count_1`, `.test_old_logic_case_157_solar_rocket_count_2`, `.test_old_logic_case_159_solar_hyper_count_9`, `.test_old_logic_case_161_solar_pulse_count_9`, `.test_old_logic_case_162_solar_aqua_count_1`, `.test_old_logic_case_163_solar_bobble_count_4`, `.test_corvette_pulse_7_modules`, `.test_corvette_7_modules_non_pulse`, `.test_old_logic_case_085_sentinel_bobble_count_4`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._
- **Are the 58 inferred relationships involving `Grid` (e.g. with `SocketIORequest` and `AdjacencyType`) actually correct?**
  _`Grid` has 58 INFERRED edges - model-reasoned connections that need verification._
- **What connects `generate_solves.sh script`, `name`, `version` to the rest of the system?**
  _50 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `place_module` be split into smaller, more focused modules?**
  _Cohesion score 0.11614401858304298 - nodes in this community are weakly interconnected._
- **Should `Grid` be split into smaller, more focused modules?**
  _Cohesion score 0.031228070175438598 - nodes in this community are weakly interconnected._