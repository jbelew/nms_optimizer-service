import unittest
from src.optimization.helpers import determine_window_dimensions
from src.data_loader import get_module_data


class TestOldDetermineWindowDimensionsBehavior(unittest.TestCase):
    """
    This class captures the behavior of `determine_window_dimensions`
    as it was at commit c1576e5 (the "old" version).

    Failures here indicate a change in behavior from that commit.
    The goal is to ensure the new implementation (which uses external JSON
    overrides and a standard profile) produces the exact same results
    as the old hardcoded logic for these specific inputs.
    """

    def test_old_logic_case_000_standard_cyclotron_count_5(self):
        """Old Logic: standard cyclotron (5 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(5, "cyclotron", "standard", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_001_standard_infra_count_5(self):
        """Old Logic: standard infra (5 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(5, "infra", "standard", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_002_standard_phase_count_5(self):
        """Old Logic: standard phase (5 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(5, "phase", "standard", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_003_standard_photon_count_5(self):
        """Old Logic: standard photon (5 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(5, "photon", "standard", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_004_standard_positron_count_5(self):
        """Old Logic: standard positron (5 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(5, "positron", "standard", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_005_standard_rocket_count_2(self):
        """Old Logic: standard rocket (2 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(2, "rocket", "standard", modules=modules_data)
        self.assertEqual((w, h), (2, 1))

    def test_old_logic_case_006_standard_shield_count_5(self):
        """Old Logic: standard shield (5 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(5, "shield", "standard", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_007_standard_hyper_count_9(self):
        """Old Logic: standard hyper (9 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(9, "hyper", "standard", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_008_standard_launch_count_6(self):
        """Old Logic: standard launch (6 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(6, "launch", "standard", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_009_standard_pulse_count_8(self):
        """Old Logic: standard pulse (8 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(8, "pulse", "standard", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_010_standard_aqua_count_1(self):
        """Old Logic: standard aqua (1 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(1, "aqua", "standard", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_011_standard_bobble_count_4(self):
        """Old Logic: standard bobble (4 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(4, "bobble", "standard", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_012_standard_conflict_scanner_count_1(self):
        """Old Logic: standard conflict_scanner (1 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(1, "conflict_scanner", "standard", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_013_standard_economy_scanner_count_1(self):
        """Old Logic: standard economy_scanner (1 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(1, "economy_scanner", "standard", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_014_standard_cargo_scanner_count_1(self):
        """Old Logic: standard cargo_scanner (1 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(1, "cargo_scanner", "standard", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_015_standard_trails_count_12(self):
        """Old Logic: standard trails (12 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(12, "trails", "standard", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_016_standard_teleporter_count_1(self):
        """Old Logic: standard teleporter (1 modules)"""
        modules_data = get_module_data("standard")
        w, h = determine_window_dimensions(1, "teleporter", "standard", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_017_atlantid_mining_count_7(self):
        """Old Logic: atlantid mining (7 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(7, "mining", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_018_atlantid_analysis_count_1(self):
        """Old Logic: atlantid analysis (1 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(1, "analysis", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_019_atlantid_fishing_count_1(self):
        """Old Logic: atlantid fishing (1 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(1, "fishing", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_020_atlantid_gravatino_count_1(self):
        """Old Logic: atlantid gravatino (1 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(1, "gravatino", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_021_atlantid_scanner_count_7(self):
        """Old Logic: atlantid scanner (7 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(7, "scanner", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_022_atlantid_survey_count_1(self):
        """Old Logic: atlantid survey (1 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(1, "survey", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_023_atlantid_blaze_javelin_count_6(self):
        """Old Logic: atlantid blaze-javelin (6 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(6, "blaze-javelin", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_024_atlantid_bolt_caster_count_9(self):
        """Old Logic: atlantid bolt-caster (9 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(9, "bolt-caster", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_025_atlantid_geology_count_4(self):
        """Old Logic: atlantid geology (4 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(4, "geology", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_026_atlantid_neutron_count_5(self):
        """Old Logic: atlantid neutron (5 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(5, "neutron", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_027_atlantid_plasma_launcher_count_4(self):
        """Old Logic: atlantid plasma-launcher (4 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(4, "plasma-launcher", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_028_atlantid_pulse_spitter_count_7(self):
        """Old Logic: atlantid pulse-spitter (7 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(7, "pulse-spitter", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_029_atlantid_scatter_count_5(self):
        """Old Logic: atlantid scatter (5 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(5, "scatter", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_030_atlantid_cloaking_count_1(self):
        """Old Logic: atlantid cloaking (1 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(1, "cloaking", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_031_atlantid_combat_count_1(self):
        """Old Logic: atlantid combat (1 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(1, "combat", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_032_atlantid_voltaic_amplifier_count_1(self):
        """Old Logic: atlantid voltaic-amplifier (1 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(1, "voltaic-amplifier", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_033_atlantid_paralysis_count_1(self):
        """Old Logic: atlantid paralysis (1 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(1, "paralysis", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_034_atlantid_personal_count_1(self):
        """Old Logic: atlantid personal (1 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(1, "personal", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_035_atlantid_terrian_count_1(self):
        """Old Logic: atlantid terrian (1 modules)"""
        modules_data = get_module_data("atlantid")
        w, h = determine_window_dimensions(1, "terrian", "atlantid", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_036_corvette_cyclotron_count_5(self):
        """Old Logic: corvette cyclotron (5 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(5, "cyclotron", "corvette", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_037_corvette_infra_count_5(self):
        """Old Logic: corvette infra (5 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(5, "infra", "corvette", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_038_corvette_phase_count_5(self):
        """Old Logic: corvette phase (5 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(5, "phase", "corvette", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_039_corvette_photon_count_5(self):
        """Old Logic: corvette photon (5 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(5, "photon", "corvette", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_040_corvette_positron_count_5(self):
        """Old Logic: corvette positron (5 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(5, "positron", "corvette", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_041_corvette_rocket_count_2(self):
        """Old Logic: corvette rocket (2 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(2, "rocket", "corvette", modules=modules_data)
        self.assertEqual((w, h), (2, 1))

    def test_old_logic_case_042_corvette_shield_count_5(self):
        """Old Logic: corvette shield (5 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(5, "shield", "corvette", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_043_corvette_hyper_count_9(self):
        """Old Logic: corvette hyper (9 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(9, "hyper", "corvette", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_044_corvette_cockpit_count_1(self):
        """Old Logic: corvette cockpit (1 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(1, "cockpit", "corvette", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_045_corvette_launch_count_6(self):
        """Old Logic: corvette launch (6 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(6, "launch", "corvette", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_046_corvette_pulse_count_8(self):
        """Old Logic: corvette pulse (8 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(8, "pulse", "corvette", modules=modules_data)
        self.assertEqual((w, h), (4, 2))

    def test_old_logic_case_047_corvette_habitation_count_3(self):
        """Old Logic: corvette habitation (3 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(3, "habitation", "corvette", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_048_corvette_aqua_count_1(self):
        """Old Logic: corvette aqua (1 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(1, "aqua", "corvette", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_049_corvette_bobble_count_4(self):
        """Old Logic: corvette bobble (4 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(4, "bobble", "corvette", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_050_corvette_conflict_scanner_count_1(self):
        """Old Logic: corvette conflict_scanner (1 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(1, "conflict_scanner", "corvette", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_051_corvette_economy_scanner_count_1(self):
        """Old Logic: corvette economy_scanner (1 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(1, "economy_scanner", "corvette", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_052_corvette_cargo_scanner_count_1(self):
        """Old Logic: corvette cargo_scanner (1 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(1, "cargo_scanner", "corvette", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_053_corvette_trails_count_12(self):
        """Old Logic: corvette trails (12 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(12, "trails", "corvette", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_054_corvette_teleporter_count_1(self):
        """Old Logic: corvette teleporter (1 modules)"""
        modules_data = get_module_data("corvette")
        w, h = determine_window_dimensions(1, "teleporter", "corvette", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_055_standard_mt_mining_count_6(self):
        """Old Logic: standard-mt mining (6 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(6, "mining", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_056_standard_mt_analysis_count_1(self):
        """Old Logic: standard-mt analysis (1 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(1, "analysis", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_057_standard_mt_fishing_count_1(self):
        """Old Logic: standard-mt fishing (1 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(1, "fishing", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_058_standard_mt_gravatino_count_1(self):
        """Old Logic: standard-mt gravatino (1 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(1, "gravatino", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_059_standard_mt_scanner_count_7(self):
        """Old Logic: standard-mt scanner (7 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(7, "scanner", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_060_standard_mt_survey_count_1(self):
        """Old Logic: standard-mt survey (1 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(1, "survey", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_061_standard_mt_blaze_javelin_count_6(self):
        """Old Logic: standard-mt blaze-javelin (6 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(6, "blaze-javelin", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_062_standard_mt_bolt_caster_count_9(self):
        """Old Logic: standard-mt bolt-caster (9 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(9, "bolt-caster", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_063_standard_mt_geology_count_4(self):
        """Old Logic: standard-mt geology (4 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(4, "geology", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_064_standard_mt_neutron_count_5(self):
        """Old Logic: standard-mt neutron (5 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(5, "neutron", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_065_standard_mt_plasma_launcher_count_4(self):
        """Old Logic: standard-mt plasma-launcher (4 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(4, "plasma-launcher", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_066_standard_mt_pulse_spitter_count_7(self):
        """Old Logic: standard-mt pulse-spitter (7 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(7, "pulse-spitter", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_067_standard_mt_scatter_count_5(self):
        """Old Logic: standard-mt scatter (5 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(5, "scatter", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_068_standard_mt_cloaking_count_1(self):
        """Old Logic: standard-mt cloaking (1 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(1, "cloaking", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_069_standard_mt_combat_count_1(self):
        """Old Logic: standard-mt combat (1 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(1, "combat", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_070_standard_mt_voltaic_amplifier_count_1(self):
        """Old Logic: standard-mt voltaic-amplifier (1 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(1, "voltaic-amplifier", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_071_standard_mt_paralysis_count_1(self):
        """Old Logic: standard-mt paralysis (1 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(1, "paralysis", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_072_standard_mt_personal_count_1(self):
        """Old Logic: standard-mt personal (1 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(1, "personal", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_073_standard_mt_terrian_count_1(self):
        """Old Logic: standard-mt terrian (1 modules)"""
        modules_data = get_module_data("standard-mt")
        w, h = determine_window_dimensions(1, "terrian", "standard-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_074_sentinel_cyclotron_count_5(self):
        """Old Logic: sentinel cyclotron (5 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(5, "cyclotron", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_075_sentinel_infra_count_5(self):
        """Old Logic: sentinel infra (5 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(5, "infra", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_076_sentinel_phase_count_5(self):
        """Old Logic: sentinel phase (5 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(5, "phase", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_077_sentinel_positron_count_5(self):
        """Old Logic: sentinel positron (5 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(5, "positron", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_078_sentinel_rocket_count_2(self):
        """Old Logic: sentinel rocket (2 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(2, "rocket", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (2, 1))

    def test_old_logic_case_079_sentinel_photon_count_5(self):
        """Old Logic: sentinel photon (5 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(5, "photon", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_080_sentinel_shield_count_5(self):
        """Old Logic: sentinel shield (5 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(5, "shield", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_081_sentinel_launch_count_6(self):
        """Old Logic: sentinel launch (6 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(6, "launch", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_082_sentinel_hyper_count_9(self):
        """Old Logic: sentinel hyper (9 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(9, "hyper", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_083_sentinel_pulse_count_8(self):
        """Old Logic: sentinel pulse (8 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(8, "pulse", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_084_sentinel_aqua_count_1(self):
        """Old Logic: sentinel aqua (1 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(1, "aqua", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_085_sentinel_bobble_count_4(self):
        """Old Logic: sentinel bobble (4 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(4, "bobble", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_086_sentinel_pilot_count_1(self):
        """Old Logic: sentinel pilot (1 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(1, "pilot", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_087_sentinel_conflict_scanner_count_1(self):
        """Old Logic: sentinel conflict_scanner (1 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(1, "conflict_scanner", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_088_sentinel_economy_scanner_count_1(self):
        """Old Logic: sentinel economy_scanner (1 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(1, "economy_scanner", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_089_sentinel_cargo_scanner_count_1(self):
        """Old Logic: sentinel cargo_scanner (1 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(1, "cargo_scanner", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_090_sentinel_trails_count_12(self):
        """Old Logic: sentinel trails (12 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(12, "trails", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_091_sentinel_teleporter_count_1(self):
        """Old Logic: sentinel teleporter (1 modules)"""
        modules_data = get_module_data("sentinel")
        w, h = determine_window_dimensions(1, "teleporter", "sentinel", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_092_nomad_fusion_count_4(self):
        """Old Logic: nomad fusion (4 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(4, "fusion", "nomad", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_093_nomad_icarus_count_1(self):
        """Old Logic: nomad icarus (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "icarus", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_094_nomad_boost_count_4(self):
        """Old Logic: nomad boost (4 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(4, "boost", "nomad", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_095_nomad_slide_count_1(self):
        """Old Logic: nomad slide (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "slide", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_096_nomad_grip_count_1(self):
        """Old Logic: nomad grip (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "grip", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_097_nomad_drift_count_1(self):
        """Old Logic: nomad drift (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "drift", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_098_nomad_mining_count_5(self):
        """Old Logic: nomad mining (5 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(5, "mining", "nomad", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_099_nomad_mounted_count_4(self):
        """Old Logic: nomad mounted (4 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(4, "mounted", "nomad", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_100_nomad_flamethrower_count_1(self):
        """Old Logic: nomad flamethrower (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "flamethrower", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_101_nomad_thermal_count_1(self):
        """Old Logic: nomad thermal (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "thermal", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_102_nomad_cold_count_1(self):
        """Old Logic: nomad cold (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "cold", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_103_nomad_radiation_count_1(self):
        """Old Logic: nomad radiation (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "radiation", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_104_nomad_toxic_count_1(self):
        """Old Logic: nomad toxic (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "toxic", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_105_nomad_cyclops_count_1(self):
        """Old Logic: nomad cyclops (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "cyclops", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_106_nomad_radar_count_1(self):
        """Old Logic: nomad radar (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "radar", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_107_nomad_amplifier_count_1(self):
        """Old Logic: nomad amplifier (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "amplifier", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_108_nomad_power_count_1(self):
        """Old Logic: nomad power (1 modules)"""
        modules_data = get_module_data("nomad")
        w, h = determine_window_dimensions(1, "power", "nomad", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_109_minotaur_daedalus_count_7(self):
        """Old Logic: minotaur daedalus (7 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(7, "daedalus", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_110_minotaur_ariadnes_count_1(self):
        """Old Logic: minotaur ariadnes (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "ariadnes", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_111_minotaur_icarus_count_1(self):
        """Old Logic: minotaur icarus (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "icarus", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_112_minotaur_minotaur_laser_count_6(self):
        """Old Logic: minotaur minotaur-laser (6 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(6, "minotaur-laser", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_113_minotaur_minotaur_count_4(self):
        """Old Logic: minotaur minotaur (4 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(4, "minotaur", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_114_minotaur_hardframe_right_count_1(self):
        """Old Logic: minotaur hardframe-right (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "hardframe-right", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_115_minotaur_liquidator_right_count_4(self):
        """Old Logic: minotaur liquidator-right (4 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(4, "liquidator-right", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_116_minotaur_liquidator_left_count_1(self):
        """Old Logic: minotaur liquidator-left (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "liquidator-left", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_117_minotaur_environment_count_1(self):
        """Old Logic: minotaur environment (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "environment", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_118_minotaur_cyclops_count_1(self):
        """Old Logic: minotaur cyclops (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "cyclops", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_119_minotaur_array_count_1(self):
        """Old Logic: minotaur array (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "array", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_120_minotaur_ai_count_1(self):
        """Old Logic: minotaur ai (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "ai", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_121_minotaur_bore_count_1(self):
        """Old Logic: minotaur bore (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "bore", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_122_minotaur_hardframe_body_count_1(self):
        """Old Logic: minotaur hardframe-body (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "hardframe-body", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_123_minotaur_liquidator_body_count_1(self):
        """Old Logic: minotaur liquidator-body (1 modules)"""
        modules_data = get_module_data("minotaur")
        w, h = determine_window_dimensions(1, "liquidator-body", "minotaur", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_124_freighter_hyper_count_11(self):
        """Old Logic: freighter hyper (11 modules)"""
        modules_data = get_module_data("freighter")
        w, h = determine_window_dimensions(11, "hyper", "freighter", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_125_freighter_interstellar_count_1(self):
        """Old Logic: freighter interstellar (1 modules)"""
        modules_data = get_module_data("freighter")
        w, h = determine_window_dimensions(1, "interstellar", "freighter", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_126_freighter_matterbeam_count_1(self):
        """Old Logic: freighter matterbeam (1 modules)"""
        modules_data = get_module_data("freighter")
        w, h = determine_window_dimensions(1, "matterbeam", "freighter", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_127_freighter_fleet_fuel_count_3(self):
        """Old Logic: freighter fleet-fuel (3 modules)"""
        modules_data = get_module_data("freighter")
        w, h = determine_window_dimensions(3, "fleet-fuel", "freighter", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_128_freighter_fleet_speed_count_3(self):
        """Old Logic: freighter fleet-speed (3 modules)"""
        modules_data = get_module_data("freighter")
        w, h = determine_window_dimensions(3, "fleet-speed", "freighter", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_129_freighter_fleet_combat_count_3(self):
        """Old Logic: freighter fleet-combat (3 modules)"""
        modules_data = get_module_data("freighter")
        w, h = determine_window_dimensions(3, "fleet-combat", "freighter", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_130_freighter_fleet_exploration_count_3(self):
        """Old Logic: freighter fleet-exploration (3 modules)"""
        modules_data = get_module_data("freighter")
        w, h = determine_window_dimensions(3, "fleet-exploration", "freighter", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_131_freighter_fleet_mining_count_3(self):
        """Old Logic: freighter fleet-mining (3 modules)"""
        modules_data = get_module_data("freighter")
        w, h = determine_window_dimensions(3, "fleet-mining", "freighter", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_132_freighter_fleet_trade_count_3(self):
        """Old Logic: freighter fleet-trade (3 modules)"""
        modules_data = get_module_data("freighter")
        w, h = determine_window_dimensions(3, "fleet-trade", "freighter", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_133_colossus_fusion_count_4(self):
        """Old Logic: colossus fusion (4 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(4, "fusion", "colossus", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_134_colossus_icarus_count_1(self):
        """Old Logic: colossus icarus (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "icarus", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_135_colossus_boost_count_4(self):
        """Old Logic: colossus boost (4 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(4, "boost", "colossus", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_136_colossus_slide_count_1(self):
        """Old Logic: colossus slide (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "slide", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_137_colossus_grip_count_1(self):
        """Old Logic: colossus grip (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "grip", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_138_colossus_drift_count_1(self):
        """Old Logic: colossus drift (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "drift", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_139_colossus_mining_count_5(self):
        """Old Logic: colossus mining (5 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(5, "mining", "colossus", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_140_colossus_mounted_count_4(self):
        """Old Logic: colossus mounted (4 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(4, "mounted", "colossus", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_141_colossus_flamethrower_count_1(self):
        """Old Logic: colossus flamethrower (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "flamethrower", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_142_colossus_thermal_count_1(self):
        """Old Logic: colossus thermal (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "thermal", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_143_colossus_cold_count_1(self):
        """Old Logic: colossus cold (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "cold", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_144_colossus_radiation_count_1(self):
        """Old Logic: colossus radiation (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "radiation", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_145_colossus_toxic_count_1(self):
        """Old Logic: colossus toxic (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "toxic", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_146_colossus_radar_count_1(self):
        """Old Logic: colossus radar (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "radar", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_147_colossus_cyclops_count_1(self):
        """Old Logic: colossus cyclops (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "cyclops", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_148_colossus_excavation_count_1(self):
        """Old Logic: colossus excavation (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "excavation", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_149_colossus_amplifier_count_1(self):
        """Old Logic: colossus amplifier (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "amplifier", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_150_colossus_power_count_1(self):
        """Old Logic: colossus power (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "power", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_151_colossus_mineral_count_1(self):
        """Old Logic: colossus mineral (1 modules)"""
        modules_data = get_module_data("colossus")
        w, h = determine_window_dimensions(1, "mineral", "colossus", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_152_solar_cyclotron_count_5(self):
        """Old Logic: solar cyclotron (5 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(5, "cyclotron", "solar", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_153_solar_infra_count_5(self):
        """Old Logic: solar infra (5 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(5, "infra", "solar", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_154_solar_phase_count_5(self):
        """Old Logic: solar phase (5 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(5, "phase", "solar", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_155_solar_photon_count_5(self):
        """Old Logic: solar photon (5 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(5, "photon", "solar", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_156_solar_positron_count_5(self):
        """Old Logic: solar positron (5 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(5, "positron", "solar", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_157_solar_rocket_count_2(self):
        """Old Logic: solar rocket (2 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(2, "rocket", "solar", modules=modules_data)
        self.assertEqual((w, h), (2, 1))

    def test_old_logic_case_158_solar_shield_count_5(self):
        """Old Logic: solar shield (5 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(5, "shield", "solar", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_159_solar_hyper_count_9(self):
        """Old Logic: solar hyper (9 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(9, "hyper", "solar", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_160_solar_launch_count_6(self):
        """Old Logic: solar launch (6 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(6, "launch", "solar", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_161_solar_pulse_count_9(self):
        """Old Logic: solar pulse (9 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(9, "pulse", "solar", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_162_solar_aqua_count_1(self):
        """Old Logic: solar aqua (1 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(1, "aqua", "solar", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_163_solar_bobble_count_4(self):
        """Old Logic: solar bobble (4 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(4, "bobble", "solar", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_164_solar_conflict_scanner_count_1(self):
        """Old Logic: solar conflict_scanner (1 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(1, "conflict_scanner", "solar", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_165_solar_economy_scanner_count_1(self):
        """Old Logic: solar economy_scanner (1 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(1, "economy_scanner", "solar", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_166_solar_cargo_scanner_count_1(self):
        """Old Logic: solar cargo_scanner (1 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(1, "cargo_scanner", "solar", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_167_solar_trails_count_12(self):
        """Old Logic: solar trails (12 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(12, "trails", "solar", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_168_solar_teleporter_count_1(self):
        """Old Logic: solar teleporter (1 modules)"""
        modules_data = get_module_data("solar")
        w, h = determine_window_dimensions(1, "teleporter", "solar", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_169_pilgrim_fusion_count_4(self):
        """Old Logic: pilgrim fusion (4 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(4, "fusion", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_170_pilgrim_icarus_count_1(self):
        """Old Logic: pilgrim icarus (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "icarus", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_171_pilgrim_boost_count_4(self):
        """Old Logic: pilgrim boost (4 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(4, "boost", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_172_pilgrim_slide_count_1(self):
        """Old Logic: pilgrim slide (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "slide", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_173_pilgrim_grip_count_1(self):
        """Old Logic: pilgrim grip (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "grip", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_174_pilgrim_drift_count_1(self):
        """Old Logic: pilgrim drift (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "drift", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_175_pilgrim_mining_count_5(self):
        """Old Logic: pilgrim mining (5 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(5, "mining", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_176_pilgrim_mounted_count_4(self):
        """Old Logic: pilgrim mounted (4 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(4, "mounted", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_177_pilgrim_flamethrower_count_1(self):
        """Old Logic: pilgrim flamethrower (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "flamethrower", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_178_pilgrim_thermal_count_1(self):
        """Old Logic: pilgrim thermal (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "thermal", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_179_pilgrim_cold_count_1(self):
        """Old Logic: pilgrim cold (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "cold", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_180_pilgrim_radiation_count_1(self):
        """Old Logic: pilgrim radiation (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "radiation", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_181_pilgrim_toxic_count_1(self):
        """Old Logic: pilgrim toxic (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "toxic", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_182_pilgrim_cyclops_count_1(self):
        """Old Logic: pilgrim cyclops (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "cyclops", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_183_pilgrim_radar_count_1(self):
        """Old Logic: pilgrim radar (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "radar", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_184_pilgrim_amplifier_count_1(self):
        """Old Logic: pilgrim amplifier (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "amplifier", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_185_pilgrim_power_count_1(self):
        """Old Logic: pilgrim power (1 modules)"""
        modules_data = get_module_data("pilgrim")
        w, h = determine_window_dimensions(1, "power", "pilgrim", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_186_nautilon_humboldt_count_5(self):
        """Old Logic: nautilon humboldt (5 modules)"""
        modules_data = get_module_data("nautilon")
        w, h = determine_window_dimensions(5, "humboldt", "nautilon", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_187_nautilon_icarus_count_1(self):
        """Old Logic: nautilon icarus (1 modules)"""
        modules_data = get_module_data("nautilon")
        w, h = determine_window_dimensions(1, "icarus", "nautilon", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_188_nautilon_dredging_count_1(self):
        """Old Logic: nautilon dredging (1 modules)"""
        modules_data = get_module_data("nautilon")
        w, h = determine_window_dimensions(1, "dredging", "nautilon", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_189_nautilon_tethys_count_1(self):
        """Old Logic: nautilon tethys (1 modules)"""
        modules_data = get_module_data("nautilon")
        w, h = determine_window_dimensions(1, "tethys", "nautilon", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_190_nautilon_nautilon_count_4(self):
        """Old Logic: nautilon nautilon (4 modules)"""
        modules_data = get_module_data("nautilon")
        w, h = determine_window_dimensions(4, "nautilon", "nautilon", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_191_nautilon_cyclops_count_1(self):
        """Old Logic: nautilon cyclops (1 modules)"""
        modules_data = get_module_data("nautilon")
        w, h = determine_window_dimensions(1, "cyclops", "nautilon", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_192_nautilon_sonar_count_1(self):
        """Old Logic: nautilon sonar (1 modules)"""
        modules_data = get_module_data("nautilon")
        w, h = determine_window_dimensions(1, "sonar", "nautilon", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_193_living_grafted_count_4(self):
        """Old Logic: living grafted (4 modules)"""
        modules_data = get_module_data("living")
        w, h = determine_window_dimensions(4, "grafted", "living", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_194_living_spewing_count_4(self):
        """Old Logic: living spewing (4 modules)"""
        modules_data = get_module_data("living")
        w, h = determine_window_dimensions(4, "spewing", "living", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_195_living_scream_count_4(self):
        """Old Logic: living scream (4 modules)"""
        modules_data = get_module_data("living")
        w, h = determine_window_dimensions(4, "scream", "living", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_196_living_assembly_count_5(self):
        """Old Logic: living assembly (5 modules)"""
        modules_data = get_module_data("living")
        w, h = determine_window_dimensions(5, "assembly", "living", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_197_living_singularity_count_5(self):
        """Old Logic: living singularity (5 modules)"""
        modules_data = get_module_data("living")
        w, h = determine_window_dimensions(5, "singularity", "living", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_198_living_pulsing_count_4(self):
        """Old Logic: living pulsing (4 modules)"""
        modules_data = get_module_data("living")
        w, h = determine_window_dimensions(4, "pulsing", "living", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_199_living_saline_count_1(self):
        """Old Logic: living saline (1 modules)"""
        modules_data = get_module_data("living")
        w, h = determine_window_dimensions(1, "saline", "living", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_200_living_bobble_count_4(self):
        """Old Logic: living bobble (4 modules)"""
        modules_data = get_module_data("living")
        w, h = determine_window_dimensions(4, "bobble", "living", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_201_living_scanners_count_2(self):
        """Old Logic: living scanners (2 modules)"""
        modules_data = get_module_data("living")
        w, h = determine_window_dimensions(2, "scanners", "living", modules=modules_data)
        self.assertEqual((w, h), (2, 1))

    def test_old_logic_case_202_living_trails_count_12(self):
        """Old Logic: living trails (12 modules)"""
        modules_data = get_module_data("living")
        w, h = determine_window_dimensions(12, "trails", "living", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_203_exosuit_jetpack_count_8(self):
        """Old Logic: exosuit jetpack (8 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(8, "jetpack", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_204_exosuit_refiner_count_1(self):
        """Old Logic: exosuit refiner (1 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(1, "refiner", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_205_exosuit_life_count_6(self):
        """Old Logic: exosuit life (6 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(6, "life", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_206_exosuit_core_health_count_3(self):
        """Old Logic: exosuit core_health (3 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(3, "core_health", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_207_exosuit_anomaly_count_1(self):
        """Old Logic: exosuit anomaly (1 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(1, "anomaly", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_208_exosuit_hazard_count_1(self):
        """Old Logic: exosuit hazard (1 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(1, "hazard", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_209_exosuit_pressure_count_1(self):
        """Old Logic: exosuit pressure (1 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(1, "pressure", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_210_exosuit_coolant_count_4(self):
        """Old Logic: exosuit coolant (4 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(4, "coolant", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_211_exosuit_radiation_count_4(self):
        """Old Logic: exosuit radiation (4 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(4, "radiation", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_212_exosuit_aeration_count_4(self):
        """Old Logic: exosuit aeration (4 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(4, "aeration", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_213_exosuit_thermic_count_4(self):
        """Old Logic: exosuit thermic (4 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(4, "thermic", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_214_exosuit_toxin_count_4(self):
        """Old Logic: exosuit toxin (4 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(4, "toxin", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_215_exosuit_protection_count_4(self):
        """Old Logic: exosuit protection (4 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(4, "protection", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_216_exosuit_defense_count_3(self):
        """Old Logic: exosuit defense (3 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(3, "defense", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_217_exosuit_rebuilt_count_3(self):
        """Old Logic: exosuit rebuilt (3 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(3, "rebuilt", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_218_exosuit_forbidden_count_3(self):
        """Old Logic: exosuit forbidden (3 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(3, "forbidden", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_219_exosuit_translators_count_4(self):
        """Old Logic: exosuit translators (4 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(4, "translators", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_220_exosuit_hazmat_count_1(self):
        """Old Logic: exosuit hazmat (1 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(1, "hazmat", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_221_exosuit_nutrient_count_1(self):
        """Old Logic: exosuit nutrient (1 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(1, "nutrient", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_222_exosuit_skiff_count_1(self):
        """Old Logic: exosuit skiff (1 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(1, "skiff", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_223_exosuit_trade_count_1(self):
        """Old Logic: exosuit trade (1 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(1, "trade", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_224_exosuit_exocraft_count_1(self):
        """Old Logic: exosuit exocraft (1 modules)"""
        modules_data = get_module_data("exosuit")
        w, h = determine_window_dimensions(1, "exocraft", "exosuit", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_225_staves_mining_count_6(self):
        """Old Logic: staves mining (6 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(6, "mining", "staves", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_226_staves_analysis_count_1(self):
        """Old Logic: staves analysis (1 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(1, "analysis", "staves", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_227_staves_fishing_count_1(self):
        """Old Logic: staves fishing (1 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(1, "fishing", "staves", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_228_staves_gravatino_count_1(self):
        """Old Logic: staves gravatino (1 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(1, "gravatino", "staves", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_229_staves_scanner_count_7(self):
        """Old Logic: staves scanner (7 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(7, "scanner", "staves", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_230_staves_survey_count_1(self):
        """Old Logic: staves survey (1 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(1, "survey", "staves", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_231_staves_blaze_javelin_count_6(self):
        """Old Logic: staves blaze-javelin (6 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(6, "blaze-javelin", "staves", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_232_staves_bolt_caster_count_9(self):
        """Old Logic: staves bolt-caster (9 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(9, "bolt-caster", "staves", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_233_staves_geology_count_4(self):
        """Old Logic: staves geology (4 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(4, "geology", "staves", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_234_staves_neutron_count_5(self):
        """Old Logic: staves neutron (5 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(5, "neutron", "staves", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_235_staves_plasma_launcher_count_4(self):
        """Old Logic: staves plasma-launcher (4 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(4, "plasma-launcher", "staves", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_236_staves_pulse_spitter_count_7(self):
        """Old Logic: staves pulse-spitter (7 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(7, "pulse-spitter", "staves", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_237_staves_scatter_count_5(self):
        """Old Logic: staves scatter (5 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(5, "scatter", "staves", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_238_staves_cloaking_count_1(self):
        """Old Logic: staves cloaking (1 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(1, "cloaking", "staves", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_239_staves_combat_count_1(self):
        """Old Logic: staves combat (1 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(1, "combat", "staves", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_240_staves_voltaic_amplifier_count_1(self):
        """Old Logic: staves voltaic-amplifier (1 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(1, "voltaic-amplifier", "staves", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_241_staves_paralysis_count_1(self):
        """Old Logic: staves paralysis (1 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(1, "paralysis", "staves", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_242_staves_personal_count_1(self):
        """Old Logic: staves personal (1 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(1, "personal", "staves", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_243_staves_terrian_count_1(self):
        """Old Logic: staves terrian (1 modules)"""
        modules_data = get_module_data("staves")
        w, h = determine_window_dimensions(1, "terrian", "staves", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_244_sentinel_mt_mining_count_7(self):
        """Old Logic: sentinel-mt mining (7 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(7, "mining", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_245_sentinel_mt_analysis_count_1(self):
        """Old Logic: sentinel-mt analysis (1 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(1, "analysis", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_246_sentinel_mt_fishing_count_1(self):
        """Old Logic: sentinel-mt fishing (1 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(1, "fishing", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_247_sentinel_mt_gravatino_count_1(self):
        """Old Logic: sentinel-mt gravatino (1 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(1, "gravatino", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_248_sentinel_mt_scanner_count_7(self):
        """Old Logic: sentinel-mt scanner (7 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(7, "scanner", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_249_sentinel_mt_survey_count_1(self):
        """Old Logic: sentinel-mt survey (1 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(1, "survey", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_250_sentinel_mt_blaze_javelin_count_6(self):
        """Old Logic: sentinel-mt blaze-javelin (6 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(6, "blaze-javelin", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_251_sentinel_mt_bolt_caster_count_9(self):
        """Old Logic: sentinel-mt bolt-caster (9 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(9, "bolt-caster", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (4, 3))

    def test_old_logic_case_252_sentinel_mt_geology_count_4(self):
        """Old Logic: sentinel-mt geology (4 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(4, "geology", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_253_sentinel_mt_neutron_count_5(self):
        """Old Logic: sentinel-mt neutron (5 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(5, "neutron", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_254_sentinel_mt_plasma_launcher_count_4(self):
        """Old Logic: sentinel-mt plasma-launcher (4 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(4, "plasma-launcher", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_255_sentinel_mt_pulse_spitter_count_7(self):
        """Old Logic: sentinel-mt pulse-spitter (7 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(7, "pulse-spitter", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 3))

    def test_old_logic_case_256_sentinel_mt_scatter_count_5(self):
        """Old Logic: sentinel-mt scatter (5 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(5, "scatter", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_257_sentinel_mt_cloaking_count_1(self):
        """Old Logic: sentinel-mt cloaking (1 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(1, "cloaking", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_258_sentinel_mt_combat_count_1(self):
        """Old Logic: sentinel-mt combat (1 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(1, "combat", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_259_sentinel_mt_voltaic_amplifier_count_1(self):
        """Old Logic: sentinel-mt voltaic-amplifier (1 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(1, "voltaic-amplifier", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_260_sentinel_mt_paralysis_count_1(self):
        """Old Logic: sentinel-mt paralysis (1 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(1, "paralysis", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_261_sentinel_mt_personal_count_1(self):
        """Old Logic: sentinel-mt personal (1 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(1, "personal", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_262_sentinel_mt_terrian_count_1(self):
        """Old Logic: sentinel-mt terrian (1 modules)"""
        modules_data = get_module_data("sentinel-mt")
        w, h = determine_window_dimensions(1, "terrian", "sentinel-mt", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_263_roamer_fusion_count_4(self):
        """Old Logic: roamer fusion (4 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(4, "fusion", "roamer", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_264_roamer_icarus_count_1(self):
        """Old Logic: roamer icarus (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "icarus", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_265_roamer_boost_count_4(self):
        """Old Logic: roamer boost (4 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(4, "boost", "roamer", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_266_roamer_slide_count_1(self):
        """Old Logic: roamer slide (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "slide", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_267_roamer_grip_count_1(self):
        """Old Logic: roamer grip (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "grip", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_268_roamer_drift_count_1(self):
        """Old Logic: roamer drift (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "drift", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_269_roamer_mining_count_5(self):
        """Old Logic: roamer mining (5 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(5, "mining", "roamer", modules=modules_data)
        self.assertEqual((w, h), (3, 2))

    def test_old_logic_case_270_roamer_mounted_count_4(self):
        """Old Logic: roamer mounted (4 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(4, "mounted", "roamer", modules=modules_data)
        self.assertEqual((w, h), (2, 2))

    def test_old_logic_case_271_roamer_flamethrower_count_1(self):
        """Old Logic: roamer flamethrower (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "flamethrower", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_272_roamer_thermal_count_1(self):
        """Old Logic: roamer thermal (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "thermal", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_273_roamer_cold_count_1(self):
        """Old Logic: roamer cold (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "cold", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_274_roamer_radiation_count_1(self):
        """Old Logic: roamer radiation (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "radiation", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_275_roamer_toxic_count_1(self):
        """Old Logic: roamer toxic (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "toxic", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_276_roamer_cyclops_count_1(self):
        """Old Logic: roamer cyclops (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "cyclops", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_277_roamer_radar_count_1(self):
        """Old Logic: roamer radar (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "radar", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_278_roamer_amplifier_count_1(self):
        """Old Logic: roamer amplifier (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "amplifier", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))

    def test_old_logic_case_279_roamer_power_count_1(self):
        """Old Logic: roamer power (1 modules)"""
        modules_data = get_module_data("roamer")
        w, h = determine_window_dimensions(1, "power", "roamer", modules=modules_data)
        self.assertEqual((w, h), (1, 1))
