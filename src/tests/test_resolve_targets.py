# src/tests/test_resolve_targets.py
import subprocess
import sys
import unittest
from scripts.training.resolve_targets import resolve_targets, RECOMMENDED_TARGETS


class TestResolveTargets(unittest.TestCase):
    def test_resolve_recommended(self):
        expected = [
            ("corvette", "launch"),
            ("corvette", "pulse"),
            ("corvette", "photonix"),
            ("freighter", "hyper"),
        ]
        self.assertEqual(resolve_targets("recommended"), expected)
        self.assertEqual(resolve_targets(""), expected)

    def test_resolve_explicit_ship_tech_pairs(self):
        targets = resolve_targets("corvette/launch,freighter/hyper")
        self.assertEqual(targets, [("corvette", "launch"), ("freighter", "hyper")])

        # Test space-separated and mixed separators
        targets = resolve_targets("corvette/launch freighter/hyper,standard/pulse")
        self.assertEqual(
            targets,
            [("corvette", "launch"), ("freighter", "hyper"), ("standard", "pulse")],
        )

    def test_resolve_legacy_ship_and_techs(self):
        targets = resolve_targets("corvette", "launch,pulse")
        self.assertEqual(targets, [("corvette", "launch"), ("corvette", "pulse")])

    def test_resolve_legacy_category(self):
        targets = resolve_targets("corvette", "Hyperdrive")
        self.assertIn(("corvette", "launch"), targets)
        self.assertIn(("corvette", "pulse"), targets)
        self.assertIn(("corvette", "photonix"), targets)
        self.assertIn(("corvette", "hyper"), targets)

        targets_weapons = resolve_targets("corvette", "Weaponry")
        self.assertIn(("corvette", "infra"), targets_weapons)
        self.assertIn(("corvette", "phase"), targets_weapons)

    def test_cli_invocation(self):
        result = subprocess.run(
            [sys.executable, "scripts/training/resolve_targets.py", "recommended"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.strip()
        expected = " ".join(RECOMMENDED_TARGETS)
        self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
