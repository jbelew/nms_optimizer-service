import unittest
from src.modules_utils import get_tech_modules


class TestGetTechModules(unittest.TestCase):
    def setUp(self):
        self.modules = {
            "hauler": {
                "types": {
                    "tech": [
                        {"key": "test_tech", "type": "max", "modules": [{"id": "max_module", "type": "bonus"}]},
                        {"key": "test_tech", "modules": [{"id": "normal_module", "type": "bonus"}]},
                    ]
                }
            }
        }

    def test_get_tech_modules_returns_normal_modules(self):
        tech_modules = get_tech_modules(self.modules["hauler"], "hauler", "test_tech")
        # Type guard: ensure tech_modules is not None or empty
        self.assertIsNotNone(tech_modules)
        if isinstance(tech_modules, list):
            self.assertEqual(len(tech_modules), 1)
            self.assertEqual(tech_modules[0]["id"], "normal_module")


if __name__ == "__main__":
    unittest.main()
