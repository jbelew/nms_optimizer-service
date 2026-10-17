# scripts/training/resolve_targets.py
"""
Utility module to resolve training and data generation target specifications.

Supports:
- "recommended": Tier 1 high-benefit targets (corvette/launch, corvette/pulse,
  corvette/photonix, freighter/hyper).
- Explicit comma/space-separated "ship/tech" pairs: "corvette/launch,freighter/hyper".
- Legacy single-ship syntax: ship="corvette", target="launch,pulse" or category name.
"""

from typing import List, Tuple
import sys
from src.data_loader import get_all_module_data

RECOMMENDED_TARGETS: List[str] = [
    "corvette/launch",
    "corvette/pulse",
    "corvette/photonix",
    "freighter/hyper",
]


def resolve_targets(arg1: str, arg2: str = "") -> List[Tuple[str, str]]:
    """Resolves target specifications into a list of (ship, tech) tuples.

    Args:
        arg1 (str): First argument ('recommended', 'ship/tech', or ship key).
        arg2 (str): Second argument (tech keys, category, or empty).

    Returns:
        List[Tuple[str, str]]: List of (ship, tech) tuples.
    """
    arg1 = (arg1 or "").strip()
    arg2 = (arg2 or "").strip()

    if not arg1 or arg1 == "recommended":
        return [tuple(t.split("/", 1)) for t in RECOMMENDED_TARGETS]  # type: ignore

    if "/" in arg1 or "/" in arg2:
        combined = f"{arg1} {arg2}".replace(",", " ")
        results: List[Tuple[str, str]] = []
        for token in combined.split():
            token = token.strip()
            if "/" in token:
                parts = token.split("/", 1)
                results.append((parts[0].strip(), parts[1].strip()))
        return results

    # Legacy: arg1 is ship, arg2 is tech key(s) or category
    ship = arg1
    target = arg2
    all_data = get_all_module_data()
    ship_data = all_data.get(ship, {})
    types = ship_data.get("types", {})

    if target in types:
        tech_keys = [t["key"] for t in types[target] if isinstance(t, dict) and "key" in t]
    elif target:
        tech_keys = [t.strip() for t in target.replace(",", " ").split() if t.strip()]
    else:
        tech_keys = []
        for cat in types.values():
            for t in cat:
                if isinstance(t, dict) and "key" in t:
                    tech_keys.append(t["key"])

    return [(ship, tk) for tk in tech_keys]


if __name__ == "__main__":
    a1 = sys.argv[1] if len(sys.argv) > 1 else "recommended"
    a2 = sys.argv[2] if len(sys.argv) > 2 else ""
    resolved = resolve_targets(a1, a2)
    print(" ".join(f"{s}/{t}" for s, t in resolved))
