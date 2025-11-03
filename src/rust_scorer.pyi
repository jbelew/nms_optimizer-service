# src/rust_scorer.pyi
"""
This is a stub file for the 'rust_scorer' native module.
It provides type hints to static analysis tools like Pylance,
which cannot introspect the compiled Rust code.
"""
from typing import Any, Callable, List, Optional, Dict

class AdjacencyType:
    Greater: Any
    Lesser: Any
    NoAdjacency: Any

class ModuleType:
    Core: Any
    Bonus: Any
    Upgrade: Any
    Cosmetic: Any
    Reactor: Any
    Atlantid: Any

class Module:
    def __init__(
        self,
        *,
        id: str,
        label: str,
        tech: str,
        module_type: "ModuleType",
        bonus: float,
        adjacency: "AdjacencyType",
        sc_eligible: bool,
        image: Optional[str],
    ) -> None: ...

class Cell: ...
class Grid: ...

def calculate_grid_score(grid: Grid, tech: str, apply_supercharge_first: bool = False) -> float: ...
def simulated_annealing(
    grid_json: str,
    tech_modules: List[Module],
    tech: str,
    initial_temperature: float,
    cooling_rate: float,
    stopping_temperature: float,
    iterations_per_temp: int,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> tuple[str, float]: ...
