# Agent Guidelines

NMS Optimizer Service is the Python (Flask + Rust `rust_scorer` + PyTorch) backend solver for No Man's Sky technology layout optimization.

## Core Directives

- **Virtual Environment**: Always invoke Python tools via `./venv/bin/` (`./venv/bin/python`, `./venv/bin/pytest`, `./venv/bin/ruff`, `./venv/bin/pyright`).
- **Python Documentation**: Document all public functions, classes, methods, and non-obvious helpers with docstrings including `Args:` and `Returns:` sections, alongside Python type annotations.
- **Domain Language**: Follow canonical domain terms in [`../nms_optimizer-web/GLOSSARY.md`](../nms_optimizer-web/GLOSSARY.md) when naming functions, parameters, docstrings, and tests.
- **Commits**: Follow Angular / Conventional Commit conventions (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`), enforced by Husky `commit-msg` (`commitlint`). Run `git pull --ff-only` before committing on `main` because CI pushes automated Commitizen release commits.

## Verification Gate (Definition of Done)

Before marking any code task complete, execute this sequence (also enforced on staged `.py` files on `git commit` via `.husky/pre-commit`):

1. `./venv/bin/pyright <touched-files>` — Must report 0 errors.
2. `./venv/bin/ruff check --fix <touched-files>` — Must report 0 lint errors.
3. `./venv/bin/ruff format --check <touched-files>` — Format touched files with `./venv/bin/ruff format <touched-files>`.
4. `./venv/bin/pytest <targeted-test-files>` (and `./venv/bin/pytest -m "not slow"` before finalizing).

## Key Architecture & Domain Notes

- **Optimization Pipeline (`src/optimization/`)**:
  - `core.py`: Main orchestrator (`optimize_placement`, `_prepare_optimization_run`, `_opportunity_rank`).
  - `windowing.py`: Solver Window scanning (`find_supercharged_opportunities`, `_scan_grid_with_window`, `calculate_window_score`) and localized subgrid extraction.
  - `refinement.py`: ML placement (`_handle_ml_opportunity`) and simulated annealing / permutation refinement (`_handle_sa_refine_opportunity`, `simulated_annealing`, `refine_placement`).
- **Pulse / Photonix Core (`PC`) Remapping**: In `src/solve_map_utils.py` (`filter_solves`), when `pulse` includes the `PC` module, `solve_data` is remapped to the `photonix` solve map without mutating the `tech` parameter so `get_tech_modules` continues to fetch `pulse` modules.
- **Telemetry & Analytics (`src/routes/analytics.py`)**: Serves `/analytics/popular_data`, `/analytics/performance_data` (BigQuery primary with GA4 Data API fallback; `GA_PROPERTY_ID = 484727815`), and `/api/events`; follow the cross-repo schema contract in [`../nms_optimizer-web/docs/adr/0003-ga4-bigquery-telemetry-contract.md`](../nms_optimizer-web/docs/adr/0003-ga4-bigquery-telemetry-contract.md).
