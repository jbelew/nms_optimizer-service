# Pyright Type Checking Improvements

## Current Status
- **ruff**: Fully fixed (all checks pass)
- **pyright**: 46 errors identified, not yet fixed
- **Tests**: All 397 tests pass with reverted code

## Problem Statement
We have two different `Grid` types that cause type checking issues:

1. **`grid_utils.Grid`** (Python class)
   - Constructor: `Grid(width, height)`
   - Builds cells internally
   - Used throughout the codebase

2. **`rust_scorer.Grid`** (PyO3 binding)
   - Constructor: `Grid(width, height, cells)`
   - Requires pre-built cells list
   - Used by Rust scoring functions

This dual-Grid situation is the root cause of most pyright errors.

## Key Issues to Address

### 1. Grid Type Conversion (HIGH PRIORITY)
**Problem**: `calculate_grid_score()` and other functions need to convert `grid_utils.Grid` → `rust_scorer.Grid`

**Solution**: Create conversion functions in `bonus_calculations.py`:
```python
def python_grid_to_rust_grid(python_grid: Grid) -> RustGrid:
    """Convert grid_utils.Grid to rust_scorer.Grid"""
    # Build rust cells from python grid
    # Return RustGrid(width, height, cells)
```

**Files affected**:
- `src/bonus_calculations.py` - add conversion function
- `src/optimization/training.py` - use converter in rust_calculate_grid_score calls
- All places calling `rust_calculate_grid_score` must convert the grid first

### 2. Return Type Hints
**Problem**: Functions like `_prepare_optimization_run()` have inconsistent return types

**Solution**: Add explicit return type hints:
```python
def _prepare_optimization_run(
    grid: Grid,
    modules: dict,
    ship: str,
    tech: str,
    available_modules: list[str] | None,
) -> tuple[list[dict], list[dict]] | tuple[Grid, float, float, str]:
    """
    Returns either success tuple or error tuple
    """
```

**Files affected**:
- `src/optimization/core.py` - `_prepare_optimization_run()`
- `src/optimization/training.py` - `refine_placement_for_training()`
- Other core functions with complex returns

### 3. Rust Module Type Stubs (OPTIONAL)
**Problem**: PyO3-generated bindings don't have proper type information

**Solution**: Create `.pyi` stub files or suppress errors:
- Create `rust_scorer.pyi` with proper type signatures
- OR use `# type: ignore` on Rust function calls

**Files affected**:
- `rust_scorer/` - create stub files
- `src/optimization/refinement.py` - suppress `simulated_annealing` call errors

### 4. Test File Type Hints
**Problem**: Test files have Optional type issues

**Solution**: Add proper type guards:
```python
# Before:
result = find_supercharged_opportunities(...)
self.assertEqual(len(result), 4)

# After:
result = find_supercharged_opportunities(...)
self.assertIsNotNone(result)
if result is not None:
    self.assertEqual(len(result), 4)
```

**Files affected**:
- `src/tests/test_optimization_core.py`
- `src/tests/test_optimization.py`
- `src/tests/test_optimization_windowing.py`
- `src/tests/test_optimization_refinement.py`
- `src/tests/test_pattern_matching.py`

## Implementation Plan

### Phase 1: Grid Conversion (MUST DO)
1. [ ] Add `python_grid_to_rust_grid()` to `bonus_calculations.py`
2. [ ] Add conversion helper functions for Rust types
3. [ ] Update all `rust_calculate_grid_score()` calls to convert grid
4. [ ] Update `calculate_grid_score()` to use conversion function
5. [ ] Test all bonus calculation tests pass

### Phase 2: Return Type Hints
1. [ ] Add type hints to `_prepare_optimization_run()`
2. [ ] Add type hints to `optimize_placement()`
3. [ ] Add type hints to `refine_placement_for_training()`
4. [ ] Verify no type conflicts introduced

### Phase 3: Test File Guards
1. [ ] Add type guards to test files
2. [ ] Suppress unavoidable Rust binding errors with `# type: ignore`
3. [ ] Run pyright - target: <30 errors

### Phase 4: Rust Stubs (OPTIONAL)
1. [ ] Evaluate if worth creating `.pyi` files
2. [ ] OR accept remaining Rust-related errors with suppressions

## Testing Strategy
1. Run `python -m unittest discover` after each phase
2. Run `ruff check .` to ensure no regressions
3. Run `pyright src/` to track error reduction

## Notes for Next Session
- Do NOT try to be clever with type conversions - keep it explicit and simple
- The Rust type system is PyO3-generated; perfect type safety may be impossible
- Focus on Python code quality first, accept Rust binding limitations
- All tests MUST pass before committing changes
- The pre-commit hook is now configured to run tests - this is good!
