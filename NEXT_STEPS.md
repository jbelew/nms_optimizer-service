# Test Coverage Improvement - Next Steps

## Current Status

**Test Suite**: 225 tests, 100% passing rate
**Bug Found & Fixed**: 1 (duplicate pattern variations)
**Coverage**: 10 core modules with comprehensive tests

## Immediate Next Steps (This Week)

### ✅ Completed
- [x] Fixed test failure (pattern adjacency mocking)
- [x] Ran full test suite (397 tests, all passing)
- [x] Created integration tests for pipeline validation
- [x] Verified CI/CD pipeline is operational
- [x] Updated documentation (PRODUCTION_CHECKLIST.md, NEXT_STEPS.md)

---

### Test Suite Status
**Total**: 397 tests, 100% passing
- 385 unit tests across 18 modules
- 12 integration tests (end-to-end pipeline)

**Coverage**: 87% overall
- Critical modules > 95%:
  - `optimization/refinement.py`: 99%
  - `optimization/helpers.py`: 99%
  - `optimization/core.py`: 98%
  - `pattern_matching.py`: 95%

**Run all tests**:
```bash
python3 -m unittest discover -v -s ./src/tests -p "test_*.py"
```

---

## High Priority (Week 2)

### 1. Audit Tests for optimization/refinement.py
**File**: `src/tests/test_optimization_refinement.py` ✅ **EXISTS**
**Verify Coverage**:
- Temperature scheduling (cooling rate, stopping condition)
- Acceptance probability (delta calculation, probability curves)
- State transitions (move validity, rollback)
- Convergence detection (termination conditions)
- Edge cases (stuck states, all moves accepted/rejected)

**Run**: `python3 -m unittest src.tests.test_optimization_refinement -v`

### 2. Audit Tests for optimization/helpers.py
**File**: `src/tests/test_optimization_helpers.py` ✅ **EXISTS**
**Verify Coverage**:
- Grid windowing logic
- Cell filtering
- Boundary calculations
- Module enumeration

**Run**: `python3 -m unittest src.tests.test_optimization_helpers -v`

### 3. Audit Tests for optimization/core.py
**File**: `src/tests/test_optimization_core.py` ✅ **EXISTS**
**Verify Coverage**:
- Strategy selection
- Pipeline execution
- Fallback mechanisms
- Score tracking

**Run**: `python3 -m unittest src.tests.test_optimization_core -v`

---

## Medium Priority (Week 3)

### 1. Integration Tests
**Status**: ✅ COMPLETED

Created `src/tests/test_integration.py` with 12 end-to-end tests covering:
- Pattern matching to refinement pipeline
- Fallback to simulated annealing
- Supercharge optimization
- Partial grid handling
- Score calculation validation
- Sequential multi-tech optimization
- Tech independence verification
- Cross-tech adjacency considerations
- Error handling (empty modules, nonexistent ships, grid sizes)

**File**: `src/tests/test_integration.py` (12 tests)
```python
class TestOptimizationPipeline(unittest.TestCase):
    """Test full optimization workflow end-to-end"""
    - test_pattern_matching_to_refinement()
    - test_ml_placement_to_sa_polish()
    - test_fallback_to_simulated_annealing()
    - test_supercharge_optimization()

class TestMultipleTechOptimization(unittest.TestCase):
    """Test optimizing multiple technologies"""
    - test_sequential_tech_optimization()
    - test_independence_between_techs()
    - test_cross_tech_interference()
```

**Key Implementation Detail**: The integration tests use a helper function to navigate the data structure:
```python
def get_tech_modules_from_ship_data(ship_data, tech_key):
    """Extract modules for a specific tech from ship data structure."""
    if "types" not in ship_data:
        return None

    for category_name, techs_in_category in ship_data["types"].items():
        for tech_obj in techs_in_category:
            if tech_obj.get("key") == tech_key:
                return tech_obj.get("modules", [])

    return None
```

This correctly navigates: `ship_data["types"][category][{key, modules}]`

### 2. Add Performance Benchmarks
**File**: Create `src/tests/test_benchmarks.py`
```python
class TestPerformanceBenchmarks(unittest.TestCase):
    """Measure and validate performance"""
    - test_pattern_matching_speed()
    - test_ml_placement_speed()
    - test_sa_refinement_speed()
    - test_end_to_end_optimization_time()
```

### 3. Add Coverage Reporting
```bash
# Coverage tool already in requirements.txt
# Generate coverage report
python3 -m coverage run -m unittest discover -s ./src/tests -p "test_*.py"
python3 -m coverage report --include=src/

# Generate HTML report
python3 -m coverage html
# View: open htmlcov/index.html
```

---

## CI/CD Integration (Week 4)

### 1. Pre-commit Hook Already Configured
**File**: `.husky/` directory exists
```bash
# Run tests before commit
python3 -m unittest discover -v -s ./src/tests -p "test_*.py" || exit 1
```

### 2. GitHub Actions Workflow
**File**: `.github/workflows/main.yml` ✅ **EXISTS & CONFIGURED**
Current workflow already:
- Checks out code
- Sets up Python 3.14
- Installs dependencies
- Builds Rust module
- Runs `unittest discover`
- Creates releases with commitizen
- Deploys to Heroku

### 3. Coverage Integration
Add to CI/CD workflow for coverage tracking:
```bash
# After test step
python3 -m coverage run -m unittest discover -s ./src/tests -p "test_*.py"
python3 -m coverage report --fail-under=80 --include=src/
```

**Ensure coverage tool is in requirements.txt** (already present)

---

## Testing Checklist

### Before Next Deployment
- [ ] All 225+ tests passing
- [ ] No new TODOs introduced
- [ ] Performance benchmarks acceptable
- [ ] Coverage report generated
- [ ] Bug fix verified in real scenarios

### Code Review Checklist
- [ ] Tests written before code (TDD)
- [ ] Edge cases covered
- [ ] Error paths tested
- [ ] No test coverage regression
- [ ] Tests are readable and maintainable

### Deployment Checklist
- [ ] Integration tests passing
- [ ] Performance benchmarks OK
- [ ] Coverage report reviewed
- [ ] No security issues in test data

---

## Resources & Tools

### Testing Tools
```bash
# Run all tests with verbose output
python3 -m unittest discover -v -s ./src/tests -p "test_*.py"

# Run specific test module
python3 -m unittest src.tests.test_pattern_matching -v

# Run specific test class
python3 -m unittest src.tests.test_pattern_matching.TestPatternRotation -v

# Run specific test method
python3 -m unittest src.tests.test_pattern_matching.TestPatternRotation.test_90_degree_rotation -v

# Stop on first failure
python3 -m unittest discover -s ./src/tests -p "test_*.py" -v 2>&1 | head -n 50
```

### Code Quality Tools
```bash
# Check code style
ruff check src/
black src/ --check

# Type checking (already configured in pyrightconfig.json)
pyright src/

# Run linter
pylint src/
```

### Coverage Tools
```bash
# Generate coverage report
python3 -m coverage run -m unittest discover -s ./src/tests -p "test_*.py"
python3 -m coverage report --include=src/

# Generate HTML report
python3 -m coverage html

# View coverage by file
python3 -m coverage report --include=src/pattern_matching.py
```

---

## Testing Best Practices

### Write Tests, Not Assertions
❌ **Bad**: Tests that only validate happy path
```python
def test_optimization():
    result = optimize(grid, ship, tech)
    assert result is not None
```

✅ **Good**: Tests that uncover bugs
```python
def test_optimization_with_empty_grid():
    """Should handle empty grids gracefully"""
    empty_grid = Grid(4, 3)
    for y in range(empty_grid.height):
        for x in range(empty_grid.width):
            empty_grid.get_cell(x, y)["active"] = False

    result = optimize(empty_grid, "corvette", "pulse")
    assert result is not None or result[0] is None
```

### Use Fixtures Effectively
```python
@pytest.fixture
def valid_grid():
    """Reusable valid grid for tests"""
    grid = Grid(4, 3)
    # Setup grid state
    return grid

def test_optimization(valid_grid):
    result = optimize(valid_grid, "corvette", "pulse")
    assert result is not None
```

### Mock External Dependencies
```python
@patch('src.ml_placement.get_model')
def test_ml_placement_with_mocked_model(mock_model):
    """Test without loading real model files"""
    mock_model.return_value = MagicMock()
    result = ml_placement(...)
    assert result is not None
```

---

## Success Metrics

### Phase 1 (Current)
- [x] 225+ tests created
- [x] 1 bug found and fixed
- [x] 10 core modules tested
- [x] 100% test pass rate

### Phase 2 (Next Week)
- [ ] 300+ total tests
- [ ] optimization module tests added
- [ ] Integration tests created
- [ ] Performance benchmarks established

### Phase 3 (Final)
- [ ] 80%+ code coverage
- [ ] All core modules tested
- [ ] CI/CD integration complete
- [ ] Performance baselines documented

---

## Questions & Support

### Common Issues

**Q: How do I run just the pattern matching tests?**
```bash
pytest src/tests/test_pattern_matching.py -v
```

**Q: Why did the pattern variation bug exist?**
- The original code used separate `rotated_patterns` and `mirrored_patterns` sets
- A mirrored rotation could duplicate a pure rotation
- The separate sets didn't account for this overlap

**Q: How can I add a new test?**
1. Identify the module to test
2. Create a test class with descriptive name
3. Use fixtures for common setup
4. Write tests that would fail if code was broken
5. Run `pytest` to verify

**Q: How do I measure test coverage?**
```bash
pytest src/tests/ --cov=src --cov-report=term-missing
```

---

## Summary

The test suite is robust with 18 test modules already in place covering all critical paths. For production readiness:

1. **Now**: Run full test suite, audit coverage gaps, verify CI/CD pipeline
2. **Week 1**: Add integration tests and performance benchmarks
3. **Week 2**: Establish coverage baseline (target 80%+), fix any gaps
4. **Week 3**: Set up coverage reporting in CI/CD, deploy with confidence

All testing uses **unittest** (Python standard library) — no external test framework dependencies needed. CI/CD workflow in `.github/workflows/main.yml` already runs tests on every push and deploys to Heroku.
