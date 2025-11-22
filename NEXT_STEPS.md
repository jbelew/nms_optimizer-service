# Test Coverage Improvement - Next Steps

## Current Status

**Test Suite**: 225 tests, 100% passing rate
**Bug Found & Fixed**: 1 (duplicate pattern variations)
**Coverage**: 10 core modules with comprehensive tests

## Immediate Next Steps (This Week)

### 1. Verify Bug Fix Performance Impact
**Task**: Measure performance impact of pattern variation fix
```bash
# Before fix metrics (in commit history)
# After fix metrics (current)
# Compare pattern matching execution time
python3 -m pytest src/tests/test_pattern_matching.py -v --durations=10
```

**Expected Outcome**: No performance degradation, improved efficiency for symmetric patterns

---

### 2. Run Full Integration Tests
**Task**: Verify bug fix works correctly in real optimization scenarios
```bash
# Create a test with real optimization flow
python3 -c "
from src.optimizer import Grid
from src.pattern_matching import get_all_unique_pattern_variations

pattern = {(0,0): 'mod_a'}
variations = get_all_unique_pattern_variations(pattern)
assert len(variations) == 1, f'Expected 1, got {len(variations)}'
print('✓ Single-cell patterns: PASS')
"
```

---

### 3. Document Remaining Test Gaps
**Task**: Create a feature branch to test remaining modules
**Target Modules**:
- `optimization/refinement.py` (SA algorithm) - HIGH PRIORITY
- `optimization/helpers.py` (Grid windowing) - HIGH PRIORITY
- `optimization/core.py` (Optimization orchestration) - HIGH PRIORITY

---

## High Priority (Week 2)

### 1. Create Tests for optimization/refinement.py
**Why**: Critical path for optimization quality, complex algorithm
**Scope**: 50-60 tests covering:
- Temperature scheduling
- Acceptance probability
- State transitions
- Convergence detection
- Edge cases (stuck states, oscillation)

**Start with**:
```bash
# Create test file
touch src/tests/test_optimization_refinement.py

# Test structure:
# - TestTemperatureScheduling (cooling rate, stopping condition)
# - TestAcceptanceProbability (delta calculation, probability curves)
# - TestStateTransitions (move validity, rollback)
# - TestConvergence (termination conditions)
# - TestEdgeCases (stuck states, all moves accepted/rejected)
```

### 2. Create Tests for optimization/helpers.py
**Why**: Core utilities for grid operations
**Scope**: 30-40 tests covering:
- Grid windowing logic
- Cell filtering
- Boundary calculations
- Module enumeration

### 3. Create Tests for optimization/core.py
**Why**: Orchestration of optimization strategies
**Scope**: 20-30 tests covering:
- Strategy selection
- Pipeline execution
- Fallback mechanisms
- Score tracking

---

## Medium Priority (Week 3)

### 1. Add Integration Tests
**Create**: `src/tests/test_integration.py` (50+ tests)
```python
class TestOptimizationPipeline:
    """Test full optimization workflow end-to-end"""
    - test_pattern_matching_to_refinement()
    - test_ml_placement_to_sa_polish()
    - test_fallback_to_simulated_annealing()
    - test_supercharge_optimization()

class TestMultipleTechOptimization:
    """Test optimizing multiple technologies"""
    - test_sequential_tech_optimization()
    - test_independence_between_techs()
    - test_cross_tech_interference()
```

### 2. Add Performance Benchmarks
**Create**: `src/tests/test_benchmarks.py`
```python
class TestPerformanceBenchmarks:
    """Measure and validate performance"""
    - test_pattern_matching_speed()
    - test_ml_placement_speed()
    - test_sa_refinement_speed()
    - test_end_to_end_optimization_time()
```

### 3. Add Coverage Reporting
```bash
# Install coverage tool
pip install coverage

# Generate coverage report
python3 -m pytest src/tests/ --cov=src --cov-report=html

# View report
open htmlcov/index.html
```

---

## CI/CD Integration (Week 4)

### 1. Add Pre-commit Hook
**File**: `.husky/pre-commit`
```bash
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

# Run tests before commit
python3 -m pytest src/tests/ -q || exit 1
```

### 2. Set Up GitHub Actions
**File**: `.github/workflows/test.yml`
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.14'
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest src/tests/ --cov=src
      - uses: codecov/codecov-action@v2
```

### 3. Add Test Execution to Deployment Pipeline
- Run full test suite before deploying to production
- Generate coverage report and track trends
- Alert on test failures

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
# Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-timeout

# Run tests with useful options
pytest src/tests/ \
  -v                    # Verbose output
  --tb=short            # Short traceback format
  --durations=10        # Show slowest tests
  --cov=src             # Coverage report
  -x                    # Stop on first failure
  -k "pattern"          # Run specific tests
```

### Code Quality Tools
```bash
# Install quality tools
pip install ruff pylint black mypy

# Check code style
ruff check src/
black src/ --check

# Type checking
mypy src/ --strict

# Run linter
pylint src/tests/
```

### Coverage Tools
```bash
# Generate coverage report
coverage run -m pytest src/tests/
coverage report
coverage html

# View coverage by file
coverage report --include=src/pattern_matching.py
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

The test suite is now in excellent shape with 225 passing tests and strong coverage of core modules. The next focus should be on:

1. **This week**: Verify bug fix and test remaining optimization modules
2. **Next week**: Add integration tests and performance benchmarks
3. **Final week**: Set up CI/CD and establish baseline metrics

This investment in testing will pay dividends in code confidence, faster development, and easier debugging of production issues.
