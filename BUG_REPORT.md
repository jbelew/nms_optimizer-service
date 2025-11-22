# Bug Report: Pattern Variation Duplication

## Summary
Found and fixed a bug in `src/pattern_matching.py` where the `get_all_unique_pattern_variations()` function generates duplicate pattern variations for symmetric patterns.

## Bug Details

**Function:** `get_all_unique_pattern_variations()` (lines 207-251)

**Issue:** Duplicate pattern variations are generated and returned when a pattern is symmetric.

### Examples of the Bug:

1. **Single-cell patterns**: Expected 1 variation, got 2
   ```python
   pattern = {(0, 0): "module_a"}
   variations = get_all_unique_pattern_variations(pattern)
   # Before fix: 2 variations (original + duplicate mirror)
   # After fix: 1 variation
   ```

2. **Uniform 2×2 square**: Expected 1 variation, got 3
   ```python
   pattern = {(0, 0): "A", (1, 0): "A", (0, 1): "A", (1, 1): "A"}
   variations = get_all_unique_pattern_variations(pattern)
   # Before fix: 3 variations (duplicates of symmetric rotations/mirrors)
   # After fix: 1 variation
   ```

## Root Cause

The original implementation used separate tracking sets for rotations and mirrors:
- `rotated_patterns` set - tracked rotations
- `mirrored_patterns` set - tracked mirrors

However, **mirrors of rotations could be identical to other rotations**, but the code didn't check for this overlap. A mirrored pattern was only validated against other mirrored patterns, not against rotations already in the list.

## Impact

- **Performance:** Wasted computation checking duplicate patterns during pattern matching
- **Correctness:** For symmetric patterns (most common in NMS), optimization may explore redundant variations
- **Memory:** Unnecessary list growth for common patterns

## Solution Implemented

Replaced the dual-set approach with a **single unified set** that tracks all unique patterns regardless of whether they came from rotation or mirroring.

**Key changes:**
1. Use single `unique_patterns` set with sorted tuple representation for comparison
2. Add all rotations first (0°, 90°, 180°, 270°)
3. Then add all mirrors of each rotation
4. Helper function `add_unique_pattern()` ensures each pattern is added only once

## Testing

### Tests Created:
- **42 new tests** for pattern_matching.py covering:
  - Pattern rotation (6 tests)
  - Pattern mirroring (8 tests)
  - Variation generation (6 tests) ← **Found the bug here**
  - Pattern application (9 tests)
  - Adjacency scoring (6 tests)
  - Pattern extraction (4 tests)
  - Edge cases (3 tests)

### Test Results:
- Before fix: **2 failures** in variation generation tests
- After fix: **All 144 tests pass** ✅

### Related Tests:
- `test_single_cell_has_one_variation()` - Now passes
- `test_square_pattern_symmetry()` - Now passes
- `test_no_duplicate_variations()` - Validates deduplication works

## Files Modified

1. **src/pattern_matching.py** - Fixed `get_all_unique_pattern_variations()` function
2. **src/tests/test_pattern_matching.py** - New comprehensive test suite (103 lines, 42 tests)

## Verification

Run the tests to verify the fix:
```bash
source venv/bin/activate
python3 -m pytest src/tests/test_pattern_matching.py::TestPatternVariationGeneration -v
```

All tests now pass:
```
test_single_cell_has_one_variation PASSED
test_square_pattern_symmetry PASSED
test_2x1_line_has_multiple_variations PASSED
test_all_variations_preserve_module_count PASSED
test_no_duplicate_variations PASSED
test_variations_always_include_original PASSED
```

## Recommendations

1. ✅ **DONE** - Fixed the duplicate variation bug
2. Add more adversarial tests for edge cases
3. Create tests for ml_placement.py (complex ML integration)
4. Add integration tests for full optimization pipeline
5. Set up continuous testing in CI/CD pipeline
