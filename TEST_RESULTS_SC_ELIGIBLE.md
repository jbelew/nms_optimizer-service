# Adversarial Test Results for sc_eligible Enforcement

## Tests Run
Created 11 comprehensive adversarial tests covering edge cases for sc_eligible flag enforcement.

## Results Summary
- **Passed**: 8 tests
- **Failed**: 3 tests (found real bugs)

## Failed Tests

### 1. test_non_eligible_modules_skip_supercharged
**Status**: FAILED  
**Issue**: `place_all_modules_in_empty_slots` is STILL placing non-eligible modules in supercharged slots  
**Expected**: Non-eligible modules should skip supercharged slots and use non-supercharged slots instead  
**Actual**: Non-eligible modules are being placed in supercharged slots  
**Root Cause**: The `continue` statement in helpers.py's `place_all_modules_in_empty_slots` only skips the current iteration but doesn't guarantee finding a non-supercharged slot - if there are no non-supercharged slots, the outer loop will wrap and the module never gets placed.

### 2. test_refine_placement_rejects_invalid_permutations  
**Status**: FAILED  
**Issue**: `refine_placement` is placing non-eligible modules in supercharged slots  
**Expected**: Permutations that violate sc_eligible constraint should be rejected  
**Actual**: Invalid permutations are being accepted and scored  
**Root Cause**: The validation check I added runs BEFORE module placement but the constraint is being violated anyway. Need to investigate if the grid cells are being properly checked for supercharged status.

### 3. test_all_supercharged_slots_non_eligible
**Status**: FAILED  
**Issue**: Edge case where ALL slots are supercharged and ALL modules are non-eligible  
**Expected**: Constraint should be maintained - don't place modules  
**Actual**: Non-eligible modules are being placed in supercharged slots  
**Root Cause**: When there are NO valid placement options (all supercharged, all modules non-eligible), the current implementation falls back to violating the constraint rather than refusing to place.

## Passed Tests

### Successfully Validated:
1. ✓ `test_windowing_rejects_all_non_eligible` - Correctly returns None when all modules non-sc_eligible
2. ✓ `test_windowing_accepts_all_eligible` - Correctly finds windows when all modules sc_eligible  
3. ✓ `test_mixed_modules_placement` - Handles mix of eligible/non-eligible correctly
4. ✓ `test_no_supercharged_slots_non_eligible` - Non-eligible modules work fine with no supercharged slots
5. ✓ `test_invariant_supercharged_vs_eligible` - Skip logic is sound
6. ✓ `test_invariant_non_supercharged_always_ok` - Non-supercharged slots accept any module
7. ✓ `test_core_placement_skips_supercharged_for_non_eligible` - Logic looks correct
8. ✓ `test_eligible_modules_prefer_supercharged` - Eligible modules prioritize supercharged

## Recommendations

### Critical Fixes Needed:

1. **helpers.py `place_all_modules_in_empty_slots`**:
   - The current `continue` logic doesn't work with nested loops
   - Need a two-pass approach:
     - Pass 1: Place modules preferentially in non-supercharged slots for non-eligible modules
     - Pass 2: Place remaining eligible modules in supercharged slots

2. **refinement.py `refine_placement`**:
   - The constraint validation may not be checking grid cells properly
   - Need to verify that `grid.get_cell(x, y)["supercharged"]` is correctly populated

3. **Edge Case Handling**:
   - When ALL slots are supercharged and ALL modules are non-eligible:
     - This is an impossible constraint scenario
     - Should either: (a) return error, (b) fall back to allow it (with warning), or (c) return no placement
     - Current implementation is violating the constraint - need explicit policy

## Test Coverage Notes

The adversarial tests successfully expose real implementation bugs that would cause:
- Module bonuses to be miscalculated (non-eligible in supercharged means incorrect scoring)
- Game mechanic violations (supercharged penalties for non-eligible modules)
- Potential server-side inconsistencies with client-side validation

These tests should be kept and run as part of regression testing.
