# Improvement: Window Scoring for No-Solve Cases

## Problem

When no solve map exists for a ship/tech combination, `optimize_placement()` needed an improved strategy:

1. **Single module case**: Wastes computation with window scoring. Should just find best empty cell.
2. **Multi-module case**: Simple column-filling produces suboptimal placement. Should use window scoring.

**Example from logs (before fix)**:
```
2025-11-22 10:19:30,635 - INFO - No solve found for ship: 'corvette' -- tech: 'aqua'. Placing modules in empty slots.
2025-11-22 10:19:30,636 - INFO - Final Score (No Solve Map): 0.0000 (0.00% of potential 0.0000)
```

## Solution

Use window scoring to intelligently find optimal placement areas instead of naively filling the first available slots.

**Key insight**: Don't just find the best window location, then use it for initial placement via Simulated Annealing (not refinement). Refinement requires existing modules to work with, but we have an empty grid in the no-solve case.

### Implementation

Modified the "no solve available" path in `optimize_placement()` (lines 130-244) with two strategies:

#### Single-Module Path (Fast)

For single modules (no window scoring needed):

```python
if num_modules == 1:
    # Find best empty cell, preferring edges/corners
    best_cell = None
    for y in range(height):
        for x in range(width):
            if cell is active and empty:
                is_edge = (x==0 or x==width-1 or y==0 or y==height-1)
                if best_cell is None or (is_edge and not best_cell[2]):
                    best_cell = (x, y, is_edge)
    
    # Place module at selected location
    grid.set_module(best_x, best_y, module_id)
    return grid
```

**Benefits**:
- ✓ No window scoring overhead for single modules
- ✓ Prefers edges/corners to preserve central grid space
- ✓ Fast execution for common case
- ✓ Separate "Single Module" solve method name

#### Multi-Module Path (Window Scoring)

For multiple modules:

1. **Determine optimal window size** based on module count and tech/ship combo
   ```python
   num_modules = len(tech_modules)
   w, h = determine_window_dimensions(num_modules, tech, ship)
   ```

2. **Scan grid for best window positions** using _scan_grid_with_window
   ```python
   best_score_scan, best_pos_scan = _scan_grid_with_window(
       grid_for_scan.copy(), w, h, num_modules, tech, require_supercharge=False
   )
   ```

3. **Try rotated dimensions** if width ≠ height
   ```python
   if w != h:
       score_rotated, pos_rotated = _scan_grid_with_window(
           grid_for_scan.copy(), h, w, num_modules, tech, require_supercharge=False
       )
       if score_rotated > best_score_scan:
           # Use rotated dimensions
   ```

4. **Apply Simulated Annealing** for initial placement
   ```python
   solved_grid, solved_bonus = simulated_annealing(
       grid_for_sa_window, ship, modules, tech, grid,
       progress_callback=progress_callback,
       stage="no_solve_sa_initial_placement", ...
   )
   solve_method = "No Solve (Windowed SA)"
   ```
   
   Note: Uses full SA for initial placement (not refinement), since we're placing modules for the first time in a no-solve scenario.

5. **Fallback to simple placement** if no suitable window found
   ```python
   if not best_pos_scan:
       solved_grid = place_all_modules_in_empty_slots(...)
       solve_method = "Initial Placement (No Solve)"
   ```

### Code Changes

**File**: `src/optimization/core.py`  
**Lines**: 130-197  
**Changes**:
- Replaced simple linear placement with intelligent window scanning
- Added dimension rotation support
- Integrated SA refinement for optimal scoring
- Preserved fallback for edge cases

**File**: `src/tests/test_optimization.py`  
**Changes**:
- Updated `test_optimize_no_solve_map_available` to mock new window scanning
- Test now validates window scoring attempt and fallback behavior

### Benefits

1. **Better module placement** for ships without official solves
2. **Considers adjacency bonuses** through window scoring
3. **Respects ship/tech-specific dimensions** (e.g., pulse corvette 4x3)
4. **Handles grid rotation** for non-square windows
5. **Graceful fallback** if scanning finds no suitable areas

### Return Values

When a no-solve case occurs:

| Scenario | solve_method | Behavior |
|----------|--------------|----------|
| Suitable window found | `"No Solve (Windowed SA)"` | Uses SA on optimal window |
| No suitable window | `"Initial Placement (No Solve)"` | Falls back to simple placement |

### Example Log Output (After Fix)

```
INFO - No solve found for ship: 'corvette' -- tech: 'aqua'. Using window scoring to find optimal placement.
INFO - Found optimal window via scoring: 4x3 at (2, 1) with score 15.25
INFO - Final Score (No Solve Map): 12.5000 (100.00% of potential 0.0000) using method 'No Solve (Windowed SA)'
```

vs (if no suitable window):

```
INFO - No solve found for ship: 'corvette' -- tech: 'aqua'. Using window scoring to find optimal placement.
WARNING - No suitable window found via scoring. Falling back to simple placement.
INFO - Final Score (No Solve Map): 5.0000 (100.00% of potential 0.0000) using method 'Initial Placement (No Solve)'
```

## Testing

- ✅ All 374 tests passing
- ✅ test_optimize_no_solve_map_available updated and passing
- ✅ Full test suite: 11.02 seconds
- ✅ No regressions

## Related Code

- `_scan_grid_with_window()` - Finds best window positions
- `_handle_sa_refine_opportunity()` - Refines modules in window via SA
- `determine_window_dimensions()` - Calculates optimal window size
- `place_all_modules_in_empty_slots()` - Fallback simple placement

## Edge Cases Handled

1. ✅ Grid too small for calculated window → fallback to simple placement
2. ✅ No empty slots in any window → fallback to simple placement
3. ✅ Non-square windows → tries both orientations
4. ✅ Different ship/tech combos → uses tech-specific dimension rules
5. ✅ Single module → uses 1x1 window
