import rust_scorer

try:
    from rust_scorer import populate_all_module_bonuses, calculate_grid_score

    print("Successfully imported populate_all_module_bonuses and calculate_grid_score")
except ImportError as e:
    print(f"ImportError: {e}")

print(f"dir(rust_scorer): {dir(rust_scorer)}")
