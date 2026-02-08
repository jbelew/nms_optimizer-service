from src.grid_utils import Grid
from src.optimization.refinement import simulated_annealing


def test_reproduce_panic():
    # Setup a minimal grid
    grid = Grid(width=5, height=5)

    # Add a cell with greater_1 adjacency
    cell = grid.get_cell(0, 0)
    cell.update(
        {
            "module": "M1",
            "label": "Test Module",
            "tech": "jetpack",
            "type": "upgrade",
            "bonus": 0.3,
            "adjacency": "greater_1",
            "sc_eligible": True,
            "active": True,
            "supercharged": False,
            "image": None,
            "value": 0,
            "total": 0.0,
            "adjacency_bonus": 0.0,
        }
    )

    ship = "exosuit"
    tech = "jetpack"

    # Correct structure for modules data
    modules_data = {
        "types": {
            "Movement Systems": [
                {
                    "key": tech,
                    "label": "Jetpack",
                    "modules": [
                        {
                            "id": "M1",
                            "label": "Test Module",
                            "type": "upgrade",
                            "bonus": 0.3,
                            "adjacency": "greater_1",
                            "sc_eligible": True,
                            "image": None,
                        },
                        {
                            "id": "M2",
                            "label": "Test Module 2",
                            "type": "upgrade",
                            "bonus": 0.3,
                            "adjacency": "greater_1",
                            "sc_eligible": True,
                            "image": None,
                        },
                        {
                            "id": "M3",
                            "label": "Test Module 3",
                            "type": "upgrade",
                            "bonus": 0.3,
                            "adjacency": "greater_1",
                            "sc_eligible": True,
                            "image": None,
                        },
                        {
                            "id": "M4",
                            "label": "Test Module 4",
                            "type": "upgrade",
                            "bonus": 0.3,
                            "adjacency": "greater_1",
                            "sc_eligible": True,
                            "image": None,
                        },
                        {
                            "id": "M5",
                            "label": "Test Module 5",
                            "type": "upgrade",
                            "bonus": 0.3,
                            "adjacency": "greater_1",
                            "sc_eligible": True,
                            "image": None,
                        },
                        {
                            "id": "M6",
                            "label": "Test Module 6",
                            "type": "upgrade",
                            "bonus": 0.3,
                            "adjacency": "greater_1",
                            "sc_eligible": True,
                            "image": None,
                        },
                    ],
                }
            ]
        }
    }

    full_grid = grid.copy()

    print("Running simulated_annealing which should trigger the error...")
    try:
        # Increase number of modules to >= 6 to trigger SA path
        simulated_annealing(
            grid=grid,
            ship=ship,
            modules=modules_data,
            tech=tech,
            full_grid=full_grid,
            num_sa_runs=1,
            iterations_per_temp=1,
        )
        print("Success! (Wait, this should have failed...)")
    except KeyError as e:
        print(f"Caught expected Python KeyError: {e}")
    except Exception as e:
        print(f"Caught expected exception: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_reproduce_panic()
