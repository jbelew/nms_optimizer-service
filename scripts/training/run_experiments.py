import subprocess
import os
import argparse

# Define the parameter combinations to test
experiments = [
    # Baseline
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
    },
    # Experiment 1: Higher initial temperature
    {
        "initial_temperature": 5000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
    },
    # Experiment 2: Even higher initial temperature
    {
        "initial_temperature": 7000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
    },
    # Experiment 3: Slower cooling
    {
        "initial_temperature": 5000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
    },
    # Experiment 4: Even slower cooling
    {
        "initial_temperature": 5000,
        "cooling_rate": 0.99,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
    },
    # Experiment 5: More iterations per temp
    {
        "initial_temperature": 5000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 50,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
    },
    # Experiment 6: Even more iterations per temp
    {
        "initial_temperature": 5000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 75,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
    },
    # Experiment 7: More steps without improvement
    {
        "initial_temperature": 5000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 50,
        "max_steps_without_improvement": 200,
        "num_sa_runs": 2,
    },
    # Experiment 8: Even more steps without improvement
    {
        "initial_temperature": 5000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 50,
        "max_steps_without_improvement": 250,
        "num_sa_runs": 2,
    },
    # Experiment 9: More SA runs
    {
        "initial_temperature": 5000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 50,
        "max_steps_without_improvement": 200,
        "num_sa_runs": 4,
    },
    # Experiment 10: Even more SA runs
    {
        "initial_temperature": 5000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 50,
        "max_steps_without_improvement": 200,
        "num_sa_runs": 6,
    },
    # Combinations
    # High temp, slow cooling
    {
        "initial_temperature": 7000,
        "cooling_rate": 0.99,
        "iterations_per_temp": 50,
        "max_steps_without_improvement": 200,
        "num_sa_runs": 4,
    },
    # High temp, slow cooling, more iterations
    {
        "initial_temperature": 7000,
        "cooling_rate": 0.99,
        "iterations_per_temp": 75,
        "max_steps_without_improvement": 200,
        "num_sa_runs": 4,
    },
    # High temp, very slow cooling
    {
        "initial_temperature": 9000,
        "cooling_rate": 0.99,
        "iterations_per_temp": 50,
        "max_steps_without_improvement": 200,
        "num_sa_runs": 4,
    },
    # My previous "best"
    {
        "initial_temperature": 6000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 50,
        "max_steps_without_improvement": 200,
        "num_sa_runs": 4,
    },
    # More exploration
    {
        "initial_temperature": 9000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 6,
    },
    {
        "initial_temperature": 7000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 75,
        "max_steps_without_improvement": 250,
        "num_sa_runs": 4,
    },
    {
        "initial_temperature": 5000,
        "cooling_rate": 0.99,
        "iterations_per_temp": 75,
        "max_steps_without_improvement": 250,
        "num_sa_runs": 6,
    },
    {
        "initial_temperature": 9000,
        "cooling_rate": 0.99,
        "iterations_per_temp": 75,
        "max_steps_without_improvement": 250,
        "num_sa_runs": 6,
    },
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.99,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
    },
    {
        "initial_temperature": 9000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 75,
        "max_steps_without_improvement": 250,
        "num_sa_runs": 6,
    },
]


def run_experiments(start_index, end_index):
    output_dir = "benchmark_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(start_index, end_index):
        if i >= len(experiments):
            break

        params = experiments[i]
        experiment_name = f"experiment_{i+1}"
        output_file = os.path.join(output_dir, f"{experiment_name}.log")

        command = ["python3", "scripts/training/benchmark.py", "--num_runs=20", "--max_processing_time=20.0"]

        for key, value in params.items():
            command.append(f"--{key}={value}")

        print(f"--- Running {experiment_name} ---")
        print(f"Parameters: {params}")
        print(f"Outputting to: {output_file}")

        with open(output_file, "w") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)

        print(f"--- {experiment_name} complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark experiments in batches.")
    parser.add_argument("--start", type=int, required=True, help="Starting experiment index.")
    parser.add_argument("--end", type=int, required=True, help="Ending experiment index (exclusive).")
    args = parser.parse_args()

    run_experiments(args.start, args.end)
