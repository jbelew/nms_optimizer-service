import subprocess
import os
import argparse
import sys

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
    # Experiment 22: Tweak of exp 13
    {
        "initial_temperature": 7000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 75,
        "max_steps_without_improvement": 200,
        "num_sa_runs": 4,
    },
    # Experiment 23: Tweak of exp 13
    {
        "initial_temperature": 7000,
        "cooling_rate": 0.99,
        "iterations_per_temp": 75,
        "max_steps_without_improvement": 200,
        "num_sa_runs": 5,
    },
    # Experiment 24: Higher initial swap probability
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.75,
        "final_swap_probability": 0.25,
    },
    # Experiment 25: Lower initial swap probability
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.35,
        "final_swap_probability": 0.25,
    },
    # Experiment 26: Higher final swap probability
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.55,
        "final_swap_probability": 0.45,
    },
    # Experiment 27: Lower final swap probability
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.55,
        "final_swap_probability": 0.05,
    },
    # Experiment 28: Both higher
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.75,
        "final_swap_probability": 0.45,
    },
    # Experiment 29: Both lower
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 35,
        "max_steps_without_improvement": 150,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.35,
        "final_swap_probability": 0.05,
    },
    # Experiment 30: Try to reduce std dev
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 70,
        "max_steps_without_improvement": 300,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.75,
        "final_swap_probability": 0.25,
    },
    # Experiment 31: Speed up exp 30
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 60,
        "max_steps_without_improvement": 250,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.75,
        "final_swap_probability": 0.25,
    },
    # Experiment 32: Tune cooling rate of exp 30
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.995,
        "iterations_per_temp": 70,
        "max_steps_without_improvement": 300,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.75,
        "final_swap_probability": 0.25,
    },
    # Experiment 33: Tune cooling rate of exp 30
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.99,
        "iterations_per_temp": 70,
        "max_steps_without_improvement": 300,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.75,
        "final_swap_probability": 0.25,
    },
    # Experiment 34: exp 30 with 1 sa run
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.999,
        "iterations_per_temp": 70,
        "max_steps_without_improvement": 300,
        "num_sa_runs": 1,
        "initial_swap_probability": 0.75,
        "final_swap_probability": 0.25,
    },
    # Experiment 35: Try to reduce std dev
    {
        "initial_temperature": 3000,
        "cooling_rate": 0.998,
        "iterations_per_temp": 70,
        "max_steps_without_improvement": 300,
        "num_sa_runs": 2,
        "initial_swap_probability": 0.75,
        "final_swap_probability": 0.25,
    },
]


def run_experiments(start_index, end_index):
    output_dir = "benchmark_results_swap_exp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(start_index, end_index):
        if i >= len(experiments):
            break

        params = experiments[i]
        experiment_name = f"experiment_{i+1}"
        output_file = os.path.join(output_dir, f"{experiment_name}.log")

        command = [sys.executable, "scripts/training/benchmark.py", "--num_runs=20", "--max_processing_time=60.0"]

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
