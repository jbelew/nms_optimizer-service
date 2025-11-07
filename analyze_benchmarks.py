import os
import re
from rich.console import Console
from rich.table import Table
import argparse


def parse_benchmark_log(file_path):
    """Parses a single benchmark log file."""
    params = {}
    results = {}
    with open(file_path, "r") as f:
        content = f.read()

    # Using regex to be more robust
    param_matches = re.findall(r"  (\w+): ([\d\.]+)", content)
    for key, value in param_matches:
        try:
            if "." in value:
                params[key] = float(value)
            else:
                params[key] = int(value)
        except ValueError:
            params[key] = value

    mean_score_match = re.search(r"Mean Score: ([\d\.]+)", content)
    if mean_score_match:
        results["Mean Score"] = float(mean_score_match.group(1))

    median_score_match = re.search(r"Median Score: ([\d\.]+)", content)
    if median_score_match:
        results["Median Score"] = float(median_score_match.group(1))

    best_score_match = re.search(r"Best Score: ([\d\.]+)", content)
    if best_score_match:
        results["Best Score"] = float(best_score_match.group(1))

    std_dev_match = re.search(r"Standard Deviation: ([\d\.]+)", content)
    if std_dev_match:
        results["Standard Deviation"] = float(std_dev_match.group(1))

    avg_time_match = re.search(r"Average Run Time: ([\d\.]+)s", content)
    if avg_time_match:
        results["Average Run Time"] = float(avg_time_match.group(1))

    return {"params": params, "results": results, "file": os.path.basename(file_path)}


def analyze_benchmarks(directory):
    """Analyzes all benchmark logs in a directory."""
    all_results = []
    for filename in sorted(os.listdir(directory)):
        if filename.startswith("experiment_") and filename.endswith(".log"):
            file_path = os.path.join(directory, filename)
            parsed_data = parse_benchmark_log(file_path)
            if parsed_data["results"]:  # Only include files with results
                all_results.append(parsed_data)
    return all_results


def print_results(results):
    """Prints the analysis results in a formatted table."""
    # Sort by a composite score: mean_score / (std_dev + 0.0001)
    # This rewards high scores and penalizes high variance.
    # Add a small epsilon to std_dev to avoid division by zero.
    sorted_results = sorted(
        results,
        key=lambda x: x["results"].get("Mean Score", 0) / (x["results"].get("Standard Deviation", 999) + 0.0001),
        reverse=True,
    )

    table = Table(title="Benchmark Analysis", show_header=True, header_style="bold magenta")
    table.add_column("Experiment", style="dim", width=15)
    table.add_column("Mean Score", justify="right")
    table.add_column("Best Score", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Composite Score", justify="right")
    table.add_column("Parameters")

    console = Console()

    for res in sorted_results:
        params_str = ", ".join(f"{k.replace('_', ' ').title()}={v}" for k, v in res["params"].items())
        mean_score = res["results"].get("Mean Score", 0)
        std_dev = res["results"].get("Standard Deviation", 999)
        composite_score = mean_score / (std_dev + 0.0001)

        table.add_row(
            res["file"],
            f"{mean_score:.4f}",
            f"{res['results'].get('Best Score', 'N/A'):.4f}",
            f"{std_dev:.4f}",
            f"{res['results'].get('Average Run Time', 'N/A'):.2f}",
            f"{composite_score:.2f}",
            params_str,
        )

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark logs.")
    parser.add_argument("directory", type=str, help="Path to the directory containing benchmark log files.")
    args = parser.parse_args()

    if os.path.isdir(args.directory):
        results = analyze_benchmarks(args.directory)
        if results:
            print_results(results)
        else:
            print(f"No benchmark results found in the directory: {args.directory}")
    else:
        print(f"Directory not found: {args.directory}")
