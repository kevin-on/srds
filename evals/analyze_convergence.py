#!/usr/bin/env python3
"""
Analyze iteration-by-iteration convergence speed from experiments.
Show how convergence values change across iterations for each experiment.
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_experiment_name(dir_path):
    """Extract experiment configuration from directory name"""
    dir_name = os.path.basename(dir_path)

    # Extract adaptive parameter (ad0, ad2, ad4, etc.)
    match = re.search(r"ad(\d+)", dir_name)
    if match:
        return f"ad{match.group(1)}"

    # Extract sequential steps (sequential_ddim50, sequential_ddim100, etc.)
    match = re.search(r"sequential_ddim(\d+)", dir_name)
    if match:
        return f"sequential_{match.group(1)}"

    # Extract other algorithm types
    if "srds" in dir_name:
        return "srds"
    elif "sparareal" in dir_name:
        return "sparareal"
    elif "sparatts" in dir_name:
        return "sparatts"

    return "unknown"


def extract_legend_name(dir_path):
    """Extract legend name from directory name (everything after timestamp)"""
    dir_name = os.path.basename(dir_path)

    # Remove timestamp pattern (YYYYMMDD_HHMMSS_)
    match = re.match(r"\d{8}_\d{6}_(.+)", dir_name)
    if match:
        return match.group(1)

    return dir_name


def get_iteration_convergence_data(csv_file_path):
    """Get convergence data for all iterations from convergence table"""
    try:
        df = pd.read_csv(csv_file_path, index_col=0)

        # Get the last column (final timestep) - this represents final convergence
        final_timestep_col = df.columns[-1]
        iteration_values = df[final_timestep_col].values

        return iteration_values

    except Exception as e:
        print(f"Error reading {csv_file_path}: {e}")
        return None


def get_gt_trajectory_errors(csv_file_path):
    """Get GT trajectory errors for all iterations from GT errors table"""
    try:
        df = pd.read_csv(csv_file_path, index_col=0)

        # Get the last column (final timestep) - this represents final GT error
        final_timestep_col = df.columns[-1]
        iteration_values = df[final_timestep_col].values

        return iteration_values

    except Exception as e:
        print(f"Error reading {csv_file_path}: {e}")
        return None


def extract_l1_and_mse_from_log(log_file_path):
    """Extract L1 distances and MSE from log.txt file"""
    l1_distances = []
    mse_values = []

    try:
        with open(log_file_path) as f:
            content = f.read()

        # Find all L1 distance entries
        # Pattern: "SRDS Iteration X: L1=Y.YYYYYY (tolerance=0.1) - continuing"
        l1_pattern = r"SRDS Iteration (\d+): L1=([\d.]+) \(tolerance=0\.1\) - continuing"
        l1_matches = re.findall(l1_pattern, content)

        for iteration, l1_dist in l1_matches:
            l1_distances.append((int(iteration), float(l1_dist)))

        # Find MSE entries if they exist
        # Pattern: "SRDS Iteration X: MSE=Y.YYYYYY (tolerance=0.1) - continuing"
        mse_pattern = r"SRDS Iteration (\d+): MSE=([\d.]+) \(tolerance=0\.1\) - continuing"
        mse_matches = re.findall(mse_pattern, content)

        for iteration, mse_val in mse_matches:
            mse_values.append((int(iteration), float(mse_val)))

    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")

    return l1_distances, mse_values


def analyze_experiment_iterations(exp_dir):
    """Analyze iteration convergence for a single experiment"""
    exp_name = extract_experiment_name(exp_dir)

    # Find all prompt directories
    prompt_dirs = glob.glob(os.path.join(exp_dir, "prompt*"))
    prompt_dirs.sort()

    all_convergence_data = []
    all_l1_data = []
    all_mse_data = []
    all_gt_error_data = []

    print(f"\nAnalyzing experiment: {exp_name}")
    print(f"Found {len(prompt_dirs)} prompts")

    for prompt_dir in prompt_dirs:
        prompt_name = os.path.basename(prompt_dir)

        # Get convergence table
        convergence_csv = os.path.join(prompt_dir, "_trajectory_convergences_table.csv")
        if os.path.exists(convergence_csv):
            convergence_values = get_iteration_convergence_data(convergence_csv)
            if convergence_values is not None:
                all_convergence_data.append(convergence_values)
                print(f"  {prompt_name}: {len(convergence_values)} iterations")

        # Get GT trajectory errors table
        gt_errors_csv = os.path.join(prompt_dir, "_gt_trajectory_errors_table.csv")
        if os.path.exists(gt_errors_csv):
            gt_error_values = get_gt_trajectory_errors(gt_errors_csv)
            if gt_error_values is not None:
                all_gt_error_data.append(gt_error_values)
                print(f"  {prompt_name}: {len(gt_error_values)} GT error iterations")

        # Get L1 and MSE from log
        log_file = os.path.join(prompt_dir, "log.txt")
        if os.path.exists(log_file):
            l1_distances, mse_values = extract_l1_and_mse_from_log(log_file)
            if l1_distances:
                l1_vals = [l1_dist for _, l1_dist in l1_distances]
                all_l1_data.append(l1_vals)
            if mse_values:
                mse_vals = [mse_val for _, mse_val in mse_values]
                all_mse_data.append(mse_vals)

    # Process convergence data
    convergence_result = None
    if all_convergence_data:
        max_iterations = max(len(data) for data in all_convergence_data)
        padded_data = []
        for data in all_convergence_data:
            padded = np.full(max_iterations, np.nan)
            padded[: len(data)] = data
            padded_data.append(padded)

        iteration_means = np.nanmean(padded_data, axis=0)
        iteration_stds = np.nanstd(padded_data, axis=0)
        convergence_result = (iteration_means, iteration_stds, max_iterations)

    # Process L1 data
    l1_result = None
    if all_l1_data:
        max_iterations_l1 = max(len(data) for data in all_l1_data)
        padded_l1_data = []
        for data in all_l1_data:
            padded = np.full(max_iterations_l1, np.nan)
            padded[: len(data)] = data
            padded_l1_data.append(padded)

        l1_means = np.nanmean(padded_l1_data, axis=0)
        l1_stds = np.nanstd(padded_l1_data, axis=0)
        l1_result = (l1_means, l1_stds, max_iterations_l1)

    # Process MSE data
    mse_result = None
    if all_mse_data:
        max_iterations_mse = max(len(data) for data in all_mse_data)
        padded_mse_data = []
        for data in all_mse_data:
            padded = np.full(max_iterations_mse, np.nan)
            padded[: len(data)] = data
            padded_mse_data.append(padded)

        mse_means = np.nanmean(padded_mse_data, axis=0)
        mse_stds = np.nanstd(padded_mse_data, axis=0)
        mse_result = (mse_means, mse_stds, max_iterations_mse)

    # Process GT error data
    gt_error_result = None
    if all_gt_error_data:
        max_iterations_gt = max(len(data) for data in all_gt_error_data)
        padded_gt_data = []
        for data in all_gt_error_data:
            padded = np.full(max_iterations_gt, np.nan)
            padded[: len(data)] = data
            padded_gt_data.append(padded)

        gt_error_means = np.nanmean(padded_gt_data, axis=0)
        gt_error_stds = np.nanstd(padded_gt_data, axis=0)
        gt_error_result = (gt_error_means, gt_error_stds, max_iterations_gt)

    return exp_name, convergence_result, l1_result, mse_result, gt_error_result


def main():
    parser = argparse.ArgumentParser(description="Analyze iteration convergence from experiments")
    parser.add_argument(
        "--folder", type=str, required=True, help="Base folder containing experiment results"
    )
    args = parser.parse_args()

    base_dir = args.folder

    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(base_dir, "2025*"))
    exp_dirs.sort()

    print(f"Found {len(exp_dirs)} experiments in {base_dir}")

    results = []
    for exp_dir in exp_dirs:
        exp_name, convergence_result, l1_result, mse_result, gt_error_result = (
            analyze_experiment_iterations(exp_dir)
        )
        if (
            convergence_result is not None
            or l1_result is not None
            or mse_result is not None
            or gt_error_result is not None
        ):
            results.append((exp_name, convergence_result, l1_result, mse_result, gt_error_result))

    # Create plots
    if results:
        # Create figure with 3 subplots (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

        # Plot 1: Convergence curves (left)
        ax1 = axes[0]
        for i, (exp_name, convergence_result, _, _, _) in enumerate(results):
            if convergence_result is not None:
                means, stds, max_iter = convergence_result
                iterations = np.arange(1, max_iter + 1)
                valid_mask = ~np.isnan(means)
                valid_iterations = iterations[valid_mask]
                valid_means = means[valid_mask]
                valid_stds = stds[valid_mask]

                # Get legend name from exp_dir
                exp_dir = exp_dirs[i]
                legend_name = extract_legend_name(exp_dir)

                ax1.plot(
                    valid_iterations,
                    valid_means,
                    "o-",
                    label=f"{legend_name}",
                    color=colors[i],
                    linewidth=2,
                    markersize=4,
                )

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Final Step Convergence Value")
        ax1.set_title("Convergence Speed Across Iterations (Log Scale)")
        ax1.legend(loc="best", fontsize=9, framealpha=0.8)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")  # Log scale

        # Plot 2: L1 distances (right)
        ax2 = axes[1]
        for i, (exp_name, _, l1_result, _, _) in enumerate(results):
            if l1_result is not None:
                means, stds, max_iter = l1_result
                iterations = np.arange(1, max_iter + 1)
                valid_mask = ~np.isnan(means)
                valid_iterations = iterations[valid_mask]
                valid_means = means[valid_mask]
                valid_stds = stds[valid_mask]

                # Get legend name from exp_dir
                exp_dir = exp_dirs[i]
                legend_name = extract_legend_name(exp_dir)

                ax2.plot(
                    valid_iterations,
                    valid_means,
                    "o-",
                    label=f"{legend_name}",
                    color=colors[i],
                    linewidth=2,
                    markersize=4,
                )

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("L1 Distance")
        ax2.set_title("L1 Distance Across Iterations (Log Scale)")
        ax2.legend(loc="best", fontsize=9, framealpha=0.8)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")  # Log scale

        # Plot 3: GT Error curves (right)
        ax3 = axes[2]
        for i, (exp_name, _, _, _, gt_error_result) in enumerate(results):
            if gt_error_result is not None:
                means, stds, max_iter = gt_error_result
                iterations = np.arange(1, max_iter + 1)
                valid_mask = ~np.isnan(means)
                valid_iterations = iterations[valid_mask]
                valid_means = means[valid_mask]
                valid_stds = stds[valid_mask]

                # Get legend name from exp_dir
                exp_dir = exp_dirs[i]
                legend_name = extract_legend_name(exp_dir)

                ax3.plot(
                    valid_iterations,
                    valid_means,
                    "o-",
                    label=f"{legend_name}",
                    color=colors[i],
                    linewidth=2,
                    markersize=4,
                )

        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("GT Error (Final Step)")
        ax3.set_title("GT Error Across Iterations (Log Scale)")
        ax3.legend(loc="best", fontsize=9, framealpha=0.8)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")  # Log scale

        plt.tight_layout()

        # Save with folder name in the specified folder
        folder_name = os.path.basename(base_dir)
        output_file = os.path.join(base_dir, f"convergence_analysis_{folder_name}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nGraph saved as: {output_file}")
        plt.show()

        # Prepare summary data for CSV export
        summary_data = []

        # Print convergence speed analysis
        print("\n" + "=" * 80)
        print("CONVERGENCE SPEED ANALYSIS")
        print("=" * 80)

        for i, (exp_name, convergence_result, l1_result, mse_result, gt_error_result) in enumerate(
            results
        ):
            exp_dir = exp_dirs[i]
            legend_name = extract_legend_name(exp_dir)

            print(f"\n{legend_name}:")

            # Initialize summary row
            summary_row = {"experiment": legend_name, "exp_name": exp_name}

            if convergence_result is not None:
                means, _, _ = convergence_result
                valid_means = means[~np.isnan(means)]
                if len(valid_means) >= 2:
                    total_improvement = valid_means[0] - valid_means[-1]
                    avg_improvement_per_iter = total_improvement / (len(valid_means) - 1)
                    print(
                        f"  Convergence - Initial: {valid_means[0]:.4f}, Final: {valid_means[-1]:.4f}"
                    )
                    print(f"  Convergence - Total improvement: {total_improvement:.4f}")
                    print(f"  Convergence - Avg improvement/iter: {avg_improvement_per_iter:.4f}")

                    summary_row["convergence_initial"] = valid_means[0]
                    summary_row["convergence_final"] = valid_means[-1]
                    summary_row["convergence_total_improvement"] = total_improvement
                    summary_row["convergence_avg_improvement_per_iter"] = avg_improvement_per_iter
                    summary_row["convergence_iterations"] = len(valid_means)

            if l1_result is not None:
                means, _, _ = l1_result
                valid_means = means[~np.isnan(means)]
                if len(valid_means) >= 2:
                    total_improvement = valid_means[0] - valid_means[-1]
                    avg_improvement_per_iter = total_improvement / (len(valid_means) - 1)
                    print(
                        f"  L1 Distance - Initial: {valid_means[0]:.4f}, Final: {valid_means[-1]:.4f}"
                    )
                    print(f"  L1 Distance - Total improvement: {total_improvement:.4f}")
                    print(f"  L1 Distance - Avg improvement/iter: {avg_improvement_per_iter:.4f}")

                    summary_row["l1_initial"] = valid_means[0]
                    summary_row["l1_final"] = valid_means[-1]
                    summary_row["l1_total_improvement"] = total_improvement
                    summary_row["l1_avg_improvement_per_iter"] = avg_improvement_per_iter
                    summary_row["l1_iterations"] = len(valid_means)

            if mse_result is not None:
                means, _, _ = mse_result
                valid_means = means[~np.isnan(means)]
                if len(valid_means) >= 2:
                    total_improvement = valid_means[0] - valid_means[-1]
                    avg_improvement_per_iter = total_improvement / (len(valid_means) - 1)
                    print(f"  MSE - Initial: {valid_means[0]:.4f}, Final: {valid_means[-1]:.4f}")
                    print(f"  MSE - Total improvement: {total_improvement:.4f}")
                    print(f"  MSE - Avg improvement/iter: {avg_improvement_per_iter:.4f}")

                    summary_row["mse_initial"] = valid_means[0]
                    summary_row["mse_final"] = valid_means[-1]
                    summary_row["mse_total_improvement"] = total_improvement
                    summary_row["mse_avg_improvement_per_iter"] = avg_improvement_per_iter
                    summary_row["mse_iterations"] = len(valid_means)

            if gt_error_result is not None:
                means, _, _ = gt_error_result
                valid_means = means[~np.isnan(means)]
                if len(valid_means) >= 2:
                    total_improvement = valid_means[0] - valid_means[-1]
                    avg_improvement_per_iter = total_improvement / (len(valid_means) - 1)
                    print(
                        f"  GT Error - Initial: {valid_means[0]:.4f}, Final: {valid_means[-1]:.4f}"
                    )
                    print(f"  GT Error - Total improvement: {total_improvement:.4f}")
                    print(f"  GT Error - Avg improvement/iter: {avg_improvement_per_iter:.4f}")

                    summary_row["gt_error_initial"] = valid_means[0]
                    summary_row["gt_error_final"] = valid_means[-1]
                    summary_row["gt_error_total_improvement"] = total_improvement
                    summary_row["gt_error_avg_improvement_per_iter"] = avg_improvement_per_iter
                    summary_row["gt_error_iterations"] = len(valid_means)

            summary_data.append(summary_row)

        # Save summary to CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_file = os.path.join(base_dir, f"convergence_summary_{folder_name}.csv")
            summary_df.to_csv(csv_file, index=False)
            print(f"\nConvergence summary saved to: {csv_file}")

    else:
        print("No valid results found!")


if __name__ == "__main__":
    main()
