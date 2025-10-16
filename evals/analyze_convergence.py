#!/usr/bin/env python3
"""
Analyze iteration-by-iteration convergence speed from experiments.
Show how convergence values change across iterations for each experiment.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import glob
import argparse

def extract_experiment_name(dir_path):
    """Extract experiment configuration from directory name"""
    dir_name = os.path.basename(dir_path)
    # Extract adaptive parameter (ad0, ad2, ad4, etc.)
    match = re.search(r'ad(\d+)', dir_name)
    if match:
        return f"ad{match.group(1)}"
    return "unknown"

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

def extract_l1_and_mse_from_log(log_file_path):
    """Extract L1 distances and MSE from log.txt file"""
    l1_distances = []
    mse_values = []
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
            
        # Find all L1 distance entries
        # Pattern: "SRDS Iteration X: L1=Y.YYYYYY (tolerance=0.1) - continuing"
        l1_pattern = r'SRDS Iteration (\d+): L1=([\d.]+) \(tolerance=0\.1\) - continuing'
        l1_matches = re.findall(l1_pattern, content)
        
        for iteration, l1_dist in l1_matches:
            l1_distances.append((int(iteration), float(l1_dist)))
        
        # Find MSE entries if they exist
        # Pattern: "SRDS Iteration X: MSE=Y.YYYYYY (tolerance=0.1) - continuing"
        mse_pattern = r'SRDS Iteration (\d+): MSE=([\d.]+) \(tolerance=0\.1\) - continuing'
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
            padded[:len(data)] = data
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
            padded[:len(data)] = data
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
            padded[:len(data)] = data
            padded_mse_data.append(padded)
        
        mse_means = np.nanmean(padded_mse_data, axis=0)
        mse_stds = np.nanstd(padded_mse_data, axis=0)
        mse_result = (mse_means, mse_stds, max_iterations_mse)
    
    return exp_name, convergence_result, l1_result, mse_result

def main():
    parser = argparse.ArgumentParser(description='Analyze iteration convergence from experiments')
    parser.add_argument('--folder', type=str, required=True, 
                       help='Base folder containing experiment results')
    args = parser.parse_args()
    
    base_dir = args.folder
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(base_dir, "2025*"))
    exp_dirs.sort()
    
    print(f"Found {len(exp_dirs)} experiments in {base_dir}")
    
    results = []
    for exp_dir in exp_dirs:
        exp_name, convergence_result, l1_result, mse_result = analyze_experiment_iterations(exp_dir)
        if convergence_result is not None or l1_result is not None or mse_result is not None:
            results.append((exp_name, convergence_result, l1_result, mse_result))
    
    # Create plots
    if results:
        # Create figure with 2 subplots (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        
        # Plot 1: Convergence curves (left)
        ax1 = axes[0]
        for i, (exp_name, convergence_result, _, _) in enumerate(results):
            if convergence_result is not None:
                means, stds, max_iter = convergence_result
                iterations = np.arange(1, max_iter + 1)
                valid_mask = ~np.isnan(means)
                valid_iterations = iterations[valid_mask]
                valid_means = means[valid_mask]
                valid_stds = stds[valid_mask]
                
                ax1.plot(valid_iterations, valid_means, 'o-', 
                        label=f'{exp_name}', color=colors[i], linewidth=2, markersize=4)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Final Step Convergence Value')
        ax1.set_title('Convergence Speed Across Iterations (Log Scale)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale
        
        # Plot 2: L1 distances (right)
        ax2 = axes[1]
        for i, (exp_name, _, l1_result, _) in enumerate(results):
            if l1_result is not None:
                means, stds, max_iter = l1_result
                iterations = np.arange(1, max_iter + 1)
                valid_mask = ~np.isnan(means)
                valid_iterations = iterations[valid_mask]
                valid_means = means[valid_mask]
                valid_stds = stds[valid_mask]
                
                ax2.plot(valid_iterations, valid_means, 'o-', 
                        label=f'{exp_name}', color=colors[i], linewidth=2, markersize=4)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('L1 Distance')
        ax2.set_title('L1 Distance Across Iterations (Log Scale)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale
        
        plt.tight_layout()
        
        # Save with folder name in the specified folder
        folder_name = os.path.basename(base_dir)
        output_file = os.path.join(base_dir, f'iteration_convergence_analysis_{folder_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nGraph saved as: {output_file}")
        plt.show()
        
        # Print convergence speed analysis
        print("\n" + "="*80)
        print("CONVERGENCE SPEED ANALYSIS")
        print("="*80)
        
        for exp_name, convergence_result, l1_result, mse_result in results:
            print(f"\n{exp_name}:")
            
            if convergence_result is not None:
                means, _, _ = convergence_result
                valid_means = means[~np.isnan(means)]
                if len(valid_means) >= 2:
                    total_improvement = valid_means[0] - valid_means[-1]
                    avg_improvement_per_iter = total_improvement / (len(valid_means) - 1)
                    print(f"  Convergence - Initial: {valid_means[0]:.4f}, Final: {valid_means[-1]:.4f}")
                    print(f"  Convergence - Total improvement: {total_improvement:.4f}")
                    print(f"  Convergence - Avg improvement/iter: {avg_improvement_per_iter:.4f}")
            
            if l1_result is not None:
                means, _, _ = l1_result
                valid_means = means[~np.isnan(means)]
                if len(valid_means) >= 2:
                    total_improvement = valid_means[0] - valid_means[-1]
                    avg_improvement_per_iter = total_improvement / (len(valid_means) - 1)
                    print(f"  L1 Distance - Initial: {valid_means[0]:.4f}, Final: {valid_means[-1]:.4f}")
                    print(f"  L1 Distance - Total improvement: {total_improvement:.4f}")
                    print(f"  L1 Distance - Avg improvement/iter: {avg_improvement_per_iter:.4f}")
            
            if mse_result is not None:
                means, _, _ = mse_result
                valid_means = means[~np.isnan(means)]
                if len(valid_means) >= 2:
                    total_improvement = valid_means[0] - valid_means[-1]
                    avg_improvement_per_iter = total_improvement / (len(valid_means) - 1)
                    print(f"  MSE - Initial: {valid_means[0]:.4f}, Final: {valid_means[-1]:.4f}")
                    print(f"  MSE - Total improvement: {total_improvement:.4f}")
                    print(f"  MSE - Avg improvement/iter: {avg_improvement_per_iter:.4f}")
    
    else:
        print("No valid results found!")

if __name__ == "__main__":
    main()