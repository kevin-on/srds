#!/usr/bin/env python3
"""
Analyze PickScore across iterations for experiments.
Measure PickScore for final samples at each iteration and compare across experiments.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import glob
import argparse
from PIL import Image
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from reward.pickscore import PickScoreInferencer
except ImportError:
    print("Warning: PickScoreInferencer not found. Please ensure the module is available.")
    PickScoreInferencer = None

try:
    from reward.clip_score import CLIPScoreInferencer
except ImportError:
    print("Warning: CLIPScoreInferencer not found. Please ensure the module is available.")
    CLIPScoreInferencer = None

def extract_experiment_name(dir_path):
    """Extract experiment configuration from directory name"""
    dir_name = os.path.basename(dir_path)
    # Extract adaptive parameter (ad0, ad2, ad4, etc.)
    match = re.search(r'ad(\d+)', dir_name)
    if match:
        return f"ad{match.group(1)}"
    return "unknown"

def extract_prompt_from_dir(prompt_dir):
    """Extract prompt text from directory name"""
    dir_name = os.path.basename(prompt_dir)
    # Extract prompt from directory name like "prompt0_a_majestic_mountain_landscape"
    match = re.match(r'prompt\d+_(.+)', dir_name)
    if match:
        return match.group(1).replace('_', ' ')
    return dir_name

def get_iteration_images(prompt_dir):
    """Get images from each iteration in a prompt directory"""
    iteration_images = {}
    
    # Look for iteration-specific image files with different naming patterns
    image_patterns = [
        "srds_iteration_*.png",  # srds_iteration_0.png, srds_iteration_1.png, etc.
        "iteration_*.png",       # iteration_5.png
        "sample_iteration_*.png" # sample_iteration_5.png
    ]
    
    for pattern in image_patterns:
        image_files = glob.glob(os.path.join(prompt_dir, pattern))
        
        for img_file in image_files:
            # Extract iteration number from filename
            match = re.search(r'(?:srds_)?(?:sample_)?iteration_(\d+)\.png', os.path.basename(img_file))
            if match:
                iteration_num = int(match.group(1))
                try:
                    img = Image.open(img_file)
                    iteration_images[iteration_num] = img
                    print(f"    Found iteration {iteration_num}: {os.path.basename(img_file)}")
                except Exception as e:
                    print(f"Error loading image {img_file}: {e}")
    
    # Also check for other final images
    final_image_files = [
        "srds_final.png",
        "final_image.png",
        "sample_final.png",
        "srds_initialized.png"
    ]
    
    for filename in final_image_files:
        img_path = os.path.join(prompt_dir, filename)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                # Use special indices for non-iteration images
                if "initialized" in filename:
                    iteration_images[-1] = img  # Initial image
                else:
                    # Don't add final image as separate iteration, it's usually same as last iteration
                    pass
                print(f"    Found special image: {filename}")
                break
            except Exception as e:
                print(f"Error loading special image {img_path}: {e}")
    
    return iteration_images

def calculate_metric_for_iterations(iteration_images, prompt_text, metric, inferencer):
    """Calculate specified metric for each iteration"""
    if inferencer is None:
        print(f"{metric.upper()} inferencer not available, returning dummy scores")
        return {iter_num: 0.5 for iter_num in iteration_images.keys()}
    
    scores = {}
    
    for iteration_num, image in iteration_images.items():
        try:
            if metric == 'pickscore':
                score = inferencer.score([image], prompt_text)[0]
            elif metric == 'clip':
                score = inferencer.score([image], prompt_text)[0]
            elif metric == 'hps':
                # For HPS, we would need to implement HPS scoring
                score = 0.5  # Placeholder
            elif metric == 'mse':
                # For MSE, we would need ground truth images
                score = 0.5  # Placeholder
            elif metric == 'l1':
                # For L1, we would need ground truth images
                score = 0.5  # Placeholder
            else:
                score = 0.5
            
            scores[iteration_num] = score
            print(f"  Iteration {iteration_num}: {metric.upper()} = {score:.4f}")
        except Exception as e:
            print(f"Error calculating {metric} for iteration {iteration_num}: {e}")
            scores[iteration_num] = 0.0
    
    return scores

def analyze_experiment_metric(exp_dir, metric, inferencer):
    """Analyze specified metric for a single experiment"""
    exp_name = extract_experiment_name(exp_dir)
    
    # Find all prompt directories
    prompt_dirs = glob.glob(os.path.join(exp_dir, "prompt*"))
    prompt_dirs.sort()
    
    all_iteration_scores = {}
    all_final_scores = []
    
    print(f"\nAnalyzing experiment: {exp_name}")
    print(f"Found {len(prompt_dirs)} prompts")
    
    for prompt_dir in prompt_dirs:
        prompt_name = os.path.basename(prompt_dir)
        prompt_text = extract_prompt_from_dir(prompt_dir)
        
        print(f"\nProcessing {prompt_name}")
        print(f"Prompt: {prompt_text}")
        
        # Get images for each iteration
        iteration_images = get_iteration_images(prompt_dir)
        
        if not iteration_images:
            print(f"  No iteration images found in {prompt_dir}")
            continue
        
        print(f"  Found {len(iteration_images)} iteration images")
        
        # Calculate metric for each iteration
        scores = calculate_metric_for_iterations(
            iteration_images, prompt_text, metric, inferencer
        )
        
        # Store scores by iteration
        for iter_num, score in scores.items():
            if iter_num not in all_iteration_scores:
                all_iteration_scores[iter_num] = []
            all_iteration_scores[iter_num].append(score)
        
        # Store final score (highest iteration number, excluding special indices)
        final_score = None
        if scores:
            # Get the highest iteration number (excluding special indices like -1)
            regular_iters = [k for k in scores.keys() if k >= 0]
            if regular_iters:
                max_iter = max(regular_iters)
                final_score = scores[max_iter]
        
        if final_score is not None:
            all_final_scores.append(final_score)
    
    # Calculate statistics for each iteration
    if all_iteration_scores:
        iteration_stats = {}
        for iter_num, scores in all_iteration_scores.items():
            iteration_stats[iter_num] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
        
        return exp_name, iteration_stats, all_final_scores
    else:
        return exp_name, None, []

def main():
    parser = argparse.ArgumentParser(description='Analyze iteration convergence from experiments')
    parser.add_argument('--folder', type=str, required=True, 
                       help='Base folder containing experiment results')
    parser.add_argument('--metric', type=str, default='pickscore',
                       choices=['pickscore', 'clip', 'hps', 'mse', 'l1'],
                       help='Metric to use for evaluation (default: pickscore)')
    args = parser.parse_args()
    
    base_dir = args.folder
    
    # Initialize metric inferencer based on selected metric
    inferencer = None
    if args.metric == 'pickscore' and PickScoreInferencer is not None:
        try:
            print("Initializing PickScore inferencer...")
            inferencer = PickScoreInferencer(device='cuda')
            print("PickScore inferencer initialized successfully")
        except Exception as e:
            print(f"Error initializing PickScore inferencer: {e}")
            print("Will use dummy scores")
    elif args.metric == 'clip' and CLIPScoreInferencer is not None:
        try:
            print("Initializing CLIP inferencer...")
            inferencer = CLIPScoreInferencer(device='cuda')
            print("CLIP inferencer initialized successfully")
        except Exception as e:
            print(f"Error initializing CLIP inferencer: {e}")
            print("Will use dummy scores")
    else:
        print(f"{args.metric.upper()} inferencer not available, using dummy scores")
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(base_dir, "2025*"))
    exp_dirs.sort()
    
    print(f"Found {len(exp_dirs)} experiments in {base_dir}")
    
    results = []
    for exp_dir in exp_dirs:
        exp_name, iteration_stats, final_scores = analyze_experiment_metric(exp_dir, args.metric, inferencer)
        if iteration_stats is not None:
            results.append((exp_name, iteration_stats, final_scores))
    
    # Create plots
    if results:
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        
        # Plot 1: Metric across iterations (left)
        ax1 = axes[0]
        for i, (exp_name, iteration_stats, final_scores) in enumerate(results):
            # Filter out special iterations (like -1)
            regular_iterations = [k for k in sorted(iteration_stats.keys()) if k >= 0]
            means = [iteration_stats[iter_num]['mean'] for iter_num in regular_iterations]
            
            ax1.plot(regular_iterations, means, 'o-', 
                    label=f'{exp_name}', color=colors[i], linewidth=2.5, markersize=6)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel(args.metric.upper())
        ax1.set_title(f'{args.metric.upper()} Across Iterations')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Adjust y-axis scale to better separate the curves
        all_means = []
        for _, iteration_stats, _ in results:
            regular_iterations = [k for k in sorted(iteration_stats.keys()) if k >= 0]
            means = [iteration_stats[iter_num]['mean'] for iter_num in regular_iterations]
            all_means.extend(means)
        
        if all_means:
            y_min = min(all_means) * 0.998  # Slightly below minimum
            y_max = max(all_means) * 1.002  # Slightly above maximum
            ax1.set_ylim(y_min, y_max)
        
        # Plot 2: Final scores box plot (right)
        ax2 = axes[1]
        exp_names = [result[0] for result in results]
        final_score_data = [result[2] for result in results]
        
        bp = ax2.boxplot(final_score_data, labels=exp_names, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel(f'Final {args.metric.upper()}')
        ax2.set_title(f'Final {args.metric.upper()} Distribution by Experiment')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with folder name in the specified folder
        folder_name = os.path.basename(base_dir)
        output_file = os.path.join(base_dir, f'image_score_{args.metric}_{folder_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nGraph saved as: {output_file}")
        plt.show()
        
        # Print analysis summary
        print("\n" + "="*80)
        print(f"{args.metric.upper()} ITERATION ANALYSIS")
        print("="*80)
        
        for exp_name, iteration_stats, final_scores in results:
            print(f"\n{exp_name}:")
            # Filter out special iterations
            regular_iterations = [k for k in sorted(iteration_stats.keys()) if k >= 0]
            means = [iteration_stats[iter_num]['mean'] for iter_num in regular_iterations]
            stds = [iteration_stats[iter_num]['std'] for iter_num in regular_iterations]
            
            if len(means) >= 2:
                initial_score = means[0]
                final_score = means[-1]
                improvement = final_score - initial_score
                
                print(f"  Initial {args.metric.upper()}: {initial_score:.4f} ± {stds[0]:.4f}")
                print(f"  Final {args.metric.upper()}: {final_score:.4f} ± {stds[-1]:.4f}")
                print(f"  Improvement: {improvement:.4f}")
                print(f"  Iterations: {len(regular_iterations)}")
                
                # Find best iteration
                best_iter_idx = np.argmax(means)
                best_iter = regular_iterations[best_iter_idx]
                best_score = means[best_iter_idx]
                print(f"  Best iteration: {best_iter} ({args.metric.upper()}: {best_score:.4f})")
            
            # Final scores statistics
            if final_scores:
                print(f"  Final scores mean: {np.mean(final_scores):.4f} ± {np.std(final_scores):.4f}")
                print(f"  Final scores range: [{np.min(final_scores):.4f}, {np.max(final_scores):.4f}]")
    
    else:
        print("No valid results found!")

if __name__ == "__main__":
    main()
