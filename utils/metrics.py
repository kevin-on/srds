import os
import json
import pandas as pd
from typing import Dict, Any


def save_key_metrics_summary(output_dir: str, algorithm: str):
    """Save key metrics summary to a separate file"""
    
    summary = {
        'algorithm': algorithm,
        'metrics': {}
    }
    
    # 1. Load trajectory errors to get iteration count and final step changes
    trajectory_file = os.path.join(output_dir, "trajectory_errors_table.csv")
    if os.path.exists(trajectory_file):
        try:
            df = pd.read_csv(trajectory_file)
            if 'Final_Image_Error' in df.columns:
                errors = df['Final_Image_Error'].values
                summary['metrics']['total_iterations'] = len(errors)
                summary['metrics']['initial_error'] = float(errors[0]) if len(errors) > 0 else None
                summary['metrics']['final_error'] = float(errors[-1]) if len(errors) > 0 else None
                
                # Calculate iteration-by-iteration changes for final step
                if len(errors) > 1:
                    changes = []
                    for i in range(1, len(errors)):
                        change = float(errors[i-1] - errors[i])
                        changes.append(change)
                    summary['metrics']['iteration_changes'] = changes
                    summary['metrics']['avg_change_per_iteration'] = float(sum(changes) / len(changes))
        except Exception as e:
            print(f"Warning: Could not load trajectory errors: {e}")
    
    # 2. Load log file to get segment selection info (for sparareal)
    if algorithm == "sparareal":
        log_file = os.path.join(output_dir, "log.txt")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                
                # Extract segment selection information
                original_selections = 0
                total_selections = 0
                selection_details = []
                
                for line in log_content.split('\n'):
                    if "is_original:" in line:
                        total_selections += 1
                        is_original = "True" in line
                        if is_original:
                            original_selections += 1
                        
                        # Extract timestep and candidate info
                        if "Timestep" in line and "Selected candidate" in line:
                            try:
                                parts = line.split("Timestep")[1].split(":")[0].strip()
                                timestep = int(parts)
                                candidate_part = line.split("Selected candidate")[1].split("(")[0].strip()
                                candidate = int(candidate_part)
                                selection_details.append({
                                    'timestep': timestep,
                                    'candidate': candidate,
                                    'is_original': is_original
                                })
                            except:
                                pass
                
                if total_selections > 0:
                    summary['metrics']['segment_selection'] = {
                        'total_selections': total_selections,
                        'original_selections': original_selections,
                        'alternative_selections': total_selections - original_selections,
                        'original_selection_rate': original_selections / total_selections,
                        'alternative_selection_rate': (total_selections - original_selections) / total_selections,
                        'selection_details': selection_details
                    }
                    
            except Exception as e:
                print(f"Warning: Could not parse log file: {e}")
    
    # Save summary to JSON file
    summary_file = os.path.join(output_dir, "key_metrics.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Key metrics saved to: {summary_file}")
    
    # Also save a simple text summary
    text_summary_file = os.path.join(output_dir, "summary.txt")
    with open(text_summary_file, 'w') as f:
        f.write(f"=== {algorithm.upper()} EXPERIMENT SUMMARY ===\n\n")
        
        if 'total_iterations' in summary['metrics']:
            f.write(f"Total Iterations: {summary['metrics']['total_iterations']}\n")
        
        if 'iteration_changes' in summary['metrics']:
            f.write(f"Average Change per Iteration: {summary['metrics']['avg_change_per_iteration']:.6f}\n")
            f.write(f"Iteration Changes: {[f'{c:.6f}' for c in summary['metrics']['iteration_changes']]}\n")
        
        if algorithm == "sparareal" and 'segment_selection' in summary['metrics']:
            sel = summary['metrics']['segment_selection']
            f.write(f"\nSegment Selection:\n")
            f.write(f"  Original selections: {sel['original_selections']}/{sel['total_selections']} ({sel['original_selection_rate']:.1%})\n")
            f.write(f"  Alternative selections: {sel['alternative_selections']}/{sel['total_selections']} ({sel['alternative_selection_rate']:.1%})\n")
    
    print(f"Text summary saved to: {text_summary_file}")


def load_key_metrics(output_dir: str) -> Dict[str, Any]:
    """Load key metrics from a completed experiment"""
    
    metrics_file = os.path.join(output_dir, "key_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load key metrics: {e}")
    
    return {}


def get_convergence_metrics(output_dir: str) -> Dict[str, Any]:
    """Get convergence-related metrics from trajectory errors"""
    
    trajectory_file = os.path.join(output_dir, "trajectory_errors_table.csv")
    if not os.path.exists(trajectory_file):
        return {}
    
    try:
        df = pd.read_csv(trajectory_file)
        if 'Final_Image_Error' not in df.columns:
            return {}
        
        errors = df['Final_Image_Error'].values
        if len(errors) == 0:
            return {}
        
        metrics = {
            'total_iterations': len(errors),
            'initial_error': float(errors[0]),
            'final_error': float(errors[-1]),
        }
        
        if len(errors) > 1:
            changes = []
            for i in range(1, len(errors)):
                change = float(errors[i-1] - errors[i])
                changes.append(change)
            metrics['iteration_changes'] = changes
            metrics['avg_change_per_iteration'] = float(sum(changes) / len(changes))
        
        return metrics
        
    except Exception as e:
        print(f"Warning: Could not load convergence metrics: {e}")
        return {}


def get_segment_selection_metrics(output_dir: str) -> Dict[str, Any]:
    """Get segment selection metrics from log file"""
    
    log_file = os.path.join(output_dir, "log.txt")
    if not os.path.exists(log_file):
        return {}
    
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        original_selections = 0
        total_selections = 0
        selection_details = []
        
        for line in log_content.split('\n'):
            if "is_original:" in line:
                total_selections += 1
                is_original = "True" in line
                if is_original:
                    original_selections += 1
                
                # Extract timestep and candidate info
                if "Timestep" in line and "Selected candidate" in line:
                    try:
                        parts = line.split("Timestep")[1].split(":")[0].strip()
                        timestep = int(parts)
                        candidate_part = line.split("Selected candidate")[1].split("(")[0].strip()
                        candidate = int(candidate_part)
                        selection_details.append({
                            'timestep': timestep,
                            'candidate': candidate,
                            'is_original': is_original
                        })
                    except:
                        pass
        
        if total_selections > 0:
            return {
                'total_selections': total_selections,
                'original_selections': original_selections,
                'alternative_selections': total_selections - original_selections,
                'original_selection_rate': original_selections / total_selections,
                'alternative_selection_rate': (total_selections - original_selections) / total_selections,
                'selection_details': selection_details
            }
        
        return {}
        
    except Exception as e:
        print(f"Warning: Could not load segment selection metrics: {e}")
        return {}
