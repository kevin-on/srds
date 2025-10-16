#!/usr/bin/env python3
"""
Generate HTML gallery for experiment results.
Displays final images from each prompt in each experiment folder.
"""

import os
import glob
import re
import argparse
from pathlib import Path

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
    # Extract prompt text after prompt number
    match = re.search(r'prompt\d+_(.+)', dir_name)
    if match:
        return match.group(1).replace('_', ' ')
    return dir_name

def find_final_images(prompt_dir):
    """Find final image files in a prompt directory"""
    final_images = []
    
    # Look for various final image patterns
    final_image_patterns = [
        "srds_final.png",
        "final_image.png", 
        "sample_final.png",
        "srds_initialized.png"
    ]
    
    for pattern in final_image_patterns:
        img_path = os.path.join(prompt_dir, pattern)
        if os.path.exists(img_path):
            final_images.append({
                'filename': pattern,
                'path': img_path,
                'relative_path': os.path.relpath(img_path, prompt_dir)
            })
            break  # Take the first found image
    
    return final_images

def generate_html_gallery(base_dir):
    """Generate HTML gallery for all experiments"""
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(base_dir, "2025*"))
    exp_dirs.sort()
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Method Comparison - {os.path.basename(base_dir)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 10px; background: #fff; }}
        h1 {{ text-align: center; color: #333; }}
        .prompt-section {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
        .prompt-title {{ font-weight: bold; margin-bottom: 10px; color: #555; }}
        .methods-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
        .method {{ text-align: center; }}
        .method-name {{ font-size: 12px; margin-bottom: 5px; color: #666; }}
        .method-image {{ width: 120px; height: 120px; object-fit: cover; border: 1px solid #ccc; }}
        .stats {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1>Method Comparison by Prompt</h1>
    <p style="text-align: center; font-size: 12px; color: #666;">Generated from: {base_dir}</p>
"""
    
    # Collect all prompts and their methods
    prompt_data = {}
    
    for exp_dir in exp_dirs:
        exp_name = extract_experiment_name(exp_dir)
        prompt_dirs = glob.glob(os.path.join(exp_dir, "prompt*"))
        prompt_dirs.sort()
        
        for prompt_dir in prompt_dirs:
            prompt_name = os.path.basename(prompt_dir)
            prompt_text = extract_prompt_from_dir(prompt_dir)
            
            if prompt_text not in prompt_data:
                prompt_data[prompt_text] = {}
            
            final_images = find_final_images(prompt_dir)
            if final_images:
                img = final_images[0]
                relative_img_path = os.path.relpath(img['path'], base_dir)
                prompt_data[prompt_text][exp_name] = {
                    'image_path': relative_img_path,
                    'filename': img['filename']
                }
    
    # Generate HTML by prompt
    for prompt_text, methods in prompt_data.items():
        html_content += f"""
    <div class="prompt-section">
        <div class="prompt-title">{prompt_text}</div>
        <div class="methods-grid">
"""
        
        for method_name, data in methods.items():
            html_content += f"""
            <div class="method">
                <div class="method-name">{method_name}</div>
                <img src="{data['image_path']}" alt="{method_name}" class="method-image">
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    # Add summary
    total_prompts = len(prompt_data)
    total_methods = len(exp_dirs)
    
    html_content += f"""
    <div class="stats">
        Summary: {total_methods} methods, {total_prompts} prompts
    </div>
</body>
</html>
"""
    
    return html_content

def main():
    parser = argparse.ArgumentParser(description='Generate HTML gallery for experiment results')
    parser.add_argument('--folder', type=str, required=True,
                       help='Base folder containing experiment results (e.g., output/adaptive_10)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output HTML file name (default: gallery_{folder_name}.html)')
    args = parser.parse_args()
    
    base_dir = args.folder
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist!")
        return
    
    # Generate HTML content
    print(f"Generating HTML gallery for: {base_dir}")
    html_content = generate_html_gallery(base_dir)
    
    # Determine output filename (save in the specified folder)
    if args.output:
        output_file = args.output
    else:
        folder_name = os.path.basename(base_dir)
        output_file = os.path.join(base_dir, f"gallery_{folder_name}.html")
    
    # Save HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML gallery saved as: {output_file}")
    print(f"📁 Open in browser to view results from {base_dir}")
    
    # Print summary
    exp_dirs = glob.glob(os.path.join(base_dir, "2025*"))
    exp_dirs.sort()
    
    total_prompts = 0
    for exp_dir in exp_dirs:
        prompt_dirs = glob.glob(os.path.join(exp_dir, "prompt*"))
        total_prompts += len(prompt_dirs)
    
    print(f"\n📊 Summary:")
    print(f"   - Methods: {len(exp_dirs)}")
    print(f"   - Prompts: {total_prompts}")
    print(f"   - HTML file: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
