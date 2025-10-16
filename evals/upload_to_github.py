#!/usr/bin/env python3
"""
Upload HTML gallery to GitHub and enable GitHub Pages sharing.
"""

import os
import subprocess
import argparse
from pathlib import Path

def run_command(command, cwd=None):
    """Run shell command and return result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, 
                              capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def upload_to_github(html_file_path, commit_message=None):
    """Upload HTML file to GitHub"""
    
    if not os.path.exists(html_file_path):
        print(f"❌ Error: File {html_file_path} not found!")
        return False
    
    # Get repository root
    repo_root = run_command("git rev-parse --show-toplevel")[1].strip()
    
    # Create commit message
    if not commit_message:
        filename = os.path.basename(html_file_path)
        commit_message = f"Update {filename}"
    
    print(f"📁 Repository root: {repo_root}")
    print(f"📄 HTML file: {html_file_path}")
    
    # Add file to git
    success, output = run_command(f"git add {html_file_path}", cwd=repo_root)
    if not success:
        print(f"❌ Error adding file: {output}")
        return False
    
    # Commit changes
    success, output = run_command(f'git commit -m "{commit_message}"', cwd=repo_root)
    if not success:
        if "nothing to commit" in output:
            print("ℹ️  No changes to commit")
        else:
            print(f"❌ Error committing: {output}")
            return False
    
    # Push to GitHub
    success, output = run_command("git push origin main", cwd=repo_root)
    if not success:
        print(f"❌ Error pushing: {output}")
        return False
    
    print("✅ Successfully uploaded to GitHub!")
    return True

def get_github_pages_url(repo_root):
    """Get GitHub Pages URL for the repository"""
    try:
        # Get remote URL
        success, output = run_command("git remote get-url origin", cwd=repo_root)
        if not success:
            return None
        
        # Parse GitHub URL
        remote_url = output.strip()
        if "github.com" in remote_url:
            # Extract username/repo from URL
            if remote_url.startswith("git@"):
                # SSH format: git@github.com:username/repo.git
                parts = remote_url.replace("git@github.com:", "").replace(".git", "")
            else:
                # HTTPS format: https://github.com/username/repo.git
                parts = remote_url.replace("https://github.com/", "").replace(".git", "")
            
            username, repo_name = parts.split("/")
            pages_url = f"https://{username}.github.io/{repo_name}"
            return pages_url
        
    except Exception as e:
        print(f"Warning: Could not determine GitHub Pages URL: {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Upload HTML gallery to GitHub')
    parser.add_argument('html_file', type=str,
                       help='Path to HTML file to upload')
    parser.add_argument('--message', '-m', type=str,
                       help='Commit message')
    args = parser.parse_args()
    
    html_file = args.html_file
    
    # Convert to absolute path
    if not os.path.isabs(html_file):
        html_file = os.path.abspath(html_file)
    
    print(f"🚀 Uploading {html_file} to GitHub...")
    
    # Upload to GitHub
    success = upload_to_github(html_file, args.message)
    
    if success:
        # Try to get GitHub Pages URL
        repo_root = run_command("git rev-parse --show-toplevel")[1].strip()
        pages_url = get_github_pages_url(repo_root)
        
        if pages_url:
            # Get relative path from repo root
            rel_path = os.path.relpath(html_file, repo_root)
            full_url = f"{pages_url}/{rel_path}"
            
            print(f"\n🌐 GitHub Pages URL:")
            print(f"   {full_url}")
            print(f"\n📋 Instructions:")
            print(f"   1. Go to your GitHub repository")
            print(f"   2. Settings → Pages")
            print(f"   3. Source: Deploy from a branch")
            print(f"   4. Branch: main, Folder: / (root)")
            print(f"   5. Wait 1-2 minutes for deployment")
            print(f"   6. Access your gallery at the URL above")
        else:
            print(f"\n📋 Manual setup needed:")
            print(f"   1. Go to your GitHub repository")
            print(f"   2. Settings → Pages")
            print(f"   3. Enable GitHub Pages")
            print(f"   4. Share the generated URL")
    else:
        print("❌ Upload failed!")

if __name__ == "__main__":
    main()
