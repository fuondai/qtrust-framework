#!/usr/bin/env python3
"""
QTrust Project Cleaner

This script automatically finds and deletes all log, csv, png, json, txt, and pkl files in the project directory.
It helps to clean up the project by removing generated files before committing or sharing code.
"""

import os
import sys
import glob
import shutil
from datetime import datetime

def find_files(root_dir, extensions):
    """
    Find all files with specified extensions in the given directory and its subdirectories.
    
    Args:
        root_dir (str): The root directory to search in
        extensions (list): List of file extensions to look for (without dot)
        
    Returns:
        list: A list of file paths
    """
    files = []
    for ext in extensions:
        pattern = os.path.join(root_dir, "**", f"*.{ext}")
        found_files = glob.glob(pattern, recursive=True)
        files.extend(found_files)
    return files

def create_backup(files, backup_dir):
    """
    Create a backup of files before deletion
    
    Args:
        files (list): List of file paths to backup
        backup_dir (str): Directory to store backups
        
    Returns:
        bool: True if backup was successful, False otherwise
    """
    if not files:
        return True
    
    try:
        os.makedirs(backup_dir, exist_ok=True)
        for file_path in files:
            rel_path = os.path.relpath(file_path, os.getcwd())
            backup_path = os.path.join(backup_dir, rel_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(file_path, backup_path)
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def delete_files(files, no_confirm=False):
    """
    Delete the specified files
    
    Args:
        files (list): List of file paths to delete
        no_confirm (bool): If True, delete without confirmation
        
    Returns:
        int: Number of files deleted
    """
    if not files:
        print("No files found to delete.")
        return 0
    
    # Group files by type for better reporting
    file_types = {}
    for file_path in files:
        ext = os.path.splitext(file_path)[1][1:]  # Remove the dot
        if ext not in file_types:
            file_types[ext] = []
        file_types[ext].append(file_path)
    
    # Print summary
    print("\nFiles to be deleted:")
    for ext, type_files in file_types.items():
        print(f"  {ext.upper()} files: {len(type_files)}")
    
    # Ask for confirmation
    if not no_confirm:
        confirmation = input("\nAre you sure you want to delete these files? (y/N): ")
        if confirmation.lower() != 'y':
            print("Operation cancelled.")
            return 0
    
    # Delete files
    deleted_count = 0
    for file_path in files:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    return deleted_count

def clean_empty_dirs(root_dir, no_confirm=False):
    """
    Remove empty directories
    
    Args:
        root_dir (str): Root directory to start cleaning from
        no_confirm (bool): If True, delete without confirmation
        
    Returns:
        int: Number of directories deleted
    """
    empty_dirs = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Skip the root directory itself
        if dirpath == root_dir:
            continue
            
        # If directory is empty, add to list
        if not dirnames and not filenames:
            empty_dirs.append(dirpath)
    
    if not empty_dirs:
        return 0
        
    print(f"\nFound {len(empty_dirs)} empty directories.")
    
    # Ask for confirmation
    if not no_confirm:
        confirmation = input("\nAre you sure you want to delete these directories? (y/N): ")
        if confirmation.lower() != 'y':
            print("Directory cleanup cancelled.")
            return 0
    
    # Delete empty directories
    deleted_count = 0
    for dir_path in empty_dirs:
        try:
            os.rmdir(dir_path)
            print(f"Removed empty directory: {dir_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting directory {dir_path}: {e}")
    
    return deleted_count

def main():
    """Main function"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Clean up generated files in the QTrust project")
    parser.add_argument("-b", "--backup", action="store_true", help="Create backup before deletion")
    parser.add_argument("-y", "--yes", action="store_true", help="Delete without confirmation")
    parser.add_argument("-t", "--types", default="log,csv,png,json,txt,pkl", 
                        help="File types to delete (comma-separated, default: log,csv,png,json,txt,pkl)")
    parser.add_argument("-d", "--directory", default=".", help="Root directory to search in (default: current directory)")
    parser.add_argument("--skip-dirs", default=".git,venv,env,.mypy_cache,.pytest_cache", 
                        help="Directories to skip (comma-separated)")
    parser.add_argument("--clean-empty", action="store_true", help="Also remove empty directories")
    args = parser.parse_args()
    
    # Prepare file types
    file_extensions = [ext.strip() for ext in args.types.split(",")]
    
    # Find files
    root_dir = os.path.abspath(args.directory)
    files = find_files(root_dir, file_extensions)
    
    # Filter out files in skip directories
    skip_dirs = [os.path.join(root_dir, d.strip()) for d in args.skip_dirs.split(",")]
    filtered_files = []
    for file_path in files:
        should_skip = False
        for skip_dir in skip_dirs:
            if file_path.startswith(skip_dir):
                should_skip = True
                break
        if not should_skip:
            filtered_files.append(file_path)
    
    # Create backup if requested
    if args.backup and filtered_files:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"backup_{timestamp}"
        print(f"Creating backup in {backup_dir}...")
        if not create_backup(filtered_files, backup_dir):
            print("Backup failed. Aborting operation.")
            return 1
    
    # Delete files
    deleted_count = delete_files(filtered_files, args.yes)
    
    print(f"\nOperation completed: {deleted_count} files deleted.")
    
    # Clean empty directories if requested
    if args.clean_empty:
        print("\nLooking for empty directories...")
        dirs_deleted = clean_empty_dirs(root_dir, args.yes)
        print(f"Directory cleanup completed: {dirs_deleted} directories removed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 