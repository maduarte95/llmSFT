#!/usr/bin/env python3
"""
Data utilities for CSV file selection and handling
Provides flexible CSV file detection and selection for run scripts.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict


def get_csv_files(directory: str = ".") -> List[Path]:
    """Get all CSV files in the specified directory."""
    current_dir = Path(directory)
    return list(current_dir.glob("*.csv"))


def categorize_csv_files(csv_files: List[Path]) -> Dict[str, List[Path]]:
    """Categorize CSV files by their likely purpose based on filename patterns."""
    categories = {
        "filtered_data": [],
        "with_switches": [],
        "with_labels": [],
        "with_predictions": [],
        "analysis_output": [],
        "other": []
    }
    
    for file in csv_files:
        filename = file.name.lower()
        
        if "filtered" in filename and ("analysis" in filename or "data" in filename):
            categories["filtered_data"].append(file)
        elif "switch" in filename and ("llm" in filename or "with" in filename):
            categories["with_switches"].append(file)
        elif "label" in filename and ("llm" in filename or "with" in filename):
            categories["with_labels"].append(file)
        elif "prediction" in filename and ("llm" in filename or "with" in filename):
            categories["with_predictions"].append(file)
        elif any(term in filename for term in ["analysis", "statistics", "metrics"]):
            categories["analysis_output"].append(file)
        else:
            categories["other"].append(file)
    
    return categories


def display_csv_options(csv_files: List[Path], categorized: Dict[str, List[Path]]):
    """Display CSV files organized by category with numbered options."""
    print("\n📁 Available CSV files:")
    print("=" * 50)
    
    file_index = 1
    index_to_file = {}
    
    for category, files in categorized.items():
        if files:
            # Convert category key to display name
            display_names = {
                "filtered_data": "Filtered Data Files",
                "with_switches": "Files with Switch Classifications",
                "with_labels": "Files with Labels",
                "with_predictions": "Files with Predictions",
                "analysis_output": "Analysis Output Files",
                "other": "Other CSV Files"
            }
            
            print(f"\n🔹 {display_names[category]}:")
            for file in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True):
                file_size = file.stat().st_size / 1024  # KB
                mod_time = file.stat().st_mtime
                mod_time_str = pd.Timestamp(mod_time, unit='s').strftime('%Y-%m-%d %H:%M')
                
                print(f"  {file_index:2d}. {file.name}")
                print(f"      Size: {file_size:.1f} KB | Modified: {mod_time_str}")
                
                index_to_file[file_index] = file
                file_index += 1
    
    return index_to_file


def select_csv_file_interactive(
    preferred_patterns: Optional[List[str]] = None,
    required_columns: Optional[List[str]] = None,
    script_purpose: str = "processing"
) -> Optional[str]:
    """
    Interactive CSV file selection with smart defaults and validation.
    
    Args:
        preferred_patterns: List of filename patterns to prefer (e.g., ["*filtered*"])
        required_columns: List of column names that must be present in the selected file
        script_purpose: Description of what the script does (for user guidance)
    
    Returns:
        Path to selected CSV file or None if cancelled
    """
    current_dir = Path(".")
    csv_files = get_csv_files()
    
    if not csv_files:
        print("❌ No CSV files found in current directory.")
        return None
    
    # Try auto-detection first if patterns provided
    if preferred_patterns:
        for pattern in preferred_patterns:
            matching_files = list(current_dir.glob(pattern))
            if matching_files:
                # Sort by modification time (newest first)
                best_match = max(matching_files, key=lambda f: f.stat().st_mtime)
                print(f"✅ Auto-detected file: {best_match}")
                
                # Validate columns if required
                if required_columns and not validate_csv_columns(best_match, required_columns):
                    print(f"⚠️  Auto-detected file missing required columns: {required_columns}")
                    print("Falling back to interactive selection...")
                else:
                    # Ask for confirmation
                    response = input("Use this file? (y/n/list): ").strip().lower()
                    if response in ['y', 'yes', '']:
                        return str(best_match)
                    elif response in ['n', 'no']:
                        break
                    # 'list' or anything else falls through to interactive selection
                break
    
    # Interactive selection
    categorized = categorize_csv_files(csv_files)
    index_to_file = display_csv_options(csv_files, categorized)
    
    print(f"\n🎯 Purpose: {script_purpose}")
    if required_columns:
        print(f"📋 Required columns: {', '.join(required_columns)}")
    
    while True:
        try:
            print(f"\nSelect a file (1-{len(index_to_file)}) or 'q' to quit:")
            choice = input("Choice: ").strip()
            
            if choice.lower() in ['q', 'quit', 'exit']:
                return None
            
            file_num = int(choice)
            if file_num in index_to_file:
                selected_file = index_to_file[file_num]
                
                # Validate columns if required
                if required_columns and not validate_csv_columns(selected_file, required_columns):
                    print(f"❌ Selected file missing required columns: {', '.join(required_columns)}")
                    print("Please select a different file.")
                    continue
                
                print(f"✅ Selected: {selected_file}")
                return str(selected_file)
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(index_to_file)}.")
        
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n\nSelection cancelled.")
            return None


def validate_csv_columns(file_path: Path, required_columns: List[str]) -> bool:
    """
    Validate that a CSV file contains the required columns.
    
    Args:
        file_path: Path to the CSV file
        required_columns: List of column names that must be present
    
    Returns:
        True if all required columns are present, False otherwise
    """
    try:
        # Read just the header to check columns
        df = pd.read_csv(file_path, nrows=0)
        file_columns = set(df.columns)
        required_set = set(required_columns)
        
        missing = required_set - file_columns
        if missing:
            print(f"❌ Missing columns: {', '.join(missing)}")
            print(f"Available columns: {', '.join(sorted(file_columns))}")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return False


def find_data_file_flexible(
    preferred_patterns: Optional[List[str]] = None,
    required_columns: Optional[List[str]] = None,
    script_purpose: str = "processing",
    auto_select: bool = False
) -> str:
    """
    Find and select a data file with flexible options.
    
    Args:
        preferred_patterns: List of filename patterns to prefer
        required_columns: List of column names that must be present
        script_purpose: Description of script purpose for user guidance
        auto_select: If True, automatically select the first matching file without prompting
    
    Returns:
        Path to selected file
    
    Raises:
        SystemExit: If no suitable file is found or user cancels
    """
    if auto_select and preferred_patterns:
        # Try to auto-select the first matching pattern
        current_dir = Path(".")
        for pattern in preferred_patterns:
            matching_files = list(current_dir.glob(pattern))
            if matching_files:
                best_match = max(matching_files, key=lambda f: f.stat().st_mtime)
                
                # Validate columns if required
                if not required_columns or validate_csv_columns(best_match, required_columns):
                    print(f"✅ Auto-selected file: {best_match}")
                    return str(best_match)
    
    # Fall back to interactive selection
    selected_file = select_csv_file_interactive(
        preferred_patterns=preferred_patterns,
        required_columns=required_columns,
        script_purpose=script_purpose
    )
    
    if not selected_file:
        print("❌ No file selected. Exiting.")
        sys.exit(1)
    
    return selected_file


# Predefined configurations for different script types
SCRIPT_CONFIGS = {
    "classification": {
        "preferred_patterns": [
            "filtered_data_for_analysis.csv",
            "*filtered*analysis*.csv",
            "*filtered*.csv"
        ],
        "required_columns": ["playerID", "category", "word_index", "text"],
        "script_purpose": "LLM switch classification"
    },
    
    "labeling": {
        "preferred_patterns": [
            "*switch*llm*.csv",
            "*switchLLM*.csv", 
            "*with_llm_switches*.csv",
            "data_with_llm_switches*.csv"
        ],
        "required_columns": ["playerID", "category", "switchLLM"],
        "script_purpose": "LLM group labeling (requires switch classification results)"
    },
    
    "prediction": {
        "preferred_patterns": [
            "filtered_data_for_analysis.csv",
            "*filtered*analysis*.csv",
            "*filtered*.csv",
            "data_with_llm_switches*.csv"
        ],
        "required_columns": ["playerID", "category", "word_index", "text"],
        "script_purpose": "LLM switch prediction"
    }
}


def get_data_file_for_script(script_type: str, auto_select: bool = False) -> str:
    """
    Get appropriate data file for a specific script type.
    
    Args:
        script_type: Type of script ("classification", "labeling", "prediction")
        auto_select: Whether to auto-select without user interaction
    
    Returns:
        Path to selected data file
    """
    if script_type not in SCRIPT_CONFIGS:
        raise ValueError(f"Unknown script type: {script_type}. Available: {list(SCRIPT_CONFIGS.keys())}")
    
    config = SCRIPT_CONFIGS[script_type]
    
    return find_data_file_flexible(
        preferred_patterns=config["preferred_patterns"],
        required_columns=config["required_columns"],
        script_purpose=config["script_purpose"],
        auto_select=auto_select
    )