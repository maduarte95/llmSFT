#!/usr/bin/env python3
"""
Majority Class Labeling Script
Labels groups based on majority class from snafu categories instead of LLM calls.
Uses ground truth switches to create groups, then determines the most common category.
"""

import pandas as pd
import numpy as np
import ast
import random
from collections import Counter
import sys
from pathlib import Path


def parse_snafu_categories(cat_string):
    """
    Parse snafu category string into a list of categories.
    
    Args:
        cat_string: String representation of categories (e.g., "['Pets', 'Canine']")
        
    Returns:
        List of categories or empty list if parsing fails
    """
    if pd.isna(cat_string) or cat_string == '':
        return []
    
    try:
        # Handle string representation of Python lists
        if isinstance(cat_string, str):
            # Remove any extra whitespace and evaluate as Python literal
            cleaned = cat_string.strip()
            if cleaned.startswith('[') and cleaned.endswith(']'):
                return ast.literal_eval(cleaned)
            else:
                # Single category not in list format
                return [cleaned]
        elif isinstance(cat_string, list):
            return cat_string
        else:
            return []
    except (ValueError, SyntaxError):
        print(f"Warning: Could not parse category string: {cat_string}")
        return []


def has_ground_truth_switches(sequence_data):
    """
    Check if a player sequence has ground truth switch data.
    
    Args:
        sequence_data: DataFrame with word sequence data for one player
        
    Returns:
        True if sequence has switch data, False otherwise
    """
    gt_switches = sequence_data['switch_ground_truth'].copy()
    
    # Convert empty strings to NaN
    gt_switches = gt_switches.replace('', np.nan)
    
    # Check for any non-NaN values at positions other than index 0
    non_zero_indices = sequence_data['word_index'] != 0
    has_switches = gt_switches[non_zero_indices].notna().any()
    
    return has_switches


def create_groups_from_switches(sequence_data):
    """
    Create groups from ground truth switches.
    
    Args:
        sequence_data: DataFrame with word sequence data sorted by word_index
        
    Returns:
        List of group dictionaries with 'words' and 'categories' keys
    """
    # Convert ground truth switches to numeric
    gt_switches = sequence_data['switch_ground_truth'].copy()
    gt_switches = gt_switches.replace('', np.nan)
    gt_switches_numeric = pd.to_numeric(gt_switches, errors='coerce')
    
    # For index 0, treat as start of first group (no switch)
    first_row_mask = sequence_data['word_index'] == 0
    if first_row_mask.any():
        gt_switches_numeric[first_row_mask] = 0.0
    
    # Create groups
    groups = []
    current_group_words = []
    current_group_categories = []
    
    for idx, row in sequence_data.iterrows():
        word = row['text']
        switch_value = gt_switches_numeric.loc[idx]
        categories = parse_snafu_categories(row['snafu_cat'])
        
        # Check if this word starts a new group (switch=1)
        if switch_value == 1.0 and current_group_words:
            # Finish the previous group
            groups.append({
                'words': current_group_words.copy(),
                'categories': current_group_categories.copy()
            })
            current_group_words = []
            current_group_categories = []
        
        # Add current word and categories to the current group
        current_group_words.append(word)
        current_group_categories.extend(categories)
    
    # Add final group if there are remaining words
    if current_group_words:
        groups.append({
            'words': current_group_words,
            'categories': current_group_categories
        })
    
    return groups


def determine_majority_category(categories):
    """
    Determine the majority category from a list of categories.
    In case of tie, choose randomly.
    
    Args:
        categories: List of category strings
        
    Returns:
        String of the majority category or None if no categories
    """
    if not categories:
        return None
    
    # Count categories
    category_counts = Counter(categories)
    
    # Find the maximum count
    max_count = max(category_counts.values())
    
    # Get all categories with the maximum count
    majority_categories = [cat for cat, count in category_counts.items() if count == max_count]
    
    # If tie, choose randomly
    if len(majority_categories) > 1:
        random.seed(42)  # For reproducible results
        return random.choice(majority_categories)
    else:
        return majority_categories[0]


def process_player_sequence(sequence_data):
    """
    Process a single player sequence to assign majority class labels.
    
    Args:
        sequence_data: DataFrame with word sequence data for one player
        
    Returns:
        DataFrame with snafu_gt_label column filled
    """
    result_data = sequence_data.copy()
    result_data['snafu_gt_label'] = None
    
    # Check if player has ground truth switches
    if not has_ground_truth_switches(sequence_data):
        # Leave labels blank for players without ground truth
        return result_data
    
    # Some players have all switches as 0 - check for this case
    gt_switches = sequence_data['switch_ground_truth'].copy()
    gt_switches = gt_switches.replace('', np.nan)
    gt_switches_numeric = pd.to_numeric(gt_switches, errors='coerce')
    
    # Check if all non-NaN switches are 0
    valid_switches = gt_switches_numeric[sequence_data['word_index'] != 0].dropna()
    if len(valid_switches) > 0 and (valid_switches == 0).all():
        # All switches are 0 - treat as one big group
        all_categories = []
        for _, row in sequence_data.iterrows():
            categories = parse_snafu_categories(row['snafu_cat'])
            all_categories.extend(categories)
        
        majority_label = determine_majority_category(all_categories)
        result_data['snafu_gt_label'] = majority_label
        return result_data
    
    # Create groups from switches
    groups = create_groups_from_switches(sequence_data)
    
    # Assign majority class labels to each group
    word_idx = 0
    for group in groups:
        majority_label = determine_majority_category(group['categories'])
        
        # Apply this label to all words in the group
        for word in group['words']:
            if word_idx < len(sequence_data):
                row_idx = sequence_data.iloc[word_idx].name
                result_data.loc[row_idx, 'snafu_gt_label'] = majority_label
                word_idx += 1
    
    return result_data


def main():
    """Main execution function."""
    print("=" * 60)
    print("Majority Class Labeling Script")
    print("=" * 60)
    
    # Load data
    print("\nStep 1: Loading data...")
    data_file = "categorized_pred_animals.csv"
    
    if not Path(data_file).exists():
        print(f"Error: Data file '{data_file}' not found!")
        print("Please ensure the file is in the current directory.")
        sys.exit(1)
    
    try:
        data = pd.read_csv(data_file)
        print(f"Data loaded successfully")
        print(f"   Shape: {data.shape}")
        print(f"   Unique players: {data['playerID'].nunique()}")
        
        # Check required columns
        required_columns = ['text', 'switch_ground_truth', 'snafu_cat', 'playerID', 'word_index']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(data.columns)}")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Process data
    print("\nStep 2: Processing player sequences...")
    
    result_data = data.copy()
    result_data['snafu_gt_label'] = None
    
    player_groups = data.groupby('playerID')
    processed_players = 0
    skipped_players = 0
    
    for player_id, player_data in player_groups:
        sequence_data = player_data.sort_values('word_index')
        
        # Process this player's sequence
        processed_sequence = process_player_sequence(sequence_data)
        
        # Update the result data
        for idx in processed_sequence.index:
            result_data.loc[idx, 'snafu_gt_label'] = processed_sequence.loc[idx, 'snafu_gt_label']
        
        # Check if player was processed or skipped
        if processed_sequence['snafu_gt_label'].notna().any():
            processed_players += 1
        else:
            skipped_players += 1
    
    print(f"   Processed players: {processed_players}")
    print(f"   Skipped players (no ground truth): {skipped_players}")
    
    # Save results
    print("\nStep 3: Saving results...")
    
    output_file = "data_with_majority_class_labels.csv"
    result_data.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Analysis summary
    print("\nStep 4: Analysis summary...")
    
    total_rows = len(result_data)
    labeled_rows = result_data['snafu_gt_label'].notna().sum()
    
    print(f"Total rows: {total_rows:,}")
    print(f"Rows with majority class labels: {labeled_rows:,}")
    print(f"Coverage: {labeled_rows/total_rows:.1%}")
    
    if labeled_rows > 0:
        # Show label distribution
        label_counts = result_data['snafu_gt_label'].value_counts()
        print(f"\nLabel distribution:")
        for label, count in label_counts.head(10).items():
            print(f"   '{label}': {count} occurrences")
        
        if len(label_counts) > 10:
            print(f"   ... and {len(label_counts) - 10} more labels")
    
    print("\n" + "=" * 60)
    print("Majority class labeling completed successfully!")
    print(f"Output file: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)