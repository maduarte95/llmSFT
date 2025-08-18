"""
Word Prediction Analysis
========================

This script analyzes word prediction accuracy by word index, addressing data quality issues
like normalization, NaN handling, and improved matching criteria.

Usage:
    uv run python word_prediction_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def improve_normalization(text):
    """
    Apply comprehensive text normalization as mentioned in the data description.
    
    Parameters:
    text: str, input text to normalize
    
    Returns:
    str, normalized text
    """
    if pd.isna(text):
        return np.nan
    
    text = str(text).lower()
    # Remove punctuation (hyphens, apostrophes, commas, periods)
    text = re.sub(r'[-\',.]', '', text)
    # Remove whitespace
    text = re.sub(r'\s+', '', text)
    
    return text

def find_csv_files():
    """
    Find all CSV files in the current directory and subdirectories.
    
    Returns:
    list, paths to CSV files
    """
    csv_files = []
    
    # Search in current directory and common subdirectories
    search_dirs = ['.', 'data', 'process_for_labels', 'analysis_output', 'analysis_output/word_pred_data']
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.csv') and not file.startswith('.'):
                        csv_files.append(os.path.join(root, file))
    
    return sorted(csv_files)

def select_csv_file(csv_files):
    """
    Allow user to select a CSV file from the available options.
    
    Parameters:
    csv_files: list, paths to CSV files
    
    Returns:
    str, selected file path or 'combine' for combining multiple files
    """
    if not csv_files:
        print("No CSV files found in the current directory or subdirectories.")
        return None
    
    print("Available CSV files:")
    print("=" * 60)
    
    for i, file_path in enumerate(csv_files, 1):
        # Get file size for display
        try:
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            print(f"{i:2d}. {file_path} ({size_mb:.1f} MB)")
        except:
            print(f"{i:2d}. {file_path}")
    
    # Add option to combine word prediction files
    word_pred_files = [f for f in csv_files if any(pred in f.lower() for pred in ['animals_preds', 'clothes_preds', 'supermarket_preds'])]
    if word_pred_files:
        print(f"{len(csv_files) + 1:2d}. Combine word prediction files (animals_preds.csv, clothes_preds.csv, supermarket_preds.csv)")
    
    print("=" * 60)
    
    max_choice = len(csv_files) + (1 if word_pred_files else 0)
    
    while True:
        try:
            choice = input(f"Select a file (1-{max_choice}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(csv_files):
                selected_file = csv_files[choice_num - 1]
                print(f"Selected: {selected_file}")
                return selected_file
            elif choice_num == len(csv_files) + 1 and word_pred_files:
                print("Selected: Combine word prediction files")
                return 'combine'
            else:
                print(f"Please enter a number between 1 and {max_choice}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None

def load_and_process_word_pred_data(data_source="analysis_output/word_pred_data"):
    """
    Load and process word prediction data from either a single CSV file or multiple files.
    
    Parameters:
    data_source: str, either a single CSV file path or directory containing word prediction files
    
    Returns:
    pandas.DataFrame, processed data
    """
    if data_source == 'combine':
        # Original behavior - combine specific files
        data_dir = "analysis_output/word_pred_data"
        files = ['animals_preds.csv', 'clothes_preds.csv', 'supermarket_preds.csv']
        all_data = []
        
        for file in files:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                all_data.append(df)
                print(f"Loaded {file}: {len(df)} rows")
            else:
                print(f"Warning: {file} not found")
        
        if not all_data:
            raise ValueError("No word prediction files found")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
    elif os.path.isfile(data_source):
        # Load single CSV file
        print(f"Loading single file: {data_source}")
        combined_df = pd.read_csv(data_source)
        print(f"Loaded: {len(combined_df)} rows")
        
    else:
        raise ValueError(f"Invalid data source: {data_source}")
    
    # Check for required columns
    required_cols = ['text', 'predicted']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(combined_df.columns)}")
        return combined_df
    
    # Apply improved normalization
    combined_df['text_improved_norm'] = combined_df['text'].apply(improve_normalization)
    combined_df['predicted_improved_norm'] = combined_df['predicted'].apply(improve_normalization)
    
    # Create improved correct column
    combined_df['correct_improved'] = (
        combined_df['text_improved_norm'] == combined_df['predicted_improved_norm']
    )
    
    # Remove rows where index is 0 (always NaN for predictions according to notes)
    if 'index' in combined_df.columns:
        combined_df = combined_df[combined_df['index'] != 0].copy()
    
    # Remove rows with NaN predictions
    combined_df = combined_df.dropna(subset=['predicted']).copy()
    
    print(f"Processed dataset: {len(combined_df)} rows after processing")
    
    if 'category' in combined_df.columns:
        print(f"Categories: {combined_df['category'].value_counts().to_dict()}")
    
    return combined_df

def calculate_accuracy_by_word_index(df, accuracy_col='correct_improved', window_size=5):
    """
    Calculate accuracy by word index windows with SEM.
    
    Parameters:
    df: pandas.DataFrame, processed word prediction data
    accuracy_col: str, column name for accuracy calculation
    window_size: int, size of word index windows
    
    Returns:
    pandas.DataFrame, accuracy by word index windows with SEM
    """
    # Create word index windows
    df = df.copy()
    df['index_window'] = ((df['index'] - 1) // window_size) * window_size + 1
    df['index_window_label'] = df['index_window'].astype(str) + '-' + (df['index_window'] + window_size - 1).astype(str)
    
    # Calculate accuracy by window
    accuracy_by_window = df.groupby('index_window')[accuracy_col].agg(['mean', 'count', 'sem']).reset_index()
    accuracy_by_window.columns = ['index_window', 'accuracy', 'count', 'sem']
    
    # Create window labels for plotting
    accuracy_by_window['window_label'] = (accuracy_by_window['index_window'].astype(str) + '-' + 
                                         (accuracy_by_window['index_window'] + window_size - 1).astype(str))
    
    accuracy_by_window = accuracy_by_window.round(3)
    
    return accuracy_by_window

def plot_accuracy_by_word_index(accuracy_df, output_dir="analysis_output/plots/word_prediction", timestamp=None):
    """
    Create visualization of word prediction accuracy by word index windows (similar to switch prediction plots).
    
    Parameters:
    accuracy_df: pandas.DataFrame, accuracy data by word index windows with SEM
    output_dir: str, directory to save plots
    timestamp: str, timestamp for filename
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the plot - simpler style similar to switch prediction plots
    plt.figure(figsize=(10, 6))
    
    # Use window midpoints for x-axis
    x_pos = range(len(accuracy_df))
    
    # Plot accuracy line with SEM shading
    plt.plot(x_pos, accuracy_df['accuracy'], 
             color='steelblue', linewidth=2, marker='o', markersize=4)
    
    # Add SEM shading
    plt.fill_between(x_pos, 
                     accuracy_df['accuracy'] - accuracy_df['sem'],
                     accuracy_df['accuracy'] + accuracy_df['sem'],
                     alpha=0.3, color='steelblue')
    
    # Customize the plot
    plt.xlabel('Word Index Window', fontsize=12)
    plt.ylabel('Word Prediction Accuracy', fontsize=12)
    plt.title('Word Prediction Accuracy by Word Index Window', fontsize=14)
    
    # Set x-axis labels to show window ranges
    plt.xticks(x_pos, accuracy_df['window_label'], rotation=45)
    
    # Set y-axis limits
    plt.ylim(0, max(0.3, accuracy_df['accuracy'].max() + accuracy_df['sem'].max() + 0.05))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"word_prediction_accuracy_by_window_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    
    plt.show()

def generate_summary_statistics(df, accuracy_df, output_dir="analysis_output/statistics/word_prediction", timestamp=None):
    """
    Generate and save summary statistics.
    
    Parameters:
    df: pandas.DataFrame, full processed data
    accuracy_df: pandas.DataFrame, accuracy by word index
    output_dir: str, directory to save statistics
    timestamp: str, timestamp for filename
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall statistics
    stats = {
        'total_predictions': len(df),
        'overall_accuracy_original': df['correct'].mean() if 'correct' in df.columns else np.nan,
        'overall_accuracy_improved': df['correct_improved'].mean(),
        'unique_participants': df['playerID'].nunique(),
        'unique_sequences': df['sourceParticipantId'].nunique(),
        'categories': df['category'].value_counts().to_dict(),
        'max_word_index': df['index'].max(),
        'min_word_index': df['index'].min(),
        'accuracy_by_category': df.groupby('category')['correct_improved'].mean().to_dict()
    }
    
    # Save accuracy by word index windows
    accuracy_filename = f"word_prediction_by_window_{timestamp}.csv"
    accuracy_path = os.path.join(output_dir, accuracy_filename)
    accuracy_df.to_csv(accuracy_path, index=False)
    print(f"Accuracy by window saved to: {accuracy_path}")
    
    # Print summary
    print("\n=== WORD PREDICTION ANALYSIS SUMMARY ===")
    print(f"Total predictions analyzed: {stats['total_predictions']}")
    print(f"Overall accuracy (original): {stats['overall_accuracy_original']:.3f}")
    print(f"Overall accuracy (improved): {stats['overall_accuracy_improved']:.3f}")
    print(f"Unique participants: {stats['unique_participants']}")
    print(f"Word index range: {stats['min_word_index']} - {stats['max_word_index']}")
    print(f"Categories: {stats['categories']}")
    print(f"Accuracy by category: {stats['accuracy_by_category']}")
    
    return stats

def main():
    """Main analysis function."""
    print("Word Prediction Analysis")
    print("=" * 60)
    
    # Find available CSV files
    csv_files = find_csv_files()
    
    # Let user select file or combine option
    data_source = select_csv_file(csv_files)
    if data_source is None:
        return
    
    # Load and process data
    try:
        df = load_and_process_word_pred_data(data_source)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Check if we have the minimum required columns for word prediction analysis
    if 'index' not in df.columns:
        print("Warning: 'index' column not found. Word index analysis may not work properly.")
        print("Available columns:", list(df.columns))
        
        # Ask user if they want to continue
        try:
            continue_choice = input("Continue anyway? (y/n): ").strip().lower()
            if continue_choice != 'y':
                return
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    # Calculate accuracy by word index (if index column exists)
    if 'index' in df.columns:
        accuracy_df = calculate_accuracy_by_word_index(df)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visualization
        plot_accuracy_by_word_index(accuracy_df, timestamp=timestamp)
        
        # Generate summary statistics
        stats = generate_summary_statistics(df, accuracy_df, timestamp=timestamp)
    else:
        # Just generate basic statistics without word index analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("\n=== BASIC WORD PREDICTION ANALYSIS ===")
        print(f"Total predictions analyzed: {len(df)}")
        if 'correct_improved' in df.columns:
            print(f"Overall accuracy (improved): {df['correct_improved'].mean():.3f}")
        if 'playerID' in df.columns:
            print(f"Unique participants: {df['playerID'].nunique()}")
        if 'category' in df.columns:
            print(f"Categories: {df['category'].value_counts().to_dict()}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()