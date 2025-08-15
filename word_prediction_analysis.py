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

def load_and_process_word_pred_data(data_dir="analysis_output/word_pred_data"):
    """
    Load and combine all word prediction CSV files with improved processing.
    
    Parameters:
    data_dir: str, directory containing word prediction data
    
    Returns:
    pandas.DataFrame, combined and processed data
    """
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
        raise ValueError("No data files found")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Apply improved normalization
    combined_df['text_improved_norm'] = combined_df['text'].apply(improve_normalization)
    combined_df['predicted_improved_norm'] = combined_df['predicted'].apply(improve_normalization)
    
    # Create improved correct column
    combined_df['correct_improved'] = (
        combined_df['text_improved_norm'] == combined_df['predicted_improved_norm']
    )
    
    # Remove rows where index is 0 (always NaN for predictions according to notes)
    combined_df = combined_df[combined_df['index'] != 0].copy()
    
    # Remove rows with NaN predictions
    combined_df = combined_df.dropna(subset=['predicted']).copy()
    
    print(f"Combined dataset: {len(combined_df)} rows after processing")
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

def plot_accuracy_by_word_index(accuracy_df, output_dir="analysis_output/plots", timestamp=None):
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

def generate_summary_statistics(df, accuracy_df, output_dir="analysis_output/statistics", timestamp=None):
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
    print("Starting Word Prediction Analysis...")
    
    # Load and process data
    try:
        df = load_and_process_word_pred_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate accuracy by word index
    accuracy_df = calculate_accuracy_by_word_index(df)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create visualization
    plot_accuracy_by_word_index(accuracy_df, timestamp=timestamp)
    
    # Generate summary statistics
    stats = generate_summary_statistics(df, accuracy_df, timestamp=timestamp)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()