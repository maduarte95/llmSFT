"""
Results Summary Generator
=========================

This script generates a readable summary of the LLM switch analysis results.

Usage:
    uv run python generate_results_summary.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def find_summary_files(output_dir="analysis_output"):
    """
    Find all summary statistics files in the output directory and subdirectories.
    """
    stats_dir = os.path.join(output_dir, "statistics")
    summary_files = []
    
    # Walk through all directories to find summary statistics files
    for root, dirs, files in os.walk(stats_dir):
        for f in files:
            if f.startswith("summary_statistics_"):
                full_path = os.path.join(root, f)
                # Get relative path from stats_dir for display
                rel_path = os.path.relpath(full_path, stats_dir)
                summary_files.append((full_path, rel_path, f))
    
    return summary_files

def choose_summary_file(output_dir="analysis_output"):
    """
    Let user choose which summary statistics file to analyze.
    """
    summary_files = find_summary_files(output_dir)
    
    if not summary_files:
        raise FileNotFoundError("No summary statistics files found")
    
    print("Found summary statistics files:")
    print("=" * 60)
    
    for i, (full_path, rel_path, filename) in enumerate(summary_files, 1):
        # Get file modification time for display
        mod_time = datetime.fromtimestamp(os.path.getmtime(full_path))
        print(f"{i}. {rel_path}")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    while True:
        try:
            choice = input(f"Enter your choice (1-{len(summary_files)}) or 'latest' for most recent: ").strip().lower()
            
            if choice == 'latest':
                # Get the most recent file by modification time
                latest_file = max(summary_files, key=lambda x: os.path.getmtime(x[0]))
                return latest_file[0], latest_file[2]
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(summary_files):
                selected = summary_files[choice_num - 1]
                return selected[0], selected[2]
            else:
                print(f"Please enter a number between 1 and {len(summary_files)}")
        
        except ValueError:
            print("Please enter a valid number or 'latest'")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None, None

def load_chosen_results(output_dir="analysis_output"):
    """
    Load the user-chosen analysis results.
    """
    file_path, filename = choose_summary_file(output_dir)
    
    if file_path is None:
        return None
    
    # Load the data
    summary_stats = pd.read_csv(file_path)
    
    print(f"\nLoaded results from: {filename}")
    print(f"Full path: {file_path}")
    return summary_stats

def create_results_summary(summary_stats):
    """
    Create a readable summary of the results.
    """
    print("=" * 80)
    print("LLM SWITCH ANALYSIS RESULTS SUMMARY")
    print("=" * 80)
    
    # Get accuracy data
    accuracy_data = summary_stats[summary_stats['metric_type'] == 'accuracy'].copy()
    
    # Overall performance (across all categories)
    print("\n1. OVERALL PERFORMANCE (All Categories Combined)")
    print("-" * 50)
    
    overall_means = accuracy_data.groupby('method')['accuracy_mean'].mean()
    overall_sems = accuracy_data.groupby('method')['accuracy_sem'].mean()
    
    # Group methods
    human_methods = [m for m in overall_means.index if 'Human' in m]
    llm_methods = [m for m in overall_means.index if 'LLM' in m]
    baseline_methods = [m for m in overall_means.index if 'Baseline' in m]
    
    print("\nHuman Performance:")
    for method in human_methods:
        print(f"  {method:<20}: {overall_means[method]:.3f} ± {overall_sems[method]:.3f}")
    
    print("\nLLM Performance:")
    for method in llm_methods:
        print(f"  {method:<20}: {overall_means[method]:.3f} ± {overall_sems[method]:.3f}")
    
    print("\nBaseline Performance:")
    for method in baseline_methods:
        print(f"  {method:<20}: {overall_means[method]:.3f} ± {overall_sems[method]:.3f}")
    
    # Performance by category
    print("\n\n2. PERFORMANCE BY CATEGORY")
    print("-" * 50)
    
    categories = accuracy_data['category'].unique()
    for category in sorted(categories):
        print(f"\n{category.upper()}:")
        cat_data = accuracy_data[accuracy_data['category'] == category]
        
        for method in ['Human_Retrospective', 'Human_Predicted', 'LLM_Retrospective', 'LLM_Predicted']:
            if method in cat_data['method'].values:
                row = cat_data[cat_data['method'] == method].iloc[0]
                print(f"  {method:<20}: {row['accuracy_mean']:.3f} ± {row['accuracy_sem']:.3f} (n={int(row['accuracy_count'])})")
    
    # Key comparisons
    print("\n\n3. KEY COMPARISONS")
    print("-" * 50)
    
    print("\nPrediction Task (Human vs LLM):")
    human_pred = overall_means.get('Human_Predicted', 0)
    llm_pred = overall_means.get('LLM_Predicted', 0)
    diff_pred = llm_pred - human_pred
    print(f"  Human Predicted:  {human_pred:.3f}")
    print(f"  LLM Predicted:    {llm_pred:.3f}")
    print(f"  Difference:       {diff_pred:+.3f} ({'LLM better' if diff_pred > 0 else 'Human better'})")
    
    print("\nIdentification Task (Human vs LLM):")
    human_retro = overall_means.get('Human_Retrospective', 0)
    llm_retro = overall_means.get('LLM_Retrospective', 0)
    diff_retro = llm_retro - human_retro
    print(f"  Human Retrospective: {human_retro:.3f}")
    print(f"  LLM Retrospective:   {llm_retro:.3f}")
    print(f"  Difference:          {diff_retro:+.3f} ({'LLM better' if diff_retro > 0 else 'Human better'})")
    
    print("\nPrediction vs Identification (within method):")
    print(f"  Human: Retro={human_retro:.3f}, Pred={human_pred:.3f}, Diff={human_retro-human_pred:+.3f}")
    print(f"  LLM:   Retro={llm_retro:.3f}, Pred={llm_pred:.3f}, Diff={llm_retro-llm_pred:+.3f}")
    
    # Performance vs baselines
    print("\n\n4. PERFORMANCE VS BASELINES")
    print("-" * 50)
    
    zeror_baseline = overall_means.get('ZeroR_Baseline', 0)
    random_baseline = overall_means.get('Random_Baseline', 0)
    
    print(f"\nZeroR Baseline (Majority Class): {zeror_baseline:.3f}")
    print(f"Random Baseline:                 {random_baseline:.3f}")
    print("\nMethod performance vs ZeroR:")
    
    for method in ['Human_Predicted', 'Human_Retrospective', 'LLM_Predicted', 'LLM_Retrospective']:
        if method in overall_means.index:
            diff = overall_means[method] - zeror_baseline
            status = "Above" if diff > 0 else "Below"
            print(f"  {method:<20}: {overall_means[method]:.3f} ({status} baseline by {abs(diff):.3f})")
    
    # Sample sizes
    print("\n\n5. SAMPLE SIZES")
    print("-" * 50)
    
    print("\nParticipants per category:")
    for category in sorted(categories):
        cat_data = accuracy_data[accuracy_data['category'] == category]
        n_participants = cat_data['accuracy_count'].iloc[0] if len(cat_data) > 0 else 0
        print(f"  {category:<20}: {int(n_participants)} participants")
    
    total_participants = accuracy_data['accuracy_count'].sum() / len(accuracy_data['method'].unique())
    print(f"\nTotal participants: {int(total_participants)}")
    
    return overall_means

def create_metric_summary(summary_stats):
    """
    Create summary for all metrics (accuracy, precision, recall, F1).
    """
    print("\n\n6. ALL METRICS SUMMARY")
    print("-" * 50)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    methods = ['Human_Predicted', 'Human_Retrospective', 'LLM_Predicted', 'LLM_Retrospective']
    
    # Create summary table
    results_table = []
    
    for method in methods:
        row = {'Method': method}
        for metric in metrics:
            metric_data = summary_stats[summary_stats['metric_type'] == metric]
            if method in metric_data['method'].values:
                overall_mean = metric_data[metric_data['method'] == method][f'{metric}_mean'].mean()
                row[metric.title()] = f"{overall_mean:.3f}"
            else:
                row[metric.title()] = "N/A"
        results_table.append(row)
    
    # Print table
    print(f"\n{'Method':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 65)
    for row in results_table:
        print(f"{row['Method']:<25} {row['Accuracy']:<10} {row['Precision']:<10} {row['Recall']:<10} {row['F1']:<10}")

def main():
    """
    Generate and display results summary.
    """
    try:
        # Load results
        summary_stats = load_chosen_results()
        
        if summary_stats is None:
            print("No file selected. Exiting.")
            return
        
        # Create summaries
        overall_means = create_results_summary(summary_stats)
        create_metric_summary(summary_stats)
        
        print("\n\n7. FILES GENERATED")
        print("-" * 50)
        print("\nPlots created:")
        print("  - Overall comparison plots (accuracy, precision, recall, F1)")
        print("  - Category-wise prediction plots")  
        print("  - Category-wise identification plots")
        print("  - Accuracy by word index window plot")
        print("  - Precision by word index window plot")
        print("  - Recall by word index window plot") 
        print("  - F1 by word index window plot")
        print("\nStatistics files:")
        print("  - Participant-level metrics")
        print("  - Summary statistics by category and method")
        print("  - Window-based data for all metrics (accuracy, precision, recall, F1)")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    main()