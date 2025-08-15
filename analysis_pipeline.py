"""
LLM Switch Analysis Pipeline
=============================

This script analyzes the accuracies of LLM vs human switch detection (both predictive and retrospective)
compared to ground truth switches, including visualization and basic statistics.

Requirements:
    - pandas
    - numpy  
    - matplotlib
    - seaborn
    - scikit-learn

Usage:
    uv run python analysis_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class SwitchAnalysisPipeline:
    def __init__(self, data_path, output_dir="analysis_output"):
        """
        Initialize the analysis pipeline.
        
        Parameters:
        data_path: str, path to the CSV file with switch data
        output_dir: str, directory to save outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "statistics"), exist_ok=True)
        
        # Define switch column mappings
        self.switch_columns = {
            'Human_Retrospective': 'switch',
            'Human_Predicted': 'predictedSwitch_shifted_int', 
            'LLM_Retrospective': 'switchLLM',
            'LLM_Predicted': 'predicted_switch_llm'
        }
        
        self.data = None
        self.filtered_data = None
        
    def load_and_filter_data(self):
        """
        Load data and apply filtering criteria based on the notebook.
        """
        print("=" * 60)
        print("LOADING AND FILTERING DATA")
        print("=" * 60)
        
        # Load the data
        self.data = pd.read_csv(self.data_path)
        print(f"Original data shape: {self.data.shape}")
        print(f"Unique players: {self.data['playerID'].nunique()}")
        print(f"Categories: {list(self.data['category'].unique())}")
        
        # Check for required columns
        required_cols = ['switch_ground_truth'] + list(self.switch_columns.values())
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
        
        # Filter criteria from notebook:
        # 1. Remove word_index 0 (first word doesn't have a switch)
        # 2. Remove rows with missing predictions (-1 or NaN)
        # 3. Filter out participants with extreme switch rates
        # 4. Remove duplicates by sourceParticipantId
        
        print("\nApplying filtering criteria...")
        
        # Step 1: Remove word_index 0
        data_filtered = self.data[self.data['word_index'] > 0].copy()
        print(f"After removing word_index 0: {data_filtered.shape}")
        
        # Step 2: Convert columns to numeric and handle missing values
        numeric_cols = ['switch_ground_truth'] + list(self.switch_columns.values())
        for col in numeric_cols:
            if col in data_filtered.columns:
                data_filtered[col] = pd.to_numeric(data_filtered[col], errors='coerce')
        
        # Remove rows with missing ground truth
        data_filtered = data_filtered.dropna(subset=['switch_ground_truth'])
        print(f"After removing missing ground truth: {data_filtered.shape}")
        
        # Remove rows with -1 in predictive columns
        for col in ['predictedSwitch_shifted_int', 'predicted_switch_llm']:
            if col in data_filtered.columns:
                initial_count = len(data_filtered)
                data_filtered = data_filtered[data_filtered[col] != -1]
                data_filtered = data_filtered.dropna(subset=[col])
                print(f"After cleaning {col}: {len(data_filtered)} (removed {initial_count - len(data_filtered)})")
        
        # Step 3: Filter out participants with extreme switch rates or zero switches
        self._filter_extreme_participants(data_filtered)
        
        # Step 4: Remove duplicates by sourceParticipantId (random selection)
        self._remove_duplicates()
        
        print(f"\nFinal filtered data shape: {self.filtered_data.shape}")
        print(f"Final number of participants: {self.filtered_data['playerID'].nunique()}")
        print("Participants per category:")
        print(self.filtered_data.groupby('category')['playerID'].nunique())
        
        # Save filtered data
        filtered_path = os.path.join(self.output_dir, f"filtered_data_{self.timestamp}.csv")
        self.filtered_data.to_csv(filtered_path, index=False)
        print(f"\nFiltered data saved to: {filtered_path}")
        
    def _filter_extreme_participants(self, data):
        """
        Filter out participants with extreme switch rates or zero switches.
        """
        print("\nFiltering extreme participants...")
        
        # Calculate switch rates per participant
        participant_stats = []
        for player_id in data['playerID'].unique():
            player_data = data[data['playerID'] == player_id]
            
            if len(player_data) > 0:
                category = player_data['category'].iloc[0]
                total_words = len(player_data)
                
                stats = {
                    'playerID': player_id,
                    'category': category,
                    'total_words': total_words
                }
                
                # Calculate switch counts and rates for each type
                for method_name, col_name in self.switch_columns.items():
                    if col_name in player_data.columns:
                        switches = player_data[col_name].sum()
                        rate = switches / total_words
                        stats[f'{method_name}_switches'] = switches
                        stats[f'{method_name}_rate'] = rate
                
                # Ground truth
                gt_switches = player_data['switch_ground_truth'].sum()
                gt_rate = gt_switches / total_words
                stats['gt_switches'] = gt_switches
                stats['gt_rate'] = gt_rate
                
                participant_stats.append(stats)
        
        switch_rates_df = pd.DataFrame(participant_stats)
        
        # Detect outliers using IQR method for rates
        def detect_outliers_iqr(data, column_name):
            Q1 = data[column_name].quantile(0.25)
            Q3 = data[column_name].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            return data[data[column_name] > upper_bound]['playerID'].tolist()
        
        # Find participants with extreme rates
        extreme_participants = set()
        rate_columns = [col for col in switch_rates_df.columns if col.endswith('_rate')]
        
        for col in rate_columns:
            if col in switch_rates_df.columns:
                outliers = detect_outliers_iqr(switch_rates_df, col)
                extreme_participants.update(outliers)
                print(f"High outliers in {col}: {len(outliers)}")
        
        # Find participants with zero switches in any measure
        switch_columns = [col for col in switch_rates_df.columns if col.endswith('_switches')]
        for col in switch_columns:
            if col in switch_rates_df.columns:
                zero_switches = switch_rates_df[switch_rates_df[col] == 0]['playerID'].tolist()
                extreme_participants.update(zero_switches)
                print(f"Zero switches in {col}: {len(zero_switches)}")
        
        print(f"Total participants to remove: {len(extreme_participants)}")
        
        # Apply filtering
        self.filtered_data = data[~data['playerID'].isin(extreme_participants)].copy()
        print(f"Participants after filtering: {self.filtered_data['playerID'].nunique()}")
        
    def _remove_duplicates(self):
        """
        Remove duplicates by randomly selecting one playerID per sourceParticipantId-category combination.
        """
        print("\nRemoving duplicates...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        duplicates_removed = 0
        for category in self.filtered_data['category'].unique():
            category_data = self.filtered_data[self.filtered_data['category'] == category]
            
            # Find sourceParticipantIds with multiple playerIDs
            source_player_counts = category_data.groupby('sourceParticipantId')['playerID'].nunique()
            duplicated_sources = source_player_counts[source_player_counts > 1].index
            
            for source_id in duplicated_sources:
                players_for_source = category_data[category_data['sourceParticipantId'] == source_id]['playerID'].unique()
                
                # Randomly select one to keep
                player_to_keep = np.random.choice(players_for_source)
                players_to_remove = [p for p in players_for_source if p != player_to_keep]
                
                # Remove the unwanted playerIDs
                for player_to_remove in players_to_remove:
                    remove_mask = (
                        (self.filtered_data['sourceParticipantId'] == source_id) & 
                        (self.filtered_data['category'] == category) & 
                        (self.filtered_data['playerID'] == player_to_remove)
                    )
                    self.filtered_data = self.filtered_data[~remove_mask]
                    duplicates_removed += 1
        
        print(f"Duplicate combinations removed: {duplicates_removed}")
        
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate classification metrics.
        """
        # Remove NaN values
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'accuracy': np.nan,
                'precision': np.nan, 
                'recall': np.nan,
                'f1': np.nan,
                'n_samples': 0
            }
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_clean, y_pred_clean)
        precision = precision_score(y_true_clean, y_pred_clean, zero_division=0)
        recall = recall_score(y_true_clean, y_pred_clean, zero_division=0)
        f1 = f1_score(y_true_clean, y_pred_clean, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': len(y_true_clean)
        }
    
    def calculate_participant_metrics(self):
        """
        Calculate per-participant metrics for all methods.
        """
        print("=" * 60)
        print("CALCULATING PER-PARTICIPANT METRICS")
        print("=" * 60)
        
        participant_metrics = []
        
        for player_id in self.filtered_data['playerID'].unique():
            player_data = self.filtered_data[self.filtered_data['playerID'] == player_id]
            
            if len(player_data) == 0:
                continue
                
            category = player_data['category'].iloc[0]
            y_true = player_data['switch_ground_truth'].values
            
            # Calculate metrics for each method
            for method_name, col_name in self.switch_columns.items():
                if col_name in player_data.columns:
                    y_pred = player_data[col_name].values
                    metrics = self.calculate_metrics(y_true, y_pred)
                    
                    participant_metrics.append({
                        'playerID': player_id,
                        'category': category,
                        'method': method_name,
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1': metrics['f1'],
                        'n_samples': metrics['n_samples']
                    })
        
        self.participant_metrics = pd.DataFrame(participant_metrics)
        
        # Calculate baselines
        self._calculate_baselines()
        
        print(f"Metrics calculated for {len(self.participant_metrics)} method-participant combinations")
        
        # Save participant metrics
        metrics_path = os.path.join(self.output_dir, "statistics", f"participant_metrics_{self.timestamp}.csv")
        self.participant_metrics.to_csv(metrics_path, index=False)
        print(f"Participant metrics saved to: {metrics_path}")
        
    def _calculate_baselines(self):
        """
        Calculate baseline performance (chance and random rate).
        """
        print("Calculating baseline performance...")
        
        baseline_metrics = []
        
        for player_id in self.filtered_data['playerID'].unique():
            player_data = self.filtered_data[self.filtered_data['playerID'] == player_id]
            
            if len(player_data) > 0:
                category = player_data['category'].iloc[0]
                y_true = player_data['switch_ground_truth'].values
                
                # Base rate (proportion of switches)
                base_rate = y_true.mean()
                
                # ZeroR baseline (always predict majority class)
                majority_pred = np.full_like(y_true, 1 if base_rate >= 0.5 else 0)
                zeror_metrics = self.calculate_metrics(y_true, majority_pred)
                
                # Random baseline (predict according to base rate)
                np.random.seed(42)  # For reproducibility
                random_pred = np.random.binomial(1, base_rate, size=len(y_true))
                random_metrics = self.calculate_metrics(y_true, random_pred)
                
                baseline_metrics.extend([
                    {
                        'playerID': player_id,
                        'category': category,
                        'method': 'ZeroR_Baseline',
                        'accuracy': zeror_metrics['accuracy'],
                        'precision': zeror_metrics['precision'],
                        'recall': zeror_metrics['recall'],
                        'f1': zeror_metrics['f1'],
                        'n_samples': zeror_metrics['n_samples']
                    },
                    {
                        'playerID': player_id,
                        'category': category,
                        'method': 'Random_Baseline',
                        'accuracy': random_metrics['accuracy'],
                        'precision': random_metrics['precision'],
                        'recall': random_metrics['recall'],
                        'f1': random_metrics['f1'],
                        'n_samples': random_metrics['n_samples']
                    }
                ])
        
        # Add baselines to participant metrics
        baseline_df = pd.DataFrame(baseline_metrics)
        self.participant_metrics = pd.concat([self.participant_metrics, baseline_df], ignore_index=True)
        
    def create_summary_statistics(self):
        """
        Create summary statistics by category and method.
        """
        print("Creating summary statistics...")
        
        # Group by category and method
        summary_stats = []
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            metric_summary = self.participant_metrics.groupby(['category', 'method']).agg({
                metric: ['mean', 'std', 'sem', 'count']
            }).round(4)
            
            # Flatten column names
            metric_summary.columns = [f'{metric}_{stat}' for stat in ['mean', 'std', 'sem', 'count']]
            metric_summary = metric_summary.reset_index()
            metric_summary['metric_type'] = metric
            
            summary_stats.append(metric_summary)
        
        self.summary_stats = pd.concat(summary_stats, ignore_index=True)
        
        # Save summary statistics
        summary_path = os.path.join(self.output_dir, "statistics", f"summary_statistics_{self.timestamp}.csv")
        self.summary_stats.to_csv(summary_path, index=False)
        print(f"Summary statistics saved to: {summary_path}")
        
    def run_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print("Starting LLM Switch Analysis Pipeline")
        print("=" * 60)
        
        # Step 1: Load and filter data
        self.load_and_filter_data()
        
        # Step 2: Calculate metrics
        self.calculate_participant_metrics()
        
        # Step 3: Create summary statistics
        self.create_summary_statistics()
        
        # Step 4: Create visualizations
        self.create_all_visualizations()
        
        print("\nAnalysis pipeline completed successfully!")
        print(f"All outputs saved to: {self.output_dir}")
    
    def create_all_visualizations(self):
        """
        Create all required visualizations.
        """
        print("=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create visualizations for each metric
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            print(f"\nCreating plots for {metric}...")
            
            # A. Overall comparison (switch prediction vs identification)
            self._plot_overall_comparison(metric)
            
            # B. Switch prediction by category
            self._plot_by_category(metric, task_type='prediction')
            
            # C. Switch identification by category  
            self._plot_by_category(metric, task_type='identification')
            
        # D. Accuracy by word index window (only for accuracy)
        self._plot_accuracy_by_word_index()
        
        # D2. Additional metrics by word index window  
        self._plot_metrics_by_word_index(['precision', 'recall', 'f1'])
        
        print("All visualizations created!")
    
    def _plot_overall_comparison(self, metric):
        """
        Create bar plot comparing switch prediction vs identification (overall).
        Plot A: Two sets of bars for prediction vs identification accuracy.
        """
        # Define method groupings
        prediction_methods = ['Human_Predicted', 'LLM_Predicted', 'Random_Baseline', 'ZeroR_Baseline']
        identification_methods = ['Human_Retrospective', 'LLM_Retrospective', 'Random_Baseline', 'ZeroR_Baseline']
        
        # Get metric data
        metric_data = self.summary_stats[self.summary_stats['metric_type'] == metric].copy()
        
        # Calculate overall means (across all categories)
        overall_stats = self.participant_metrics.groupby('method').agg({
            metric: ['mean', 'sem']
        }).round(4)
        overall_stats.columns = [f'{metric}_{stat}' for stat in ['mean', 'sem']]
        overall_stats = overall_stats.reset_index()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Define colors
        colors = {
            'Human_Predicted': '#ff7f7f',      # Light red
            'Human_Retrospective': '#ff4444',   # Red  
            'LLM_Predicted': '#7fbfff',        # Light blue
            'LLM_Retrospective': '#4444ff',     # Blue
            'Random_Baseline': '#cccccc',       # Light gray
            'ZeroR_Baseline': '#999999'        # Dark gray
        }
        
        # Plot 1: Switch Prediction
        prediction_data = overall_stats[overall_stats['method'].isin(prediction_methods)]
        x_pos = np.arange(len(prediction_data))
        
        bars1 = ax1.bar(x_pos, prediction_data[f'{metric}_mean'], 
                       yerr=prediction_data[f'{metric}_sem'],
                       color=[colors[m] for m in prediction_data['method']],
                       capsize=5, alpha=0.8)
        
        ax1.set_title(f'Switch Prediction {metric.title()} (All Categories)', fontweight='bold')
        ax1.set_ylabel(f'{metric.title()}')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.replace('_', '\n') for m in prediction_data['method']], rotation=0)
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val, sem in zip(bars1, prediction_data[f'{metric}_mean'], prediction_data[f'{metric}_sem']):
            ax1.text(bar.get_x() + bar.get_width()/2, val + sem + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Switch Identification  
        identification_data = overall_stats[overall_stats['method'].isin(identification_methods)]
        x_pos = np.arange(len(identification_data))
        
        bars2 = ax2.bar(x_pos, identification_data[f'{metric}_mean'],
                       yerr=identification_data[f'{metric}_sem'], 
                       color=[colors[m] for m in identification_data['method']],
                       capsize=5, alpha=0.8)
        
        ax2.set_title(f'Switch Identification {metric.title()} (All Categories)', fontweight='bold')
        ax2.set_ylabel(f'{metric.title()}')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.replace('_', '\n') for m in identification_data['method']], rotation=0)
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val, sem in zip(bars2, identification_data[f'{metric}_mean'], identification_data[f'{metric}_sem']):
            ax2.text(bar.get_x() + bar.get_width()/2, val + sem + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", f"{metric}_overall_comparison_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Overall comparison plot saved: {plot_path}")
    
    def _plot_by_category(self, metric, task_type):
        """
        Create bar plots separated by category.
        Plot B: Switch prediction by category
        Plot C: Switch identification by category
        """
        # Define methods based on task type
        if task_type == 'prediction':
            methods = ['Human_Predicted', 'LLM_Predicted', 'Random_Baseline', 'ZeroR_Baseline']
            title_prefix = 'Switch Prediction'
        else:  # identification
            methods = ['Human_Retrospective', 'LLM_Retrospective', 'Random_Baseline', 'ZeroR_Baseline'] 
            title_prefix = 'Switch Identification'
        
        # Get metric data for these methods
        metric_data = self.summary_stats[
            (self.summary_stats['metric_type'] == metric) & 
            (self.summary_stats['method'].isin(methods))
        ].copy()
        
        # Get categories
        categories = sorted(self.filtered_data['category'].unique())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define colors
        colors = {
            'Human_Predicted': '#ff7f7f',
            'Human_Retrospective': '#ff4444', 
            'LLM_Predicted': '#7fbfff',
            'LLM_Retrospective': '#4444ff',
            'Random_Baseline': '#cccccc',
            'ZeroR_Baseline': '#999999'
        }
        
        # Bar width and positions
        n_methods = len(methods)
        bar_width = 0.8 / n_methods
        x = np.arange(len(categories))
        
        # Plot bars for each method
        for i, method in enumerate(methods):
            method_data = metric_data[metric_data['method'] == method]
            method_data = method_data.set_index('category').reindex(categories).reset_index()
            
            means = method_data[f'{metric}_mean'].fillna(0)
            sems = method_data[f'{metric}_sem'].fillna(0)
            
            bar_positions = x + (i - (n_methods-1)/2) * bar_width
            
            bars = ax.bar(bar_positions, means, bar_width, 
                         yerr=sems, capsize=3,
                         label=method.replace('_', ' '),
                         color=colors[method], alpha=0.8)
            
            # Add value labels
            for bar, val, sem in zip(bars, means, sems):
                if val > 0:  # Only label non-zero values
                    ax.text(bar.get_x() + bar.get_width()/2, val + sem + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Customize plot
        ax.set_title(f'{title_prefix} {metric.title()} by Category', fontweight='bold', fontsize=14)
        ax.set_ylabel(f'{metric.title()}', fontsize=12)
        ax.set_xlabel('Category', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", f"{metric}_{task_type}_by_category_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  {title_prefix} by category plot saved: {plot_path}")
    
    def _plot_accuracy_by_word_index(self):
        """
        Create line plot of accuracy by word index window.
        Plot D: Accuracy across word index windows with confidence intervals.
        """
        print("Creating accuracy by word index plot...")
        
        # Only use prediction methods for this analysis
        methods_to_plot = ['Human_Predicted', 'LLM_Predicted']
        
        # Get word index range
        word_indices = self.filtered_data['word_index'].values
        min_idx, max_idx = word_indices.min(), word_indices.max()
        
        # Create 5 equal windows
        window_edges = np.linspace(min_idx, max_idx, 6)
        window_centers = (window_edges[:-1] + window_edges[1:]) / 2
        window_labels = [f'{int(window_edges[i])}-{int(window_edges[i+1])}' for i in range(5)]
        
        # Calculate accuracy for each window and method
        window_results = []
        
        for method_name, col_name in self.switch_columns.items():
            if method_name not in methods_to_plot or col_name not in self.filtered_data.columns:
                continue
                
            method_accuracies = []
            
            for i in range(5):
                # Define window
                window_min = window_edges[i]
                window_max = window_edges[i+1]
                
                # Get data in this window
                window_data = self.filtered_data[
                    (self.filtered_data['word_index'] >= window_min) & 
                    (self.filtered_data['word_index'] < window_max)
                ]
                
                if len(window_data) == 0:
                    continue
                
                # Calculate per-participant accuracy in this window
                participant_accuracies = []
                
                for player_id in window_data['playerID'].unique():
                    player_window_data = window_data[window_data['playerID'] == player_id]
                    
                    if len(player_window_data) > 0:
                        y_true = player_window_data['switch_ground_truth'].values
                        y_pred = player_window_data[col_name].values
                        
                        # Remove NaN values
                        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
                        if mask.sum() > 0:
                            accuracy = (y_true[mask] == y_pred[mask]).mean()
                            participant_accuracies.append(accuracy)
                
                if len(participant_accuracies) > 0:
                    window_results.append({
                        'method': method_name,
                        'window': i,
                        'window_center': window_centers[i],
                        'window_label': window_labels[i],
                        'accuracy': np.array(participant_accuracies),
                        'mean_accuracy': np.mean(participant_accuracies),
                        'sem_accuracy': np.std(participant_accuracies) / np.sqrt(len(participant_accuracies)),
                        'n_participants': len(participant_accuracies)
                    })
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors for methods
        method_colors = {
            'Human_Predicted': '#ff7f7f',
            'LLM_Predicted': '#7fbfff'
        }
        
        # Plot lines with confidence intervals
        for method in methods_to_plot:
            method_results = [r for r in window_results if r['method'] == method]
            
            if len(method_results) > 0:
                x_vals = [r['window_center'] for r in method_results]
                y_vals = [r['mean_accuracy'] for r in method_results]
                yerr_vals = [r['sem_accuracy'] for r in method_results]
                
                # Plot line
                ax.plot(x_vals, y_vals, 'o-', color=method_colors[method], 
                       linewidth=2, markersize=6, label=method.replace('_', ' '))
                
                # Plot confidence interval
                ax.fill_between(x_vals, 
                               np.array(y_vals) - np.array(yerr_vals),
                               np.array(y_vals) + np.array(yerr_vals),
                               alpha=0.3, color=method_colors[method])
        
        # Customize plot
        ax.set_title('Accuracy by Word Index Window', fontweight='bold', fontsize=14)
        ax.set_xlabel('Word Index', fontsize=12)
        ax.set_ylabel('Mean Accuracy per Participant', fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-tick labels to show window ranges
        ax.set_xticks(window_centers)
        ax.set_xticklabels(window_labels, rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", f"accuracy_by_word_index_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Accuracy by word index plot saved: {plot_path}")
        
        # Save window results as CSV
        window_results_flat = []
        for result in window_results:
            for acc in result['accuracy']:
                window_results_flat.append({
                    'method': result['method'],
                    'window': result['window'],
                    'window_label': result['window_label'],
                    'window_center': result['window_center'],
                    'accuracy': acc
                })
        
        window_df = pd.DataFrame(window_results_flat)
        window_path = os.path.join(self.output_dir, "statistics", f"accuracy_by_window_{self.timestamp}.csv")
        window_df.to_csv(window_path, index=False)
        print(f"  Window accuracy data saved: {window_path}")
    
    def _plot_metrics_by_word_index(self, metrics):
        """
        Create line plots for precision, recall, and F1 by word index window.
        Similar to accuracy plot but for other metrics.
        """
        print("Creating additional metrics by word index plots...")
        
        # Only use prediction methods for this analysis
        methods_to_plot = ['Human_Predicted', 'LLM_Predicted']
        
        # Get word index range
        word_indices = self.filtered_data['word_index'].values
        min_idx, max_idx = word_indices.min(), word_indices.max()
        
        # Create 5 equal windows
        window_edges = np.linspace(min_idx, max_idx, 6)
        window_centers = (window_edges[:-1] + window_edges[1:]) / 2
        window_labels = [f'{int(window_edges[i])}-{int(window_edges[i+1])}' for i in range(5)]
        
        # Colors for methods
        method_colors = {
            'Human_Predicted': '#ff7f7f',
            'LLM_Predicted': '#7fbfff'
        }
        
        for metric in metrics:
            print(f"  Creating {metric} by word index plot...")
            
            # Calculate metric for each window and method
            window_results = []
            
            for method_name, col_name in self.switch_columns.items():
                if method_name not in methods_to_plot or col_name not in self.filtered_data.columns:
                    continue
                    
                for i in range(5):
                    # Define window
                    window_min = window_edges[i]
                    window_max = window_edges[i+1]
                    
                    # Get data in this window
                    window_data = self.filtered_data[
                        (self.filtered_data['word_index'] >= window_min) & 
                        (self.filtered_data['word_index'] < window_max)
                    ]
                    
                    if len(window_data) == 0:
                        continue
                    
                    # Calculate per-participant metric in this window
                    participant_metrics = []
                    
                    for player_id in window_data['playerID'].unique():
                        player_window_data = window_data[window_data['playerID'] == player_id]
                        
                        if len(player_window_data) > 0:
                            y_true = player_window_data['switch_ground_truth'].values
                            y_pred = player_window_data[col_name].values
                            
                            # Calculate metric using the existing method
                            metric_result = self.calculate_metrics(y_true, y_pred)
                            metric_value = metric_result[metric]
                            
                            if not np.isnan(metric_value):
                                participant_metrics.append(metric_value)
                    
                    if len(participant_metrics) > 0:
                        window_results.append({
                            'method': method_name,
                            'window': i,
                            'window_center': window_centers[i],
                            'window_label': window_labels[i],
                            'metric_values': np.array(participant_metrics),
                            'mean_metric': np.mean(participant_metrics),
                            'sem_metric': np.std(participant_metrics) / np.sqrt(len(participant_metrics)),
                            'n_participants': len(participant_metrics)
                        })
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot lines with confidence intervals
            for method in methods_to_plot:
                method_results = [r for r in window_results if r['method'] == method]
                
                if len(method_results) > 0:
                    x_vals = [r['window_center'] for r in method_results]
                    y_vals = [r['mean_metric'] for r in method_results]
                    yerr_vals = [r['sem_metric'] for r in method_results]
                    
                    # Plot line
                    ax.plot(x_vals, y_vals, 'o-', color=method_colors[method], 
                           linewidth=2, markersize=6, label=method.replace('_', ' '))
                    
                    # Plot confidence interval
                    ax.fill_between(x_vals, 
                                   np.array(y_vals) - np.array(yerr_vals),
                                   np.array(y_vals) + np.array(yerr_vals),
                                   alpha=0.3, color=method_colors[method])
            
            # Customize plot
            ax.set_title(f'{metric.title()} by Word Index Window', fontweight='bold', fontsize=14)
            ax.set_xlabel('Word Index', fontsize=12)
            ax.set_ylabel(f'Mean {metric.title()} per Participant', fontsize=12)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set x-tick labels to show window ranges
            ax.set_xticks(window_centers)
            ax.set_xticklabels(window_labels, rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "plots", f"{metric}_by_word_index_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    {metric.title()} by word index plot saved: {plot_path}")
            
            # Save window results as CSV
            window_results_flat = []
            for result in window_results:
                for metric_val in result['metric_values']:
                    window_results_flat.append({
                        'method': result['method'],
                        'window': result['window'],
                        'window_label': result['window_label'],
                        'window_center': result['window_center'],
                        metric: metric_val
                    })
            
            if window_results_flat:
                window_df = pd.DataFrame(window_results_flat)
                window_path = os.path.join(self.output_dir, "statistics", f"{metric}_by_window_{self.timestamp}.csv")
                window_df.to_csv(window_path, index=False)
                print(f"    Window {metric} data saved: {window_path}")

def main():
    """
    Main function to run the analysis.
    """
    # Configuration
    data_path = "data/data_with_word_predictions_llama_3.3_70b_prediction_20250815_044406.csv"
    output_dir = "analysis_output"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please check the file path and try again.")
        return
    
    # Run analysis
    pipeline = SwitchAnalysisPipeline(data_path, output_dir)
    pipeline.run_analysis()

if __name__ == "__main__":
    main()