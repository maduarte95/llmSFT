"""
Statistical Significance Analysis for LLM Switch Detection
=======================================================

This script performs comprehensive statistical testing on F1 scores from the LLM switch analysis pipeline.
It evaluates assumptions for paired t-tests, selects appropriate statistical tests, performs pairwise 
comparisons, corrects for multiple comparisons, calculates effect sizes, and creates plots with 
significance markers.

Features:
- Assumption testing (normality via Shapiro-Wilk)
- Automatic test selection (paired t-test vs Wilcoxon signed-rank)
- Pairwise comparisons between all method combinations
- Holm-Bonferroni multiple comparisons correction
- Effect size calculations (Cohen's d and r)
- Visualization with statistical annotations using statannotations

Requirements:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scipy
    - statannotations

Usage:
    uv run python statistical_significance_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
import warnings
import os
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any
import argparse

# Import statannotations
try:
    from statannotations.Annotator import Annotator
    STATANNOTATIONS_AVAILABLE = True
except ImportError:
    print("Warning: statannotations not available. Install with: uv add statannotations")
    STATANNOTATIONS_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class StatisticalSignificanceAnalyzer:
    """
    Comprehensive statistical analysis for switch detection method comparisons.
    """
    
    def __init__(self, participant_metrics_path: str, output_dir: str = "significance_analysis_output"):
        """
        Initialize the statistical significance analyzer.
        
        Parameters:
        participant_metrics_path: str, path to participant_metrics.csv from analysis pipeline
        output_dir: str, directory to save outputs
        """
        self.participant_metrics_path = participant_metrics_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "statistics"), exist_ok=True)
        
        # Define methods to compare (excluding baselines for main comparisons)
        self.main_methods = [
            'Human_Retrospective',
            'Human_Predicted', 
            'LLM_Retrospective',
            'LLM_Predicted'
        ]
        
        # Include baselines for additional comparisons
        self.all_methods = self.main_methods + ['Random_Baseline', 'ZeroR_Baseline']
        
        # Initialize data containers
        self.data = None
        self.paired_data = None
        self.statistical_results = []
        
    def load_data(self):
        """
        Load and prepare participant metrics data for statistical analysis.
        """
        print("=" * 60)
        print("LOADING PARTICIPANT METRICS DATA")
        print("=" * 60)
        
        # Load data
        self.data = pd.read_csv(self.participant_metrics_path)
        print(f"Loaded data shape: {self.data.shape}")
        print(f"Available methods: {sorted(self.data['method'].unique())}")
        print(f"Available categories: {sorted(self.data['category'].unique())}")
        print(f"Unique participants: {self.data['playerID'].nunique()}")
        
        # Prepare paired data for statistical testing
        self._prepare_paired_data()
        
    def _prepare_paired_data(self):
        """
        Prepare data for paired statistical tests by organizing by playerID.
        """
        print("\nPreparing paired data for statistical analysis...")
        
        # Create pivot table with playerID as index, methods as columns
        pivot_data = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            # Pivot data: playerID × method
            metric_pivot = self.data.pivot_table(
                index='playerID', 
                columns='method', 
                values=metric, 
                aggfunc='first'  # Should be only one value per player-method combination
            )
            
            # Only keep participants who have data for all main methods
            complete_participants = metric_pivot.dropna(subset=self.main_methods)
            
            pivot_data[metric] = complete_participants
            
            print(f"  {metric}: {len(complete_participants)} participants with complete data")
        
        self.paired_data = pivot_data
        
        # Print sample size info
        total_complete = len(self.paired_data['f1'])
        print(f"\nTotal participants with complete F1 data: {total_complete}")
        
        if total_complete < 5:
            print("WARNING: Very small sample size may affect statistical power!")
        
    def test_normality_assumptions(self, metric: str = 'f1') -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Test normality assumptions for paired differences between methods.
        
        Parameters:
        metric: str, metric to test (default: 'f1')
        
        Returns:
        Dict with normality test results for each method pair
        """
        print(f"\n{'='*60}")
        print(f"TESTING NORMALITY ASSUMPTIONS FOR {metric.upper()}")
        print("="*60)
        
        normality_results = {}
        metric_data = self.paired_data[metric]
        
        # Test all pairwise combinations
        method_pairs = list(combinations(self.main_methods, 2))
        
        for method1, method2 in method_pairs:
            # Calculate paired differences
            diff = metric_data[method1] - metric_data[method2]
            diff_clean = diff.dropna()
            
            if len(diff_clean) < 3:
                print(f"  {method1} vs {method2}: Insufficient data (n={len(diff_clean)})")
                continue
            
            # Shapiro-Wilk test for normality
            if len(diff_clean) <= 5000:  # Shapiro-Wilk limitation
                stat, p_value = shapiro(diff_clean)
                test_name = "Shapiro-Wilk"
            else:
                # Use Anderson-Darling for larger samples
                from scipy.stats import anderson
                result = anderson(diff_clean, dist='norm')
                stat = result.statistic
                # Convert to approximate p-value (rough approximation)
                p_value = 0.05 if stat > result.critical_values[2] else 0.5
                test_name = "Anderson-Darling"
            
            # Interpret results
            is_normal = p_value > 0.05
            recommended_test = "Paired t-test" if is_normal else "Wilcoxon signed-rank"
            
            normality_results[(method1, method2)] = {
                'n_samples': len(diff_clean),
                'test_statistic': stat,
                'p_value': p_value,
                'is_normal': is_normal,
                'test_name': test_name,
                'recommended_test': recommended_test,
                'differences': diff_clean.values
            }
            
            print(f"  {method1} vs {method2}:")
            print(f"    {test_name}: statistic={stat:.4f}, p={p_value:.4f}")
            print(f"    Normal: {is_normal}, Recommended: {recommended_test}")
        
        # Print clear summary of assumption testing results
        self._print_assumption_summary(normality_results)
        
        # Create distribution plots
        self._create_distribution_plots(metric_data, normality_results, metric)
        
        return normality_results
    
    def test_normality_assumptions_targeted(self, metric: str, target_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Test normality assumptions for only the targeted pairs.
        """
        print(f"\n{'='*60}")
        print(f"TESTING NORMALITY ASSUMPTIONS FOR {metric.upper()} (TARGETED PAIRS)")
        print("="*60)
        
        normality_results = {}
        metric_data = self.paired_data[metric]
        
        for method1, method2 in target_pairs:
            # Calculate paired differences
            diff = metric_data[method1] - metric_data[method2]
            diff_clean = diff.dropna()
            
            if len(diff_clean) < 3:
                print(f"  {method1} vs {method2}: Insufficient data (n={len(diff_clean)})")
                continue
            
            # Shapiro-Wilk test for normality
            stat, p_value = shapiro(diff_clean)
            test_name = "Shapiro-Wilk"
            
            # Interpret results
            is_normal = p_value > 0.05
            recommended_test = "Paired t-test" if is_normal else "Wilcoxon signed-rank"
            
            normality_results[(method1, method2)] = {
                'n_samples': len(diff_clean),
                'test_statistic': stat,
                'p_value': p_value,
                'is_normal': is_normal,
                'test_name': test_name,
                'recommended_test': recommended_test,
                'differences': diff_clean.values
            }
            
            print(f"  {method1} vs {method2}:")
            print(f"    {test_name}: statistic={stat:.4f}, p={p_value:.4f}")
            print(f"    Normal: {is_normal}, Recommended: {recommended_test}")
        
        # Print clear summary of assumption testing results
        self._print_assumption_summary(normality_results)
        
        # Create distribution plots for targeted pairs only
        self._create_distribution_plots_targeted(metric, target_pairs, normality_results)
        
        return normality_results
    
    def _print_assumption_summary(self, normality_results: Dict[Tuple[str, str], Dict[str, Any]]):
        """
        Print a clear summary of which pairs meet normality assumptions.
        """
        print(f"\n{'='*60}")
        print("ASSUMPTION TESTING SUMMARY")
        print("="*60)
        
        # Separate results by assumption outcome
        normal_pairs = []
        non_normal_pairs = []
        
        for pair, results in normality_results.items():
            if results['is_normal']:
                normal_pairs.append((pair, results))
            else:
                non_normal_pairs.append((pair, results))
        
        print(f"\n[PASS] PAIRS MEETING NORMALITY ASSUMPTIONS (n={len(normal_pairs)}):")
        print("   -> Will use PAIRED T-TEST")
        if normal_pairs:
            for (method1, method2), results in normal_pairs:
                print(f"   * {method1} vs {method2}: p={results['p_value']:.4f}")
        else:
            print("   * None")
        
        print(f"\n[FAIL] PAIRS NOT MEETING NORMALITY ASSUMPTIONS (n={len(non_normal_pairs)}):")
        print("   -> Will use WILCOXON SIGNED-RANK TEST")
        if non_normal_pairs:
            for (method1, method2), results in non_normal_pairs:
                print(f"   * {method1} vs {method2}: p={results['p_value']:.4f}")
        else:
            print("   * None")
        
        print(f"\nNote: alpha = 0.05 significance level used for normality testing")
    
    def _create_distribution_plots(self, metric_data: pd.DataFrame, 
                                  normality_results: Dict[Tuple[str, str], Dict[str, Any]], 
                                  metric: str):
        """
        Create distribution plots for raw values and paired differences.
        """
        print(f"\n{'='*60}")
        print("CREATING DISTRIBUTION PLOTS")
        print("="*60)
        
        # 1. Raw F1 distributions by method
        self._plot_raw_distributions(metric_data, metric)
        
        # 2. Paired differences distributions
        self._plot_difference_distributions(normality_results, metric)
    
    def _plot_raw_distributions(self, metric_data: pd.DataFrame, metric: str):
        """
        Plot histograms of raw metric values for each method.
        """
        # Create subplots for each method
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        colors = {
            'Human_Retrospective': '#ff4444',   # Red  
            'Human_Predicted': '#ff7f7f',      # Light red
            'LLM_Retrospective': '#4444ff',     # Blue
            'LLM_Predicted': '#7fbfff',        # Light blue
            'Random_Baseline': '#cccccc',       # Light gray
            'ZeroR_Baseline': '#999999'        # Dark gray
        }
        
        for i, method in enumerate(self.main_methods):
            ax = axes[i]
            data = metric_data[method].dropna()
            
            # Histogram
            ax.hist(data, bins=15, alpha=0.7, color=colors[method], edgecolor='black')
            
            # Add statistics
            mean_val = data.mean()
            std_val = data.std()
            median_val = data.median()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            # Formatting
            ax.set_title(f'{method.replace("_", " ")}\n(n={len(data)}, std={std_val:.3f})', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{metric.upper()} Score')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_xlim(0, 1)
        
        plt.suptitle(f'Distribution of Raw {metric.upper()} Scores by Method', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", f"{metric}_raw_distributions_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Raw {metric} distributions plot saved: {plot_path}")
    
    def _plot_difference_distributions(self, normality_results: Dict[Tuple[str, str], Dict[str, Any]], metric: str):
        """
        Plot histograms of paired differences for assumption checking.
        """
        n_pairs = len(normality_results)
        if n_pairs == 0:
            return
        
        # Calculate grid dimensions
        n_cols = 3
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()
        
        for i, ((method1, method2), results) in enumerate(normality_results.items()):
            ax = axes[i]
            differences = results['differences']
            
            # Histogram
            color = 'green' if results['is_normal'] else 'red'
            ax.hist(differences, bins=15, alpha=0.7, color=color, edgecolor='black')
            
            # Add statistics
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            
            ax.axvline(mean_diff, color='darkblue', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_diff:.3f}')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, 
                      label='No difference')
            
            # Add normal curve overlay for comparison
            x_range = np.linspace(differences.min(), differences.max(), 100)
            normal_curve = stats.norm.pdf(x_range, mean_diff, std_diff)
            # Scale to match histogram
            normal_curve = normal_curve * len(differences) * (differences.max() - differences.min()) / 15
            ax.plot(x_range, normal_curve, 'k--', alpha=0.8, linewidth=2, label='Normal fit')
            
            # Formatting
            normality_status = "[PASS] Normal" if results['is_normal'] else "[FAIL] Non-normal"
            test_used = "Paired t-test" if results['is_normal'] else "Wilcoxon signed-rank"
            
            ax.set_title(f'{method1.replace("_", " ")} - {method2.replace("_", " ")}\n'
                        f'{normality_status} (p={results["p_value"]:.4f})\n'
                        f'-> {test_used}', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel(f'{metric.upper()} Difference ({method1} - {method2})')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for i in range(n_pairs, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Distribution of Paired Differences for {metric.upper()} Scores\n'
                    f'(Used for Normality Assumption Testing)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", f"{metric}_difference_distributions_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Paired differences distributions plot saved: {plot_path}")
    
    def _create_distribution_plots_targeted(self, metric: str, target_pairs: List[Tuple[str, str]], 
                                           normality_results: Dict[Tuple[str, str], Dict[str, Any]]):
        """
        Create distribution plots for targeted pairs only.
        """
        print(f"\n{'='*60}")
        print("CREATING DISTRIBUTION PLOTS (TARGETED PAIRS)")
        print("="*60)
        
        # 1. Raw F1 distributions by method (only the methods we're testing)
        target_methods = set()
        for method1, method2 in target_pairs:
            target_methods.add(method1)
            target_methods.add(method2)
        
        metric_data = self.paired_data[metric]
        
        # Create subplots for target methods only
        n_methods = len(target_methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 6))
        if n_methods == 1:
            axes = [axes]
        
        colors = {
            'Human_Retrospective': '#ff4444',   # Red  
            'Human_Predicted': '#ff7f7f',      # Light red
            'LLM_Retrospective': '#4444ff',     # Blue
            'LLM_Predicted': '#7fbfff',        # Light blue
            'Random_Baseline': '#cccccc',       # Light gray
            'ZeroR_Baseline': '#999999'        # Dark gray
        }
        
        for i, method in enumerate(sorted(target_methods)):
            ax = axes[i]
            data = metric_data[method].dropna()
            
            # Histogram
            ax.hist(data, bins=15, alpha=0.7, color=colors[method], edgecolor='black')
            
            # Add statistics
            mean_val = data.mean()
            std_val = data.std()
            median_val = data.median()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            # Formatting
            ax.set_title(f'{method.replace("_", " ")}\n(n={len(data)}, std={std_val:.3f})', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{metric.upper()} Score')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_xlim(0, 1)
        
        plt.suptitle(f'Distribution of Raw {metric.upper()} Scores by Method (Targeted)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", f"{metric}_raw_distributions_targeted_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Raw {metric} distributions plot (targeted) saved: {plot_path}")
        
        # 2. Paired differences distributions
        self._plot_difference_distributions_targeted(normality_results, metric)
    
    def _plot_difference_distributions_targeted(self, normality_results: Dict[Tuple[str, str], Dict[str, Any]], metric: str):
        """
        Plot histograms of paired differences for targeted pairs only.
        """
        n_pairs = len(normality_results)
        if n_pairs == 0:
            return
        
        fig, axes = plt.subplots(1, n_pairs, figsize=(8*n_pairs, 6))
        if n_pairs == 1:
            axes = [axes]
        
        for i, ((method1, method2), results) in enumerate(normality_results.items()):
            ax = axes[i]
            differences = results['differences']
            
            # Histogram
            color = 'green' if results['is_normal'] else 'red'
            ax.hist(differences, bins=15, alpha=0.7, color=color, edgecolor='black')
            
            # Add statistics
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            
            ax.axvline(mean_diff, color='darkblue', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_diff:.3f}')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, 
                      label='No difference')
            
            # Add normal curve overlay for comparison
            x_range = np.linspace(differences.min(), differences.max(), 100)
            normal_curve = stats.norm.pdf(x_range, mean_diff, std_diff)
            # Scale to match histogram
            normal_curve = normal_curve * len(differences) * (differences.max() - differences.min()) / 15
            ax.plot(x_range, normal_curve, 'k--', alpha=0.8, linewidth=2, label='Normal fit')
            
            # Formatting
            normality_status = "[PASS] Normal" if results['is_normal'] else "[FAIL] Non-normal"
            test_used = "Paired t-test" if results['is_normal'] else "Wilcoxon signed-rank"
            
            ax.set_title(f'{method1.replace("_", " ")} - {method2.replace("_", " ")}\n'
                        f'{normality_status} (p={results["p_value"]:.4f})\n'
                        f'-> {test_used}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{metric.upper()} Difference ({method1} - {method2})')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        
        plt.suptitle(f'Distribution of Paired Differences for {metric.upper()} Scores (Targeted)\n'
                    f'(Used for Normality Assumption Testing)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", f"{metric}_difference_distributions_targeted_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Paired differences distributions plot (targeted) saved: {plot_path}")
    
    def perform_targeted_comparisons(self, metric: str = 'f1', force_test_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform targeted statistical comparisons for two-panel plot.
        Only compares: Human_Predicted vs LLM_Predicted and Human_Retrospective vs LLM_Retrospective
        
        Parameters:
        metric: str, metric to analyze (default: 'f1')
        force_test_type: Optional[str], force test type ('parametric', 'nonparametric', or None for automatic)
        
        Returns:
        List of dictionaries containing test results
        """
        print(f"\n{'='*60}")
        print(f"PERFORMING TARGETED COMPARISONS FOR {metric.upper()}")
        print("Comparing within task types (including baselines):")
        print("  - Prediction: Human_Predicted vs LLM_Predicted + baseline comparisons")
        print("  - Identification: Human_Retrospective vs LLM_Retrospective + baseline comparisons")
        if force_test_type:
            print(f"FORCED TEST TYPE: {force_test_type.upper()}")
        print("="*60)
        
        # Define the specific pairs we want to compare (including baselines)
        target_pairs = [
            # Prediction task comparisons
            ('Human_Predicted', 'LLM_Predicted'),
            ('Human_Predicted', 'Random_Baseline'),
            ('Human_Predicted', 'ZeroR_Baseline'),
            ('LLM_Predicted', 'Random_Baseline'),
            ('LLM_Predicted', 'ZeroR_Baseline'),
            
            # Identification task comparisons
            ('Human_Retrospective', 'LLM_Retrospective'),
            ('Human_Retrospective', 'Random_Baseline'),
            ('Human_Retrospective', 'ZeroR_Baseline'),
            ('LLM_Retrospective', 'Random_Baseline'),
            ('LLM_Retrospective', 'ZeroR_Baseline')
        ]
        
        # Test normality assumptions for only these pairs (unless forcing a specific test type)
        if force_test_type is None:
            normality_results = self.test_normality_assumptions_targeted(metric, target_pairs)
        else:
            print(f"\nSkipping normality testing - using forced test type: {force_test_type}")
            normality_results = {}
        
        comparison_results = []
        metric_data = self.paired_data[metric]
        
        for method1, method2 in target_pairs:
            # Skip if either method not in data
            if method1 not in metric_data.columns or method2 not in metric_data.columns:
                continue
            
            # Get paired data
            data1 = metric_data[method1]
            data2 = metric_data[method2]
            
            # Remove NaN pairs
            mask = ~(data1.isna() | data2.isna())
            data1_clean = data1[mask]
            data2_clean = data2[mask]
            
            if len(data1_clean) < 3:
                print(f"  {method1} vs {method2}: Insufficient data (n={len(data1_clean)})")
                continue
            
            # Select appropriate test based on force_test_type or normality results
            if force_test_type == 'parametric':
                use_parametric = True
            elif force_test_type == 'nonparametric':
                use_parametric = False
            else:
                # Use normality testing results
                pair_key = (method1, method2) if (method1, method2) in normality_results else (method2, method1)
                
                if pair_key in normality_results:
                    use_parametric = normality_results[pair_key]['is_normal']
                else:
                    # Default to non-parametric for baseline comparisons or untested pairs
                    use_parametric = False
            
            # Perform statistical test
            if use_parametric:
                # Paired t-test
                statistic, p_value = ttest_rel(data1_clean, data2_clean)
                test_name = "Paired t-test"
                
                # Calculate Cohen's d (effect size)
                diff = data1_clean - data2_clean
                effect_size = np.mean(diff) / np.std(diff, ddof=1)
                effect_size_name = "Cohen's d"
                
            else:
                # Wilcoxon signed-rank test
                if np.all(data1_clean == data2_clean):
                    # All differences are zero
                    statistic, p_value = 0, 1.0
                else:
                    statistic, p_value = wilcoxon(data1_clean, data2_clean)
                test_name = "Wilcoxon signed-rank"
                
                # Calculate effect size r = Z/sqrt(N)
                from scipy.stats import norm
                z_score = norm.ppf(1 - p_value/2) if p_value > 0 else 0  # Two-tailed
                effect_size = z_score / np.sqrt(len(data1_clean))
                effect_size_name = "r"
            
            # Store results
            result = {
                'method1': method1,
                'method2': method2,
                'metric': metric,
                'n_samples': len(data1_clean),
                'mean1': np.mean(data1_clean),
                'mean2': np.mean(data2_clean),
                'std1': np.std(data1_clean, ddof=1),
                'std2': np.std(data2_clean, ddof=1),
                'test_name': test_name,
                'test_statistic': statistic,
                'p_value': p_value,
                'effect_size': effect_size,
                'effect_size_name': effect_size_name,
                'is_parametric': use_parametric
            }
            
            comparison_results.append(result)
            
            print(f"  {method1} vs {method2}:")
            print(f"    n={len(data1_clean)}, {test_name}")
            print(f"    M1={np.mean(data1_clean):.3f}±{np.std(data1_clean, ddof=1):.3f}, "
                  f"M2={np.mean(data2_clean):.3f}±{np.std(data2_clean, ddof=1):.3f}")
            print(f"    Statistic={statistic:.4f}, p={p_value:.4f}, "
                  f"{effect_size_name}={effect_size:.3f}")
        
        return comparison_results
    
    def correct_multiple_comparisons_by_task(self, comparison_results: List[Dict[str, Any]], 
                                           method: str = 'holm') -> List[Dict[str, Any]]:
        """
        Apply multiple comparisons correction separately for prediction and identification tasks.
        
        Parameters:
        comparison_results: List of comparison result dictionaries
        method: str, correction method ('holm', 'bonferroni', 'fdr_bh')
        
        Returns:
        Updated results with corrected p-values
        """
        print(f"\n{'='*60}")
        print(f"APPLYING MULTIPLE COMPARISONS CORRECTION BY TASK ({method.upper()})")
        print("="*60)
        
        # Separate comparisons by task type
        prediction_comparisons = []
        identification_comparisons = []
        
        prediction_methods = {'Human_Predicted', 'LLM_Predicted', 'Random_Baseline', 'ZeroR_Baseline'}
        identification_methods = {'Human_Retrospective', 'LLM_Retrospective', 'Random_Baseline', 'ZeroR_Baseline'}
        
        for result in comparison_results:
            methods_in_comparison = {result['method1'], result['method2']}
            
            # Check if this is a prediction task comparison
            if methods_in_comparison.issubset(prediction_methods):
                prediction_comparisons.append(result)
            # Check if this is an identification task comparison  
            elif methods_in_comparison.issubset(identification_methods):
                identification_comparisons.append(result)
        
        print(f"Prediction task comparisons: {len(prediction_comparisons)}")
        print(f"Identification task comparisons: {len(identification_comparisons)}")
        
        # Apply correction to prediction comparisons
        if prediction_comparisons:
            print(f"\nApplying {method} correction to PREDICTION comparisons:")
            p_values_pred = [result['p_value'] for result in prediction_comparisons]
            rejected_pred, p_corrected_pred, _, _ = multipletests(
                p_values_pred, method=method, alpha=0.05
            )
            
            for i, result in enumerate(prediction_comparisons):
                result['p_corrected'] = p_corrected_pred[i]
                result['significant'] = rejected_pred[i]
                result['correction_method'] = f"{method}_prediction"
            
            n_sig_raw_pred = sum(1 for p in p_values_pred if p < 0.05)
            n_sig_corr_pred = sum(rejected_pred)
            print(f"  Before correction: {n_sig_raw_pred}/{len(p_values_pred)} significant")
            print(f"  After correction: {n_sig_corr_pred}/{len(p_values_pred)} significant")
        
        # Apply correction to identification comparisons
        if identification_comparisons:
            print(f"\nApplying {method} correction to IDENTIFICATION comparisons:")
            p_values_id = [result['p_value'] for result in identification_comparisons]
            rejected_id, p_corrected_id, _, _ = multipletests(
                p_values_id, method=method, alpha=0.05
            )
            
            for i, result in enumerate(identification_comparisons):
                result['p_corrected'] = p_corrected_id[i]
                result['significant'] = rejected_id[i]
                result['correction_method'] = f"{method}_identification"
            
            n_sig_raw_id = sum(1 for p in p_values_id if p < 0.05)
            n_sig_corr_id = sum(rejected_id)
            print(f"  Before correction: {n_sig_raw_id}/{len(p_values_id)} significant")
            print(f"  After correction: {n_sig_corr_id}/{len(p_values_id)} significant")
        
        # Show significant results by task
        pred_significant = [r for r in prediction_comparisons if r['significant']]
        id_significant = [r for r in identification_comparisons if r['significant']]
        
        if pred_significant:
            print(f"\nSignificant PREDICTION results after correction:")
            for result in pred_significant:
                print(f"  {result['method1']} vs {result['method2']}: "
                      f"p_corrected={result['p_corrected']:.4f}, "
                      f"{result['effect_size_name']}={result['effect_size']:.3f}")
        
        if id_significant:
            print(f"\nSignificant IDENTIFICATION results after correction:")
            for result in id_significant:
                print(f"  {result['method1']} vs {result['method2']}: "
                      f"p_corrected={result['p_corrected']:.4f}, "
                      f"{result['effect_size_name']}={result['effect_size']:.3f}")
        
        if not pred_significant and not id_significant:
            print("\nNo significant results after correction in either task.")
        
        return comparison_results
    
    def create_two_panel_plot(self, comparison_results: List[Dict[str, Any]], metric: str = 'f1'):
        """
        Create the exact two-panel plot from analysis_pipeline with significance markers.
        Left: Switch Prediction (Human_Predicted vs LLM_Predicted vs baselines)  
        Right: Switch Identification (Human_Retrospective vs LLM_Retrospective vs baselines)
        
        Args:
            comparison_results: List of comparison dictionaries with p-values and test info
            metric: str, metric to plot (f1, recall, etc.)
        """
        print(f"\n{'='*60}")
        print(f"CREATING TWO-PANEL {metric.upper()} SIGNIFICANCE PLOT")
        print("="*60)
        
        if not STATANNOTATIONS_AVAILABLE:
            print("Warning: statannotations not available. Creating plot without significance markers.")
            self._create_basic_two_panel_plot()
            return
        
        # Define method groupings (matching analysis_pipeline exactly)
        prediction_methods = ['Human_Predicted', 'LLM_Predicted', 'Random_Baseline', 'ZeroR_Baseline']
        identification_methods = ['Human_Retrospective', 'LLM_Retrospective', 'Random_Baseline', 'ZeroR_Baseline']
        
        # Calculate summary statistics for each method
        metric_data = self.paired_data[metric].copy()
        summary_stats = []
        for method in self.all_methods:
            if method in metric_data.columns:
                values = metric_data[method].dropna()
                summary_stats.append({
                    'method': method,
                    'mean': values.mean(),
                    'sem': values.std() / np.sqrt(len(values)),
                    'std': values.std(),
                    'n': len(values)
                })
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Define colors (matching analysis_pipeline exactly)
        colors = {
            'Human_Predicted': '#ff7f7f',      # Light red
            'Human_Retrospective': '#ff4444',   # Red  
            'LLM_Predicted': '#7fbfff',        # Light blue
            'LLM_Retrospective': '#4444ff',     # Blue
            'Random_Baseline': '#cccccc',       # Light gray
            'ZeroR_Baseline': '#999999'        # Dark gray
        }
        
        # LEFT PANEL: Switch Prediction
        prediction_data = summary_df[summary_df['method'].isin(prediction_methods)]
        self._create_panel_with_significance(ax1, prediction_data, prediction_methods, 
                                           colors, comparison_results, 
                                           f'Switch Prediction {metric.upper()} (All Categories)',
                                           panel_type='prediction', metric=metric)
        
        # RIGHT PANEL: Switch Identification  
        identification_data = summary_df[summary_df['method'].isin(identification_methods)]
        self._create_panel_with_significance(ax2, identification_data, identification_methods,
                                            colors, comparison_results,
                                            f'Switch Identification {metric.upper()} (All Categories)', 
                                            panel_type='identification', metric=metric)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", 
                                f"{metric}_two_panel_significance_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Two-panel {metric.upper()} significance plot saved: {plot_path}")
    
    def _create_panel_with_significance(self, ax, panel_data, methods, colors, 
                                      comparison_results, title, panel_type, metric='f1'):
        """
        Create one panel of the two-panel plot with significance annotations.
        
        Args:
            metric: str, metric to plot (f1, recall, etc.)
        """
        # Prepare data for seaborn/statannotations (following tutorial pattern)
        plot_data = []
        metric_data = self.paired_data[metric]
        
        for method in methods:
            if method in metric_data.columns:
                values = metric_data[method].dropna()
                for value in values:
                    plot_data.append({'method': method, metric: value})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create seaborn bar plot
        method_palette = [colors.get(m, '#cccccc') for m in methods]
        
        # Plot parameters for statannotations (following tutorial pattern)
        plotting_parameters = {
            'data': plot_df,
            'x': 'method',
            'y': metric,
            'palette': method_palette
        }
        
        sns.barplot(ax=ax, **plotting_parameters, capsize=0.1, errwidth=2, alpha=0.8)
        
        # Find significant pairs for this panel
        significant_pairs = []
        panel_methods_set = set(methods)
        
        # Define which comparisons belong to each panel (only those involving methods shown in the panel)
        prediction_panel_methods = {'Human_Predicted', 'LLM_Predicted', 'Random_Baseline', 'ZeroR_Baseline'}
        identification_panel_methods = {'Human_Retrospective', 'LLM_Retrospective', 'Random_Baseline', 'ZeroR_Baseline'}
        
        # Get the actual methods shown in this panel
        panel_methods_shown = set(methods)
        
        for result in comparison_results:
            # Check if this comparison belongs to the current panel
            comparison_methods = {result['method1'], result['method2']}
            
            # Check if both methods in the comparison are actually shown in this panel
            both_methods_in_panel = comparison_methods.issubset(panel_methods_shown)
            
            # Check if this is the right task type
            is_right_task = ((panel_type == 'prediction' and 
                             comparison_methods.issubset(prediction_panel_methods)) or
                            (panel_type == 'identification' and 
                             comparison_methods.issubset(identification_panel_methods)))
            
            if both_methods_in_panel and is_right_task:
                # Add ALL pairs for annotations (significant or not)
                significant_pairs.append((result['method1'], result['method2']))
                status = "SIGNIFICANT" if result['significant'] else "NON-SIGNIFICANT"
                print(f"    {status} pair for {panel_type}: {result['method1']} vs {result['method2']}")
                print(f"    p_corrected={result['p_corrected']:.6f}")
            else:
                print(f"    Skipping pair (not both methods in panel): {result['method1']} vs {result['method2']}")
                print(f"      Panel methods: {panel_methods_shown}")
                print(f"      Comparison methods: {comparison_methods}")
                print(f"      Both in panel: {both_methods_in_panel}, Right task: {is_right_task}")
        
        print(f"  {title}: Adding {len(significant_pairs)} annotations (all comparisons)")
        
        # Add significance annotations if any (following tutorial pattern exactly)
        if significant_pairs:
            try:
                # Initialize annotator with same parameters as barplot (KEY!)
                annotator = Annotator(ax, significant_pairs, **plotting_parameters)
                annotator.configure(text_format='star', loc='outside', verbose=False)
                
                # Get p-values for significant pairs
                pvalues = []
                for pair in significant_pairs:
                    for result in comparison_results:
                        if ((result['method1'] == pair[0] and result['method2'] == pair[1]) or
                            (result['method1'] == pair[1] and result['method2'] == pair[0])):
                            pvalues.append(result['p_corrected'])
                            print(f"      Setting p-value for {pair}: {result['p_corrected']:.6f}")
                            break
                
                if pvalues:
                    annotator.set_pvalues(pvalues)
                    annotator.annotate()
                    print(f"      Successfully added {len(pvalues)} annotations")
                else:
                    print(f"      No p-values found for pairs: {significant_pairs}")
                    
            except Exception as e:
                print(f"    Warning: Could not add annotations to {title}: {e}")
                import traceback
                traceback.print_exc()
        
        # Customize plot (matching analysis_pipeline format exactly)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
        ax.set_xlabel('')
        ax.set_xticklabels([m.replace('_', '\n') for m in methods])
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    def _create_basic_two_panel_plot(self):
        """
        Create basic two-panel plot without statannotations (fallback).
        """
        # Define method groupings
        prediction_methods = ['Human_Predicted', 'LLM_Predicted', 'Random_Baseline', 'ZeroR_Baseline']
        identification_methods = ['Human_Retrospective', 'LLM_Retrospective', 'Random_Baseline', 'ZeroR_Baseline']
        
        f1_data = self.paired_data['f1'].copy()
        
        # Calculate summary statistics
        summary_stats = []
        for method in self.all_methods:
            if method in f1_data.columns:
                values = f1_data[method].dropna()
                summary_stats.append({
                    'method': method,
                    'mean': values.mean(),
                    'sem': values.std() / np.sqrt(len(values))
                })
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        colors = {
            'Human_Predicted': '#ff7f7f',
            'Human_Retrospective': '#ff4444',
            'LLM_Predicted': '#7fbfff',
            'LLM_Retrospective': '#4444ff',
            'Random_Baseline': '#cccccc',
            'ZeroR_Baseline': '#999999'
        }
        
        # Left panel: Prediction
        prediction_data = summary_df[summary_df['method'].isin(prediction_methods)]
        x_pos = np.arange(len(prediction_data))
        ax1.bar(x_pos, prediction_data['mean'], 
               yerr=prediction_data['sem'],
               color=[colors.get(m, '#cccccc') for m in prediction_data['method']],
               capsize=5, alpha=0.8)
        
        ax1.set_title('Switch Prediction F1 (All Categories)', fontweight='bold', fontsize=14)
        ax1.set_ylabel('F1 Score')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.replace('_', '\n') for m in prediction_data['method']])
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Right panel: Identification
        identification_data = summary_df[summary_df['method'].isin(identification_methods)]
        x_pos = np.arange(len(identification_data))
        ax2.bar(x_pos, identification_data['mean'], 
               yerr=identification_data['sem'],
               color=[colors.get(m, '#cccccc') for m in identification_data['method']],
               capsize=5, alpha=0.8)
        
        ax2.set_title('Switch Identification F1 (All Categories)', fontweight='bold', fontsize=14)
        ax2.set_ylabel('F1 Score')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.replace('_', '\n') for m in identification_data['method']])
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "plots", 
                                f"f1_two_panel_basic_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Basic two-panel F1 plot saved: {plot_path}")

    def save_results(self, comparison_results: List[Dict[str, Any]]):
        """
        Save statistical analysis results to CSV files.
        """
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print("="*60)
        
        # Save detailed results
        results_df = pd.DataFrame(comparison_results)
        results_path = os.path.join(self.output_dir, "statistics", 
                                   f"statistical_comparisons_{self.timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Detailed results saved: {results_path}")
        
        # Save summary of significant results
        significant_results = [r for r in comparison_results if r['significant']]
        if significant_results:
            sig_df = pd.DataFrame(significant_results)
            sig_path = os.path.join(self.output_dir, "statistics", 
                                   f"significant_results_{self.timestamp}.csv")
            sig_df.to_csv(sig_path, index=False)
            print(f"Significant results saved: {sig_path}")
        
        # Save summary statistics
        summary_stats = []
        for metric in ['f1']:  # Can extend to other metrics
            metric_data = self.paired_data[metric]
            for method in self.main_methods:
                if method in metric_data.columns:
                    values = metric_data[method].dropna()
                    summary_stats.append({
                        'metric': metric,
                        'method': method,
                        'n': len(values),
                        'mean': values.mean(),
                        'std': values.std(),
                        'sem': values.std() / np.sqrt(len(values)),
                        'min': values.min(),
                        'max': values.max(),
                        'median': values.median()
                    })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_path = os.path.join(self.output_dir, "statistics", 
                                   f"method_summary_{self.timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Method summary saved: {summary_path}")
    
    def run_complete_analysis(self, metric: str = 'f1', force_test_type: Optional[str] = None):
        """
        Run the complete statistical significance analysis pipeline.
        
        Parameters:
        metric: str, metric to analyze (default: 'f1')
        force_test_type: Optional[str], force test type ('parametric', 'nonparametric', or None for automatic)
        """
        print("Statistical Significance Analysis Pipeline")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Perform targeted comparisons
        comparison_results = self.perform_targeted_comparisons(metric, force_test_type)
        
        # Step 3: Correct for multiple comparisons by task type
        comparison_results = self.correct_multiple_comparisons_by_task(comparison_results, method='holm')
        
        # Step 4: Create visualizations
        self.create_two_panel_plot(comparison_results, metric)
        
        # Step 5: Save results
        self.save_results(comparison_results)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"All outputs saved to: {self.output_dir}")
        
        return comparison_results


def find_participant_metrics_files():
    """
    Find all participant_metrics.csv files in the analysis output directory.
    """
    csv_files = []
    
    # Search in analysis_output directory
    search_dirs = ['analysis_output', '.']
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.startswith('participant_metrics') and file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
    
    return sorted(csv_files, key=os.path.getmtime, reverse=True)  # Most recent first


def select_metrics_file(csv_files):
    """
    Allow user to select a participant metrics file.
    """
    if not csv_files:
        print("No participant_metrics.csv files found.")
        return None
    
    print("Available participant metrics files:")
    print("="*60)
    
    for i, file_path in enumerate(csv_files, 1):
        try:
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"{i:2d}. {file_path}")
            print(f"    Size: {size_mb:.1f} MB, Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(f"{i:2d}. {file_path}")
    
    print("="*60)
    
    while True:
        try:
            choice = input(f"Select a file (1-{len(csv_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(csv_files):
                selected_file = csv_files[choice_num - 1]
                print(f"Selected: {selected_file}")
                return selected_file
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def main():
    """
    Main function to run the statistical significance analysis.
    """
    parser = argparse.ArgumentParser(description='Statistical Significance Analysis for LLM Switch Detection')
    parser.add_argument('--input', '-i', type=str, help='Path to participant_metrics.csv file')
    parser.add_argument('--output', '-o', type=str, default='significance_analysis_output', 
                       help='Output directory (default: significance_analysis_output)')
    parser.add_argument('--metric', '-m', type=str, default='f1', 
                       choices=['accuracy', 'precision', 'recall', 'f1'],
                       help='Metric to analyze (default: f1)')
    parser.add_argument('--force-test', '-t', type=str, default=None,
                       choices=['parametric', 'nonparametric'],
                       help='Force test type: parametric (paired t-test) or nonparametric (Wilcoxon signed-rank). If not specified, automatic selection based on normality testing.')
    
    args = parser.parse_args()
    
    # Get input file
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}")
            return
        metrics_file = args.input
    else:
        # Find and select metrics file
        csv_files = find_participant_metrics_files()
        metrics_file = select_metrics_file(csv_files)
        if metrics_file is None:
            return
    
    # Run analysis
    analyzer = StatisticalSignificanceAnalyzer(metrics_file, args.output)
    results = analyzer.run_complete_analysis(args.metric, args.force_test)
    
    return results


if __name__ == "__main__":
    main()