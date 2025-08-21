"""
Semantic Distance Analysis Pipeline
==================================

This script analyzes semantic distance between text and different types of labels:
- SNAFU distance: between "text" and "snafu_gt_label"
- Human distance: between "text" and "label"  
- LLM distance: between "text" and "labelLLM"

Creates a bar plot comparing these distances with the title:
"Semantic distance between words and cluster-based labels"

Usage:
    uv run python semantic_distance_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json
from scipy.spatial.distance import cosine
from typing import Dict, List
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class SemanticDistanceAnalyzer:
    def __init__(self, human_embeddings_path: str, llm_embeddings_path: str, output_dir: str = "analysis_output"):
        """Initialize the semantic distance analyzer."""
        self.human_embeddings_path = human_embeddings_path
        self.llm_embeddings_path = llm_embeddings_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots", "semantic_distance"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "statistics", "semantic_distance"), exist_ok=True)
        
        # Data containers
        self.text_embeddings = {}
        self.label_embeddings = {}
        self.snafu_embeddings = {}
        self.llm_embeddings = {}
        self.human_data = None
        self.llm_data = None
        self.results_df = None
        
    def load_embeddings(self):
        """Load embeddings from pickle files."""
        print("=" * 60)
        print("LOADING EMBEDDINGS")
        print("=" * 60)
        
        # Load human embeddings
        print("Loading human embeddings...")
        with open(self.human_embeddings_path, 'rb') as f:
            human_data = pickle.load(f)
        
        self.text_embeddings = human_data['text_embeddings']
        self.label_embeddings = human_data['label_embeddings']
        self.snafu_embeddings = human_data['snafu_embeddings']
        
        print(f"  Text embeddings: {len(self.text_embeddings)}")
        print(f"  Human label embeddings: {len(self.label_embeddings)}")
        print(f"  SNAFU embeddings: {len(self.snafu_embeddings)}")
        
        # Load LLM embeddings
        print("\\nLoading LLM embeddings...")
        with open(self.llm_embeddings_path, 'rb') as f:
            llm_data = pickle.load(f)
        
        self.llm_embeddings = llm_data['llm_embeddings']
        print(f"  LLM label embeddings: {len(self.llm_embeddings)}")
        
    def load_data_mappings(self):
        """Load data mappings to connect texts with their labels."""
        print("\\nLoading data mappings...")
        
        # Load human data mapping
        human_mapping_path = os.path.join(os.path.dirname(self.human_embeddings_path), "human_data_mapping.csv")
        self.human_data = pd.read_csv(human_mapping_path)
        print(f"Human data mapping loaded: {self.human_data.shape}")
        
        # Load LLM data mapping
        llm_mapping_path = os.path.join(os.path.dirname(self.llm_embeddings_path), "llm_data_mapping.csv")
        self.llm_data = pd.read_csv(llm_mapping_path)
        print(f"LLM data mapping loaded: {self.llm_data.shape}")
        
        # Align datasets
        self._align_datasets()
        
    def _align_datasets(self):
        """Align human and LLM datasets to ensure consistent comparisons."""
        print("\\nAligning datasets...")
        
        # Find common data points
        human_keys = set(zip(self.human_data['playerID'], self.human_data['word_index']))
        llm_keys = set(zip(self.llm_data['playerID'], self.llm_data['word_index']))
        common_keys = human_keys.intersection(llm_keys)
        
        print(f"Human data points: {len(human_keys)}")
        print(f"LLM data points: {len(llm_keys)}")
        print(f"Common data points: {len(common_keys)}")
        
        # Filter to common keys
        def filter_by_keys(df, keys):
            return df[df.apply(lambda row: (row['playerID'], row['word_index']) in keys, axis=1)]
        
        self.human_data = filter_by_keys(self.human_data, common_keys)
        self.llm_data = filter_by_keys(self.llm_data, common_keys)
        
        # Sort for consistent ordering
        self.human_data = self.human_data.sort_values(['playerID', 'word_index']).reset_index(drop=True)
        self.llm_data = self.llm_data.sort_values(['playerID', 'word_index']).reset_index(drop=True)
        
        print(f"Aligned human data: {self.human_data.shape}")
        print(f"Aligned LLM data: {self.llm_data.shape}")
        
    def calculate_distances(self):
        """Calculate semantic distances for all text-label pairs."""
        print("=" * 60)
        print("CALCULATING SEMANTIC DISTANCES")
        print("=" * 60)
        
        results = []
        
        for idx in range(len(self.human_data)):
            human_row = self.human_data.iloc[idx]
            llm_row = self.llm_data.iloc[idx]
            
            # Verify alignment
            assert human_row['playerID'] == llm_row['playerID'], "Data alignment error"
            assert human_row['word_index'] == llm_row['word_index'], "Data alignment error"
            
            # Use original text for display, normalized for embedding lookup
            text = human_row['text']
            human_label = human_row['label']
            snafu_gt_label = human_row['snafu_gt_label']
            llm_label = llm_row['labelLLM']
            
            # Get normalized versions for embedding lookup
            text_normalized = human_row['text_normalized']
            human_label_normalized = human_row['label_normalized']
            snafu_gt_label_normalized = human_row['snafu_gt_label_normalized']
            llm_label_normalized = llm_row['labelLLM_normalized']
            
            # Check if we have all required embeddings (using normalized versions)
            if (text_normalized in self.text_embeddings and 
                human_label_normalized in self.label_embeddings and 
                snafu_gt_label_normalized in self.snafu_embeddings and 
                llm_label_normalized in self.llm_embeddings):
                
                # Get embeddings using normalized keys
                text_emb = np.array(self.text_embeddings[text_normalized])
                human_label_emb = np.array(self.label_embeddings[human_label_normalized])
                snafu_emb = np.array(self.snafu_embeddings[snafu_gt_label_normalized])
                llm_label_emb = np.array(self.llm_embeddings[llm_label_normalized])
                
                # Calculate cosine distances
                snafu_distance = cosine(text_emb, snafu_emb)
                human_distance = cosine(text_emb, human_label_emb)
                llm_distance = cosine(text_emb, llm_label_emb)
                
                # Store result
                results.append({
                    'playerID': human_row['playerID'],
                    'category': human_row['category'],
                    'word_index': human_row['word_index'],
                    'text': text,
                    'human_label': human_label,
                    'snafu_gt_label': snafu_gt_label,
                    'llm_label': llm_label,
                    'snafu_distance': snafu_distance,
                    'human_distance': human_distance,
                    'llm_distance': llm_distance
                })
            
            # Progress reporting
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(self.human_data)} rows")
        
        self.results_df = pd.DataFrame(results)
        print(f"\\nCalculated distances for {len(self.results_df)} data points")
        
    def create_summary_statistics(self):
        """Create summary statistics for distances."""
        print("=" * 60)
        print("CREATING SUMMARY STATISTICS")
        print("=" * 60)
        
        # Overall statistics
        overall_stats = {
            'SNAFU_distance': {
                'mean': float(self.results_df['snafu_distance'].mean()),
                'std': float(self.results_df['snafu_distance'].std()),
                'sem': float(self.results_df['snafu_distance'].sem()),
                'median': float(self.results_df['snafu_distance'].median()),
                'count': len(self.results_df)
            },
            'Human_distance': {
                'mean': float(self.results_df['human_distance'].mean()),
                'std': float(self.results_df['human_distance'].std()),
                'sem': float(self.results_df['human_distance'].sem()),
                'median': float(self.results_df['human_distance'].median()),
                'count': len(self.results_df)
            },
            'LLM_distance': {
                'mean': float(self.results_df['llm_distance'].mean()),
                'std': float(self.results_df['llm_distance'].std()),
                'sem': float(self.results_df['llm_distance'].sem()),
                'median': float(self.results_df['llm_distance'].median()),
                'count': len(self.results_df)
            }
        }
        
        print("Overall Distance Statistics:")
        print("=" * 40)
        for dist_type, stats in overall_stats.items():
            print(f"{dist_type}: {stats['mean']:.4f} ± {stats['sem']:.4f} (n={stats['count']})")
        
        # Statistics by category
        category_stats = []
        for category in self.results_df['category'].unique():
            cat_data = self.results_df[self.results_df['category'] == category]
            
            for dist_col in ['snafu_distance', 'human_distance', 'llm_distance']:
                category_stats.append({
                    'category': category,
                    'distance_type': dist_col,
                    'mean': cat_data[dist_col].mean(),
                    'std': cat_data[dist_col].std(),
                    'sem': cat_data[dist_col].sem(),
                    'median': cat_data[dist_col].median(),
                    'count': len(cat_data)
                })
        
        category_stats_df = pd.DataFrame(category_stats)
        
        print("\\nDistance Statistics by Category:")
        print("=" * 40)
        print(category_stats_df.round(4))
        
        # Save statistics
        overall_stats_path = os.path.join(self.output_dir, "statistics", "semantic_distance", f"overall_statistics_{self.timestamp}.json")
        with open(overall_stats_path, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        print(f"\\nOverall statistics saved to: {overall_stats_path}")
        
        category_stats_path = os.path.join(self.output_dir, "statistics", "semantic_distance", f"category_statistics_{self.timestamp}.csv")
        category_stats_df.to_csv(category_stats_path, index=False)
        print(f"Category statistics saved to: {category_stats_path}")
        
    def create_distance_barplot(self):
        """Create a bar plot comparing semantic distances with player-level variability."""
        print("=" * 60)
        print("CREATING DISTANCE BAR PLOT WITH PLAYER VARIABILITY")
        print("=" * 60)
        
        # Calculate per-player means first
        player_means = []
        for player_id in self.results_df['playerID'].unique():
            player_data = self.results_df[self.results_df['playerID'] == player_id]
            player_means.append({
                'playerID': player_id,
                'snafu_mean': player_data['snafu_distance'].mean(),
                'human_mean': player_data['human_distance'].mean(),
                'llm_mean': player_data['llm_distance'].mean(),
                'n_words': len(player_data)
            })
        
        player_means_df = pd.DataFrame(player_means)
        print(f"Calculated means for {len(player_means_df)} players")
        print(f"Words per player: {player_means_df['n_words'].describe()}")
        
        # Calculate means and SEMs across players (not individual words)
        snafu_mean = player_means_df['snafu_mean'].mean()
        snafu_sem = player_means_df['snafu_mean'].sem()
        
        human_mean = player_means_df['human_mean'].mean()
        human_sem = player_means_df['human_mean'].sem()
        
        llm_mean = player_means_df['llm_mean'].mean()
        llm_sem = player_means_df['llm_mean'].sem()
        
        print(f"Player-level variability:")
        print(f"  SNAFU: {snafu_mean:.4f} ± {snafu_sem:.4f} (SEM across {len(player_means_df)} players)")
        print(f"  Human: {human_mean:.4f} ± {human_sem:.4f} (SEM across {len(player_means_df)} players)")
        print(f"  LLM: {llm_mean:.4f} ± {llm_sem:.4f} (SEM across {len(player_means_df)} players)")
        
        # Prepare plot data
        plot_data = pd.DataFrame({
            'Distance_Type': ['SNAFU', 'Human', 'LLM'],
            'Mean_Distance': [snafu_mean, human_mean, llm_mean],
            'SEM': [snafu_sem, human_sem, llm_sem]
        })
        
        # Store player means for further analysis
        self.player_means_df = player_means_df
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        # Create bar plot
        bars = ax.bar(plot_data['Distance_Type'], plot_data['Mean_Distance'], 
                     yerr=plot_data['SEM'], capsize=5, color=colors, alpha=0.8)
        
        # Customize plot
        ax.set_title('Semantic distance between words and cluster-based labels\n(Player-level means ± SEM)', 
                    fontweight='bold', fontsize=14)
        ax.set_ylabel('Cosine Distance (Mean per Player)', fontsize=12)
        ax.set_xlabel('Label Type', fontsize=12)
        ax.set_ylim(0, max(plot_data['Mean_Distance']) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels with SEM
        for bar, val, sem in zip(bars, plot_data['Mean_Distance'], plot_data['SEM']):
            ax.text(bar.get_x() + bar.get_width()/2, val + sem + 0.01,
                   f'{val:.3f}±{sem:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", "semantic_distance", f"semantic_distance_barplot_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distance bar plot saved: {plot_path}")
        
    def create_category_comparison_plot(self):
        """Create a grouped bar plot comparing distances by category."""
        print("Creating category comparison plot...")
        
        categories = self.results_df['category'].unique()
        
        # Calculate means and SEMs by category
        category_means = {}
        category_sems = {}
        
        for category in categories:
            cat_data = self.results_df[self.results_df['category'] == category]
            
            category_means[category] = {
                'SNAFU': cat_data['snafu_distance'].mean(),
                'Human': cat_data['human_distance'].mean(),
                'LLM': cat_data['llm_distance'].mean()
            }
            
            category_sems[category] = {
                'SNAFU': cat_data['snafu_distance'].sem(),
                'Human': cat_data['human_distance'].sem(),
                'LLM': cat_data['llm_distance'].sem()
            }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Settings
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        distance_types = ['SNAFU', 'Human', 'LLM']
        bar_width = 0.25
        x = np.arange(len(categories))
        
        # Create grouped bars
        for i, dist_type in enumerate(distance_types):
            means = [category_means[cat][dist_type] for cat in categories]
            sems = [category_sems[cat][dist_type] for cat in categories]
            
            bars = ax.bar(x + i * bar_width, means, bar_width, 
                         yerr=sems, capsize=3, label=dist_type, 
                         color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar, val, sem in zip(bars, means, sems):
                ax.text(bar.get_x() + bar.get_width()/2, val + sem + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Customize plot
        ax.set_title('Semantic Distance by Category', fontweight='bold', fontsize=14)
        ax.set_ylabel('Cosine Distance', fontsize=12)
        ax.set_xlabel('Category', fontsize=12)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", "semantic_distance", f"semantic_distance_by_category_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Category comparison plot saved: {plot_path}")
        
    def save_detailed_results(self):
        """Save detailed distance results."""
        print("Saving detailed results...")
        
        # Save individual word-level results
        detailed_path = os.path.join(self.output_dir, "statistics", "semantic_distance", f"detailed_distances_{self.timestamp}.csv")
        self.results_df.to_csv(detailed_path, index=False)
        print(f"Detailed word-level distances saved to: {detailed_path}")
        
        # Save player-level means
        if hasattr(self, 'player_means_df'):
            player_means_path = os.path.join(self.output_dir, "statistics", "semantic_distance", f"player_means_{self.timestamp}.csv")
            self.player_means_df.to_csv(player_means_path, index=False)
            print(f"Player-level means saved to: {player_means_path}")
        
    def run_analysis(self):
        """Run the complete semantic distance analysis pipeline."""
        print("Starting Semantic Distance Analysis Pipeline")
        print("=" * 60)
        
        # Step 1: Load embeddings
        self.load_embeddings()
        
        # Step 2: Load data mappings
        self.load_data_mappings()
        
        # Step 3: Calculate distances
        self.calculate_distances()
        
        # Step 4: Create summary statistics
        self.create_summary_statistics()
        
        # Step 5: Create visualizations
        self.create_distance_barplot()
        self.create_category_comparison_plot()
        
        # Step 6: Save detailed results
        self.save_detailed_results()
        
        print("\\nSemantic distance analysis completed successfully!")
        print(f"All outputs saved to: {self.output_dir}")

def find_embedding_files():
    """Find embedding files in the embeddings directory."""
    embeddings_dir = "embeddings"
    if not os.path.exists(embeddings_dir):
        return [], []
    
    human_files = []
    llm_files = []
    
    for file in os.listdir(embeddings_dir):
        if file.endswith('.pkl'):
            file_path = os.path.join(embeddings_dir, file)
            if 'human' in file.lower():
                human_files.append(file_path)
            elif 'llm' in file.lower():
                llm_files.append(file_path)
    
    return sorted(human_files), sorted(llm_files)

def select_embedding_files():
    """Allow user to select embedding files."""
    human_files, llm_files = find_embedding_files()
    
    if not human_files:
        print("No human embedding files found in embeddings/ directory.")
        print("Please run generate_embeddings_human.py first.")
        return None, None
    
    if not llm_files:
        print("No LLM embedding files found in embeddings/ directory.")
        print("Please run generate_embeddings_llm.py first.")
        return None, None
    
    print("Available human embedding files:")
    for i, file_path in enumerate(human_files, 1):
        print(f"{i}. {file_path}")
    
    while True:
        try:
            choice = input(f"Select human embeddings file (1-{len(human_files)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(human_files):
                human_file = human_files[choice_num - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(human_files)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\\nAvailable LLM embedding files:")
    for i, file_path in enumerate(llm_files, 1):
        print(f"{i}. {file_path}")
    
    while True:
        try:
            choice = input(f"Select LLM embeddings file (1-{len(llm_files)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(llm_files):
                llm_file = llm_files[choice_num - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(llm_files)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\\nSelected files:")
    print(f"  Human: {human_file}")
    print(f"  LLM: {llm_file}")
    
    return human_file, llm_file

def main():
    """Main function to run the semantic distance analysis."""
    print("Semantic Distance Analysis Pipeline")
    print("=" * 60)
    
    # Find and select embedding files
    human_file, llm_file = select_embedding_files()
    if human_file is None or llm_file is None:
        return
    
    # Check if files exist
    if not os.path.exists(human_file):
        print(f"Error: Human embeddings file not found at {human_file}")
        return
    
    if not os.path.exists(llm_file):
        print(f"Error: LLM embeddings file not found at {llm_file}")
        return
    
    # Run analysis
    analyzer = SemanticDistanceAnalyzer(human_file, llm_file)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()