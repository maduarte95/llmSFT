"""
Complete Semantic Analysis Pipeline
==================================

Single script that:
1. Generates embeddings using sentence-transformers with task-specific prompts
2. Uses built-in similarity functions (model.similarity_pairwise) for efficiency
3. Calculates semantic distances (1 - similarity) between texts and labels
4. Creates comprehensive visualizations and statistics

Analyzes semantic distance between text and different types of labels:
- SNAFU distance: between "text" and "snafu_gt_label"
- Human distance: between "text" and "label"  
- LLM distance: between "text" and "labelLLM"

Usage:
    uv run python semantic_analysis_complete.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
import re
from sentence_transformers import SentenceTransformer, SimilarityFunction
from typing import List, Dict, Any
import warnings
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def normalize_text(text):
    """Apply comprehensive text normalization for consistent embeddings."""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Remove punctuation (hyphens, apostrophes, commas, periods)
    text = re.sub(r'[-\',.]', '', text)
    # Remove whitespace
    text = re.sub(r'\s+', '', text)
    
    return text

def configure_similarity_function(model_name):
    """Configure the appropriate similarity function based on the model."""
    model_name_lower = model_name.lower()
    
    if "minilm" in model_name_lower or "mpnet" in model_name_lower:
        # Both MiniLM and mpnet models produce normalized embeddings
        # Use DOT_PRODUCT for better performance (equivalent to cosine on normalized embeddings)
        return SimilarityFunction.DOT_PRODUCT
    else:
        # For other models, use cosine as fallback
        return SimilarityFunction.COSINE

class CompleteSemanticAnalyzer:
    def __init__(self, data_path: str, model_name: str = "all-mpnet-base-v2", output_dir: str = "analysis_output", 
                 use_prompts: bool = True, normalize_embeddings: bool = False):
        """Initialize the complete semantic analyzer."""
        self.data_path = data_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_prompts = use_prompts
        self.normalize_embeddings = normalize_embeddings
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots", "semantic_analysis"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "statistics", "semantic_analysis"), exist_ok=True)
        
        # Initialize model
        print(f"Loading sentence-transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.similarity_fn_name = configure_similarity_function(model_name)
        print(f"Using similarity function: {self.model.similarity_fn_name}")
        print(f"Using prompts: {self.use_prompts}")
        print(f"Normalize embeddings: {self.normalize_embeddings}")
        
        # Define task-specific prompts
        if self.use_prompts:
            # self.prompts = {
            #     'word': 'Represent this word for semantic matching with a subcategory label in the category "animals": ',
            #     'label': 'Represent this subcategory label for semantic matching with words from the "animals" category: ',
            #     'snafu': 'Represent this subcategory label for semantic matching with words from the "animals" category: '
            # }
            self.prompts = {
                'word': 'To which subcategory of "animals" does this word belong? Word: ',
                'label': 'Subcategory:  ',
                'snafu': 'Subcategory: '
            }
        else:
            # No prompts - use raw text
            self.prompts = {
                'word': None,
                'label': None,
                'snafu': None
            }
        
        # Data containers
        self.data = None
        self.text_embeddings = {}
        self.label_embeddings = {}
        self.snafu_embeddings = {}
        self.llm_embeddings = {}
        self.results_df = None
        
    def load_and_clean_data(self):
        """Load data and remove rows with missing labels."""
        print("=" * 60)
        print("LOADING AND CLEANING DATA")
        print("=" * 60)
        
        self.data = pd.read_csv(self.data_path)
        print(f"Original data shape: {self.data.shape}")
        
        # Check required columns
        required_cols = ['text', 'playerID', 'category', 'word_index']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check which label columns are available
        available_label_cols = []
        for col in ['label', 'snafu_gt_label', 'labelLLM']:
            if col in self.data.columns:
                available_label_cols.append(col)
        
        if not available_label_cols:
            raise ValueError("No label columns found. Need at least one of: label, snafu_gt_label, labelLLM")
        
        print(f"Available label columns: {available_label_cols}")
        
        # Remove rows with missing text
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=['text'])
        print(f"After removing missing text: {len(self.data)} (removed {initial_count - len(self.data)})")
        
        # Remove word_index 0 (first word doesn't have a switch)
        self.data = self.data[self.data['word_index'] > 0].copy()
        print(f"After removing word_index 0: {len(self.data)}")
        
        # Apply participant filtering
        self._filter_extreme_participants()
        self._remove_duplicates()
        
        print(f"Final cleaned data shape: {self.data.shape}")
        print(f"Unique participants: {self.data['playerID'].nunique()}")
        
    def _filter_extreme_participants(self):
        """Filter out participants with extreme switch rates or zero switches."""
        print("\\nFiltering extreme participants...")
        
        participant_stats = []
        for player_id in self.data['playerID'].unique():
            player_data = self.data[self.data['playerID'] == player_id]
            
            if len(player_data) > 0:
                category = player_data['category'].iloc[0]
                total_words = len(player_data)
                
                if 'switch' in player_data.columns:
                    switches = player_data['switch'].sum()
                    rate = switches / total_words
                    
                    participant_stats.append({
                        'playerID': player_id,
                        'category': category,
                        'total_words': total_words,
                        'switches': switches,
                        'rate': rate
                    })
        
        if participant_stats:
            switch_rates_df = pd.DataFrame(participant_stats)
            
            Q1 = switch_rates_df['rate'].quantile(0.25)
            Q3 = switch_rates_df['rate'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            
            extreme_participants = set()
            outliers = switch_rates_df[switch_rates_df['rate'] > upper_bound]['playerID'].tolist()
            zero_switches = switch_rates_df[switch_rates_df['switches'] == 0]['playerID'].tolist()
            
            extreme_participants.update(outliers)
            extreme_participants.update(zero_switches)
            
            print(f"High outliers: {len(outliers)}")
            print(f"Zero switches: {len(zero_switches)}")
            print(f"Total participants to remove: {len(extreme_participants)}")
            
            self.data = self.data[~self.data['playerID'].isin(extreme_participants)].copy()
            print(f"Participants after filtering: {self.data['playerID'].nunique()}")
        else:
            print("No switch data available for participant filtering")
    
    def _remove_duplicates(self):
        """Remove duplicates by randomly selecting one playerID per sourceParticipantId-category combination."""
        print("\\nRemoving duplicates...")
        
        if 'sourceParticipantId' not in self.data.columns:
            print("No sourceParticipantId column found, skipping duplicate removal")
            return
        
        np.random.seed(42)
        
        duplicates_removed = 0
        for category in self.data['category'].unique():
            category_data = self.data[self.data['category'] == category]
            
            source_player_counts = category_data.groupby('sourceParticipantId')['playerID'].nunique()
            duplicated_sources = source_player_counts[source_player_counts > 1].index
            
            for source_id in duplicated_sources:
                players_for_source = category_data[category_data['sourceParticipantId'] == source_id]['playerID'].unique()
                
                player_to_keep = np.random.choice(players_for_source)
                players_to_remove = [p for p in players_for_source if p != player_to_keep]
                
                for player_to_remove in players_to_remove:
                    remove_mask = (
                        (self.data['sourceParticipantId'] == source_id) & 
                        (self.data['category'] == category) & 
                        (self.data['playerID'] == player_to_remove)
                    )
                    self.data = self.data[~remove_mask]
                    duplicates_removed += 1
        
        print(f"Duplicate combinations removed: {duplicates_removed}")
    
    def generate_embeddings(self):
        """Generate embeddings for all unique texts and labels."""
        print("=" * 60)
        print("GENERATING EMBEDDINGS")
        print("=" * 60)
        
        # Apply normalization
        self.data['text_normalized'] = self.data['text'].apply(normalize_text)
        
        # Get unique texts
        unique_texts = self.data['text_normalized'].unique().tolist()
        unique_texts = [text for text in unique_texts if text and text.strip()]
        print(f"Unique texts to embed: {len(unique_texts)}")
        
        # Generate text embeddings
        print("Generating text embeddings...")
        text_embeddings_array = self.model.encode(
            unique_texts, 
            prompt=self.prompts['word'],
            convert_to_numpy=True, 
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True
        )
        
        for text, embedding in zip(unique_texts, text_embeddings_array):
            self.text_embeddings[text] = embedding
        
        # Generate label embeddings for each available type
        if 'label' in self.data.columns:
            self.data['label_normalized'] = self.data['label'].apply(normalize_text)
            unique_labels = self.data['label_normalized'].dropna().unique().tolist()
            unique_labels = [label for label in unique_labels if label and label.strip()]
            
            if unique_labels:
                print(f"Generating human label embeddings: {len(unique_labels)}")
                label_embeddings_array = self.model.encode(
                    unique_labels,
                    prompt=self.prompts['label'],
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=True
                )
                
                for label, embedding in zip(unique_labels, label_embeddings_array):
                    self.label_embeddings[label] = embedding
        
        if 'snafu_gt_label' in self.data.columns:
            self.data['snafu_gt_label_normalized'] = self.data['snafu_gt_label'].apply(normalize_text)
            unique_snafu = self.data['snafu_gt_label_normalized'].dropna().unique().tolist()
            unique_snafu = [snafu for snafu in unique_snafu if snafu and snafu.strip()]
            
            if unique_snafu:
                print(f"Generating SNAFU embeddings: {len(unique_snafu)}")
                snafu_embeddings_array = self.model.encode(
                    unique_snafu,
                    prompt=self.prompts['snafu'],
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=True
                )
                
                for snafu, embedding in zip(unique_snafu, snafu_embeddings_array):
                    self.snafu_embeddings[snafu] = embedding
        
        if 'labelLLM' in self.data.columns:
            self.data['labelLLM_normalized'] = self.data['labelLLM'].apply(normalize_text)
            unique_llm = self.data['labelLLM_normalized'].dropna().unique().tolist()
            unique_llm = [llm for llm in unique_llm if llm and llm.strip()]
            
            if unique_llm:
                print(f"Generating LLM label embeddings: {len(unique_llm)}")
                llm_embeddings_array = self.model.encode(
                    unique_llm,
                    prompt=self.prompts['label'],
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=True
                )
                
                for llm, embedding in zip(unique_llm, llm_embeddings_array):
                    self.llm_embeddings[llm] = embedding
        
        print(f"\\nEmbeddings generated:")
        print(f"  Texts: {len(self.text_embeddings)}")
        print(f"  Human labels: {len(self.label_embeddings)}")
        print(f"  SNAFU categories: {len(self.snafu_embeddings)}")
        print(f"  LLM labels: {len(self.llm_embeddings)}")
    
    def calculate_distances(self):
        """Calculate semantic distances using efficient similarity_pairwise method."""
        print("=" * 60)
        print("CALCULATING SEMANTIC DISTANCES")
        print("=" * 60)
        
        results = []
        
        for idx, row in self.data.iterrows():
            text_key = row['text_normalized']
            
            if text_key not in self.text_embeddings:
                continue
            
            text_emb = self.text_embeddings[text_key]
            text_emb_batch = np.array([text_emb])
            
            result = {
                'playerID': row['playerID'],
                'category': row['category'],
                'word_index': row['word_index'],
                'text': row['text'],
                'snafu_distance': np.nan,
                'human_distance': np.nan,
                'llm_distance': np.nan
            }
            
            # Add original labels for reference
            if 'snafu_gt_label' in row:
                result['snafu_gt_label'] = row['snafu_gt_label']
            if 'label' in row:
                result['human_label'] = row['label']
            if 'labelLLM' in row:
                result['llm_label'] = row['labelLLM']
            
            # Calculate SNAFU distance
            if ('snafu_gt_label_normalized' in row and 
                pd.notna(row['snafu_gt_label_normalized']) and
                row['snafu_gt_label_normalized'] in self.snafu_embeddings):
                
                snafu_emb = self.snafu_embeddings[row['snafu_gt_label_normalized']]
                snafu_emb_batch = np.array([snafu_emb])
                similarity = self.model.similarity_pairwise(text_emb_batch, snafu_emb_batch)[0].item()
                result['snafu_distance'] = 1.0 - similarity
            
            # Calculate Human distance
            if ('label_normalized' in row and 
                pd.notna(row['label_normalized']) and
                row['label_normalized'] in self.label_embeddings):
                
                human_emb = self.label_embeddings[row['label_normalized']]
                human_emb_batch = np.array([human_emb])
                similarity = self.model.similarity_pairwise(text_emb_batch, human_emb_batch)[0].item()
                result['human_distance'] = 1.0 - similarity
            
            # Calculate LLM distance
            if ('labelLLM_normalized' in row and 
                pd.notna(row['labelLLM_normalized']) and
                row['labelLLM_normalized'] in self.llm_embeddings):
                
                llm_emb = self.llm_embeddings[row['labelLLM_normalized']]
                llm_emb_batch = np.array([llm_emb])
                similarity = self.model.similarity_pairwise(text_emb_batch, llm_emb_batch)[0].item()
                result['llm_distance'] = 1.0 - similarity
            
            # Only add if we calculated at least one distance
            if not (np.isnan(result['snafu_distance']) and 
                    np.isnan(result['human_distance']) and 
                    np.isnan(result['llm_distance'])):
                results.append(result)
            
            # Progress reporting
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(self.data)} rows")
        
        self.results_df = pd.DataFrame(results)
        print(f"\\nCalculated distances for {len(self.results_df)} data points")
        
        # Show sample statistics
        for col in ['snafu_distance', 'human_distance', 'llm_distance']:
            valid_count = self.results_df[col].notna().sum()
            if valid_count > 0:
                mean_dist = self.results_df[col].mean()
                print(f"  {col}: {valid_count} valid, mean = {mean_dist:.4f}")
    
    def create_summary_statistics(self):
        """Create and save summary statistics."""
        print("=" * 60)
        print("CREATING SUMMARY STATISTICS")
        print("=" * 60)
        
        # Overall statistics
        overall_stats = {}
        
        for dist_type in ['snafu_distance', 'human_distance', 'llm_distance']:
            valid_data = self.results_df[dist_type].dropna()
            if len(valid_data) > 0:
                overall_stats[dist_type] = {
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()),
                    'sem': float(valid_data.sem()),
                    'median': float(valid_data.median()),
                    'count': len(valid_data)
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
                valid_data = cat_data[dist_col].dropna()
                if len(valid_data) > 0:
                    category_stats.append({
                        'category': category,
                        'distance_type': dist_col,
                        'mean': valid_data.mean(),
                        'std': valid_data.std(),
                        'sem': valid_data.sem(),
                        'median': valid_data.median(),
                        'count': len(valid_data)
                    })
        
        category_stats_df = pd.DataFrame(category_stats)
        
        # Save statistics
        overall_stats_path = os.path.join(self.output_dir, "statistics", "semantic_analysis", f"overall_statistics_{self.timestamp}.json")
        with open(overall_stats_path, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        print(f"\\nOverall statistics saved to: {overall_stats_path}")
        
        if len(category_stats_df) > 0:
            category_stats_path = os.path.join(self.output_dir, "statistics", "semantic_analysis", f"category_statistics_{self.timestamp}.csv")
            category_stats_df.to_csv(category_stats_path, index=False)
            print(f"Category statistics saved to: {category_stats_path}")
        
        return overall_stats, category_stats_df
    
    def create_main_barplot(self):
        """Create the main bar plot comparing semantic distances."""
        print("=" * 60)
        print("CREATING MAIN DISTANCE BAR PLOT")
        print("=" * 60)
        
        # Calculate per-player means
        player_means = []
        for player_id in self.results_df['playerID'].unique():
            player_data = self.results_df[self.results_df['playerID'] == player_id]
            
            player_mean = {'playerID': player_id, 'n_words': len(player_data)}
            
            for dist_col in ['snafu_distance', 'human_distance', 'llm_distance']:
                valid_data = player_data[dist_col].dropna()
                if len(valid_data) > 0:
                    player_mean[f"{dist_col.replace('_distance', '_mean')}"] = valid_data.mean()
                else:
                    player_mean[f"{dist_col.replace('_distance', '_mean')}"] = np.nan
            
            player_means.append(player_mean)
        
        player_means_df = pd.DataFrame(player_means)
        print(f"Calculated means for {len(player_means_df)} players")
        
        # Calculate means and SEMs across players
        plot_data_list = []
        
        for dist_type in ['snafu_mean', 'human_mean', 'llm_mean']:
            valid_data = player_means_df[dist_type].dropna()
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                sem_val = valid_data.sem()
                
                label_name = dist_type.replace('_mean', '').upper()
                plot_data_list.append({
                    'Distance_Type': label_name,
                    'Mean_Distance': mean_val,
                    'SEM': sem_val
                })
                
                print(f"  {label_name}: {mean_val:.4f} ± {sem_val:.4f} (SEM across {len(valid_data)} players)")
        
        if not plot_data_list:
            print("No valid data for plotting!")
            return
        
        plot_data = pd.DataFrame(plot_data_list)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        # Create bar plot
        bars = ax.bar(plot_data['Distance_Type'], plot_data['Mean_Distance'], 
                     yerr=plot_data['SEM'], capsize=5, color=colors[:len(plot_data)], alpha=0.8)
        
        # Customize plot
        ax.set_title('Semantic distance between words and cluster-based labels\\n(Player-level means ± SEM)', 
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
        plot_path = os.path.join(self.output_dir, "plots", "semantic_analysis", f"semantic_distance_barplot_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Main distance bar plot saved: {plot_path}")
        
        # Save player means for other analyses
        player_means_path = os.path.join(self.output_dir, "statistics", "semantic_analysis", f"player_means_{self.timestamp}.csv")
        player_means_df.to_csv(player_means_path, index=False)
        print(f"Player-level means saved to: {player_means_path}")
    
    def create_category_comparison_plot(self):
        """Create grouped bar plot comparing distances by category."""
        if self.results_df['category'].nunique() <= 1:
            print("Not enough categories for comparison plot")
            return
            
        print("Creating category comparison plot...")
        
        categories = self.results_df['category'].unique()
        
        # Calculate means and SEMs by category
        category_means = {}
        category_sems = {}
        
        for category in categories:
            cat_data = self.results_df[self.results_df['category'] == category]
            
            category_means[category] = {}
            category_sems[category] = {}
            
            for dist_col in ['snafu_distance', 'human_distance', 'llm_distance']:
                valid_data = cat_data[dist_col].dropna()
                if len(valid_data) > 0:
                    label_name = dist_col.replace('_distance', '').upper()
                    category_means[category][label_name] = valid_data.mean()
                    category_sems[category][label_name] = valid_data.sem()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Settings
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        distance_types = ['SNAFU', 'HUMAN', 'LLM']
        bar_width = 0.25
        x = np.arange(len(categories))
        
        # Create grouped bars
        for i, dist_type in enumerate(distance_types):
            means = []
            sems = []
            for cat in categories:
                if dist_type in category_means[cat]:
                    means.append(category_means[cat][dist_type])
                    sems.append(category_sems[cat][dist_type])
                else:
                    means.append(0)
                    sems.append(0)
            
            bars = ax.bar(x + i * bar_width, means, bar_width, 
                         yerr=sems, capsize=3, label=dist_type, 
                         color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar, val, sem in zip(bars, means, sems):
                if val > 0:
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
        plot_path = os.path.join(self.output_dir, "plots", "semantic_analysis", f"semantic_distance_by_category_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Category comparison plot saved: {plot_path}")
    
    def create_word_label_visualization(self):
        """Create word-label distance visualization for first player."""
        print("Creating word-label distance visualization...")
        
        # Get first player with sufficient data
        first_player_id = self.results_df['playerID'].iloc[0]
        first_player_data = self.results_df[self.results_df['playerID'] == first_player_id].copy()
        
        if len(first_player_data) == 0:
            print("No data for word-label visualization")
            return
            
        print(f"Creating visualization for player: {first_player_id}")
        
        # Limit to first 15 words for readability
        display_data = first_player_data.head(15)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Prepare plot data
        plot_data = []
        for idx, row in display_data.iterrows():
            word_label = f"W{row['word_index']}: {row['text'][:15]}..."
            
            if not np.isnan(row['snafu_distance']):
                plot_data.append({'Word': word_label, 'Label_Type': 'SNAFU', 
                                'Distance': row['snafu_distance'], 'Word_Index': row['word_index']})
            if not np.isnan(row['human_distance']):
                plot_data.append({'Word': word_label, 'Label_Type': 'Human', 
                                'Distance': row['human_distance'], 'Word_Index': row['word_index']})
            if not np.isnan(row['llm_distance']):
                plot_data.append({'Word': word_label, 'Label_Type': 'LLM', 
                                'Distance': row['llm_distance'], 'Word_Index': row['word_index']})
        
        if not plot_data:
            print("No valid data for word-label visualization")
            return
            
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar plot
        sns.barplot(data=plot_df, x='Word', y='Distance', hue='Label_Type', 
                   palette=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        
        plt.title(f'Word-to-Label Distances\\nPlayer: {first_player_id}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Cosine Distance', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Label Type')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", "semantic_analysis", f"word_label_distances_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Word-label distances plot saved: {plot_path}")
    
    def save_detailed_results(self):
        """Save all detailed results."""
        print("Saving detailed results...")
        
        # Save word-level results
        detailed_path = os.path.join(self.output_dir, "statistics", "semantic_analysis", f"detailed_distances_{self.timestamp}.csv")
        self.results_df.to_csv(detailed_path, index=False)
        print(f"Detailed word-level distances saved to: {detailed_path}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Complete Semantic Analysis Pipeline")
        print("=" * 60)
        
        # Step 1: Load and clean data
        self.load_and_clean_data()
        
        # Step 2: Generate embeddings
        self.generate_embeddings()
        
        # Step 3: Calculate distances
        self.calculate_distances()
        
        if len(self.results_df) == 0:
            print("No valid results found. Cannot proceed with analysis.")
            return
        
        # Step 4: Create statistics
        self.create_summary_statistics()
        
        # Step 5: Create visualizations
        self.create_main_barplot()
        self.create_category_comparison_plot()
        self.create_word_label_visualization()
        
        # Step 6: Save detailed results
        self.save_detailed_results()
        
        print("\\nComplete semantic analysis finished successfully!")
        print(f"All outputs saved to: {self.output_dir}")
        
        # Summary
        print("\\nFiles created:")
        plots_dir = os.path.join(self.output_dir, "plots", "semantic_analysis")
        stats_dir = os.path.join(self.output_dir, "statistics", "semantic_analysis")
        
        for file in os.listdir(plots_dir):
            if file.endswith(f"{self.timestamp}.png"):
                print(f"  Plot: {file}")
                
        for file in os.listdir(stats_dir):
            if file.endswith(f"{self.timestamp}.csv") or file.endswith(f"{self.timestamp}.json"):
                print(f"  Data: {file}")

def find_csv_files():
    """Find all CSV files in the project directories."""
    csv_files = []
    search_dirs = ['data', 'process_for_labels', 'analysis_output', 'files_from_exp1', 'old_files', 'claude_orig_attempt']
    
    # Current directory
    try:
        for file in os.listdir('.'):
            if file.endswith('.csv') and not file.startswith('.') and os.path.isfile(file):
                csv_files.append(file)
    except:
        pass
    
    # Search directories
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                if '.venv' in root or 'venv' in root or '__pycache__' in root:
                    continue
                for file in files:
                    if file.endswith('.csv') and not file.startswith('.'):
                        csv_files.append(os.path.join(root, file))
    
    return sorted(csv_files)

def select_csv_file(csv_files):
    """Allow user to select a CSV file."""
    if not csv_files:
        print("No CSV files found.")
        return None
    
    print("Available CSV files:")
    print("=" * 60)
    
    for i, file_path in enumerate(csv_files, 1):
        try:
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            print(f"{i:2d}. {file_path} ({size_mb:.1f} MB)")
        except:
            print(f"{i:2d}. {file_path}")
    
    print("=" * 60)
    
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
            print("\\nExiting...")
            return None

def select_model():
    """Allow user to select sentence-transformer model."""
    models = [
        "all-mpnet-base-v2",
        "all-MiniLM-L6-v2", 
        "all-MiniLM-L12-v2",
        "paraphrase-multilingual-mpnet-base-v2"
    ]
    
    print("\\nAvailable sentence-transformer models:")
    print("=" * 60)
    for i, model in enumerate(models, 1):
        similarity_fn = configure_similarity_function(model)
        print(f"{i}. {model} (will use {similarity_fn})")
    print("=" * 60)
    
    while True:
        try:
            choice = input(f"Select model (1-{len(models)}) or press Enter for default: ").strip()
            
            if choice == "":
                selected_model = models[0]
                break
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(models):
                selected_model = models[choice_num - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(models)}")
                
        except ValueError:
            print("Please enter a valid number or press Enter for default")
        except KeyboardInterrupt:
            print("\\nExiting...")
            return None
    
    similarity_fn = configure_similarity_function(selected_model)
    print(f"Selected model: {selected_model}")
    print(f"Will use similarity function: {similarity_fn}")
    return selected_model

def main():
    """Main function."""
    print("Complete Semantic Analysis Pipeline")
    print("=" * 60)
    
    # Select data file
    csv_files = find_csv_files()
    data_path = select_csv_file(csv_files)
    if data_path is None:
        return
    
    # Select model
    model = select_model()
    if model is None:
        return
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Selected file not found at {data_path}")
        return
    
    # Configuration options
    print("\nConfiguration Options:")
    print("=" * 30)
    
    # Prompt configuration
    use_prompts_input = input("Use task-specific prompts? (y/N): ").strip().lower()
    use_prompts = use_prompts_input in ['y', 'yes']
    
    # Normalization configuration
    normalize_input = input("Normalize embeddings explicitly? (y/N): ").strip().lower()
    normalize_embeddings = normalize_input in ['y', 'yes']
    
    print(f"\nConfiguration:")
    print(f"  Use prompts: {use_prompts}")
    print(f"  Normalize embeddings: {normalize_embeddings}")
    print(f"  Model: {model}")
    
    # Run complete analysis
    analyzer = CompleteSemanticAnalyzer(
        data_path=data_path, 
        model_name=model, 
        use_prompts=use_prompts,
        normalize_embeddings=normalize_embeddings
    )
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()