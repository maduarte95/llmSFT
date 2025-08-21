"""
Word-to-Label Distance Visualization
===================================

This script visualizes the distances between each word and its corresponding labels:
- Each word wi has its own human_label_i, snafu_gt_label_i, and llm_label_i
- We calculate distance(wi, human_label_i), distance(wi, snafu_gt_label_i), distance(wi, llm_label_i)
- Shows which label type is semantically closest to each word

Usage:
    uv run python visualize_word_label_distances.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import re
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings('ignore')

def normalize_text(text):
    """
    Apply comprehensive text normalization for consistent embeddings.
    
    Parameters:
    text: str, input text to normalize
    
    Returns:
    str, normalized text
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Remove punctuation (hyphens, apostrophes, commas, periods)
    text = re.sub(r'[-\',.]', '', text)
    # Remove whitespace
    text = re.sub(r'\s+', '', text)
    
    return text

class WordLabelDistanceVisualizer:
    def __init__(self, human_embeddings_path: str, llm_embeddings_path: str, output_dir: str = "analysis_output"):
        """Initialize the word-label distance visualizer."""
        self.human_embeddings_path = human_embeddings_path
        self.llm_embeddings_path = llm_embeddings_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(os.path.join(output_dir, "plots", "word_label_distances"), exist_ok=True)
        
        # Data containers
        self.text_embeddings = {}
        self.label_embeddings = {}
        self.snafu_embeddings = {}
        self.llm_embeddings = {}
        self.human_data = None
        self.llm_data = None
        self.first_player_data = None
        
    def load_embeddings(self):
        """Load embeddings from pickle files."""
        print("Loading embeddings...")
        
        # Load human embeddings
        with open(self.human_embeddings_path, 'rb') as f:
            human_data = pickle.load(f)
        
        self.text_embeddings = human_data['text_embeddings']
        self.label_embeddings = human_data['label_embeddings']
        self.snafu_embeddings = human_data['snafu_embeddings']
        
        # Load LLM embeddings
        with open(self.llm_embeddings_path, 'rb') as f:
            llm_data = pickle.load(f)
        
        self.llm_embeddings = llm_data['llm_embeddings']
        
        print(f"Loaded embeddings: {len(self.text_embeddings)} texts, {len(self.label_embeddings)} human labels, {len(self.snafu_embeddings)} SNAFU categories, {len(self.llm_embeddings)} LLM labels")
        
    def load_data_mappings(self):
        """Load data mappings and get first player data."""
        print("Loading data mappings...")
        
        # Load human data mapping
        human_mapping_path = os.path.join(os.path.dirname(self.human_embeddings_path), "human_data_mapping.csv")
        self.human_data = pd.read_csv(human_mapping_path)
        
        # Load LLM data mapping
        llm_mapping_path = os.path.join(os.path.dirname(self.llm_embeddings_path), "llm_data_mapping.csv")
        self.llm_data = pd.read_csv(llm_mapping_path)
        
        # Get first player
        first_player_id = self.human_data['playerID'].iloc[0]
        print(f"First player ID: {first_player_id}")
        
        # Get data for first player
        human_first_player = self.human_data[self.human_data['playerID'] == first_player_id].copy()
        llm_first_player = self.llm_data[self.llm_data['playerID'] == first_player_id].copy()
        
        # Align the data
        human_keys = set(zip(human_first_player['playerID'], human_first_player['word_index']))
        llm_keys = set(zip(llm_first_player['playerID'], llm_first_player['word_index']))
        common_keys = human_keys.intersection(llm_keys)
        
        def filter_by_keys(df, keys):
            return df[df.apply(lambda row: (row['playerID'], row['word_index']) in keys, axis=1)]
        
        human_first_player = filter_by_keys(human_first_player, common_keys)
        llm_first_player = filter_by_keys(llm_first_player, common_keys)
        
        # Sort both for consistent ordering
        human_first_player = human_first_player.sort_values('word_index').reset_index(drop=True)
        llm_first_player = llm_first_player.sort_values('word_index').reset_index(drop=True)
        
        # Combine the data
        self.first_player_data = human_first_player.copy()
        self.first_player_data['llm_label'] = llm_first_player['labelLLM']
        self.first_player_data['llm_label_normalized'] = llm_first_player['labelLLM_normalized']
        
        print(f"First player data points: {len(self.first_player_data)}")
        print(f"Category: {self.first_player_data['category'].iloc[0]}")
        
        # Filter to only rows where we have all embeddings
        valid_rows = []
        for idx, row in self.first_player_data.iterrows():
            if (row['text_normalized'] in self.text_embeddings and 
                row['label_normalized'] in self.label_embeddings and 
                row['snafu_gt_label_normalized'] in self.snafu_embeddings and 
                row['llm_label_normalized'] in self.llm_embeddings):
                valid_rows.append(idx)
        
        self.first_player_data = self.first_player_data.loc[valid_rows].reset_index(drop=True)
        print(f"Valid data points with all embeddings: {len(self.first_player_data)}")
        
    def calculate_word_label_distances(self):
        """Calculate distances between each word and its corresponding labels."""
        print("Calculating word-to-label distances...")
        
        distances_data = []
        
        for idx, row in self.first_player_data.iterrows():
            text = row['text']
            human_label = row['label']
            snafu_gt_label = row['snafu_gt_label']
            llm_label = row['llm_label']
            word_index = row['word_index']
            
            # Get embeddings using normalized versions
            text_emb = np.array(self.text_embeddings[row['text_normalized']])
            human_emb = np.array(self.label_embeddings[row['label_normalized']])
            snafu_emb = np.array(self.snafu_embeddings[row['snafu_gt_label_normalized']])
            llm_emb = np.array(self.llm_embeddings[row['llm_label_normalized']])
            
            # Calculate distances between this word and its labels
            snafu_dist = cosine(text_emb, snafu_emb)
            human_dist = cosine(text_emb, human_emb)
            llm_dist = cosine(text_emb, llm_emb)
            
            # Find closest label type
            distances = [('SNAFU', snafu_dist), ('Human', human_dist), ('LLM', llm_dist)]
            closest_type = min(distances, key=lambda x: x[1])[0]
            
            distances_data.append({
                'word_index': word_index,
                'text': text,
                'human_label': human_label,
                'snafu_gt_label': snafu_gt_label,
                'llm_label': llm_label,
                'snafu_distance': snafu_dist,
                'human_distance': human_dist,
                'llm_distance': llm_dist,
                'closest_type': closest_type
            })
        
        self.distances_df = pd.DataFrame(distances_data)
        print(f"Calculated distances for {len(self.distances_df)} word-label pairs")
        
    def create_distance_comparison_plot(self):
        """Create a plot comparing distances for each word."""
        print("Creating distance comparison plot...")
        
        # Prepare data for plotting
        plot_data = []
        
        for idx, row in self.distances_df.iterrows():
            word_label = f"W{row['word_index']}: {row['text'][:20]}..."
            
            plot_data.extend([
                {'Word': word_label, 'Label_Type': 'SNAFU', 'Distance': row['snafu_distance'], 'Word_Index': row['word_index']},
                {'Word': word_label, 'Label_Type': 'Human', 'Distance': row['human_distance'], 'Word_Index': row['word_index']},
                {'Word': word_label, 'Label_Type': 'LLM', 'Distance': row['llm_distance'], 'Word_Index': row['word_index']}
            ])
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Create grouped bar plot
        sns.barplot(data=plot_df, x='Word', y='Distance', hue='Label_Type', 
                   palette=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        
        plt.title(f'Word-to-Label Distances\\nPlayer: {self.first_player_data["playerID"].iloc[0]}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Cosine Distance', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Label Type')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", "word_label_distances", "word_label_distances_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distance comparison plot saved: {plot_path}")
        
    def create_heatmap_visualization(self):
        """Create a heatmap showing distances and labels for each word."""
        print("Creating distance heatmap with labels...")
        
        # Prepare matrix data and annotation labels
        n_words = len(self.distances_df)
        distance_matrix = np.zeros((n_words, 3))  # 3 label types
        annotation_labels = np.empty((n_words, 3), dtype=object)
        
        for idx, row in self.distances_df.iterrows():
            # Distance values
            distance_matrix[idx, 0] = row['snafu_distance']
            distance_matrix[idx, 1] = row['human_distance'] 
            distance_matrix[idx, 2] = row['llm_distance']
            
            # Create annotation text with label + distance
            snafu_label = row['snafu_gt_label'][:15] + '...' if len(row['snafu_gt_label']) > 15 else row['snafu_gt_label']
            human_label = row['human_label'][:15] + '...' if len(row['human_label']) > 15 else row['human_label']
            llm_label = row['llm_label'][:15] + '...' if len(row['llm_label']) > 15 else row['llm_label']
            
            annotation_labels[idx, 0] = f"{snafu_label}\\n({row['snafu_distance']:.3f})"
            annotation_labels[idx, 1] = f"{human_label}\\n({row['human_distance']:.3f})"
            annotation_labels[idx, 2] = f"{llm_label}\\n({row['llm_distance']:.3f})"
        
        # Create word labels
        word_labels = [f"W{row['word_index']}: {row['text'][:25]}..." 
                      for idx, row in self.distances_df.iterrows()]
        
        # Create heatmap
        plt.figure(figsize=(12, max(8, len(word_labels) * 0.8)))
        
        sns.heatmap(distance_matrix,
                   xticklabels=['SNAFU', 'Human', 'LLM'],
                   yticklabels=word_labels,
                   annot=annotation_labels,
                   fmt='',  # Use custom annotation format
                   cmap='viridis_r',  # Darker = closer
                   cbar_kws={'label': 'Cosine Distance'},
                   annot_kws={'fontsize': 8, 'ha': 'center', 'va': 'center'})
        
        plt.title(f'Word-to-Label Distance Heatmap\\nPlayer: {self.first_player_data["playerID"].iloc[0]}\\n(Showing Label + Distance)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Label Type', fontsize=12)
        plt.ylabel('Words', fontsize=12)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", "word_label_distances", "word_label_distances_heatmap.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distance heatmap with labels saved: {plot_path}")
        
    def create_closest_label_summary(self):
        """Create a summary showing which label type is closest to each word."""
        print("Creating closest label summary...")
        
        # Count which label type is closest most often
        closest_counts = self.distances_df['closest_type'].value_counts()
        
        # Create pie chart
        plt.figure(figsize=(8, 6))
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        plt.pie(closest_counts.values, labels=closest_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title(f'Closest Label Type Distribution\\nPlayer: {self.first_player_data["playerID"].iloc[0]}', 
                 fontsize=14, fontweight='bold')
        plt.axis('equal')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", "word_label_distances", "closest_label_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Closest label distribution saved: {plot_path}")
        
        # Print summary statistics
        print("\\nSummary Statistics:")
        print(f"Average SNAFU distance: {self.distances_df['snafu_distance'].mean():.4f}")
        print(f"Average Human distance: {self.distances_df['human_distance'].mean():.4f}")  
        print(f"Average LLM distance: {self.distances_df['llm_distance'].mean():.4f}")
        print("\\nClosest label type counts:")
        print(closest_counts)
        
        # Save detailed results
        detailed_path = os.path.join(self.output_dir, "plots", "word_label_distances", "detailed_word_label_distances.csv")
        self.distances_df.to_csv(detailed_path, index=False)
        print(f"\\nDetailed results saved: {detailed_path}")
        
    def run_visualization(self):
        """Run the complete word-label distance visualization pipeline."""
        print("Starting Word-Label Distance Visualization")
        print("=" * 60)
        
        # Step 1: Load embeddings
        self.load_embeddings()
        
        # Step 2: Load data mappings and get first player
        self.load_data_mappings()
        
        if len(self.first_player_data) == 0:
            print("No valid data found for first player!")
            return
        
        # Step 3: Calculate word-label distances
        self.calculate_word_label_distances()
        
        # Step 4: Create visualizations
        self.create_distance_comparison_plot()
        self.create_heatmap_visualization()
        self.create_closest_label_summary()
        
        print("\\nWord-label distance visualization completed successfully!")
        print(f"All outputs saved to: {os.path.join(self.output_dir, 'plots', 'word_label_distances')}")

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
    
    if not human_files or not llm_files:
        print("Please ensure both human and LLM embedding files exist in embeddings/ directory.")
        return None, None
    
    print("Available human embedding files:")
    for i, file_path in enumerate(human_files, 1):
        print(f"{i}. {file_path}")
    
    choice = input(f"Select human embeddings file (1-{len(human_files)}) or press Enter for first: ").strip()
    if choice == "":
        human_file = human_files[0]
    else:
        human_file = human_files[int(choice) - 1]
    
    print(f"\\nAvailable LLM embedding files:")
    for i, file_path in enumerate(llm_files, 1):
        print(f"{i}. {file_path}")
    
    choice = input(f"Select LLM embeddings file (1-{len(llm_files)}) or press Enter for first: ").strip()
    if choice == "":
        llm_file = llm_files[0]
    else:
        llm_file = llm_files[int(choice) - 1]
    
    print(f"\\nSelected files:")
    print(f"  Human: {human_file}")
    print(f"  LLM: {llm_file}")
    
    return human_file, llm_file

def main():
    """Main function to run the word-label distance visualization."""
    print("Word-Label Distance Visualization")
    print("=" * 60)
    
    # Find and select embedding files
    human_file, llm_file = select_embedding_files()
    if human_file is None or llm_file is None:
        return
    
    # Run visualization
    visualizer = WordLabelDistanceVisualizer(human_file, llm_file)
    visualizer.run_visualization()

if __name__ == "__main__":
    main()