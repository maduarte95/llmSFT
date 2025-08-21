"""
Human Labels Embeddings Generation Script
=========================================

This script generates embeddings for text and human labels using Together AI's embedding API.
Saves embeddings to an /embeddings directory for reuse in semantic similarity analysis.

Usage:
    uv run python generate_embeddings_human.py
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import re
from together import Together
from typing import List, Dict, Any
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

class HumanEmbeddingsGenerator:
    def __init__(self, data_path: str, model: str = "togethercomputer/m2-bert-80M-32k-retrieval", output_dir: str = "embeddings"):
        """
        Initialize the embeddings generator for human labels.
        
        Parameters:
        data_path: str, path to the CSV file with data
        model: str, embedding model to use
        output_dir: str, directory to save embeddings
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = model
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Together client
        self.client = Together()
        
        self.data = None
        self.text_embeddings = {}
        self.label_embeddings = {}
        self.snafu_embeddings = {}
        
    def load_and_clean_data(self):
        """
        Load data and remove rows with missing labels.
        Apply the same filtering criteria as analysis_pipeline.py.
        """
        print("=" * 60)
        print("LOADING AND CLEANING DATA")
        print("=" * 60)
        
        # Load the data
        self.data = pd.read_csv(self.data_path)
        print(f"Original data shape: {self.data.shape}")
        
        # Check required columns
        required_cols = ['text', 'label', 'snafu_gt_label', 'playerID', 'category', 'word_index']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing labels
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=['text', 'label', 'snafu_gt_label'])
        print(f"After removing missing labels: {len(self.data)} (removed {initial_count - len(self.data)})")
        
        # Remove word_index 0 (first word doesn't have a switch)
        self.data = self.data[self.data['word_index'] > 0].copy()
        print(f"After removing word_index 0: {len(self.data)}")
        
        # Apply participant filtering (same as analysis_pipeline.py)
        self._filter_extreme_participants()
        self._remove_duplicates()
        
        print(f"Final cleaned data shape: {self.data.shape}")
        print(f"Unique participants: {self.data['playerID'].nunique()}")
        print("Participants per category:")
        print(self.data.groupby('category')['playerID'].nunique())
        
    def _filter_extreme_participants(self):
        """
        Filter out participants with extreme switch rates or zero switches.
        Based on analysis_pipeline.py logic.
        """
        print("\nFiltering extreme participants...")
        
        # Calculate switch rates per participant
        participant_stats = []
        for player_id in self.data['playerID'].unique():
            player_data = self.data[self.data['playerID'] == player_id]
            
            if len(player_data) > 0:
                category = player_data['category'].iloc[0]
                total_words = len(player_data)
                
                # Check if we have switch columns for filtering
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
            
            # Detect outliers using IQR method
            Q1 = switch_rates_df['rate'].quantile(0.25)
            Q3 = switch_rates_df['rate'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            
            # Find participants with extreme rates or zero switches
            extreme_participants = set()
            outliers = switch_rates_df[switch_rates_df['rate'] > upper_bound]['playerID'].tolist()
            zero_switches = switch_rates_df[switch_rates_df['switches'] == 0]['playerID'].tolist()
            
            extreme_participants.update(outliers)
            extreme_participants.update(zero_switches)
            
            print(f"High outliers: {len(outliers)}")
            print(f"Zero switches: {len(zero_switches)}")
            print(f"Total participants to remove: {len(extreme_participants)}")
            
            # Apply filtering
            self.data = self.data[~self.data['playerID'].isin(extreme_participants)].copy()
            print(f"Participants after filtering: {self.data['playerID'].nunique()}")
        else:
            print("No switch data available for participant filtering")
    
    def _remove_duplicates(self):
        """
        Remove duplicates by randomly selecting one playerID per sourceParticipantId-category combination.
        """
        print("\nRemoving duplicates...")
        
        if 'sourceParticipantId' not in self.data.columns:
            print("No sourceParticipantId column found, skipping duplicate removal")
            return
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        duplicates_removed = 0
        for category in self.data['category'].unique():
            category_data = self.data[self.data['category'] == category]
            
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
                        (self.data['sourceParticipantId'] == source_id) & 
                        (self.data['category'] == category) & 
                        (self.data['playerID'] == player_to_remove)
                    )
                    self.data = self.data[~remove_mask]
                    duplicates_removed += 1
        
        print(f"Duplicate combinations removed: {duplicates_removed}")
    
    def get_unique_texts(self):
        """
        Get unique texts and labels for embedding generation with normalization.
        """
        print("\nExtracting and normalizing unique texts and labels...")
        
        # Apply normalization to all text fields
        self.data['text_normalized'] = self.data['text'].apply(normalize_text)
        self.data['label_normalized'] = self.data['label'].apply(normalize_text)
        self.data['snafu_gt_label_normalized'] = self.data['snafu_gt_label'].apply(normalize_text)
        
        # Get unique normalized values
        unique_texts = self.data['text_normalized'].unique().tolist()
        unique_labels = self.data['label_normalized'].unique().tolist()
        unique_snafu = self.data['snafu_gt_label_normalized'].unique().tolist()
        
        # Remove any empty strings or NaN values
        unique_texts = [text for text in unique_texts if text and text.strip()]
        unique_labels = [label for label in unique_labels if label and label.strip()]
        unique_snafu = [snafu for snafu in unique_snafu if snafu and snafu.strip()]
        
        print(f"Unique normalized texts: {len(unique_texts)}")
        print(f"Unique normalized human labels: {len(unique_labels)}")
        print(f"Unique normalized SNAFU categories: {len(unique_snafu)}")
        
        return unique_texts, unique_labels, unique_snafu
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> Dict[str, List[float]]:
        """
        Generate embeddings for a list of texts using batch processing.
        
        Parameters:
        texts: List of texts to embed
        batch_size: Number of texts to process in each batch
        
        Returns:
        Dictionary mapping text to embedding vector
        """
        embeddings = {}
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)")
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                # Store embeddings
                for j, embedding_data in enumerate(response.data):
                    text = batch[j]
                    embedding = embedding_data.embedding
                    embeddings[text] = embedding
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Process texts individually if batch fails
                for text in batch:
                    try:
                        response = self.client.embeddings.create(
                            model=self.model,
                            input=text
                        )
                        embeddings[text] = response.data[0].embedding
                    except Exception as e2:
                        print(f"Error processing individual text '{text[:50]}...': {e2}")
                        
        return embeddings
    
    def generate_all_embeddings(self):
        """
        Generate embeddings for all unique texts, labels, and SNAFU categories.
        """
        print("=" * 60)
        print("GENERATING EMBEDDINGS")
        print("=" * 60)
        
        unique_texts, unique_labels, unique_snafu = self.get_unique_texts()
        
        # Generate embeddings for texts
        print("\nGenerating text embeddings...")
        self.text_embeddings = self.generate_embeddings_batch(unique_texts)
        
        # Generate embeddings for human labels
        print("\nGenerating human label embeddings...")
        self.label_embeddings = self.generate_embeddings_batch(unique_labels)
        
        # Generate embeddings for SNAFU categories
        print("\nGenerating SNAFU category embeddings...")
        self.snafu_embeddings = self.generate_embeddings_batch(unique_snafu)
        
        print(f"\nEmbeddings generated:")
        print(f"  Texts: {len(self.text_embeddings)}")
        print(f"  Human labels: {len(self.label_embeddings)}")
        print(f"  SNAFU categories: {len(self.snafu_embeddings)}")
    
    def save_embeddings(self):
        """
        Save embeddings to files with metadata.
        """
        print("=" * 60)
        print("SAVING EMBEDDINGS")
        print("=" * 60)
        
        # Create metadata
        metadata = {
            'model': self.model,
            'data_path': self.data_path,
            'data_shape': self.data.shape,
            'n_participants': self.data['playerID'].nunique(),
            'categories': self.data['category'].unique().tolist(),
            'embedding_dims': len(list(self.text_embeddings.values())[0]) if self.text_embeddings else 0,
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save embeddings as pickle files (preserving exact float precision)
        embeddings_data = {
            'text_embeddings': self.text_embeddings,
            'label_embeddings': self.label_embeddings,
            'snafu_embeddings': self.snafu_embeddings,
            'metadata': metadata
        }
        
        # Save combined file
        embeddings_path = os.path.join(self.output_dir, "human_embeddings.pkl")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        print(f"Human embeddings saved to: {embeddings_path}")
        
        # Save metadata as JSON for easy inspection
        metadata_path = os.path.join(self.output_dir, "human_embeddings_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
        
        # Save data mapping for analysis (including normalized versions)
        data_mapping = self.data[['text', 'label', 'snafu_gt_label', 'text_normalized', 'label_normalized', 'snafu_gt_label_normalized', 'playerID', 'category', 'word_index']].copy()
        mapping_path = os.path.join(self.output_dir, "human_data_mapping.csv")
        data_mapping.to_csv(mapping_path, index=False)
        print(f"Data mapping saved to: {mapping_path}")
    
    def run_generation(self):
        """
        Run the complete embeddings generation pipeline.
        """
        print("Starting Human Labels Embeddings Generation")
        print("=" * 60)
        
        # Step 1: Load and clean data
        self.load_and_clean_data()
        
        # Step 2: Generate embeddings
        self.generate_all_embeddings()
        
        # Step 3: Save embeddings
        self.save_embeddings()
        
        print("\nEmbeddings generation completed successfully!")
        print(f"All outputs saved to: {self.output_dir}")

def find_csv_files():
    """Find all CSV files in the current directory and subdirectories."""
    csv_files = []
    search_dirs = ['.', 'data', 'process_for_labels', 'analysis_output']
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.csv') and not file.startswith('.'):
                        csv_files.append(os.path.join(root, file))
    
    return sorted(csv_files)

def select_csv_file(csv_files):
    """Allow user to select a CSV file from the available options."""
    if not csv_files:
        print("No CSV files found in the current directory or subdirectories.")
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
            print("\nExiting...")
            return None

def select_embedding_model():
    """Allow user to select embedding model."""
    models = [
        "togethercomputer/m2-bert-80M-32k-retrieval",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5-vllm",
        "intfloat/multilingual-e5-large-instruct"
    ]
    
    print("\nAvailable embedding models:")
    print("=" * 60)
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    print("=" * 60)
    
    while True:
        try:
            choice = input(f"Select embedding model (1-{len(models)}) or press Enter for default: ").strip()
            
            if choice == "":
                selected_model = models[0]  # Default
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
            print("\nExiting...")
            return None
    
    print(f"Selected model: {selected_model}")
    return selected_model

def main():
    """Main function to run the embeddings generation."""
    print("Human Labels Embeddings Generation")
    print("=" * 60)
    
    # Find available CSV files
    csv_files = find_csv_files()
    
    # Let user select file
    data_path = select_csv_file(csv_files)
    if data_path is None:
        return
    
    # Let user select model
    model = select_embedding_model()
    if model is None:
        return
    
    # Check if selected file exists
    if not os.path.exists(data_path):
        print(f"Error: Selected file not found at {data_path}")
        return
    
    # Run embeddings generation
    generator = HumanEmbeddingsGenerator(data_path, model)
    generator.run_generation()

if __name__ == "__main__":
    main()