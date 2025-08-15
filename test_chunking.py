#!/usr/bin/env python3
"""
Test script to verify chunking logic for switch prediction.
Run with: uv run python test_chunking.py
"""

import pandas as pd
from base_classifiers import BaseSwitchPredictor
from typing import List, Dict, Any

class TestPredictor(BaseSwitchPredictor):
    """Test implementation to access chunking functionality."""
    
    def __init__(self):
        # Initialize without provider for testing
        self.provider = None
        self.config = None
    
    def process_batch_results(self, batch_id: str, original_data: pd.DataFrame, requests_metadata: List[Dict]) -> pd.DataFrame:
        """Not needed for testing."""
        pass

def test_chunking():
    """Test the chunking logic with sample data."""
    print("🧪 Testing Switch Prediction Chunking Logic")
    print("=" * 50)
    
    # Load real data
    try:
        data = pd.read_csv('filtered_data_for_analysis.csv')
        print(f"✅ Loaded real data: {data.shape}")
        print(f"   Players: {data['playerID'].nunique()}")
        print(f"   Categories: {list(data['category'].unique())}")
    except Exception as e:
        print(f"❌ Could not load real data: {e}")
        print("📝 Creating sample data instead...")
        
        # Create sample data if real data not available
        data = pd.DataFrame({
            'playerID': ['player1'] * 6 + ['player2'] * 4,
            'category': ['animals'] * 6 + ['animals'] * 4,
            'word_index': [0, 1, 2, 3, 4, 5] + [0, 1, 2, 3],
            'text': ['cat', 'dog', 'lion', 'tiger', 'apple', 'banana'] + ['bird', 'fish', 'cow', 'sheep']
        })
        print(f"✅ Created sample data: {data.shape}")
    
    # Initialize test predictor
    predictor = TestPredictor()
    
    # Test with first player only for clarity
    first_player = data['playerID'].iloc[0]
    first_category = data['category'].iloc[0]
    test_data = data[(data['playerID'] == first_player) & (data['category'] == first_category)].head(6)
    
    print(f"\n📊 Test data for {first_player} ({first_category}):")
    print(test_data[['word_index', 'text']].to_string(index=False))
    
    # Create chunks
    print(f"\n🔍 Generating chunks...")
    chunks = predictor.create_chunk_data(test_data)
    prediction_chunks = [c for c in chunks if c['has_next_word']]
    
    print(f"Generated {len(chunks)} total chunks, {len(prediction_chunks)} prediction chunks\n")
    
    # Display chunks in detail
    print("📋 CHUNK DETAILS:")
    print("-" * 60)
    
    for i, chunk in enumerate(chunks):
        chunk_words_str = " → ".join(chunk['chunk_words'])
        has_next = "✅" if chunk['has_next_word'] else "❌"
        next_word = chunk['next_word'] if chunk['next_word'] else "None"
        
        print(f"Chunk {i+1:2d}: [{chunk_words_str}]")
        print(f"           End Index: {chunk['chunk_end_index']}")
        print(f"           Has Next: {has_next}")
        print(f"           Next Word: {next_word}")
        print(f"           Can Predict: {'YES' if chunk['has_next_word'] else 'NO'}")
        print()
    
    # Test with multiple players
    print("🔍 Testing with multiple players...")
    all_chunks = predictor.create_chunk_data(data)
    all_prediction_chunks = [c for c in all_chunks if c['has_next_word']]
    
    print(f"Total chunks across all players: {len(all_chunks)}")
    print(f"Total prediction chunks: {len(all_prediction_chunks)}")
    
    # Group by player to verify separation
    chunks_by_player = {}
    for chunk in all_chunks:
        player_id = chunk['player_id']
        if player_id not in chunks_by_player:
            chunks_by_player[player_id] = []
        chunks_by_player[player_id].append(chunk)
    
    print(f"\n👥 Chunks per player:")
    for player_id, player_chunks in chunks_by_player.items():
        prediction_count = sum(1 for c in player_chunks if c['has_next_word'])
        total_count = len(player_chunks)
        print(f"   {player_id}: {total_count} total, {prediction_count} prediction chunks")
    
    # Verify no cross-contamination between players
    print(f"\n🔒 Verifying player separation...")
    issues = []
    for chunk in all_chunks:
        player_id = chunk['player_id']
        category = chunk['category']
        
        # Check that all words in chunk belong to same player
        player_data = data[(data['playerID'] == player_id) & (data['category'] == category)]
        player_words = player_data.sort_values('word_index')['text'].tolist()
        
        # Verify chunk words are consecutive from this player's sequence
        chunk_end_idx = chunk['chunk_end_index']
        expected_words = player_words[:chunk_end_idx + 1]
        
        if chunk['chunk_words'] != expected_words:
            issues.append(f"Player {player_id}: chunk mismatch at index {chunk_end_idx}")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ All chunks properly separated by player - no cross-contamination!")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Total sequences: {data.groupby(['playerID', 'category']).ngroups}")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Prediction chunks: {len(all_prediction_chunks)}")
    print(f"   Players tested: {len(chunks_by_player)}")
    print(f"   Chunking integrity: {'✅ PASS' if not issues else '❌ FAIL'}")

if __name__ == "__main__":
    test_chunking()