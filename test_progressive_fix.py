#!/usr/bin/env python3
"""
Test script to verify the progressive classification fixes work correctly.
"""

import pandas as pd
import sys
sys.path.append('.')

from base_classifiers import BaseProgressiveSwitchClassifier

class TestProgressiveClassifier(BaseProgressiveSwitchClassifier):
    def __init__(self):
        # Mock provider for testing
        self.provider = None
        self.config = None
    
    def process_batch_results(self, batch_id, original_data, requests_metadata):
        # Not needed for this test
        pass

def test_chunk_creation():
    """Test that chunk creation follows the correct progressive logic."""
    
    # Create test data: [cat, dog, horse, fish]
    test_data = pd.DataFrame({
        'playerID': ['P1', 'P1', 'P1', 'P1'],
        'category': ['animals', 'animals', 'animals', 'animals'],
        'word_index': [0, 1, 2, 3],
        'text': ['cat', 'dog', 'horse', 'fish']
    })
    
    classifier = TestProgressiveClassifier()
    chunks = classifier.create_progressive_chunk_data(test_data)
    
    print(f"Test data: {test_data['text'].tolist()}")
    print(f"Generated {len(chunks)} chunks:")
    
    target_words_classified = set()
    
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk['chunk_words']} -> classify word {chunk['target_word_index']}: '{chunk['target_word']}'")
        target_words_classified.add(chunk['target_word_index'])
    
    # Expected progressive logic:
    # Chunk 1: [cat, dog] -> classify cat (index 0)
    # Chunk 2: [cat, dog, horse] -> classify dog (index 1)  
    # Chunk 3: [cat, dog, horse, fish] -> classify horse (index 2)
    # Chunk 4: [cat, dog, horse, fish] -> classify fish (index 3)
    
    expected_chunks = [
        (['cat', 'dog'], 0, 'cat'),
        (['cat', 'dog', 'horse'], 1, 'dog'),
        (['cat', 'dog', 'horse', 'fish'], 2, 'horse'),
        (['cat', 'dog', 'horse', 'fish'], 3, 'fish')
    ]
    
    print(f"\nExpected progression:")
    for i, (exp_chunk, exp_idx, exp_word) in enumerate(expected_chunks):
        print(f"  Expected Chunk {i+1}: {exp_chunk} -> classify word {exp_idx}: '{exp_word}'")
    
    # Check that all words are classified
    expected_word_indices = set(range(len(test_data)))
    print(f"\nExpected word indices to classify: {expected_word_indices}")
    print(f"Actual word indices classified: {target_words_classified}")
    
    # Verify the exact progression
    success = True
    if len(chunks) != len(expected_chunks):
        print(f"FAILURE: Expected {len(expected_chunks)} chunks, got {len(chunks)}")
        success = False
    
    for i, (chunk, (exp_chunk, exp_idx, exp_word)) in enumerate(zip(chunks, expected_chunks)):
        if chunk['chunk_words'] != exp_chunk:
            print(f"FAILURE: Chunk {i+1} words mismatch. Expected {exp_chunk}, got {chunk['chunk_words']}")
            success = False
        if chunk['target_word_index'] != exp_idx:
            print(f"FAILURE: Chunk {i+1} target index mismatch. Expected {exp_idx}, got {chunk['target_word_index']}")
            success = False
        if chunk['target_word'] != exp_word:
            print(f"FAILURE: Chunk {i+1} target word mismatch. Expected '{exp_word}', got '{chunk['target_word']}'")
            success = False
    
    if expected_word_indices == target_words_classified and success:
        print("SUCCESS: All words classified with correct progressive logic!")
        return True
    else:
        missing = expected_word_indices - target_words_classified
        extra = target_words_classified - expected_word_indices
        if missing or extra:
            print(f"FAILURE: Missing {missing}, Extra {extra}")
        return False

def test_edge_cases():
    """Test edge cases like 2-word sequences."""
    
    # Test 2-word sequence
    test_data = pd.DataFrame({
        'playerID': ['P1', 'P1'],
        'category': ['animals', 'animals'],
        'word_index': [0, 1],
        'text': ['cat', 'dog']
    })
    
    classifier = TestProgressiveClassifier()
    chunks = classifier.create_progressive_chunk_data(test_data)
    
    print(f"\n2-word test: {test_data['text'].tolist()}")
    print(f"Generated {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk['chunk_words']} -> classify word {chunk['target_word_index']}: '{chunk['target_word']}'")
    
    # Should have 2 chunks: [cat, dog] -> classify cat, and [cat, dog] -> classify dog
    expected_chunks = 2
    if len(chunks) == expected_chunks:
        print(f"SUCCESS: 2-word sequence generates {expected_chunks} chunks")
        return True
    else:
        print(f"FAILURE: Expected {expected_chunks} chunks, got {len(chunks)}")
        return False

if __name__ == "__main__":
    print("Testing progressive classification fixes...")
    print("="*50)
    
    success1 = test_chunk_creation()
    success2 = test_edge_cases()
    
    print("\n" + "="*50)
    if success1 and success2:
        print("ALL TESTS PASSED! The fixes work correctly.")
    else:
        print("Some tests failed. Please check the implementation.")