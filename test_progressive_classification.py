#!/usr/bin/env python3
"""
Test script for progressive classification implementation.
Creates a small test dataset and validates the chunking logic.
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_classifiers import BaseProgressiveSwitchClassifier
from anthropic_providers import AnthropicProvider


class TestProgressiveClassifier(BaseProgressiveSwitchClassifier):
    """Test implementation of progressive classifier for validation."""
    
    def __init__(self):
        # Create a dummy provider for testing chunking logic
        provider = None  # We won't actually make API calls
        super().__init__(provider, config=None)
    
    def process_batch_results(self, batch_id, original_data, requests_metadata):
        """Dummy implementation for testing."""
        return original_data


def create_test_data():
    """Create a small test dataset."""
    test_data = []
    
    # Player 1, Category 1: 4 words
    for i, word in enumerate(['cat', 'dog', 'hamster', 'bird']):
        test_data.append({
            'playerID': 'P1',
            'category': 'animals',
            'word_index': i,
            'text': word
        })
    
    # Player 2, Category 1: 3 words
    for i, word in enumerate(['apple', 'banana', 'orange']):
        test_data.append({
            'playerID': 'P2',
            'category': 'fruits',
            'word_index': i,
            'text': word
        })
    
    return pd.DataFrame(test_data)


def test_progressive_chunking():
    """Test the progressive chunking logic."""
    print("Testing Progressive Classification Chunking Logic")
    print("=" * 50)
    
    # Create test data
    data = create_test_data()
    print(f"Test data created with {len(data)} rows")
    print("\nTest data:")
    print(data.to_string())
    
    # Create test classifier
    classifier = TestProgressiveClassifier()
    
    # Test chunking
    print(f"\nTesting progressive chunk creation...")
    chunks = classifier.create_progressive_chunk_data(data)
    
    print(f"Created {len(chunks)} chunks")
    
    # Analyze chunks
    expected_chunks = []
    
    # For P1 (4 words: cat, dog, hamster, bird):
    # Chunk 1: [cat, dog] -> classify 'cat' (index 0)
    # Chunk 2: [cat, dog, hamster] -> classify 'dog' (index 1) 
    # Chunk 3: [cat, dog, hamster, bird] -> classify 'hamster' (index 2)
    expected_chunks.extend([
        {'player': 'P1', 'chunk': ['cat', 'dog'], 'target': 'cat', 'target_idx': 0},
        {'player': 'P1', 'chunk': ['cat', 'dog', 'hamster'], 'target': 'dog', 'target_idx': 1},
        {'player': 'P1', 'chunk': ['cat', 'dog', 'hamster', 'bird'], 'target': 'hamster', 'target_idx': 2}
    ])
    
    # For P2 (3 words: apple, banana, orange):
    # Chunk 1: [apple, banana] -> classify 'apple' (index 0)
    # Chunk 2: [apple, banana, orange] -> classify 'banana' (index 1)
    expected_chunks.extend([
        {'player': 'P2', 'chunk': ['apple', 'banana'], 'target': 'apple', 'target_idx': 0},
        {'player': 'P2', 'chunk': ['apple', 'banana', 'orange'], 'target': 'banana', 'target_idx': 1}
    ])
    
    print(f"\nExpected {len(expected_chunks)} chunks")
    
    # Validate chunks
    errors = []
    
    if len(chunks) != len(expected_chunks):
        errors.append(f"Chunk count mismatch: got {len(chunks)}, expected {len(expected_chunks)}")
    
    for i, (actual, expected) in enumerate(zip(chunks, expected_chunks)):
        print(f"\nChunk {i+1}:")
        print(f"  Player: {actual['player_id']} (expected: {expected['player']})")
        print(f"  Chunk words: {actual['chunk_words']} (expected: {expected['chunk']})")
        print(f"  Target word: '{actual['target_word']}' at index {actual['target_word_index']} (expected: '{expected['target']}' at {expected['target_idx']})")
        
        # Validate this chunk
        if actual['player_id'] != expected['player']:
            errors.append(f"Chunk {i+1}: Player mismatch")
        if actual['chunk_words'] != expected['chunk']:
            errors.append(f"Chunk {i+1}: Chunk words mismatch")
        if actual['target_word'] != expected['target']:
            errors.append(f"Chunk {i+1}: Target word mismatch")
        if actual['target_word_index'] != expected['target_idx']:
            errors.append(f"Chunk {i+1}: Target index mismatch")
    
    # Report results
    print(f"\nValidation Results:")
    print("-" * 20)
    
    if errors:
        print(f"FAILED: {len(errors)} errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("SUCCESS: All chunks validated correctly!")
        print(f"  - Created {len(chunks)} chunks as expected")
        print(f"  - All target words and indices match expected values")
        print(f"  - Progressive chunking logic working correctly")
        return True


def test_prompt_creation():
    """Test prompt creation."""
    print(f"\n\nTesting Prompt Creation")
    print("=" * 30)
    
    classifier = TestProgressiveClassifier()
    
    # Test example: chunk [cat, dog, hamster] to classify 'dog' at index 1
    chunk_words = ['cat', 'dog', 'hamster']
    category = 'animals'
    target_word_index = 1
    target_word = 'dog'
    
    prompt = classifier.create_progressive_classification_prompt(
        chunk_words, category, target_word_index, target_word
    )
    
    print(f"Test case:")
    print(f"  Chunk: {chunk_words}")
    print(f"  Target: '{target_word}' at index {target_word_index}")
    print(f"  Category: {category}")
    
    print(f"\nGenerated prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # Basic validation
    validation_checks = [
        (f"Word sequence from category \"{category}\"" in prompt, "Category mentioned"),
        (f"Word at position {target_word_index}: \"{target_word}\"" in prompt, "Target word identified"),
        ("JSON object" in prompt, "JSON response format specified"),
        ("switch" in prompt.lower(), "Switch classification mentioned"),
        ("reasoning" in prompt.lower(), "Reasoning requested")
    ]
    
    print(f"\nPrompt validation:")
    all_passed = True
    for check, description in validation_checks:
        status = "PASS" if check else "FAIL"
        print(f"  {status}: {description}")
        if not check:
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("Progressive Classification Implementation Test")
    print("=" * 50)
    
    try:
        # Test 1: Chunking logic
        chunking_success = test_progressive_chunking()
        
        # Test 2: Prompt creation
        prompt_success = test_prompt_creation()
        
        # Final results
        print(f"\n\nFinal Test Results")
        print("=" * 20)
        
        if chunking_success and prompt_success:
            print("ALL TESTS PASSED!")
            print("Progressive classification implementation is working correctly.")
            print("\nTo run with real data:")
            print("python run_llm_classification_progressive.py [config_name]")
            return True
        else:
            print("SOME TESTS FAILED!")
            print("Please check the implementation.")
            return False
            
    except Exception as e:
        print(f"ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)