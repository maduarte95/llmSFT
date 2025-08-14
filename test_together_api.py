#!/usr/bin/env python3
"""
TogetherAI API Test Script
Quick test to verify TogetherAI integration is working without running full pipeline.
"""

import os
import sys
import pandas as pd
import json
from dotenv import load_dotenv
from together_providers import TogetherAISwitchClassifier, TogetherAIGroupLabeler


def load_environment():
    """Load TogetherAI API key."""
    load_dotenv()
    
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("❌ Error: TOGETHER_API_KEY not found in .env file")
        print("Please ensure your .env file contains:")
        print("TOGETHER_API_KEY=your_api_key_here")
        sys.exit(1)
    
    print("✅ TogetherAI API key loaded successfully")
    return api_key


def test_single_request():
    """Test a single API request to TogetherAI."""
    print("\n🔍 Test 1: Single API Request")
    print("-" * 40)
    
    try:
        api_key = load_environment()
        classifier = TogetherAISwitchClassifier(api_key=api_key)
        
        # Test with a simple word sequence
        test_words = ["dog", "cat", "lion", "tiger", "apple"]
        test_category = "animals"
        
        print(f"Testing with words: {test_words}")
        print(f"Category: {test_category}")
        
        # Create prompt
        prompt = classifier.create_switch_identification_prompt(test_words, test_category)
        print(f"Prompt length: {len(prompt)} characters")
        
        # Make single request
        print("Making API request...")
        messages = [{"role": "user", "content": prompt}]
        response_text = classifier.provider.make_single_request(messages, max_tokens=1500)
        
        print(f"✅ API request successful!")
        print(f"Response length: {len(response_text)} characters")
        
        # Try to parse JSON response
        try:
            # Clean response text
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            response_data = json.loads(cleaned_text)
            
            if 'word_classifications' in response_data:
                classifications = response_data['word_classifications']
                print(f"✅ Valid JSON with {len(classifications)} word classifications")
                
                # Show results
                for item in classifications:
                    idx = item.get('index', 'N/A')
                    word = item.get('word', 'N/A')
                    switch = item.get('switch', 'N/A')
                    print(f"  {idx}. {word} → {switch}")
                
                # Calculate switch rate
                switches = [item.get('switch', 0) for item in classifications if 'switch' in item]
                if switches:
                    switch_rate = sum(switches) / len(switches)
                    print(f"Switch rate: {switch_rate:.3f}")
            else:
                print(f"❌ Missing 'word_classifications' key")
                print(f"Available keys: {list(response_data.keys())}")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response preview: {response_text[:300]}...")
            return False
            
    except Exception as e:
        print(f"❌ API request failed: {e}")
        return False
    
    return True


def test_dry_run():
    """Test the dry run functionality with mock data."""
    print("\n🧪 Test 2: Dry Run with Mock Data")
    print("-" * 40)
    
    try:
        api_key = load_environment()
        classifier = TogetherAISwitchClassifier(api_key=api_key)
        
        # Create mock data
        mock_data = pd.DataFrame({
            'playerID': ['test_player_1', 'test_player_1', 'test_player_1', 'test_player_1'],
            'category': ['animals', 'animals', 'animals', 'animals'],
            'word_index': [0, 1, 2, 3],
            'text': ['dog', 'cat', 'lion', 'tiger']
        })
        
        print(f"Mock data created: {len(mock_data)} rows")
        print("Sample data:")
        print(mock_data)
        
        # Run dry run test
        print("\nRunning dry run...")
        success = classifier.dry_run_test(mock_data, num_tests=1)
        
        if success:
            print("✅ Dry run successful!")
        else:
            print("❌ Dry run failed!")
            return False
            
    except Exception as e:
        print(f"❌ Dry run test failed: {e}")
        return False
    
    return True


def test_group_labeling():
    """Test group labeling with mock data."""
    print("\n🏷️ Test 3: Group Labeling Test")
    print("-" * 40)
    
    try:
        api_key = load_environment()
        labeler = TogetherAIGroupLabeler(api_key=api_key)
        
        # Create mock data with switchLLM column
        mock_data = pd.DataFrame({
            'playerID': ['test_player_1', 'test_player_1', 'test_player_1', 'test_player_1'],
            'category': ['animals', 'animals', 'animals', 'animals'],
            'word_index': [0, 1, 2, 3],
            'text': ['dog', 'cat', 'lion', 'tiger'],
            'switchLLM': [1, 0, 1, 0]  # Two groups: [dog, cat] and [lion, tiger]
        })
        
        print(f"Mock data with switchLLM: {len(mock_data)} rows")
        print("Sample data:")
        print(mock_data)
        
        # Test group creation
        sequence_data = mock_data.sort_values('word_index')
        groups = labeler.create_groups_from_switches(sequence_data)
        
        print(f"\nCreated {len(groups)} groups:")
        for i, group in enumerate(groups):
            print(f"  Group {i+1}: {group['words']} (indices {group['word_indices']})")
        
        # Test prompt creation
        prompt = labeler.create_group_labeling_prompt(groups, 'animals')
        print(f"\nPrompt created (length: {len(prompt)} chars)")
        
        # Test single API call
        print("Making group labeling API request...")
        messages = [{"role": "user", "content": prompt}]
        response_text = labeler.provider.make_single_request(messages, max_tokens=1000)
        
        print(f"✅ Group labeling API request successful!")
        
        # Try to parse response
        try:
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            response_data = json.loads(cleaned_text)
            
            if 'group_labels' in response_data:
                group_labels = response_data['group_labels']
                print(f"✅ Valid JSON with {len(group_labels)} group labels")
                
                for label_data in group_labels:
                    group_num = label_data.get('group_number', 'N/A')
                    words = label_data.get('words', [])
                    label = label_data.get('label', 'N/A')
                    print(f"  Group {group_num}: {words} → '{label}'")
            else:
                print(f"❌ Missing 'group_labels' key")
                print(f"Available keys: {list(response_data.keys())}")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response preview: {response_text[:300]}...")
            return False
            
    except Exception as e:
        print(f"❌ Group labeling test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 TogetherAI API Integration Tests")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: Single API request
    test_results.append(test_single_request())
    
    # Test 2: Dry run with mock data
    test_results.append(test_dry_run())
    
    # Test 3: Group labeling
    test_results.append(test_group_labeling())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    test_names = [
        "Single API Request",
        "Dry Run Test", 
        "Group Labeling Test"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests passed! TogetherAI integration is working correctly.")
        print("You can now use the unified scripts with TogetherAI.")
    else:
        print("⚠️ Some tests failed. Please check your API key and connection.")
        print("Make sure TOGETHER_API_KEY is set in your .env file.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)