#!/usr/bin/env python3
"""
Debug script to see actual TogetherAI response for group labeling
"""

import os
import sys
import pandas as pd
import json
from dotenv import load_dotenv
from together_providers import TogetherAIGroupLabeler

def main():
    print("Debug TogetherAI Group Labeling Response")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("Error: TOGETHER_API_KEY not found")
        return
    
    print("API key loaded successfully")
    
    # Create labeler
    labeler = TogetherAIGroupLabeler(api_key=api_key)
    
    # Create test data similar to your actual data
    mock_data = pd.DataFrame({
        'playerID': ['test_player'] * 11,
        'category': ['animals'] * 11,
        'word_index': list(range(11)),
        'text': ['cat', 'dog', 'lamb', 'tiger', 'lion', 'bear', 'goat', 'sheep', 'bird', 'lizard', 'iguana'],
        'switchLLM': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # This should create multiple groups
    })
    
    print(f"Created test data with {len(mock_data)} rows")
    
    # Create groups from switches
    groups = labeler.create_groups_from_switches(mock_data)
    print(f"Created {len(groups)} groups:")
    for i, group in enumerate(groups):
        print(f"  Group {i+1}: {group['words']}")
    
    # Create prompt
    prompt = labeler.create_group_labeling_prompt(groups, 'animals')
    print(f"\nPrompt length: {len(prompt)} characters")
    print("Prompt preview:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    # Make API call
    print("\nMaking API request...")
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response_text = labeler.provider.make_single_request(messages, max_tokens=1000)
        
        print("\n" + "="*50)
        print("RAW RESPONSE:")
        print("="*50)
        print(response_text)
        print("="*50)
        
        # Try to parse it
        print("\nTrying to parse response...")
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        print("Cleaned text:")
        print(cleaned_text)
        
        try:
            response_data = json.loads(cleaned_text)
            print(f"\nSuccessfully parsed JSON!")
            print(f"Keys in response: {list(response_data.keys())}")
            
            if 'group_labels' in response_data:
                group_labels = response_data['group_labels']
                print(f"Found {len(group_labels)} group labels")
                for i, label_data in enumerate(group_labels):
                    print(f"  Label {i+1}: {label_data}")
            else:
                print("No 'group_labels' key found in response")
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            
    except Exception as e:
        print(f"API call failed: {e}")

if __name__ == "__main__":
    main()