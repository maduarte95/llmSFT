#!/usr/bin/env python3
"""
Batch Response Inspector
Inspects the raw response from Anthropic batch API to debug issues.
"""

import os
import sys
import pandas as pd
import anthropic
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in .env file")
        sys.exit(1)
    
    print("✅ API key loaded successfully")
    return api_key

def inspect_batch_results(batch_id: str, api_key: str):
    """
    Inspect detailed results from a batch.
    
    Args:
        batch_id: The batch ID to inspect
        api_key: Anthropic API key
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    print(f"🔍 Inspecting batch: {batch_id}")
    print("=" * 60)
    
    try:
        # Get batch info
        batch = client.messages.batches.retrieve(batch_id)
        print(f"Batch Status: {batch.processing_status}")
        print(f"Request Counts: {batch.request_counts}")
        print()
        
        # Get all results
        results = list(client.messages.batches.results(batch_id))
        print(f"Total Results: {len(results)}")
        print()
        
        for i, result in enumerate(results):
            print(f"Result {i+1}/{len(results)}:")
            print(f"  Custom ID: {result.custom_id}")
            print(f"  Result Type: {result.result.type}")
            
            if result.result.type == "succeeded":
                try:
                    # Get the response text
                    response_text = result.result.message.content[0].text
                    print(f"  Response Length: {len(response_text)} characters")
                    
                    # Try to parse as JSON
                    try:
                        # Clean response text
                        cleaned_text = response_text.strip()
                        if cleaned_text.startswith("```json"):
                            cleaned_text = cleaned_text[7:]
                        if cleaned_text.endswith("```"):
                            cleaned_text = cleaned_text[:-3]
                        cleaned_text = cleaned_text.strip()
                        
                        response_data = json.loads(cleaned_text)
                        
                        if "switches" in response_data:
                            switches = response_data["switches"]
                            print(f"  Switches Array Length: {len(switches)}")
                            print(f"  Switches Sample: {switches[:10]}...")
                            print(f"  Switch Rate: {sum(switches) / len(switches):.3f}")
                        else:
                            print(f"  JSON Keys: {list(response_data.keys())}")
                        
                    except json.JSONDecodeError as e:
                        print(f"  ❌ JSON Parse Error: {e}")
                        print(f"  Raw Response Preview: {response_text[:200]}...")
                    
                    # Show full response if requested
                    show_full = input(f"  Show full response for result {i+1}? (y/N): ").strip().lower()
                    if show_full == 'y':
                        print(f"  Full Response:")
                        print("-" * 40)
                        print(response_text)
                        print("-" * 40)
                    
                except Exception as e:
                    print(f"  ❌ Error processing response: {e}")
            
            elif result.result.type == "errored":
                print(f"  ❌ Error: {result.result.error}")
            
            print()
    
    except Exception as e:
        print(f"❌ Error inspecting batch: {e}")
        return

def inspect_sequence_data(player_id: str, category: str):
    """
    Inspect the actual sequence data for a specific player.
    
    Args:
        player_id: Player ID to inspect
        category: Category to inspect
    """
    print(f"🔍 Inspecting sequence data for {player_id} ({category})")
    print("=" * 60)
    
    # Find data file
    current_dir = Path(".")
    data_files = list(current_dir.glob("filtered_data_for_analysis.csv"))
    if not data_files:
        data_files = list(current_dir.glob("*llm*.csv"))
    
    if not data_files:
        print("❌ No data file found")
        return
    
    data_file = data_files[0]
    print(f"📁 Loading data from: {data_file}")
    
    try:
        data = pd.read_csv(data_file)
        
        # Filter for specific player
        player_data = data[(data['playerID'] == player_id) & (data['category'] == category)]
        
        if len(player_data) == 0:
            print(f"❌ No data found for {player_id} ({category})")
            return
        
        # Sort by word_index
        player_data = player_data.sort_values('word_index')
        
        print(f"📊 Sequence Analysis:")
        print(f"  Total words: {len(player_data)}")
        print(f"  Word indices: {player_data['word_index'].min()} to {player_data['word_index'].max()}")
        print(f"  Category: {category}")
        print()
        
        print(f"📝 Complete word sequence:")
        for idx, row in player_data.iterrows():
            word_idx = row['word_index']
            word = row['text']
            switch = row.get('switch', 'N/A')
            switch_llm = row.get('switchLLM', 'N/A')
            print(f"  {word_idx:2d}. {word:<20} (human: {switch}, LLM: {switch_llm})")
        
        # Look for potential issues
        print(f"\n🔍 Potential Issues:")
        
        # Check for duplicate words
        word_counts = player_data['text'].value_counts()
        duplicates = word_counts[word_counts > 1]
        if len(duplicates) > 0:
            print(f"  ⚠️ Duplicate words found:")
            for word, count in duplicates.items():
                print(f"    '{word}': appears {count} times")
        
        # Check for special characters or formatting
        special_words = []
        for word in player_data['text']:
            if any(char in word for char in ['"', "'", '\n', '\r', '\t']) or len(word.strip()) != len(word):
                special_words.append(word)
        
        if special_words:
            print(f"  ⚠️ Words with special characters or whitespace:")
            for word in special_words:
                print(f"    '{repr(word)}'")
        
        # Check for very long words
        long_words = player_data[player_data['text'].str.len() > 20]
        if len(long_words) > 0:
            print(f"  ⚠️ Very long words (>20 chars):")
            for _, row in long_words.iterrows():
                print(f"    {row['word_index']}. '{row['text']}' ({len(row['text'])} chars)")
        
        # Check for empty or null words
        empty_words = player_data[player_data['text'].str.strip().eq('') | player_data['text'].isna()]
        if len(empty_words) > 0:
            print(f"  ⚠️ Empty or null words:")
            for _, row in empty_words.iterrows():
                print(f"    {row['word_index']}. '{repr(row['text'])}'")
        
        if len(duplicates) == 0 and len(special_words) == 0 and len(long_words) == 0 and len(empty_words) == 0:
            print(f"  ✅ No obvious issues found in sequence data")
        
    except Exception as e:
        print(f"❌ Error inspecting sequence: {e}")

def main():
    """Main execution function."""
    print("🔍 Batch Response Inspector")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load environment
    api_key = load_environment()
    
    print("\n📋 What would you like to inspect?")
    print("1. Inspect specific batch results")
    print("2. Inspect problematic sequence data")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        print("\n🔍 Batch Inspection")
        batch_id = input("Enter batch ID (e.g., msgbatch_01BEDLMnEMMC3KSQY8ZCwLR3): ").strip()
        if batch_id:
            inspect_batch_results(batch_id, api_key)
    
    if choice in ["2", "3"]:
        print("\n🔍 Sequence Data Inspection")
        player_id = input("Enter player ID (default: 01JYPG7PZJNHF56HQPDWJP54A1): ").strip()
        if not player_id:
            player_id = "01JYPG7PZJNHF56HQPDWJP54A1"
        
        category = input("Enter category (default: supermarket items): ").strip()
        if not category:
            category = "supermarket items"
        
        inspect_sequence_data(player_id, category)

if __name__ == "__main__":
    main()