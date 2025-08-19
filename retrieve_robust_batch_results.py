#!/usr/bin/env python3
"""
Retrieve results from an already completed Anthropic robust batch.
Use this when a robust batch completed but result processing failed.
Specifically designed for single-word classification batches.
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from anthropic_providers import AnthropicProvider


def load_environment():
    """Load Anthropic API key."""
    load_dotenv()
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        sys.exit(1)
    
    print("Anthropic API key loaded successfully")
    return api_key


def retrieve_and_process_robust_results(batch_id: str, data_file: str):
    """Retrieve results from completed robust batch and process them."""
    
    # Load API key and create provider
    api_key = load_environment()
    provider = AnthropicProvider(api_key=api_key)
    
    # Load original data
    print(f"Loading original data from: {data_file}")
    try:
        original_data = pd.read_csv(data_file)
        print(f"Data loaded: {original_data.shape}")
        print(f"   Unique players: {original_data['playerID'].nunique()}")
        print(f"   Categories: {list(original_data['category'].unique())}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Check batch status first
    print(f"Checking batch status: {batch_id}")
    try:
        batch_info = provider.client.messages.batches.retrieve(batch_id)
        print(f"Batch status: {batch_info.processing_status}")
        
        if batch_info.processing_status != "ended":
            print(f"Batch is not completed. Status: {batch_info.processing_status}")
            return
            
    except Exception as e:
        print(f"Error checking batch status: {e}")
        sys.exit(1)
    
    # Reconstruct metadata for robust requests (single-word classification)
    print("Reconstructing robust request metadata...")
    mock_requests = []
    request_index = 0
    
    # Group by player and category like the original robust script
    player_groups = original_data.groupby(['playerID', 'category'])
    
    for (player_id, category), group in player_groups:
        sequence_data = group.sort_values('word_index')
        words = sequence_data['text'].tolist()
        
        if len(words) < 2:  # Need at least 2 words, skip word 0
            continue
        
        # Create metadata for each word (except word 0 which is always a switch)
        for word_idx in range(1, len(words)):  # Start from 1, skip word 0
            target_word = words[word_idx]
            # Use the same custom ID format as the original robust script
            custom_id = f"single_word_claude-sonnet-4-switches_{request_index:06d}"
            
            mock_request = {
                "custom_id": custom_id,
                "metadata": {
                    "player_id": player_id,
                    "category": category,
                    "target_word_index": word_idx,
                    "target_word": target_word,
                    "full_word_sequence": words,
                    "sequence_length": len(words),
                    "request_id": request_index,
                    "config_name": "claude-sonnet-4-switches",
                    "config_version": "1.0"
                }
            }
            
            mock_requests.append(mock_request)
            request_index += 1
    
    print(f"Reconstructed {len(mock_requests)} robust request metadata entries")
    
    # Create metadata lookup
    metadata_lookup = {req["custom_id"]: req["metadata"] for req in mock_requests}
    
    # Start with copy of original data
    result_data = original_data.copy()
    
    # Initialize switchLLM and reasoning_switch columns if they don't exist
    if 'switchLLM' not in result_data.columns:
        result_data['switchLLM'] = None
    if 'reasoning_switch' not in result_data.columns:
        result_data['reasoning_switch'] = None
    
    # Set word 0 to always be a switch for all players (robust logic)
    for (player_id, category), group in result_data.groupby(['playerID', 'category']):
        first_word_mask = (result_data['playerID'] == player_id) & \
                         (result_data['category'] == category) & \
                         (result_data['word_index'] == 0)
        result_data.loc[first_word_mask, 'switchLLM'] = 1
    
    # Get results from Anthropic using the fixed logic
    print("Processing batch results...")
    batch_results = provider.process_batch_results(batch_id)
    
    successful_results = 0
    failed_results = 0
    
    # Process each result using the same logic as the fixed robust script
    for result in batch_results:
        # Handle Anthropic MessageBatchIndividualResponse object
        if hasattr(result, 'custom_id'):
            custom_id = result.custom_id
            if result.result.type == "succeeded":
                response_text = result.result.message.content[0].text
                success = True
            else:
                success = False
        else:
            print("Unexpected result format - expected Anthropic format")
            failed_results += 1
            continue
        
        if success:
            try:
                # Clean and parse JSON response
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                response_data = json.loads(response_text)
                
                # Get metadata for this request
                metadata = metadata_lookup.get(custom_id, {})
                player_id = metadata.get('player_id')
                category = metadata.get('category')
                target_word_index = metadata.get('target_word_index')
                target_word = metadata.get('target_word')
                
                if not player_id or target_word_index is None:
                    print(f"  Missing metadata for {custom_id}")
                    failed_results += 1
                    continue
                
                # Extract single word classification
                switch_value = response_data.get('switch')
                reasoning_switch = response_data.get('reasoning_switch', '')
                
                # Validate response
                if switch_value not in [0, 1]:
                    print(f"  Invalid switch value for {player_id} word {target_word_index}: {switch_value}")
                    failed_results += 1
                    continue
                
                # Apply classification to the specific word
                word_mask = (result_data['playerID'] == player_id) & \
                           (result_data['category'] == category) & \
                           (result_data['word_index'] == target_word_index)
                
                if not word_mask.any():
                    print(f"  No matching row for {player_id} word {target_word_index}")
                    failed_results += 1
                    continue
                
                result_data.loc[word_mask, 'switchLLM'] = switch_value
                result_data.loc[word_mask, 'reasoning_switch'] = reasoning_switch
                
                successful_results += 1
                if successful_results <= 10:  # Show first few for debugging
                    print(f"  {player_id} word {target_word_index} ('{target_word}'): {switch_value}")
            
            except Exception as e:
                print(f"  Error processing result for {custom_id}: {e}")
                failed_results += 1
        else:
            print(f"  Anthropic result failed for {custom_id}: {result.result.type}")
            failed_results += 1
    
    print(f"\nRobust classification recovery summary:")
    print(f"- Successful: {successful_results}")
    print(f"- Failed: {failed_results}")
    print(f"- Total rows: {len(result_data)}")
    print(f"- Rows with LLM classifications: {result_data['switchLLM'].notna().sum()}")
    
    if result_data['switchLLM'].notna().sum() > 0:
        print(f"- LLM switch rate: {result_data['switchLLM'].mean():.3f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"data_with_switches_robust_recovered_{timestamp}.csv"
    
    result_data.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return result_data, output_file


def main():
    """Main execution function."""
    if len(sys.argv) != 3:
        print("Usage: python retrieve_robust_batch_results.py <batch_id> <data_file>")
        print("Example: python retrieve_robust_batch_results.py msgbatch_013CyQee7Sh8Cq5qN77dNyDk merged_trials_data_unfiltered.csv")
        sys.exit(1)
    
    batch_id = sys.argv[1]
    data_file = sys.argv[2]
    
    print("Robust Batch Results Recovery")
    print(f"Batch ID: {batch_id}")
    print(f"Data file: {data_file}")
    print("=" * 60)
    
    try:
        result_data, output_file = retrieve_and_process_robust_results(batch_id, data_file)
        
        print("\n" + "=" * 60)
        print("Recovery completed successfully!")
        print(f"Output file: {output_file}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during recovery: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()