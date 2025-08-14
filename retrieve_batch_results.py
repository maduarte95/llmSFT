#!/usr/bin/env python3
"""
Retrieve results from an already completed TogetherAI batch.
Use this when a batch completed but result processing failed.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from together_providers import TogetherAISwitchClassifier
from config_system import ConfigManager


def load_environment():
    """Load TogetherAI API key."""
    load_dotenv()
    
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("❌ Error: TOGETHER_API_KEY not found in .env file")
        sys.exit(1)
    
    print("✅ TogetherAI API key loaded successfully")
    return api_key


def retrieve_and_process_results(batch_id: str, data_file: str, config_name: str):
    """Retrieve results from completed batch and process them."""
    
    # Load API key and create classifier
    api_key = load_environment()
    classifier = TogetherAISwitchClassifier(api_key=api_key)
    
    # Load original data
    print(f"📊 Loading original data from: {data_file}")
    try:
        original_data = pd.read_csv(data_file)
        print(f"✅ Data loaded: {original_data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Check batch status first
    print(f"🔍 Checking batch status: {batch_id}")
    try:
        batch_info = classifier.provider.client.batches.get_batch(batch_id)
        print(f"Batch status: {batch_info.status}")
        
        if "completed" not in str(batch_info.status).lower():
            print(f"❌ Batch is not completed yet. Current status: {batch_info.status}")
            return
            
    except Exception as e:
        print(f"❌ Error checking batch status: {e}")
        sys.exit(1)
    
    # Retrieve results
    print(f"📥 Retrieving batch results...")
    try:
        # Get batch results
        batch_results = classifier.provider.process_batch_results(batch_id)
        print(f"✅ Retrieved {len(batch_results)} results")
        
        # Reconstruct the original requests metadata for processing
        # We need this to map results back to the original data
        print("🔄 Reconstructing request metadata...")
        
        # Prepare mock requests metadata (we need this for result processing)
        mock_requests = []
        request_index = 0
        
        for (player_id, category), group in original_data.groupby(['playerID', 'category']):
            sequence_data = group.sort_values('word_index')
            complete_word_sequence = sequence_data['text'].tolist()
            
            if not complete_word_sequence:
                continue
            
            custom_id = f"switch_{config_name}_{request_index:04d}"
            
            mock_request = {
                "custom_id": custom_id,
                "metadata": {
                    "player_id": player_id,
                    "category": category,
                    "total_words": len(sequence_data),
                    "sequence_length": len(complete_word_sequence),
                    "word_sequence": complete_word_sequence,
                    "request_index": request_index,
                    "config_name": config_name
                }
            }
            
            mock_requests.append(mock_request)
            request_index += 1
        
        print(f"✅ Reconstructed {len(mock_requests)} request metadata entries")
        
        # Process results using the existing method
        print("🔄 Processing results...")
        result_data = classifier.process_batch_results(batch_id, original_data, mock_requests)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"data_with_switches_{config_name}_retrieved_{timestamp}.csv"
        
        result_data.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Save metadata
        config_manager = ConfigManager()
        config = config_manager.get_config(config_name)
        if config:
            additional_info = {
                'retrieved_batch': True,
                'original_batch_id': batch_id,
                'num_results': len(batch_results),
                'num_players': original_data['playerID'].nunique(),
                'categories': list(original_data['category'].unique()),
                'data_shape': list(original_data.shape)
            }
            
            config_manager.save_run_metadata(
                config=config,
                output_file=output_file,
                batch_id=batch_id,
                additional_info=additional_info
            )
        
        # Show summary
        print("\n📈 Results Summary:")
        total_rows = len(result_data)
        rows_with_llm = result_data['switchLLM'].notna().sum()
        llm_switch_rate = result_data['switchLLM'].mean() if rows_with_llm > 0 else 0
        
        print(f"Total rows: {total_rows:,}")
        print(f"Rows with LLM classifications: {rows_with_llm:,}")
        print(f"LLM switch rate: {llm_switch_rate:.3f}")
        
        print(f"\n🎉 Results successfully retrieved from batch {batch_id}!")
        
    except Exception as e:
        print(f"❌ Error processing results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function."""
    print("🔄 TogetherAI Batch Results Retriever")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python retrieve_batch_results.py <batch_id> [data_file] [config_name]")
        print("\nExample:")
        print("  python retrieve_batch_results.py b592a21f-46eb-43c0-905e-78c8b1f7edbe")
        print("  python retrieve_batch_results.py b592a21f-46eb... merged_trials_data_unfiltered.csv llama-3.1-70b-switches")
        sys.exit(1)
    
    batch_id = sys.argv[1]
    
    # Try to find data file
    if len(sys.argv) > 2:
        data_file = sys.argv[2]
    else:
        # Try to find data file automatically
        current_dir = Path(".")
        patterns = [
            "filtered_data_for_analysis.csv",
            "*filtered*.csv",
            "merged_trials_data_unfiltered.csv",
            "*.csv"
        ]
        
        data_file = None
        for pattern in patterns:
            files = list(current_dir.glob(pattern))
            if files:
                data_file = str(files[0])
                break
        
        if not data_file:
            print("❌ Could not find data file. Please specify it as second argument.")
            sys.exit(1)
    
    # Get config name
    if len(sys.argv) > 3:
        config_name = sys.argv[3]
    else:
        config_name = "retrieved_batch"
    
    print(f"Batch ID: {batch_id}")
    print(f"Data file: {data_file}")
    print(f"Config name: {config_name}")
    print()
    
    retrieve_and_process_results(batch_id, data_file, config_name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)