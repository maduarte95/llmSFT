#!/usr/bin/env python3
"""
Retrieve results from an already completed Anthropic labeling batch.
Use this when a batch completed but result processing failed.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from anthropic_providers import AnthropicGroupLabeler
from config_system import ConfigManager


def load_environment():
    """Load Anthropic API key."""
    load_dotenv()
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in .env file")
        sys.exit(1)
    
    print("✅ Anthropic API key loaded successfully")
    return api_key


def retrieve_and_process_results(batch_id: str, data_file: str, config_name: str):
    """Retrieve results from completed labeling batch and process them."""
    
    # Load API key and create labeler
    api_key = load_environment()
    labeler = AnthropicGroupLabeler(api_key=api_key)
    
    # Load original data (should already have switchLLM column)
    print(f"📊 Loading original data from: {data_file}")
    try:
        original_data = pd.read_csv(data_file)
        print(f"✅ Data loaded: {original_data.shape}")
        
        # Verify switchLLM column exists
        if 'switchLLM' not in original_data.columns:
            print("❌ Error: Data file must contain 'switchLLM' column from switch classification")
            print("   Please run switch classification first or use a file with switch results")
            sys.exit(1)
            
        print("✅ Found switchLLM column - data is ready for labeling")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Check batch status first
    print(f"🔍 Checking batch status: {batch_id}")
    try:
        batch_status = labeler.provider.client.batches.retrieve(batch_id)
        print(f"Batch status: {batch_status.processing_status}")
        
        if batch_status.processing_status != "ended":
            print(f"❌ Batch is not completed yet. Current status: {batch_status.processing_status}")
            return
            
    except Exception as e:
        print(f"❌ Error checking batch status: {e}")
        sys.exit(1)
    
    # Retrieve results
    print(f"📥 Retrieving batch results...")
    try:
        # Get batch results
        batch_results = labeler.provider.process_batch_results(batch_id)
        print(f"✅ Retrieved {len(batch_results)} results")
        
        # Extract the actual config name from batch results
        actual_config_name = config_name  # Default fallback
        
        if batch_results and hasattr(batch_results[0], 'custom_id'):
            sample_id = batch_results[0].custom_id
            if sample_id.startswith('label_') and sample_id.count('_') >= 2:
                parts = sample_id.split('_')
                actual_config_name = '_'.join(parts[1:-1])  # Everything between label_ and final _number
                print(f"🔍 Detected actual config name from batch: {actual_config_name}")
        
        # Reconstruct the original requests metadata for labeling processing
        print("🔄 Reconstructing labeling request metadata...")
        mock_requests = []
        request_index = 0
        
        for (player_id, category), group in original_data.groupby(['playerID', 'category']):
            sequence_data = group.sort_values('word_index')
            
            if len(sequence_data) == 0:
                continue
            
            # Extract groups based on switchLLM column
            groups = []
            current_group = []
            
            for _, row in sequence_data.iterrows():
                if row['switchLLM'] == 1:  # Start of new group
                    if current_group:  # Save previous group
                        groups.append(current_group)
                    current_group = [row['text']]  # Start new group
                else:  # Continue current group
                    if current_group:  # Only add if we have a group started
                        current_group.append(row['text'])
            
            # Don't forget the last group
            if current_group:
                groups.append(current_group)
            
            if not groups:
                continue
                
            custom_id = f"label_{actual_config_name}_{request_index:04d}"
            
            mock_request = {
                "custom_id": custom_id,
                "metadata": {
                    "player_id": player_id,
                    "category": category,
                    "groups": groups,
                    "total_words": len(sequence_data),
                    "num_groups": len(groups),
                    "request_index": request_index,
                    "config_name": actual_config_name
                }
            }
            
            mock_requests.append(mock_request)
            request_index += 1
        
        print(f"✅ Reconstructed {len(mock_requests)} labeling request metadata entries")
        
        # Process results using the existing method
        print("🔄 Processing labeling results...")
        result_data = labeler.process_batch_results(batch_id, original_data, mock_requests)
        
        # Save results with same naming pattern as original script
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_config_name = actual_config_name.replace(' ', '_').replace('-', '_')
        output_file = f"data_with_labels_{safe_config_name}_retrieved_{timestamp}.csv"
        
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
                'data_shape': list(original_data.shape),
                'task_type': 'labeling'
            }
            
            config_manager.save_run_metadata(
                config=config,
                output_file=output_file,
                batch_id=batch_id,
                additional_info=additional_info
            )
        
        # Show summary
        print("\n📈 Labeling Results Summary:")
        total_rows = len(result_data)
        rows_with_labels = result_data['labelLLM'].notna().sum()
        unique_labels = result_data['labelLLM'].nunique() if rows_with_labels > 0 else 0
        
        print(f"Total rows: {total_rows:,}")
        print(f"Rows with LLM labels: {rows_with_labels:,}")
        print(f"Unique labels generated: {unique_labels}")
        
        if rows_with_labels > 0:
            print(f"Sample labels: {list(result_data['labelLLM'].dropna().unique()[:5])}")
        
        print(f"\n🎉 Labeling results successfully retrieved from batch {batch_id}!")
        
    except Exception as e:
        print(f"❌ Error processing results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function."""
    print("🔄 Anthropic Labeling Batch Results Retriever")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python retrieve_anthropic_labeling_batch_results.py <batch_id> [data_file] [config_name]")
        print("\nExample:")
        print("  python retrieve_anthropic_labeling_batch_results.py batch_abc123")
        print("  python retrieve_anthropic_labeling_batch_results.py batch_abc123 data_with_switches_claude_20250815.csv claude-sonnet-4-labels")
        print("\nNote: Data file should contain results from switch classification (with switchLLM column)")
        sys.exit(1)
    
    batch_id = sys.argv[1]
    
    # Try to find data file (should be switch classification results)
    if len(sys.argv) > 2:
        data_file = sys.argv[2]
    else:
        # Try to find switch classification results automatically
        current_dir = Path(".")
        patterns = [
            "*switch*llm*.csv",
            "*switches*.csv", 
            "data_with_switches_*.csv",
            "*.csv"
        ]
        
        data_file = None
        for pattern in patterns:
            files = list(current_dir.glob(pattern))
            if files:
                # Sort by modification time, newest first
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                data_file = str(files[0])
                break
        
        if not data_file:
            print("❌ Could not find switch classification results file.")
            print("   Please specify the data file with switchLLM column as second argument.")
            sys.exit(1)
    
    # Get config name
    if len(sys.argv) > 3:
        config_name = sys.argv[3]
    else:
        config_name = "retrieved_labeling_batch"
    
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