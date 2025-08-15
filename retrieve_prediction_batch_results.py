#!/usr/bin/env python3
"""
Retrieve results from an already completed TogetherAI prediction batch.
Use this when a batch completed but result processing failed.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from together_providers import TogetherAISwitchPredictor
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
    """Retrieve results from completed prediction batch and process them."""
    
    # Load API key and create predictor
    api_key = load_environment()
    predictor = TogetherAISwitchPredictor(api_key=api_key)
    
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
        batch_info = predictor.provider.client.batches.get_batch(batch_id)
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
        batch_results = predictor.provider.process_batch_results(batch_id)
        print(f"✅ Retrieved {len(batch_results)} results")
        
        # Extract the actual config name from batch results
        actual_custom_ids = [result.get('custom_id') for result in batch_results[:5]]
        actual_config_name = config_name  # Default fallback
        
        if actual_custom_ids and actual_custom_ids[0]:
            # Extract config name from actual custom_id format: pred_{config_name}_{number}
            sample_id = actual_custom_ids[0]
            if sample_id.startswith('pred_') and sample_id.count('_') >= 2:
                parts = sample_id.split('_')
                actual_config_name = '_'.join(parts[1:-1])  # Everything between pred_ and final _number
                print(f"🔍 Detected actual config name from batch: {actual_config_name}")
        
        # Regenerate mock requests with correct config name
        print("🔄 Regenerating request metadata with correct config name...")
        mock_requests = []
        chunk_index = 0
        
        for (player_id, category), group in original_data.groupby(['playerID', 'category']):
            sequence_data = group.sort_values('word_index')
            words = sequence_data['text'].tolist()
            
            if len(words) < 2:  # Need at least 2 words to predict next switch
                continue
            
            # Create incremental chunks exactly as in base_classifiers.py: [0], [0,1], [0,1,2], etc.
            for end_idx in range(len(words)):
                chunk_words = words[:end_idx + 1]
                
                # For prediction, we need to know if there's a next word
                has_next_word = end_idx < len(words) - 1
                next_word = words[end_idx + 1] if has_next_word else None
                
                # Only include chunks that have a next word (prediction chunks)
                if has_next_word:
                    custom_id = f"pred_{actual_config_name}_{chunk_index:06d}"
                    
                    mock_request = {
                        "custom_id": custom_id,
                        "metadata": {
                            "chunk_id": chunk_index,
                            "player_id": player_id,
                            "category": category,
                            "chunk_words": chunk_words,
                            "chunk_end_index": end_idx,
                            "next_word": next_word,
                            "sequence_length": len(words)
                        }
                    }
                    
                    mock_requests.append(mock_request)
                
                chunk_index += 1
        
        print(f"✅ Reconstructed {len(mock_requests)} prediction request metadata entries")
        
        # Process results using the existing method
        print("🔄 Processing prediction results...")
        result_data = predictor.process_batch_results(batch_id, original_data, mock_requests)
        
        # Save results with same naming pattern as original script
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_config_name = actual_config_name.replace(' ', '_').replace('-', '_')
        output_file = f"data_with_predictions_{safe_config_name}_retrieved_{timestamp}.csv"
        
        result_data.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Create word-level prediction DataFrame
        print("🔄 Creating word-level prediction DataFrame...")
        word_level_data = original_data.copy()
        word_level_data['predicted_switch_llm'] = None
        word_level_data['prediction_confidence_llm'] = None
        word_level_data['prediction_reasoning_llm'] = None
        
        # Map chunk predictions to individual words
        successful_mappings = 0
        
        for _, chunk_row in result_data.iterrows():
            player_id = chunk_row['player_id']
            category = chunk_row['category']
            chunk_end_index = chunk_row['chunk_end_index']
            
            # The prediction is for the word AFTER the chunk_end_index
            predicted_word_index = chunk_end_index + 1
            
            # Find the corresponding row in original data
            mask = (
                (word_level_data['playerID'] == player_id) & 
                (word_level_data['category'] == category) & 
                (word_level_data['word_index'] == predicted_word_index)
            )
            
            matching_rows = word_level_data[mask]
            if len(matching_rows) == 1:
                # Apply the prediction to this word
                word_level_data.loc[mask, 'predicted_switch_llm'] = chunk_row['prediction_llm']
                word_level_data.loc[mask, 'prediction_confidence_llm'] = chunk_row['confidence_llm']
                word_level_data.loc[mask, 'prediction_reasoning_llm'] = chunk_row['reasoning_llm']
                successful_mappings += 1
            elif len(matching_rows) > 1:
                print(f"⚠️  Warning: Multiple matches for {player_id}, {category}, word_index {predicted_word_index}")
            # If no matches, that's expected for the last word in each sequence
        
        # Save word-level results
        word_output_file = f"data_with_word_predictions_{safe_config_name}_retrieved_{timestamp}.csv"
        word_level_data.to_csv(word_output_file, index=False)
        
        print(f"✅ Word-level predictions saved to: {word_output_file}")
        print(f"✅ Successfully mapped {successful_mappings} chunk predictions to words")
        
        # Show word-level summary
        total_words = len(word_level_data)
        words_with_predictions = word_level_data['predicted_switch_llm'].notna().sum()
        if words_with_predictions > 0:
            avg_confidence = word_level_data['prediction_confidence_llm'].mean()
            switch_rate = word_level_data['predicted_switch_llm'].mean()
            print(f"📊 Word-level summary:")
            print(f"   Total words: {total_words:,}")
            print(f"   Words with predictions: {words_with_predictions:,}")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Predicted switch rate: {switch_rate:.3f}")
        
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
                'task_type': 'prediction'
            }
            
            config_manager.save_run_metadata(
                config=config,
                output_file=output_file,
                batch_id=batch_id,
                additional_info=additional_info
            )
        
        # Show summary
        print("\n📈 Prediction Results Summary:")
        total_rows = len(result_data)
        
        if 'prediction_llm' in result_data.columns:
            rows_with_llm = result_data['prediction_llm'].notna().sum()
            llm_prediction_rate = result_data['prediction_llm'].mean() if rows_with_llm > 0 else 0
            print(f"Total prediction chunks: {total_rows:,}")
            print(f"Chunks with LLM predictions: {rows_with_llm:,}")
            print(f"LLM switch prediction rate: {llm_prediction_rate:.3f}")
        else:
            print(f"Total rows: {total_rows:,}")
            print("Note: Check column names in output file for prediction results")
        
        print(f"\n🎉 Prediction results successfully retrieved from batch {batch_id}!")
        
    except Exception as e:
        print(f"❌ Error processing results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function."""
    print("🔄 TogetherAI Prediction Batch Results Retriever")
    print("=" * 55)
    
    if len(sys.argv) < 2:
        print("Usage: python retrieve_prediction_batch_results.py <batch_id> [data_file] [config_name]")
        print("\nExample:")
        print("  python retrieve_prediction_batch_results.py 7a633bc9-652b-4f82-b3de-76bad72fc1bd")
        print("  python retrieve_prediction_batch_results.py 7a633bc9-652b... filtered_data_for_analysis.csv llama-3.1-70b-prediction")
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
        config_name = "retrieved_prediction_batch"
    
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