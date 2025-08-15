#!/usr/bin/env python3
"""
Configuration-Driven LLM Switch Prediction Script
Uses YAML configuration files for model selection and parameter control.
Supports custom prompt templates and reproducible experiments.
"""

import os
import sys
import pandas as pd
import time
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Import configuration system
from config_system import ConfigManager, ModelConfig

# Import provider-specific predictors
from anthropic_providers import AnthropicSwitchPredictor, AnthropicProvider
from together_providers import TogetherAISwitchPredictor, TogetherAIProvider


def load_environment(provider: str) -> str:
    """Load environment variables from .env file."""
    load_dotenv()
    
    if provider == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ Error: ANTHROPIC_API_KEY not found in .env file")
            sys.exit(1)
    elif provider == "together":
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            print("❌ Error: TOGETHER_API_KEY not found in .env file")
            sys.exit(1)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    print(f"✅ {provider.title()} API key loaded successfully")
    return api_key


def find_data_file():
    """Find the filtered data CSV file."""
    current_dir = Path(".")
    
    patterns = [
        "filtered_data_for_analysis.csv",
        "*filtered*analysis*.csv",
        "*filtered*.csv",
        "data_with_llm_switches*.csv"
    ]
    
    for pattern in patterns:
        files = list(current_dir.glob(pattern))
        if files:
            file_path = files[0]
            print(f"✅ Found data file: {file_path}")
            return str(file_path)
    
    print("❌ Error: Could not find data file")
    csv_files = list(current_dir.glob("*.csv"))
    if csv_files:
        print("\nAvailable CSV files:")
        for f in csv_files:
            print(f"  - {f}")
        
        while True:
            filename = input("\nPlease enter the filename to use: ").strip()
            if Path(filename).exists():
                return filename
            else:
                print(f"File '{filename}' not found. Please try again.")
    else:
        print("No CSV files found in current directory.")
        sys.exit(1)


class ConfigurablePredictor:
    """Wrapper that applies config parameters to existing predictors."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        
        # Create base predictor with config
        if config.provider == "anthropic":
            self.predictor = AnthropicSwitchPredictor(
                api_key=api_key, 
                model=config.model,
                config=config
            )
        elif config.provider == "together":
            self.predictor = TogetherAISwitchPredictor(
                api_key=api_key,
                model=config.model,
                config=config
            )
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        # Apply custom parameters
        self.predictor.provider.default_params = config.parameters
    
    def prepare_batch_requests(self, data: pd.DataFrame):
        """Prepare batch requests with custom parameters from config."""
        chunks = self.predictor.create_chunk_data(data)
        
        # Only process chunks that have a next word to predict
        prediction_chunks = [chunk for chunk in chunks if chunk['has_next_word']]
        
        requests = []
        
        for chunk in prediction_chunks:
            # Use configurable prompt (will use custom template if available)
            prompt = self.predictor.create_switch_prediction_prompt(
                chunk['chunk_words'], 
                chunk['category']
            )
            custom_id = f"pred_{self.config.name}_{chunk['chunk_id']:06d}"
            
            # Apply custom parameters from config
            request_params = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Add all parameters from config
            request_params.update(self.config.parameters)
            
            request = {
                "custom_id": custom_id,
                "params": request_params,
                "metadata": {
                    "chunk_id": chunk['chunk_id'],
                    "player_id": chunk['player_id'],
                    "category": chunk['category'],
                    "chunk_words": chunk['chunk_words'],
                    "chunk_end_index": chunk['chunk_end_index'],
                    "next_word": chunk['next_word'],
                    "sequence_length": chunk['sequence_length'],
                    "config_name": self.config.name,
                    "config_version": self.config.version
                }
            }
            
            requests.append(request)
        
        print(f"Prepared {len(requests)} batch requests using config: {self.config.name}")
        print(f"Total chunks: {len(chunks)}, Prediction chunks: {len(prediction_chunks)}")
        return requests
    
    def run_prediction(self, data: pd.DataFrame):
        """Run prediction with config parameters and metadata tracking."""
        print(f"Starting prediction with config: {self.config.name}")
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model}")
        print(f"Parameters: {self.config.parameters}")
        
        if self.config.prompt_template:
            print("✅ Using custom prompt template from config")
        else:
            print("📝 Using default prompt template")
        
        # Prepare requests with config parameters
        batch_requests = self.prepare_batch_requests(data)
        
        if len(batch_requests) == 0:
            print("No valid requests to process. Returning empty DataFrame.")
            return pd.DataFrame(), None, {}
        
        # Run dry run test
        if not self.predictor.dry_run_test(data, num_tests=min(2, len(batch_requests))):
            print("❌ Dry run failed. Please check your configuration and data.")
            return pd.DataFrame(), None, {}
        
        # Create batch
        batch_id = self.predictor.provider.create_batch(batch_requests)
        
        # Monitor batch completion
        final_batch = self.predictor.provider.monitor_batch(batch_id)
        
        # Process results with metadata
        result_data = self.predictor.process_batch_results(batch_id, data, batch_requests)
        
        # Add config metadata info
        additional_info = {
            'num_requests': len(batch_requests),
            'num_players': data['playerID'].nunique(),
            'categories': list(data['category'].unique()),
            'data_shape': list(data.shape),
            'custom_prompt_used': self.config.prompt_template is not None
        }
        
        return result_data, batch_id, additional_info


def print_separator(char="=", length=60):
    """Print a separator line."""
    print(char * length)


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def main():
    """Main execution function."""
    print_separator()
    print("🔮 Configuration-Driven LLM Switch Prediction")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    # Step 1: Load configurations
    print("\n🔧 Step 1: Loading configurations...")
    config_manager = ConfigManager()
    config_manager.print_config_summary()
    
    # Step 2: Select configuration
    print("\n🤖 Step 2: Configuration selection...")
    
    # Option to specify config via command line
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        config = config_manager.get_config(config_name)
        if not config:
            print(f"❌ Configuration '{config_name}' not found")
            sys.exit(1)
        print(f"✅ Using config from command line: {config_name}")
    else:
        # Interactive selection
        config = config_manager.interactive_config_selector("prediction")
        if not config:
            print("❌ No configuration selected")
            sys.exit(1)
    
    # Validate configuration
    issues = config_manager.validate_config(config)
    if issues:
        print(f"❌ Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    print(f"✅ Selected configuration: {config.name}")
    print(f"   Provider: {config.provider}")
    print(f"   Model: {config.model}")
    print(f"   Parameters: {config.parameters}")
    if config.prompt_template:
        print(f"   Custom prompt: Yes ({len(config.prompt_template)} chars)")
    else:
        print(f"   Custom prompt: No (using default)")
    
    # Step 3: Load environment
    print(f"\n📁 Step 3: Loading {config.provider.title()} environment...")
    api_key = load_environment(config.provider)
    
    # Step 4: Load data
    print("\n📊 Step 4: Loading data...")
    data_file = find_data_file()
    
    try:
        data = pd.read_csv(data_file)
        print(f"✅ Data loaded successfully")
        print(f"   Shape: {data.shape}")
        print(f"   Unique players: {data['playerID'].nunique()}")
        print(f"   Categories: {list(data['category'].unique())}")
        
        # Calculate expected chunks
        total_chunks = 0
        prediction_chunks = 0
        for (player_id, category), group in data.groupby(['playerID', 'category']):
            sequence_length = len(group)
            if sequence_length >= 2:
                total_chunks += sequence_length
                prediction_chunks += sequence_length - 1
        
        print(f"   Expected total chunks: {total_chunks:,}")
        print(f"   Expected prediction chunks: {prediction_chunks:,}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Step 5: Create configurable predictor
    print(f"\n🤖 Step 5: Initializing predictor with config...")
    try:
        predictor = ConfigurablePredictor(config, api_key)
        print("✅ Predictor initialized with configuration")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        sys.exit(1)
    
    # Step 6: Run prediction
    print(f"\n🚀 Step 6: Running prediction...")
    start_time = time.time()
    
    try:
        result_data, batch_id, additional_info = predictor.run_prediction(data)
        
        if result_data.empty:
            print("❌ No prediction results returned")
            sys.exit(1)
        
        # Generate output filename with config name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_config_name = config.name.replace(' ', '_').replace('-', '_')
        output_file = f"data_with_predictions_{safe_config_name}_{timestamp}.csv"
        
        result_data.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Create word-level prediction DataFrame
        print("🔄 Creating word-level prediction DataFrame...")
        word_level_data = data.copy()
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
        word_output_file = f"data_with_word_predictions_{safe_config_name}_{timestamp}.csv"
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
        config_manager.save_run_metadata(
            config=config,
            output_file=output_file,
            batch_id=batch_id,
            additional_info=additional_info
        )
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Total processing time: {format_duration(elapsed_time)}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 7: Analysis summary
    print("\n📈 Step 7: Analysis summary...")
    try:
        total_chunks = len(result_data)
        avg_confidence = result_data['confidence_llm'].mean()
        switch_rate = result_data['prediction_llm'].mean()
        
        print(f"Total chunks processed: {total_chunks:,}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Predicted switch rate: {switch_rate:.3f}")
        
        # Per-category breakdown
        print(f"\nPer-category analysis:")
        for category in result_data['category'].unique():
            cat_data = result_data[result_data['category'] == category]
            count = len(cat_data)
            rate = cat_data['prediction_llm'].mean()
            conf = cat_data['confidence_llm'].mean()
            print(f"  {category}: {count:,} chunks, {rate:.3f} switch rate, {conf:.3f} avg confidence")
        
        # Configuration details
        print(f"\nConfiguration details:")
        print(f"  Name: {config.name}")
        print(f"  Provider: {config.provider}")
        print(f"  Model: {config.model}")
        print(f"  Version: {config.version}")
        print(f"  Custom prompt: {'Yes' if config.prompt_template else 'No'}")
        print(f"  Processing time: {format_duration(elapsed_time)}")
        
    except Exception as e:
        print(f"⚠️ Error in final analysis: {e}")
    
    # Completion
    print_separator()
    print("🎉 Configuration-driven prediction pipeline completed successfully!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_file}")
    print(f"Metadata: {Path(output_file).stem}_metadata.json")
    print(f"Configuration: {config.name} (v{config.version})")
    print_separator()


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