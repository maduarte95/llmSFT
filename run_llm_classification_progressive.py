#!/usr/bin/env python3
"""
Progressive LLM Switch Classification Script
Uses progressive classification where the LLM sees chunks of increasing size 
and classifies the second-to-last word using future context.
Uses YAML configuration files for model selection and parameter control.
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

# Import provider-specific classifiers
from anthropic_providers import AnthropicProgressiveSwitchClassifier, AnthropicProvider
from together_providers import TogetherAIProgressiveSwitchClassifier, TogetherAIProvider

# Import flexible data utilities
from data_utils import get_data_file_for_script


def load_environment(provider: str) -> str:
    """Load environment variables from .env file."""
    load_dotenv()
    
    if provider == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not found in .env file")
            sys.exit(1)
    elif provider == "together":
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            print("ERROR: TOGETHER_API_KEY not found in .env file")
            sys.exit(1)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    print(f"SUCCESS: {provider.title()} API key loaded successfully")
    return api_key


def find_data_file():
    """Find the filtered data CSV file using flexible selection."""
    return get_data_file_for_script("classification")


class ProgressiveConfigurableClassifier:
    """Progressive classifier that uses future context to classify words."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        
        # Check if IRT timing should be included (backward compatible)
        self.include_irt = getattr(config, 'include_irt', False)
        
        # Create base classifier
        if config.provider == "anthropic":
            self.classifier = AnthropicProgressiveSwitchClassifier(
                api_key=api_key, 
                model=config.model, 
                config=config
            )
        elif config.provider == "together":
            self.classifier = TogetherAIProgressiveSwitchClassifier(
                api_key=api_key, 
                model=config.model, 
                config=config
            )
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        # Pass IRT setting to classifier
        self.classifier.include_irt = self.include_irt
        
        # Apply custom parameters
        self.classifier.provider.default_params = config.parameters
    
    def prepare_batch_requests(self, data: pd.DataFrame):
        """Prepare batch requests for progressive classification."""
        chunks = self.classifier.create_progressive_chunk_data(data)
        
        requests = []
        
        for chunk in chunks:
            # Create prompt for this specific chunk and target word
            prompt = self.classifier.create_progressive_classification_prompt(
                chunk['chunk_words'],
                chunk['category'],
                chunk['target_word_index'],
                chunk['target_word'],
                chunk.get('chunk_irt_values')  # Pass IRT values if available
            )
            
            custom_id = f"prog_class_{self.config.name}_{chunk['chunk_id']:06d}"
            
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
                    "target_word_index": chunk['target_word_index'],
                    "target_word": chunk['target_word'],
                    "sequence_length": chunk['sequence_length'],
                    "config_name": self.config.name,
                    "config_version": self.config.version
                }
            }
            
            requests.append(request)
        
        print(f"Prepared {len(requests)} progressive classification requests using config: {self.config.name}")
        return requests
    
    def run_classification(self, data: pd.DataFrame):
        """Run progressive classification."""
        print(f"Starting progressive classification with config: {self.config.name}")
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model}")
        print(f"Parameters: {self.config.parameters}")
        
        # Prepare requests for progressive classification
        batch_requests = self.prepare_batch_requests(data)
        
        # Run dry run test
        if len(batch_requests) > 0:
            print(f"SUCCESS: Prepared {len(batch_requests)} progressive classification requests")
        else:
            print("ERROR: No requests prepared. Please check your data.")
            return data.copy()
        
        # Create batch
        batch_id = self.classifier.provider.create_batch(batch_requests)
        
        # Monitor batch completion
        final_batch = self.classifier.provider.monitor_batch(batch_id)
        
        # Process results
        result_data = self.classifier.process_batch_results(batch_id, data, batch_requests)
        
        # Add config metadata to results
        config_manager = ConfigManager()
        additional_info = {
            'num_requests': len(batch_requests),
            'num_players': data['playerID'].nunique(),
            'categories': list(data['category'].unique()),
            'data_shape': list(data.shape),
            'progressive_format': True
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
    print("PROGRESSIVE LLM Switch Classification (Future Context Format)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    # Step 1: Load configurations
    print("\nSTEP 1: Loading configurations...")
    config_manager = ConfigManager()
    config_manager.print_config_summary()
    
    # Step 2: Select configuration
    print("\nSTEP 2: Configuration selection...")
    
    # Option to specify config via command line
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        config = config_manager.get_config(config_name)
        if not config:
            print(f"ERROR: Configuration '{config_name}' not found")
            sys.exit(1)
        print(f"SUCCESS: Using config from command line: {config_name}")
    else:
        # Interactive selection
        config = config_manager.interactive_config_selector("switches")
        if not config:
            print("ERROR: No configuration selected")
            sys.exit(1)
    
    # Validate configuration
    issues = config_manager.validate_config(config)
    if issues:
        print(f"ERROR: Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    print(f"SUCCESS: Selected configuration: {config.name}")
    print(f"   Provider: {config.provider}")
    print(f"   Model: {config.model}")
    print(f"   Parameters: {config.parameters}")
    
    # Step 3: Load environment
    print(f"\nSTEP 3: Loading {config.provider.title()} environment...")
    api_key = load_environment(config.provider)
    
    # Step 4: Load data
    print("\nSTEP 4: Loading data...")
    data_file = find_data_file()
    
    try:
        data = pd.read_csv(data_file)
        print(f"SUCCESS: Data loaded successfully")
        print(f"   Shape: {data.shape}")
        print(f"   Unique players: {data['playerID'].nunique()}")
        print(f"   Categories: {list(data['category'].unique())}")
        
        # Calculate expected chunks for progressive classification
        total_chunks = 0
        for (player_id, category), group in data.groupby(['playerID', 'category']):
            sequence_length = len(group)
            if sequence_length >= 2:
                # Progressive classification: chunks from [0,1] to [0,1,...,n-1]
                # Each chunk classifies the second-to-last word
                total_chunks += sequence_length - 1
        
        print(f"   Expected progressive classification chunks: {total_chunks:,}")
        
    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        sys.exit(1)
    
    # Step 5: Create progressive classifier
    print(f"\nSTEP 5: Initializing progressive classifier...")
    try:
        classifier = ProgressiveConfigurableClassifier(config, api_key)
        print("SUCCESS: Progressive classifier initialized with configuration")
    except Exception as e:
        print(f"ERROR: Error initializing classifier: {e}")
        sys.exit(1)
    
    # Step 6: Run progressive classification
    print(f"\nSTEP 6: Running progressive classification...")
    start_time = time.time()
    
    try:
        result_data, batch_id, additional_info = classifier.run_classification(data)
        
        # Generate output filename with progressive indicator
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_config_name = config.name.replace(' ', '_').replace('-', '_')
        output_file = f"data_with_switches_progressive_{safe_config_name}_{timestamp}.csv"
        
        result_data.to_csv(output_file, index=False)
        print(f"SUCCESS: Results saved to: {output_file}")
        
        # Save metadata
        config_manager.save_run_metadata(
            config=config,
            output_file=output_file,
            batch_id=batch_id,
            additional_info=additional_info
        )
        
        elapsed_time = time.time() - start_time
        print(f"TIMING: Total processing time: {format_duration(elapsed_time)}")
        
    except Exception as e:
        print(f"ERROR: Error during classification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 7: Analysis summary
    print("\nSTEP 7: Analysis summary...")
    try:
        total_rows = len(result_data)
        rows_with_llm_prog = result_data['switchLLM_prog'].notna().sum()
        llm_prog_switch_rate = result_data['switchLLM_prog'].mean()
        
        print(f"Total rows: {total_rows:,}")
        print(f"Rows with LLM progressive classifications: {rows_with_llm_prog:,}")
        print(f"LLM progressive switch rate: {llm_prog_switch_rate:.3f}")
        
        # Compare with human classifications if available
        if 'switch' in result_data.columns:
            human_switch_rate = result_data['switch'].mean()
            print(f"Human switch rate: {human_switch_rate:.3f}")
            
            both_exist = result_data[['switch', 'switchLLM_prog']].notna().all(axis=1)
            if both_exist.sum() > 0:
                agreement = (result_data.loc[both_exist, 'switch'] == result_data.loc[both_exist, 'switchLLM_prog']).mean()
                print(f"Human-LLM progressive agreement: {agreement:.3f}")
        
        # Compare with other LLM methods if available
        if 'switchLLM' in result_data.columns:
            robust_switch_rate = result_data['switchLLM'].mean()
            print(f"LLM robust switch rate: {robust_switch_rate:.3f}")
            
            both_llm_exist = result_data[['switchLLM', 'switchLLM_prog']].notna().all(axis=1)
            if both_llm_exist.sum() > 0:
                llm_agreement = (result_data.loc[both_llm_exist, 'switchLLM'] == result_data.loc[both_llm_exist, 'switchLLM_prog']).mean()
                print(f"Robust-Progressive LLM agreement: {llm_agreement:.3f}")
        
        # Configuration details
        print(f"\nConfiguration details:")
        print(f"  Name: {config.name}")
        print(f"  Provider: {config.provider}")
        print(f"  Model: {config.model}")
        print(f"  Version: {config.version}")
        print(f"  Progressive format: Future Context Classification")
        print(f"  Processing time: {format_duration(elapsed_time)}")
        
    except Exception as e:
        print(f"WARNING: Error in final analysis: {e}")
    
    # Completion
    print_separator()
    print("SUCCESS: Progressive classification pipeline completed successfully!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_file}")
    print(f"Metadata: {Path(output_file).stem}_metadata.json")
    print(f"Configuration: {config.name} (v{config.version}) - Progressive Format")
    print_separator()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWARNING: Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)