#!/usr/bin/env python3
"""
Configuration-Driven LLM Switch Classification Script
Uses YAML configuration files for model selection and parameter control.
Supports reproducible experiments and easy model comparison.
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
from anthropic_providers import AnthropicSwitchClassifier, AnthropicProvider
from together_providers import TogetherAISwitchClassifier, TogetherAIProvider


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
        "*filtered*.csv"
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


def create_enhanced_provider(config: ModelConfig, api_key: str):
    """Create provider with custom parameters from config."""
    if config.provider == "anthropic":
        provider = AnthropicProvider(api_key=api_key, model=config.model)
        # Override parameters if needed
        provider.default_params = config.parameters
        return provider
    elif config.provider == "together":
        provider = TogetherAIProvider(api_key=api_key, model=config.model)
        provider.default_params = config.parameters
        return provider
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


class ConfigurableClassifier:
    """Wrapper that applies config parameters to existing classifiers."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        
        # Create base classifier
        if config.provider == "anthropic":
            self.classifier = AnthropicSwitchClassifier(api_key=api_key, model=config.model)
        elif config.provider == "together":
            self.classifier = TogetherAISwitchClassifier(api_key=api_key, model=config.model)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        # Apply custom parameters
        self.classifier.provider.default_params = config.parameters
    
    def create_switch_identification_prompt(self, word_sequence, category):
        """Create prompt using config's custom template if available."""
        if self.config.prompt_template:
            # Use custom prompt template
            formatted_words = "\n".join([f"{i}. {word}" for i, word in enumerate(word_sequence)])
            return self.config.prompt_template.format(
                category=category,
                word_sequence=formatted_words,
                num_words=len(word_sequence)
            )
        else:
            # Use default prompt
            return self.classifier.create_switch_identification_prompt(word_sequence, category)
    
    def prepare_batch_requests(self, data: pd.DataFrame):
        """Prepare batch requests with custom parameters."""
        batch_requests = []
        request_index = 0
        
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            sequence_data = group.sort_values('word_index')
            complete_word_sequence = sequence_data['text'].tolist()
            
            if not complete_word_sequence:
                continue
            
            # Use configurable prompt
            prompt = self.create_switch_identification_prompt(complete_word_sequence, category)
            custom_id = f"switch_{self.config.name}_{request_index:04d}"
            
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
                    "player_id": player_id,
                    "category": category,
                    "total_words": len(sequence_data),
                    "sequence_length": len(complete_word_sequence),
                    "word_sequence": complete_word_sequence,
                    "request_index": request_index,
                    "config_name": self.config.name,
                    "config_version": self.config.version
                }
            }
            
            batch_requests.append(request)
            request_index += 1
        
        print(f"Prepared {len(batch_requests)} batch requests using config: {self.config.name}")
        return batch_requests
    
    def run_classification(self, data: pd.DataFrame):
        """Run classification with config parameters and metadata tracking."""
        print(f"Starting classification with config: {self.config.name}")
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model}")
        print(f"Parameters: {self.config.parameters}")
        
        # Prepare requests with config parameters
        batch_requests = self.prepare_batch_requests(data)
        
        # Run dry run test
        if not self.classifier.dry_run_test(data, num_tests=min(2, len(batch_requests))):
            print("❌ Dry run failed. Please check your configuration and data.")
            return data.copy()
        
        # Create batch
        batch_id = self.classifier.provider.create_batch(batch_requests)
        
        # Monitor batch completion
        final_batch = self.classifier.provider.monitor_batch(batch_id)
        
        # Process results with metadata
        result_data = self.classifier.process_batch_results(batch_id, data, batch_requests)
        
        # Add config metadata to results
        config_manager = ConfigManager()
        additional_info = {
            'num_requests': len(batch_requests),
            'num_players': data['playerID'].nunique(),
            'categories': list(data['category'].unique()),
            'data_shape': list(data.shape)
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
    print("🚀 Configuration-Driven LLM Switch Classification")
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
        config = config_manager.interactive_config_selector("switches")
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
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Step 5: Create configurable classifier
    print(f"\n🤖 Step 5: Initializing classifier with config...")
    try:
        classifier = ConfigurableClassifier(config, api_key)
        print("✅ Classifier initialized with configuration")
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        sys.exit(1)
    
    # Step 6: Run classification
    print(f"\n🚀 Step 6: Running classification...")
    start_time = time.time()
    
    try:
        result_data, batch_id, additional_info = classifier.run_classification(data)
        
        # Generate output filename with config name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_config_name = config.name.replace(' ', '_').replace('-', '_')
        output_file = f"data_with_switches_{safe_config_name}_{timestamp}.csv"
        
        result_data.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
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
        print(f"❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 7: Analysis summary
    print("\n📈 Step 7: Analysis summary...")
    try:
        total_rows = len(result_data)
        rows_with_llm = result_data['switchLLM'].notna().sum()
        llm_switch_rate = result_data['switchLLM'].mean()
        
        print(f"Total rows: {total_rows:,}")
        print(f"Rows with LLM classifications: {rows_with_llm:,}")
        print(f"LLM switch rate: {llm_switch_rate:.3f}")
        
        # Compare with human classifications if available
        if 'switch' in result_data.columns:
            human_switch_rate = result_data['switch'].mean()
            print(f"Human switch rate: {human_switch_rate:.3f}")
            
            both_exist = result_data[['switch', 'switchLLM']].notna().all(axis=1)
            if both_exist.sum() > 0:
                agreement = (result_data.loc[both_exist, 'switch'] == result_data.loc[both_exist, 'switchLLM']).mean()
                print(f"Human-LLM agreement: {agreement:.3f}")
        
        # Configuration details
        print(f"\nConfiguration details:")
        print(f"  Name: {config.name}")
        print(f"  Provider: {config.provider}")
        print(f"  Model: {config.model}")
        print(f"  Version: {config.version}")
        print(f"  Processing time: {format_duration(elapsed_time)}")
        
    except Exception as e:
        print(f"⚠️ Error in final analysis: {e}")
    
    # Completion
    print_separator()
    print("🎉 Configuration-driven pipeline completed successfully!")
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