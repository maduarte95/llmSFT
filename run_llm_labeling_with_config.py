#!/usr/bin/env python3
"""
Configuration-Driven LLM Group Labeling Script
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

# Import provider-specific labelers
from anthropic_providers import AnthropicGroupLabeler, AnthropicProvider
from together_providers import TogetherAIGroupLabeler, TogetherAIProvider

# Import flexible data utilities
from data_utils import get_data_file_for_script


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
    """Find data file with switchLLM column using flexible selection."""
    return get_data_file_for_script("labeling")


class ConfigurableLabeler:
    """Wrapper that applies config parameters to existing labelers."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        
        # Create base labeler with config
        if config.provider == "anthropic":
            self.labeler = AnthropicGroupLabeler(api_key=api_key, model=config.model)
        elif config.provider == "together":
            self.labeler = TogetherAIGroupLabeler(api_key=api_key, model=config.model)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        # Apply custom parameters
        self.labeler.provider.default_params = config.parameters
        
        # Store config for prompt customization
        self.labeler.config = config
    
    def create_group_labeling_prompt(self, groups, category):
        """Create prompt using config's custom template if available."""
        if self.config.prompt_template:
            # Use custom prompt template
            formatted_groups = []
            for i, group in enumerate(groups):
                words_str = ", ".join(group['words'])
                formatted_groups.append(f"Group {i+1}: {words_str}")
            groups_text = "\n".join(formatted_groups)
            
            return self.config.prompt_template.format(
                category=category,
                groups_text=groups_text,
                num_groups=len(groups)
            )
        else:
            # Use default prompt
            return self.labeler.create_group_labeling_prompt(groups, category)
    
    def prepare_batch_requests(self, data: pd.DataFrame):
        """Prepare batch requests with custom parameters from config."""
        requests = []
        request_index = 0
        
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            sequence_data = group.sort_values('word_index')
            
            if 'switchLLM' not in sequence_data.columns or sequence_data['switchLLM'].isna().all():
                print(f"Skipping player {player_id} ({category}) - no switchLLM data")
                continue
            
            groups = self.labeler.create_groups_from_switches(sequence_data)
            
            if len(groups) == 0:
                print(f"Skipping player {player_id} ({category}) - no groups found")
                continue
            
            # Use configurable prompt (will use custom template if available)
            prompt = self.create_group_labeling_prompt(groups, category)
            custom_id = f"label_{self.config.name}_{request_index:04d}"
            
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
                    "num_groups": len(groups),
                    "groups": groups,
                    "request_index": request_index,
                    "config_name": self.config.name,
                    "config_version": self.config.version
                }
            }
            
            requests.append(request)
            request_index += 1
        
        print(f"Prepared {len(requests)} batch requests using config: {self.config.name}")
        return requests
    
    def run_labeling(self, data: pd.DataFrame):
        """Run labeling with config parameters and metadata tracking."""
        print(f"Starting group labeling with config: {self.config.name}")
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model}")
        print(f"Parameters: {self.config.parameters}")
        
        if self.config.prompt_template:
            print("✅ Using custom prompt template from config")
        else:
            print("📝 Using default prompt template")
        
        if 'switchLLM' not in data.columns:
            raise ValueError("switchLLM column not found. Please run switch classification first.")
        
        # Prepare requests with config parameters
        batch_requests = self.prepare_batch_requests(data)
        
        if len(batch_requests) == 0:
            print("No valid requests to process. Returning original data.")
            result_data = data.copy()
            result_data['labelLLM'] = None
            return result_data, None, {}
        
        # Run dry run test
        if not self.labeler.dry_run_test(data, num_tests=min(2, len(batch_requests))):
            print("❌ Dry run failed. Please check your configuration and data.")
            return data.copy(), None, {}
        
        # Create batch
        batch_id = self.labeler.provider.create_batch(batch_requests)
        
        # Monitor batch completion
        final_batch = self.labeler.provider.monitor_batch(batch_id)
        
        # Process results with metadata
        result_data = self.labeler.process_batch_results(batch_id, data, batch_requests)
        
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
    print("🏷️ Configuration-Driven LLM Group Labeling")
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
        config = config_manager.interactive_config_selector("labels")
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
        
        # Check for switchLLM column
        if 'switchLLM' not in data.columns:
            print("❌ Error: switchLLM column not found!")
            print("This script requires switch classification results.")
            print("Please run switch classification first using:")
            print("  - run_llm_classification_unified.py")
            print("  - run_llm_classification_with_config.py")
            sys.exit(1)
        
        switch_data_count = data['switchLLM'].notna().sum()
        print(f"   Rows with switch classifications: {switch_data_count:,}")
        
        if switch_data_count == 0:
            print("❌ Error: No switch classification data found!")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Step 5: Create configurable labeler
    print(f"\n🤖 Step 5: Initializing labeler with config...")
    try:
        labeler = ConfigurableLabeler(config, api_key)
        print("✅ Labeler initialized with configuration")
    except Exception as e:
        print(f"❌ Error initializing labeler: {e}")
        sys.exit(1)
    
    # Step 6: Run labeling
    print(f"\n🚀 Step 6: Running group labeling...")
    start_time = time.time()
    
    try:
        result_data, batch_id, additional_info = labeler.run_labeling(data)
        
        # Generate output filename with config name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_config_name = config.name.replace(' ', '_').replace('-', '_')
        output_file = f"data_with_labels_{safe_config_name}_{timestamp}.csv"
        
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
        print(f"❌ Error during labeling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 7: Analysis summary
    print("\n📈 Step 7: Analysis summary...")
    try:
        total_rows = len(result_data)
        rows_with_labels = result_data['labelLLM'].notna().sum()
        
        print(f"Total rows: {total_rows:,}")
        print(f"Rows with LLM labels: {rows_with_labels:,}")
        
        if rows_with_labels > 0:
            # Count unique labels
            unique_labels = result_data['labelLLM'].dropna().nunique()
            print(f"Unique labels generated: {unique_labels}")
            
            # Show most common labels
            top_labels = result_data['labelLLM'].value_counts().head(5)
            print(f"\nTop 5 most common labels:")
            for label, count in top_labels.items():
                print(f"  '{label}': {count} occurrences")
        
        # Per-category breakdown
        print(f"\nPer-category analysis:")
        for category in result_data['category'].unique():
            cat_data = result_data[result_data['category'] == category]
            label_count = cat_data['labelLLM'].notna().sum()
            total_count = len(cat_data)
            coverage = label_count / total_count if total_count > 0 else 0
            print(f"  {category}: {label_count:,}/{total_count:,} labeled ({coverage:.1%})")
        
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
    print("🎉 Configuration-driven labeling pipeline completed successfully!")
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