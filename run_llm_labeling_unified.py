#!/usr/bin/env python3
"""
Unified LLM Group Labeling Script
Supports both Anthropic and TogetherAI APIs with provider selection.
"""

import os
import sys
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import provider-specific labelers
from anthropic_providers import AnthropicGroupLabeler
from together_providers import TogetherAIGroupLabeler


def load_environment(provider: str) -> str:
    """Load environment variables from .env file."""
    load_dotenv()
    
    if provider == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ Error: ANTHROPIC_API_KEY not found in .env file")
            print("Please ensure your .env file contains:")
            print("ANTHROPIC_API_KEY=your_api_key_here")
            sys.exit(1)
    elif provider == "together":
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            print("❌ Error: TOGETHER_API_KEY not found in .env file")
            print("Please ensure your .env file contains:")
            print("TOGETHER_API_KEY=your_api_key_here")
            sys.exit(1)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    print(f"✅ {provider.title()} API key loaded successfully")
    return api_key


def choose_provider() -> tuple[str, str]:
    """Let user choose between Anthropic and TogetherAI providers."""
    print("🤖 Choose your LLM provider:")
    print("1. Anthropic Claude (claude-sonnet-4-20250514)")
    print("2. TogetherAI (meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo)")
    print("3. Custom model (specify provider and model)")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            return "anthropic", "claude-sonnet-4-20250514"
        elif choice == "2":
            return "together", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        elif choice == "3":
            return choose_custom_model()
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def choose_custom_model() -> tuple[str, str]:
    """Let user specify custom provider and model."""
    print("\n📝 Custom model selection:")
    
    while True:
        provider = input("Enter provider (anthropic/together): ").strip().lower()
        if provider in ["anthropic", "together"]:
            break
        print("Invalid provider. Please enter 'anthropic' or 'together'.")
    
    if provider == "anthropic":
        print("Popular Anthropic models:")
        print("- claude-sonnet-4-20250514")
        print("- claude-3-5-sonnet-20241022")
        print("- claude-3-5-haiku-20241022")
        default_model = "claude-sonnet-4-20250514"
    else:  # together
        print("Popular TogetherAI models:")
        print("- meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        print("- meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        print("- Qwen/Qwen2.5-72B-Instruct-Turbo")
        print("- deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
        default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    
    model = input(f"Enter model name (default: {default_model}): ").strip()
    if not model:
        model = default_model
    
    return provider, model


def find_data_file():
    """Find data file with switchLLM column."""
    current_dir = Path(".")
    # Use set to avoid duplicates when file matches multiple patterns
    data_files = list(set(list(current_dir.glob("*switch*.csv")) + list(current_dir.glob("*llm*.csv"))))
    
    if not data_files:
        print("❌ No data files found with switch/llm in name")
        print("Please ensure you have run the switch classification first")
        sys.exit(1)
    
    # Try to find the most recent file or let user choose
    if len(data_files) == 1:
        data_file = data_files[0]
        print(f"✅ Found data file: {data_file}")
        return str(data_file)
    else:
        print("📁 Multiple data files found:")
        for i, f in enumerate(data_files):
            print(f"  {i+1}. {f}")
        
        while True:
            try:
                choice = int(input("Enter file number: ")) - 1
                if 0 <= choice < len(data_files):
                    data_file = data_files[choice]
                    print(f"✅ Selected: {data_file}")
                    return str(data_file)
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")


def create_labeler(provider: str, model: str, api_key: str):
    """Create the appropriate labeler based on provider."""
    if provider == "anthropic":
        return AnthropicGroupLabeler(api_key=api_key, model=model)
    elif provider == "together":
        return TogetherAIGroupLabeler(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


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
    print("🚀 Unified LLM Group Labeling Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    # Step 1: Choose provider and model
    print("\n🤖 Step 1: Provider and model selection...")
    provider, model = choose_provider()
    print(f"Selected: {provider.title()} - {model}")
    
    # Step 2: Load environment
    print(f"\n📁 Step 2: Loading {provider.title()} environment...")
    api_key = load_environment(provider)
    
    # Step 3: Find and load data
    print("\n📊 Step 3: Loading data...")
    data_file = find_data_file()
    
    try:
        data = pd.read_csv(data_file)
        print(f"✅ Data loaded successfully")
        print(f"   Shape: {data.shape}")
        print(f"   Unique players: {data['playerID'].nunique()}")
        print(f"   Categories: {list(data['category'].unique())}")
        
        # Check for switchLLM column
        if 'switchLLM' not in data.columns:
            print("❌ Error: switchLLM column not found")
            print("Please run switch classification first using run_llm_classification_unified.py")
            sys.exit(1)
        
        # Count how many players have switchLLM data
        players_with_switches = data.groupby(['playerID', 'category'])['switchLLM'].apply(lambda x: x.notna().any()).sum()
        print(f"   Players with switchLLM data: {players_with_switches}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Step 4: Initialize labeler
    print(f"\n🤖 Step 4: Initializing {provider.title()} labeler...")
    try:
        labeler = create_labeler(provider, model, api_key)
        print("✅ Group labeler initialized")
    except Exception as e:
        print(f"❌ Error initializing labeler: {e}")
        sys.exit(1)
    
    # Step 5: Run labeling
    print(f"\n🚀 Step 5: Running {provider.title()} group labeling pipeline...")
    start_time = time.time()
    
    try:
        result_data = labeler.run_group_labeling(data)
        
        # Generate output filename with timestamp and provider
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        provider_short = provider[:4]  # "anth" or "toge"
        input_name = Path(data_file).stem
        output_file = f"{input_name}_with_labels_{provider_short}_{timestamp}.csv"
        
        result_data.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        print(f"⏱️ Total processing time: {format_duration(elapsed_time)}")
        
    except Exception as e:
        print(f"❌ Error during labeling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 6: Final analysis
    print("\n📈 Step 6: Analysis summary...")
    try:
        total_rows = len(result_data)
        rows_with_labels = result_data['labelLLM'].notna().sum()
        
        print(f"Total rows: {total_rows:,}")
        print(f"Rows with labelLLM: {rows_with_labels:,}")
        
        # Show sample labels
        labeled_data = result_data[result_data['labelLLM'].notna()]
        if len(labeled_data) > 0:
            print(f"\nSample labels:")
            # Group by player and category, then take first row of each group
            sample_groups = labeled_data.groupby(['playerID', 'category'])
            for (player_id, category), group in list(sample_groups)[:3]:  # Show first 3 player-category combinations
                first_row = group.iloc[0]  # Get first row of this group
                print(f"  Player {player_id} ({category}): {first_row['text']} → {first_row['labelLLM']}")
        
        # Per-category breakdown
        print(f"\nPer-category analysis:")
        for category in result_data['category'].unique():
            cat_data = result_data[result_data['category'] == category]
            labeled_count = cat_data['labelLLM'].notna().sum()
            total_count = len(cat_data)
            coverage = (labeled_count / total_count) * 100 if total_count > 0 else 0
            print(f"  {category}: {labeled_count:,}/{total_count:,} words labeled ({coverage:.1f}%)")
        
        # Provider-specific metrics
        print(f"\nProvider details:")
        print(f"  Provider: {provider.title()}")
        print(f"  Model: {model}")
        print(f"  Processing time: {format_duration(elapsed_time)}")
        if rows_with_labels > 0:
            labels_per_minute = (rows_with_labels / elapsed_time) * 60
            print(f"  Processing speed: {labels_per_minute:.1f} labels/minute")
        
    except Exception as e:
        print(f"⚠️ Error in final analysis: {e}")
        print("Results were saved successfully, but analysis summary failed.")
    
    # Completion
    print_separator()
    print("🎉 Pipeline completed successfully!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_file}")
    print(f"Provider: {provider.title()} ({model})")
    print_separator()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user")
        print("Any submitted batch will continue processing on the provider's servers.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)