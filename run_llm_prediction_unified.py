#!/usr/bin/env python3
"""
Unified LLM Switch Prediction Script
Supports both Anthropic and TogetherAI APIs with provider selection.
Predicts whether the next word will be a switch for incremental chunks.
"""

import os
import sys
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Import provider-specific predictors
from anthropic_providers import AnthropicSwitchPredictor
from together_providers import TogetherAISwitchPredictor


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
    """Find the filtered data CSV file."""
    current_dir = Path(".")
    
    # Common file name patterns
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
    print("Expected file patterns:")
    for pattern in patterns:
        print(f"  - {pattern}")
    
    # List available CSV files
    csv_files = list(current_dir.glob("*.csv"))
    if csv_files:
        print("\nAvailable CSV files:")
        for f in csv_files:
            print(f"  - {f}")
        
        # Ask user to specify
        while True:
            filename = input("\nPlease enter the filename to use: ").strip()
            if Path(filename).exists():
                return filename
            else:
                print(f"File '{filename}' not found. Please try again.")
    else:
        print("No CSV files found in current directory.")
        sys.exit(1)


def create_predictor(provider: str, model: str, api_key: str):
    """Create the appropriate predictor based on provider."""
    if provider == "anthropic":
        return AnthropicSwitchPredictor(api_key=api_key, model=model)
    elif provider == "together":
        return TogetherAISwitchPredictor(api_key=api_key, model=model)
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
    print("🔮 Unified LLM Switch Prediction Pipeline")
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
        
        # Count player-category combinations
        player_category_combos = data.groupby(['playerID', 'category']).size().shape[0]
        print(f"   Player-category combinations: {player_category_combos}")
        
        # Calculate expected chunks
        total_chunks = 0
        prediction_chunks = 0
        for (player_id, category), group in data.groupby(['playerID', 'category']):
            sequence_length = len(group)
            if sequence_length >= 2:
                total_chunks += sequence_length
                prediction_chunks += sequence_length - 1  # Can't predict after last word
        
        print(f"   Expected total chunks: {total_chunks:,}")
        print(f"   Expected prediction chunks: {prediction_chunks:,}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Step 4: Initialize predictor
    print(f"\n🤖 Step 4: Initializing {provider.title()} predictor...")
    try:
        predictor = create_predictor(provider, model, api_key)
        print("✅ Predictor initialized")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        sys.exit(1)
    
    # Step 5: Run prediction
    print(f"\n🚀 Step 5: Running {provider.title()} prediction pipeline...")
    print("   Note: This creates incremental chunks (word 0, words 0-1, words 0-2, etc.)")
    print("   Each chunk predicts whether the NEXT word will be a switch")
    start_time = time.time()
    
    try:
        result_data = predictor.run_switch_prediction(data)
        
        if result_data.empty:
            print("❌ No prediction results returned")
            sys.exit(1)
        
        # Generate output filename with timestamp and provider
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        provider_short = provider[:4]  # "anth" or "toge"
        output_file = f"data_with_llm_predictions_{provider_short}_{timestamp}.csv"
        
        result_data.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        print(f"⏱️ Total processing time: {format_duration(elapsed_time)}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 6: Final analysis
    print("\n📈 Step 6: Analysis summary...")
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
        
        # Chunk length analysis
        print(f"\nChunk length analysis:")
        chunk_stats = result_data.groupby('chunk_length').agg({
            'prediction_llm': ['count', 'mean'],
            'confidence_llm': 'mean'
        }).round(3)
        
        for length in sorted(result_data['chunk_length'].unique())[:10]:  # Show first 10 lengths
            length_data = result_data[result_data['chunk_length'] == length]
            count = len(length_data)
            rate = length_data['prediction_llm'].mean()
            conf = length_data['confidence_llm'].mean()
            print(f"  Length {length}: {count:,} chunks, {rate:.3f} switch rate, {conf:.3f} avg confidence")
        
        if result_data['chunk_length'].max() > 10:
            print(f"  ... and {len(result_data['chunk_length'].unique()) - 10} more lengths")
        
        # Provider-specific metrics
        print(f"\nProvider details:")
        print(f"  Provider: {provider.title()}")
        print(f"  Model: {model}")
        print(f"  Processing time: {format_duration(elapsed_time)}")
        if total_chunks > 0:
            chunks_per_minute = (total_chunks / elapsed_time) * 60
            print(f"  Processing speed: {chunks_per_minute:.1f} chunks/minute")
        
    except Exception as e:
        print(f"⚠️ Error in final analysis: {e}")
        print("Results were saved successfully, but analysis summary failed.")
    
    # Completion
    print_separator()
    print("🎉 Pipeline completed successfully!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_file}")
    print(f"Provider: {provider.title()} ({model})")
    print("\n📋 Output format:")
    print("  - chunk_id: Unique chunk identifier")
    print("  - player_id: Participant ID")
    print("  - category: Task category")
    print("  - chunk_end_index: Last word index in chunk")
    print("  - chunk_words: Words in chunk (comma-separated)")
    print("  - next_word: The word being predicted")
    print("  - prediction_llm: 0=continue group, 1=start new group")
    print("  - confidence_llm: Model confidence (0.0-1.0)")
    print("  - reasoning_llm: Model's reasoning")
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