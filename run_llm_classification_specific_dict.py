#!/usr/bin/env python3
"""
LLM Switch Classification Script - Specific Sequences
Runs LLM classification for specific player sequences (e.g., to fix failed ones).
"""

import os
import sys
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Import your classifier
from llm_switch_classifier_dict import LLMSwitchClassifierDict

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in .env file")
        print("Please ensure your .env file contains:")
        print("ANTHROPIC_API_KEY=your_api_key_here")
        sys.exit(1)
    
    print("✅ API key loaded successfully")
    return api_key

def find_data_file():
    """Find the data CSV file."""
    current_dir = Path(".")
    
    # Look for files with LLM data first, then original filtered data
    patterns = [
        "*llm_switches*.csv",
        "*with_llm*.csv", 
        "filtered_data_for_analysis.csv",
        "*filtered*.csv"
    ]
    
    for pattern in patterns:
        files = list(current_dir.glob(pattern))
        if files:
            # If multiple files, get the most recent one
            file_path = max(files, key=lambda f: f.stat().st_mtime)
            print(f"✅ Found data file: {file_path}")
            return str(file_path)
    
    print("❌ Error: Could not find data file")
    sys.exit(1)

def find_missing_or_failed_sequences(data: pd.DataFrame):
    """
    Find sequences that don't have switchLLM data or might need reprocessing.
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        List of (playerID, category) tuples that need processing
    """
    missing_sequences = []
    
    # Group by playerID and category
    player_groups = data.groupby(['playerID', 'category'])
    
    for (player_id, category), group in player_groups:
        # Check if switchLLM column exists and has data
        if 'switchLLM' not in data.columns:
            missing_sequences.append((player_id, category))
        else:
            # Check if this player has any non-null switchLLM values
            player_switch_data = group['switchLLM']
            if player_switch_data.isna().all():
                missing_sequences.append((player_id, category))
    
    return missing_sequences

def get_user_sequence_selection(data: pd.DataFrame):
    """
    Let user select which sequences to process.
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        List of (playerID, category) tuples to process
    """
    # Find missing sequences automatically
    missing_sequences = find_missing_or_failed_sequences(data)
    
    print(f"\n📊 Analysis of sequences:")
    print(f"   Total unique sequences: {data.groupby(['playerID', 'category']).ngroups}")
    
    if 'switchLLM' in data.columns:
        completed_sequences = data.groupby(['playerID', 'category'])['switchLLM'].apply(lambda x: x.notna().any()).sum()
        print(f"   Sequences with switchLLM data: {completed_sequences}")
        print(f"   Sequences missing switchLLM data: {len(missing_sequences)}")
    else:
        print(f"   Sequences missing switchLLM data: {len(missing_sequences)} (no switchLLM column)")
    
    if len(missing_sequences) > 0:
        print(f"\n❌ Missing/failed sequences found:")
        for i, (player_id, category) in enumerate(missing_sequences):
            player_data = data[(data['playerID'] == player_id) & (data['category'] == category)]
            word_count = len(player_data)
            print(f"   {i+1}. {player_id} ({category}) - {word_count} words")
    
    # User selection options
    print(f"\n📝 Processing options:")
    print(f"1. Process all missing/failed sequences ({len(missing_sequences)} sequences)")
    print(f"2. Process specific sequence(s) by player ID")
    print(f"3. Process all sequences (complete reprocessing)")
    print(f"4. Show sequence details and choose")
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                if len(missing_sequences) == 0:
                    print("✅ No missing sequences found!")
                    return []
                return missing_sequences
            
            elif choice == "2":
                return get_specific_sequences_by_id(data)
            
            elif choice == "3":
                # Return all sequences
                all_sequences = []
                for (player_id, category), group in data.groupby(['playerID', 'category']):
                    all_sequences.append((player_id, category))
                return all_sequences
            
            elif choice == "4":
                return get_sequences_with_details(data)
            
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            sys.exit(0)

def get_specific_sequences_by_id(data: pd.DataFrame):
    """Let user specify sequences by player ID."""
    sequences = []
    
    print(f"\n📋 Available player IDs:")
    unique_combinations = data.groupby(['playerID', 'category']).size().reset_index()
    for idx, row in unique_combinations.iterrows():
        player_id = row['playerID']
        category = row['category']
        word_count = row[0]
        
        # Check if has switchLLM data
        if 'switchLLM' in data.columns:
            player_data = data[(data['playerID'] == player_id) & (data['category'] == category)]
            has_data = player_data['switchLLM'].notna().any()
            status = "✅" if has_data else "❌"
        else:
            status = "❌"
        
        print(f"   {status} {player_id} ({category}) - {word_count} words")
    
    print(f"\nEnter player IDs to process (one per line, empty line to finish):")
    
    while True:
        player_id = input("Player ID: ").strip()
        if not player_id:
            break
            
        # Find matching sequences
        matching = data[data['playerID'] == player_id].groupby(['playerID', 'category'])
        if len(matching) == 0:
            print(f"❌ No data found for player {player_id}")
            continue
        
        for (pid, category), group in matching:
            sequences.append((pid, category))
            word_count = len(group)
            print(f"   ✅ Added: {pid} ({category}) - {word_count} words")
    
    return sequences

def get_sequences_with_details(data: pd.DataFrame):
    """Show detailed sequence info and let user choose."""
    print(f"\n📊 Detailed sequence information:")
    
    sequences_info = []
    for (player_id, category), group in data.groupby(['playerID', 'category']):
        word_count = len(group)
        
        # Check switchLLM status
        if 'switchLLM' in data.columns:
            has_switch_data = group['switchLLM'].notna().any()
            switch_rate = group['switchLLM'].mean() if has_switch_data else None
        else:
            has_switch_data = False
            switch_rate = None
        
        # Check if has human switch data for comparison
        has_human_data = 'switch' in data.columns and group['switch'].notna().any()
        human_switch_rate = group['switch'].mean() if has_human_data else None
        
        sequences_info.append({
            'player_id': player_id,
            'category': category,
            'word_count': word_count,
            'has_switch_data': has_switch_data,
            'switch_rate': switch_rate,
            'human_switch_rate': human_switch_rate
        })
    
    # Display table
    print(f"{'#':<3} {'Player ID':<30} {'Category':<18} {'Words':<6} {'LLM':<5} {'Switch Rate':<12} {'Human Rate':<12}")
    print("-" * 90)
    
    for i, info in enumerate(sequences_info):
        status = "✅" if info['has_switch_data'] else "❌"
        switch_rate_str = f"{info['switch_rate']:.3f}" if info['switch_rate'] is not None else "N/A"
        human_rate_str = f"{info['human_switch_rate']:.3f}" if info['human_switch_rate'] is not None else "N/A"
        
        print(f"{i+1:<3} {info['player_id']:<30} {info['category']:<18} {info['word_count']:<6} {status:<5} {switch_rate_str:<12} {human_rate_str:<12}")
    
    print(f"\nEnter sequence numbers to process (comma-separated, e.g., '1,3,5' or 'all'):")
    selection = input("Selection: ").strip()
    
    if selection.lower() == 'all':
        return [(info['player_id'], info['category']) for info in sequences_info]
    
    try:
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        selected_sequences = []
        for idx in indices:
            if 0 <= idx < len(sequences_info):
                info = sequences_info[idx]
                selected_sequences.append((info['player_id'], info['category']))
            else:
                print(f"❌ Invalid sequence number: {idx + 1}")
        
        return selected_sequences
    except ValueError:
        print("❌ Invalid selection format")
        return []

def print_separator(char="=", length=60):
    """Print a separator line."""
    print(char * length)

def main():
    """Main execution function."""
    print_separator()
    print("🎯 LLM Switch Classification - Specific Sequences")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    # Step 1: Load environment
    print("\n📁 Step 1: Loading environment...")
    api_key = load_environment()
    
    # Step 2: Find and load data
    print("\n📊 Step 2: Loading data...")
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
    
    # Step 3: Get sequence selection
    print("\n🎯 Step 3: Selecting sequences to process...")
    target_sequences = get_user_sequence_selection(data)
    
    if len(target_sequences) == 0:
        print("✅ No sequences to process. Exiting.")
        sys.exit(0)
    
    print(f"\n📝 Selected {len(target_sequences)} sequences for processing:")
    for player_id, category in target_sequences:
        player_data = data[(data['playerID'] == player_id) & (data['category'] == category)]
        word_count = len(player_data)
        print(f"   • {player_id} ({category}) - {word_count} words")
    
    # Confirm processing
    confirm = input(f"\nProceed with processing these {len(target_sequences)} sequences? (y/N): ").strip().lower()
    if confirm != 'y':
        print("👋 Processing cancelled.")
        sys.exit(0)
    
    # Step 4: Filter data for selected sequences
    print("\n🔄 Step 4: Filtering data for selected sequences...")
    
    # Create mask for selected sequences
    mask = pd.Series([False] * len(data))
    for player_id, category in target_sequences:
        sequence_mask = (data['playerID'] == player_id) & (data['category'] == category)
        mask = mask | sequence_mask
    
    filtered_data = data[mask].copy()
    print(f"✅ Filtered to {len(filtered_data)} rows for processing")
    
    # Step 5: Initialize classifier
    print("\n🤖 Step 5: Initializing LLM classifier...")
    classifier = LLMSwitchClassifierDict(api_key=api_key)
    print("✅ Classifier initialized")
    
    # Step 6: Run classification
    print("\n🚀 Step 6: Running LLM classification...")
    try:
        result_data = classifier.run_switch_classification(filtered_data)
        
        # Merge results back to original data
        print("\n🔄 Step 7: Merging results back to original dataset...")
        
        # Update the original data with new results
        updated_data = data.copy()
        
        for idx, row in result_data.iterrows():
            # Find matching row in original data
            original_mask = (
                (updated_data['playerID'] == row['playerID']) & 
                (updated_data['category'] == row['category']) &
                (updated_data['word_index'] == row['word_index'])
            )
            
            # Update switchLLM value
            if 'switchLLM' in row and pd.notna(row['switchLLM']):
                updated_data.loc[original_mask, 'switchLLM'] = row['switchLLM']
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data_with_llm_switches_{timestamp}.csv"
        updated_data.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Show summary
        print("\n📈 Step 8: Final summary...")
        total_rows = len(updated_data)
        llm_rows = updated_data['switchLLM'].notna().sum()
        print(f"Total rows: {total_rows}")
        print(f"Rows with LLM classifications: {llm_rows}")
        print(f"LLM switch rate: {updated_data['switchLLM'].mean():.3f}")
        
        # Show results for processed sequences
        print(f"\nResults for processed sequences:")
        for player_id, category in target_sequences:
            player_data = updated_data[(updated_data['playerID'] == player_id) & (updated_data['category'] == category)]
            has_llm_data = player_data['switchLLM'].notna().any()
            word_count = len(player_data)
            status = "✅" if has_llm_data else "❌"
            
            if has_llm_data:
                switch_rate = player_data['switchLLM'].mean()
                print(f"   {status} {player_id} ({category}): {word_count} words, switch rate: {switch_rate:.3f}")
            else:
                print(f"   {status} {player_id} ({category}): {word_count} words, FAILED")
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n🎉 Processing completed successfully!")
    print_separator()

if __name__ == "__main__":
    main()