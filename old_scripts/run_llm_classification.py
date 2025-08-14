#!/usr/bin/env python3
"""
LLM Switch Classification Script
Runs the complete pipeline with automatic polling and progress monitoring.
"""

import os
import sys
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Import your classifier
from llm_switch_classifier import LLMSwitchClassifier

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
    """Find the filtered data CSV file."""
    current_dir = Path(".")
    
    # Common file name patterns
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

def poll_batch_status(classifier, batch_id, check_interval=30):
    """
    Poll batch status until completion with progress updates.
    
    Args:
        classifier: LLMSwitchClassifier instance
        batch_id: Batch ID to monitor
        check_interval: Seconds between status checks
    
    Returns:
        Final batch object when processing is complete
    """
    print(f"🔄 Starting automatic polling (checking every {check_interval}s)")
    start_time = time.time()
    check_count = 0
    
    while True:
        check_count += 1
        current_time = time.time()
        elapsed = current_time - start_time
        
        try:
            batch = classifier.client.messages.batches.retrieve(batch_id)
            
            # Calculate progress
            counts = batch.request_counts
            total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
            completed = counts.succeeded + counts.errored + counts.canceled + counts.expired
            progress_pct = (completed / total * 100) if total > 0 else 0
            
            # Print status update
            print(f"\n📊 Status Check #{check_count} (after {format_duration(elapsed)})")
            print(f"Status: {batch.processing_status}")
            print(f"Progress: {completed}/{total} ({progress_pct:.1f}%)")
            print(f"  ✅ Succeeded: {counts.succeeded}")
            print(f"  🔄 Processing: {counts.processing}")
            print(f"  ❌ Errored: {counts.errored}")
            print(f"  🚫 Canceled: {counts.canceled}")
            print(f"  ⏰ Expired: {counts.expired}")
            
            # Check if complete
            if batch.processing_status == "ended":
                print(f"\n🎉 Batch processing completed after {format_duration(elapsed)}!")
                
                # Final summary
                if counts.errored > 0 or counts.expired > 0 or counts.canceled > 0:
                    print(f"⚠️ Note: {counts.errored + counts.expired + counts.canceled} requests failed")
                
                return batch
                
            elif batch.processing_status in ["canceled", "failed"]:
                print(f"\n❌ Batch processing failed with status: {batch.processing_status}")
                raise Exception(f"Batch failed with status: {batch.processing_status}")
            
            # Estimate remaining time
            if completed > 0 and counts.processing > 0:
                avg_time_per_request = elapsed / completed
                estimated_remaining = avg_time_per_request * counts.processing
                print(f"📈 Estimated time remaining: {format_duration(estimated_remaining)}")
            
            print(f"⏳ Next check in {check_interval}s...")
            
        except Exception as e:
            print(f"❌ Error checking batch status: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)
            continue
        
        time.sleep(check_interval)

def main():
    """Main execution function."""
    print_separator()
    print("🚀 LLM Switch Classification Pipeline")
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
        
        # Count player-category combinations
        player_category_combos = data.groupby(['playerID', 'category']).size().shape[0]
        print(f"   Player-category combinations: {player_category_combos}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Step 3: Initialize classifier
    print("\n🤖 Step 3: Initializing LLM classifier...")
    classifier = LLMSwitchClassifier(api_key=api_key)
    print("✅ Classifier initialized")
    
    # Step 4: Prepare batch requests
    print("\n📝 Step 4: Preparing batch requests...")
    try:
        requests = classifier.prepare_batch_requests(data)
        print(f"✅ Prepared {len(requests)} batch requests")
        
        # Show sample request IDs
        print("Sample request IDs:")
        for i, req in enumerate(requests[:3]):
            print(f"   {i+1}. {req['custom_id']}")
        if len(requests) > 3:
            print(f"   ... and {len(requests) - 3} more")
            
    except Exception as e:
        print(f"❌ Error preparing requests: {e}")
        sys.exit(1)
    
    # Step 4.5: Dry run testing
    print("\n🧪 Step 4.5: Running dry run tests...")
    try:
        # Test with 2 sample requests to validate format
        dry_run_success = classifier.dry_run_test(data, num_tests=min(2, len(requests)))
        
        if not dry_run_success:
            print("❌ Dry run tests failed!")
            print("Please check your data format and prompts before proceeding.")
            
            # Ask user if they want to continue anyway
            response = input("\nDo you want to continue with the full batch anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("Exiting due to dry run failure.")
                sys.exit(1)
            else:
                print("⚠️ Proceeding despite dry run failure...")
        else:
            print("✅ Dry run successful - format validated!")
            
    except Exception as e:
        print(f"⚠️ Dry run testing failed with error: {e}")
        print("This might be a temporary issue. Continuing with batch submission...")
    
    # Step 5: Submit batch
    print("\n📤 Step 5: Submitting batch to Anthropic...")
    try:
        batch_id = classifier.create_batch(requests)
        print(f"✅ Batch submitted successfully!")
        print(f"   Batch ID: {batch_id}")
        
        # Save batch ID for recovery
        with open(f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
            f.write(f"Batch ID: {batch_id}\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write(f"Data file: {data_file}\n")
            f.write(f"Total requests: {len(requests)}\n")
        
    except Exception as e:
        print(f"❌ Error submitting batch: {e}")
        sys.exit(1)
    
    # Step 6: Monitor batch with polling
    print("\n⏱️ Step 6: Monitoring batch progress...")
    try:
        final_batch = poll_batch_status(classifier, batch_id, check_interval=30)
    except Exception as e:
        print(f"❌ Error during batch monitoring: {e}")
        print(f"You can manually check batch status with ID: {batch_id}")
        sys.exit(1)
    
    # Step 7: Process results
    print("\n🔄 Step 7: Processing results...")
    try:
        result_data = classifier.process_batch_results(batch_id, data, requests)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"data_with_llm_switches_{timestamp}.csv"
        
        result_data.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error processing results: {e}")
        print(f"You can manually process results later with batch ID: {batch_id}")
        sys.exit(1)
    
    # Step 8: Final analysis
    print("\n📈 Step 8: Analysis summary...")
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
            
            # Calculate agreement
            both_exist = result_data[['switch', 'switchLLM']].notna().all(axis=1)
            if both_exist.sum() > 0:
                agreement = (result_data.loc[both_exist, 'switch'] == result_data.loc[both_exist, 'switchLLM']).mean()
                print(f"Human-LLM agreement: {agreement:.3f}")
        
        # Per-category breakdown
        print("\nPer-category analysis:")
        category_stats = result_data.groupby('category').agg({
            'switchLLM': ['count', 'mean']
        }).round(3)
        
        for category in result_data['category'].unique():
            cat_data = result_data[result_data['category'] == category]
            count = cat_data['switchLLM'].count()
            rate = cat_data['switchLLM'].mean()
            print(f"  {category}: {count:,} words, {rate:.3f} switch rate")
        
    except Exception as e:
        print(f"⚠️ Error in final analysis: {e}")
        print("Results were saved successfully, but analysis summary failed.")
    
    # Completion
    print_separator()
    print("🎉 Pipeline completed successfully!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_file}")
    print_separator()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user")
        print("Any submitted batch will continue processing on Anthropic's servers.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)