#!/usr/bin/env python3
"""
Ground Truth-Based Group Labeling Script
Labels groups based on ground truth switches instead of LLM switches.
Uses YAML configuration files for model selection and parameter control.
Supports custom prompt templates and reproducible experiments.
"""

import os
import sys
import pandas as pd
import time
import json
import numpy as np
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

# Import animal categories utility
from animal_categories import load_animal_categories


def load_environment(provider: str) -> str:
    """Load environment variables from .env file."""
    load_dotenv()
    
    if provider == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found in .env file")
            sys.exit(1)
    elif provider == "together":
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            print("Error: TOGETHER_API_KEY not found in .env file")
            sys.exit(1)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    print(f"{provider.title()} API key loaded successfully")
    return api_key


def find_data_file():
    """Find data file with switch_ground_truth column using flexible selection."""
    return get_data_file_for_script("labeling")


def has_complete_ground_truth_data(sequence_data: pd.DataFrame) -> bool:
    """
    Check if a player sequence has complete ground truth data.
    
    Args:
        sequence_data: DataFrame with word sequence data for one player
        
    Returns:
        True if sequence has complete ground truth data, False otherwise
    """
    gt_switches = sequence_data['switch_ground_truth'].copy()
    
    # Convert empty strings to NaN for easier handling
    gt_switches = gt_switches.replace('', np.nan)
    
    # Check for NaN values at positions other than index 0 (index 0 is expected to be NaN)
    non_zero_indices = sequence_data['word_index'] != 0
    missing_gt_data = gt_switches[non_zero_indices].isna()
    
    return not missing_gt_data.any()


def create_ground_truth_groups(sequence_data: pd.DataFrame) -> list:
    """
    Create groups from ground truth switches.
    Assumes sequence has complete ground truth data (checked separately).
    
    Args:
        sequence_data: DataFrame with word sequence data sorted by word_index
        
    Returns:
        List of group dictionaries with expected format for base labeler
    """
    # Convert ground truth switches to numeric, handling index 0 case
    gt_switches = sequence_data['switch_ground_truth'].copy()
    gt_switches = gt_switches.replace('', np.nan)
    gt_switches_numeric = pd.to_numeric(gt_switches, errors='coerce')
    
    # For index 0, treat as start of first group (no switch)
    first_row_mask = sequence_data['word_index'] == 0
    if first_row_mask.any():
        gt_switches_numeric[first_row_mask] = 0.0
    
    # Create groups based on ground truth switches
    groups = []
    current_group_words = []
    current_group_start_idx = None
    
    for word_index, (idx, row) in enumerate(sequence_data.iterrows()):
        word = row['text']
        switch_value = gt_switches_numeric.loc[idx]
        
        # Initialize group start for first word
        if current_group_start_idx is None:
            current_group_start_idx = word_index
        
        # Check if this word starts a new group (switch=1)
        if switch_value == 1.0 and current_group_words:
            # This word starts a new group, so finish the previous group first
            groups.append({
                'start_word_index': current_group_start_idx,
                'end_word_index': word_index - 1,
                'words': current_group_words.copy(),
                'word_indices': list(range(current_group_start_idx, word_index))
            })
            current_group_words = []
            current_group_start_idx = word_index
        
        # Add current word to the current group
        current_group_words.append(word)
    
    # Add final group if there are remaining words
    if current_group_words:
        final_end_idx = len(sequence_data) - 1
        groups.append({
            'start_word_index': current_group_start_idx,
            'end_word_index': final_end_idx,
            'words': current_group_words.copy(),
            'word_indices': list(range(current_group_start_idx, len(sequence_data)))
        })
    
    return groups


class GroundTruthLabeler:
    """Ground truth-based labeler with custom prompt handling and dual provider support."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        
        # Create provider
        if config.provider == "anthropic":
            self.provider = AnthropicProvider(api_key=api_key, model=config.model)
        elif config.provider == "together":
            self.provider = TogetherAIProvider(api_key=api_key, model=config.model)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        # Apply custom parameters
        self.provider.default_params = config.parameters
    
    def create_group_labeling_prompt(self, groups, category):
        """Create prompt using config's custom template with animal categories support."""
        if self.config.prompt_template:
            # Format groups for prompt
            formatted_groups = []
            for i, group in enumerate(groups):
                words_str = ", ".join(group['words'])
                formatted_groups.append(f"Group {i+1}: {words_str}")
            groups_text = "\n".join(formatted_groups)
            
            # Load animal categories if needed
            animal_categories = ""
            if '{animal_categories}' in self.config.prompt_template:
                animal_categories = load_animal_categories()
            
            return self.config.prompt_template.format(
                category=category,
                groups_text=groups_text,
                num_groups=len(groups),
                animal_categories=animal_categories
            )
        else:
            # Use basic default prompt
            formatted_groups = []
            for i, group in enumerate(groups):
                words_str = ", ".join(group['words'])
                formatted_groups.append(f"Group {i+1}: {words_str}")
            groups_text = "\n".join(formatted_groups)
            
            return f"""Analyze these word groups from a verbal fluency task for {category} and provide descriptive labels:

{groups_text}

Please provide a JSON response with the following format:
{{
  "group_labels": [
    {{
      "group_number": 1,
      "words": ["word1", "word2"],
      "label": "descriptive label"
    }}
  ]
}}"""
    
    def prepare_batch_requests(self, data: pd.DataFrame):
        """Prepare batch requests using ground truth switches - ANIMALS ONLY."""
        # Filter for animals category only
        animals_data = data[data['category'] == 'animals'].copy()
        
        if len(animals_data) == 0:
            print("No animals category data found. This script processes animals only.")
            return []
        
        print(f"Processing animals category only: {len(animals_data):,} rows out of {len(data):,} total")
        
        # Check if we'll be using enhanced animal categories
        if self.config.prompt_template and '{animal_categories}' in self.config.prompt_template:
            print("Using enhanced prompts with complete animal categories from animals_by_category.txt")
        
        requests = []
        request_index = 0
        skipped_players = 0
        
        player_groups = animals_data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            sequence_data = group.sort_values('word_index')
            
            # Check if this player has complete ground truth data
            if not has_complete_ground_truth_data(sequence_data):
                print(f"Skipping player {player_id} ({category}) - missing ground truth data")
                skipped_players += 1
                continue
            
            # Create groups from ground truth switches
            groups = create_ground_truth_groups(sequence_data)
            
            if len(groups) == 0:
                print(f"Skipping player {player_id} ({category}) - no valid groups")
                skipped_players += 1
                continue
            
            # Create prompt
            prompt = self.create_group_labeling_prompt(groups, category)
            custom_id = f"gt_label_{self.config.name}_{request_index:04d}"
            
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
                    "config_version": self.config.version,
                    "labeling_type": "ground_truth"
                }
            }
            
            requests.append(request)
            request_index += 1
        
        print(f"Prepared {len(requests)} batch requests using config: {self.config.name}")
        if skipped_players > 0:
            print(f"Skipped {skipped_players} players due to missing ground truth data")
        
        return requests
    
    def dry_run_test(self, data: pd.DataFrame, num_tests: int = 2) -> bool:
        """Run dry run test using ground truth data - ANIMALS ONLY."""
        print(f"Running ground truth dry run test with {num_tests} requests...")
        
        # Filter to animals data
        animals_data = data[data['category'] == 'animals'].copy()
        
        if len(animals_data) == 0:
            print("No animals data found for dry run")
            return False
        
        test_requests = []
        player_groups = animals_data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            if len(test_requests) >= num_tests:
                break
            
            sequence_data = group.sort_values('word_index')
            
            # Check ground truth data
            if not has_complete_ground_truth_data(sequence_data):
                continue
            
            # Create groups
            groups = create_ground_truth_groups(sequence_data)
            
            if len(groups) == 0:
                continue
            
            # Create prompt
            prompt = self.create_group_labeling_prompt(groups, category)
            
            test_requests.append({
                'player_id': player_id,
                'category': category,
                'prompt': prompt,
                'groups': groups
            })
        
        if len(test_requests) == 0:
            print("No valid test requests found")
            return False
        
        # Test each request
        for i, test_req in enumerate(test_requests):
            print(f"\nTest {i+1}/{len(test_requests)}: Player {test_req['player_id']} ({test_req['category']})")
            print(f"   Groups to label: {len(test_req['groups'])}")
            
            try:
                messages = [{"role": "user", "content": test_req['prompt']}]
                response_text = self.provider.make_single_request(messages, max_tokens=1000)
                
                # Parse and validate response
                try:
                    cleaned_text = response_text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:]
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()
                    
                    response_data = json.loads(cleaned_text)
                    group_labels = response_data.get('group_labels', [])
                    
                    if len(group_labels) == len(test_req['groups']):
                        print(f"   Valid response with {len(group_labels)} labels")
                        
                        for j, (group, label_data) in enumerate(zip(test_req['groups'], group_labels[:3])):
                            words_str = ", ".join(group['words'])
                            label = label_data.get('label', 'NO_LABEL')
                            print(f"      Group {j+1}: {words_str} → '{label}'")
                        
                        if len(group_labels) > 3:
                            print(f"      ... and {len(group_labels) - 3} more groups")
                    else:
                        print(f"   Label count mismatch: expected {len(test_req['groups'])}, got {len(group_labels)}")
                        return False
                        
                except json.JSONDecodeError as e:
                    print(f"   Invalid JSON response: {e}")
                    print(f"   Response preview: {response_text[:200]}...")
                    return False
                    
            except Exception as e:
                print(f"   API error: {e}")
                return False
        
        print("Ground truth dry run completed successfully")
        return True
    
    def process_batch_results(self, batch_id: str, original_data: pd.DataFrame, batch_requests: list):
        """Process batch results and add ground truth labels to data."""
        # Get batch results from provider
        if self.config.provider == "anthropic":
            # For Anthropic, get results from the batch
            batch = self.provider.client.messages.batches.retrieve(batch_id)
            results_url = batch.results_url
            
            if not results_url:
                print("No results URL available")
                result_data = original_data.copy()
                result_data['snafu_gt_label'] = None
                return result_data
            
            # Download and parse results
            import requests
            response = requests.get(results_url)
            batch_results = []
            for line in response.text.strip().split('\n'):
                if line.strip():
                    batch_results.append(json.loads(line))
                    
        elif self.config.provider == "together":
            # For Together, get results from the batch
            batch = self.provider.client.batches.retrieve(batch_id)
            output_file_id = batch.output_file_id
            
            if not output_file_id:
                print("No output file available")
                result_data = original_data.copy()
                result_data['snafu_gt_label'] = None
                return result_data
            
            # Download and parse results
            file_content = self.provider.client.files.content(output_file_id)
            batch_results = []
            for line in file_content.decode('utf-8').strip().split('\n'):
                if line.strip():
                    batch_results.append(json.loads(line))
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
        
        # Initialize result data
        result_data = original_data.copy()
        result_data['snafu_gt_label'] = None
        
        # Process each result
        for result in batch_results:
            try:
                custom_id = result.get('custom_id', '')
                
                # Find matching request
                matching_request = None
                for req in batch_requests:
                    if req['custom_id'] == custom_id:
                        matching_request = req
                        break
                
                if not matching_request:
                    print(f"Warning: No matching request found for {custom_id}")
                    continue
                
                player_id = matching_request['metadata']['player_id']
                category = matching_request['metadata']['category']
                groups = matching_request['metadata']['groups']
                
                # Extract response content (provider-specific)
                if self.config.provider == "anthropic":
                    if result.get('result', {}).get('type') == 'succeeded':
                        response_content = result['result']['message']['content'][0]['text']
                    else:
                        print(f"Warning: Failed result for {custom_id}")
                        continue
                elif self.config.provider == "together":
                    if 'response' in result and 'body' in result['response']:
                        response_content = result['response']['body']['choices'][0]['message']['content']
                    else:
                        print(f"Warning: Invalid result format for {custom_id}")
                        continue
                
                # Parse labels
                try:
                    cleaned_text = response_content.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:]
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()
                    
                    response_data = json.loads(cleaned_text)
                    group_labels = response_data.get('group_labels', [])
                    
                    if len(group_labels) != len(groups):
                        print(f"Warning: Label count mismatch for {player_id}")
                        continue
                    
                    # Apply labels to data
                    sequence_data = result_data[
                        (result_data['playerID'] == player_id) & 
                        (result_data['category'] == category)
                    ].sort_values('word_index')
                    
                    # Recreate ground truth groups to match with labels
                    gt_groups = create_ground_truth_groups(sequence_data)
                    
                    # Apply labels to each word in each group
                    word_idx = 0
                    for group_idx, (group, label_info) in enumerate(zip(gt_groups, group_labels)):
                        label = label_info.get('label', f'Group {group_idx + 1}')
                        
                        for word in group['words']:
                            if word_idx < len(sequence_data):
                                row_idx = sequence_data.iloc[word_idx].name
                                result_data.loc[row_idx, 'snafu_gt_label'] = label
                                word_idx += 1
                    
                    print(f"Applied ground truth labels for player {player_id} ({category}) - {len(gt_groups)} groups")
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing response for {custom_id}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error processing result {result.get('custom_id', 'unknown')}: {e}")
                continue
        
        return result_data
    
    def run_labeling(self, data: pd.DataFrame):
        """Run ground truth-based labeling with config parameters and metadata tracking."""
        print(f"Starting ground truth group labeling with config: {self.config.name}")
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model}")
        print(f"Parameters: {self.config.parameters}")
        
        if self.config.prompt_template:
            print("Using custom prompt template from config")
        else:
            print("Using default prompt template")
        
        if 'switch_ground_truth' not in data.columns:
            raise ValueError("switch_ground_truth column not found. Please ensure ground truth switch data is available.")
        
        # Check for ground truth data availability
        gt_data = data['switch_ground_truth'].replace('', np.nan)
        total_rows = len(data)
        missing_gt_count = gt_data.isna().sum()
        
        print(f"Ground truth data availability:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Missing ground truth: {missing_gt_count:,} ({missing_gt_count/total_rows:.1%})")
        
        # Prepare requests with ground truth groups
        batch_requests = self.prepare_batch_requests(data)
        
        if len(batch_requests) == 0:
            print("No valid requests to process. Returning original data.")
            result_data = data.copy()
            result_data['snafu_gt_label'] = None
            return result_data, None, {}
        
        # Run dry run test
        if not self.dry_run_test(data, num_tests=min(2, len(batch_requests))):
            print("Dry run failed. Please check your configuration and data.")
            return data.copy(), None, {}
        
        # Create batch
        batch_id = self.provider.create_batch(batch_requests)
        
        # Monitor batch completion
        final_batch = self.provider.monitor_batch(batch_id)
        
        # Process results
        result_data = self.process_batch_results(batch_id, data, batch_requests)
        
        # Add config metadata info
        additional_info = {
            'num_requests': len(batch_requests),
            'num_players': data['playerID'].nunique(),
            'categories': list(data['category'].unique()),
            'data_shape': list(data.shape),
            'custom_prompt_used': self.config.prompt_template is not None,
            'labeling_type': 'ground_truth',
            'missing_gt_percentage': missing_gt_count/total_rows
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
    print("Ground Truth-Based LLM Group Labeling")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    # Step 1: Load configurations
    print("\nStep 1: Loading configurations...")
    config_manager = ConfigManager()
    
    # Print config summary without emojis to avoid encoding issues
    try:
        config_manager.print_config_summary()
    except UnicodeEncodeError:
        print("Configuration summary (emoji encoding issue avoided):")
        print(f"Total configurations loaded: {len(config_manager.configs)}")
        for name, config in config_manager.configs.items():
            print(f"  - {name} ({config.provider}/{config.model})")
    
    # Step 2: Select configuration
    print("\nStep 2: Configuration selection...")
    
    # Option to specify config via command line
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        config = config_manager.get_config(config_name)
        if not config:
            print(f"Configuration '{config_name}' not found")
            sys.exit(1)
        print(f"Using config from command line: {config_name}")
    else:
        # Interactive selection
        config = config_manager.interactive_config_selector("labels")
        if not config:
            print("No configuration selected")
            sys.exit(1)
    
    # Validate configuration
    issues = config_manager.validate_config(config)
    if issues:
        print(f"Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    print(f"Selected configuration: {config.name}")
    print(f"   Provider: {config.provider}")
    print(f"   Model: {config.model}")
    print(f"   Parameters: {config.parameters}")
    if config.prompt_template:
        print(f"   Custom prompt: Yes ({len(config.prompt_template)} chars)")
    else:
        print(f"   Custom prompt: No (using default)")
    
    # Step 3: Load environment
    print(f"\nStep 3: Loading {config.provider.title()} environment...")
    api_key = load_environment(config.provider)
    
    # Step 4: Load data
    print("\nStep 4: Loading data...")
    data_file = find_data_file()
    
    try:
        data = pd.read_csv(data_file)
        print(f"Data loaded successfully")
        print(f"   Shape: {data.shape}")
        print(f"   Unique players: {data['playerID'].nunique()}")
        print(f"   Categories: {list(data['category'].unique())}")
        
        # Check for animals category
        if 'animals' not in data['category'].values:
            print("Error: No 'animals' category found in data!")
            print("This script is designed to process animals category only.")
            sys.exit(1)
        
        animals_count = len(data[data['category'] == 'animals'])
        print(f"   Animals category rows: {animals_count:,}")
        
        # Check for switch_ground_truth column
        if 'switch_ground_truth' not in data.columns:
            print("Error: switch_ground_truth column not found!")
            print("This script requires ground truth switch data.")
            print("Available columns:", list(data.columns))
            sys.exit(1)
        
        gt_data = data['switch_ground_truth'].replace('', np.nan)
        gt_data_count = gt_data.notna().sum()
        print(f"   Rows with ground truth switches: {gt_data_count:,}")
        
        if gt_data_count == 0:
            print("Error: No ground truth switch data found!")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Step 5: Create ground truth labeler
    print(f"\nStep 5: Initializing ground truth labeler with config...")
    try:
        labeler = GroundTruthLabeler(config, api_key)
        print("Ground truth labeler initialized with configuration")
    except Exception as e:
        print(f"Error initializing labeler: {e}")
        sys.exit(1)
    
    # Step 6: Run ground truth labeling
    print(f"\nStep 6: Running ground truth group labeling...")
    start_time = time.time()
    
    try:
        result_data, batch_id, additional_info = labeler.run_labeling(data)
        
        # Generate output filename with ground truth and animals indicator
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_config_name = config.name.replace(' ', '_').replace('-', '_')
        output_file = f"data_with_ground_truth_labels_animals_{safe_config_name}_{timestamp}.csv"
        
        result_data.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
        # Save metadata
        config_manager.save_run_metadata(
            config=config,
            output_file=output_file,
            batch_id=batch_id,
            additional_info=additional_info
        )
        
        elapsed_time = time.time() - start_time
        print(f"Total processing time: {format_duration(elapsed_time)}")
        
    except Exception as e:
        print(f"Error during labeling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 7: Analysis summary
    print("\nStep 7: Analysis summary...")
    try:
        total_rows = len(result_data)
        rows_with_gt_labels = result_data.get('snafu_gt_label', pd.Series()).notna().sum()
        
        print(f"Total rows: {total_rows:,}")
        print(f"Rows with ground truth labels: {rows_with_gt_labels:,}")
        
        if rows_with_gt_labels > 0:
            # Count unique labels
            if 'snafu_gt_label' in result_data.columns:
                unique_labels = result_data['snafu_gt_label'].dropna().nunique()
                print(f"Unique ground truth labels generated: {unique_labels}")
                
                # Show most common labels
                top_labels = result_data['snafu_gt_label'].value_counts().head(5)
                print(f"\nTop 5 most common ground truth labels:")
                for label, count in top_labels.items():
                    print(f"  '{label}': {count} occurrences")
            else:
                print("Warning: snafu_gt_label column not found")
        
        # Per-category breakdown
        print(f"\nPer-category analysis:")
        for category in result_data['category'].unique():
            cat_data = result_data[result_data['category'] == category]
            if 'snafu_gt_label' in result_data.columns:
                label_count = cat_data['snafu_gt_label'].notna().sum()
            else:
                label_count = 0
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
        print(f"  Labeling type: Ground Truth")
        print(f"  Processing time: {format_duration(elapsed_time)}")
        
    except Exception as e:
        print(f"Error in final analysis: {e}")
    
    # Completion
    print_separator()
    print("Ground truth-based labeling pipeline completed successfully!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_file}")
    print(f"Metadata: {Path(output_file).stem}_metadata.json")
    print(f"Configuration: {config.name} (v{config.version})")
    print_separator()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)