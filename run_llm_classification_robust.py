#!/usr/bin/env python3
"""
Robust LLM Switch Classification Script
Eliminates word reordering issues by having LLM output only switch values without repeating words.
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
from anthropic_providers import AnthropicSwitchClassifier, AnthropicProvider
from together_providers import TogetherAISwitchClassifier, TogetherAIProvider

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
    """Find the filtered data CSV file using flexible selection."""
    return get_data_file_for_script("classification")


class RobustConfigurableClassifier:
    """Robust classifier that shows full sequence but classifies one word at a time."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        
        # Check if IRT timing should be included (backward compatible)
        self.include_irt = getattr(config, 'include_irt', False)
        
        # Create base classifier
        if config.provider == "anthropic":
            self.classifier = AnthropicSwitchClassifier(api_key=api_key, model=config.model)
        elif config.provider == "together":
            self.classifier = TogetherAISwitchClassifier(api_key=api_key, model=config.model)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        # Apply custom parameters
        self.classifier.provider.default_params = config.parameters
    
    def create_single_word_classification_data(self, data: pd.DataFrame):
        """Create requests for each word to be classified individually (except word 0)."""
        requests_data = []
        request_index = 0
        
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            sequence_data = group.sort_values('word_index')
            words = sequence_data['text'].tolist()
            
            # Extract IRT values if available and requested
            irt_values = None
            if self.include_irt and 'irt' in sequence_data.columns:
                irt_values = sequence_data['irt'].tolist()
            
            if len(words) < 2:  # Need at least 2 words, skip word 0
                continue
            
            # Create one request for each word (except word 0 which is always a switch)
            for word_idx in range(1, len(words)):  # Start from 1, skip word 0
                target_word = words[word_idx]
                
                request_data = {
                    'request_id': request_index,
                    'player_id': player_id,
                    'category': category,
                    'full_word_sequence': words,
                    'target_word_index': word_idx,
                    'target_word': target_word,
                    'sequence_length': len(words)
                }
                
                # Add IRT values if available
                if irt_values is not None:
                    request_data['irt_values'] = irt_values
                
                requests_data.append(request_data)
                request_index += 1
        
        return requests_data
    
    def create_robust_switch_prompt(self, full_sequence, category, target_word_index, target_word, irt_values=None):
        """Create a prompt that shows full sequence but asks to classify one specific word."""
        # Format words with optional IRT timing information
        if irt_values is not None and self.include_irt:
            formatted_words = "\n".join([
                f"{i}. {word} ({irt_values[i]:.0f}ms)" if i < len(irt_values) and irt_values[i] is not None 
                else f"{i}. {word}" 
                for i, word in enumerate(full_sequence)
            ])
        else:
            formatted_words = "\n".join([f"{i}. {word}" for i, word in enumerate(full_sequence)])
        
        # Use custom prompt template if available
        if self.config.prompt_template:
            # For custom templates, adapt to single word classification
            template_vars = {
                'category': category,
                'word_sequence': formatted_words,
                'target_word_index': target_word_index,
                'target_word': target_word,
                'num_words': len(full_sequence)
            }
            # Add IRT context for custom templates
            if irt_values is not None and self.include_irt:
                template_vars['irt_context'] = "Each word shows its inter-item response time (IRT) in milliseconds, indicating the time since the previous word."
            else:
                template_vars['irt_context'] = ""
            
            prompt = self.config.prompt_template.format(**template_vars)
            return prompt
        
        # Default robust prompt for single word classification
        # Add IRT timing context if available
        irt_instructions = ""
        if irt_values is not None and self.include_irt:
            irt_instructions = "\n- Each word shows its inter-item response time (IRT) in milliseconds - the time interval since the previous word\n- Longer IRTs (>2000ms) may indicate cognitive switches between thematic groups\n- Consider both semantic relationships AND timing patterns in your decision"
        
        prompt = f"""You are participating in a verbal fluency experiment. You will see a complete sequence of words from the category "{category}" that were produced by another participant.

Your task is to determine if ONE SPECIFIC WORD in this sequence starts a new thematic group or continues the current group.

Instructions:
- Look at the complete sequence of words below
- Consider how people naturally cluster related words together when naming items from a category
- Focus on the relationship between the target word and the words that came before it{irt_instructions}
- Decide if the target word starts a new thematic group (1) or continues the current group (0)

Complete word sequence from category "{category}":
{formatted_words}

TARGET WORD TO CLASSIFY:
Word at position {target_word_index}: "{target_word}"

Question: Does the word "{target_word}" at position {target_word_index} start a NEW thematic group (1) or CONTINUE the current thematic group (0)?

Consider the context of all words that came before position {target_word_index} when making your decision.

RESPONSE FORMAT:
Respond with ONLY a JSON object containing your classification.

{{
  "switch": 0_or_1,
  "reasoning_switch": "brief explanation of your decision"
}}

Your response:"""
        
        return prompt
    
    def prepare_batch_requests(self, data: pd.DataFrame):
        """Prepare batch requests for single-word classification."""
        batch_requests = []
        
        # Get all word classification requests
        word_requests = self.create_single_word_classification_data(data)
        
        for req_data in word_requests:
            # Create prompt for this specific word
            prompt = self.create_robust_switch_prompt(
                req_data['full_word_sequence'],
                req_data['category'],
                req_data['target_word_index'],
                req_data['target_word'],
                req_data.get('irt_values')  # Pass IRT values if available
            )
            
            custom_id = f"single_word_{self.config.name}_{req_data['request_id']:06d}"
            
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
                    "player_id": req_data['player_id'],
                    "category": req_data['category'],
                    "target_word_index": req_data['target_word_index'],
                    "target_word": req_data['target_word'],
                    "full_word_sequence": req_data['full_word_sequence'],
                    "sequence_length": req_data['sequence_length'],
                    "request_id": req_data['request_id'],
                    "config_name": self.config.name,
                    "config_version": self.config.version
                }
            }
            
            batch_requests.append(request)
        
        print(f"Prepared {len(batch_requests)} single-word classification requests using config: {self.config.name}")
        return batch_requests
    
    def process_robust_batch_results(self, batch_id: str, original_data: pd.DataFrame, requests_metadata: list) -> pd.DataFrame:
        """Process batch results from single-word classification."""
        print(f"Processing single-word classification results from {batch_id}...")
        
        # Create metadata lookup
        metadata_lookup = {req["custom_id"]: req["metadata"] for req in requests_metadata}
        
        # Start with copy of original data
        result_data = original_data.copy()
        
        # Initialize switchLLM and reasoning_switch columns if they don't exist
        if 'switchLLM' not in result_data.columns:
            result_data['switchLLM'] = None
        if 'reasoning_switch' not in result_data.columns:
            result_data['reasoning_switch'] = None
        
        # Set word 0 to always be a switch for all players
        for (player_id, category), group in result_data.groupby(['playerID', 'category']):
            first_word_mask = (result_data['playerID'] == player_id) & \
                             (result_data['category'] == category) & \
                             (result_data['word_index'] == 0)
            result_data.loc[first_word_mask, 'switchLLM'] = 1
        
        successful_results = 0
        failed_results = 0
        
        # Get results from provider
        batch_results = self.classifier.provider.process_batch_results(batch_id)
        
        # Process each result
        for result in batch_results:
            # Handle both provider formats: Anthropic (object) vs Together AI (dict)
            if hasattr(result, 'custom_id'):
                # Anthropic format - MessageBatchIndividualResponse object
                custom_id = result.custom_id
                if result.result.type == "succeeded":
                    response_text = result.result.message.content[0].text
                    success = True
                else:
                    success = False
            else:
                # Together AI format - dictionary
                custom_id = result.get("custom_id")
                if "response" in result and result["response"]["status_code"] == 200:
                    response_body = result["response"]["body"]
                    response_text = response_body["choices"][0]["message"]["content"]
                    success = True
                else:
                    success = False
            
            if success:
                try:
                    
                    # Clean and parse JSON response
                    response_text = response_text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    response_data = json.loads(response_text)
                    
                    # Get metadata for this request
                    metadata = metadata_lookup.get(custom_id, {})
                    player_id = metadata.get('player_id')
                    category = metadata.get('category')
                    target_word_index = metadata.get('target_word_index')
                    target_word = metadata.get('target_word')
                    
                    if not player_id or target_word_index is None:
                        print(f"  ❌ Missing metadata for {custom_id}")
                        failed_results += 1
                        continue
                    
                    # Extract single word classification
                    switch_value = response_data.get('switch')
                    reasoning_switch = response_data.get('reasoning_switch', '')
                    
                    # Validate response
                    if switch_value not in [0, 1]:
                        print(f"  ❌ Invalid switch value for {player_id} word {target_word_index}: {switch_value}")
                        failed_results += 1
                        continue
                    
                    # Apply classification to the specific word
                    word_mask = (result_data['playerID'] == player_id) & \
                               (result_data['category'] == category) & \
                               (result_data['word_index'] == target_word_index)
                    
                    if not word_mask.any():
                        print(f"  ❌ No matching row for {player_id} word {target_word_index}")
                        failed_results += 1
                        continue
                    
                    result_data.loc[word_mask, 'switchLLM'] = switch_value
                    result_data.loc[word_mask, 'reasoning_switch'] = reasoning_switch
                    
                    successful_results += 1
                    if successful_results <= 10:  # Show first few for debugging
                        print(f"  ✅ {player_id} word {target_word_index} ('{target_word}'): {switch_value}")
                
                except Exception as e:
                    print(f"  ❌ Error processing result for {custom_id}: {e}")
                    failed_results += 1
            else:
                # Log failed results with provider context
                if hasattr(result, 'custom_id'):
                    print(f"  ❌ Anthropic result failed for {custom_id}: {result.result.type}")
                else:
                    print(f"  ❌ Together AI result failed for {custom_id}")
                failed_results += 1
        
        print(f"\nSingle-word classification summary:")
        print(f"- Successful: {successful_results}")
        print(f"- Failed: {failed_results}")
        print(f"- Total rows: {len(result_data)}")
        print(f"- Rows with LLM classifications: {result_data['switchLLM'].notna().sum()}")
        
        if result_data['switchLLM'].notna().sum() > 0:
            print(f"- LLM switch rate: {result_data['switchLLM'].mean():.3f}")
        
        # Compare with existing switch column if available
        if 'switch' in result_data.columns:
            human_switch_rate = result_data['switch'].mean()
            print(f"- Human switch rate: {human_switch_rate:.3f}")
            
            both_exist = result_data[['switch', 'switchLLM']].notna().all(axis=1)
            if both_exist.sum() > 0:
                agreement = (result_data.loc[both_exist, 'switch'] == result_data.loc[both_exist, 'switchLLM']).mean()
                print(f"- Human-LLM agreement: {agreement:.3f}")
        
        # Convert to int format
        result_data['switchLLM'] = pd.to_numeric(result_data['switchLLM'], errors='coerce').astype('Int64')
        
        return result_data
    
    def run_classification(self, data: pd.DataFrame):
        """Run robust single-word classification."""
        print(f"Starting robust single-word classification with config: {self.config.name}")
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model}")
        print(f"Parameters: {self.config.parameters}")
        
        # Prepare requests for single-word classification
        batch_requests = self.prepare_batch_requests(data)
        
        # Run dry run test (adapted for single-word format)
        if len(batch_requests) > 0:
            print(f"✅ Prepared {len(batch_requests)} single-word classification requests")
        else:
            print("❌ No requests prepared. Please check your data.")
            return data.copy()
        
        # Create batch
        batch_id = self.classifier.provider.create_batch(batch_requests)
        
        # Monitor batch completion
        final_batch = self.classifier.provider.monitor_batch(batch_id)
        
        # Process results with single-word format
        result_data = self.process_robust_batch_results(batch_id, data, batch_requests)
        
        # Add config metadata to results
        config_manager = ConfigManager()
        additional_info = {
            'num_requests': len(batch_requests),
            'num_players': data['playerID'].nunique(),
            'categories': list(data['category'].unique()),
            'data_shape': list(data.shape),
            'single_word_format': True
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
    print("🚀 Robust LLM Switch Classification (Switch-Values-Only Format)")
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
    
    # Step 5: Create robust classifier
    print(f"\n🤖 Step 5: Initializing robust classifier...")
    try:
        classifier = RobustConfigurableClassifier(config, api_key)
        print("✅ Robust classifier initialized with configuration")
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        sys.exit(1)
    
    # Step 6: Run robust classification
    print(f"\n🚀 Step 6: Running robust classification...")
    start_time = time.time()
    
    try:
        result_data, batch_id, additional_info = classifier.run_classification(data)
        
        # Generate output filename with robust indicator
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_config_name = config.name.replace(' ', '_').replace('-', '_')
        output_file = f"data_with_switches_robust_{safe_config_name}_{timestamp}.csv"
        
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
        print(f"  Robust format: Single-word classification")
        print(f"  Processing time: {format_duration(elapsed_time)}")
        
    except Exception as e:
        print(f"⚠️ Error in final analysis: {e}")
    
    # Completion
    print_separator()
    print("🎉 Robust single-word classification pipeline completed successfully!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_file}")
    print(f"Metadata: {Path(output_file).stem}_metadata.json")
    print(f"Configuration: {config.name} (v{config.version}) - Single-Word Format")
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