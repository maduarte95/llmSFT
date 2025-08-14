#!/usr/bin/env python3
"""
Anthropic API provider implementation for LLM-based switch classification and group labeling.
Wraps existing functionality to work with the new base class architecture.
"""

import anthropic
import pandas as pd
import json
import time
from typing import List, Dict, Any
from base_classifiers import BaseLLMProvider, BaseSwitchClassifier, BaseGroupLabeler


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider implementation."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        super().__init__(api_key, model)
    
    def initialize_client(self):
        """Initialize the Anthropic client."""
        self._client = anthropic.Anthropic(api_key=self.api_key)
    
    def create_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Create a batch with the prepared requests using Anthropic Batch API."""
        print("Creating Anthropic batch...")
        
        # Convert to the format expected by the API
        api_requests = []
        for req in requests:
            api_request = anthropic.types.messages.batch_create_params.Request(
                custom_id=req["custom_id"],
                params=anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
                    model=req["params"]["model"],
                    max_tokens=req["params"]["max_tokens"],
                    messages=req["params"]["messages"]
                )
            )
            api_requests.append(api_request)
        
        # Create the batch
        message_batch = self.client.messages.batches.create(requests=api_requests)
        
        print(f"Created Anthropic batch with ID: {message_batch.id}")
        print(f"Processing status: {message_batch.processing_status}")
        
        return message_batch.id
    
    def monitor_batch(self, batch_id: str, check_interval: int = 30) -> Dict[str, Any]:
        """Monitor batch processing status and wait for completion."""
        print(f"Monitoring Anthropic batch {batch_id}...")
        
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            
            print(f"Status: {batch.processing_status}")
            print(f"Request counts: {batch.request_counts}")
            
            if batch.processing_status == "ended":
                print("Anthropic batch processing completed!")
                return batch
            elif batch.processing_status in ["canceled", "failed"]:
                raise Exception(f"Batch processing failed with status: {batch.processing_status}")
            
            time.sleep(check_interval)
    
    def process_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Process batch results and return list of results."""
        results = []
        for result in self.client.messages.batches.results(batch_id):
            results.append(result)
        return results
    
    def make_single_request(self, messages: List[Dict[str, str]], max_tokens: int = 1500) -> str:
        """Make a single API request for testing."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages
        )
        return response.content[0].text


class AnthropicSwitchClassifier(BaseSwitchClassifier):
    """Anthropic-based switch classification (wrapping existing functionality)."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        provider = AnthropicProvider(api_key, model)
        super().__init__(provider)
    
    def process_batch_results(self, batch_id: str, original_data: pd.DataFrame, requests_metadata: List[Dict]) -> pd.DataFrame:
        """Process Anthropic batch results and add LLM classifications to the original data."""
        print(f"Processing Anthropic batch results from {batch_id}...")
        
        # Create metadata lookup
        metadata_lookup = {req["custom_id"]: req["metadata"] for req in requests_metadata}
        
        # Start with copy of original data
        result_data = original_data.copy()
        
        # Initialize switchLLM column if it doesn't exist
        if 'switchLLM' not in result_data.columns:
            result_data['switchLLM'] = None
        
        successful_results = 0
        failed_results = 0
        
        # Get results from Anthropic
        batch_results = self.provider.process_batch_results(batch_id)
        
        # Process each result
        for result in batch_results:
            if result.result.type == "succeeded":
                try:
                    # Parse the LLM response
                    response_text = result.result.message.content[0].text
                    
                    # Clean response text (remove potential markdown)
                    response_text = response_text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    response_data = json.loads(response_text)
                    
                    # Get metadata for this request
                    custom_id = result.custom_id
                    metadata = metadata_lookup.get(custom_id, {})
                    player_id = metadata.get('player_id')
                    category = metadata.get('category')
                    expected_word_sequence = metadata.get('word_sequence', [])
                    
                    if not player_id:
                        print(f"  ❌ Missing metadata for {custom_id}")
                        failed_results += 1
                        continue
                    
                    # Extract and validate word classifications
                    if 'word_classifications' not in response_data:
                        print(f"  ❌ Missing 'word_classifications' for {player_id}")
                        failed_results += 1
                        continue
                    
                    classifications = response_data['word_classifications']
                    expected_length = len(expected_word_sequence)
                    
                    if len(classifications) != expected_length:
                        print(f"  ❌ Length mismatch for {player_id}: expected {expected_length}, got {len(classifications)}")
                        failed_results += 1
                        continue
                    
                    # Validate and extract switch values
                    switch_values = [None] * expected_length
                    classification_errors = []
                    
                    for item in classifications:
                        idx = item.get('index')
                        word = item.get('word')
                        switch = item.get('switch')
                        
                        if idx is None or idx < 0 or idx >= expected_length:
                            classification_errors.append(f"Invalid index: {idx}")
                            continue
                        
                        if word != expected_word_sequence[idx]:
                            classification_errors.append(f"Word mismatch at {idx}: expected '{expected_word_sequence[idx]}', got '{word}'")
                            continue
                        
                        if switch not in [0, 1]:
                            classification_errors.append(f"Invalid switch value at {idx}: {switch}")
                            continue
                        
                        switch_values[idx] = switch
                    
                    if classification_errors or None in switch_values:
                        print(f"  ❌ Classification errors for {player_id}: {classification_errors[:3]}...")
                        failed_results += 1
                        continue
                    
                    # Force word 0 to be a switch (1)
                    switch_values[0] = 1
                    
                    # Apply classifications to the data
                    player_mask = (result_data['playerID'] == player_id) & (result_data['category'] == category)
                    player_data = result_data[player_mask].sort_values('word_index')
                    
                    if len(player_data) != expected_length:
                        print(f"  ❌ Data length mismatch for {player_id}: data has {len(player_data)}, expected {expected_length}")
                        failed_results += 1
                        continue
                    
                    # Update the DataFrame with classifications
                    indices = player_data.index
                    result_data.loc[indices, 'switchLLM'] = switch_values
                    
                    successful_results += 1
                    print(f"  ✅ Processed {player_id} ({category}): {len(switch_values)} classifications applied")
                
                except Exception as e:
                    print(f"  ❌ Error processing result for {result.custom_id}: {e}")
                    failed_results += 1
            else:
                print(f"  ❌ Failed result for {result.custom_id}: {result.result}")
                failed_results += 1
        
        print(f"\nClassification summary:")
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


class AnthropicGroupLabeler(BaseGroupLabeler):
    """Anthropic-based group labeling (wrapping existing functionality)."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        provider = AnthropicProvider(api_key, model)
        super().__init__(provider)
    
    def process_batch_results(self, batch_id: str, original_data: pd.DataFrame, requests_metadata: List[Dict]) -> pd.DataFrame:
        """Process Anthropic batch results and add LLM group labels to the original data."""
        print(f"Processing Anthropic group labeling results from batch {batch_id}...")
        
        # Create metadata lookup
        metadata_lookup = {req["custom_id"]: req["metadata"] for req in requests_metadata}
        
        # Start with copy of original data
        result_data = original_data.copy()
        result_data['labelLLM'] = None  # Initialize new column
        
        successful_results = 0
        failed_results = 0
        
        # Get results from Anthropic
        batch_results = self.provider.process_batch_results(batch_id)
        
        # Process each result
        for result in batch_results:
            if result.result.type == "succeeded":
                try:
                    # Parse the LLM response
                    response_text = result.result.message.content[0].text
                    
                    # Clean response text (remove potential markdown)
                    response_text = response_text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    response_data = json.loads(response_text)
                    
                    # Get metadata for this request
                    custom_id = result.custom_id
                    metadata = metadata_lookup.get(custom_id, {})
                    player_id = metadata.get('player_id')
                    category = metadata.get('category')
                    groups = metadata.get('groups', [])
                    
                    if not player_id or not groups:
                        print(f"  ❌ Missing metadata for {custom_id}")
                        failed_results += 1
                        continue
                    
                    # Extract group labels from response
                    group_labels_data = response_data.get('group_labels', [])
                    
                    if len(group_labels_data) != len(groups):
                        print(f"  ❌ Label count mismatch for {player_id}: expected {len(groups)}, got {len(group_labels_data)}")
                        failed_results += 1
                        continue
                    
                    # Map labels to word indices
                    for i, (group, label_data) in enumerate(zip(groups, group_labels_data)):
                        label = label_data.get('label', '')
                        word_indices = group['word_indices']
                        
                        # Apply label to all words in this group
                        mask = (result_data['playerID'] == player_id) & \
                               (result_data['category'] == category) & \
                               (result_data['word_index'].isin(word_indices))
                        
                        result_data.loc[mask, 'labelLLM'] = label
                    
                    successful_results += 1
                    print(f"  ✅ Processed {player_id} ({category}): {len(groups)} groups labeled")
                
                except Exception as e:
                    print(f"  ❌ Error processing result for {result.custom_id}: {e}")
                    failed_results += 1
            else:
                print(f"  ❌ Failed result for {result.custom_id}: {result.result}")
                failed_results += 1
        
        print(f"\nLabeling summary:")
        print(f"- Successful: {successful_results}")
        print(f"- Failed: {failed_results}")
        print(f"- Rows with labels: {result_data['labelLLM'].notna().sum()}")
        
        return result_data