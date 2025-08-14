#!/usr/bin/env python3
"""
LLM Switch Classification with Dictionary Response Format
Modified version that uses index-word-switch dictionary format to prevent counting errors.
"""

import pandas as pd
import numpy as np
import anthropic
import json
import time
from typing import List, Dict, Any
from datetime import datetime

class LLMSwitchClassifierDict:
    """
    LLM-based switch classification using dictionary response format.
    This format should prevent the LLM from accidentally skipping words.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the classifier with Anthropic client."""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        
    def create_switch_identification_prompt(self, word_sequence: List[str], category: str) -> str:
        """
        Create a prompt for switch identification using dictionary format.
        
        Args:
            word_sequence: Complete list of words from a player sequence (including word 0)
            category: Category (animals, clothing items, supermarket items)
            
        Returns:
            Formatted prompt string
        """
        
        # Format the complete word sequence
        formatted_words = "\n".join([f"{i}. {word}" for i, word in enumerate(word_sequence)])
        
        prompt = f"""You are participating in a verbal fluency experiment. You will see a sequence of words from the category "{category}" that were produced by another participant.

Your task is to identify which words start new groups of related words. This is similar to how people naturally cluster related words together when naming items from a category.

Instructions:
- Look at the complete sequence of words below
- For each word, decide if it starts a new thematic group (1) or continues the current group (0)
- Consider the context of all previous words when making your decision
- If you feel several words in a row are not related to each other, you can mark multiple consecutive words as starting new groups
- There is no right or wrong answer - people have different opinions about how words are related

Word sequence from category "{category}":
{formatted_words}

IMPORTANT: You must classify ALL {len(word_sequence)} words. Do not skip any words.

Respond with a JSON object containing a "word_classifications" array where each element is a dictionary with the word index, the word itself (as written above), and your switch classification.

Example format for a 4-word sequence:
{{
    "word_classifications": [
        {{"index": 0, "word": "first_word", "switch": 1}},
        {{"index": 1, "word": "second_word", "switch": 0}},
        {{"index": 2, "word": "third_word", "switch": 1}},
        {{"index": 3, "word": "fourth_word", "switch": 0}}
    ]
}}

Your response must include exactly {len(word_sequence)} word classifications, covering indices 0 through {len(word_sequence)-1}.
Each "switch" value should be either 1 (starts new group) or 0 (continues current group).

Your response must be valid JSON only. Do not include any other text or explanation."""

        return prompt
    
    def dry_run_test(self, data: pd.DataFrame, num_tests: int = 2) -> bool:
        """
        Test a few requests with the regular Messages API before submitting the batch.
        """
        print(f"🧪 Running dry run with {num_tests} test requests...")
        
        # Get first few player sequences
        test_sequences = []
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            if len(test_sequences) >= num_tests:
                break
                
            sequence_data = group.sort_values('word_index')
            word_sequence = sequence_data['text'].tolist()
            
            if len(word_sequence) > 0:
                test_sequences.append({
                    'player_id': player_id,
                    'category': category,
                    'word_sequence': word_sequence,
                    'expected_length': len(word_sequence)
                })
        
        if len(test_sequences) == 0:
            print("❌ No valid test sequences found")
            return False
        
        # Test each sequence
        for i, test_seq in enumerate(test_sequences):
            print(f"\n🔍 Testing request {i+1}/{len(test_sequences)}: {test_seq['player_id']} ({test_seq['category']})")
            print(f"   Expected classifications: {test_seq['expected_length']}")
            
            try:
                # Create prompt
                prompt = self.create_switch_identification_prompt(test_seq['word_sequence'], test_seq['category'])
                
                # Make the API call
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,  # Increased for dictionary format
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                response_text = response.content[0].text
                
                # Try to parse JSON response
                try:
                    # Clean response text
                    response_text = response_text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    response_data = json.loads(response_text)
                    
                    # Check for word_classifications
                    if 'word_classifications' not in response_data:
                        print(f"   ❌ Missing 'word_classifications' key")
                        return False
                    
                    classifications = response_data['word_classifications']
                    
                    if len(classifications) != test_seq['expected_length']:
                        print(f"   ❌ Length mismatch: expected {test_seq['expected_length']}, got {len(classifications)}")
                        return False
                    
                    # Verify all indices are present and correct
                    expected_indices = set(range(test_seq['expected_length']))
                    actual_indices = set(item['index'] for item in classifications)
                    
                    if expected_indices != actual_indices:
                        missing = expected_indices - actual_indices
                        extra = actual_indices - expected_indices
                        print(f"   ❌ Index mismatch. Missing: {missing}, Extra: {extra}")
                        return False
                    
                    # Verify words match
                    word_mismatches = []
                    for item in classifications:
                        idx = item['index']
                        expected_word = test_seq['word_sequence'][idx]
                        actual_word = item['word']
                        if expected_word != actual_word:
                            word_mismatches.append(f"Index {idx}: expected '{expected_word}', got '{actual_word}'")
                    
                    if word_mismatches:
                        print(f"   ❌ Word mismatches: {word_mismatches[:3]}...")  # Show first 3
                        return False
                    
                    # Calculate switch rate
                    switches = [item['switch'] for item in classifications]
                    switch_rate = sum(switches) / len(switches)
                    
                    print(f"   ✅ Test {i+1} passed")
                    print(f"     Classifications: {len(classifications)}")
                    print(f"     Switch rate: {switch_rate:.3f}")
                    print(f"     Sample switches: {switches[:5]}")
                    
                except json.JSONDecodeError as e:
                    print(f"   ❌ Invalid JSON response: {e}")
                    print(f"   Response preview: {response_text[:200]}...")
                    return False
                    
            except Exception as e:
                print(f"   ❌ API call failed: {e}")
                return False
        
        print(f"\n✅ All {len(test_sequences)} dry run tests passed!")
        print("🚀 Ready to submit full batch")
        return True

    def prepare_batch_requests(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Prepare batch requests for all unique playerID sequences.
        """
        requests = []
        request_index = 0
        
        # Group by playerID to get complete sequences
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            # Sort by word_index to ensure correct order
            sequence_data = group.sort_values('word_index')
            
            # Extract complete word sequence (including word 0)
            complete_word_sequence = sequence_data['text'].tolist()
            
            # Skip if empty sequence
            if not complete_word_sequence:
                continue
                
            # Create prompt with complete sequence
            prompt = self.create_switch_identification_prompt(complete_word_sequence, category)
            
            # Create shorter custom_id to stay under 64-character limit
            custom_id = f"switch_dict_{request_index:04d}"
            
            # Create batch request
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": self.model,
                    "max_tokens": 1500,  # Increased for dictionary format
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                # Store metadata for result processing
                "metadata": {
                    "player_id": player_id,
                    "category": category,
                    "total_words": len(sequence_data),
                    "sequence_length": len(complete_word_sequence),
                    "word_sequence": complete_word_sequence,  # Store for verification
                    "request_index": request_index
                }
            }
            
            requests.append(request)
            request_index += 1
            
        print(f"Prepared {len(requests)} batch requests for switch classification (dictionary format)")
        return requests
    
    def create_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Create a batch with the prepared requests."""
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
        
        print(f"Created batch with ID: {message_batch.id}")
        print(f"Processing status: {message_batch.processing_status}")
        
        return message_batch.id
    
    def monitor_batch(self, batch_id: str, check_interval: int = 30) -> Dict[str, Any]:
        """Monitor batch processing status and wait for completion."""
        print(f"Monitoring batch {batch_id}...")
        
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            
            print(f"Status: {batch.processing_status}")
            print(f"Request counts: {batch.request_counts}")
            
            if batch.processing_status == "ended":
                print("Batch processing completed!")
                return batch
            elif batch.processing_status in ["canceled", "failed"]:
                raise Exception(f"Batch processing failed with status: {batch.processing_status}")
            
            time.sleep(check_interval)
    
    def process_batch_results(self, batch_id: str, original_data: pd.DataFrame, requests_metadata: List[Dict]) -> pd.DataFrame:
        """
        Process batch results and add LLM classifications to the original data.
        """
        print(f"Processing results from batch {batch_id} (dictionary format)...")
        
        # Create metadata lookup
        metadata_lookup = {req["custom_id"]: req["metadata"] for req in requests_metadata}
        
        # Start with copy of original data
        result_data = original_data.copy()
        
        # Initialize switchLLM column if it doesn't exist
        if 'switchLLM' not in result_data.columns:
            result_data['switchLLM'] = None
        
        successful_results = 0
        failed_results = 0
        
        # Process each result
        for result in self.client.messages.batches.results(batch_id):
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
                    
                    # Extract word classifications from response
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
                    
                    # Verify indices and words match
                    classification_errors = []
                    switch_values = [None] * expected_length  # Initialize array
                    
                    for item in classifications:
                        idx = item.get('index')
                        word = item.get('word')
                        switch = item.get('switch')
                        
                        # Check index is valid
                        if idx is None or idx < 0 or idx >= expected_length:
                            classification_errors.append(f"Invalid index: {idx}")
                            continue
                        
                        # Check word matches expected
                        if word != expected_word_sequence[idx]:
                            classification_errors.append(f"Word mismatch at {idx}: expected '{expected_word_sequence[idx]}', got '{word}'")
                            continue
                        
                        # Check switch value is valid
                        if switch not in [0, 1]:
                            classification_errors.append(f"Invalid switch value at {idx}: {switch}")
                            continue
                        
                        switch_values[idx] = switch
                    
                    if classification_errors:
                        print(f"  ❌ Classification errors for {player_id}: {classification_errors[:3]}...")
                        failed_results += 1
                        continue
                    
                    # Check for any missing classifications
                    if None in switch_values:
                        missing_indices = [i for i, val in enumerate(switch_values) if val is None]
                        print(f"  ❌ Missing classifications for {player_id} at indices: {missing_indices}")
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
                    
                    # Update the DataFrame with final classifications
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
            
            # Calculate agreement on rows where both exist
            both_exist = result_data[['switch', 'switchLLM']].notna().all(axis=1)
            if both_exist.sum() > 0:
                agreement = (result_data.loc[both_exist, 'switch'] == result_data.loc[both_exist, 'switchLLM']).mean()
                print(f"- Human-LLM agreement: {agreement:.3f}")
        
        return result_data

    def run_switch_classification(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete switch classification pipeline with batching.
        """
        print("Starting LLM switch classification with dictionary format...")
        
        # Step 1: Prepare batch requests
        requests = self.prepare_batch_requests(data)
        
        # Step 2: Run dry run test
        if not self.dry_run_test(data, num_tests=min(2, len(requests))):
            print("❌ Dry run failed. Please check your data and try again.")
            return data.copy()
        
        # Step 3: Create batch
        batch_id = self.create_batch(requests)
        
        # Step 4: Monitor batch completion
        final_batch = self.monitor_batch(batch_id)
        
        # Step 5: Process results
        result_data = self.process_batch_results(batch_id, data, requests)
        
        # Step 6: Convert switchLLM to int format matching original switch column
        result_data['switchLLM'] = pd.to_numeric(result_data['switchLLM'], errors='coerce').astype('Int64')
        
        print("LLM switch classification completed!")
        return result_data


def main():
    """
    Example usage of the Dictionary-based LLM Switch Classifier
    """
    
    # Initialize classifier
    classifier = LLMSwitchClassifierDict()
    
    print("Dictionary-based LLM Switch Classifier")
    print("This version uses index-word-switch dictionaries to prevent counting errors.")
    print("\nTo run with your data:")
    print("1. Load your filtered_data_for_analysis.csv")
    print("2. Ensure it has columns: playerID, text, word_index, category")
    print("3. Set your Anthropic API key")
    print("4. Run: result_data = classifier.run_switch_classification(data)")


if __name__ == "__main__":
    main()