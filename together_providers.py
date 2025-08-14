#!/usr/bin/env python3
"""
TogetherAI API provider implementation for LLM-based switch classification and group labeling.
"""

import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from together import Together
from base_classifiers import BaseLLMProvider, BaseSwitchClassifier, BaseGroupLabeler


class TogetherAIProvider(BaseLLMProvider):
    """TogetherAI API provider implementation."""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        super().__init__(api_key, model)
    
    def initialize_client(self):
        """Initialize the Together API client."""
        self._client = Together(api_key=self.api_key)
    
    def create_batch(self, batch_requests: List[Dict[str, Any]]) -> str:
        """Create a batch with the prepared requests using Together SDK."""
        print("Creating TogetherAI batch...")
        
        # Step 1: Create JSONL content
        jsonl_lines = []
        for req in batch_requests:
            # Convert to TogetherAI batch format
            batch_request = {
                "custom_id": req["custom_id"],
                "body": {
                    "model": req["params"]["model"],
                    "max_tokens": req["params"]["max_tokens"],
                    "messages": req["params"]["messages"],
                    "response_format": {
                        "type": "json_object"
                    }
                }
            }
            # Add other parameters from config
            for param_key, param_value in req["params"].items():
                if param_key not in ["model", "max_tokens", "messages"]:
                    batch_request["body"][param_key] = param_value
            
            jsonl_lines.append(json.dumps(batch_request))
        
        jsonl_content = "\n".join(jsonl_lines)
        
        print(f"Content length: {len(jsonl_content)} characters")
        print(f"First line preview: {jsonl_lines[0][:100]}...")
        
        # Step 2: Create temporary file and upload using Together SDK
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
            temp_file.write(jsonl_content)
            temp_file_path = temp_file.name
        
        try:
            # Upload file using Together SDK
            print(f"Uploading file using Together SDK...")
            file_resp = self.client.files.upload(file=temp_file_path, purpose="batch-api")
            file_id = file_resp.id
            print(f"✅ Uploaded batch file with ID: {file_id}")
            
            # Step 3: Create batch using Together SDK
            print("Creating batch...")
            batch_resp = self.client.batches.create_batch(
                file_id,
                endpoint="/v1/chat/completions"
            )
            
            batch_id = batch_resp.id
            print(f"✅ Created TogetherAI batch with ID: {batch_id}")
            print(f"Status: {batch_resp.status}")
            
            return batch_id
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def monitor_batch(self, batch_id: str, check_interval: int = 30) -> Dict[str, Any]:
        """Monitor batch processing status and wait for completion."""
        print(f"Monitoring TogetherAI batch {batch_id}...")
        
        while True:
            batch_info = self.client.batches.get_batch(batch_id)
            status = batch_info.status
            
            print(f"Status: {status}")
            
            # Print request counts if available
            if hasattr(batch_info, 'request_counts'):
                print(f"Request counts: {batch_info.request_counts}")
            
            # Handle both string and enum status values
            status_str = str(status).lower()
            
            if "completed" in status_str:
                print("TogetherAI batch processing completed!")
                return batch_info.__dict__
            elif any(failed_status in status_str for failed_status in ["failed", "cancelled", "expired"]):
                raise Exception(f"Batch processing failed with status: {status}")
            
            time.sleep(check_interval)
    
    def process_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Process batch results and return list of results."""
        # Get batch info to find output file
        batch_info = self.client.batches.get_batch(batch_id)
        output_file_id = batch_info.output_file_id
        
        if not output_file_id:
            raise Exception("No output file ID found in batch info")
        
        # Download results file using Together SDK
        self.client.files.retrieve_content(id=output_file_id, output="temp_batch_output.jsonl")
        
        # Read the downloaded file
        with open("temp_batch_output.jsonl", "r") as f:
            file_content = f.read()
        
        # Clean up temp file
        import os
        os.remove("temp_batch_output.jsonl")
        
        # Parse JSONL results
        results = []
        for line in file_content.strip().split('\n'):
            if line.strip():
                result = json.loads(line)
                results.append(result)
        
        return results
    
    def make_single_request(self, messages: List[Dict[str, str]], max_tokens: int = 1500) -> str:
        """Make a single API request for testing."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            response_format={
                "type": "json_object"
            }
        )
        
        return response.choices[0].message.content


class TogetherAISwitchClassifier(BaseSwitchClassifier):
    """TogetherAI-based switch classification."""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        provider = TogetherAIProvider(api_key, model)
        super().__init__(provider)
    
    def create_switch_identification_prompt(self, word_sequence: List[str], category: str) -> str:
        """Create a prompt for switch identification using TogetherAI JSON mode."""
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

You must respond with ONLY a JSON object containing a "word_classifications" array where each element is a dictionary with the word index, the word itself (EXACTLY as written above - no corrections), and your switch classification.

The response must include exactly {len(word_sequence)} word classifications, covering indices 0 through {len(word_sequence)-1}. Do NOT change the order of words or indices. Do NOT skip indices.
Each "switch" value should be either 1 (starts new group) or 0 (continues current group).


Only respond in JSON format. Do not include any other text or explanation."""
        
        return prompt
    
    def process_batch_results(self, batch_id: str, original_data: pd.DataFrame, requests_metadata: List[Dict]) -> pd.DataFrame:
        """Process TogetherAI batch results and add LLM classifications to the original data."""
        print(f"Processing TogetherAI batch results from {batch_id}...")
        
        # Create metadata lookup
        metadata_lookup = {req["custom_id"]: req["metadata"] for req in requests_metadata}
        
        # Start with copy of original data
        result_data = original_data.copy()
        
        # Initialize switchLLM column if it doesn't exist
        if 'switchLLM' not in result_data.columns:
            result_data['switchLLM'] = None
        
        successful_results = 0
        failed_results = 0
        
        # Get results from TogetherAI
        batch_results = self.provider.process_batch_results(batch_id)
        
        # Process each result
        for result in batch_results:
            custom_id = result.get("custom_id")
            
            if "response" in result and result["response"]["status_code"] == 200:
                try:
                    # Parse the LLM response
                    response_body = result["response"]["body"]
                    response_text = response_body["choices"][0]["message"]["content"]
                    
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
                    print(f"  ❌ Error processing result for {custom_id}: {e}")
                    failed_results += 1
            else:
                print(f"  ❌ Failed result for {custom_id}: {result}")
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


class TogetherAIGroupLabeler(BaseGroupLabeler):
    """TogetherAI-based group labeling."""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        provider = TogetherAIProvider(api_key, model)
        super().__init__(provider)
    
    def create_group_labeling_prompt(self, groups: List[Dict[str, Any]], category: str) -> str:
        """Create a prompt for group labeling using TogetherAI JSON mode."""
        formatted_groups = []
        for i, group in enumerate(groups):
            words_str = ", ".join(group['words'])
            formatted_groups.append(f"Group {i+1}: {words_str}")
        
        groups_text = "\n".join(formatted_groups)
        
        prompt = f"""You are participating in a verbal fluency experiment. You will see groups of related words from the category "{category}" that were identified by another participant.

Your task is to provide a short, descriptive label for what you feel each group of words has in common. Think about the theme or concept that connects the words in each group.

Instructions:
- Look at each group of words
- Think about what theme or concept connects these words together  
- Enter a short, descriptive label that best describes what the words have in common
- There might be different strategies to group words together. There is no right or wrong answer

Groups from category "{category}":
{groups_text}

Examples of good labels:
- For "banana, lemon, corn" → "yellow foods" or "yellow items"
- For "strawberry, blackberry, cranberry" → "berries" 
- For "papaya, pear, passion fruit" → "words starting with P" or "tropical fruits"

You must respond with ONLY a JSON object containing labels for each group.
The response should have exactly {len(groups)} group labels (one for each group shown above).

Only respond in JSON format. Do not include any other text."""
        
        return prompt
    
    def process_batch_results(self, batch_id: str, original_data: pd.DataFrame, requests_metadata: List[Dict]) -> pd.DataFrame:
        """Process TogetherAI batch results and add LLM group labels to the original data."""
        print(f"Processing TogetherAI group labeling results from batch {batch_id}...")
        
        # Create metadata lookup
        metadata_lookup = {req["custom_id"]: req["metadata"] for req in requests_metadata}
        
        # Start with copy of original data
        result_data = original_data.copy()
        result_data['labelLLM'] = None  # Initialize new column
        
        successful_results = 0
        failed_results = 0
        
        # Get results from TogetherAI
        batch_results = self.provider.process_batch_results(batch_id)
        
        # Process each result
        for result in batch_results:
            custom_id = result.get("custom_id")
            
            if "response" in result and result["response"]["status_code"] == 200:
                try:
                    # Parse the LLM response
                    response_body = result["response"]["body"]
                    response_text = response_body["choices"][0]["message"]["content"]
                    
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
                    print(f"  ❌ Error processing result for {custom_id}: {e}")
                    failed_results += 1
            else:
                print(f"  ❌ Failed result for {custom_id}: {result}")
                failed_results += 1
        
        print(f"\nLabeling summary:")
        print(f"- Successful: {successful_results}")
        print(f"- Failed: {failed_results}")
        print(f"- Rows with labels: {result_data['labelLLM'].notna().sum()}")
        
        return result_data