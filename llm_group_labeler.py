#!/usr/bin/env python3
"""
LLM Group Labeling Script
Labels groups created by the switchLLM column from the LLM switch classification.
Replicates the human labeling task described in LabellingOtherInstructions.
"""

import os
import sys
import pandas as pd
import anthropic
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

class LLMGroupLabeler:
    """
    LLM-based group labeling for verbal fluency sequences using Anthropic's Batch API.
    Replicates human group labeling task from LabellingOther component.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the labeler with Anthropic client."""
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("API key not found. Set ANTHROPIC_API_KEY in .env file or pass as parameter.")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
    
    def create_groups_from_switches(self, sequence_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create word groups from switchLLM column.
        
        Args:
            sequence_data: DataFrame for one player, sorted by word_index
            
        Returns:
            List of group dictionaries with words and metadata
        """
        groups = []
        current_group = []
        current_group_start_idx = 0
        
        for idx, row in sequence_data.iterrows():
            word_index = row['word_index']
            word = row['text']
            switch = row['switchLLM']
            
            # If this is a switch (1) and we have a current group, save it and start new one
            if switch == 1 and len(current_group) > 0:
                groups.append({
                    'start_word_index': current_group_start_idx,
                    'end_word_index': word_index - 1,
                    'words': current_group.copy(),
                    'word_indices': list(range(current_group_start_idx, word_index))
                })
                current_group = [word]
                current_group_start_idx = word_index
            else:
                current_group.append(word)
                if word_index == 0:  # First word always starts a group
                    current_group_start_idx = 0
        
        # Add the final group
        if len(current_group) > 0:
            final_end_idx = sequence_data['word_index'].max()
            groups.append({
                'start_word_index': current_group_start_idx,
                'end_word_index': final_end_idx,
                'words': current_group.copy(),
                'word_indices': list(range(current_group_start_idx, final_end_idx + 1))
            })
        
        return groups
    
    def create_group_labeling_prompt(self, groups: List[Dict[str, Any]], category: str) -> str:
        """
        Create a prompt for group labeling that mirrors human instructions.
        
        Args:
            groups: List of word groups from create_groups_from_switches
            category: Category (animals, clothing items, supermarket items)
            
        Returns:
            Formatted prompt string
        """
        # Format groups for display
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

Respond with a JSON object containing labels for each group:
{{
    "group_labels": [
        {{"group_number": 1, "words": ["word1", "word2", "word3"], "label": "descriptive label"}},
        {{"group_number": 2, "words": ["word4", "word5"], "label": "another label"}},
        ...
    ]
}}

The response should have exactly {len(groups)} group labels (one for each group shown above).

Your response must be valid JSON only. Do not include any other text."""

        return prompt
    
    def prepare_batch_requests(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Prepare batch requests for all unique playerID sequences that have switchLLM data.
        
        Args:
            data: DataFrame with switchLLM column from LLM switch classification
            
        Returns:
            List of batch request objects with metadata for result mapping
        """
        requests = []
        request_index = 0
        
        # Group by playerID and category  
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            # Sort by word_index to ensure correct order
            sequence_data = group.sort_values('word_index')
            
            # Skip if no switchLLM data
            if 'switchLLM' not in sequence_data.columns or sequence_data['switchLLM'].isna().all():
                print(f"Skipping player {player_id} ({category}) - no switchLLM data")
                continue
            
            # Create groups from switches
            groups = self.create_groups_from_switches(sequence_data)
            
            if len(groups) == 0:
                print(f"Skipping player {player_id} ({category}) - no groups found")
                continue
            
            # Create prompt
            prompt = self.create_group_labeling_prompt(groups, category)
            
            # Create batch request
            custom_id = f"label_req_{request_index:04d}"
            
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": self.model,
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ]
                },
                "metadata": {
                    "player_id": player_id,
                    "category": category,
                    "num_groups": len(groups),
                    "groups": groups,  # Store groups for result mapping
                    "request_index": request_index
                }
            }
            
            requests.append(request)
            request_index += 1
        
        print(f"Prepared {len(requests)} batch requests for group labeling")
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
        Process batch results and add LLM group labels to the original data.
        
        Args:
            batch_id: Completed batch ID
            original_data: Original DataFrame to add results to
            requests_metadata: Metadata from the original requests for mapping
            
        Returns:
            DataFrame with added labelLLM column
        """
        print(f"Processing group labeling results from batch {batch_id}...")
        
        # Create metadata lookup
        metadata_lookup = {req["custom_id"]: req["metadata"] for req in requests_metadata}
        
        # Start with copy of original data
        result_data = original_data.copy()
        result_data['labelLLM'] = None  # Initialize new column
        
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
    
    def dry_run_test(self, data: pd.DataFrame, num_tests: int = 2) -> bool:
        """
        Test a few requests with the regular Messages API before submitting the batch.
        This validates request format and prompt effectiveness.
        
        Args:
            data: DataFrame to test with
            num_tests: Number of test requests to make
            
        Returns:
            True if tests pass, False otherwise
        """
        print(f"🧪 Running dry run test with {num_tests} requests...")
        
        # Get first few player sequences that have switchLLM data
        test_requests = []
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            if len(test_requests) >= num_tests:
                break
                
            sequence_data = group.sort_values('word_index')
            
            # Skip if no switchLLM data
            if 'switchLLM' not in sequence_data.columns or sequence_data['switchLLM'].isna().all():
                continue
            
            # Create groups from switches
            groups = self.create_groups_from_switches(sequence_data)
            
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
            print("❌ No valid test requests found")
            return False
        
        # Test each request
        for i, test_req in enumerate(test_requests):
            print(f"\n📝 Test {i+1}/{len(test_requests)}: Player {test_req['player_id']} ({test_req['category']})")
            print(f"   Groups to label: {len(test_req['groups'])}")
            
            try:
                # Make the API call
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": test_req['prompt']
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
                    group_labels = response_data.get('group_labels', [])
                    
                    if len(group_labels) == len(test_req['groups']):
                        print(f"   ✅ Valid response with {len(group_labels)} labels")
                        
                        # Show sample labels
                        for j, (group, label_data) in enumerate(zip(test_req['groups'], group_labels[:3])):  # Show first 3
                            words_str = ", ".join(group['words'])
                            label = label_data.get('label', 'NO_LABEL')
                            print(f"      Group {j+1}: {words_str} → '{label}'")
                        
                        if len(group_labels) > 3:
                            print(f"      ... and {len(group_labels) - 3} more groups")
                            
                    else:
                        print(f"   ❌ Label count mismatch: expected {len(test_req['groups'])}, got {len(group_labels)}")
                        return False
                        
                except json.JSONDecodeError as e:
                    print(f"   ❌ Invalid JSON response: {e}")
                    print(f"   Response: {response_text[:200]}...")
                    return False
                    
            except Exception as e:
                print(f"   ❌ API call failed: {e}")
                return False
        
        print(f"\n✅ Dry run successful! All {len(test_requests)} test requests passed.")
        print("🚀 Ready to submit full batch")
        return True

    def run_group_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete group labeling pipeline with batching.
        
        Args:
            data: DataFrame with switchLLM column from switch classification
            
        Returns:
            DataFrame with added labelLLM column
        """
        print("Starting LLM group labeling with batching...")
        
        # Verify switchLLM column exists
        if 'switchLLM' not in data.columns:
            raise ValueError("switchLLM column not found. Please run switch classification first.")
        
        # Step 1: Prepare batch requests
        requests = self.prepare_batch_requests(data)
        
        if len(requests) == 0:
            print("No valid requests to process. Returning original data.")
            result_data = data.copy()
            result_data['labelLLM'] = None
            return result_data
        
        # Step 2: Run dry run test
        print(f"\n📝 Step 2: Running dry run test...")
        if not self.dry_run_test(data, num_tests=min(2, len(requests))):
            print("❌ Dry run failed. Please check your data and try again.")
            return data.copy()
        
        # Step 3: Create batch
        print(f"\n🚀 Step 3: Creating batch...")
        batch_id = self.create_batch(requests)
        
        # Step 4: Monitor batch completion
        print(f"\n⏳ Step 4: Monitoring batch completion...")
        final_batch = self.monitor_batch(batch_id)
        
        # Step 5: Process results
        print(f"\n📊 Step 5: Processing results...")
        result_data = self.process_batch_results(batch_id, data, requests)
        
        print("LLM group labeling completed!")
        return result_data


def main():
    """Main execution function."""
    print("🚀 LLM Group Labeling Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load environment
    try:
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ Error: ANTHROPIC_API_KEY not found in .env file")
            sys.exit(1)
        print("✅ API key loaded successfully")
    except Exception as e:
        print(f"❌ Error loading environment: {e}")
        sys.exit(1)
    
    # Find data file with switchLLM column
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
    else:
        print("📁 Multiple data files found:")
        for i, f in enumerate(data_files):
            print(f"  {i+1}. {f}")
        
        while True:
            try:
                choice = int(input("Enter file number: ")) - 1
                if 0 <= choice < len(data_files):
                    data_file = data_files[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Load data
    try:
        data = pd.read_csv(data_file)
        print(f"✅ Data loaded from: {data_file}")
        print(f"   Shape: {data.shape}")
        print(f"   Unique players: {data['playerID'].nunique()}")
        print(f"   Categories: {list(data['category'].unique())}")
        
        # Check for switchLLM column
        if 'switchLLM' not in data.columns:
            print("❌ Error: switchLLM column not found")
            print("Please run switch classification first using run_llm_classification.py")
            sys.exit(1)
        
        # Count how many players have switchLLM data
        players_with_switches = data.groupby(['playerID', 'category'])['switchLLM'].apply(lambda x: x.notna().any()).sum()
        print(f"   Players with switchLLM data: {players_with_switches}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Initialize labeler
    try:
        labeler = LLMGroupLabeler(api_key=api_key)
        print("✅ Group labeler initialized")
    except Exception as e:
        print(f"❌ Error initializing labeler: {e}")
        sys.exit(1)
    
    # Run labeling
    try:
        result_data = labeler.run_group_labeling(data)
        
        # Save results
        output_file = data_file.stem + "_with_labels.csv"
        result_data.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Show summary
        print("\nFinal Summary:")
        print(f"- Total rows: {len(result_data)}")
        print(f"- Rows with labelLLM: {result_data['labelLLM'].notna().sum()}")
        
        # Show sample labels
        labeled_data = result_data[result_data['labelLLM'].notna()]
        if len(labeled_data) > 0:
            print("\nSample labels:")
            # Group by player and category, then take first row of each group
            sample_groups = labeled_data.groupby(['playerID', 'category'])
            for (player_id, category), group in list(sample_groups)[:3]:  # Show first 3 player-category combinations
                first_row = group.iloc[0]  # Get first row of this group
                print(f"  Player {player_id} ({category}): {first_row['text']} → {first_row['labelLLM']}")
        
    except Exception as e:
        print(f"❌ Error during labeling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()