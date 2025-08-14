import pandas as pd
import numpy as np
import anthropic
import json
import time
from typing import List, Dict, Any
from datetime import datetime

class LLMSwitchClassifier:
    """
    LLM-based switch classification for verbal fluency sequences using Anthropic's Batch API.
    Replicates human retrospective switch identification task.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the classifier with Anthropic client."""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        
    def create_switch_identification_prompt(self, word_sequence: List[str], category: str) -> str:
        """
        Create a prompt for switch identification that mirrors human instructions.
        
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
- The first word (word with index 0) is always considered a new group (1)

Word sequence from category "{category}":
{formatted_words}

Respond with a JSON object containing:
- "switches": An array of 0s and 1s for all words (0 through {len(word_sequence)-1}), where 1 indicates the word starts a new group and 0 indicates it continues the current group

The switches array should have exactly {len(word_sequence)} elements (one for each word in the sequence). Make sure the array contains exactly {len(word_sequence)} integers, each either 0 or 1.

Example format:
{{
    "switches": [1, 0, 0, 1, 0, 1, 0, 0]
}}

Your response must be valid JSON only. Do not include any other text."""

        return prompt
    
    def dry_run_test(self, data: pd.DataFrame, num_tests: int = 2) -> bool:
        """
        Test a few requests with the regular Messages API before submitting the batch.
        This validates request format and prompt effectiveness.
        
        Args:
            data: DataFrame with player sequences
            num_tests: Number of test requests to run
            
        Returns:
            True if tests pass, False otherwise
        """
        print(f"🧪 Running dry run with {num_tests} test requests...")
        
        # Prepare a few sample requests
        requests = self.prepare_batch_requests(data)
        if len(requests) == 0:
            print("❌ No requests to test")
            return False
        
        # Test the first few requests
        test_requests = requests[:min(num_tests, len(requests))]
        
        for i, request in enumerate(test_requests):
            print(f"\n🔍 Testing request {i+1}/{len(test_requests)}: {request['custom_id']}")
            
            try:
                # Send request to regular Messages API
                response = self.client.messages.create(
                    model=request['params']['model'],
                    max_tokens=request['params']['max_tokens'],
                    messages=request['params']['messages']
                )
                
                # Basic validation of the response
                response_text = response.content[0].text
                expected_length = request['metadata']['sequence_length']
                
                # Simple validation
                try:
                    # Clean response text
                    cleaned = response_text.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()
                    
                    response_data = json.loads(cleaned)
                    switches = response_data.get('switches', [])
                    
                    if len(switches) == expected_length and all(s in [0, 1] for s in switches):
                        print(f"  ✅ Test {i+1} passed")
                        print(f"     Response length: {len(switches)}")
                        print(f"     Switch rate: {sum(switches)/len(switches):.3f}")
                        print(f"     Sample switches: {switches[:min(5, len(switches))]}")
                    else:
                        print(f"  ❌ Test {i+1} failed - invalid response format")
                        print(f"     Expected length: {expected_length}, got: {len(switches)}")
                        return False
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  ❌ Test {i+1} failed - JSON parsing error: {e}")
                    return False
                    
            except Exception as e:
                print(f"  ❌ Test {i+1} failed with error: {e}")
                return False
        
        print(f"\n✅ All {len(test_requests)} dry run tests passed!")
        print("🚀 Ready to submit full batch")
        return True

    def prepare_batch_requests(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Prepare batch requests for all unique playerID sequences.
        
        Args:
            data: DataFrame with columns including playerID, text, word_index, category
            
        Returns:
            List of batch request objects with metadata for result mapping
        """
        requests = []
        
        # Group by playerID to get complete sequences
        player_groups = data.groupby(['playerID', 'category'])
        
        # Create a mapping for shorter custom_ids (Anthropic has 64-char limit)
        request_index = 0
        
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
            # Format: "switch_req_{index}" (max ~15 chars)
            custom_id = f"switch_req_{request_index:04d}"
            
            # Create batch request
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
                # Store metadata for result processing
                "metadata": {
                    "player_id": player_id,
                    "category": category,
                    "total_words": len(sequence_data),
                    "sequence_length": len(complete_word_sequence),
                    "request_index": request_index
                }
            }
            
            requests.append(request)
            request_index += 1
            
        print(f"Prepared {len(requests)} batch requests for switch classification")
        return requests
    
    def create_batch(self, requests: List[Dict[str, Any]]) -> str:
        """
        Create a batch with the prepared requests.
        
        Args:
            requests: List of batch request objects
            
        Returns:
            Batch ID string
        """
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
        """
        Monitor batch processing status and wait for completion.
        
        Args:
            batch_id: Batch ID to monitor
            check_interval: Seconds between status checks
            
        Returns:
            Final batch object when processing is complete
        """
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
        Word 0 is automatically set to 1 (switch) regardless of LLM output.
        
        Args:
            batch_id: Completed batch ID
            original_data: Original DataFrame to add results to
            requests_metadata: Metadata from the original requests for mapping
            
        Returns:
            DataFrame with added LLM classification columns
        """
        print(f"Processing results from batch {batch_id}...")
        
        # Create metadata lookup for easier access
        metadata_lookup = {req["custom_id"]: req["metadata"] for req in requests_metadata}
        
        # Retrieve results
        results = {}
        successful_results = 0
        failed_results = 0
        
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
                    player_id = metadata.get("player_id")
                    category = metadata.get("category")
                    
                    if player_id and category:
                        results[(player_id, category)] = {
                            'llm_switches': response_data.get('switches', []),
                            'metadata': metadata
                        }
                        successful_results += 1
                    else:
                        print(f"Missing metadata for {custom_id}")
                        failed_results += 1
                    
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Error parsing result for {result.custom_id}: {e}")
                    failed_results += 1
                    continue
                    
            else:
                print(f"Failed result for {result.custom_id}: {result.result.type}")
                failed_results += 1
        
        print(f"Results summary:")
        print(f"- Successfully processed: {successful_results}")
        print(f"- Failed: {failed_results}")
        
        # Add results to original data
        data_with_llm = original_data.copy()
        
        # Initialize LLM columns
        data_with_llm['switchLLM'] = np.nan
        
        # Apply results to each player sequence
        processed_players = 0
        for (player_id, category), result_data in results.items():
            # Get the player's data
            mask = (data_with_llm['playerID'] == player_id) & (data_with_llm['category'] == category)
            player_data = data_with_llm[mask].sort_values('word_index')
            
            if len(player_data) == 0:
                print(f"No data found for player {player_id}, category {category}")
                continue
            
            # Get LLM switches and metadata
            llm_switches = result_data['llm_switches']
            metadata = result_data['metadata']
            
            print(f"Processing player {player_id} ({category}):")
            print(f"  - Total words in data: {len(player_data)}")
            print(f"  - LLM returned {len(llm_switches)} classifications")
            
            # Verify lengths match
            if len(llm_switches) == len(player_data):
                # Force word 0 to be a switch (1), keep LLM results for words 1+
                final_switches = llm_switches.copy()
                final_switches[0] = 1  # Force first word to be a switch
                
                # Update the DataFrame with final classifications
                indices = player_data.index
                data_with_llm.loc[indices, 'switchLLM'] = final_switches
                processed_players += 1
                print(f"  ✅ Successfully applied classifications (word 0 forced to 1)")
            else:
                print(f"  ❌ Length mismatch: Expected {len(player_data)}, got {len(llm_switches)}")
            print()
        
        print(f"Successfully processed {processed_players} player sequences")
        
        return data_with_llm
    
    def run_switch_classification(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete switch classification pipeline with batching.
        
        Args:
            data: DataFrame with columns: playerID, text, word_index, category
            
        Returns:
            DataFrame with added switchLLM column
        """
        print("Starting LLM switch classification with batching...")
        
        # Step 1: Prepare batch requests
        requests = self.prepare_batch_requests(data)
        
        # Step 2: Create batch
        batch_id = self.create_batch(requests)
        
        # Step 3: Monitor batch completion
        final_batch = self.monitor_batch(batch_id)
        
        # Step 4: Process results (pass requests for metadata)
        result_data = self.process_batch_results(batch_id, data, requests)
        
        # Step 5: Convert switchLLM to int format matching original switch column
        result_data['switchLLM'] = pd.to_numeric(result_data['switchLLM'], errors='coerce').astype('Int64')
        
        print("LLM switch classification completed!")
        print(f"Classification summary:")
        print(f"- Total rows: {len(result_data)}")
        print(f"- Rows with LLM classifications: {result_data['switchLLM'].notna().sum()}")
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


def main():
    """
    Example usage of the LLM Switch Classifier
    """
    
    # Initialize classifier
    classifier = LLMSwitchClassifier()
    
    # Load your data
    # data = pd.read_csv("path_to_your_filtered_data_for_analysis.csv")
    
    # For this example, create sample data structure
    sample_data = pd.DataFrame({
        'playerID': ['player1'] * 5 + ['player2'] * 4,
        'text': ['cat', 'dog', 'lion', 'tiger', 'shark', 
                 'apple', 'banana', 'carrot', 'lettuce'],
        'word_index': [0, 1, 2, 3, 4, 0, 1, 2, 3],
        'category': ['animals'] * 5 + ['supermarket items'] * 4,
        'switch': [1, 0, 1, 0, 1, 1, 0, 1, 0],  # Original human classifications
        'switch_ground_truth': [1, 0, 1, 0, 1, 1, 0, 1, 0]
    })
    
    print("Sample data structure:")
    print(sample_data.head())
    
    # Run classification
    # result_data = classifier.run_switch_classification(sample_data)
    
    # Save results
    # result_data.to_csv("data_with_llm_switches.csv", index=False)
    
    print("\nTo run with your actual data:")
    print("1. Load your filtered_data_for_analysis.csv")
    print("2. Ensure it has columns: playerID, text, word_index, category")
    print("3. Set your Anthropic API key")
    print("4. Run: result_data = classifier.run_switch_classification(data)")


if __name__ == "__main__":
    main()