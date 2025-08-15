#!/usr/bin/env python3
"""
Abstract base classes for LLM-based switch classification and group labeling.
Supports multiple API providers (Anthropic, TogetherAI, etc.).
"""

from abc import ABC, abstractmethod
import pandas as pd
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime


class BaseLLMProvider(ABC):
    """Abstract base class for LLM API providers."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    @abstractmethod
    def initialize_client(self):
        """Initialize the API client."""
        pass
    
    @abstractmethod
    def create_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Create a batch with the prepared requests."""
        pass
    
    @abstractmethod
    def monitor_batch(self, batch_id: str, check_interval: int = 30) -> Dict[str, Any]:
        """Monitor batch processing status and wait for completion."""
        pass
    
    @abstractmethod
    def process_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Process batch results and return list of results."""
        pass
    
    @abstractmethod
    def make_single_request(self, messages: List[Dict[str, str]], max_tokens: int = 1500) -> str:
        """Make a single API request for testing."""
        pass
    
    @property
    def client(self):
        """Get initialized client."""
        if self._client is None:
            self.initialize_client()
        return self._client


class BaseSwitchClassifier(ABC):
    """Abstract base class for switch classification."""
    
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
    
    def create_switch_identification_prompt(self, word_sequence: List[str], category: str) -> str:
        """Create a prompt for switch identification using dictionary format."""
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
    
    def prepare_batch_requests(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare batch requests for all unique playerID sequences."""
        requests = []
        request_index = 0
        
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            sequence_data = group.sort_values('word_index')
            complete_word_sequence = sequence_data['text'].tolist()
            
            if not complete_word_sequence:
                continue
            
            prompt = self.create_switch_identification_prompt(complete_word_sequence, category)
            custom_id = f"switch_dict_{request_index:04d}"
            
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": self.provider.model,
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                "metadata": {
                    "player_id": player_id,
                    "category": category,
                    "total_words": len(sequence_data),
                    "sequence_length": len(complete_word_sequence),
                    "word_sequence": complete_word_sequence,
                    "request_index": request_index
                }
            }
            
            requests.append(request)
            request_index += 1
        
        print(f"Prepared {len(requests)} batch requests for switch classification")
        return requests
    
    def dry_run_test(self, data: pd.DataFrame, num_tests: int = 2) -> bool:
        """Test a few requests with the regular API before submitting the batch."""
        print(f"🧪 Running dry run with {num_tests} test requests...")
        
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
        
        for i, test_seq in enumerate(test_sequences):
            print(f"\n🔍 Testing request {i+1}/{len(test_sequences)}: {test_seq['player_id']} ({test_seq['category']})")
            print(f"   Expected classifications: {test_seq['expected_length']}")
            
            try:
                prompt = self.create_switch_identification_prompt(test_seq['word_sequence'], test_seq['category'])
                messages = [{"role": "user", "content": prompt}]
                
                response_text = self.provider.make_single_request(messages, max_tokens=1500)
                
                # Parse and validate response
                try:
                    cleaned_text = response_text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:]
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()
                    
                    response_data = json.loads(cleaned_text)
                    
                    if 'word_classifications' not in response_data:
                        print(f"   ❌ Missing 'word_classifications' key")
                        return False
                    
                    classifications = response_data['word_classifications']
                    
                    if len(classifications) != test_seq['expected_length']:
                        print(f"   ❌ Length mismatch: expected {test_seq['expected_length']}, got {len(classifications)}")
                        return False
                    
                    # Verify indices and words
                    expected_indices = set(range(test_seq['expected_length']))
                    actual_indices = set(item['index'] for item in classifications)
                    
                    if expected_indices != actual_indices:
                        missing = expected_indices - actual_indices
                        extra = actual_indices - expected_indices
                        print(f"   ❌ Index mismatch. Missing: {missing}, Extra: {extra}")
                        return False
                    
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
        return True
    
    def run_switch_classification(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the complete switch classification pipeline with batching."""
        print("Starting LLM switch classification...")
        
        # Step 1: Prepare batch requests
        requests = self.prepare_batch_requests(data)
        
        # Step 2: Run dry run test
        if not self.dry_run_test(data, num_tests=min(2, len(requests))):
            print("❌ Dry run failed. Please check your data and try again.")
            return data.copy()
        
        # Step 3: Create batch
        batch_id = self.provider.create_batch(requests)
        
        # Step 4: Monitor batch completion
        final_batch = self.provider.monitor_batch(batch_id)
        
        # Step 5: Process results
        result_data = self.process_batch_results(batch_id, data, requests)
        
        print("LLM switch classification completed!")
        return result_data
    
    @abstractmethod
    def process_batch_results(self, batch_id: str, original_data: pd.DataFrame, requests_metadata: List[Dict]) -> pd.DataFrame:
        """Process batch results and add LLM classifications to the original data."""
        pass


class BaseSwitchPredictor(ABC):
    """Abstract base class for switch prediction."""
    
    def __init__(self, provider: BaseLLMProvider, config=None):
        self.provider = provider
        self.config = config
    
    def create_chunk_data(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create chunks for each playerID with incremental word sequences."""
        chunks = []
        chunk_index = 0
        
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            sequence_data = group.sort_values('word_index')
            words = sequence_data['text'].tolist()
            
            if len(words) < 2:  # Need at least 2 words to predict next switch
                continue
            
            # Create incremental chunks: [0], [0,1], [0,1,2], etc.
            for end_idx in range(len(words)):
                chunk_words = words[:end_idx + 1]
                
                # For prediction, we need to know if there's a next word
                has_next_word = end_idx < len(words) - 1
                next_word = words[end_idx + 1] if has_next_word else None
                
                chunks.append({
                    'chunk_id': chunk_index,
                    'player_id': player_id,
                    'category': category,
                    'chunk_words': chunk_words,
                    'chunk_end_index': end_idx,
                    'has_next_word': has_next_word,
                    'next_word': next_word,
                    'sequence_length': len(words)
                })
                
                chunk_index += 1
        
        return chunks
    
    def create_switch_prediction_prompt(self, chunk_words: List[str], category: str) -> str:
        """Create a prompt for predicting if the next word will be a switch."""
        formatted_words = "\n".join([f"{i}. {word}" for i, word in enumerate(chunk_words)])
        
        # Use custom prompt template if available in config
        if self.config and self.config.prompt_template:
            return self.config.prompt_template.format(
                category=category,
                chunk_words=formatted_words,
                num_words=len(chunk_words),
                chunk_length=len(chunk_words)
            )
        
        # Default prompt
        prompt = f"""You are participating in a verbal fluency experiment. You will see a partial sequence of words from the category "{category}" that were produced by a participant.

Your task is to predict whether the NEXT word (if there is one) will start a new thematic group or continue the current group.

Think about:
- What thematic group(s) the current words belong to
- Whether the participant might switch to a different theme for their next word
- Common patterns in verbal fluency tasks (people often cluster related words together)

Current word sequence from category "{category}":
{formatted_words}

Based on this sequence, predict whether the NEXT word will:
- 1: Start a new thematic group (switch)
- 0: Continue the current thematic group (no switch)

Consider the overall flow and grouping patterns. There is no right or wrong answer - this is about predicting human behavior patterns.

Respond with a JSON object containing your prediction:
{{
    "prediction": 1,
    "confidence": 0.75,
    "reasoning": "brief explanation of your reasoning"
}}

The "prediction" must be either 0 or 1.
The "confidence" should be a number between 0 and 1.
The "reasoning" should be a brief explanation (1-2 sentences).

Your response must be valid JSON only. Do not include any other text."""
        
        return prompt
    
    def prepare_batch_requests(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare batch requests for all chunks."""
        chunks = self.create_chunk_data(data)
        
        # Only process chunks that have a next word to predict
        prediction_chunks = [chunk for chunk in chunks if chunk['has_next_word']]
        
        requests = []
        
        for chunk in prediction_chunks:
            prompt = self.create_switch_prediction_prompt(chunk['chunk_words'], chunk['category'])
            custom_id = f"pred_{chunk['chunk_id']:06d}"
            
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": self.provider.model,
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                "metadata": {
                    "chunk_id": chunk['chunk_id'],
                    "player_id": chunk['player_id'],
                    "category": chunk['category'],
                    "chunk_words": chunk['chunk_words'],
                    "chunk_end_index": chunk['chunk_end_index'],
                    "next_word": chunk['next_word'],
                    "sequence_length": chunk['sequence_length']
                }
            }
            
            requests.append(request)
        
        print(f"Prepared {len(requests)} batch requests for switch prediction")
        print(f"Total chunks: {len(chunks)}, Prediction chunks: {len(prediction_chunks)}")
        return requests
    
    def dry_run_test(self, data: pd.DataFrame, num_tests: int = 2) -> bool:
        """Test a few requests with the regular API before submitting the batch."""
        print(f"🧪 Running dry run with {num_tests} test requests...")
        
        chunks = self.create_chunk_data(data)
        prediction_chunks = [chunk for chunk in chunks if chunk['has_next_word']]
        
        if len(prediction_chunks) == 0:
            print("❌ No valid prediction chunks found")
            return False
        
        test_chunks = prediction_chunks[:num_tests]
        
        for i, chunk in enumerate(test_chunks):
            print(f"\n🔍 Testing prediction {i+1}/{len(test_chunks)}: {chunk['player_id']} ({chunk['category']})")
            print(f"   Chunk: {chunk['chunk_words']} → predicting for next word: {chunk['next_word']}")
            
            try:
                prompt = self.create_switch_prediction_prompt(chunk['chunk_words'], chunk['category'])
                messages = [{"role": "user", "content": prompt}]
                
                response_text = self.provider.make_single_request(messages, max_tokens=500)
                
                # Parse and validate response
                try:
                    cleaned_text = response_text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:]
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()
                    
                    response_data = json.loads(cleaned_text)
                    
                    prediction = response_data.get('prediction')
                    confidence = response_data.get('confidence')
                    reasoning = response_data.get('reasoning', '')
                    
                    if prediction not in [0, 1]:
                        print(f"   ❌ Invalid prediction value: {prediction}")
                        return False
                    
                    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                        print(f"   ❌ Invalid confidence value: {confidence}")
                        return False
                    
                    print(f"   ✅ Test {i+1} passed")
                    print(f"     Prediction: {prediction} (confidence: {confidence:.2f})")
                    print(f"     Reasoning: {reasoning[:50]}...")
                    
                except json.JSONDecodeError as e:
                    print(f"   ❌ Invalid JSON response: {e}")
                    print(f"   Response preview: {response_text[:200]}...")
                    return False
            
            except Exception as e:
                print(f"   ❌ API call failed: {e}")
                return False
        
        print(f"\n✅ All {len(test_chunks)} dry run tests passed!")
        return True
    
    def run_switch_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the complete switch prediction pipeline with batching."""
        print("Starting LLM switch prediction...")
        
        # Step 1: Prepare batch requests
        requests = self.prepare_batch_requests(data)
        
        if len(requests) == 0:
            print("No valid requests to process. Returning original data.")
            return data.copy()
        
        # Step 2: Run dry run test
        if not self.dry_run_test(data, num_tests=min(2, len(requests))):
            print("❌ Dry run failed. Please check your data and try again.")
            return data.copy()
        
        # Step 3: Create batch
        batch_id = self.provider.create_batch(requests)
        
        # Step 4: Monitor batch completion
        final_batch = self.provider.monitor_batch(batch_id)
        
        # Step 5: Process results
        result_data = self.process_batch_results(batch_id, data, requests)
        
        print("LLM switch prediction completed!")
        return result_data
    
    @abstractmethod
    def process_batch_results(self, batch_id: str, original_data: pd.DataFrame, requests_metadata: List[Dict]) -> pd.DataFrame:
        """Process batch results and add LLM predictions to the original data."""
        pass


class BaseGroupLabeler(ABC):
    """Abstract base class for group labeling."""
    
    def __init__(self, provider: BaseLLMProvider, config=None):
        self.provider = provider
        self.config = config
    
    def create_groups_from_switches(self, sequence_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create word groups from switchLLM column."""
        groups = []
        current_group = []
        current_group_start_idx = 0
        
        for idx, row in sequence_data.iterrows():
            word_index = row['word_index']
            word = row['text']
            switch = row['switchLLM']
            
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
                if word_index == 0:
                    current_group_start_idx = 0
        
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
        """Create a prompt for group labeling that mirrors human instructions."""
        formatted_groups = []
        for i, group in enumerate(groups):
            words_str = ", ".join(group['words'])
            formatted_groups.append(f"Group {i+1}: {words_str}")
        
        groups_text = "\n".join(formatted_groups)
        
        # Use custom prompt template if available in config
        if self.config and self.config.prompt_template:
            return self.config.prompt_template.format(
                category=category,
                groups_text=groups_text,
                num_groups=len(groups)
            )
        
        # Default prompt
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
        """Prepare batch requests for all unique playerID sequences that have switchLLM data."""
        requests = []
        request_index = 0
        
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            sequence_data = group.sort_values('word_index')
            
            if 'switchLLM' not in sequence_data.columns or sequence_data['switchLLM'].isna().all():
                print(f"Skipping player {player_id} ({category}) - no switchLLM data")
                continue
            
            groups = self.create_groups_from_switches(sequence_data)
            
            if len(groups) == 0:
                print(f"Skipping player {player_id} ({category}) - no groups found")
                continue
            
            prompt = self.create_group_labeling_prompt(groups, category)
            custom_id = f"label_req_{request_index:04d}"
            
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": self.provider.model,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                "metadata": {
                    "player_id": player_id,
                    "category": category,
                    "num_groups": len(groups),
                    "groups": groups,
                    "request_index": request_index
                }
            }
            
            requests.append(request)
            request_index += 1
        
        print(f"Prepared {len(requests)} batch requests for group labeling")
        return requests
    
    def dry_run_test(self, data: pd.DataFrame, num_tests: int = 2) -> bool:
        """Test a few requests with the regular API before submitting the batch."""
        print(f"🧪 Running dry run test with {num_tests} requests...")
        
        test_requests = []
        player_groups = data.groupby(['playerID', 'category'])
        
        for (player_id, category), group in player_groups:
            if len(test_requests) >= num_tests:
                break
            
            sequence_data = group.sort_values('word_index')
            
            if 'switchLLM' not in sequence_data.columns or sequence_data['switchLLM'].isna().all():
                continue
            
            groups = self.create_groups_from_switches(sequence_data)
            
            if len(groups) == 0:
                continue
            
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
        
        for i, test_req in enumerate(test_requests):
            print(f"\n📝 Test {i+1}/{len(test_requests)}: Player {test_req['player_id']} ({test_req['category']})")
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
                        print(f"   ✅ Valid response with {len(group_labels)} labels")
                        
                        for j, (group, label_data) in enumerate(zip(test_req['groups'], group_labels[:3])):
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
        return True
    
    def run_group_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the complete group labeling pipeline with batching."""
        print("Starting LLM group labeling with batching...")
        
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
        if not self.dry_run_test(data, num_tests=min(2, len(requests))):
            print("❌ Dry run failed. Please check your data and try again.")
            return data.copy()
        
        # Step 3: Create batch
        batch_id = self.provider.create_batch(requests)
        
        # Step 4: Monitor batch completion
        final_batch = self.provider.monitor_batch(batch_id)
        
        # Step 5: Process results
        result_data = self.process_batch_results(batch_id, data, requests)
        
        print("LLM group labeling completed!")
        return result_data
    
    @abstractmethod
    def process_batch_results(self, batch_id: str, original_data: pd.DataFrame, requests_metadata: List[Dict]) -> pd.DataFrame:
        """Process batch results and add LLM group labels to the original data."""
        pass