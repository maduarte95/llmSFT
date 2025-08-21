#!/usr/bin/env python3
"""
Utility functions for loading and formatting animal category data.
"""

import os
from pathlib import Path


def load_animal_categories(file_path: str = "animals_by_category.txt") -> str:
    """
    Load animal categories from file and format for use in prompts.
    
    Args:
        file_path: Path to the animals_by_category.txt file
        
    Returns:
        Formatted string with all animal categories
    """
    try:
        if not os.path.exists(file_path):
            print(f"Warning: Animal categories file not found at {file_path}")
            return ""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"Warning: Animal categories file is empty")
            return ""
        
        # Format the content for better readability in prompts
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line and ':' in line:
                # Split category name and animals
                category, animals = line.split(':', 1)
                category = category.strip()
                animals = animals.strip()
                
                # Format for better readability
                formatted_lines.append(f"- {category}: {animals}")
        
        return '\n'.join(formatted_lines)
    
    except Exception as e:
        print(f"Error loading animal categories: {e}")
        return ""


def get_enhanced_animal_prompt_template() -> str:
    """
    Get the enhanced prompt template with animal categories dynamically loaded.
    
    Returns:
        Prompt template string with {animal_categories} placeholder
    """
    return '''You are analyzing word groups from a verbal fluency task where participants named animals.

Your task is to provide short, descriptive labels for each group based on the semantic relationships between the words.

Here are the complete animal categories for reference:
{animal_categories}

Groups to label:
{groups_text}

Please provide a JSON response with the following format:
{{
  "group_labels": [
    {{
      "group_number": 1,
      "words": ["word1", "word2", "word3"],
      "label": "descriptive label for this group"
    }},
    {{
      "group_number": 2,
      "words": ["word4", "word5"],
      "label": "descriptive label for this group"
    }}
  ]
}}

Guidelines:
- Labels should be concise but descriptive (2-4 words)
- Focus on the most obvious semantic relationship based on the categories above
- Use the category names when appropriate (e.g., "African animals", "Farm animals", "Birds")
- Consider subcategories like "big cats", "marine mammals", "arctic animals"
- If words don't fit a clear category, use descriptive terms like "large mammals", "small pets", etc.'''


def create_animal_prompt_with_categories(groups_text: str, category: str) -> str:
    """
    Create a complete prompt with animal categories loaded dynamically.
    
    Args:
        groups_text: Formatted groups text
        category: The category (should be "animals")
        
    Returns:
        Complete prompt with animal categories included
    """
    if category != "animals":
        # For non-animals, use a basic prompt
        return f"""Analyze these word groups and provide descriptive labels:

{groups_text}

Please provide a JSON response with group labels."""
    
    # Load animal categories dynamically
    animal_categories = load_animal_categories()
    
    if not animal_categories:
        print("Warning: Could not load animal categories, using basic prompt")
        return f"""Analyze these animal word groups and provide descriptive labels:

{groups_text}

Please provide a JSON response with group labels."""
    
    # Use enhanced template with loaded categories
    template = get_enhanced_animal_prompt_template()
    return template.format(
        animal_categories=animal_categories,
        groups_text=groups_text
    )