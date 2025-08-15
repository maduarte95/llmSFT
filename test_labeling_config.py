#!/usr/bin/env python3
"""
Test script to verify group labeling config functionality.
Run with: uv run python test_labeling_config.py
"""

from config_system import ConfigManager

def test_labeling_configs():
    """Test loading labeling configs and custom prompt templates."""
    print('🏷️ Testing Group Labeling Config Functionality')
    print('=' * 50)
    
    config_manager = ConfigManager()
    
    # Check labeling configs
    labeling_configs = config_manager.list_configs('labels')
    print(f'Found {len(labeling_configs)} labeling configs:')
    for config in labeling_configs:
        has_custom = "✅ Custom" if config.prompt_template else "📝 Default"
        print(f'  - {config.name} ({has_custom})')
    
    if len(labeling_configs) == 0:
        print('❌ No labeling configs found!')
        return False
    
    # Test the creative labeling config
    creative_config = config_manager.get_config('experimental-high-creativity-labels')
    if creative_config:
        print(f'\n✅ Testing custom labeling config: {creative_config.name}')
        print(f'Description: {creative_config.description}')
        print(f'Custom prompt template: {creative_config.prompt_template is not None}')
        
        if creative_config.prompt_template:
            # Test prompt formatting with sample groups
            sample_groups_text = "Group 1: banana, lemon, corn\nGroup 2: strawberry, blackberry, cranberry"
            
            try:
                formatted_prompt = creative_config.prompt_template.format(
                    category='foods',
                    groups_text=sample_groups_text,
                    num_groups=2
                )
                print(f'\n📝 Sample formatted prompt (first 400 chars):')
                print(formatted_prompt[:400] + '...')
                print('\n✅ Custom labeling prompt template formatting works!')
                return True
            except Exception as e:
                print(f'❌ Prompt formatting failed: {e}')
                return False
        else:
            print('📝 Config has no custom prompt template')
            return True
    else:
        print('❌ Creative labeling config not found')
        return False

def test_available_variables():
    """Show available template variables for labeling."""
    print('\n📋 Available Template Variables for Group Labeling:')
    print('  {category} - The verbal fluency category (e.g., "animals")')
    print('  {groups_text} - Formatted groups ("Group 1: cat, dog\\nGroup 2: fish, bird")')
    print('  {num_groups} - Number of groups to label')
    
    print('\n💡 Example Custom Template Usage:')
    example_template = '''
🏷️ CREATIVE LABELING TASK 🏷️

Category: {category}
Number of groups: {num_groups}

Groups to label:
{groups_text}

Your task: Create imaginative, SHORT labels for each group!
Think outside the box - consider semantic, phonetic, or cultural connections.

Respond in JSON format only.'''
    
    print(example_template)

if __name__ == "__main__":
    success = test_labeling_configs()
    test_available_variables()
    
    if success:
        print('\n🎉 Group Labeling Config Functionality is working!')
        print('\n📋 Usage:')
        print('1. Run: uv run python run_llm_labeling_with_config.py')
        print('2. Select "experimental-high-creativity-labels" for custom prompts')
        print('3. Or create your own labeling config with custom prompt_template')
        print('\n💡 Note: Labeling requires data with switchLLM column from classification step')
    else:
        print('\n❌ There are issues with the labeling config setup')