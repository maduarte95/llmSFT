#!/usr/bin/env python3
"""
Simple test script to verify custom prompt template functionality.
Run with: uv run python test_custom_prompt.py
"""

from config_system import ConfigManager

def test_config_loading():
    """Test loading prediction configs and custom prompt templates."""
    print('🔧 Testing config loading...')
    config_manager = ConfigManager()

    # Check if our new configs are loaded
    configs = config_manager.list_configs('prediction')
    print(f'Found {len(configs)} prediction configs:')
    for config in configs:
        has_custom = "✅ Custom" if config.prompt_template else "📝 Default"
        print(f'  - {config.name} ({has_custom})')

    # Test the custom prompt config
    custom_config = config_manager.get_config('claude-sonnet-4-prediction-custom')
    if custom_config:
        print(f'\n✅ Loaded custom config: {custom_config.name}')
        print(f'Custom prompt template: {custom_config.prompt_template is not None}')
        
        if custom_config.prompt_template:
            # Test prompt formatting
            test_words = ['cat', 'dog', 'fish']
            formatted_words = '\n'.join([f'{i}. {word}' for i, word in enumerate(test_words)])
            
            try:
                formatted_prompt = custom_config.prompt_template.format(
                    category='animals',
                    chunk_words=formatted_words,
                    num_words=len(test_words),
                    chunk_length=len(test_words)
                )
                print(f'\n📝 Sample formatted prompt (first 300 chars):')
                print(formatted_prompt[:300] + '...')
                print('\n✅ Prompt template formatting works!')
                return True
            except Exception as e:
                print(f'❌ Prompt formatting failed: {e}')
                return False
    else:
        print('❌ Custom config not found')
        return False

if __name__ == "__main__":
    success = test_config_loading()
    if success:
        print('\n🎉 Custom prompt template functionality is working!')
        print('\n📋 Usage:')
        print('1. Run: uv run python run_llm_prediction_with_config.py')
        print('2. Select "claude-sonnet-4-prediction-custom" config')
        print('3. The custom prompt with emojis will be used!')
    else:
        print('\n❌ There are issues with the custom prompt setup')