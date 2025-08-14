# Unified LLM Switch Classification & Group Labeling

This project provides LLM-based analysis of verbal fluency data with support for multiple API providers (Anthropic and TogetherAI).

## Features

- **Multi-provider support**: Choose between Anthropic Claude and TogetherAI models
- **Switch classification**: Identify thematic group boundaries in word sequences
- **Group labeling**: Generate descriptive labels for word groups
- **Batch processing**: Efficient processing of large datasets
- **Backward compatibility**: Original Anthropic-only scripts still work

## Quick Start

### 1. Installation with UV (Recommended)

This project uses [UV](https://docs.astral.sh/uv/) for fast dependency management:

```bash
# Initialize the project and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

#### Alternative: Manual Installation

```bash
pip install pandas anthropic requests python-dotenv
```

#### Development Installation

```bash
# Install with development dependencies
uv sync --extra dev

# Install with all optional dependencies (dev + jupyter)
uv sync --extra all
```

### 2. API Setup

Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```bash
# For Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_key_here

# For TogetherAI (Llama, Qwen, etc.)
TOGETHER_API_KEY=your_together_key_here
```

### 3. Data Format

Your CSV file should contain:
- `playerID`: Unique participant identifier
- `text`: Individual words from verbal fluency task
- `word_index`: Order of word in sequence (0, 1, 2, ...)
- `category`: Task category (e.g., "animals", "supermarket items")

### 4. Usage

#### Switch Classification
```bash
# Using UV
uv run python run_llm_classification_unified.py

# Or with activated virtual environment
python run_llm_classification_unified.py

# Quick test of TogetherAI integration
uv run python test_together_api.py
```

#### Group Labeling (after switch classification)
```bash
# Using UV
uv run python run_llm_labeling_unified.py

# Or with activated virtual environment
python run_llm_labeling_unified.py
```

## Available Scripts

### New Unified Scripts (Recommended)
- `run_llm_classification_unified.py` - Multi-provider switch classification
- `run_llm_labeling_unified.py` - Multi-provider group labeling

### Original Anthropic-Only Scripts (Still Supported)
- `run_llm_classification_dict.py` - Anthropic switch classification
- `llm_group_labeler.py` - Anthropic group labeling
- `run_llm_classification_specific_dict.py` - Process specific sequences

### Utility Scripts
- `inspect_batch_response.py` - Debug batch processing issues

## Provider Comparison

| Provider | Models | Batch API | JSON Mode | Cost |
|----------|--------|-----------|-----------|------|
| **Anthropic** | Claude Sonnet 4, Claude 3.5 | ✅ | ✅ | Higher |
| **TogetherAI** | Llama 3.1, Qwen 2.5, DeepSeek | ✅ | ✅ | 50% lower |

## Architecture

The codebase uses an abstract base class architecture:

- `base_classifiers.py` - Abstract base classes
- `anthropic_providers.py` - Anthropic implementation
- `together_providers.py` - TogetherAI implementation

This allows easy extension to additional providers.

## Workflow

1. **Switch Classification**: 
   - Input: Raw verbal fluency data
   - Output: Data with `switchLLM` column (0/1 for group boundaries)

2. **Group Labeling**:
   - Input: Data with `switchLLM` column
   - Output: Data with `labelLLM` column (descriptive group labels)

## Example Output

### Switch Classification
```
word_index  text        switchLLM
0           dog         1          # First word, always 1
1           cat         0          # Same group (pets)
2           lion        1          # New group (wild animals)
3           tiger       0          # Same group
```

### Group Labeling
```
word_index  text        switchLLM  labelLLM
0           dog         1          "pets"
1           cat         0          "pets"  
2           lion        1          "wild animals"
3           tiger       0          "wild animals"
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure your `.env` file contains the correct API key
   - Check the key name matches: `ANTHROPIC_API_KEY` or `TOGETHER_API_KEY`

2. **Missing switchLLM Column**
   - Run switch classification before group labeling
   - Use `run_llm_classification_unified.py` first

3. **Batch Processing Fails**
   - Use `inspect_batch_response.py` to debug
   - Check your data format matches requirements

4. **File Not Found**
   - Ensure your CSV file matches expected patterns
   - Use `*switch*.csv` or `*llm*.csv` naming for group labeling

### Performance Tips

- **TogetherAI**: 50% lower cost, good for large datasets
- **Anthropic**: Higher quality, better for research applications
- **Batch Size**: 1,000-10,000 requests per batch is optimal
- **Model Choice**: Larger models (70B+) generally perform better

## Contributing

To add a new provider:

1. Create a new provider class inheriting from `BaseLLMProvider`
2. Implement required methods for batch processing
3. Create classifier and labeler classes
4. Add provider option to unified scripts

## License

This project is for research use. Please follow each API provider's terms of service.