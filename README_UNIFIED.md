# Unified LLM Verbal Fluency Analysis

This project provides comprehensive LLM-based analysis of verbal fluency data with support for multiple API providers (Anthropic and TogetherAI).

## Features

- **Multi-provider support**: Choose between Anthropic Claude and TogetherAI models
- **Switch classification**: Identify thematic group boundaries in word sequences
- **Switch prediction**: Predict whether the next word will start a new group
- **Group labeling**: Generate descriptive labels for word groups
- **Comprehensive analysis pipeline** (**NEW!**): Automated statistical analysis and visualization
  - Per-participant and summary metrics (accuracy, precision, recall, F1, TPR/FPR)
  - Multiple baseline comparisons (ZeroR, Random)
  - 20+ publication-ready visualizations with confidence intervals
  - Word index window analysis for temporal performance trends
- **Custom prompt templates**: Fully customizable prompts via YAML configs
- **Batch processing**: Efficient processing of large datasets
- **Configuration-driven**: Reproducible experiments with YAML configs
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
# Basic unified script
uv run python run_llm_classification_unified.py

# With custom prompt templates and configs
uv run python run_llm_classification_with_config.py
```

#### Switch Prediction (NEW!)
```bash
# Predict next word switches with incremental chunks
uv run python run_llm_prediction_unified.py

# With custom prompt templates and configs
uv run python run_llm_prediction_with_config.py
```

#### Group Labeling (after switch classification)
```bash
# Basic unified script
uv run python run_llm_labeling_unified.py

# With custom prompt templates and configs
uv run python run_llm_labeling_with_config.py
```

#### Analysis & Visualization (**NEW!**)
```bash
# Comprehensive analysis pipeline with visualizations and statistics
uv run python analysis_pipeline.py

# Generate readable summary of results  
uv run python generate_results_summary.py
```

#### Testing & Validation
```bash
# Test custom prompt templates
uv run python test_custom_prompt.py

# Test chunking logic for prediction
uv run python test_chunking.py

# Test labeling config functionality
uv run python test_labeling_config.py
```

## Available Scripts

### Core Analysis Scripts
- `run_llm_classification_unified.py` - Multi-provider switch classification
- `run_llm_prediction_unified.py` - **NEW!** Multi-provider switch prediction
- `run_llm_labeling_unified.py` - Multi-provider group labeling

### Configuration-Driven Scripts (Recommended)
- `run_llm_classification_with_config.py` - Switch classification with custom prompts
- `run_llm_prediction_with_config.py` - **NEW!** Switch prediction with custom prompts
- `run_llm_labeling_with_config.py` - **NEW!** Group labeling with custom prompts

### Testing & Validation Scripts
- `test_custom_prompt.py` - Test custom prompt template functionality
- `test_chunking.py` - Test incremental chunking logic for prediction
- `test_labeling_config.py` - Test labeling configuration functionality

### Original Anthropic-Only Scripts (Still Supported)
- `run_llm_classification_dict.py` - Anthropic switch classification
- `llm_group_labeler.py` - Anthropic group labeling
- `run_llm_classification_specific_dict.py` - Process specific sequences

### Batch Retrieval Scripts (NEW!)
**TogetherAI Retrieval:**
- `retrieve_batch_results.py` - Retrieve switch classification results
- `retrieve_prediction_batch_results.py` - Retrieve prediction results  
- `retrieve_labeling_batch_results.py` - Retrieve labeling results

**Anthropic Retrieval:**
- `retrieve_anthropic_batch_results.py` - Retrieve switch classification results
- `retrieve_anthropic_prediction_batch_results.py` - Retrieve prediction results
- `retrieve_anthropic_labeling_batch_results.py` - Retrieve labeling results

### Utility Scripts
- `inspect_batch_response.py` - Debug batch processing issues
- `config_system.py` - Configuration management system

### Analysis & Visualization Scripts (**NEW!**)
- `analysis_pipeline.py` - **Comprehensive analysis pipeline with visualizations and statistics**
  - Automated data filtering and preprocessing
  - Per-participant and summary metrics (accuracy, precision, recall, F1)
  - TPR/FPR confusion matrix analysis
  - Baseline comparisons (ZeroR, Random)
  - Multiple visualization types (bar plots, line plots with confidence intervals)
  - Word index window analysis for all metrics
  - Timestamped outputs for reproducibility
- `generate_results_summary.py` - Generate readable summary reports of analysis results

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

## Analysis Workflows

### 1. Traditional Classification & Labeling Workflow
1. **Switch Classification**: 
   - Input: Raw verbal fluency data
   - Output: Data with `switchLLM` column (0/1 for group boundaries)
   - Process: Analyze complete sequences to identify thematic switches

2. **Group Labeling**:
   - Input: Data with `switchLLM` column
   - Output: Data with `labelLLM` column (descriptive group labels)
   - Process: Generate labels for word groups identified by switches

### 2. NEW! Switch Prediction Workflow
1. **Switch Prediction**:
   - Input: Raw verbal fluency data
   - Output: Chunk-based predictions in new DataFrame
   - Process: Create incremental chunks and predict next-word switches
   - Example: `[cat]` → predict `dog`, `[cat, dog]` → predict `lamb`, etc.

### 3. Custom Prompt Workflows
All workflows support custom prompt templates through YAML configuration files for:
- Experimenting with different instruction styles
- Adding domain-specific context
- Optimizing for specific research questions

## Example Output

### Switch Classification
```
word_index  text        switchLLM
0           dog         1          # First word, always 1
1           cat         0          # Same group (pets)
2           lion        1          # New group (wild animals)
3           tiger       0          # Same group
```

### Switch Prediction (NEW!)
```
chunk_id  player_id  chunk_words  next_word  prediction_llm  confidence_llm
0         p001       cat          dog        0               0.85
1         p001       cat, dog     lion       1               0.92
2         p001       cat, dog, lion tiger    0               0.78
```

### Group Labeling
```
word_index  text        switchLLM  labelLLM
0           dog         1          "pets"
1           cat         0          "pets"  
2           lion        1          "wild animals"
3           tiger       0          "wild animals"
```

## Custom Prompt Templates

### Overview
All analysis tasks support custom prompt templates through YAML configuration files, allowing you to:
- Experiment with different instruction styles
- Add domain-specific context or examples
- Optimize prompts for specific research questions
- Maintain reproducible experiments with version control

### Available Template Variables

#### Switch Classification
- `{category}` - The verbal fluency category
- `{word_sequence}` - Formatted word list ("0. cat\n1. dog\n2. fish")
- `{num_words}` - Number of words in sequence

#### Switch Prediction
- `{category}` - The verbal fluency category
- `{chunk_words}` - Formatted chunk words
- `{num_words}` - Number of words in chunk
- `{chunk_length}` - Same as num_words

#### Group Labeling
- `{category}` - The verbal fluency category
- `{groups_text}` - Formatted groups ("Group 1: cat, dog\nGroup 2: fish, bird")
- `{num_groups}` - Number of groups to label

### Example Custom Configurations

#### Switch Prediction with Emojis
```yaml
# configs/anthropic_claude_sonnet4_prediction_custom.yaml
prompt_template: |
  🧠 VERBAL FLUENCY PREDICTION TASK 🧠
  
  You're analyzing human word production patterns in a "{category}" verbal fluency task.
  
  CURRENT SEQUENCE ({chunk_length} words):
  {chunk_words}
  
  🎯 YOUR MISSION: Predict if the participant's NEXT word will start a new semantic cluster...
```

#### Creative Group Labeling
```yaml
# configs/experimental_high_creativity_labels.yaml
prompt_template: |
  Your task is to provide SHORT, CREATIVE, and DESCRIPTIVE labels...
  Consider multiple possible relationships: semantic, phonetic, cultural, functional...
```

### Creating Custom Configs
1. Copy an existing config file from `configs/`
2. Modify the `prompt_template` field with your custom prompt
3. Use the available template variables in your prompt
4. Run with: `uv run python run_llm_*_with_config.py`

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure your `.env` file contains the correct API key
   - Check the key name matches: `ANTHROPIC_API_KEY` or `TOGETHER_API_KEY`

2. **Missing switchLLM Column** (for labeling)
   - Run switch classification before group labeling
   - Use `run_llm_classification_unified.py` first

3. **Batch Processing Fails**
   - **NEW!** Use retrieval scripts if batch completes but processing fails:
     - TogetherAI: `retrieve_batch_results.py`, `retrieve_prediction_batch_results.py`, `retrieve_labeling_batch_results.py`
     - Anthropic: `retrieve_anthropic_batch_results.py`, `retrieve_anthropic_prediction_batch_results.py`, `retrieve_anthropic_labeling_batch_results.py`
   - Use `inspect_batch_response.py` to debug batch content issues
   - Check your data format matches requirements

4. **File Not Found**
   - Classification/Prediction: Use `filtered_data_for_analysis.csv` pattern
   - Labeling: Use `*switch*llm*.csv` pattern from classification output

5. **Custom Prompt Template Errors**
   - Check variable names match available options
   - Test template formatting with `test_custom_prompt.py`
   - Ensure YAML syntax is correct (use `|` for multi-line templates)

6. **Chunking Issues** (for prediction)
   - Verify player separation with `test_chunking.py` 
   - Check that `playerID` and `word_index` columns are properly formatted

### Performance Tips

- **TogetherAI**: 50% lower cost, good for large datasets
- **Anthropic**: Higher quality, better for research applications
- **Batch Size**: 1,000-10,000 requests per batch is optimal
- **Model Choice**: Larger models (70B+) generally perform better

## Script Guide & Parameters

### 🚀 Main Analysis Scripts

#### Switch Classification
```bash
# Basic unified script
uv run python run_llm_classification_unified.py

# With custom config (recommended)
uv run python run_llm_classification_with_config.py
```
**Input:** Raw verbal fluency data (CSV with `playerID`, `text`, `word_index`, `category`)
**Output:** Original data + `switchLLM` column (0/1 for group boundaries)

#### Switch Prediction  
```bash
# Basic unified script
uv run python run_llm_prediction_unified.py

# With custom config (recommended)
uv run python run_llm_prediction_with_config.py
```
**Input:** Raw verbal fluency data (same as classification)
**Output:** 
- `data_with_predictions_{config}_{timestamp}.csv` - Chunk-based predictions
- `data_with_word_predictions_{config}_{timestamp}.csv` - **NEW!** Word-level predictions

#### Group Labeling
```bash
# Basic unified script
uv run python run_llm_labeling_unified.py

# With custom config (recommended)
uv run python run_llm_labeling_with_config.py
```
**Input:** Data with `switchLLM` column (from classification step)
**Output:** Original data + `labelLLM` column (descriptive group labels)

### 🔧 Batch Retrieval Scripts

Use these when a batch completes but processing fails:

#### TogetherAI Retrieval
```bash
# Switch classification
python retrieve_batch_results.py <batch_id> [data_file] [config_name]

# Prediction (NEW!)
python retrieve_prediction_batch_results.py <batch_id> [data_file] [config_name]

# Labeling (NEW!)
python retrieve_labeling_batch_results.py <batch_id> [data_file] [config_name]
```

#### Anthropic Retrieval
```bash
# Switch classification
python retrieve_anthropic_batch_results.py <batch_id> [data_file] [config_name]

# Prediction (NEW!)
python retrieve_anthropic_prediction_batch_results.py <batch_id> [data_file] [config_name]

# Labeling (NEW!)
python retrieve_anthropic_labeling_batch_results.py <batch_id> [data_file] [config_name]
```

**Parameters:**
- `<batch_id>` - Required batch ID from the failed run
- `[data_file]` - Optional, auto-detects if not provided
- `[config_name]` - Optional, used for output filename

### 🧪 Testing & Validation Scripts

```bash
# Test custom prompt templates
uv run python test_custom_prompt.py

# Test chunking logic for prediction
uv run python test_chunking.py

# Test labeling config functionality
uv run python test_labeling_config.py
```

### 📊 Output File Formats

#### Switch Classification Output
```csv
playerID,text,word_index,category,switchLLM
p001,dog,0,animals,1
p001,cat,1,animals,0
p001,lion,2,animals,1
```

#### Prediction Output (NEW!)
**Chunk-based:** `data_with_predictions_{config}_{timestamp}.csv`
```csv
chunk_id,player_id,category,chunk_words,next_word,prediction_llm,confidence_llm
0,p001,animals,cat,dog,0,0.85
1,p001,animals,"cat, dog",lion,1,0.92
```

**Word-level:** `data_with_word_predictions_{config}_{timestamp}.csv`
```csv
playerID,text,word_index,category,predicted_switch_llm,prediction_confidence_llm
p001,dog,1,animals,0,0.85
p001,lion,2,animals,1,0.92
```

#### Labeling Output
```csv
playerID,text,word_index,category,switchLLM,labelLLM
p001,dog,0,animals,1,pets
p001,cat,1,animals,0,pets
p001,lion,2,animals,1,wild animals
```

#### Analysis Pipeline Output (**NEW!**)
**Per-Participant Metrics:** `participant_metrics_{timestamp}.csv`
```csv
playerID,category,method,accuracy,precision,recall,f1,n_samples
p001,animals,Human_Predicted,0.756,0.45,0.62,0.52,45
p001,animals,LLM_Predicted,0.822,0.67,0.58,0.62,45
```

**TPR/FPR Confusion Matrix:** `participant_tpr_fpr_{timestamp}.csv` 
```csv
playerID,category,method,TPR,FPR,TP,FP,TN,FN,n_samples
p001,animals,Human_Predicted,0.625,0.15,10,3,17,6,36
p001,animals,LLM_Predicted,0.750,0.12,12,2,15,7,36
```

**Summary Statistics:** `summary_statistics_{timestamp}.csv`, `tpr_fpr_summary_{timestamp}.csv`

**Window Analysis:** `{metric}_by_window_{timestamp}.csv` (for accuracy, precision, recall, f1)

**Visualizations:** 20+ plots in PNG format including:
- Overall comparison plots (accuracy, precision, recall, F1)
- Category-wise performance plots
- Word index window trends with confidence intervals

### ⚙️ Configuration Files

All config-driven scripts use YAML files in `configs/` directory:

```yaml
# Example: configs/anthropic_claude_sonnet4_prediction.yaml
provider: anthropic
model: claude-sonnet-4-20250514
prompt_template: |
  Your task is to predict whether the next word will start a new semantic group...
  
  Current sequence: {chunk_words}
  Category: {category}
  
  Respond with JSON: {"prediction": 0 or 1, "confidence": 0.0-1.0, "reasoning": "explanation"}

parameters:
  max_tokens: 500
  temperature: 0.1
```

### 🔄 Typical Workflow

1. **Switch Classification** → Get group boundaries
2. **Group Labeling** → Label the identified groups
3. **Switch Prediction** → Alternative approach for real-time prediction

Or use **Prediction only** for prospective/incremental analysis.~

**CREATE LLM classifications:**
```uv run run_llm_classification_robust.py```
   -> NOW WITH IRT
	-> output:..._switches.csv
```uv run run_llm_prediction_with_config.py```
   - NOW WITH IRT
	-> choose "no", then choose choose file with switches
	-> output:..._predictions.csv
```uv run run_llm_labeling_with_config.py```
	-> choose file data_with_word_predictions_*
	->output: ...labels.csv

**ANALYZE DATA**
```uv run analysis_pipeline.csv```
-> choose ..labels.csv
- ...filtered.csv -> contains filtered data

```uv run python statistical_significance_analysis.py --force-test nonparametric```

**RECOVER BATCH**
SWITCHES - ANTHROPIC
```uv run python retrieve_robust_batch_results.py msgbatch_01CAkQdXg9cFhxRcffTqeCx6 merged_trials_data_unfiltered.csv```

## Contributing

To add a new provider:

1. Create a new provider class inheriting from `BaseLLMProvider`
2. Implement required methods for batch processing
3. Create classifier and labeler classes
4. Add provider option to unified scripts

## License

This project is for research use. Please follow each API provider's terms of service.