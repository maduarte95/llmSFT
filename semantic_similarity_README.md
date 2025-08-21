# Semantic Distance Analysis Pipeline

This pipeline analyzes semantic distance between text and different types of labels using either Together AI's embedding models or local sentence-transformers models.

## Overview

The pipeline compares three types of semantic distance:
- **SNAFU distance**: between "text" and "snafu_gt_label" 
- **Human distance**: between "text" and "label"
- **LLM distance**: between "text" and "labelLLM"

The final output is a bar plot titled "Semantic distance between words and cluster-based labels".

## Embedding Generation Methods

The pipeline supports two embedding generation approaches:

### A. Together AI Embeddings (API-based)
Uses Together AI's embedding API for remote embedding generation.

### B. Sentence-Transformers Embeddings (Local)
Uses local sentence-transformers models with task-specific prompts for better semantic matching.

## Scripts

### Together AI Scripts

### 1. `generate_embeddings_human.py`
Generates embeddings for text, human labels, and SNAFU categories.

**Features:**
- Interactive model selection from 4 available embedding models
- Text normalization for consistent embeddings (removes punctuation, case, whitespace)
- Applies same filtering criteria as `analysis_pipeline.py` (removes extreme participants, duplicates, etc.)
- Batch processing for efficient API usage
- Saves embeddings as pickle files with metadata
- Saves data mapping for analysis (with both original and normalized text)

**Output files:**
- `embeddings/human_embeddings.pkl` - Combined embeddings file
- `embeddings/human_embeddings_metadata.json` - Metadata 
- `embeddings/human_data_mapping.csv` - Data mapping

### 2. `generate_embeddings_llm.py`
Generates embeddings for LLM labels separately to allow reuse of human embeddings across different LLM runs.

**Features:**
- Interactive model selection from 4 available embedding models
- Text normalization for consistent embeddings
- Same filtering and batch processing as human script
- Compatible with existing human embeddings

**Output files:**
- `embeddings/llm_embeddings.pkl` - LLM embeddings file
- `embeddings/llm_embeddings_metadata.json` - Metadata
- `embeddings/llm_data_mapping.csv` - Data mapping

### Sentence-Transformers Scripts

### 3. `generate_embeddings_sentencetransformers_human.py`
Generates embeddings for text, human labels, and SNAFU categories using local sentence-transformers.

**Features:**
- **Task-specific prompts**: Uses specialized prompts to improve semantic matching
  - Words: `"Represent this word for semantic matching with a subcategory label in the category "animals": "`
  - Labels: `"Represent this subcategory label for semantic matching with words from the "animals" category: "`
- Interactive model selection from 6 sentence-transformer models
- **No API costs**: Runs locally without API dependencies
- Text normalization for consistent embeddings
- Same filtering logic as Together AI scripts for consistency
- Batch processing for efficiency

**Output files:**
- `embeddings/sentencetransformers_human_embeddings.pkl` - Combined embeddings file
- `embeddings/sentencetransformers_human_embeddings_metadata.json` - Metadata
- `embeddings/human_data_mapping.csv` - Data mapping (compatible with analysis scripts)

### 4. `generate_embeddings_sentencetransformers_llm.py`
Generates embeddings for LLM labels using sentence-transformers.

**Features:**
- **Consistent prompts**: Uses same label prompt as human script for comparable embeddings
- Interactive model selection (should match human embedding model)
- Local processing without API dependencies
- Compatible with existing sentence-transformers human embeddings

**Output files:**
- `embeddings/sentencetransformers_llm_embeddings.pkl` - LLM embeddings file
- `embeddings/sentencetransformers_llm_embeddings_metadata.json` - Metadata
- `embeddings/llm_data_mapping.csv` - Data mapping (compatible with analysis scripts)

### Analysis Scripts

### 5. `semantic_distance_analysis.py`
Analyzes semantic distances and creates visualizations.

**Features:**
- **Works with both embedding types**: Automatically detects Together AI or sentence-transformers embeddings
- **Player-level analysis**: Calculates mean distances per player, then computes means and SEM across players
- Aligns datasets to ensure consistent comparisons
- Calculates cosine distances using scipy
- Creates bar plots with meaningful error bars showing between-player variability
- Generates summary statistics

**Output files:**
- `analysis_output/plots/semantic_distance/semantic_distance_barplot_*.png` - Main bar plot (player-level means ± SEM)
- `analysis_output/plots/semantic_distance/semantic_distance_by_category_*.png` - Category comparison
- `analysis_output/statistics/semantic_distance/overall_statistics_*.json` - Overall stats
- `analysis_output/statistics/semantic_distance/category_statistics_*.csv` - Category stats
- `analysis_output/statistics/semantic_distance/detailed_distances_*.csv` - Individual word-level calculations
- `analysis_output/statistics/semantic_distance/player_means_*.csv` - **NEW**: Per-player mean distances

### 6. `visualize_word_label_distances.py`
Creates detailed visualizations for the first player's word-to-label distances.

**Features:**
- Shows distances between each word and its corresponding labels
- Creates grouped bar plot, heatmap with labels, and summary statistics
- Displays original text for readability while using normalized text for accurate distance calculations
- Shows which label type is semantically closest to each word

**Output files:**
- `analysis_output/plots/word_label_distances/word_label_distances_comparison.png` - Grouped bar plot
- `analysis_output/plots/word_label_distances/word_label_distances_heatmap.png` - Heatmap with labels and distances
- `analysis_output/plots/word_label_distances/closest_label_distribution.png` - Pie chart of closest label types
- `analysis_output/plots/word_label_distances/detailed_word_label_distances.csv` - All calculations

## Available Embedding Models

### Together AI Models
When running Together AI embedding scripts, you can choose from:

1. **togethercomputer/m2-bert-80M-32k-retrieval** (default)
   - Smaller, faster, good for experimentation
2. **BAAI/bge-large-en-v1.5**
   - Larger model, potentially better quality embeddings
3. **BAAI/bge-base-en-v1.5-vllm**
   - Optimized version, good balance of speed/quality
4. **intfloat/multilingual-e5-large-instruct**
   - Instruction-tuned, may work better for complex label types

### Sentence-Transformers Models
When running sentence-transformers scripts, you can choose from:

1. **all-mpnet-base-v2** (default)
   - High-quality general-purpose model, 768 dimensions
2. **all-MiniLM-L6-v2**
   - Faster, smaller model, 384 dimensions, good for experimentation
3. **all-MiniLM-L12-v2**
   - Balance of speed and quality, 384 dimensions
4. **paraphrase-multilingual-mpnet-base-v2**
   - Multilingual support, 768 dimensions
5. **sentence-transformers/all-mpnet-base-v2**
   - Alternative path to model #1
6. **sentence-transformers/all-MiniLM-L6-v2**
   - Alternative path to model #2

## Usage

### Step 1: Generate Human Embeddings
```bash
uv run python generate_embeddings_human.py
```
- Select your CSV file when prompted
- Choose an embedding model (or press Enter for default)
- This creates embeddings for text, human labels, and SNAFU categories

### Step 2: Generate LLM Embeddings  
```bash
uv run python generate_embeddings_llm.py
```
- Select the same CSV file
- Choose the same embedding model for consistency
- This creates embeddings for LLM labels

### Step 3: Run Distance Analysis
```bash
uv run python semantic_distance_analysis.py
```
- Select the embedding files when prompted
- This calculates distances and creates the main comparison plots

### Step 4: Visualize Individual Player (Optional)
```bash
uv run python visualize_word_label_distances.py
```
- Select the embedding files when prompted
- This creates detailed visualizations for the first player's data

## Requirements

- `together` - Together AI Python client
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` - Plotting
- `seaborn` - Statistical plotting
- `scipy` - Cosine distance calculation
- `python-dotenv` - Environment variable loading

## Data Requirements

Your CSV file must contain these columns:
- `text` - The text to analyze
- `label` - Human-provided labels
- `snafu_gt_label` - SNAFU ground truth category labels  
- `labelLLM` - LLM-provided labels
- `playerID` - Participant identifier
- `category` - Experimental category
- `word_index` - Word position index

Optional columns for filtering:
- `switch` - Switch indicators (for participant filtering)
- `sourceParticipantId` - For duplicate removal

## Key Features

### Text Normalization
All text is normalized before embedding generation:
- Converts to lowercase
- Removes punctuation (hyphens, apostrophes, commas, periods)
- Removes whitespace
- This ensures consistent embeddings regardless of text formatting

### Distance Calculation
- Uses scipy's cosine distance function
- Distance range: 0-2 (lower = more similar)
- 0 = identical, 1 = orthogonal, 2 = opposite

### Backward Compatibility
- Scripts automatically handle both normalized and original embeddings
- Visualization script includes normalization for older embedding files

## Notes

- The pipeline maintains compatibility with both Anthropic and Together providers as specified in CLAUDE.md
- All scripts use the same participant filtering logic as `analysis_pipeline.py` for consistency
- Embeddings are saved as pickle files to preserve exact float precision
- The pipeline handles missing data gracefully and provides detailed progress reporting
- Random seeds are set for reproducible duplicate removal (seed=42)
- Model selection is consistent across human and LLM embedding generation

## Troubleshooting

- Make sure you have a valid Together AI API key set in your environment (via .env file)
- Ensure your CSV file contains all required columns
- Check that you have sufficient API quota for embedding generation
- Verify file paths when selecting embedding files for analysis
- Use the same embedding model for both human and LLM embeddings for consistency
- If embeddings were generated with different models, regenerate them with the same model