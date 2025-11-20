# Step 2 Notebooks

Interactive Jupyter notebooks for analysis, testing, and visualization of the preprocessing pipeline.

## Available Notebooks

### `RAG_Implementation.ipynb`
**Purpose**: Demonstrates and tests the complete RAG (Retrieval-Augmented Generation) text processing pipeline

**Features**:
- Medical NER with scispacy (en_core_sci_md)
- Entity-based + semantic sentence retrieval
- Claude Sonnet 4.5 summarization via LangChain
- ClinicalBERT tokenization
- Includes configuration verification and results checklist

**Requirements**: ANTHROPIC_API_KEY environment variable

### `analyze_test_results.ipynb`
**Purpose**: Analyzes results from test runs of the preprocessing pipeline

**Features**:
- Visualizes processing times per modality
- Analyzes missing data patterns
- Shows sample processed images
- Generates summary statistics

### `explore_cohort_outputs.ipynb`
**Purpose**: Explores the normal cohort data from Step 1

**Features**:
- Cohort demographics and statistics
- Distribution analyses
- Data quality checks

## Usage

```bash
# From step2_preprocessing directory
cd notebooks
jupyter notebook

# Or launch specific notebook
jupyter notebook RAG_Implementation.ipynb
```

## Notes

- All notebooks expect to be run from the `step2_preprocessing/` directory
- Relative paths are configured accordingly (e.g., `../config/config.yaml`)
- Large output cells may be cleared before committing
