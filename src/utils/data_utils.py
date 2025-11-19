"""
Data utility functions.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def save_cohort(cohort_df: pd.DataFrame, output_path: Path,
                format: str = 'csv') -> str:
    """
    Save cohort to file.

    Args:
        cohort_df: Cohort DataFrame
        output_path: Output file path
        format: File format ('csv', 'parquet', 'pickle')

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'csv':
        cohort_df.to_csv(output_path, index=False)
    elif format == 'parquet':
        cohort_df.to_parquet(output_path, index=False)
    elif format == 'pickle':
        cohort_df.to_pickle(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Cohort saved to: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return str(output_path)


def load_cohort(input_path: Path, format: str = None) -> pd.DataFrame:
    """
    Load cohort from file.

    Args:
        input_path: Input file path
        format: File format (auto-detected if None)

    Returns:
        Cohort DataFrame
    """
    input_path = Path(input_path)

    if format is None:
        format = input_path.suffix[1:]  # Remove leading dot

    logger.info(f"Loading cohort from: {input_path}")

    if format == 'csv':
        df = pd.read_csv(input_path)
    elif format == 'parquet':
        df = pd.read_parquet(input_path)
    elif format in ['pickle', 'pkl']:
        df = pd.read_pickle(input_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Loaded {len(df):,} records")

    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.

    Args:
        df: DataFrame to optimize

    Returns:
        Optimized DataFrame
    """
    logger.info("Optimizing DataFrame dtypes...")

    initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

    # Downcast integers
    int_columns = df.select_dtypes(include=['int64']).columns
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # Downcast floats
    float_columns = df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Convert object columns with few unique values to category
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')

    final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

    logger.info(f"Memory usage reduced: {initial_memory:.2f} MB -> {final_memory:.2f} MB")
    logger.info(f"Reduction: {(initial_memory - final_memory) / initial_memory * 100:.1f}%")

    return df


def create_summary_statistics(cohort_df: pd.DataFrame) -> Dict:
    """
    Create comprehensive summary statistics for cohort.

    Args:
        cohort_df: Cohort DataFrame

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_cases': len(cohort_df),
        'unique_subjects': cohort_df['subject_id'].nunique(),
        'unique_studies': cohort_df['study_id'].nunique(),
        'columns': list(cohort_df.columns),
        'dtypes': cohort_df.dtypes.astype(str).to_dict()
    }

    # Numeric columns statistics
    numeric_cols = cohort_df.select_dtypes(include=[np.number]).columns
    summary['numeric_stats'] = {}
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': float(cohort_df[col].mean()) if cohort_df[col].notna().any() else None,
            'median': float(cohort_df[col].median()) if cohort_df[col].notna().any() else None,
            'std': float(cohort_df[col].std()) if cohort_df[col].notna().any() else None,
            'min': float(cohort_df[col].min()) if cohort_df[col].notna().any() else None,
            'max': float(cohort_df[col].max()) if cohort_df[col].notna().any() else None,
            'missing': int(cohort_df[col].isna().sum())
        }

    # Categorical columns statistics
    categorical_cols = cohort_df.select_dtypes(include=['object', 'category']).columns
    summary['categorical_stats'] = {}
    for col in categorical_cols:
        value_counts = cohort_df[col].value_counts()
        summary['categorical_stats'][col] = {
            'unique_values': int(cohort_df[col].nunique()),
            'top_values': value_counts.head(10).to_dict(),
            'missing': int(cohort_df[col].isna().sum())
        }

    return summary


def export_summary_to_json(summary: Dict, output_path: Path) -> str:
    """
    Export summary statistics to JSON file.

    Args:
        summary: Summary dictionary
        output_path: Output file path

    Returns:
        Path to saved file
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Summary exported to: {output_path}")

    return str(output_path)


def create_data_dictionary(cohort_df: pd.DataFrame,
                          output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Create a data dictionary describing all columns.

    Args:
        cohort_df: Cohort DataFrame
        output_path: Optional path to save data dictionary

    Returns:
        Data dictionary DataFrame
    """
    data_dict = []

    for col in cohort_df.columns:
        info = {
            'Column': col,
            'Type': str(cohort_df[col].dtype),
            'Non-Null Count': cohort_df[col].notna().sum(),
            'Null Count': cohort_df[col].isna().sum(),
            'Null %': f"{cohort_df[col].isna().sum() / len(cohort_df) * 100:.2f}%",
            'Unique Values': cohort_df[col].nunique()
        }

        # Add sample values
        sample_values = cohort_df[col].dropna().unique()[:3]
        info['Sample Values'] = ', '.join(map(str, sample_values))

        data_dict.append(info)

    dd_df = pd.DataFrame(data_dict)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dd_df.to_csv(output_path, index=False)
        logger.info(f"Data dictionary saved to: {output_path}")

    return dd_df
