#!/usr/bin/env python3
"""
Generate Stratified 200-Sample Validation Cohort
Samples from full validation cohort with balanced demographics
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_age_bins(df: pd.DataFrame, age_column: str = 'anchor_age') -> pd.Series:
    """Create age bins for stratification"""
    bins = [0, 30, 45, 60, 75, 120]
    labels = ['18-30', '31-45', '46-60', '61-75', '76+']
    return pd.cut(df[age_column], bins=bins, labels=labels, include_lowest=True)


def stratified_sample(
    df: pd.DataFrame,
    n_samples: int = 200,
    stratify_columns: list = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate stratified sample from dataframe

    Args:
        df: Input dataframe
        n_samples: Number of samples to generate
        stratify_columns: Columns to stratify by (e.g., ['gender', 'age_bin'])
        random_seed: Random seed for reproducibility

    Returns:
        Stratified sample dataframe
    """
    if stratify_columns is None:
        stratify_columns = ['gender']

    # Create stratification key
    df['_strata'] = df[stratify_columns].astype(str).agg('_'.join, axis=1)

    # Calculate samples per stratum (proportional)
    strata_counts = df['_strata'].value_counts()
    strata_ratios = strata_counts / len(df)
    target_counts = (strata_ratios * n_samples).round().astype(int)

    # Adjust if sum doesn't equal n_samples (due to rounding)
    diff = n_samples - target_counts.sum()
    if diff != 0:
        # Add/subtract from largest stratum
        largest_stratum = target_counts.idxmax()
        target_counts[largest_stratum] += diff

    # Sample from each stratum
    np.random.seed(random_seed)
    sampled_indices = []

    for stratum, target_count in target_counts.items():
        stratum_data = df[df['_strata'] == stratum]
        available_count = len(stratum_data)

        # Sample with replacement if not enough samples in stratum
        replace = available_count < target_count
        if replace:
            print(f"  WARNING: Stratum '{stratum}' has only {available_count} samples "
                  f"but needs {target_count}. Sampling with replacement.")

        sampled = stratum_data.sample(n=target_count, replace=replace, random_state=random_seed)
        sampled_indices.extend(sampled.index.tolist())

    # Get final sample and remove temporary column
    result = df.loc[sampled_indices].copy()
    result = result.drop(columns=['_strata'])

    return result


def print_statistics(df: pd.DataFrame, label: str = "Dataset"):
    """Print summary statistics"""
    print(f"\n{'='*70}")
    print(f"{label} Statistics")
    print(f"{'='*70}")
    print(f"Total samples: {len(df):,}")

    # Gender distribution
    if 'gender' in df.columns:
        print(f"\nGender distribution:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            pct = 100 * count / len(df)
            print(f"  {gender}: {count:,} ({pct:.1f}%)")

    # Age distribution
    if 'anchor_age' in df.columns:
        print(f"\nAge distribution:")
        print(f"  Mean: {df['anchor_age'].mean():.1f} years")
        print(f"  Median: {df['anchor_age'].median():.1f} years")
        print(f"  Range: {df['anchor_age'].min():.0f} - {df['anchor_age'].max():.0f} years")

        # Age bins
        age_bins = create_age_bins(df)
        print(f"\n  Age groups:")
        for age_group, count in age_bins.value_counts().sort_index().items():
            pct = 100 * count / len(df)
            print(f"    {age_group}: {count:,} ({pct:.1f}%)")

    # Study characteristics
    if 'num_images' in df.columns:
        print(f"\nImages per study:")
        print(f"  Mean: {df['num_images'].mean():.1f}")
        print(f"  Median: {df['num_images'].median():.1f}")
        print(f"  Range: {df['num_images'].min():.0f} - {df['num_images'].max():.0f}")

    if 'time_to_cxr_hours' in df.columns:
        print(f"\nTime to CXR (hours):")
        print(f"  Mean: {df['time_to_cxr_hours'].mean():.1f}")
        print(f"  Median: {df['time_to_cxr_hours'].median():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified validation cohort for Lambda deployment"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input cohort CSV path (e.g., output/output_test/cohorts/normal_cohort_validation.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output cohort CSV path (e.g., step2_preprocessing/cohorts/validation_subset_200.csv)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=200,
        help='Number of samples to generate (default: 200)'
    )
    parser.add_argument(
        '--stratify-by',
        nargs='+',
        default=['gender'],
        help='Columns to stratify by (default: gender). Options: gender, anchor_age'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Stratified Cohort Generation")
    print(f"{'='*70}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Target samples: {args.n_samples}")
    print(f"Stratify by: {', '.join(args.stratify_by)}")
    print(f"Random seed: {args.random_seed}")

    # Load input cohort
    print(f"\nLoading input cohort...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} samples")

    # Handle anchor_age stratification
    stratify_columns = args.stratify_by.copy()
    if 'anchor_age' in stratify_columns:
        print(f"\nCreating age bins for stratification...")
        df['age_bin'] = create_age_bins(df)
        stratify_columns[stratify_columns.index('anchor_age')] = 'age_bin'

    # Print original statistics
    print_statistics(df, "Original Cohort")

    # Generate stratified sample
    print(f"\nGenerating stratified sample of {args.n_samples} samples...")
    sampled_df = stratified_sample(
        df,
        n_samples=args.n_samples,
        stratify_columns=stratify_columns,
        random_seed=args.random_seed
    )

    # Remove temporary age_bin column if it exists
    if 'age_bin' in sampled_df.columns:
        sampled_df = sampled_df.drop(columns=['age_bin'])

    # Print sampled statistics
    print_statistics(sampled_df, "Sampled Cohort")

    # Save output
    print(f"\nSaving to: {args.output}")
    sampled_df.to_csv(args.output, index=False)

    # Verify output
    if output_path.exists():
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"✓ Cohort saved successfully ({file_size:.1f} KB)")
    else:
        print(f"ERROR: Failed to save output file")
        return 1

    print(f"\n{'='*70}")
    print("Cohort generation complete!")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
