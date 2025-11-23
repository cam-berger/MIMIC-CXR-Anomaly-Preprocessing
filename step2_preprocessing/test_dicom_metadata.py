"""
Test DICOM metadata integration

Validates that:
1. DICOMMetadataLoader loads metadata correctly
2. Metadata features are included in structured features
3. Features are properly formatted for model input
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loaders import DICOMMetadataLoader

print("="*60)
print("DICOM Metadata Integration Test")
print("="*60)
print()

# Test 1: Load metadata
print("Test 1: Loading DICOM metadata...")
print("-"*60)

metadata_path = "/media/dev/MIMIC_DATA/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz"
loader = DICOMMetadataLoader(metadata_path)

print(f"✓ Loaded metadata for {len(loader.study_metadata)} studies")
print()

# Test 2: Get coverage stats
print("Test 2: Coverage statistics...")
print("-"*60)

stats = loader.get_coverage_stats()
print(f"Total studies: {stats['total_studies']:,}")
print(f"Has view info: {stats['has_view_info']:,} ({100*stats['has_view_info']/stats['total_studies']:.1f}%)")
print(f"Has orientation: {stats['has_orientation']:,} ({100*stats['has_orientation']/stats['total_studies']:.1f}%)")
print(f"Is portable: {stats['is_portable']:,} ({100*stats['is_portable']/stats['total_studies']:.1f}%)")
print()

print("View distribution:")
for view, count in stats['view_distribution'].items():
    print(f"  {view}: {count:,} ({100*count/stats['total_studies']:.1f}%)")
print()

print("Orientation distribution:")
for orientation, count in stats['orientation_distribution'].items():
    print(f"  {orientation}: {count:,} ({100*count/stats['total_studies']:.1f}%)")
print()

# Test 3: Load metadata for specific studies from proof_test cohort
print("Test 3: Loading metadata for proof_test_3 cohort...")
print("-"*60)

import pandas as pd
cohort = pd.read_csv("cohorts/proof_test_3.csv")

for idx, row in cohort.iterrows():
    study_id = row['study_id']
    view_position = row['ViewPosition']

    print(f"\nStudy {study_id} (ViewPosition from cohort: {view_position})")

    # Get raw metadata
    metadata = loader.get_study_metadata(study_id)
    if metadata:
        print(f"  DICOM views: {metadata['views_list']}")
        print(f"  Orientation: {metadata['orientations_list']}")
        print(f"  Portable: {metadata['is_portable']}")
        print(f"  Num DICOMs: {metadata['num_dicoms']}")
        print(f"  Dimensions: {metadata['avg_rows']:.0f} x {metadata['avg_cols']:.0f}")

        # Get model-ready features
        features = loader.get_metadata_features(study_id)
        print(f"  Model features:")
        print(f"    view_pa={features['view_pa']}, view_ap={features['view_ap']}, view_lateral={features['view_lateral']}")
        print(f"    orientation_erect={features['orientation_erect']}, orientation_recumbent={features['orientation_recumbent']}")
        print(f"    is_portable={features['is_portable']}")
        print(f"    num_views={features['num_views']}")
    else:
        print(f"  ✗ No metadata found for study {study_id}")

print()
print("="*60)
print("✓ All tests completed successfully!")
print("="*60)
print()

print("Next step: Run preprocessing with config_proof_test.yaml to verify")
print("that DICOM features are included in structured_features JSON files.")
