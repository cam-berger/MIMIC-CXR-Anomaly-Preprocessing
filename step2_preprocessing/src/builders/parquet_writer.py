"""
Parquet writer for structured and text features.

Stores tabular data in columnar Parquet format optimized for:
- Fast filtering and queries (DuckDB, pandas)
- Efficient compression
- Schema validation
- Batch organization
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class ParquetWriter:
    """
    Writer for storing structured and text features in Parquet format.

    Features:
    - Columnar storage for fast queries
    - Schema validation
    - Compression (gzip, snappy, etc.)
    - Batch partitioning
    - Metadata tracking
    """

    def __init__(
        self,
        output_dir: Path,
        compression: str = "gzip",
        schema: Optional[Dict] = None
    ):
        """
        Initialize Parquet writer.

        Args:
            output_dir: Directory for output Parquet files
            compression: Compression algorithm ("gzip", "snappy", "lz4", "zstd")
            schema: Optional schema definition for validation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self.schema = schema

        logger.info(f"Initialized ParquetWriter")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Compression: {self.compression}")

    def prepare_sample_row(self, sample: Dict) -> Dict[str, Any]:
        """
        Prepare a single sample's features for Parquet storage.

        Flattens nested dictionaries into columnar format.

        Args:
            sample: Sample dictionary with various features

        Returns:
            Flattened dictionary suitable for DataFrame row
        """
        row = {}

        # Metadata (always included)
        row['sample_id'] = sample['sample_id']
        row['subject_id'] = sample.get('subject_id')
        row['study_id'] = sample.get('study_id')
        row['hadm_id'] = sample.get('hadm_id')
        row['study_datetime'] = sample.get('study_datetime')
        row['ed_intime'] = sample.get('ed_intime')

        # Structured data (vitals, labs, medications)
        structured = sample.get('structured', {})
        for feature_name, feature_data in structured.items():
            if isinstance(feature_data, dict):
                # Flatten nested dict
                if 'value' in feature_data:
                    value = feature_data['value']
                    # Convert NOT_DONE to None for type consistency
                    row[feature_name] = None if value == "NOT_DONE" else value
                elif 'last_value' in feature_data:
                    # Convert NOT_DONE to None for numeric columns
                    last_val = feature_data['last_value']
                    mean_val = feature_data.get('mean_value')

                    row[f"{feature_name}_last"] = None if last_val == "NOT_DONE" else last_val
                    row[f"{feature_name}_mean"] = None if mean_val == "NOT_DONE" else mean_val
                    row[f"{feature_name}_count"] = feature_data.get('measurement_count')
                elif 'present' in feature_data:
                    # Medication features
                    row[f"{feature_name}_present"] = feature_data['present']
                    row[f"{feature_name}_count"] = feature_data.get('count', 0)
            else:
                # Simple value
                value = feature_data
                row[feature_name] = None if value == "NOT_DONE" else value

        # Text data (summaries, tokens)
        text = sample.get('text', {})
        if text:
            row['text_summary'] = text.get('summary', '')
            row['text_entity_count'] = text.get('entity_count', 0)
            row['text_has_content'] = len(text.get('summary', '')) > 0

            # Store token IDs as JSON string (Parquet doesn't handle lists well)
            if 'token_ids' in text:
                row['text_token_ids'] = json.dumps(text['token_ids'])

        # Processing errors
        errors = sample.get('errors', [])
        row['has_errors'] = len(errors) > 0
        row['error_count'] = len(errors)
        if errors:
            row['errors'] = json.dumps(errors)

        return row

    def write_batch(
        self,
        batch_id: int,
        samples: List[Dict],
        split: str = "train"
    ) -> Path:
        """
        Write a batch of samples to Parquet file.

        Args:
            batch_id: Batch identifier
            samples: List of sample dictionaries
            split: Dataset split ("train" or "val")

        Returns:
            Path to created Parquet file
        """
        logger.info(f"Writing batch {batch_id} with {len(samples)} samples to Parquet")

        # Prepare rows
        rows = []
        for sample in samples:
            try:
                row = self.prepare_sample_row(sample)
                rows.append(row)
            except Exception as e:
                logger.error(f"  Failed to prepare sample {sample.get('sample_id')}: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(rows)

        logger.info(f"  DataFrame shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)[:20]}...")  # Show first 20 columns

        # Validate schema if provided
        if self.schema:
            self._validate_schema(df)

        # Create output path
        output_path = self.output_dir / split / f"batch_{batch_id:04d}" / "data.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to Parquet
        df.to_parquet(
            output_path,
            compression=self.compression,
            index=False,
            engine='pyarrow'  # Use pyarrow for better performance
        )

        file_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"  Parquet batch file created: {output_path}")
        logger.info(f"  File size: {file_size_mb:.2f} MB")

        # Write metadata
        metadata_path = output_path.parent / "metadata.json"
        self._write_metadata(metadata_path, batch_id, df, output_path)

        return output_path

    def _validate_schema(self, df: pd.DataFrame):
        """
        Validate DataFrame against schema.

        Args:
            df: DataFrame to validate
        """
        required_columns = self.schema.get('required_columns', [])
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            logger.warning(f"  Missing required columns: {missing_columns}")

        # Validate dtypes if specified
        if 'dtypes' in self.schema:
            for col, expected_dtype in self.schema['dtypes'].items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if actual_dtype != expected_dtype:
                        logger.warning(
                            f"  Column '{col}' dtype mismatch: "
                            f"expected {expected_dtype}, got {actual_dtype}"
                        )

    def _write_metadata(
        self,
        metadata_path: Path,
        batch_id: int,
        df: pd.DataFrame,
        parquet_path: Path
    ):
        """
        Write batch metadata to JSON file.

        Args:
            metadata_path: Path for metadata file
            batch_id: Batch identifier
            df: DataFrame that was written
            parquet_path: Path to Parquet file
        """
        metadata = {
            'batch_id': batch_id,
            'num_samples': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'file_size_bytes': parquet_path.stat().st_size,
            'file_size_mb': parquet_path.stat().st_size / 1024 / 1024,
            'compression': self.compression,

            # Data completeness
            'completeness': {
                col: {
                    'non_null': int(df[col].notna().sum()),
                    'null': int(df[col].isna().sum()),
                    'percent_complete': float(df[col].notna().sum() / len(df) * 100)
                }
                for col in ['text_summary', 'has_errors'] if col in df.columns
            }
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Metadata written: {metadata_path}")

    def compute_checksum(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum for Parquet file.

        Args:
            file_path: Path to Parquet file

        Returns:
            SHA256 hex digest
        """
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)

        return sha256.hexdigest()

    def get_batch_stats(self, file_path: Path) -> Dict:
        """
        Get statistics for a batch Parquet file.

        Args:
            file_path: Path to Parquet file

        Returns:
            Dictionary with batch statistics
        """
        df = pd.read_parquet(file_path)

        stats = {
            'num_samples': len(df),
            'num_columns': len(df.columns),
            'file_size_mb': file_path.stat().st_size / 1024 / 1024,
            'columns': list(df.columns),

            # Data completeness
            'text_coverage': float(df['text_has_content'].sum() / len(df) * 100) if 'text_has_content' in df.columns else 0,
            'error_rate': float(df['has_errors'].sum() / len(df) * 100) if 'has_errors' in df.columns else 0,
        }

        return stats


class ParquetReader:
    """
    Reader for loading data from pre-compiled Parquet files.

    Supports:
    - Fast filtering (WHERE clauses)
    - Column projection (SELECT specific columns)
    - Multi-batch datasets
    - DuckDB integration for SQL queries
    """

    def __init__(self, parquet_paths: List[Path]):
        """
        Initialize Parquet reader.

        Args:
            parquet_paths: List of paths to Parquet files
        """
        self.parquet_paths = [Path(p) for p in parquet_paths]

        logger.info(f"Initialized ParquetReader with {len(self.parquet_paths)} files")

    def load_all(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load all data from all Parquet files.

        Args:
            columns: Optional list of columns to load (column projection)

        Returns:
            Combined DataFrame
        """
        dfs = []

        for path in self.parquet_paths:
            df = pd.read_parquet(path, columns=columns)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} samples from {len(dfs)} files")

        return combined

    def load_sample(self, sample_id: str) -> Optional[pd.Series]:
        """
        Load a single sample by ID.

        Args:
            sample_id: Sample identifier

        Returns:
            Series with sample data or None if not found
        """
        for path in self.parquet_paths:
            df = pd.read_parquet(path)
            matches = df[df['sample_id'] == sample_id]

            if len(matches) > 0:
                return matches.iloc[0]

        logger.warning(f"Sample {sample_id} not found")
        return None

    def query(self, filter_expr: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Query data using pandas query syntax.

        Args:
            filter_expr: Filter expression (e.g., "subject_id == 12345")
            columns: Optional columns to load

        Returns:
            Filtered DataFrame
        """
        df = self.load_all(columns=columns)
        result = df.query(filter_expr)

        logger.info(f"Query returned {len(result)} samples")

        return result
