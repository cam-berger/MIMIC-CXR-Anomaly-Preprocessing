"""
Text preprocessing for radiology reports and clinical notes.

Processing pipeline:
1. Load radiology reports from CXR-PRO
2. Clean and normalize text
3. (Optional) Summarize using Claude API
4. Tokenize using ClinicalBERT

Output Parquet schema:
- study_id, subject_id (keys)
- report (original text)
- report_clean (cleaned text)
- summary (Claude summary, if enabled)
- tokens (tokenized IDs as list)
- token_count (number of tokens)
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from ..config import Settings, get_settings
from ..datasets import CXRPROLoader

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, tokenization disabled")

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    logger.warning("anthropic not available, summarization disabled")


def clean_report_text(text: str) -> str:
    """
    Clean and normalize radiology report text.

    - Remove extra whitespace
    - Normalize line breaks
    - Remove common artifacts
    """
    if not text or pd.isna(text):
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove common artifacts
    text = re.sub(r"_{2,}", "", text)  # Underscores
    text = re.sub(r"-{3,}", "", text)  # Dashes

    # Strip
    text = text.strip()

    return text


class TextPreprocessor:
    """Process radiology reports and clinical notes."""

    SUMMARIZATION_PROMPT = """You are a medical expert reviewing radiology reports.
Summarize the following chest X-ray report in 2-3 concise sentences.
Focus on key clinical findings and impressions.
If no significant findings, state "No acute cardiopulmonary findings."

Report:
{report}

Summary:"""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize preprocessor."""
        self.settings = settings or get_settings()
        self.config = self.settings.preprocessing

        self._cxr_pro_loader = None
        self._tokenizer = None
        self._anthropic_client = None

    @property
    def cxr_pro_loader(self) -> CXRPROLoader:
        """CXR-PRO loader (lazy initialization)."""
        if self._cxr_pro_loader is None:
            self._cxr_pro_loader = CXRPROLoader(self.settings.paths)
        return self._cxr_pro_loader

    @property
    def tokenizer(self):
        """ClinicalBERT tokenizer (lazy initialization)."""
        if self._tokenizer is None and HAS_TRANSFORMERS:
            logger.info(f"Loading tokenizer: {self.config.tokenizer_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_model)
        return self._tokenizer

    @property
    def anthropic_client(self):
        """Anthropic client (lazy initialization)."""
        if self._anthropic_client is None and HAS_ANTHROPIC:
            api_key = self.settings.anthropic_api_key
            if api_key:
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    def process_cohort(
        self,
        cohort: pd.DataFrame,
        output_path: Path,
        enable_summarization: Optional[bool] = None,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """
        Process text data for entire cohort.

        Args:
            cohort: Cohort DataFrame (must have subject_id, study_id)
            output_path: Output parquet file path
            enable_summarization: Use Claude for summarization (uses config if None)
            batch_size: Batch size for API calls

        Returns:
            Processed text data DataFrame
        """
        logger.info(f"Processing text data for {len(cohort):,} samples...")

        enable_summarization = (
            enable_summarization
            if enable_summarization is not None
            else self.config.use_claude_summarization
        )

        # Get reports for all studies
        study_ids = set(cohort["study_id"])
        reports_df = self.cxr_pro_loader.get_reports_for_studies(study_ids)

        logger.info(f"Found reports for {len(reports_df):,} / {len(study_ids):,} studies")

        # Merge with cohort
        result = cohort[["subject_id", "study_id"]].merge(
            reports_df[["study_id", "report"]],
            on="study_id",
            how="left",
        )

        # Clean reports
        logger.info("Cleaning report text...")
        result["report_clean"] = result["report"].apply(clean_report_text)

        # Summarization (if enabled)
        if enable_summarization and self.anthropic_client:
            logger.info("Generating summaries with Claude...")
            result = self._add_summaries(result, batch_size)
        else:
            result["summary"] = result["report_clean"]  # Use cleaned report as summary

        # Tokenization
        if self.tokenizer:
            logger.info("Tokenizing text...")
            result = self._add_tokens(result)
        else:
            result["tokens"] = None
            result["token_count"] = 0

        # Add availability flag
        result["has_report"] = result["report"].notna() & (result["report"] != "")

        # Save to parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tokens list to string for parquet storage
        if "tokens" in result.columns and result["tokens"].dtype == object:
            result["tokens"] = result["tokens"].apply(
                lambda x: ",".join(map(str, x)) if x is not None else ""
            )

        result.to_parquet(output_path, index=False)
        logger.info(f"Saved text data to {output_path}")

        return result

    def _add_summaries(
        self,
        result: pd.DataFrame,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """Add Claude-generated summaries."""
        summaries = []

        # Process in batches to handle rate limits
        reports = result["report_clean"].tolist()

        for i in tqdm(range(0, len(reports), batch_size), desc="Summarizing"):
            batch = reports[i : i + batch_size]
            batch_summaries = []

            for report in batch:
                if not report or report == "":
                    batch_summaries.append("")
                    continue

                try:
                    summary = self._summarize_report(report)
                    batch_summaries.append(summary)
                except Exception as e:
                    logger.warning(f"Summarization failed: {e}")
                    batch_summaries.append(report[:500])  # Fallback to truncated report

            summaries.extend(batch_summaries)

        result["summary"] = summaries
        return result

    def _summarize_report(self, report: str) -> str:
        """Summarize a single report using Claude."""
        if not self.anthropic_client:
            return report[:500]

        prompt = self.SUMMARIZATION_PROMPT.format(report=report)

        try:
            response = self.anthropic_client.messages.create(
                model=self.config.claude_model,
                max_tokens=200,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Claude API error: {e}")
            return report[:500]

    def _add_tokens(self, result: pd.DataFrame) -> pd.DataFrame:
        """Add tokenized representations."""
        if not self.tokenizer:
            result["tokens"] = None
            result["token_count"] = 0
            return result

        tokens_list = []
        counts = []

        for text in tqdm(result["summary"].fillna(""), desc="Tokenizing"):
            if not text:
                tokens_list.append([])
                counts.append(0)
                continue

            encoded = self.tokenizer.encode(
                text,
                max_length=self.config.max_text_length,
                truncation=True,
                padding=False,
            )
            tokens_list.append(encoded)
            counts.append(len(encoded))

        result["tokens"] = tokens_list
        result["token_count"] = counts

        return result

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode tokens back to text."""
        if not self.tokenizer or not tokens:
            return ""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
