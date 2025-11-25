"""
Text preprocessing for radiology reports and clinical notes.

Processing pipeline:
1. Load radiology reports from CXR-PRO
2. Clean and normalize text
3. Build clinical context from cohort features
4. (Optional) Summarize using Claude API with full clinical context
5. Tokenize using ClinicalBERT

Output Parquet schema:
- study_id, subject_id (keys)
- report (original text)
- report_clean (cleaned text)
- clinical_context (formatted context string)
- summary (Claude summary with context, if enabled)
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


def format_clinical_context(row: pd.Series) -> str:
    """
    Format clinical context from cohort row for summarization prompt.

    Includes: demographics, triage vitals, labs, diagnoses, procedures.
    """
    context_parts = []

    # Demographics
    demographics = []
    if "anchor_age" in row and pd.notna(row.get("anchor_age")):
        demographics.append(f"{int(row['anchor_age'])} year old")
    if "gender" in row and pd.notna(row.get("gender")):
        gender_str = "male" if row["gender"] == "M" else "female"
        demographics.append(gender_str)
    if demographics:
        context_parts.append("Patient: " + " ".join(demographics))

    # Chief complaint
    if "triage_chiefcomplaint" in row and pd.notna(row.get("triage_chiefcomplaint")):
        cc = str(row["triage_chiefcomplaint"]).strip()
        if cc:
            context_parts.append(f"Chief complaint: {cc}")

    # Triage vitals
    vitals = []
    vital_mappings = [
        ("triage_temperature", "Temp", "°F"),
        ("triage_heartrate", "HR", "bpm"),
        ("triage_resprate", "RR", "/min"),
        ("triage_o2sat", "SpO2", "%"),
        ("triage_sbp", "SBP", "mmHg"),
        ("triage_dbp", "DBP", "mmHg"),
    ]
    for col, name, unit in vital_mappings:
        if col in row and pd.notna(row.get(col)):
            val = row[col]
            if isinstance(val, float):
                if col == "triage_temperature":
                    vitals.append(f"{name} {val:.1f}{unit}")
                else:
                    vitals.append(f"{name} {int(val)}{unit}")
    if vitals:
        context_parts.append("Vitals: " + ", ".join(vitals))

    # Acuity
    if "triage_acuity" in row and pd.notna(row.get("triage_acuity")):
        acuity = int(row["triage_acuity"])
        acuity_desc = {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Less urgent", 5: "Non-urgent"}
        context_parts.append(f"Triage acuity: {acuity} ({acuity_desc.get(acuity, 'Unknown')})")

    # Labs (if available)
    lab_mappings = [
        ("lab_wbc_mean", "WBC", "K/uL"),
        ("lab_hemoglobin_mean", "Hgb", "g/dL"),
        ("lab_platelets_mean", "Plt", "K/uL"),
        ("lab_glucose_mean", "Glucose", "mg/dL"),
        ("lab_creatinine_mean", "Cr", "mg/dL"),
        ("lab_sodium_mean", "Na", "mEq/L"),
        ("lab_potassium_mean", "K", "mEq/L"),
        ("lab_lactate_mean", "Lactate", "mmol/L"),
        ("lab_troponin_mean", "Troponin", "ng/mL"),
    ]
    labs = []
    for col, name, unit in lab_mappings:
        if col in row and pd.notna(row.get(col)):
            val = row[col]
            if isinstance(val, (int, float)) and not np.isnan(val):
                labs.append(f"{name} {val:.1f}")
    if labs:
        context_parts.append("Labs: " + ", ".join(labs))

    # ED Diagnoses
    if "ed_diagnoses" in row and pd.notna(row.get("ed_diagnoses")):
        dx = str(row["ed_diagnoses"]).strip()
        if dx and dx.lower() != "nan":
            # Limit to first few codes
            codes = dx.split(",")[:5]
            context_parts.append(f"ED diagnoses: {', '.join(codes)}")

    # Hospital Diagnoses (if admitted)
    if "hospital_diagnoses" in row and pd.notna(row.get("hospital_diagnoses")):
        dx = str(row["hospital_diagnoses"]).strip()
        if dx and dx.lower() != "nan":
            codes = dx.split(",")[:5]
            context_parts.append(f"Hospital diagnoses: {', '.join(codes)}")

    # Procedures
    if "procedures" in row and pd.notna(row.get("procedures")):
        proc = str(row["procedures"]).strip()
        if proc and proc.lower() != "nan":
            codes = proc.split(",")[:3]
            context_parts.append(f"Procedures: {', '.join(codes)}")

    # ED disposition
    if "disposition" in row and pd.notna(row.get("disposition")):
        disp = str(row["disposition"]).strip()
        if disp:
            context_parts.append(f"Disposition: {disp}")

    return "\n".join(context_parts)


class TextPreprocessor:
    """Process radiology reports and clinical notes with clinical context."""

    SUMMARIZATION_PROMPT = """You are a medical expert reviewing a chest X-ray case.

CLINICAL CONTEXT:
{context}

RADIOLOGY REPORT:
{report}

Based on the clinical context and radiology report above, provide a concise clinical summary (2-3 sentences) that:
1. Synthesizes the key imaging findings with the clinical presentation
2. Notes any correlation or discordance between symptoms and imaging
3. Highlights clinically significant findings or reassuring normal results

Summary:"""

    SUMMARIZATION_PROMPT_NO_CONTEXT = """You are a medical expert reviewing radiology reports.
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
        include_context: bool = True,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """
        Process text data for entire cohort.

        Args:
            cohort: Cohort DataFrame with all features (demographics, vitals, labs, etc.)
            output_path: Output parquet file path
            enable_summarization: Use Claude for summarization (uses config if None)
            include_context: Include clinical context in summarization prompt
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

        # Start with full cohort to preserve all columns for context
        result = cohort.copy()

        # Merge reports only if 'report' column doesn't already exist
        if "report" not in result.columns:
            result = result.merge(
                reports_df[["study_id", "report"]],
                on="study_id",
                how="left",
            )
        else:
            logger.info("Report column already exists in cohort, skipping merge")

        # Clean reports
        logger.info("Cleaning report text...")
        result["report_clean"] = result["report"].apply(clean_report_text)

        # Build clinical context for each row
        if include_context:
            logger.info("Building clinical context...")
            result["clinical_context"] = result.apply(format_clinical_context, axis=1)
        else:
            result["clinical_context"] = ""

        # Summarization (if enabled)
        if enable_summarization and self.anthropic_client:
            logger.info("Generating summaries with Claude (with clinical context)...")
            result = self._add_summaries(result, batch_size, include_context)
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

        # Select output columns
        output_cols = [
            "subject_id", "study_id",
            "report", "report_clean", "clinical_context",
            "summary", "tokens", "token_count", "has_report"
        ]
        result = result[[c for c in output_cols if c in result.columns]]

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
        include_context: bool = True,
    ) -> pd.DataFrame:
        """Add Claude-generated summaries with clinical context."""
        summaries = []

        for i in tqdm(range(0, len(result), batch_size), desc="Summarizing"):
            batch_df = result.iloc[i : i + batch_size]
            batch_summaries = []

            for _, row in batch_df.iterrows():
                report = row.get("report_clean", "")
                context = row.get("clinical_context", "") if include_context else ""

                if not report or report == "":
                    batch_summaries.append("")
                    continue

                try:
                    summary = self._summarize_with_context(report, context)
                    batch_summaries.append(summary)
                except Exception as e:
                    logger.warning(f"Summarization failed: {e}")
                    batch_summaries.append(report[:500])  # Fallback to truncated report

            summaries.extend(batch_summaries)

        result["summary"] = summaries
        return result

    def _summarize_with_context(self, report: str, context: str) -> str:
        """Summarize a report with clinical context using Claude."""
        if not self.anthropic_client:
            return report[:500]

        # Choose prompt based on whether we have context
        if context and context.strip():
            prompt = self.SUMMARIZATION_PROMPT.format(
                context=context,
                report=report
            )
        else:
            prompt = self.SUMMARIZATION_PROMPT_NO_CONTEXT.format(report=report)

        try:
            response = self.anthropic_client.messages.create(
                model=self.config.claude_model,
                max_tokens=300,  # Slightly more for context-aware summaries
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
