"""
What Jev reads for each study (docs/V3_PLAN.md §3-4): the `state` text, built from an arm's blocks, and the 12
questions, one per finding, in one of two formats (§3 amendment):

    noul   yes/no questions; the answer is the probability of yes (the pre-registered prompt, unchanged)
    score  10 likelihood bands; the answer is the expected band, mapped to a likelihood

Jev never sees the X-ray in v3.0, so each question asks what the radiologist's report will say: the labels come
from the report. Jev's answers come rounded to 0.01. For a yes/no probability that ties most studies on rare
findings; a score is on a 0-9 scale, so about 9 times finer. Val picks the format (§7). Anything changed here
changes `prompt_hash`, so cached answers and runs never mix two prompts.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Optional

import numpy as np

from v2.common import LABELS

FORMATS = ("noul", "score")
PROMPT_VERSION = {"noul": 1, "score": 2}  # noul is version 1, the pre-registered prompt, byte for byte
MISSING = "none recorded"
LEVELS = tuple(f"{10 * i}-{10 * (i + 1)}%" for i in range(10))  # short: they repeat in all 12 questions
# Jev replaces an asset's key wherever it appears in `state`; TypeSafe recommends a UUID per image (v3.1).
IMAGE_PLACEHOLDER = "6f1c2a7e-3b9d-4e8a-9c41-0d2b5e7f8a13"

BLOCKS = ("image", "estimates", "indication", "chief_complaint", "prior_report")
TEXT_BLOCKS = ("indication", "chief_complaint", "prior_report")  # v2.context's columns
ARMS: dict[str, tuple[str, ...]] = {
    "jev_est": ("estimates",),
    "jev_est_indication": ("estimates", "indication"),
    "jev_est_text": ("estimates", *TEXT_BLOCKS),
    "jev_text": TEXT_BLOCKS,
    # v3.1 (§8): the prepared frontal X-ray itself
    "jev_img": ("image",),
    "jev_img_est": ("image", "estimates"),
    "jev_img_est_text": ("image", "estimates", *TEXT_BLOCKS),
}
PRIMARY = "jev_est_text"
V3_0_ARMS = ("jev_est", "jev_est_indication", "jev_est_text", "jev_text")
PRIMARY_V3_1 = "jev_img_est_text"
V3_1_ARMS = ("jev_img", "jev_img_est", "jev_img_est_text")  # run in the format val chose in v3.0 (§8)

PREAMBLE = ("A patient is having a chest X-ray. Below is what was known when it was taken. A radiologist will then "
            "read the X-ray and write a report.")
TITLES = {
    "image": "The frontal chest X-ray",
    "estimates": ("An image model's reading of this X-ray: for each finding, its estimated probability that the "
                  "radiologist's report will say the finding is present"),
    "indication": "Why the X-ray was ordered (the ordering clinician's indication)",
    "chief_complaint": "Emergency department chief complaint at triage",
    "prior_report": "Findings and impression from the report of the patient's previous chest X-ray, at least a day "
                    "earlier",
}
# How each finding reads in a question ("... say that there is {name}?").
NAMES = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "pulmonary edema",
    "Enlarged Cardiomediastinum": "an enlarged cardiomediastinum",
    "Fracture": "a fracture",
    "Lung Lesion": "a lung lesion",
    "Lung Opacity": "a lung opacity",
    "Pleural Effusion": "a pleural effusion",
    "Pleural Other": "a pleural abnormality other than an effusion",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "a pneumothorax",
}
DEFINITIONS = {
    "Atelectasis": "The report describes atelectasis: collapse or incomplete expansion of lung tissue, including "
                   "plate-like, subsegmental or bibasilar atelectasis.",
    "Cardiomegaly": "The report describes an enlarged heart (cardiomegaly).",
    "Consolidation": "The report describes consolidation: airspace filled with fluid, pus or cells.",
    "Edema": "The report describes pulmonary edema or vascular congestion, interstitial or alveolar.",
    "Enlarged Cardiomediastinum": "The report describes a widened mediastinum or an enlarged cardiomediastinal "
                                  "silhouette.",
    "Fracture": "The report describes a bone fracture (rib, clavicle, spine or shoulder), acute or old.",
    "Lung Lesion": "The report describes a focal lung lesion: a nodule, mass or cavity.",
    "Lung Opacity": "The report describes any area of increased lung density (opacity, infiltrate, haziness), "
                    "whatever its cause.",
    "Pleural Effusion": "The report describes fluid in the pleural space.",
    "Pleural Other": "The report describes a pleural abnormality other than an effusion: thickening, plaques, "
                     "scarring or calcification.",
    "Pneumonia": "The report describes findings of pneumonia, or says they are consistent with it.",
    "Pneumothorax": "The report describes air in the pleural space (a pneumothorax), however small.",
}
ABSENT = {"noul": "The report will say it is absent, or will not mention it.",
          "score": "It counts as no if the report says it is absent, or does not mention it."}


def question_name(label: str) -> str:
    return label.lower().replace(" ", "_")


def run_name(arm: str, fmt: str) -> str:
    return f"{arm}_{fmt}"


def questions(fmt: str) -> dict:
    """The 12 questions in format `fmt`, keyed by question_name, in LABELS order."""
    if fmt == "noul":
        return {
            question_name(label): {
                "type": "noul",
                "instructions": f"Will the radiologist's report of this chest X-ray say that there is {NAMES[label]}?",
                "criteria": {"true": DEFINITIONS[label], "false": ABSENT["noul"]},
            }
            for label in LABELS
        }
    if fmt == "score":
        return {
            question_name(label): {
                "type": "score",
                "instructions": (f"How likely is it that the radiologist's report of this chest X-ray will say that "
                                 f"there is {NAMES[label]}? It counts as yes if: {DEFINITIONS[label]} "
                                 f"{ABSENT['score']}"),
                "criteria": list(LEVELS),
            }
            for label in LABELS
        }
    raise ValueError(f"unknown question format {fmt!r} (expected one of {', '.join(FORMATS)})")


def likelihood(question: dict, value: float) -> float:
    """
    The score v2's evaluation ranks: a yes/no answer's probability as is, a score's expected band (0 to n - 1) as
    the likelihood at that band's midpoint, (band + 0.5) / n. Monotone, so ranks and AUROC are those of the answer.
    """
    if question["type"] == "noul":
        return float(value)
    return (float(value) + 0.5) / len(question["criteria"])


def needs_estimates(arm: str) -> bool:
    return "estimates" in ARMS[arm]


def needs_image(arm: str) -> bool:
    return "image" in ARMS[arm]


def fmt_probability(p: float) -> str:
    """A percentage with one decimal: two decimals of a probability would tie most rare findings at 0.00-0.02."""
    pct = 100.0 * float(p)
    if pct < 0.05:
        return "<0.1%"
    if pct >= 99.95:
        return ">99.9%"
    return f"{pct:.1f}%"


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING
    text = " ".join(str(value).split())
    return text or MISSING


def render_state(arm: str, ctx: Mapping, estimates: Optional[np.ndarray]) -> str:
    """The `state` Jev reads for one study: `ctx` holds v2.context's text columns, `estimates` 12 probabilities."""
    parts = [PREAMBLE]
    for block in ARMS[arm]:
        if block == "image":
            parts.append(f"{TITLES['image']}: {IMAGE_PLACEHOLDER}")
        elif block == "estimates":
            if estimates is None or len(estimates) != len(LABELS):
                raise ValueError(f"arm {arm!r} shows the image model's estimates: pass 12 probabilities")
            lines = "\n".join(f"- {label}: {fmt_probability(p)}" for label, p in zip(LABELS, estimates))
            parts.append(f"{TITLES['estimates']}:\n{lines}")
        else:
            parts.append(f"{TITLES[block]}: {_text(ctx.get(block))}")
    return "\n\n".join(parts)


def prompt_hash(arm: str, fmt: str) -> str:
    """Twelve hex digits over everything that shapes a request except the study itself."""
    sample = {"indication": "INDICATION", "chief_complaint": "COMPLAINT", "prior_report": "PRIOR"}
    estimates = np.linspace(0.0, 1.0, len(LABELS)) if needs_estimates(arm) else None
    blob = json.dumps({"version": PROMPT_VERSION[fmt], "blocks": ARMS[arm], "questions": questions(fmt),
                       "state": render_state(arm, sample, estimates)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
