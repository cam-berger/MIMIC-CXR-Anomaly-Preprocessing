"""What Jev reads: the state text per arm and the 12 questions in each format (docs/V3_PLAN.md §3-4)."""

import numpy as np
import pytest

from v2.common import LABELS
from v3 import prompt

PILOT_HASH = "e99b96a72ad9"  # jev_est_text's yes/no prompt in the 2026-10-26 pilot, the pre-registered one


def ctx(**overrides):
    row = {"indication": "zebra indication", "chief_complaint": "okapi complaint", "prior_report": "quagga report"}
    row.update(overrides)
    return row


def test_arms_use_known_blocks_and_v3_0_leaves_out_the_image():
    for arm, blocks in prompt.ARMS.items():
        assert arm.startswith("jev_") and blocks and set(blocks) <= set(prompt.BLOCKS)
    assert prompt.PRIMARY in prompt.V3_0_ARMS
    assert not any(prompt.needs_image(arm) for arm in prompt.V3_0_ARMS)
    assert all(prompt.needs_image(arm) for arm in prompt.V3_1_ARMS) and prompt.PRIMARY_V3_1 in prompt.V3_1_ARMS
    assert prompt.ARMS["jev_img"] == ("image",) and not prompt.needs_estimates("jev_img")
    assert prompt.run_name("jev_est_text", "score") == "jev_est_text_score"


@pytest.mark.parametrize("fmt", prompt.FORMATS)
def test_questions_are_12_in_label_order(fmt):
    qs = prompt.questions(fmt)
    assert list(qs) == [prompt.question_name(label) for label in LABELS] and len(set(qs)) == 12
    for q in qs.values():
        assert q["type"] == fmt and "radiologist's report" in q["instructions"]
        if fmt == "noul":
            assert set(q["criteria"]) == {"true", "false"} and q["instructions"].endswith("?")
        else:
            assert q["criteria"] == list(prompt.LEVELS) and len(q["criteria"]) == 10
            assert all(len(level) <= 7 for level in q["criteria"])  # short: they repeat 12 times per request


def test_unknown_format_is_an_error():
    with pytest.raises(ValueError):
        prompt.questions("choice")


def test_likelihood_keeps_probabilities_and_maps_bands_to_midpoints():
    noul, score = prompt.questions("noul")["edema"], prompt.questions("score")["edema"]
    assert prompt.likelihood(noul, 0.37) == pytest.approx(0.37)
    assert prompt.likelihood(score, 0.0) == pytest.approx(0.05)
    assert prompt.likelihood(score, 4.5) == pytest.approx(0.5)
    assert prompt.likelihood(score, 9.0) == pytest.approx(0.95)
    assert prompt.likelihood(score, 0.01) > prompt.likelihood(score, 0.0)


def test_estimates_are_percentages_in_label_order():
    p = np.array([0.0004, 0.0123, 0.5, 0.99996] + [0.25] * 8)
    text = prompt.render_state("jev_est", ctx(), p)
    lines = [line for line in text.splitlines() if line.startswith("- ")]
    assert [line[2:].split(":")[0] for line in lines] == list(LABELS)
    assert lines[0].endswith("<0.1%") and lines[1].endswith("1.2%") and lines[2].endswith("50.0%")
    assert lines[3].endswith(">99.9%")


def test_each_arm_shows_exactly_its_own_sources():
    est = np.full(12, 0.5)
    assert not any(word in prompt.render_state("jev_est", ctx(), est) for word in ("zebra", "okapi", "quagga"))
    ind = prompt.render_state("jev_est_indication", ctx(), est)
    assert "zebra" in ind and "okapi" not in ind and "quagga" not in ind
    full = prompt.render_state("jev_est_text", ctx(), est)
    assert all(word in full for word in ("zebra", "okapi", "quagga")) and "50.0%" in full
    text_only = prompt.render_state("jev_text", ctx(), None)
    assert all(word in text_only for word in ("zebra", "okapi", "quagga")) and "%" not in text_only


def test_missing_sources_are_marked_and_whitespace_collapsed():
    text = prompt.render_state("jev_text", ctx(chief_complaint=None, prior_report="  \n ", indication=float("nan")), None)
    assert text.count(prompt.MISSING) == 3
    assert "a b c" in prompt.render_state("jev_text", ctx(indication="a\n\n b\tc"), None)


def test_estimates_are_required_where_the_arm_shows_them():
    with pytest.raises(ValueError):
        prompt.render_state("jev_est", ctx(), None)
    with pytest.raises(ValueError):
        prompt.render_state("jev_est", ctx(), np.full(11, 0.5))


def test_the_image_arm_puts_the_placeholder_in_state():
    assert prompt.IMAGE_PLACEHOLDER in prompt.render_state("jev_img_est_text", ctx(), np.full(12, 0.5))
    assert prompt.IMAGE_PLACEHOLDER not in prompt.render_state("jev_est_text", ctx(), np.full(12, 0.5))


def test_the_yes_no_format_is_still_the_pre_registered_prompt():
    assert prompt.prompt_hash("jev_est_text", "noul") == PILOT_HASH


def test_prompt_hash_is_stable_per_arm_and_format_and_follows_the_wording(monkeypatch):
    hashes = {(arm, f): prompt.prompt_hash(arm, f) for arm in prompt.ARMS for f in prompt.FORMATS}
    assert len(set(hashes.values())) == len(hashes)
    assert prompt.prompt_hash("jev_est", "score") == hashes[("jev_est", "score")]
    monkeypatch.setitem(prompt.DEFINITIONS, "Edema", "something else")
    assert all(prompt.prompt_hash("jev_est", f) != hashes[("jev_est", f)] for f in prompt.FORMATS)
