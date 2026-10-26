"""v3 end to end on the synthetic benchmark with a fake Jev: setup, the arms, the cache, the stop rules, the report."""

import json

import pytest

from v2.common import read_predictions
from v3 import jev, prompt, report, smoke
from v3 import run as v3run
from v3 import setup as v3setup


def client_for(transport, model=smoke.FAKE_MODEL, **kw):
    return jev.JevClient(model, api_key="test", transport=transport, sleep=lambda s: None, **kw)


def run_all(out, fake):
    c = client_for(fake)
    for arm in prompt.V3_0_ARMS:
        for split in ("val", "test"):
            v3run.run_arm(out, arm, split, c, concurrency=2)


def test_setup_links_the_benchmark_and_builds_both_val_combiners(v3out):
    s = json.loads((v3out / "v3_setup.json").read_text())
    assert (v3out / "manifest.parquet").is_symlink() and (v3out / "context.parquet").is_symlink()
    assert s["stack_indication"] == "test-pixels_48_probe_stack_indication"
    assert s["stack_text"] == "test-pixels_48_probe_stack_text"
    for run in (s["stack_indication"], s["stack_text"]):
        path = v3out / "predictions" / f"{run}.parquet"
        assert path.exists() and not path.is_symlink()


def test_setup_never_writes_into_the_benchmark(v3out):
    bench = json.loads((v3out / "v3_setup.json").read_text())["bench"]
    with pytest.raises(SystemExit, match="must differ"):
        v3setup.setup(bench, bench)


def test_every_arm_scores_every_study_from_its_own_inputs(v3out):
    fake = smoke.FakeJev()
    run_all(v3out, fake)
    assert smoke.check(v3out, fake) == []


def test_test_runs_only_after_val_with_the_same_prompt_and_model(v3out):
    with pytest.raises(SystemExit, match="run val first"):
        v3run.run_arm(v3out, "jev_est", "test", client_for(smoke.FakeJev()))
    v3run.run_arm(v3out, "jev_est", "val", client_for(smoke.FakeJev()))
    with pytest.raises(SystemExit, match="run val first"):
        v3run.run_arm(v3out, "jev_est", "test", client_for(smoke.FakeJev("jev-other"), model="jev-other"))


def test_the_pilot_writes_a_report_and_no_predictions(v3out):
    stats = v3run.run_arm(v3out, prompt.PRIMARY, "val", client_for(smoke.FakeJev()), limit=10)
    assert stats["n_answered"] == 10 and stats["n_failed"] == 0
    assert (v3out / "reports" / f"v3_pilot_{prompt.PRIMARY}.md").exists()
    assert not (v3out / "predictions" / f"{prompt.PRIMARY}.parquet").exists()


class Flaky(smoke.FakeJev):
    """Fails every `every`-th request with a server error until healed."""

    def __init__(self, every):
        super().__init__()
        self.every, self.healed = every, False

    def __call__(self, *args, **kwargs):
        if not self.healed and (self.calls + 1) % self.every == 0:
            self.calls += 1
            return jev.Response(500, {}, {})
        return super().__call__(*args, **kwargs)


def test_more_than_one_percent_failed_stops_and_a_rerun_asks_only_those(v3out):
    flaky = Flaky(every=5)
    with pytest.raises(SystemExit, match="no answer"):
        v3run.run_arm(v3out, "jev_est", "val", client_for(flaky, max_retries=0))
    failed_log = v3out / "logs" / "jev_est_val_failures.csv"
    assert failed_log.exists() and not (v3out / "predictions" / "jev_est.parquet").exists()
    n_failed = sum(1 for _ in failed_log.open()) - 1
    flaky.healed, before = True, flaky.calls
    stats = v3run.run_arm(v3out, "jev_est", "val", client_for(flaky))
    assert flaky.calls - before == n_failed and stats["n_failed"] == 0 and not failed_log.exists()


def test_answers_from_two_models_stop_the_run(v3out):
    c = client_for(smoke.FakeJev())
    v3run.run_arm(v3out, "jev_est", "val", c, limit=5)
    path = v3run.cache_path(v3out, "jev_est", smoke.FAKE_MODEL)
    rec = json.loads(path.read_text().splitlines()[0])
    with path.open("a") as f:
        f.write(json.dumps({**rec, "study_id": rec["study_id"] + 10**9, "model": "jev-other"}) + "\n")
    with pytest.raises(SystemExit, match="2 models"):
        v3run.run_arm(v3out, "jev_est", "val", c, limit=5)


def test_the_image_arm_needs_allow_image_and_sends_the_prepared_jpeg(v3out):
    with pytest.raises(SystemExit, match="allow-image"):
        v3run.run_arm(v3out, "jev_img_est_text", "val", client_for(smoke.FakeJev()))
    seen = []

    class Recording(smoke.FakeJev):
        def __call__(self, method, url, headers, json_body=None, files=None, timeout=None):
            seen.append((headers, files))
            return super().__call__(method, url, headers, json_body, files, timeout)

    v3run.run_arm(v3out, "jev_img_est_text", "val", client_for(Recording()), limit=3, allow_image=True)
    assert len(seen) == 3
    headers, files = seen[0]
    assert headers["TypeSafe-Danger-Zone-Images"] == "true"
    body = json.loads(files["request"][1])
    assert prompt.IMAGE_PLACEHOLDER in body["state"] and files[jev.IMAGE_PART][1][:2] == b"\xff\xd8"


def test_aliases_are_refused_before_any_request(v3out):
    with pytest.raises(SystemExit, match="alias"):
        v3run.main(["--arm", "jev_est", "--split", "val", "--model", "jev-latest", "--out", str(v3out)])


def test_the_report_before_and_after_the_arms(v3out):
    early = report.build(v3out, n_boot=20)
    assert early["j1"] is None and early["verdict"].startswith("not run")
    run_all(v3out, smoke.FakeJev())
    rep = report.main(["--out", str(v3out), "--n-boot", "20"])
    text = rep["report"].read_text()
    assert "Gate J1" in text and "The context ladder" in text and "not run" not in text.split("## Runs")[0]
    jev_rows = [row for row in rep["table"] if row["kind"] == "jev"]
    assert {row["run"] for row in jev_rows} == set(prompt.V3_0_ARMS)
    _, meta = read_predictions(v3out / "predictions" / "jev_est_text.parquet")
    assert meta["kind"] == "jev" and meta["blocks"] == list(prompt.ARMS["jev_est_text"])
