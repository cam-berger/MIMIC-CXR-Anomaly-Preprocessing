"""The Jev client: request shape, answers, retries and errors that never echo the study's text."""

import json

import pytest

from v3 import jev, prompt

SECRET_TEXT = "zebra indication from a study"
NOUL, SCORE = prompt.questions("noul"), prompt.questions("score")


def ok(questions=NOUL, value=0.3, model="jev-1.14.0"):
    answers = {name: {"type": q["type"], q["type"]: value} for name, q in questions.items()}
    return jev.Response(200, {"model": model, "answers": answers, "usage": {"input_tokens": 50, "output_tokens": 12}},
                        {})


class Scripted:
    """A transport that returns the given responses in order (an exception is raised instead of returned)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, json_body=None, files=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "json": json_body, "files": files})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(transport, **kw):
    sleeps = []
    c = jev.JevClient("jev-1.14.0", api_key="test-key", transport=transport, sleep=sleeps.append, **kw)
    return c, sleeps


def test_request_is_a_systemone_post_with_bearer_auth_and_the_questions():
    t = Scripted(ok())
    c, _ = client(t)
    answer = c.ask("some state", NOUL)
    call = t.calls[0]
    assert call["method"] == "POST" and call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["json"] == {"model": "jev-1.14.0", "state": "some state", "questions": NOUL}
    assert call["files"] is None and "TypeSafe-Danger-Zone-Images" not in call["headers"]
    assert answer.model == "jev-1.14.0" and answer.usage["input_tokens"] == 50
    assert set(answer.scores) == set(NOUL) and answer.scores["edema"] == pytest.approx(0.3)


def test_score_answers_come_back_as_the_expected_level():
    c, _ = client(Scripted(ok(SCORE, 4.37)))
    assert c.ask("s", SCORE).scores["edema"] == pytest.approx(4.37)


def test_rate_limits_and_server_errors_are_retried_with_backoff():
    t = Scripted(jev.Response(429, {}, {"Retry-After": "7"}), jev.Response(503, {}, {}), ConnectionError("reset"),
                 ok())
    c, sleeps = client(t)
    assert c.ask("s", NOUL).scores["edema"] == pytest.approx(0.3)
    assert len(t.calls) == 4 and sleeps[0] == 7.0 and len(sleeps) == 3


def test_retries_give_up_after_max_retries():
    t = Scripted(*[jev.Response(500, {}, {})] * 3)
    c, _ = client(t, max_retries=2)
    with pytest.raises(jev.JevError) as err:
        c.ask("s", NOUL)
    assert err.value.status == 500 and len(t.calls) == 3


def test_validation_errors_are_not_retried_and_never_echo_the_input():
    detail = [{"loc": ["body", "state"], "msg": "too long", "type": "value_error", "input": SECRET_TEXT,
               "ctx": {"text": SECRET_TEXT}}]
    t = Scripted(jev.Response(422, {"detail": detail}, {}))
    c, _ = client(t)
    with pytest.raises(jev.JevError) as err:
        c.ask(SECRET_TEXT, NOUL)
    assert err.value.status == 422 and len(t.calls) == 1
    assert "too long" in str(err.value) and SECRET_TEXT not in str(err.value)


@pytest.mark.parametrize("questions,answers", [
    (NOUL, {}),                                                            # no answers
    (NOUL, {name: {"type": "choice", "noul": 0.5} for name in NOUL}),        # wrong type
    (NOUL, {name: {"type": "noul", "noul": 1.5} for name in NOUL}),          # not a probability
    (SCORE, {name: {"type": "noul", "noul": 0.5} for name in SCORE}),        # a yes/no answer to a score question
    (SCORE, {name: {"type": "score", "score": 9.5} for name in SCORE}),      # beyond the last of 10 levels
])
def test_malformed_answers_are_errors(questions, answers):
    c, _ = client(Scripted(jev.Response(200, {"model": "jev-1.14.0", "answers": answers}, {})))
    with pytest.raises(jev.JevError):
        c.ask("s", questions)


def test_an_image_goes_as_a_multipart_asset_behind_the_images_header():
    t = Scripted(ok())
    c, _ = client(t)
    c.ask(f"look: {prompt.IMAGE_PLACEHOLDER}", NOUL, image=b"\xff\xd8jpeg")
    call = t.calls[0]
    assert call["headers"]["TypeSafe-Danger-Zone-Images"] == "true" and call["json"] is None
    request = json.loads(call["files"]["request"][1])
    assert request["assets"] == {prompt.IMAGE_PLACEHOLDER: {"type": "image", "part": jev.IMAGE_PART}}
    assert call["files"][jev.IMAGE_PART] == ("frontal.jpg", b"\xff\xd8jpeg", "image/jpeg")
    assert call["files"]["request"][2] == "application/json"


def test_long_context_header_is_opt_in():
    t = Scripted(ok())
    c, _ = client(t, long_context=True)
    c.ask("s", NOUL)
    assert t.calls[0]["headers"]["TypeSafe-Danger-Zone-Long-Context"] == "true"


def test_a_missing_key_stops_before_any_request(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(jev, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
        jev.JevClient("jev-1.14.0")


def test_the_key_never_appears_in_errors():
    c, _ = client(Scripted(jev.Response(401, {"detail": "bad key test-key"}, {})))
    with pytest.raises(jev.JevError) as err:
        c.ask("s", NOUL)
    assert "test-key" not in str(err.value) and err.value.status == 401
