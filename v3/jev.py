"""
A small client for TypeSafe's Jev (docs/V3_PLAN.md §3): POST /v1/systemone with a `state` and yes/no questions,
retries on rate limits and server errors, and an optional image asset (v3.1, §8).

    python -m v3.jev models          # the models this key can use (sends no MIMIC data)

The key comes from TYPESAFE_API_KEY (environment or .env); TYPESAFE_BASE_URL overrides the API's address. Errors
carry the HTTP status and the API's validation messages, never the request's text, the key, or echoed input.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from v2.common import load_dotenv, setup_logging

from .prompt import IMAGE_PLACEHOLDER

logger = logging.getLogger("v3.jev")

DEFAULT_BASE_URL = "https://api.typesafe.ai"
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
IMAGE_PART = "frontal"
MAX_BACKOFF = 60.0
# jev-1.14.0 on synthetic JPEGs (V3_PLAN §9): 256 px and smaller, and 2048 px, are "Invalid image dimensions";
# 384 to 1100 px on a side are accepted. The range here is the one tested, not TypeSafe's documented limit.
IMAGE_SIDES = (384, 1100)


class JevError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


@dataclass
class Response:
    status: int
    payload: dict
    headers: dict = field(default_factory=dict)


@dataclass
class Answer:
    model: str
    scores: dict  # question name -> the answer's value (probability of yes, or expected level)
    usage: dict
    latency: float  # seconds, the successful attempt only


Transport = Callable[..., Response]


def requests_transport(method: str, url: str, headers: dict, json_body: Optional[dict] = None,
                       files: Optional[dict] = None, timeout: Optional[float] = None) -> Response:
    import requests

    try:
        r = requests.request(method, url, headers=headers, json=json_body, files=files, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(type(exc).__name__) from None
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(type(exc).__name__) from None
    try:
        payload = r.json()
    except ValueError:
        payload = {}
    return Response(r.status_code, payload if isinstance(payload, dict) else {"data": payload}, dict(r.headers))


def build_request(model: str, state: str, questions: dict, image: Optional[bytes] = None) -> tuple[dict, Optional[dict]]:
    """(JSON body, None), or (the body, multipart parts) when an image goes with it."""
    body = {"model": model, "state": state, "questions": questions}
    if image is None:
        return body, None
    body["assets"] = {IMAGE_PLACEHOLDER: {"type": "image", "part": IMAGE_PART}}
    files = {"request": (None, json.dumps(body), "application/json"),
             IMAGE_PART: ("frontal.jpg", image, "image/jpeg")}
    return body, files


def _echoes(message: str, sent: str, width: int = 24) -> bool:
    """Whether `message` repeats any `width`-character stretch of the text that was sent (or all of a shorter text)."""
    if len(sent) < width:
        return len(sent) >= 8 and sent in message  # a few characters identify nothing
    return any(sent[i:i + width] in message for i in range(0, len(sent) - width + 1, width // 2))


def _error_message(resp: Response, sent: str = "") -> str:
    """
    The status, plus where and why for validation errors (422: `input` and `ctx` can echo the request, so they are
    dropped) and the API's own error type and short message otherwise (e.g. 400 "Invalid image dimensions."),
    unless that message repeats the text that was sent.
    """
    message = f"HTTP {resp.status}"
    detail = resp.payload.get("detail")
    if resp.status == 422 and isinstance(detail, list):
        why = ["{}: {}".format(".".join(str(p) for p in d.get("loc", [])), d.get("msg", "")) for d in detail
               if isinstance(d, dict)]
        message += " (" + "; ".join(why) + ")"
    elif isinstance(detail, dict):
        kind, text = str(detail.get("error_type", "")), str(detail.get("message", ""))
        shown = text if text and len(text) <= 120 and not _echoes(text, sent) else ""
        if kind or shown:
            message += " (" + ": ".join(p for p in (kind, shown) if p) + ")"
    return message


def parse_answer(payload: dict, questions: dict) -> tuple[str, dict]:
    """The answering model, and each question's value: a yes/no probability (`noul`) or an expected level (`score`)."""
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise JevError("malformed response: no answers", 200)
    values = {}
    for name, question in questions.items():
        kind = question["type"]
        if kind not in ("noul", "score"):
            raise ValueError(f"question {name!r}: type {kind!r} has no numeric answer")
        top = 1.0 if kind == "noul" else len(question["criteria"]) - 1
        a = answers.get(name)
        v = a.get(kind) if isinstance(a, dict) and a.get("type") == kind else None
        if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= top:
            raise JevError(f"malformed response: no {kind} value in [0, {top:g}] for {name!r}", 200)
        values[name] = float(v)
    return str(payload.get("model", "")), values


class JevClient:
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 transport: Optional[Transport] = None, timeout: float = 120.0, max_retries: int = 6,
                 backoff: float = 1.0, long_context: bool = False, sleep: Callable[[float], None] = time.sleep):
        load_dotenv()
        self._key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self._key:
            raise SystemExit("TYPESAFE_API_KEY is not set: add it to the environment or .env")
        self.model = model
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.transport = transport or requests_transport
        self.timeout, self.max_retries, self.backoff = timeout, max_retries, backoff
        self.long_context = long_context
        self.sleep = sleep

    def _headers(self, image: bool) -> dict:
        headers = {"Authorization": f"Bearer {self._key}"}
        if image:
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        if self.long_context:
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        return headers

    def _wait(self, attempt: int, resp: Optional[Response]) -> float:
        retry_after = (resp.headers or {}).get("Retry-After") if resp is not None else None
        try:
            if retry_after is not None:
                return min(float(retry_after), MAX_BACKOFF)
        except ValueError:
            pass
        return min(self.backoff * 2 ** attempt * (1 + 0.25 * random.random()), MAX_BACKOFF)

    def _send(self, method: str, path: str, body: Optional[dict] = None, files: Optional[dict] = None,
              image: bool = False) -> tuple[Response, float]:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            resp, failure = None, None
            start = time.monotonic()
            try:
                resp = self.transport(method, url, self._headers(image), json_body=None if files else body,
                                      files=files, timeout=self.timeout)
            except (ConnectionError, TimeoutError) as exc:
                failure = JevError(f"network error ({type(exc).__name__})", None)
            if resp is not None and resp.status == 200:
                return resp, time.monotonic() - start
            if resp is not None:
                sent = str((body or {}).get("state", ""))
                failure = JevError(_error_message(resp, sent).replace(self._key, "<key>"), resp.status)
                if resp.status not in RETRY_STATUS:
                    raise failure
            if attempt == self.max_retries:
                raise failure
            self.sleep(self._wait(attempt, resp))
        raise AssertionError("unreachable")

    def ask(self, state: str, questions: dict, image: Optional[bytes] = None) -> Answer:
        body, files = build_request(self.model, state, questions, image)
        resp, latency = self._send("POST", "/v1/systemone", body, files, image=image is not None)
        model, scores = parse_answer(resp.payload, questions)
        usage = resp.payload.get("usage") if isinstance(resp.payload.get("usage"), dict) else {}
        return Answer(model, scores, usage, latency)

    def models(self) -> dict:
        return self._send("GET", "/v1/models")[0].payload


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["models"])
    args = parser.parse_args(argv)
    setup_logging()
    payload = JevClient(model="").models()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
