# Small Gemini API client over plain HTTPS: upload a file, then ask a model about text, files and images.
"""Gemini REST client used by the watching and pointing code.

Only two calls are needed: `upload_file` (Files API, resumable upload) for clips, and
`generate` (generateContent) for everything else. The API key travels only in the
`x-goog-api-key` header and is never put in a URL, a log line or an error message.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from earshot import config

RETRY_STATUS = (429, 503)
MAX_TRIES = 3


class GeminiError(RuntimeError):
    """The API answered with an error, or with something that could not be used."""


def _headers(extra: dict | None = None) -> dict:
    """Auth header plus any call-specific headers."""
    return {"x-goog-api-key": config.key("GEMINI_API_KEY"), **(extra or {})}


def _fail(what: str, response: requests.Response) -> GeminiError:
    """Build an error from a response body (the body never contains the key)."""
    return GeminiError(f"{what}: HTTP {response.status_code}: {response.text[:400]}")


def upload_file(path: str | Path, mime: str, timeout: float = 120.0) -> dict:
    """Upload a local file with the Files API and wait until it is ready to use.

    Returns the file resource; its `uri` and `mimeType` go into a
    `{"file_data": {"mime_type", "file_uri"}}` part for `generate`.
    """
    path = Path(path)
    data = path.read_bytes()
    start = requests.post(
        f"{config.GEMINI_BASE}/upload/v1beta/files",
        headers=_headers({
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(data)),
            "X-Goog-Upload-Header-Content-Type": mime,
        }),
        json={"file": {"display_name": path.name}},
        timeout=30,
    )
    if start.status_code != 200 or "X-Goog-Upload-URL" not in start.headers:
        raise _fail("upload start failed", start)

    sent = requests.post(
        start.headers["X-Goog-Upload-URL"],
        headers=_headers({"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"}),
        data=data,
        timeout=timeout,
    )
    if sent.status_code != 200:
        raise _fail("upload failed", sent)
    info = sent.json()["file"]

    deadline = time.time() + timeout
    while info.get("state") != "ACTIVE":
        if info.get("state") == "FAILED":
            raise GeminiError(f"the API could not process {path.name}")
        if time.time() > deadline:
            raise GeminiError(f"{path.name} was still {info.get('state')} after {timeout:.0f} s")
        time.sleep(1.0)
        poll = requests.get(f"{config.GEMINI_BASE}/v1beta/{info['name']}", headers=_headers(), timeout=30)
        if poll.status_code != 200:
            raise _fail("upload status check failed", poll)
        info = poll.json()
    return info


def _post_generate(model: str, body: dict, timeout: float) -> requests.Response:
    """POST generateContent, retrying a busy or rate-limited answer with a short backoff."""
    url = f"{config.GEMINI_BASE}/v1beta/models/{model}:generateContent"
    for attempt in range(MAX_TRIES):
        response = requests.post(url, headers=_headers(), json=body, timeout=timeout)
        if response.status_code not in RETRY_STATUS or attempt == MAX_TRIES - 1:
            break
        time.sleep(1.5 * (attempt + 1))
    return response


def parse_json(text: str):
    """Parse a model's JSON answer, tolerating code fences and words around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0]
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if not starts or end < min(starts):
        raise GeminiError(f"the model did not answer in JSON: {text[:200]}")
    try:
        return json.loads(cleaned[min(starts):end + 1])
    except json.JSONDecodeError as err:
        raise GeminiError(f"the model's JSON could not be read: {text[:200]}") from err


def generate(model: str, parts: list[dict], json_schema: dict | None = None,
             temperature: float = 0.2, timeout: float = 180.0, thinking: str | None = None):
    """Ask `model` about `parts` and return its answer.

    Each part is `{"text": ...}`, `{"file_data": {"mime_type", "file_uri"}}` or
    `{"inline_data": {"mime_type", "data": <base64>}}`. With `json_schema` (a JSON Schema
    dict) the answer is requested as structured JSON and returned parsed. `thinking` sets
    the model's thinking level ("low" answers several times faster; None keeps the model's
    default). If the API rejects the schema or the thinking level, the question is asked
    again without them and with the schema written into the prompt. Without `json_schema`
    the answer text is returned.
    """
    generation = {"temperature": temperature}
    if json_schema is not None:
        generation["responseMimeType"] = "application/json"
        generation["responseJsonSchema"] = json_schema
    if thinking is not None:
        generation["thinkingConfig"] = {"thinkingLevel": thinking}
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation}

    response = _post_generate(model, body, timeout)
    if response.status_code == 400 and len(generation) > 1:
        plain = list(parts)
        if json_schema is not None:
            plain.append({"text": "Answer with JSON only, matching this JSON Schema:\n" + json.dumps(json_schema)})
        body = {"contents": [{"role": "user", "parts": plain}], "generationConfig": {"temperature": temperature}}
        response = _post_generate(model, body, timeout)
    if response.status_code != 200:
        raise _fail(f"{model} request failed", response)

    candidates = response.json().get("candidates") or []
    if not candidates:
        raise GeminiError(f"{model} returned no answer: {response.text[:300]}")
    answer_parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p["text"] for p in answer_parts if "text" in p and not p.get("thought"))
    if not text:
        raise GeminiError(f"{model} returned an empty answer (finish reason {candidates[0].get('finishReason')})")
    return parse_json(text) if json_schema is not None else text
