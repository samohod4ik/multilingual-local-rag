"""Small loopback client. Redirects are not followed."""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPDefaultErrorHandler, HTTPErrorProcessor, HTTPHandler, OpenerDirector, Request


def request_json(
    base: str, method: str, path: str, payload: dict[str, object] | None = None
) -> dict[str, object]:
    parsed = urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("client base must be loopback")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    opener = OpenerDirector()
    opener.add_handler(HTTPDefaultErrorHandler())
    opener.add_handler(HTTPHandler())
    opener.add_handler(HTTPErrorProcessor())
    try:
        with opener.open(req, timeout=5) as response:
            if 300 <= response.status < 400:
                raise ValueError("redirects are not followed")
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if 300 <= exc.code < 400:
            raise ValueError("redirects are not followed") from exc
        body = json.loads(exc.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("response must be an object") from exc
        body["status"] = exc.code
        return body
    if not isinstance(body, dict):
        raise ValueError("response must be an object")
    return body
