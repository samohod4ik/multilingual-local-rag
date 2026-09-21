"""Small loopback client."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


def request_json(
    base: str, method: str, path: str, payload: dict[str, object] | None = None
) -> dict[str, object]:
    if not base.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")):
        raise ValueError("client base must be loopback")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=5) as response:  # noqa: S310
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("response must be an object")
    return body
