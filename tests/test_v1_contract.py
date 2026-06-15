"""Recurrence guard for the /v1 path-drift incident (2026-06): the SDK was calling
/memory/... with no /v1 prefix and 404ing against prod. This asserts the request
layer prepends /v1 (idempotently), so the drift can't silently return.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import httpx
    from velixar import Velixar
except Exception as exc:  # deps not installed in this env
    pytest.skip(f"deps unavailable: {exc}", allow_module_level=True)


class _Resp:
    status_code = 200

    def json(self):
        return {"id": "m_test", "memories": [], "count": 0, "updated": True, "deleted": True}


def _capture(monkeypatch):
    seen = {}

    def fake_request(self, method, path, **kw):
        seen["method"] = method
        seen["path"] = path
        return _Resp()

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    return seen


def test_search_targets_v1(monkeypatch):
    seen = _capture(monkeypatch)
    Velixar(api_key="vlx_test").search("x", limit=1)
    assert seen["method"] == "GET"
    assert seen["path"].startswith("/v1/memory/search"), seen["path"]


def test_store_targets_v1(monkeypatch):
    seen = _capture(monkeypatch)
    Velixar(api_key="vlx_test").store("hello")
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/memory", seen["path"]


def test_prefix_is_idempotent(monkeypatch):
    """A path already starting with /v1/ must not become /v1/v1/..."""
    seen = _capture(monkeypatch)
    Velixar(api_key="vlx_test")._request("GET", "/v1/memory/list")
    assert seen["path"] == "/v1/memory/list", seen["path"]
