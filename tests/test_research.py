import json
from unittest.mock import patch

import pytest

from crickvision import research


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_discover_normalizes_public_metadata():
    def fake_urlopen(request, timeout):
        url = request.full_url
        if "api/datasets" in url:
            return FakeResponse([{"id": "org/cricket-data", "tags": ["license:apache-2.0"], "downloads": 12}])
        if "api/models" in url:
            return FakeResponse([{"id": "org/cricket-model", "tags": ["license:mit"], "pipeline_tag": "object-detection"}])
        return FakeResponse({"items": [{"full_name": "org/cricket-cv", "html_url": "https://github.com/org/cricket-cv", "license": {"spdx_id": "MIT"}}]})

    with patch("crickvision.research.urlopen", side_effect=fake_urlopen):
        report = research.discover_resources(limit=3)
    assert report["healthy_sources"] == 3
    assert report["sources"]["datasets"]["items"][0]["license_tag"] == "apache-2.0"
    assert report["sources"]["models"]["items"][0]["kind"] == "model"
    assert report["sources"]["repositories"]["items"][0]["license_tag"] == "MIT"
    assert "no downloads" in report["mode"]


def test_source_failure_does_not_hide_other_sources():
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("offline")
        if "api/models" in request.full_url:
            return FakeResponse([])
        return FakeResponse({"items": []})

    with patch("crickvision.research.urlopen", side_effect=fake_urlopen):
        report = research.discover_resources()
    assert report["sources"]["datasets"]["status"] == "unavailable"
    assert report["healthy_sources"] == 2
    assert report["status"] == "ok"


@pytest.mark.parametrize("limit", [0, 31, True, 1.5])
def test_rejects_invalid_limit(limit):
    with pytest.raises(ValueError):
        research.discover_resources(limit=limit)
