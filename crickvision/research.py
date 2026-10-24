"""Discover public cricket-CV datasets, models, and reference repositories.

This is a metadata scout, not an installer: it never downloads code, model weights,
or footage. Inspect source terms and provenance before using any result.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

USER_AGENT = "CrickVision-ResourceScout/0.1 (+https://github.com/kaustubhdua/CrickVision-)"
DEFAULT_LIMIT = 10
MAX_LIMIT = 30


def _get_json(url: str, timeout: float = 8.0) -> Any:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _short(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:limit]


def _dataset_item(item: dict[str, Any]) -> dict[str, Any]:
    tags = item.get("tags") or []
    license_tag = next((tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), "")
    return {
        "kind": "dataset",
        "id": item.get("id", ""),
        "url": "https://huggingface.co/datasets/" + str(item.get("id", "")),
        "license_tag": license_tag,
        "downloads": item.get("downloads"),
        "last_modified": item.get("lastModified"),
        "tags": tags[:12],
        "warning": "Metadata only. Verify dataset card, annotations, access conditions, and underlying footage rights.",
    }


def _model_item(item: dict[str, Any]) -> dict[str, Any]:
    tags = item.get("tags") or []
    license_tag = next((tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), "")
    return {
        "kind": "model",
        "id": item.get("id", ""),
        "url": "https://huggingface.co/" + str(item.get("id", "")),
        "license_tag": license_tag,
        "downloads": item.get("downloads"),
        "last_modified": item.get("lastModified"),
        "pipeline_tag": item.get("pipeline_tag"),
        "tags": tags[:12],
        "warning": "Metadata only. Verify model card, weight licence, training data, and task suitability before loading.",
    }


def _repository_item(item: dict[str, Any]) -> dict[str, Any]:
    license_info = item.get("license") or {}
    return {
        "kind": "repository",
        "id": item.get("full_name", ""),
        "url": item.get("html_url", ""),
        "description": _short(item.get("description")),
        "license_tag": license_info.get("spdx_id") or license_info.get("key", ""),
        "stars": item.get("stargazers_count"),
        "updated_at": item.get("pushed_at"),
        "language": item.get("language"),
        "warning": "Metadata only. Review source, maintenance, dependencies, licence, and security before reusing code.",
    }


def discover_resources(limit: int = DEFAULT_LIMIT, timeout: float = 8.0) -> dict[str, Any]:
    """Query public metadata APIs; individual source failures do not hide other results."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer from 1 to {MAX_LIMIT}")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    requests = [
        ("datasets", f"https://huggingface.co/api/datasets?search=cricket&limit={limit}&sort=downloads&direction=-1", _dataset_item),
        ("models", f"https://huggingface.co/api/models?search=cricket&limit={limit}&sort=downloads&direction=-1", _model_item),
        ("repositories", f"https://api.github.com/search/repositories?q=cricket+computer+vision+tracking&sort=stars&order=desc&per_page={limit}", _repository_item),
    ]
    results: dict[str, Any] = {
        "tool": "CrickVision public resource scout",
        "mode": "metadata-only; no downloads or code execution",
        "limit": limit,
        "sources": {},
        "safety_note": (
            "Public visibility is not proof of permission. Verify licences, dataset labels, "
            "model provenance, and underlying footage rights before use."
        ),
    }
    for name, url, normalize in requests:
        try:
            payload = _get_json(url, timeout=timeout)
            items = payload.get("items", []) if name == "repositories" and isinstance(payload, dict) else payload
            if not isinstance(items, list):
                raise ValueError("unexpected API response shape")
            results["sources"][name] = {
                "status": "ok",
                "count": len(items),
                "items": [normalize(item) for item in items if isinstance(item, dict)],
            }
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            results["sources"][name] = {
                "status": "unavailable",
                "count": 0,
                "items": [],
                "error": _short(f"{type(exc).__name__}: {exc}", 180),
            }
    results["healthy_sources"] = sum(source["status"] == "ok" for source in results["sources"].values())
    results["status"] = "ok" if results["healthy_sources"] else "unavailable"
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"results per source (1-{MAX_LIMIT})")
    parser.add_argument("--timeout", type=float, default=8.0, help="per-source network timeout in seconds")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    try:
        report = discover_resources(limit=args.limit, timeout=args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"CrickVision online resource scout: {report['status']}")
        print(report["safety_note"])
        for name, source in report["sources"].items():
            print(f"\n{name}: {source['status']} ({source['count']} results)")
            if source.get("error"):
                print(f"  {source['error']}")
            for item in source["items"]:
                licence = item.get("license_tag") or "licence not declared in API metadata"
                print(f"  - {item.get('id') or '(unnamed)'} [{licence}]")
                print(f"    {item.get('url', '')}")
                if item.get("description"):
                    print(f"    {item['description']}")
        print("\nNo assets were downloaded or executed.")
    return 0 if report["healthy_sources"] else 2


if __name__ == "__main__":
    sys.exit(main())
