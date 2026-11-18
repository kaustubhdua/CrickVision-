from __future__ import annotations

import hashlib
from pathlib import Path

from crickvision.provenance import sha256_file


def test_sha256_file_streams_expected_digest(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    payload = b"crickvision provenance\x00" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()
