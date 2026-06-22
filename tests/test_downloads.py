from __future__ import annotations

import hashlib
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from scripts.prepare_vqa_data import extract_zip
from vqa_project.downloads import DownloadIntegrityError, download_file, file_checksum, verify_file


def test_checksum_and_size_validation(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"verified payload")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_file(path, f"sha256:{expected}", path.stat().st_size)
    assert file_checksum(path) == expected
    with pytest.raises(DownloadIntegrityError, match="Checksum mismatch"):
        verify_file(path, "sha256:" + "0" * 64)


def test_download_resumes_partial_file(tmp_path: Path, monkeypatch) -> None:
    payload = b"0123456789" * 1024

    class FakeResponse:
        status = 206

        def __init__(self, offset: int) -> None:
            self.headers = {
                "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
                "Content-Length": str(len(payload) - offset),
            }
            self.body = BytesIO(payload[offset:])

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return self.body.read(size)

    def fake_urlopen(request, timeout):
        assert timeout == 30
        range_header = request.headers["Range"]
        offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
        return FakeResponse(offset)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    output = tmp_path / "download.bin"
    partial = output.with_suffix(".bin.part")
    partial.write_bytes(payload[:128])
    expected = hashlib.sha256(payload).hexdigest()
    result = download_file(
        "https://example.invalid/artifact",
        output,
        expected_checksum=f"sha256:{expected}",
        expected_size=len(payload),
    )
    assert result.read_bytes() == payload
    assert not partial.exists()


def test_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")
    with pytest.raises(zipfile.BadZipFile, match="Unsafe archive member"):
        extract_zip(archive_path, tmp_path / "output")
    assert not (tmp_path / "outside.txt").exists()
