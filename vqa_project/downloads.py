from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

from tqdm import tqdm


class DownloadIntegrityError(RuntimeError):
    pass


def file_checksum(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: str | Path,
    expected_checksum: str | None = None,
    expected_size: int | None = None,
) -> None:
    file_path = Path(path)
    if not file_path.is_file():
        raise DownloadIntegrityError(f"Missing downloaded file: {file_path}")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise DownloadIntegrityError(
            f"Unexpected size for {file_path}: {file_path.stat().st_size} != {expected_size} bytes"
        )
    if expected_checksum:
        algorithm, expected = expected_checksum.split(":", maxsplit=1)
        actual = file_checksum(file_path, algorithm)
        if actual.lower() != expected.lower():
            raise DownloadIntegrityError(
                f"Checksum mismatch for {file_path}: {algorithm}:{actual} != {expected_checksum}"
            )


def _response_total(response, offset: int) -> int:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        return int(content_range.rsplit("/", maxsplit=1)[1])
    content_length = int(response.headers.get("Content-Length", "0"))
    return content_length + offset if getattr(response, "status", None) == 206 else content_length


def download_file(
    url: str,
    output_path: str | Path,
    *,
    expected_checksum: str | None = None,
    expected_size: int | None = None,
    retries: int = 3,
    timeout: int = 30,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        try:
            verify_file(output, expected_checksum, expected_size)
        except DownloadIntegrityError as exc:
            print(f"invalid existing file, downloading again: {exc}")
            output.unlink()
        else:
            print(f"verified: {output}")
            return output

    partial = output.with_suffix(output.suffix + ".part")
    if partial.exists() and expected_size is not None:
        if partial.stat().st_size == expected_size:
            verify_file(partial, expected_checksum, expected_size)
            partial.replace(output)
            return output
        if partial.stat().st_size > expected_size:
            partial.unlink()

    for attempt in range(1, retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url)
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                resumed = offset > 0 and getattr(response, "status", None) == 206
                if not resumed:
                    offset = 0
                total = _response_total(response, offset)
                mode = "ab" if resumed else "wb"
                with (
                    partial.open(mode) as file,
                    tqdm(
                        total=total or None,
                        initial=offset,
                        unit="B",
                        unit_scale=True,
                        desc=output.name,
                    ) as progress,
                ):
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        file.write(chunk)
                        progress.update(len(chunk))

            if expected_size is not None and partial.stat().st_size != expected_size:
                raise DownloadIntegrityError(
                    f"Incomplete download for {output}: {partial.stat().st_size} != {expected_size} bytes"
                )
            partial.replace(output)
            verify_file(output, expected_checksum, expected_size)
            return output
        except DownloadIntegrityError:
            output.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            raise
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}. Partial data was kept at {partial} for resume.") from exc
            wait_seconds = 2 * attempt
            print(f"download failed ({attempt}/{retries}): {exc}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)
    return output
