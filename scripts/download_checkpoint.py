from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqa_project.downloads import download_file

DEFAULT_URL = "https://github.com/dongtingshuo/multimodal-vqa/releases/download/v0.3.0/best.pt"
DEFAULT_SHA256 = "0cce251f02a7b5349b90c0a6e41850168cc01a700f36fe03663887bae7dbf213"
DEFAULT_SIZE = 1409005580


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify the released VQA checkpoint.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="checkpoints/best.pt")
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    parser.add_argument("--expected-size", type=int, default=DEFAULT_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = download_file(
        args.url,
        args.output,
        expected_checksum=f"sha256:{args.sha256}" if args.sha256 else None,
        expected_size=args.expected_size,
    )
    print(f"checkpoint ready: {path.resolve()}")


if __name__ == "__main__":
    main()
