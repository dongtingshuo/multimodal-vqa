from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqa_project.downloads import download_file

DEFAULT_URL = "https://github.com/dongtingshuo/multimodal-vqa/releases/download/v0.2.0/best.pt"
DEFAULT_SHA256 = "15e15b4a0194b073a153331ad2c6b38ee39400d87e489bb6f0fc77d91e7cb22c"
DEFAULT_SIZE = 660441108


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
