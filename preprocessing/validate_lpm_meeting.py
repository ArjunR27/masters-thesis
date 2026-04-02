from __future__ import annotations

import argparse
from pathlib import Path

from lpm_preprocess_lib import print_validation_result, validate_meeting_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a generated LPM meeting directory for TreeSeg compatibility."
    )
    parser.add_argument(
        "--meeting-dir",
        required=True,
        help="Path to one meeting directory under <data_root>/<speaker>/<course>/<meeting>.",
    )
    parser.add_argument(
        "--min-slides",
        type=int,
        default=1,
        help="Minimum number of slide/OCR files expected (default: 1).",
    )
    parser.add_argument(
        "--no-require-ocr",
        action="store_true",
        help="Skip OCR file presence checks.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    meeting_dir = Path(args.meeting_dir).expanduser().resolve()
    result = validate_meeting_dir(
        meeting_dir=meeting_dir,
        require_ocr=not args.no_require_ocr,
        min_slides=args.min_slides,
    )
    print_validation_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
