#!/usr/bin/env python3
"""Download Meta CoTracker3's scaled offline checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

EXPECTED_SHA256 = "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/opt/models/cotracker3/scaled_offline.pth")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    cached = Path(
        hf_hub_download(repo_id="facebook/cotracker3", filename="scaled_offline.pth")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if cached.resolve() != output:
        shutil.copy2(cached, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"Checksum mismatch: expected {EXPECTED_SHA256}, got {digest}")
    print(f"CoTracker3 checkpoint ready: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
