#!/usr/bin/env python3
"""
Download model checkpoints from HuggingFace Hub.

Supports:
  - SAM3 (primary)   — facebook/sam3
  - SAM2 (fallback)  — facebook/sam2.1

Requirements:
  - HuggingFace account with access to the relevant model repo
  - HF_TOKEN environment variable set to your HuggingFace token

Usage:
  conda run -n sam3 python scripts/download_model.py          # download SAM3
  conda run -n sam3 python scripts/download_model.py --sam2   # download SAM2 fallback

  Or with explicit token:
  HF_TOKEN=hf_xxx conda run -n sam3 python scripts/download_model.py
"""

import argparse
import os
import sys
from pathlib import Path


SAM3_FILES = [
    ("facebook/sam3", "sam3.pt"),
    # bpe_simple_vocab_16e6.txt.gz is bundled inside the sam3 Python package
    # (sam3/assets/) and loaded via pkg_resources — no separate download needed.
]

SAM2_FILES = [
    ("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt"),
]


def download_files(dest: Path, file_specs: list, token: str):
    """Download a list of (repo_id, filename) pairs to dest."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    for repo_id, filename in file_specs:
        out_path = dest / filename
        if out_path.exists():
            size_gb = out_path.stat().st_size / 1e9
            print(f"  Skipping {filename} (already exists, {size_gb:.2f} GB)")
            continue
        print(f"Downloading {filename} from {repo_id} ...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest),
                token=token,
            )
            print(f"  Saved to {out_path}")
        except Exception as e:
            print(f"ERROR downloading {filename}: {e}")
            print("\nIf you get a 401/403 error, make sure you've:")
            print(f"  1. Accepted the model license at https://huggingface.co/{repo_id}")
            print("  2. Used a valid HF token with read permissions")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download SAM model checkpoints")
    parser.add_argument(
        "--sam2", action="store_true",
        help="Download SAM2 checkpoint (fallback model)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download both SAM3 and SAM2 checkpoints"
    )
    args = parser.parse_args()

    dest = Path(__file__).parent.parent / "pretrained_models"
    dest.mkdir(parents=True, exist_ok=True)

    # Check which files to download
    if args.all:
        files = SAM3_FILES + SAM2_FILES
        model_label = "SAM3 + SAM2"
    elif args.sam2:
        files = SAM2_FILES
        model_label = "SAM2"
    else:
        files = SAM3_FILES
        model_label = "SAM3"

    # Check if already present
    all_exist = all((dest / f).exists() for _, f in files)
    if all_exist:
        print(f"All {model_label} files already downloaded at {dest}/")
        for _, f in files:
            p = dest / f
            if p.exists():
                print(f"  - {p.stat().st_size / 1e9:.2f} GB  {f}")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Steps:")
        print("  1. Request access at the relevant HuggingFace model page")
        print("  2. Create a token at: https://huggingface.co/settings/tokens")
        print("  3. Run: export HF_TOKEN=hf_your_token_here")
        print("  4. Re-run this script")
        sys.exit(1)

    ok = download_files(dest, files, token)
    if ok:
        print(f"\nDone! {model_label} model(s) saved to {dest}/")
        print("You can now start the backend:")
        print("  bash start_backend.sh")
    else:
        print(f"\nSome downloads failed.  You can re-run to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
