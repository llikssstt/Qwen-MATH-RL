#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download Qwen3 model files from a Hugging Face mirror endpoint.

Examples:
  # Basic (mirror endpoint via env)
  export HF_ENDPOINT=https://hf-mirror.com
  python download_qwen3_from_mirror.py --repo Qwen/Qwen3-1.7B --out ./models/Qwen3-1.7B

  # With token (if required)
  export HF_TOKEN=hf_xxx
  python download_qwen3_from_mirror.py --repo Qwen/Qwen3-1.7B --out ./models/Qwen3-1.7B

  # Only download essential files (avoid training artifacts)
  python download_qwen3_from_mirror.py --repo Qwen/Qwen3-1.7B --out ./models/Qwen3-1.7B --allow "config.json" "tokenizer.*" "*.safetensors" "*.model" "*.json"

  # Specify revision
  python download_qwen3_from_mirror.py --repo Qwen/Qwen3-1.7B --revision main --out ./models/Qwen3-1.7B
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

def _parse_args():
    p = argparse.ArgumentParser(description="Download Qwen3 model via HF mirror.")
    p.add_argument("--repo", type=str, required=True, help="Model repo id, e.g. Qwen/Qwen3-1.7B")
    p.add_argument("--out", type=str, required=True, help="Output directory to store snapshot")
    p.add_argument("--revision", type=str, default=None, help="Branch/tag/commit (optional)")
    p.add_argument("--allow", nargs="*", default=None,
                   help="Optional allow patterns (glob). If set, only these files are downloaded.")
    p.add_argument("--ignore", nargs="*", default=None,
                   help="Optional ignore patterns (glob). Files matching these patterns are skipped.")
    p.add_argument("--resume", action="store_true", help="Resume download (default: on).")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume.")
    p.set_defaults(resume=True)
    p.add_argument("--local-dir-use-symlinks", type=str, default="auto",
                   choices=["auto", "true", "false"],
                   help="Symlink strategy used by huggingface_hub (auto/true/false).")
    return p.parse_args()

def main():
    args = _parse_args()

    # Mirror endpoint: e.g. https://hf-mirror.com
    hf_endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if not hf_endpoint:
        print("[WARN] HF_ENDPOINT is not set. If you want a mirror, set it, e.g.:")
        print("       export HF_ENDPOINT=https://hf-mirror.com")
    else:
        print(f"[INFO] Using HF_ENDPOINT={hf_endpoint}")

    token = os.environ.get("HF_TOKEN", None)
    if token:
        print("[INFO] HF_TOKEN is set (will use for auth if needed).")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub is not installed. Install it via:")
        print("        pip install -U huggingface_hub")
        sys.exit(1)

    allow_patterns: Optional[List[str]] = args.allow if args.allow else None
    ignore_patterns: Optional[List[str]] = args.ignore if args.ignore else None

    # Map symlink string to bool/None as huggingface_hub expects
    if args.local_dir_use_symlinks == "auto":
        local_dir_use_symlinks = "auto"
    elif args.local_dir_use_symlinks == "true":
        local_dir_use_symlinks = True
    else:
        local_dir_use_symlinks = False

    print("[INFO] Starting snapshot_download ...")
    print(f"       repo: {args.repo}")
    print(f"       out : {out_dir}")
    if args.revision:
        print(f"       rev : {args.revision}")
    if allow_patterns:
        print(f"       allow_patterns : {allow_patterns}")
    if ignore_patterns:
        print(f"       ignore_patterns: {ignore_patterns}")

    # snapshot_download will automatically use HF_ENDPOINT env var for base URL
    local_path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(out_dir),
        local_dir_use_symlinks=local_dir_use_symlinks,
        resume_download=args.resume,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        token=token,
    )

    print("[OK] Download finished.")
    print(f"     Local snapshot path: {local_path}")

if __name__ == "__main__":
    main()
