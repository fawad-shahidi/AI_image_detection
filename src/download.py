"""Phase 0: download the two core HF datasets.

Both sources are already-chunked parquet repos (14-17 files each), not giant
single zips, so plain `huggingface_hub.snapshot_download` is enough -- it
writes partial files with a `.incomplete` suffix and resumes them via HTTP
Range on retry, no custom byte-range logic needed (unlike the video
project's GenBuster zips, which required the sparse-zip trick because they
were single multi-GB archives).

Re-running this script after an interruption resumes rather than restarts.
"""
import argparse
import os
import time
from pathlib import Path

# The Xet CDN backend (us.aws.cdn.hf.co) hit repeated DNS-resolution and
# read-timeout failures on this connection during the first run -- disabling
# it falls back to plain HTTPS against the regular hub CDN, which held up.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "tiny_genimage": "TheKernel01/Tiny-GenImage",       # 8.31GB, 35K images, 8 legacy/mid generators
    "defactify": "Rajarshi-Roy-research/Defactify_Image_Dataset",  # 7.52GB, 96K images, 5 modern generators
}


def download(name: str, repo_id: str, max_retries: int = 30):
    local_dir = RAW_DIR / name
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] downloading {repo_id} -> {local_dir}", flush=True)
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                max_workers=1,  # single connection -- 2 workers thrashed the same slow uplink and made timeouts worse
            )
            print(f"[{name}] done", flush=True)
            return
        except Exception as e:
            wait = min(120, 5 * attempt)
            print(f"[{name}] attempt {attempt}/{max_retries} failed: {e!r} -- retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"[{name}] failed after {max_retries} attempts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(DATASETS), help="download just one dataset")
    args = parser.parse_args()

    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS
    for name, repo_id in targets.items():
        download(name, repo_id)


if __name__ == "__main__":
    main()
