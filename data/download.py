"""
Download open-source FPS mouse datasets for fine-tuning.

Sources:
  1. CS:GO Behavioral Cloning — TeaPearce/CounterStrike_Deathmatch (HuggingFace)
     Contains mouse dx/dy actions from gameplay recordings.

  2. Red Eclipse FPS — University of Manchester
     476 games from 45 players with keyboard+mouse events.

Usage:
    python -m data.download            # download all
    python -m data.download --csgo     # CS:GO only
    python -m data.download --redeclipse  # Red Eclipse only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


EXTERNAL_DIR = Path("data/external")


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file with progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  ⏭️  Already exists: {dest}")
        return dest

    print(f"  ⬇️  Downloading {desc or url}...")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {pct:3d}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=progress)
        print()
        return dest
    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        if dest.exists():
            dest.unlink()
        raise


def download_csgo():
    """
    Download CS:GO mouse data from HuggingFace.

    The TeaPearce/CounterStrike_Deathmatch dataset has .npy files
    containing gameplay frames + mouse actions. We download a subset
    for fine-tuning.
    """
    out_dir = EXTERNAL_DIR / "csgo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata/readme
    meta_url = "https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch/resolve/main/README.md"
    download_file(meta_url, out_dir / "README.md", "CS:GO dataset README")

    # Download a few data chunks (each ~200 files of gameplay data)
    # We only need the action data (mouse dx, dy), not the screenshots
    for chunk_id in range(1, 4):  # 3 chunks = ~600 sessions
        fname = f"hdf5_dm_scraped_dust2_chunk{chunk_id}.zip"
        url = f"https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch/resolve/main/{fname}"
        dest = out_dir / fname
        try:
            download_file(url, dest, f"CS:GO chunk {chunk_id}")
        except Exception:
            # HuggingFace might need different URL format
            print(f"  ⚠️  Could not download chunk {chunk_id}, skipping")
            continue

        # Extract if it's a zip
        if dest.exists() and dest.suffix == ".zip":
            extract_dir = out_dir / f"chunk{chunk_id}"
            if not extract_dir.exists():
                print(f"  📦 Extracting {fname}...")
                try:
                    with zipfile.ZipFile(str(dest), 'r') as zf:
                        zf.extractall(str(extract_dir))
                except zipfile.BadZipFile:
                    print(f"  ⚠️  Bad zip file, skipping")

    print(f"✅ CS:GO data → {out_dir}")


def download_redeclipse():
    """
    Download Red Eclipse FPS mouse data from University of Manchester.

    476 games × 45 players, JSON format with mouse+keyboard events.
    """
    out_dir = EXTERNAL_DIR / "redeclipse"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The dataset is hosted as a zip on the research portal
    # Try the direct download link
    urls = [
        ("https://raw.githubusercontent.com/ahurst1/RedEclipse-FPS-Data/main/data.zip",
         "Red Eclipse data (GitHub mirror)"),
        # Fallback: generate synthetic Red Eclipse-like data
    ]

    downloaded = False
    for url, desc in urls:
        try:
            dest = out_dir / "data.zip"
            download_file(url, dest, desc)
            if dest.exists():
                print(f"  📦 Extracting...")
                with zipfile.ZipFile(str(dest), 'r') as zf:
                    zf.extractall(str(out_dir))
                downloaded = True
                break
        except Exception:
            continue

    if not downloaded:
        print("  ⚠️  Could not download Red Eclipse data.")
        print("  ℹ️  The dataset may require manual download from:")
        print("      https://research.manchester.ac.uk/en/datasets/")
        print("      Search for 'Red Eclipse FPS keyboard mouse data'")

    print(f"✅ Red Eclipse data → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="📦 Download FPS mouse datasets")
    parser.add_argument("--csgo", action="store_true", help="Download CS:GO data only")
    parser.add_argument("--redeclipse", action="store_true", help="Download Red Eclipse only")
    args = parser.parse_args()

    do_all = not args.csgo and not args.redeclipse

    print("=" * 50)
    print("📦 RawAccel Studio — Dataset Downloader")
    print("=" * 50)

    if do_all or args.csgo:
        print("\n🎮 CS:GO Behavioral Cloning Dataset")
        download_csgo()

    if do_all or args.redeclipse:
        print("\n🎮 Red Eclipse FPS Dataset")
        download_redeclipse()

    print("\n✅ Done! Run: python -m data.convert")


if __name__ == "__main__":
    main()
