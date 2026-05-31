#!/usr/bin/env python3
"""Delete generated attention overlay folders from CARLA result runs."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


TARGET_DIR_NAMES = {"attention_overlay", "lidar_attention_overlay"}


def find_overlay_dirs(results_root: Path) -> list[Path]:
    """Return overlay directories under results_root without descending into them."""
    overlay_dirs: list[Path] = []

    for current_root, dirnames, _filenames in os.walk(results_root):
        matched = [name for name in dirnames if name in TARGET_DIR_NAMES]
        for name in matched:
            overlay_dirs.append(Path(current_root) / name)

        dirnames[:] = [name for name in dirnames if name not in TARGET_DIR_NAMES]

    return sorted(overlay_dirs)


def remove_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete directories named attention_overlay or lidar_attention_overlay "
            "under a results folder."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/results"),
        help="Results directory to scan. Defaults to carla_garage/results.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete matched directories. Without this flag, only prints a dry run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()

    if not results_root.exists():
        print(f"Results root does not exist: {results_root}")
        return 1
    if not results_root.is_dir():
        print(f"Results root is not a directory: {results_root}")
        return 1

    overlay_dirs = find_overlay_dirs(results_root)
    action = "Deleting" if args.delete else "Would delete"

    for overlay_dir in overlay_dirs:
        print(f"{action}: {overlay_dir}")

    if args.delete:
        for overlay_dir in overlay_dirs:
            remove_path(overlay_dir)

    print(f"{'Deleted' if args.delete else 'Found'} {len(overlay_dirs)} overlay directories.")
    if not args.delete:
        print("Run again with --delete to remove them.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
