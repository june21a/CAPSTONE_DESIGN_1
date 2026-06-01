#!/usr/bin/env python3
"""Delete generated visualization images from CARLA result runs."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "carla_garage" / "results"
TARGETS = {
    "semantic": (("vision_tasks", "semantic"),),
    "gt_semantic": (("vision_tasks_gt", "semantic"),),
    "bev_semantic": (("vision_tasks", "bev_semantic"), ("vision_tasks_gt", "bev_semantic")),
    "depth": (("vision_tasks", "depth"), ("vision_tasks_gt", "depth")),
    "detection": (("vision_tasks", "detection"),),
    "rgb_attention": (("attention_overlay",),),
    "lidar_attention": (("lidar_attention_overlay",),),
    "2d_box_overlay": (("vision_tasks_gt", "2d_box_overlay"),),
}


def target_names(requested_targets: list[str]) -> list[str]:
    if "all" in requested_targets:
        return list(TARGETS)
    return requested_targets


def experiment_roots(results_root: Path, exp_name: str) -> list[Path]:
    if exp_name == "all":
        return sorted(path for path in results_root.iterdir() if path.is_dir())
    return [results_root / exp_name]


def sensor_data_roots(results_root: Path, exp_name: str) -> list[Path]:
    roots: list[Path] = []
    for exp_root in experiment_roots(results_root, exp_name):
        if not exp_root.is_dir():
            print(f"Experiment folder does not exist: {exp_root}")
            continue
        roots.extend(sorted(path / "sensor_data" for path in exp_root.iterdir() if (path / "sensor_data").is_dir()))
    return roots


def target_dirs(sensor_root: Path, requested_targets: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for target in target_names(requested_targets):
        dirs.extend(sensor_root.joinpath(*parts) for parts in TARGETS[target])
    return dirs


def find_visualization_images(sensor_roots: list[Path], requested_targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for sensor_root in sensor_roots:
        for directory in target_dirs(sensor_root, requested_targets):
            if directory.is_dir():
                paths.extend(sorted(directory.glob("*.png")))
    return sorted(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete generated visualization PNGs under sensor_data/vision_tasks, "
            "sensor_data/vision_tasks_gt, and attention overlay folders."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Results directory to scan. Defaults to carla_garage/results.",
    )
    parser.add_argument(
        "--exp-name",
        default="all",
        help="Experiment folder under results-root to process, or 'all' for every experiment. Defaults to all.",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=["all"],
        choices=["all", *TARGETS.keys()],
        help="Visualization target(s) to delete. Defaults to all.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete matched images. Without this flag, only prints a dry run.",
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

    sensor_roots = sensor_data_roots(results_root, args.exp_name)
    if not sensor_roots:
        print(f"No sensor_data folders found for exp-name={args.exp_name!r} under {results_root}")
        return 1

    image_paths = find_visualization_images(sensor_roots, args.target)
    action = "Deleting" if args.delete else "Would delete"

    for image_path in image_paths:
        print(f"{action}: {image_path}")

    if args.delete:
        for image_path in image_paths:
            image_path.unlink()

    print(f"{'Deleted' if args.delete else 'Found'} {len(image_paths)} visualization image(s).")
    if not args.delete:
        print("Run again with --delete to remove them.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
