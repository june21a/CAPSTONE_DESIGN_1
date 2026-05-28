#!/usr/bin/env python3
"""Auto-label CARLA Garage measurements with stop/move mode labels.

The script walks route folders under a training-data root, reads
``measurements/*.json.gz``, and writes one ``mode_labels.json.gz`` file per
route folder.

Labels:
  0 = stop
  1 = move
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


STOP = 0
MOVE = 1


def load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def dump_json_gz(path: Path, data: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def infer_mode(
    measurement: dict[str, Any],
    target_speed_stop_threshold: float,
    throttle_move_threshold: float,
) -> int:
    """Infer decision mode from an autopilot measurement.

    Prefer the planner decision signals (brake and target_speed). Fall back to
    throttle only when both are unavailable.
    """
    brake = as_bool(measurement.get("brake", False))
    target_speed = measurement.get("target_speed")

    if brake:
        return STOP

    if target_speed is not None:
        return STOP if float(target_speed) <= target_speed_stop_threshold else MOVE

    throttle = measurement.get("throttle")
    if throttle is not None:
        return MOVE if float(throttle) > throttle_move_threshold else STOP

    return MOVE


def label_route(
    route_dir: Path,
    output_name: str,
    target_speed_stop_threshold: float,
    throttle_move_threshold: float,
    overwrite: bool,
) -> tuple[int, int]:
    measurements_dir = route_dir / "measurements"
    output_path = route_dir / output_name

    if not measurements_dir.is_dir():
        return 0, 0

    if output_path.exists() and not overwrite:
        labels = load_json_gz(output_path)
        return int(labels.get("summary", {}).get("num_stop", 0)), int(labels.get("summary", {}).get("num_move", 0))

    frame_labels: dict[str, int] = {}
    stop_count = 0
    move_count = 0

    for measurement_path in sorted(measurements_dir.glob("*.json.gz")):
        measurement = load_json_gz(measurement_path)
        label = infer_mode(measurement, target_speed_stop_threshold, throttle_move_threshold)
        frame_labels[measurement_path.stem.removesuffix(".json")] = label

        if label == STOP:
            stop_count += 1
        else:
            move_count += 1

    if not frame_labels:
        return 0, 0

    dump_json_gz(
        output_path,
        {
            "label_map": {"stop": STOP, "move": MOVE},
            "rule": {
                "stop": "brake == true OR target_speed <= target_speed_stop_threshold",
                "move": "otherwise",
                "target_speed_stop_threshold": target_speed_stop_threshold,
                "throttle_move_threshold": throttle_move_threshold,
            },
            "summary": {
                "num_frames": len(frame_labels),
                "num_stop": stop_count,
                "num_move": move_count,
            },
            "frames": frame_labels,
        },
    )
    return stop_count, move_count


def iter_route_dirs(data_root: Path):
    for measurements_dir in data_root.glob("*/*/measurements"):
        if measurements_dir.is_dir():
            yield measurements_dir.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label measurements with MODE: 0=stop, 1=move.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("carla_garage/training_data"),
        help="Training-data root containing scenario/route folders.",
    )
    parser.add_argument(
        "--output-name",
        default="mode_labels.json.gz",
        help="Per-route label file name.",
    )
    parser.add_argument(
        "--target-speed-stop-threshold",
        type=float,
        default=0.1,
        help="target_speed at or below this value is labeled stop.",
    )
    parser.add_argument(
        "--throttle-move-threshold",
        type=float,
        default=0.01,
        help="Fallback threshold used only if brake and target_speed are missing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files.",
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    route_count = 0
    total_stop = 0
    total_move = 0

    for route_dir in iter_route_dirs(args.data_root):
        stop_count, move_count = label_route(
            route_dir,
            args.output_name,
            args.target_speed_stop_threshold,
            args.throttle_move_threshold,
            args.overwrite,
        )
        if stop_count or move_count:
            route_count += 1
            total_stop += stop_count
            total_move += move_count

    print(f"Labeled {route_count} route folders under {args.data_root}")
    print(f"stop=0: {total_stop}")
    print(f"move=1: {total_move}")
    print(f"total: {total_stop + total_move}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
