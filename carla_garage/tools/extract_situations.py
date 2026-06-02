#!/usr/bin/env python3
"""Extract crossroads and lane-change situation frames from CARLA Garage data.

The script walks route folders under a training-data root, reads
``measurements/*.json.gz``, and writes a CSV or JSONL manifest with useful
frame-level labels and paths.

Labels:
  is_crossroad = measurement["junction"] is true, or optionally a junction scenario
  is_lane_change = command/next_command is CHANGELANELEFT/CHANGELANERIGHT, or
                   optionally a lane-interaction scenario
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable


LANE_CHANGE_COMMANDS = {5, 6}

CROSSROAD_SCENARIO_KEYWORDS = (
    "Intersection",
    "Junction",
    "VehicleTurningRoute",
    "OppositeVehicleRunningRedLight",
    "OppositeVehicleTakingPriority",
)

LANE_INTERACTION_SCENARIO_KEYWORDS = (
    "CutIn",
    "HazardAtSideLane",
    "MergerIntoSlowTraffic",
    "ConstructionObstacle",
    "InvadingTurn",
    "ParkedObstacle",
    "ParkingExit",
)


def load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def scenario_matches(scenario_name: str, keywords: Iterable[str]) -> bool:
    return any(keyword.lower() in scenario_name.lower() for keyword in keywords)


def iter_route_dirs(data_root: Path):
    for measurements_dir in data_root.glob("*/*/measurements"):
        if measurements_dir.is_dir():
            yield measurements_dir.parent


def frame_id_from_path(path: Path) -> str:
    return path.stem.removesuffix(".json")


def extract_frame(
    data_root: Path,
    route_dir: Path,
    measurement_path: Path,
    use_scenario_labels: bool,
) -> dict[str, Any]:
    measurement = load_json_gz(measurement_path)

    scenario = route_dir.parent.name
    route = route_dir.name
    frame = frame_id_from_path(measurement_path)
    command = as_int(measurement.get("command"))
    next_command = as_int(measurement.get("next_command"))

    is_crossroad_measurement = as_bool(measurement.get("junction"))
    is_lane_change_command = command in LANE_CHANGE_COMMANDS or next_command in LANE_CHANGE_COMMANDS
    is_crossroad_scenario = scenario_matches(scenario, CROSSROAD_SCENARIO_KEYWORDS)
    is_lane_interaction_scenario = scenario_matches(scenario, LANE_INTERACTION_SCENARIO_KEYWORDS)

    is_crossroad = is_crossroad_measurement or (use_scenario_labels and is_crossroad_scenario)
    is_lane_change = is_lane_change_command or (use_scenario_labels and is_lane_interaction_scenario)

    rel_route_dir = route_dir.relative_to(data_root)
    rel_measurement_path = measurement_path.relative_to(data_root)
    rel_boxes_path = rel_route_dir / "boxes" / f"{frame}.json.gz"
    rel_rgb_path = rel_route_dir / "rgb" / f"{frame}.jpg"

    return {
        "scenario": scenario,
        "route": route,
        "frame": frame,
        "is_crossroad": int(is_crossroad),
        "is_lane_change": int(is_lane_change),
        "is_crossroad_measurement": int(is_crossroad_measurement),
        "is_crossroad_scenario": int(is_crossroad_scenario),
        "is_lane_change_command": int(is_lane_change_command),
        "is_lane_interaction_scenario": int(is_lane_interaction_scenario),
        "command": command,
        "next_command": next_command,
        "speed": measurement.get("speed"),
        "target_speed": measurement.get("target_speed"),
        "steer": measurement.get("steer"),
        "throttle": measurement.get("throttle"),
        "brake": int(as_bool(measurement.get("brake"))),
        "measurement_path": str(rel_measurement_path),
        "boxes_path": str(rel_boxes_path),
        "rgb_path": str(rel_rgb_path),
    }


def wanted(row: dict[str, Any], include: str) -> bool:
    if include == "all":
        return True
    if include == "crossroads":
        return bool(row["is_crossroad"])
    if include == "lane_changes":
        return bool(row["is_lane_change"])
    return bool(row["is_crossroad"] or row["is_lane_change"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "route",
        "frame",
        "is_crossroad",
        "is_lane_change",
        "is_crossroad_measurement",
        "is_crossroad_scenario",
        "is_lane_change_command",
        "is_lane_interaction_scenario",
        "command",
        "next_command",
        "speed",
        "target_speed",
        "steer",
        "throttle",
        "brake",
        "measurement_path",
        "boxes_path",
        "rgb_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract crossroads and lane-change frame manifests from CARLA Garage training data."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("carla_garage/training_data"),
        help="Training-data root containing scenario/route folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("carla_garage/training_data/situation_manifest.csv"),
        help="Output manifest path. Use .jsonl for JSON Lines, otherwise CSV.",
    )
    parser.add_argument(
        "--include",
        choices=("positive", "crossroads", "lane_changes", "all"),
        default="positive",
        help="Which frames to write. positive means crossroads OR lane_changes.",
    )
    parser.add_argument(
        "--use-scenario-labels",
        action="store_true",
        help="Also label frames by scenario folder keywords, not only measurement fields/commands.",
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=None,
        help="Optional debug limit on number of route folders to scan.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit on number of measurement frames to scan.",
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    rows: list[dict[str, Any]] = []
    total_frames = 0
    total_routes = 0

    stop = False
    for route_dir in iter_route_dirs(args.data_root):
        if args.max_routes is not None and total_routes >= args.max_routes:
            break
        total_routes += 1
        for measurement_path in sorted((route_dir / "measurements").glob("*.json.gz")):
            if args.max_frames is not None and total_frames >= args.max_frames:
                stop = True
                break
            total_frames += 1
            row = extract_frame(args.data_root, route_dir, measurement_path, args.use_scenario_labels)
            if wanted(row, args.include):
                rows.append(row)
        if stop:
            break

    if args.output.suffix.lower() == ".jsonl":
        write_jsonl(args.output, rows)
    else:
        write_csv(args.output, rows)

    crossroad_count = sum(int(row["is_crossroad"]) for row in rows)
    lane_change_count = sum(int(row["is_lane_change"]) for row in rows)
    both_count = sum(int(row["is_crossroad"] and row["is_lane_change"]) for row in rows)

    print(f"Scanned routes: {total_routes}")
    print(f"Scanned frames: {total_frames}")
    print(f"Wrote frames: {len(rows)}")
    print(f"crossroads: {crossroad_count}")
    print(f"lane_changes: {lane_change_count}")
    print(f"both: {both_count}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
