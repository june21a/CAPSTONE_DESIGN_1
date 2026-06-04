#!/usr/bin/env python3
"""Generate plan-safety labels from paired success/failure CARLA simulator runs."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm


COLLISION_KEYS = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")
COLLISION_LOCATION_PATTERN = re.compile(
    r"\(x=([+-]?(?:\d+(?:\.\d*)?|\.\d+)), "
    r"y=([+-]?(?:\d+(?:\.\d*)?|\.\d+)), "
    r"z=([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)"
)


def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def dump_json_gz(path: Path, data: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def iter_route_dirs(data_root: Path):
    for measurements_dir in data_root.glob("*/*/measurements"):
        if measurements_dir.is_dir():
            yield measurements_dir.parent
    for measurements_dir in data_root.glob("*/*/*/measurements"):
        if measurements_dir.is_dir():
            yield measurements_dir.parent


def load_results(route_dir: Path) -> dict[str, Any] | None:
    results_path = route_dir / "results.json.gz"
    if not results_path.is_file():
        return None
    return load_json_gz(results_path)


def has_collision_result(results: dict[str, Any] | None) -> bool:
    if results is None:
        return False
    infractions = results.get("infractions", {})
    return any(bool(infractions.get(key)) for key in COLLISION_KEYS)


def collision_locations(results: dict[str, Any] | None) -> list[tuple[float, float, float]]:
    if results is None:
        return []
    locations = []
    infractions = results.get("infractions", {})
    for key in COLLISION_KEYS:
        for message in infractions.get(key, []):
            match = COLLISION_LOCATION_PATTERN.search(message)
            if match:
                locations.append(tuple(float(value) for value in match.groups()))
    return locations


def measurement_position(measurement: dict[str, Any]) -> tuple[float, float, float] | None:
    pos_global = measurement.get("pos_global")
    if isinstance(pos_global, list) and len(pos_global) >= 2:
        z = pos_global[2] if len(pos_global) >= 3 else 0.0
        return float(pos_global[0]), float(pos_global[1]), float(z)

    ego_matrix = measurement.get("ego_matrix")
    if isinstance(ego_matrix, list) and len(ego_matrix) >= 3:
        try:
            return float(ego_matrix[0][3]), float(ego_matrix[1][3]), float(ego_matrix[2][3])
        except (IndexError, TypeError, ValueError):
            return None
    return None


def infer_collision_data_frame_from_location(
    paths: list[Path],
    measurements: list[dict[str, Any]],
    locations: list[tuple[float, float, float]],
    max_distance: float,
) -> tuple[int | None, float | None]:
    if not locations:
        return None, None

    best_frame = None
    best_distance = None
    for path, measurement in zip(paths, measurements):
        position = measurement_position(measurement)
        if position is None:
            continue
        frame = int(path.stem.split(".")[0])
        for location in locations:
            distance = math.hypot(position[0] - location[0], position[1] - location[1])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_frame = frame

    if best_distance is None or best_distance > max_distance:
        return None, best_distance
    return best_frame, best_distance


def collision_data_frame(
    results: dict[str, Any] | None,
    paths: list[Path],
    measurements: list[dict[str, Any]],
    infer_from_location: bool,
    max_location_distance: float,
    data_save_freq: int,
) -> tuple[int | None, str | None, float | None]:
    if results is None:
        return None, None, None

    meta = results.get("meta", {})
    collision_frame = meta.get("collision_frame")
    if collision_frame is not None:
        return int(collision_frame) // int(data_save_freq), "meta.collision_frame", None

    data_frame = meta.get("collision_data_frame")
    if data_frame is not None:
        return int(data_frame), "meta.collision_data_frame", None

    if not infer_from_location:
        return None, None, None

    frame, distance = infer_collision_data_frame_from_location(
        paths,
        measurements,
        collision_locations(results),
        max_location_distance,
    )
    if frame is None:
        return None, None, distance
    return frame, "infraction_location_nearest_ego_pose", distance


def measurement_paths(route_dir: Path):
    return sorted((route_dir / "measurements").glob("*.json.gz"))


def focus_frame(measurement: dict[str, Any], all_frames: bool) -> bool:
    if all_frames:
        return True
    if measurement.get("junction"):
        return True
    command = measurement.get("command")
    next_command = measurement.get("next_command")
    return command in (5, 6) or next_command in (5, 6) or bool(measurement.get("changed_route"))


def candidate_from_measurement(measurement: dict[str, Any], pred_len: int, unsafe: bool) -> dict[str, Any]:
    waypoints = measurement.get("route", [])[:pred_len]
    target_speed = measurement.get("target_speed", 0.0)

    return {
        "variant": "expert",
        "waypoints": waypoints,
        "target_speed": round(float(target_speed), 4),
        "expert_target_speed": round(float(measurement.get("expert_target_speed", target_speed)), 4),
        "will_collide": int(unsafe),
    }


def select_safe_frame_keys(
    safe_frame_keys: list[str],
    unsafe_count: int,
    route_had_collision: bool,
    safe_samples_per_unsafe: float,
    max_safe_per_non_collision_route: int,
    rng: random.Random,
) -> set[str]:
    if route_had_collision and unsafe_count == 0:
        return set()
    if route_had_collision:
        safe_limit = int(round(unsafe_count * safe_samples_per_unsafe))
    else:
        safe_limit = max_safe_per_non_collision_route

    safe_limit = max(0, min(safe_limit, len(safe_frame_keys)))
    if safe_limit == 0:
        return set()
    return set(rng.sample(safe_frame_keys, safe_limit))


def label_route(
    route_dir: Path,
    pred_len: int,
    all_frames: bool,
    overwrite: bool,
    infer_collision_frame_from_location: bool,
    max_collision_location_distance: float,
    data_save_freq: int,
    safe_samples_per_unsafe: float,
    max_safe_per_non_collision_route: int,
    seed: int,
) -> tuple[int, int, int, bool, bool]:
    output_path = route_dir / "plan_safety_labels.json.gz"
    if output_path.exists() and not overwrite:
        return 0, 0, 0, False, True

    paths = measurement_paths(route_dir)
    if not paths:
        return 0, 0, 0, False, False

    measurements = [load_json_gz(path) for path in paths]
    results = load_results(route_dir)
    collision = has_collision_result(results)
    collision_frame, collision_frame_source, collision_location_distance = collision_data_frame(
        results,
        paths,
        measurements,
        infer_collision_frame_from_location,
        max_collision_location_distance,
        data_save_freq,
    )
    unresolved_collision = collision and collision_frame is None
    unsafe_window_frames = 10
    unsafe_start_frame = None if collision_frame is None else max(0, collision_frame - unsafe_window_frames)
    case_label = "collision" if collision else "success"

    candidate_frames = []
    safe_frame_keys = []
    unsafe_count = 0
    for path, measurement in zip(paths, measurements):
        frame_key = path.stem.split(".")[0]
        frame = int(frame_key)
        if collision_frame is not None and frame >= collision_frame:
            continue
        if not focus_frame(measurement, all_frames):
            continue
        unsafe = unsafe_start_frame is not None and unsafe_start_frame <= frame < collision_frame
        candidate = candidate_from_measurement(measurement, pred_len, unsafe)
        candidate_frames.append((frame_key, candidate, unsafe))
        unsafe_count += int(unsafe)
        if not unsafe:
            safe_frame_keys.append(frame_key)

    route_seed = seed + sum(ord(char) for char in str(route_dir))
    selected_safe_frame_keys = select_safe_frame_keys(
        safe_frame_keys,
        unsafe_count,
        collision,
        safe_samples_per_unsafe,
        max_safe_per_non_collision_route,
        random.Random(route_seed),
    )

    frames = {}
    safe_count = 0
    for frame_key, candidate, unsafe in candidate_frames:
        if not unsafe and frame_key not in selected_safe_frame_keys:
            continue
        frames[frame_key] = [candidate]
        safe_count += int(not unsafe)

    if not frames:
        return 0, 0, 0, unresolved_collision, False

    dump_json_gz(output_path, {
        "label_map": {"safe": 0, "unsafe_sim_collision": 1},
        "source": "carla_simulator",
        "case_label": case_label,
        "route_had_collision": collision,
        "collision_data_frame": collision_frame,
        "collision_frame_source": collision_frame_source,
        "collision_location_distance": collision_location_distance,
        "unsafe_window_frames": unsafe_window_frames,
        "safe_samples_per_unsafe": safe_samples_per_unsafe,
        "max_safe_per_non_collision_route": max_safe_per_non_collision_route,
        "pred_len": pred_len,
        "frames": frames,
    })
    return len(frames), safe_count, unsafe_count, unresolved_collision, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simulator-derived plan safety labels.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pred-len", type=int, default=8)
    parser.add_argument("--data-save-freq", type=int, default=10)
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Label every saved frame. This is the default; kept for backward-compatible commands.",
    )
    parser.add_argument(
        "--focus-frames-only",
        action="store_true",
        help="Only label junction, turn, lane-change, or changed-route frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate plan_safety_labels.json.gz even if it already exists.",
    )
    parser.add_argument(
        "--safe-samples-per-unsafe",
        type=float,
        default=1.0,
        help="For collision routes, save at most this many safe labels per unsafe label.",
    )
    parser.add_argument(
        "--max-safe-per-non-collision-route",
        type=int,
        default=0,
        help="For routes without collision, save at most this many safe labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-infer-collision-frame-from-location",
        action="store_true",
        help="Only use collision frame metadata; do not infer frame from collision location messages.",
    )
    parser.add_argument(
        "--max-collision-location-distance",
        type=float,
        default=5.0,
        help="Maximum XY distance in meters for matching a collision location to a saved ego pose.",
    )
    args = parser.parse_args()

    route_dirs = sorted(set(iter_route_dirs(args.data_root)))
    if not route_dirs:
        raise FileNotFoundError(f"No route measurement folders found under {args.data_root}")

    route_count = 0
    frame_count = 0
    safe_count = 0
    unsafe_count = 0
    unresolved_collision_count = 0
    skipped_existing_count = 0
    for route_dir in tqdm(route_dirs, desc="Generating plan-safety labels", unit="route"):
        frames, safe, unsafe, unresolved_collision, skipped_existing = label_route(
            route_dir,
            args.pred_len,
            args.all_frames or not args.focus_frames_only,
            args.overwrite,
            not args.no_infer_collision_frame_from_location,
            args.max_collision_location_distance,
            args.data_save_freq,
            args.safe_samples_per_unsafe,
            args.max_safe_per_non_collision_route,
            args.seed,
        )
        if frames:
            route_count += 1
            frame_count += frames
            safe_count += safe
            unsafe_count += unsafe
        unresolved_collision_count += int(unresolved_collision)
        skipped_existing_count += int(skipped_existing)

    print(f"Labeled routes: {route_count}")
    print(f"Labeled frames: {frame_count}")
    print(f"safe=0: {safe_count}")
    print(f"unsafe_sim_collision=1: {unsafe_count}")
    print(f"Skipped existing label files: {skipped_existing_count}")
    print(f"collision routes without usable collision frame: {unresolved_collision_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
