#!/usr/bin/env python3
"""Generate plan-safety labels from paired success/failure CARLA simulator runs."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


COLLISION_KEYS = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")
COLLISION_LOCATION_PATTERN = re.compile(
    r"\(x=([+-]?(?:\d+(?:\.\d*)?|\.\d+)), "
    r"y=([+-]?(?:\d+(?:\.\d*)?|\.\d+)), "
    r"z=([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)"
)
VISUALIZED_CLASSES = {"ego_car", "car", "walker", "static", "traffic_light", "stop_sign"}
TRAJECTORY_CLASS_COLORS = {
    "ego_car": (0, 190, 255),
    "car": (245, 145, 35),
    "walker": (40, 190, 95),
    "static": (125, 125, 125),
    "traffic_light": (215, 45, 45),
    "stop_sign": (190, 65, 180),
}
DEFAULT_TRAJECTORY_COLOR = (80, 80, 80)
DEFAULT_EGO_EXTENT = np.array([2.45, 1.06], dtype=np.float32)
DEFAULT_SIM_FPS = 20.0
DEFAULT_MAX_ACCELERATION = 1.89
DEFAULT_MAX_DECELERATION = 4.82
DEFAULT_COLLISION_REGION_RADIUS = float(np.linalg.norm(DEFAULT_EGO_EXTENT))
STRAIGHT_WAYPOINT_MAX_LATERAL_SPREAD = 0.5
STRAIGHT_WAYPOINT_MAX_HEADING_CHANGE_DEG = 5.0
MAX_CHECKED_FRAMES_BEFORE_EVENT = 10
DEFAULT_NONZERO_STEER_THRESHOLD = 1e-3
COLLISION_REGION_EGO_BOX_SAMPLE_SPACING = 0.5


def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def dump_json_gz(path: Path, data: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def iter_route_dirs(data_root: Path):
    for measurements_dir in data_root.glob("*/measurements"):
        if measurements_dir.is_dir():
            yield measurements_dir.parent
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


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_matrix(matrix: np.ndarray) -> float:
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def actor_pose_in_origin(origin_matrix: np.ndarray, actor: dict[str, Any]) -> tuple[float, float, float] | None:
    actor_matrix_raw = actor.get("matrix")
    if actor_matrix_raw is None:
        return None

    try:
        actor_matrix = np.array(actor_matrix_raw, dtype=np.float64)
        relative_matrix = np.linalg.inv(origin_matrix) @ actor_matrix
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None

    yaw = normalize_angle(yaw_from_matrix(actor_matrix) - yaw_from_matrix(origin_matrix))
    return float(relative_matrix[0, 3]), float(relative_matrix[1, 3]), yaw


def infer_collision_data_frames_from_locations(
    paths: list[Path],
    measurements: list[dict[str, Any]],
    locations: list[tuple[float, float, float]],
    max_distance: float,
) -> list[dict[str, Any]]:
    events = []
    for location_index, location in enumerate(locations):
        best_frame = None
        best_distance = None
        for path, measurement in zip(paths, measurements):
            position = measurement_position(measurement)
            if position is None:
                continue
            frame = int(path.stem.split(".")[0])
            distance = math.hypot(position[0] - location[0], position[1] - location[1])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_frame = frame

        if best_frame is None or best_distance is None or best_distance > max_distance:
            events.append({
                "frame": None,
                "source": None,
                "location_distance": best_distance,
                "location_index": location_index,
                "location": list(location),
            })
            continue

        events.append({
            "frame": best_frame,
            "source": "infraction_location_nearest_ego_pose",
            "location_distance": best_distance,
            "location_index": location_index,
            "location": list(location),
        })
    return events


def frame_in_measurement_paths(frame: int | None, paths: list[Path]) -> bool:
    if frame is None:
        return False
    frame_key = f"{int(frame):04}"
    return any(path.stem.split(".")[0] == frame_key for path in paths)


def deduplicate_collision_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame: dict[int, dict[str, Any]] = {}
    unresolved = []
    for event in events:
        frame = event.get("frame")
        if frame is None:
            unresolved.append(event)
            continue
        frame = int(frame)
        existing = by_frame.get(frame)
        if existing is None:
            event["frame"] = frame
            by_frame[frame] = event
            continue

        existing_distance = existing.get("location_distance")
        event_distance = event.get("location_distance")
        if existing_distance is None and event_distance is not None:
            by_frame[frame] = event
        elif existing_distance is not None and event_distance is not None and event_distance < existing_distance:
            by_frame[frame] = event

    resolved = sorted(by_frame.values(), key=lambda event: int(event["frame"]))
    return resolved + unresolved


def collision_data_events(
    results: dict[str, Any] | None,
    paths: list[Path],
    measurements: list[dict[str, Any]],
    infer_from_location: bool,
    max_location_distance: float,
    data_save_freq: int,
) -> list[dict[str, Any]]:
    if results is None:
        return []

    events: list[dict[str, Any]] = []
    meta = results.get("meta", {})
    data_frame = meta.get("collision_data_frame")
    if data_frame is not None:
        data_frame = int(data_frame)
        if frame_in_measurement_paths(data_frame, paths):
            events.append({
                "frame": data_frame,
                "source": "meta.collision_data_frame",
                "location_distance": None,
                "location_index": None,
                "location": None,
            })

    collision_frame = meta.get("collision_frame")
    if collision_frame is not None:
        collision_data_frame = int(collision_frame) // int(data_save_freq)
        if frame_in_measurement_paths(collision_data_frame, paths):
            events.append({
                "frame": collision_data_frame,
                "source": "meta.collision_frame",
                "location_distance": None,
                "location_index": None,
                "location": None,
            })

    if infer_from_location:
        events.extend(infer_collision_data_frames_from_locations(
            paths,
            measurements,
            collision_locations(results),
            max_location_distance,
        ))

    return deduplicate_collision_events(events)


def measurement_paths(route_dir: Path):
    return sorted((route_dir / "measurements").glob("*.json.gz"))


def delete_existing_plan_safety_labels(route_dirs: list[Path]) -> int:
    deleted = 0
    for route_dir in route_dirs:
        labels_path = route_dir / "plan_safety_labels.json.gz"
        if labels_path.is_file():
            labels_path.unlink()
            deleted += 1
    return deleted


def focus_frame(measurement: dict[str, Any], all_frames: bool) -> bool:
    if all_frames:
        return True
    if measurement.get("junction"):
        return True
    command = measurement.get("command")
    next_command = measurement.get("next_command")
    return command in (5, 6) or next_command in (5, 6) or bool(measurement.get("changed_route"))


def speed_mps(measurement: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return max(0.0, float(measurement.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def steer_value(measurement: dict[str, Any], default: float = 0.0) -> float:
    try:
        return float(measurement.get("steer", default) or default)
    except (TypeError, ValueError):
        return default


def consecutive_nonzero_steer_run_lengths(
    paths: list[Path],
    measurements: list[dict[str, Any]],
    nonzero_threshold: float,
) -> dict[str, int]:
    run_lengths: dict[str, int] = {}
    run_frame_keys: list[str] = []
    threshold = max(0.0, float(nonzero_threshold))

    def flush_run() -> None:
        if not run_frame_keys:
            return
        run_length = len(run_frame_keys)
        for run_frame_key in run_frame_keys:
            run_lengths[run_frame_key] = run_length
        run_frame_keys.clear()

    for path, measurement in zip(paths, measurements):
        frame_key = path.stem.split(".")[0]
        if abs(steer_value(measurement)) > threshold:
            run_frame_keys.append(frame_key)
        else:
            flush_run()
    flush_run()
    return run_lengths


def measurement_frame_map(
    paths: list[Path],
    measurements: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {int(path.stem.split(".")[0]): measurement for path, measurement in zip(paths, measurements)}


def ego_extent_from_boxes(route_dir: Path, frame: int | None) -> np.ndarray | None:
    if frame is None:
        return None

    boxes_path = route_dir / "boxes" / f"{frame:04}.json.gz"
    if not boxes_path.is_file():
        return None

    try:
        actors = load_json_gz(boxes_path)
    except (OSError, EOFError, json.JSONDecodeError):
        return None

    for actor in actors:
        if actor.get("class") != "ego_car":
            continue
        extent = actor.get("extent")
        if not isinstance(extent, list) or len(extent) < 2:
            return None
        try:
            return np.array([float(extent[0]), float(extent[1])], dtype=np.float32)
        except (TypeError, ValueError):
            return None
    return None


def ego_extent_from_first_available_boxes(route_dir: Path, paths: list[Path]) -> np.ndarray | None:
    for path in paths:
        frame = int(path.stem.split(".")[0])
        extent = ego_extent_from_boxes(route_dir, frame)
        if extent is not None:
            return extent
    return None


def collision_region_radius_from_extent(extent: np.ndarray) -> float:
    return float(np.linalg.norm(extent[:2].astype(np.float64)))


def resolve_collision_region_radius(
    route_dir: Path,
    paths: list[Path],
    collision_frame: int | None,
    override_radius: float | None,
) -> tuple[float, str, list[float]]:
    if override_radius is not None:
        return max(0.0, float(override_radius)), "cli_override", []

    ego_extent = ego_extent_from_boxes(route_dir, collision_frame)
    if ego_extent is not None:
        return collision_region_radius_from_extent(ego_extent), "ego_extent_collision_frame_boxes", ego_extent.tolist()

    ego_extent = ego_extent_from_first_available_boxes(route_dir, paths)
    if ego_extent is not None:
        return collision_region_radius_from_extent(ego_extent), "ego_extent_dataset_boxes", ego_extent.tolist()

    return DEFAULT_COLLISION_REGION_RADIUS, "default_ego_extent", DEFAULT_EGO_EXTENT.tolist()


def collision_point_in_ego(
    measurement: dict[str, Any],
    collision_measurement: dict[str, Any],
) -> np.ndarray | None:
    ego_matrix_raw = measurement.get("ego_matrix")
    collision_matrix_raw = collision_measurement.get("ego_matrix")
    if ego_matrix_raw is None or collision_matrix_raw is None:
        return None

    try:
        ego_matrix = np.array(ego_matrix_raw, dtype=np.float64)
        collision_matrix = np.array(collision_matrix_raw, dtype=np.float64)
        relative_matrix = np.linalg.inv(ego_matrix) @ collision_matrix
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None

    return relative_matrix[:2, 3].astype(np.float64)


def path_points_from_waypoints(waypoints: list[Any]) -> np.ndarray:
    points = [np.zeros(2, dtype=np.float64)]
    for waypoint in waypoints:
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 2:
            continue
        try:
            points.append(np.array([float(waypoint[0]), float(waypoint[1])], dtype=np.float64))
        except (TypeError, ValueError):
            continue
    return np.asarray(points, dtype=np.float64)


def normalize_ego_extent(ego_extent: list[float] | np.ndarray | None) -> np.ndarray:
    if ego_extent is None:
        return DEFAULT_EGO_EXTENT.astype(np.float64)
    try:
        extent = np.asarray(ego_extent, dtype=np.float64)
    except (TypeError, ValueError):
        return DEFAULT_EGO_EXTENT.astype(np.float64)
    if len(extent) < 2:
        return DEFAULT_EGO_EXTENT.astype(np.float64)
    return np.maximum(0.0, extent[:2])


def ego_box_intersects_circle(
    position: np.ndarray,
    yaw: float,
    extent: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> bool:
    delta = center - position
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    local_x = cos_yaw * float(delta[0]) - sin_yaw * float(delta[1])
    local_y = sin_yaw * float(delta[0]) + cos_yaw * float(delta[1])
    clamped_x = float(np.clip(local_x, -extent[0], extent[0]))
    clamped_y = float(np.clip(local_y, -extent[1], extent[1]))
    distance = math.hypot(local_x - clamped_x, local_y - clamped_y)
    return distance <= max(0.0, float(radius))


def path_distance_to_region(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
    ego_extent: list[float] | np.ndarray | None = None,
    sample_spacing: float = COLLISION_REGION_EGO_BOX_SAMPLE_SPACING,
) -> float | None:
    if len(points) == 0:
        return None

    extent = normalize_ego_extent(ego_extent)
    cumulative_distance = 0.0
    best_distance_along_path = None

    if len(points) == 1:
        if ego_box_intersects_circle(points[0], 0.0, extent, center, radius):
            return 0.0
        return None

    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-6:
            yaw = 0.0
            if index > 0:
                previous_segment = start - points[index - 1]
                if float(np.linalg.norm(previous_segment)) > 1e-6:
                    yaw = math.atan2(float(previous_segment[1]), float(previous_segment[0]))
            if ego_box_intersects_circle(start, yaw, extent, center, radius):
                if best_distance_along_path is None or cumulative_distance < best_distance_along_path:
                    best_distance_along_path = cumulative_distance
        else:
            yaw = math.atan2(float(segment[1]), float(segment[0]))
            sample_count = max(1, int(math.ceil(segment_length / max(1e-3, float(sample_spacing)))))
            for sample_index in range(sample_count + 1):
                distance_along_segment = min(
                    segment_length,
                    segment_length * float(sample_index) / float(sample_count),
                )
                position = start + segment * (distance_along_segment / segment_length)
                distance_along_path = cumulative_distance + distance_along_segment
                if ego_box_intersects_circle(position, yaw, extent, center, radius):
                    if best_distance_along_path is None or distance_along_path < best_distance_along_path:
                        best_distance_along_path = distance_along_path
                    break
        cumulative_distance += segment_length

    return best_distance_along_path


def straight_waypoint_rollout(
    waypoints: list[Any],
    max_lateral_spread: float = STRAIGHT_WAYPOINT_MAX_LATERAL_SPREAD,
    max_heading_change_deg: float = STRAIGHT_WAYPOINT_MAX_HEADING_CHANGE_DEG,
) -> bool:
    points = path_points_from_waypoints(waypoints)
    if len(points) < 3:
        return False

    path_points = points[1:]
    lateral_spread = float(path_points[:, 1].max() - path_points[:, 1].min())
    if lateral_spread > max_lateral_spread:
        return False

    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    valid_segments = segment_vectors[segment_lengths > 1e-3]
    if len(valid_segments) < 2:
        return True

    headings = np.unwrap(np.arctan2(valid_segments[:, 1], valid_segments[:, 0]))
    heading_change = math.degrees(float(headings.max() - headings.min()))
    return heading_change <= max_heading_change_deg


def velocity_rollout_distance(
    current_speed: float,
    target_speed: float,
    horizon_seconds: float,
    max_acceleration: float,
    max_deceleration: float,
    rollout_dt: float,
) -> float:
    speed = max(0.0, float(current_speed))
    target_speed = max(0.0, float(target_speed))
    remaining_time = max(0.0, float(horizon_seconds))
    distance = 0.0
    dt = max(1e-3, float(rollout_dt))

    while remaining_time > 1e-9:
        step_dt = min(dt, remaining_time)
        delta_speed = target_speed - speed
        if abs(delta_speed) < 1e-9:
            next_speed = speed
        elif delta_speed > 0.0:
            next_speed = min(target_speed, speed + max(0.0, max_acceleration) * step_dt)
        else:
            next_speed = max(target_speed, speed - max(0.0, max_deceleration) * step_dt)
        distance += 0.5 * (speed + next_speed) * step_dt
        speed = next_speed
        remaining_time -= step_dt

    return distance


def evaluate_collision_reachability(
    measurement: dict[str, Any],
    collision_measurement: dict[str, Any] | None,
    frame: int,
    collision_frame: int | None,
    pred_len: int,
    data_save_freq: int,
    sim_fps: float,
    max_acceleration: float,
    max_deceleration: float,
    rollout_dt: float,
    collision_region_radius: float,
    ego_extent: list[float] | np.ndarray | None,
    max_checked_frames_before_event: int = MAX_CHECKED_FRAMES_BEFORE_EVENT,
) -> tuple[bool, dict[str, Any]]:
    current_speed = speed_mps(measurement, "speed")
    target_speed = speed_mps(measurement, "target_speed")
    waypoints = measurement.get("route", [])[:pred_len]
    straight_rollout = straight_waypoint_rollout(waypoints)
    detail: dict[str, Any] = {
        "current_speed": round(current_speed, 4),
        "target_speed": round(target_speed, 4),
        "straight_waypoint_rollout": straight_rollout,
        "collision_region_overlap_test": "ego_box_circle",
        "ego_extent": normalize_ego_extent(ego_extent).round(4).tolist(),
        "ego_box_sample_spacing": COLLISION_REGION_EGO_BOX_SAMPLE_SPACING,
        "collision_case_outside_checked_window": False,
        "collision_case_unsafe": False,
        "collision_distance": None,
        "time_to_collision": None,
        "intersects_future_collision_region": False,
        "reachable_before_collision": False,
        "distance_to_collision_region_along_rollout": None,
        "reachable_distance_before_collision": None,
    }

    if collision_frame is None or collision_measurement is None or frame >= collision_frame:
        return False, detail

    if collision_frame - frame > max_checked_frames_before_event:
        detail["collision_case_outside_checked_window"] = True
        return False, detail

    collision_point = collision_point_in_ego(measurement, collision_measurement)
    if collision_point is None:
        return False, detail

    collision_distance = float(np.linalg.norm(collision_point))
    time_to_collision = (collision_frame - frame) * max(1, int(data_save_freq)) / max(sim_fps, 1e-6)
    detail["collision_distance"] = round(collision_distance, 4)
    detail["time_to_collision"] = round(time_to_collision, 4)

    points = path_points_from_waypoints(waypoints)
    distance_to_region = path_distance_to_region(
        points,
        collision_point,
        max(0.0, collision_region_radius),
        ego_extent,
    )
    if distance_to_region is None:
        return False, detail
    
    print(frame)
    reachable_distance = velocity_rollout_distance(
        current_speed,
        target_speed,
        time_to_collision,
        max_acceleration,
        max_deceleration,
        rollout_dt,
    )
    detail["intersects_future_collision_region"] = True
    detail["distance_to_collision_region_along_rollout"] = round(distance_to_region, 4)
    detail["reachable_distance_before_collision"] = round(reachable_distance, 4)
    detail["reachable_before_collision"] = reachable_distance >= distance_to_region
    detail["collision_case_unsafe"] = bool(detail["reachable_before_collision"])
    return bool(detail["reachable_before_collision"]), detail


def evaluate_collision_events_reachability(
    measurement: dict[str, Any],
    collision_events: list[dict[str, Any]],
    frame_to_measurement: dict[int, dict[str, Any]],
    frame: int,
    pred_len: int,
    data_save_freq: int,
    sim_fps: float,
    max_acceleration: float,
    max_deceleration: float,
    rollout_dt: float,
    default_collision_region_radius: float,
    collision_region_radii: dict[int, float],
    default_ego_extent: list[float] | np.ndarray | None,
    ego_extents: dict[int, list[float]],
) -> tuple[bool, dict[str, Any]]:
    checked_count = 0
    best_safe_detail = None
    best_future_frame = None
    resolved_event_count = sum(1 for event in collision_events if event.get("frame") is not None)

    for event_index, event in enumerate(collision_events):
        collision_frame = event.get("frame")
        if collision_frame is None:
            continue
        collision_frame = int(collision_frame)
        if frame >= collision_frame:
            continue
        
        checked_count += 1
        collision_measurement = frame_to_measurement.get(collision_frame)
        collision_region_radius = collision_region_radii.get(collision_frame, default_collision_region_radius)
        ego_extent = ego_extents.get(collision_frame, default_ego_extent)
        unsafe, detail = evaluate_collision_reachability(
            measurement,
            collision_measurement,
            frame,
            collision_frame,
            pred_len,
            data_save_freq,
            sim_fps,
            max_acceleration,
            max_deceleration,
            rollout_dt,
            collision_region_radius,
            ego_extent,
        )
        detail["collision_event_index"] = event_index
        detail["collision_event_count"] = resolved_event_count
        detail["checked_future_collision_event_count"] = checked_count
        detail["collision_event_source"] = event.get("source")
        detail["collision_event_location_distance"] = event.get("location_distance")
        detail["collision_event_location_index"] = event.get("location_index")

        if unsafe:
            return True, detail

        if best_safe_detail is None or collision_frame < best_future_frame:
            best_safe_detail = detail
            best_future_frame = collision_frame

    if best_safe_detail is not None:
        best_safe_detail["checked_future_collision_event_count"] = checked_count
        return False, best_safe_detail

    _, detail = evaluate_collision_reachability(
        measurement,
        None,
        frame,
        None,
        pred_len,
        data_save_freq,
        sim_fps,
        max_acceleration,
        max_deceleration,
        rollout_dt,
        default_collision_region_radius,
        default_ego_extent,
    )
    detail["collision_event_index"] = None
    detail["collision_event_count"] = resolved_event_count
    detail["checked_future_collision_event_count"] = checked_count
    detail["collision_event_source"] = None
    detail["collision_event_location_distance"] = None
    detail["collision_event_location_index"] = None
    return False, detail


def candidate_from_measurement(
    measurement: dict[str, Any],
    pred_len: int,
    unsafe: bool,
    safety_detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    waypoints = measurement.get("route", [])[:pred_len]
    target_speed = measurement.get("target_speed", 0.0)
    candidate = {
        "variant": "expert",
        "waypoints": waypoints,
        "target_speed": round(float(target_speed), 4),
        "expert_target_speed": round(float(measurement.get("expert_target_speed", target_speed)), 4),
        "will_collide": 0 if unsafe else 1,
    }
    if safety_detail is not None:
        candidate["safety_label_detail"] = safety_detail
    return candidate


def select_safe_frame_keys(
    safe_frame_keys: list[str],
    steering_safe_frame_keys: list[str],
    prefer_steering_safe: bool,
    unsafe_count: int,
    route_had_collision: bool,
    safe_samples_per_unsafe: float,
    max_safe_per_non_collision_route: int,
    rng: random.Random,
) -> tuple[set[str], str, int]:
    if route_had_collision and unsafe_count == 0:
        return set(), "none_collision_route_without_unsafe", 0
    if route_had_collision:
        safe_limit = int(round(unsafe_count * safe_samples_per_unsafe))
    else:
        safe_limit = max_safe_per_non_collision_route

    safe_limit = max(0, min(safe_limit, len(safe_frame_keys)))
    if safe_limit == 0:
        return set(), "none_safe_limit_zero", safe_limit

    if prefer_steering_safe and steering_safe_frame_keys:
        selected_count = min(safe_limit, len(steering_safe_frame_keys))
        return set(rng.sample(steering_safe_frame_keys, selected_count)), "consecutive_nonzero_steer", safe_limit

    selected_source = "random_safe_fallback_no_steering_safe" if prefer_steering_safe else "random_safe"
    return set(rng.sample(safe_frame_keys, safe_limit)), selected_source, safe_limit


def image_point(x: float, y: float, pixels_per_meter: float, min_y: float, max_x: float) -> tuple[int, int]:
    col = int(round((y - min_y) * pixels_per_meter))
    row = int(round((max_x - x) * pixels_per_meter))
    return col, row


def draw_oriented_box(
    draw: Any,
    pose: tuple[float, float, float],
    extent: list[float],
    color: tuple[int, int, int],
    pixels_per_meter: float,
    min_y: float,
    max_x: float,
    width: int = 2,
) -> None:
    x, y, yaw = pose
    extent_x = float(extent[0]) if len(extent) > 0 else float(DEFAULT_EGO_EXTENT[0])
    extent_y = float(extent[1]) if len(extent) > 1 else float(DEFAULT_EGO_EXTENT[1])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    corners = []
    for local_x, local_y in ((extent_x, extent_y), (extent_x, -extent_y), (-extent_x, -extent_y), (-extent_x, extent_y)):
        world_x = x + cos_yaw * local_x - sin_yaw * local_y
        world_y = y + sin_yaw * local_x + cos_yaw * local_y
        corners.append(image_point(world_x, world_y, pixels_per_meter, min_y, max_x))

    draw.line(corners + [corners[0]], fill=color, width=width)
    nose = image_point(x + cos_yaw * extent_x, y + sin_yaw * extent_x, pixels_per_meter, min_y, max_x)
    center = image_point(x, y, pixels_per_meter, min_y, max_x)
    draw.line((center, nose), fill=color, width=max(1, width))


def actor_track_id(actor: dict[str, Any], fallback_index: int) -> str:
    actor_id = actor.get("id")
    if actor_id is None:
        return f"{actor.get('class', 'actor')}:{fallback_index}"
    return f"{actor.get('class', 'actor')}:{actor_id}"


def load_actor_tracks_from_dataset(
    route_dir: Path,
    frame: int,
    horizon: int,
    origin_matrix: np.ndarray,
) -> dict[str, dict[str, Any]]:
    tracks: dict[str, dict[str, Any]] = {}
    for offset in range(horizon + 1):
        boxes_path = route_dir / "boxes" / f"{frame + offset:04}.json.gz"
        if not boxes_path.is_file():
            break

        for fallback_index, actor in enumerate(load_json_gz(boxes_path)):
            actor_class = str(actor.get("class", "actor"))
            if actor_class not in VISUALIZED_CLASSES:
                continue

            pose = actor_pose_in_origin(origin_matrix, actor)
            if pose is None:
                continue

            track_id = actor_track_id(actor, fallback_index)
            if track_id not in tracks:
                tracks[track_id] = {
                    "class": actor_class,
                    "extent": actor.get("extent", DEFAULT_EGO_EXTENT.tolist()),
                    "poses": [],
                    "frames": [],
                }
            tracks[track_id]["poses"].append(pose)
            tracks[track_id]["frames"].append(frame + offset)
            tracks[track_id]["extent"] = actor.get("extent", tracks[track_id]["extent"])
    return tracks


def unsafe_candidate(candidate: dict[str, Any], unsafe_label: int = 0) -> bool:
    return int(candidate.get("will_collide", 1)) == int(unsafe_label)


def unsafe_label_from_labels(labels: dict[str, Any]) -> int:
    label_map = labels.get("label_map", {})
    return int(label_map.get("unsafe_sim_collision", 0))


def find_rgb_image_path(route_dir: Path, frame_key: str) -> Path | None:
    frame = int(frame_key)
    frame_names = (frame_key, f"{frame:04}", f"{frame:05}")
    rgb_dirs = (
        route_dir / "rgb",
        route_dir / "rgb_front",
        route_dir / "sensor_data" / "rgb",
        route_dir / "sensor_data" / "rgb_front",
    )
    suffixes = (".jpg", ".png", ".jpeg")
    for rgb_dir in rgb_dirs:
        for name in frame_names:
            for suffix in suffixes:
                path = rgb_dir / f"{name}{suffix}"
                if path.is_file():
                    return path
    return None


def append_rgb_panel(trajectory_image: Any, route_dir: Path, frame_key: str) -> Any:
    rgb_path = find_rgb_image_path(route_dir, frame_key)
    if rgb_path is None:
        return trajectory_image

    from PIL import Image, ImageDraw

    rgb_image = Image.open(rgb_path).convert("RGB")
    target_height = trajectory_image.height
    target_width = max(1, int(round(rgb_image.width * target_height / max(rgb_image.height, 1))))
    rgb_image = rgb_image.resize((target_width, target_height), Image.Resampling.BILINEAR)

    combined = Image.new("RGB", (target_width + trajectory_image.width, target_height), (248, 248, 248))
    combined.paste(rgb_image, (0, 0))
    combined.paste(trajectory_image, (target_width, 0))
    draw = ImageDraw.Draw(combined)
    draw.rectangle((0, 0, target_width - 1, target_height - 1), outline=(0, 0, 0), width=2)
    draw.line(((target_width, 0), (target_width, target_height)), fill=(0, 0, 0), width=2)
    draw.text((8, target_height - 18), f"RGB {frame_key}", fill=(255, 255, 255))
    return combined


def draw_candidate_waypoints(
    draw: Any,
    candidate: dict[str, Any],
    pixels_per_meter: float,
    min_y: float,
    max_x: float,
) -> None:
    waypoints = candidate.get("waypoints", [])
    waypoint_points = []
    for waypoint in waypoints:
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 2:
            continue
        try:
            waypoint_points.append((float(waypoint[0]), float(waypoint[1])))
        except (TypeError, ValueError):
            continue

    image_points = [
        image_point(waypoint[0], waypoint[1], pixels_per_meter, min_y, max_x)
        for waypoint in waypoint_points
    ]
    if len(image_points) > 1:
        draw.line(image_points, fill=(35, 80, 225), width=3)
    for index, point in enumerate(image_points):
        radius = 5 if index == 0 else 3
        fill = (255, 235, 60) if index == 0 else (35, 80, 225)
        outline = (20, 20, 20) if index == 0 else (255, 255, 255)
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=fill,
            outline=outline,
            width=2 if index == 0 else 1,
        )


def candidate_status_text(candidate: dict[str, Any]) -> tuple[str, str]:
    target_speed = float(candidate.get("target_speed", 0.0) or 0.0)
    target_speed_text = f"target_speed={target_speed:.2f} m/s ({target_speed * 3.6:.1f} km/h)"

    waypoints = candidate.get("waypoints", [])
    first_waypoint = None
    if isinstance(waypoints, list):
        for waypoint in waypoints:
            if isinstance(waypoint, (list, tuple)) and len(waypoint) >= 2:
                try:
                    first_waypoint = (float(waypoint[0]), float(waypoint[1]))
                except (TypeError, ValueError):
                    first_waypoint = None
                break

    if first_waypoint is None:
        waypoint_text = "current_waypoint=None"
    else:
        waypoint_text = f"current_waypoint=({first_waypoint[0]:.2f}, {first_waypoint[1]:.2f})"
    return target_speed_text, waypoint_text


def render_dataset_trajectory_visualization(
    route_dir: Path,
    frame_key: str,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    horizon: int,
    pixels_per_meter: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    include_rgb: bool,
    unsafe_label: int,
) -> Path | None:
    measurement_path = route_dir / "measurements" / f"{int(frame_key):04}.json.gz"
    if not measurement_path.is_file():
        return None

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Trajectory visualization requires Pillow. Install it with `pip install pillow`.") from exc

    measurement = load_json_gz(measurement_path)
    origin_matrix_raw = measurement.get("ego_matrix")
    if origin_matrix_raw is None:
        return None

    origin_matrix = np.array(origin_matrix_raw, dtype=np.float64)
    tracks = load_actor_tracks_from_dataset(route_dir, int(frame_key), horizon, origin_matrix)
    if not tracks:
        return None

    width = max(1, int(round((max_y - min_y) * pixels_per_meter)))
    height = max(1, int(round((max_x - min_x) * pixels_per_meter)))
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)

    zero_x = image_point(0.0, min_y, pixels_per_meter, min_y, max_x)[1]
    zero_y = image_point(min_x, 0.0, pixels_per_meter, min_y, max_x)[0]
    draw.line(((0, zero_x), (width, zero_x)), fill=(220, 220, 220), width=1)
    draw.line(((zero_y, 0), (zero_y, height)), fill=(220, 220, 220), width=1)

    for track in tracks.values():
        poses = track["poses"]
        actor_class = track["class"]
        color = TRAJECTORY_CLASS_COLORS.get(actor_class, DEFAULT_TRAJECTORY_COLOR)
        points = [image_point(pose[0], pose[1], pixels_per_meter, min_y, max_x) for pose in poses]
        if len(points) > 1:
            draw.line(points, fill=color, width=4 if actor_class == "ego_car" else 2)
        for point_index, point in enumerate(points):
            radius = 4 if actor_class == "ego_car" else 3
            fill = color if point_index == len(points) - 1 else tuple(max(0, int(channel * 0.65)) for channel in color)
            draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=fill)
        if poses:
            draw_oriented_box(
                draw,
                poses[-1],
                track.get("extent", DEFAULT_EGO_EXTENT.tolist()),
                color,
                pixels_per_meter,
                min_y,
                max_x,
                width=3 if actor_class == "ego_car" else 2,
            )

    if candidates:
        draw_candidate_waypoints(draw, candidates[0], pixels_per_meter, min_y, max_x)

    label = "unsafe" if any(unsafe_candidate(candidate, unsafe_label) for candidate in candidates) else "safe"
    title_color = (210, 30, 30) if label == "unsafe" else (20, 135, 55)
    draw.text((8, 8), f"{frame_key} {label} recorded dataset trajectories", fill=title_color)
    draw.text((8, 24), f"horizon={horizon} saved frames, actors={len(tracks)}", fill=(35, 35, 35))
    if candidates:
        target_speed_text, waypoint_text = candidate_status_text(candidates[0])
        draw.text((8, 40), target_speed_text, fill=(20, 20, 20))
        draw.text((8, 56), waypoint_text, fill=(20, 20, 20))
    if include_rgb:
        image = append_rgb_panel(image, route_dir, frame_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{frame_key}_{label}_dataset_trajectories.png"
    image.save(output_path)
    return output_path


def maybe_render_route_trajectory_visualizations(
    route_dir: Path,
    frames: dict[str, list[dict[str, Any]]],
    output_dir: Path | None,
    horizon: int,
    max_frames_per_route: int | None,
    unsafe_only: bool,
    include_rgb: bool,
    unsafe_label: int,
    pixels_per_meter: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> int:
    if output_dir is None or not frames:
        return 0

    rendered = 0
    route_output_dir = output_dir / route_dir.parent.name / route_dir.name
    ordered_frames = [
        frame_key
        for frame_key in sorted(frames)
        if not unsafe_only or any(unsafe_candidate(candidate, unsafe_label) for candidate in frames[frame_key])
    ]
    if max_frames_per_route is not None:
        ordered_frames = ordered_frames[:max_frames_per_route]

    for frame_key in ordered_frames:
        output_path = render_dataset_trajectory_visualization(
            route_dir,
            frame_key,
            frames[frame_key],
            route_output_dir,
            horizon,
            pixels_per_meter,
            min_x,
            max_x,
            min_y,
            max_y,
            include_rgb,
            unsafe_label,
        )
        if output_path is not None:
            rendered += 1
    return rendered


def delete_route_trajectory_visualizations(output_dir: Path | None, route_dir: Path) -> None:
    if output_dir is None:
        return
    route_output_dir = output_dir / route_dir.parent.name / route_dir.name
    if route_output_dir.is_dir():
        shutil.rmtree(route_output_dir)


def label_route(
    route_dir: Path,
    pred_len: int,
    all_frames: bool,
    overwrite: bool,
    infer_collision_frame_from_location: bool,
    max_collision_location_distance: float,
    data_save_freq: int,
    sim_fps: float,
    max_acceleration: float,
    max_deceleration: float,
    rollout_dt: float,
    collision_region_radius: float | None,
    safe_samples_per_unsafe: float,
    max_safe_per_non_collision_route: int,
    safe_consecutive_nonzero_steer_frames: int,
    nonzero_steer_threshold: float,
    seed: int,
    trajectory_visualization_output_dir: Path | None,
    trajectory_visualization_horizon: int,
    trajectory_visualization_max_frames_per_route: int | None,
    trajectory_visualization_unsafe_only: bool,
    trajectory_visualization_include_rgb: bool,
    trajectory_visualization_pixels_per_meter: float,
    trajectory_visualization_min_x: float,
    trajectory_visualization_max_x: float,
    trajectory_visualization_min_y: float,
    trajectory_visualization_max_y: float,
) -> tuple[int, int, int, bool, bool, int]:
    output_path = route_dir / "plan_safety_labels.json.gz"
    if overwrite:
        delete_route_trajectory_visualizations(trajectory_visualization_output_dir, route_dir)
    if output_path.exists() and not overwrite:
        rendered_visualizations = 0
        if trajectory_visualization_output_dir is not None:
            existing_labels = load_json_gz(output_path)
            unsafe_label = unsafe_label_from_labels(existing_labels)
            rendered_visualizations = maybe_render_route_trajectory_visualizations(
                route_dir,
                existing_labels.get("frames", {}),
                trajectory_visualization_output_dir,
                trajectory_visualization_horizon,
                trajectory_visualization_max_frames_per_route,
                trajectory_visualization_unsafe_only,
                trajectory_visualization_include_rgb,
                unsafe_label,
                trajectory_visualization_pixels_per_meter,
                trajectory_visualization_min_x,
                trajectory_visualization_max_x,
                trajectory_visualization_min_y,
                trajectory_visualization_max_y,
            )
        return 0, 0, 0, False, True, rendered_visualizations

    paths = measurement_paths(route_dir)
    if not paths:
        return 0, 0, 0, False, False, 0

    measurements = [load_json_gz(path) for path in paths]
    results = load_results(route_dir)
    collision = has_collision_result(results)
    collision_events = collision_data_events(
        results,
        paths,
        measurements,
        infer_collision_frame_from_location,
        max_collision_location_distance,
        data_save_freq,
    )
    resolved_collision_events = [event for event in collision_events if event.get("frame") is not None]
    primary_collision_event = resolved_collision_events[0] if resolved_collision_events else {}
    collision_frame = primary_collision_event.get("frame")
    collision_frame_source = primary_collision_event.get("source")
    collision_location_distance = primary_collision_event.get("location_distance")
    collision_frames = [int(event["frame"]) for event in resolved_collision_events]
    last_collision_frame = max(collision_frames) if collision_frames else None
    unresolved_collision = collision and not resolved_collision_events
    frame_to_measurement = measurement_frame_map(paths, measurements)
    resolved_collision_region_radius, collision_region_radius_source, ego_extent = resolve_collision_region_radius(
        route_dir,
        paths,
        collision_frame,
        collision_region_radius,
    )
    collision_region_radii = {}
    collision_region_radius_sources = {}
    ego_extents_by_frame = {}
    for event in resolved_collision_events:
        event_frame = int(event["frame"])
        event_radius, event_radius_source, event_ego_extent = resolve_collision_region_radius(
            route_dir,
            paths,
            event_frame,
            collision_region_radius,
        )
        collision_region_radii[event_frame] = event_radius
        collision_region_radius_sources[event_frame] = event_radius_source
        ego_extents_by_frame[event_frame] = event_ego_extent
    case_label = "collision" if collision else "success"
    safe_consecutive_nonzero_steer_frames = max(0, int(safe_consecutive_nonzero_steer_frames))
    nonzero_steer_threshold = max(0.0, float(nonzero_steer_threshold))
    nonzero_steer_run_lengths = consecutive_nonzero_steer_run_lengths(
        paths,
        measurements,
        nonzero_steer_threshold,
    )
    candidate_frames = []
    safe_frame_keys = []
    steering_safe_frame_keys = []
    unsafe_count = 0
    for path, measurement in zip(paths, measurements):
        frame_key = path.stem.split(".")[0]
        frame = int(frame_key)
        if last_collision_frame is not None and frame >= last_collision_frame:
            continue
        if not focus_frame(measurement, all_frames):
            continue
        collision_unsafe, collision_detail = evaluate_collision_events_reachability(
            measurement,
            resolved_collision_events,
            frame_to_measurement,
            frame,
            pred_len,
            data_save_freq,
            sim_fps,
            max_acceleration,
            max_deceleration,
            rollout_dt,
            resolved_collision_region_radius,
            collision_region_radii,
            ego_extent,
            ego_extents_by_frame,
        )
        unsafe = collision_unsafe
        safety_detail = collision_detail
        steer = steer_value(measurement)
        nonzero_steer_run_length = nonzero_steer_run_lengths.get(frame_key, 0)
        safe_case_from_nonzero_steer_run = (
            safe_consecutive_nonzero_steer_frames > 0
            and nonzero_steer_run_length >= safe_consecutive_nonzero_steer_frames
        )
        safety_detail["steer"] = round(steer, 4)
        safety_detail["nonzero_steer_threshold"] = nonzero_steer_threshold
        safety_detail["nonzero_steer_run_length"] = nonzero_steer_run_length
        safety_detail["safe_case_from_consecutive_nonzero_steer"] = bool(safe_case_from_nonzero_steer_run)
        candidate = candidate_from_measurement(measurement, pred_len, unsafe, safety_detail)
        candidate_frames.append((frame_key, candidate, unsafe))
        unsafe_count += int(unsafe)
        if not unsafe:
            safe_frame_keys.append(frame_key)
            if safe_case_from_nonzero_steer_run:
                steering_safe_frame_keys.append(frame_key)

    route_seed = seed + sum(ord(char) for char in str(route_dir))
    selected_safe_frame_keys, selected_safe_source, safe_limit = select_safe_frame_keys(
        safe_frame_keys,
        steering_safe_frame_keys,
        safe_consecutive_nonzero_steer_frames > 0,
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
        visualization_frames = {
            frame_key: [candidate]
            for frame_key, candidate, _ in candidate_frames
        }
        rendered_visualizations = maybe_render_route_trajectory_visualizations(
            route_dir,
            visualization_frames,
            trajectory_visualization_output_dir,
            trajectory_visualization_horizon,
            trajectory_visualization_max_frames_per_route,
            trajectory_visualization_unsafe_only,
            trajectory_visualization_include_rgb,
            0,
            trajectory_visualization_pixels_per_meter,
            trajectory_visualization_min_x,
            trajectory_visualization_max_x,
            trajectory_visualization_min_y,
            trajectory_visualization_max_y,
        )
        return 0, 0, 0, unresolved_collision, False, rendered_visualizations

    dump_json_gz(output_path, {
        "label_map": {"unsafe_sim_collision": 0, "safe": 1},
        "source": "carla_simulator",
        "case_label": case_label,
        "route_had_collision": collision,
        "collision_data_frame": collision_frame,
        "collision_frame_source": collision_frame_source,
        "collision_location_distance": collision_location_distance,
        "collision_data_frames": collision_frames,
        "collision_events": collision_events,
        "collision_event_count": len(resolved_collision_events),
        "collision_case_unsafe_criterion": (
            "within_10_frames_before_each_collision_event_future_collision_region_ego_box_overlap_and_velocity_rollout_reachability"
        ),
        "max_checked_frames_before_event": MAX_CHECKED_FRAMES_BEFORE_EVENT,
        "straight_waypoint_max_lateral_spread": STRAIGHT_WAYPOINT_MAX_LATERAL_SPREAD,
        "straight_waypoint_max_heading_change_deg": STRAIGHT_WAYPOINT_MAX_HEADING_CHANGE_DEG,
        "sim_fps": sim_fps,
        "data_save_freq": data_save_freq,
        "max_acceleration": max_acceleration,
        "max_deceleration": max_deceleration,
        "rollout_dt": rollout_dt,
        "collision_region_radius": resolved_collision_region_radius,
        "collision_region_radius_source": collision_region_radius_source,
        "collision_region_radius_by_frame": collision_region_radii,
        "collision_region_radius_source_by_frame": collision_region_radius_sources,
        "ego_extent": ego_extent,
        "ego_extent_by_frame": ego_extents_by_frame,
        "ego_box_sample_spacing": COLLISION_REGION_EGO_BOX_SAMPLE_SPACING,
        "safe_samples_per_unsafe": safe_samples_per_unsafe,
        "max_safe_per_non_collision_route": max_safe_per_non_collision_route,
        "safe_consecutive_nonzero_steer_frames": safe_consecutive_nonzero_steer_frames,
        "nonzero_steer_threshold": nonzero_steer_threshold,
        "safe_candidate_count": len(safe_frame_keys),
        "steering_safe_candidate_count": len(steering_safe_frame_keys),
        "selected_safe_candidate_source": selected_safe_source,
        "selected_safe_candidate_limit": safe_limit,
        "pred_len": pred_len,
        "frames": frames,
    })
    rendered_visualizations = maybe_render_route_trajectory_visualizations(
        route_dir,
        frames,
        trajectory_visualization_output_dir,
        trajectory_visualization_horizon,
        trajectory_visualization_max_frames_per_route,
        trajectory_visualization_unsafe_only,
        trajectory_visualization_include_rgb,
        0,
        trajectory_visualization_pixels_per_meter,
        trajectory_visualization_min_x,
        trajectory_visualization_max_x,
        trajectory_visualization_min_y,
        trajectory_visualization_max_y,
    )
    return len(frames), safe_count, unsafe_count, unresolved_collision, False, rendered_visualizations


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simulator-derived plan safety labels.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pred-len", type=int, default=8)
    parser.add_argument("--data-save-freq", type=int, default=10)
    parser.add_argument(
        "--sim-fps",
        type=float,
        default=DEFAULT_SIM_FPS,
        help="Simulator ticks per second used to convert saved frame offsets into seconds.",
    )
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=DEFAULT_MAX_ACCELERATION,
        help="Maximum ego acceleration in m/s^2 for the reachability velocity rollout.",
    )
    parser.add_argument(
        "--max-deceleration",
        type=float,
        default=DEFAULT_MAX_DECELERATION,
        help="Maximum ego deceleration magnitude in m/s^2 for the reachability velocity rollout.",
    )
    parser.add_argument(
        "--rollout-dt",
        type=float,
        default=0.05,
        help="Time step in seconds for the velocity rollout reachability check.",
    )
    parser.add_argument(
        "--collision-region-radius",
        type=float,
        default=2.0,
        help=(
            "Optional radius in meters around the future collision pose treated as the collision region. "
            "Defaults to the ego_car extent from dataset boxes, falling back to the built-in ego extent."
        ),
    )
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
    parser.add_argument(
        "--safe-consecutive-nonzero-steer-frames",
        type=int,
        default=0,
        help=(
            "When greater than 0, only sample safe labels from frames that belong to a run of at least this "
            "many consecutive saved measurements with abs(steer) greater than --nonzero-steer-threshold."
        ),
    )
    parser.add_argument(
        "--nonzero-steer-threshold",
        type=float,
        default=DEFAULT_NONZERO_STEER_THRESHOLD,
        help="Absolute steering threshold used by --safe-consecutive-nonzero-steer-frames.",
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
    parser.add_argument(
        "--visualize-trajectories",
        action="store_true",
        help="Render recorded dataset trajectories for labeled frames while generating labels.",
    )
    parser.add_argument(
        "--trajectory-visualization-output-dir",
        type=str,
        help="Where to write trajectory PNGs. Defaults to DATA_ROOT/sim_plan_safety_trajectory_visualizations.",
    )
    parser.add_argument(
        "--trajectory-visualization-horizon",
        type=int,
        default=10,
        help="Number of future saved dataset frames to draw. Uses recorded boxes, not prediction.",
    )
    parser.add_argument(
        "--trajectory-visualization-max-frames-per-route",
        type=int,
        help="Maximum rendered labeled frames per route. Defaults to every labeled frame when visualization is enabled.",
    )
    parser.add_argument(
        "--trajectory-visualization-unsafe-only",
        action="store_true",
        help="Only render trajectory PNGs for frames labeled unsafe_sim_collision=0.",
    )
    parser.add_argument(
        "--trajectory-visualization-include-rgb",
        action="store_true",
        help="Write trajectory PNGs as RGB camera frame plus recorded BEV trajectory side-by-side.",
    )
    parser.add_argument("--trajectory-visualization-pixels-per-meter", type=float, default=5.0)
    parser.add_argument("--trajectory-visualization-min-x", type=float, default=-32.0)
    parser.add_argument("--trajectory-visualization-max-x", type=float, default=32.0)
    parser.add_argument("--trajectory-visualization-min-y", type=float, default=-32.0)
    parser.add_argument("--trajectory-visualization-max-y", type=float, default=32.0)
    args = parser.parse_args()

    route_dirs = sorted(set(iter_route_dirs(args.data_root)))
    if not route_dirs:
        raise FileNotFoundError(f"No route measurement folders found under {args.data_root}")

    deleted_existing_label_count = 0
    if args.overwrite:
        deleted_existing_label_count = delete_existing_plan_safety_labels(route_dirs)

    route_count = 0
    frame_count = 0
    safe_count = 0
    unsafe_count = 0
    unresolved_collision_count = 0
    skipped_existing_count = 0
    rendered_visualization_count = 0
    trajectory_visualization_output_dir = None
    if args.visualize_trajectories:
        if args.trajectory_visualization_output_dir is not None:
            trajectory_visualization_output_dir = Path(args.trajectory_visualization_output_dir)
        else:
            trajectory_visualization_output_dir = args.data_root / "sim_plan_safety_trajectory_visualizations"

    for route_dir in tqdm(route_dirs, desc="Generating plan-safety labels", unit="route"):
        frames, safe, unsafe, unresolved_collision, skipped_existing, rendered_visualizations = label_route(
            route_dir,
            args.pred_len,
            args.all_frames or not args.focus_frames_only,
            args.overwrite,
            not args.no_infer_collision_frame_from_location,
            args.max_collision_location_distance,
            args.data_save_freq,
            args.sim_fps,
            args.max_acceleration,
            args.max_deceleration,
            args.rollout_dt,
            args.collision_region_radius,
            args.safe_samples_per_unsafe,
            args.max_safe_per_non_collision_route,
            args.safe_consecutive_nonzero_steer_frames,
            args.nonzero_steer_threshold,
            args.seed,
            trajectory_visualization_output_dir,
            args.trajectory_visualization_horizon,
            args.trajectory_visualization_max_frames_per_route,
            args.trajectory_visualization_unsafe_only,
            args.trajectory_visualization_include_rgb,
            args.trajectory_visualization_pixels_per_meter,
            args.trajectory_visualization_min_x,
            args.trajectory_visualization_max_x,
            args.trajectory_visualization_min_y,
            args.trajectory_visualization_max_y,
        )
        if frames:
            route_count += 1
            frame_count += frames
            safe_count += safe
            unsafe_count += unsafe
        unresolved_collision_count += int(unresolved_collision)
        skipped_existing_count += int(skipped_existing)
        rendered_visualization_count += rendered_visualizations

    print(f"Labeled routes: {route_count}")
    print(f"Labeled frames: {frame_count}")
    print(f"safe=1: {safe_count}")
    print(f"unsafe_sim_collision=0: {unsafe_count}")
    if args.overwrite:
        print(f"Deleted existing label files before overwrite: {deleted_existing_label_count}")
    print(f"Skipped existing label files: {skipped_existing_count}")
    print(f"collision routes without usable collision frame: {unresolved_collision_count}")
    if args.visualize_trajectories:
        print(f"Recorded dataset trajectory visualizations: {rendered_visualization_count}")
        print(f"Trajectory visualization output: {trajectory_visualization_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
