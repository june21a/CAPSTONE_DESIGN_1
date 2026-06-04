#!/usr/bin/env python3
"""Generate plan-safety labels from CARLA Garage training data.

The output is one ``plan_safety_labels.json.gz`` file per route folder.
Each frame contains candidate plans with:

  will_collide = 0  # unsafe / collision predicted by offline OBB overlap
  will_collide = 1  # safe

This is an offline approximation. For guaranteed labels, replay candidate plans in
CARLA and use simulator collision events.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the script usable without optional tqdm.
    tqdm = None


VEHICLE_CLASS_ID = 0
WALKER_CLASS_ID = 1
BRAKE_RATIO = 1.1
CLIP_THROTTLE = 1.0
LATERAL_K_P = 3.118357247806046
LATERAL_K_D = 1.3782508892109167
LATERAL_K_I = 0.6406067986034124
LATERAL_SPEED_SCALE = 0.9755321901954155
LATERAL_SPEED_OFFSET = 1.9152884533402488
LATERAL_N = 6
LONGITUDINAL_PARAMS = np.array(
    [
        1.1990342347353184,
        -0.8057602384167799,
        1.710818710950062,
        0.921890257450335,
        1.556497522998393,
        -0.7013479734904027,
        1.031266635497984,
    ],
    dtype=np.float64,
)
LONGITUDINAL_MAX_ACCELERATION = 1.89
FRONT_WHEEL_BASE = -0.090769015
REAR_WHEEL_BASE = 1.4178275
STEERING_GAIN = 0.36848336
THROTTLE_THRESHOLD_DURING_FORECASTING = 0.3
THROTTLE_VALUES = np.array(
    [
        9.63873001e-01,
        4.37535692e-04,
        -3.80192912e-01,
        1.74950069e00,
        9.16787414e-02,
        -7.05461530e-02,
        -1.05996152e-03,
        6.71079346e-04,
    ],
    dtype=np.float64,
)
BRAKE_VALUES = np.array(
    [
        9.31711370e-03,
        8.20967431e-02,
        -2.83832427e-03,
        5.06587474e-05,
        -4.90357228e-07,
        2.44419284e-09,
        -4.91381935e-12,
    ],
    dtype=np.float64,
)
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


def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def dump_json_gz(path: Path, data: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_matrix(matrix: np.ndarray) -> float:
    return math.atan2(matrix[1, 0], matrix[0, 0])


def transform_box_to_ego(origin_matrix: np.ndarray, actor_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    relative_matrix = np.linalg.inv(origin_matrix) @ actor_matrix
    pos = relative_matrix[:2, 3]
    yaw = normalize_angle(yaw_from_matrix(actor_matrix) - yaw_from_matrix(origin_matrix))
    return pos, yaw


def scenario_matches(name: str, keywords: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def is_focus_frame(scenario: str, measurement: dict[str, Any], use_focus_filter: bool) -> bool:
    if not use_focus_filter:
        return True

    command = as_int(measurement.get("command"))
    next_command = as_int(measurement.get("next_command"))
    return (
        bool(measurement.get("junction"))
        or command in LANE_CHANGE_COMMANDS
        or next_command in LANE_CHANGE_COMMANDS
        or scenario_matches(scenario, CROSSROAD_SCENARIO_KEYWORDS)
        or scenario_matches(scenario, LANE_INTERACTION_SCENARIO_KEYWORDS)
    )


def get_gt_waypoints(measurements: list[dict[str, Any]]) -> np.ndarray:
    origin_matrix = np.array(measurements[0]["ego_matrix"], dtype=np.float64)
    origin_translation = origin_matrix[:3, 3:4]
    origin_rotation = origin_matrix[:3, :3]

    waypoints = []
    for measurement in measurements[1:]:
        waypoint = np.array(measurement["ego_matrix"], dtype=np.float64)[:3, 3:4]
        waypoint_ego = origin_rotation.T @ (waypoint - origin_translation)
        waypoints.append(waypoint_ego[:2, 0])
    return np.asarray(waypoints, dtype=np.float32)


def get_throttle_offline(brake: bool, target_speed: float, speed: float) -> tuple[float, bool]:
    """Match the model controller's longitudinal control without importing CARLA-bound modules."""
    if target_speed < 1e-5 or brake:
        return 0.0, True
    if target_speed < 1.0 / 3.6:
        target_speed = 1.0 / 3.6

    speed_kph = speed * 3.6
    target_speed_kph = target_speed * 3.6
    speed_error = target_speed_kph - speed_kph

    if speed_error > LONGITUDINAL_MAX_ACCELERATION:
        return 1.0, False

    if speed_kph / target_speed_kph > LONGITUDINAL_PARAMS[-1] or brake:
        return 0.0, True

    speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
    speed_norm = speed_kph / 100.0
    features = np.array(
        [
            speed_norm,
            speed_norm**2,
            100.0 * speed_error_cl,
            speed_error_cl**2,
            speed_norm * speed_error_cl,
            speed_norm**2 * speed_error_cl,
        ],
        dtype=np.float64,
    )
    throttle = float(np.clip(features @ LONGITUDINAL_PARAMS[:-1], 0.0, 1.0))
    return throttle, False


def lateral_pid_step(
    route_points: np.ndarray,
    current_speed: float,
    vehicle_position: np.ndarray,
    vehicle_heading: float,
    error_history: list[float],
) -> float:
    current_speed_kph = current_speed * 3.6
    n_lookahead = np.clip(LATERAL_SPEED_SCALE * current_speed_kph + LATERAL_SPEED_OFFSET, 24, 105) / 10
    n_lookahead = int(min(n_lookahead - 2, route_points.shape[0] - 1))
    n_lookahead = min(max(n_lookahead, 0), len(route_points) - 1)

    desired_heading_vec = route_points[n_lookahead] - vehicle_position
    yaw_path = math.atan2(float(desired_heading_vec[1]), float(desired_heading_vec[0]))
    heading_error = normalize_angle(yaw_path - vehicle_heading)
    heading_error = heading_error * 180.0 / math.pi / 90.0

    error_history.append(heading_error)
    del error_history[:-LATERAL_N]
    derivative = 0.0 if len(error_history) == 1 else error_history[-1] - error_history[-2]
    integral = float(np.mean(error_history))
    steering = LATERAL_K_P * heading_error + LATERAL_K_D * derivative + LATERAL_K_I * integral
    return round(float(np.clip(steering, -1.0, 1.0)), 3)


def bicycle_model_step(
    position: np.ndarray,
    heading: float,
    speed: float,
    steer: float,
    throttle: float,
    brake: bool,
    dt: float,
) -> tuple[np.ndarray, float, float]:
    wheel_angle = STEERING_GAIN * steer
    slip_angle = math.atan(REAR_WHEEL_BASE / (FRONT_WHEEL_BASE + REAR_WHEEL_BASE) * math.tan(wheel_angle))

    next_x = float(position[0]) + speed * math.cos(heading + slip_angle) * dt
    next_y = float(position[1]) + speed * math.sin(heading + slip_angle) * dt
    next_heading = heading + speed / REAR_WHEEL_BASE * math.sin(slip_angle) * dt

    if brake:
        speed_kph = speed * 3.6
        features = speed_kph ** np.arange(1, 8)
        next_speed = float(features @ BRAKE_VALUES) / 3.6
    else:
        throttle = float(np.clip(throttle, 0.0, 1.0))
        if throttle < THROTTLE_THRESHOLD_DURING_FORECASTING:
            next_speed = speed
        else:
            speed_kph = speed * 3.6
            features = np.array(
                [
                    speed_kph,
                    speed_kph**2,
                    throttle,
                    throttle**2,
                    speed_kph * throttle,
                    speed_kph * throttle**2,
                    speed_kph**2 * throttle,
                    speed_kph**2 * throttle**2,
                ],
                dtype=np.float64,
            )
            next_speed = float(features @ THROTTLE_VALUES) / 3.6

    return np.array([next_x, next_y], dtype=np.float32), normalize_angle(next_heading), max(0.0, next_speed)


def path_positions_for_speed(
    waypoints: np.ndarray,
    target_speed: float,
    dt: float,
    current_speed: float = 0.0,
    physics_dt: float = 1.0 / 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out controller-followed positions using the repo's tuned kinematic bicycle model."""
    if len(waypoints) == 0:
        return waypoints, np.asarray([], dtype=np.float32)

    route_points = waypoints.astype(np.float32)
    position = np.zeros((2,), dtype=np.float32)
    heading = 0.0
    speed = max(0.0, float(current_speed))
    error_history: list[float] = []
    positions = []
    yaws = []

    substeps = max(1, int(round(dt / max(physics_dt, 1e-6))))
    substep_dt = dt / substeps
    for _ in range(len(route_points)):
        for _ in range(substeps):
            brake = target_speed < 0.01 or (target_speed > 1e-6 and speed / target_speed > BRAKE_RATIO)
            steer = lateral_pid_step(route_points, speed, position, heading, error_history)
            throttle, control_brake = get_throttle_offline(brake, target_speed, speed)
            throttle = float(np.clip(throttle, 0.0, CLIP_THROTTLE))
            position, heading, speed = bicycle_model_step(
                position,
                heading,
                speed,
                steer,
                throttle,
                control_brake,
                substep_dt,
            )
        positions.append(position.copy())
        yaws.append(heading)

    return np.asarray(positions, dtype=np.float32), np.asarray(yaws, dtype=np.float32)


def yaw_for_positions(positions: np.ndarray, index: int) -> float:
    if len(positions) <= 1:
        return 0.0
    if index == 0:
        delta = positions[1] - np.zeros((2,), dtype=np.float32)
    else:
        delta = positions[index] - positions[index - 1]
    if np.linalg.norm(delta) < 1e-4:
        return 0.0
    return math.atan2(float(delta[1]), float(delta[0]))


def obb_corners(center: np.ndarray, extent: np.ndarray, yaw: float) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    local = np.array(
        [[extent[0], extent[1]], [extent[0], -extent[1]], [-extent[0], -extent[1]], [-extent[0], extent[1]]],
        dtype=np.float64,
    )
    return center[:2] + local @ rotation.T


def polygons_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    for corners in (corners_a, corners_b):
        for idx in range(4):
            edge = corners[(idx + 1) % 4] - corners[idx]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis /= norm
            proj_a = corners_a @ axis
            proj_b = corners_b @ axis
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False
    return True


def actor_box_features(origin_matrix: np.ndarray, actor: dict[str, Any]) -> list[float] | None:
    if actor.get("class") not in ("car", "walker"):
        return None

    actor_matrix = np.array(actor["matrix"], dtype=np.float64)
    pos, yaw = transform_box_to_ego(origin_matrix, actor_matrix)
    extent = actor["extent"]
    class_id = VEHICLE_CLASS_ID if actor["class"] == "car" else WALKER_CLASS_ID
    return [
        float(pos[0]),
        float(pos[1]),
        float(extent[0]),
        float(extent[1]),
        float(yaw),
        float(actor.get("speed", 0.0)),
        float(actor.get("brake", 0.0) or 0.0),
        float(class_id),
    ]


def candidate_collides(
    route_dir: Path,
    frame: int,
    origin_matrix: np.ndarray,
    ego_extent: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray | None = None,
) -> tuple[bool, list[list[float]], list[list[float]]]:
    other_boxes_for_context: list[list[float]] = []
    other_velocities_for_context: list[list[float]] = []

    for index, position in enumerate(positions):
        future_frame = frame + index + 1
        boxes_path = route_dir / "boxes" / f"{future_frame:04}.json.gz"
        if not boxes_path.is_file():
            break

        ego_yaw = float(yaws[index]) if yaws is not None and index < len(yaws) else yaw_for_positions(positions, index)
        ego_corners = obb_corners(position.astype(np.float64), ego_extent, ego_yaw)

        for actor in load_json_gz(boxes_path):
            features = actor_box_features(origin_matrix, actor)
            if features is None:
                continue
            if index == 0:
                other_boxes_for_context.append(features)
                other_velocities_for_context.append([
                    float(features[5] * math.cos(features[4])),
                    float(features[5] * math.sin(features[4])),
                ])
            actor_center = np.array(features[:2], dtype=np.float64)
            actor_extent = np.array(features[2:4], dtype=np.float64)
            actor_yaw = features[4]
            actor_corners = obb_corners(actor_center, actor_extent, actor_yaw)
            if polygons_overlap(ego_corners, actor_corners):
                return True, other_boxes_for_context, other_velocities_for_context

    return False, other_boxes_for_context, other_velocities_for_context


def perturb_waypoints(gt_waypoints: np.ndarray, rng: random.Random) -> np.ndarray:
    perturbed = gt_waypoints.copy()
    kind = rng.choice(("lateral", "rotation", "scale", "bend"))
    if kind == "lateral":
        perturbed[:, 1] += rng.choice((-1.0, 1.0)) * rng.uniform(1.0, 4.0)
    elif kind == "rotation":
        angle = math.radians(rng.uniform(-25.0, 25.0))
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
        perturbed = perturbed @ rotation.T
    elif kind == "scale":
        perturbed[:, 0] *= rng.uniform(0.6, 1.4)
        perturbed[:, 1] *= rng.uniform(0.6, 1.4)
    elif kind == "bend":
        ramp = np.linspace(0.0, 1.0, len(perturbed), dtype=np.float32)
        perturbed[:, 1] += rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 3.5) * ramp
    return perturbed.astype(np.float32)


def load_measurement_sequence(route_dir: Path, frame: int, pred_len: int) -> list[dict[str, Any]] | None:
    measurements = []
    for offset in range(pred_len + 1):
        path = route_dir / "measurements" / f"{frame + offset:04}.json.gz"
        if not path.is_file():
            return None
        measurements.append(load_json_gz(path))
    return measurements


def get_ego_extent(route_dir: Path, frame: int) -> np.ndarray:
    boxes = load_json_gz(route_dir / "boxes" / f"{frame:04}.json.gz")
    for box in boxes:
        if box.get("class") == "ego_car":
            extent = box["extent"]
            return np.array([float(extent[0]), float(extent[1])], dtype=np.float64)
    return np.array([2.45, 1.06], dtype=np.float64)


def make_candidates(
    route_dir: Path,
    frame: int,
    measurements: list[dict[str, Any]],
    num_perturbations: int,
    dt: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    origin_matrix = np.array(measurements[0]["ego_matrix"], dtype=np.float64)
    ego_extent = get_ego_extent(route_dir, frame)
    gt_waypoints = get_gt_waypoints(measurements)
    gt_target_speed = float(measurements[0].get("target_speed", 0.0))
    current_speed = float(measurements[0].get("speed", 0.0))
    first_actor_target = get_first_future_actor_target(route_dir, frame, origin_matrix)

    candidates = []
    for variant_idx in range(num_perturbations + 1):
        if variant_idx == 0:
            waypoints = gt_waypoints
            target_speed = gt_target_speed
            variant = "gt"
        elif variant_idx == 1 and first_actor_target is not None:
            waypoints = np.linspace(first_actor_target / max(len(gt_waypoints), 1), first_actor_target, len(gt_waypoints))
            target_speed = max(float(np.linalg.norm(first_actor_target) / max(dt, 1e-6)), gt_target_speed)
            variant = "actor_target"
        else:
            waypoints = perturb_waypoints(gt_waypoints, rng)
            target_speed = max(0.0, gt_target_speed * rng.uniform(0.35, 1.8) + rng.uniform(-1.0, 4.0))
            variant = "perturbed"

        timed_positions, timed_yaws = path_positions_for_speed(waypoints, target_speed, dt, current_speed)
        will_collide, other_boxes, other_velocities = candidate_collides(
            route_dir, frame, origin_matrix, ego_extent, timed_positions, timed_yaws)
        candidates.append({
            "variant": variant,
            "waypoints": waypoints.round(4).tolist(),
            "target_speed": round(float(target_speed), 4),
            "will_collide": 0 if will_collide else 1,
            "other_boxes": np.asarray(other_boxes, dtype=np.float32).round(4).tolist(),
            "other_velocities": np.asarray(other_velocities, dtype=np.float32).round(4).tolist(),
        })

    return candidates


def get_first_future_actor_target(route_dir: Path, frame: int, origin_matrix: np.ndarray) -> np.ndarray | None:
    boxes_path = route_dir / "boxes" / f"{frame + 1:04}.json.gz"
    if not boxes_path.is_file():
        return None

    best_target = None
    best_distance = float("inf")
    for actor in load_json_gz(boxes_path):
        features = actor_box_features(origin_matrix, actor)
        if features is None:
            continue
        target = np.array(features[:2], dtype=np.float32)
        distance = float(np.linalg.norm(target))
        if 2.0 < distance < best_distance:
            best_target = target
            best_distance = distance
    return best_target


def iter_route_dirs(data_root: Path):
    for measurements_dir in data_root.glob("*/*/measurements"):
        if measurements_dir.is_dir():
            yield measurements_dir.parent


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def label_route(
    route_dir: Path,
    pred_len: int,
    data_save_freq: int,
    carla_fps: int,
    num_perturbations: int,
    use_focus_filter: bool,
    overwrite: bool,
    rng: random.Random,
) -> tuple[int, int, int]:
    output_path = route_dir / "plan_safety_labels.json.gz"
    if output_path.exists() and not overwrite:
        return 0, 0, 0

    scenario = route_dir.parent.name
    measurement_paths = sorted((route_dir / "measurements").glob("*.json.gz"))
    max_frame = len(measurement_paths) - pred_len - 1
    dt = data_save_freq / float(carla_fps)

    frames = {}
    safe_count = 0
    unsafe_count = 0

    frame_iter = progress(
        range(max(0, max_frame)),
        desc=route_dir.name,
        leave=False,
        unit="frame",
    )
    for frame in frame_iter:
        measurements = load_measurement_sequence(route_dir, frame, pred_len)
        if measurements is None or not is_focus_frame(scenario, measurements[0], use_focus_filter):
            continue

        candidates = make_candidates(route_dir, frame, measurements, num_perturbations, dt, rng)
        if candidates:
            frame_key = f"{frame:04}"
            frames[frame_key] = candidates
            unsafe_count += sum(1 for candidate in candidates if candidate["will_collide"] == 0)
            safe_count += sum(1 for candidate in candidates if candidate["will_collide"] == 1)

    if frames:
        dump_json_gz(
            output_path,
            {
                "label_map": {"unsafe_will_collide": 0, "safe": 1},
                "pred_len": pred_len,
                "dt": dt,
                "frames": frames,
            },
        )

    return len(frames), safe_count, unsafe_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate plan-safety labels from existing CARLA Garage data.")
    parser.add_argument("--data-root", type=Path, default=Path("carla_garage/training_data"))
    parser.add_argument("--pred-len", type=int, default=8)
    parser.add_argument("--data-save-freq", type=int, default=5)
    parser.add_argument("--carla-fps", type=int, default=20)
    parser.add_argument("--num-perturbations", type=int, default=6)
    parser.add_argument("--all-frames", action="store_true", help="Use all frames, not just junction/lane-change focus frames.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-routes", type=int, default=None)
    args = parser.parse_args()

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    rng = random.Random(args.seed)
    route_count = 0
    frame_count = 0
    safe_count = 0
    unsafe_count = 0

    route_dirs = list(iter_route_dirs(args.data_root))
    if args.max_routes is not None:
        route_dirs = route_dirs[:args.max_routes]

    for route_dir in progress(route_dirs, desc="Routes", unit="route"):
        route_count += 1
        frames, safe, unsafe = label_route(
            route_dir=route_dir,
            pred_len=args.pred_len,
            data_save_freq=args.data_save_freq,
            carla_fps=args.carla_fps,
            num_perturbations=args.num_perturbations,
            use_focus_filter=not args.all_frames,
            overwrite=args.overwrite,
            rng=rng,
        )
        frame_count += frames
        safe_count += safe
        unsafe_count += unsafe

    print(f"Scanned routes: {route_count}")
    print(f"Labeled frames: {frame_count}")
    print(f"unsafe_will_collide=0: {unsafe_count}")
    print(f"safe=1: {safe_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
