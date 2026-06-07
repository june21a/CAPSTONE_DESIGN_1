#!/usr/bin/env python3
"""Predict vehicle motion and collision risk from saved sensor results.

Output coordinates use the current ego frame: x points forward, y points
right, and distances are in meters. Vehicles follow constant velocity and
constant yaw during prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


VEHICLE_CLASSES = {"car", "emergency_vehicle"}


def load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as infile:
    return json.load(infile)


def image_box_to_vehicle(
    box: np.ndarray,
    pixels_per_meter: float,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
  """Convert a saved BEV image box back to x-forward/y-right metric units."""
  metric = box.copy()
  metric[0] = (box[1] - origin_y) / pixels_per_meter
  metric[1] = (box[0] - origin_x) / pixels_per_meter
  metric[2] = box[3] / pixels_per_meter
  metric[3] = box[2] / pixels_per_meter
  metric[4] = -box[4]
  return metric


def normalize_boxes(
    detection: dict[str, Any],
    pixels_per_meter: float,
    image_size: float,
) -> list[dict[str, Any]]:
  boxes = np.asarray(detection.get("boxes", []), dtype=np.float64)
  if boxes.size == 0:
    return []
  boxes = boxes.reshape((-1, boxes.shape[-1]))

  class_names = detection.get("class_names", [])
  metric_limit = image_size / pixels_per_meter
  boxes_are_pixels = bool(np.any(np.abs(boxes[:, :2]) > metric_limit * 2.0))
  origin = image_size / 2.0

  normalized = []
  for raw_box in boxes:
    box = (
        image_box_to_vehicle(raw_box, pixels_per_meter, origin, origin)
        if boxes_are_pixels
        else raw_box.copy()
    )
    class_id = int(round(box[7])) if len(box) > 7 else 0
    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else "unknown"
    normalized.append({
        "box": box,
        "class": class_name,
        "class_id": class_id,
        "confidence": float(box[8]) if len(box) > 8 else 1.0,
    })
  return normalized


def match_detections(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    max_distance: float,
) -> list[tuple[int, int]]:
  candidates = []
  for prev_idx, prev in enumerate(previous):
    for curr_idx, curr in enumerate(current):
      if prev["class"] != curr["class"]:
        continue
      distance = float(np.linalg.norm(curr["position"] - prev["position"]))
      if distance <= max_distance:
        candidates.append((distance, prev_idx, curr_idx))

  matches = []
  used_previous = set()
  used_current = set()
  for _, prev_idx, curr_idx in sorted(candidates):
    if prev_idx in used_previous or curr_idx in used_current:
      continue
    used_previous.add(prev_idx)
    used_current.add(curr_idx)
    matches.append((prev_idx, curr_idx))
  return matches


def velocity_from_box(box: np.ndarray) -> np.ndarray:
  speed = max(0.0, float(box[5]))
  yaw = float(box[4])
  return np.array([speed * math.cos(yaw), speed * math.sin(yaw)], dtype=np.float64)


def prediction_times(horizon: float, step: float) -> list[float]:
  count = int(math.floor(horizon / step))
  times = [round(index * step, 10) for index in range(count + 1)]
  if not math.isclose(times[-1], horizon):
    times.append(horizon)
  return times


def bounding_box_corners(
    position: np.ndarray,
    yaw: float,
    extent: np.ndarray,
) -> np.ndarray:
  """Return corners of an oriented box; extent is the half-length/width."""
  local_corners = np.array([
      [extent[0], extent[1]],
      [extent[0], -extent[1]],
      [-extent[0], -extent[1]],
      [-extent[0], extent[1]],
  ])
  rotation = np.array([
      [math.cos(yaw), -math.sin(yaw)],
      [math.sin(yaw), math.cos(yaw)],
  ])
  return local_corners @ rotation.T + position


def bounding_boxes_overlap(first: np.ndarray, second: np.ndarray) -> bool:
  """Check two oriented rectangles using the separating axis theorem."""
  for corners in (first, second):
    edges = np.roll(corners, -1, axis=0) - corners
    for edge in edges[:2]:
      axis = np.array([-edge[1], edge[0]], dtype=np.float64)
      norm = np.linalg.norm(axis)
      if norm <= 1e-9:
        continue
      axis /= norm
      first_projection = first @ axis
      second_projection = second @ axis
      separated = (
          first_projection.max() < second_projection.min()
          or second_projection.max() < first_projection.min()
      )
      if separated:
        return False
  return True


def state_dict(
    position: np.ndarray,
    yaw: float,
    velocity: np.ndarray,
    extent: np.ndarray,
) -> dict[str, Any]:
  speed = float(np.linalg.norm(velocity))
  return {
      "position_m": {"x": float(position[0]), "y": float(position[1])},
      "yaw_rad": float(yaw),
      "yaw_deg": float(math.degrees(yaw)),
      "speed_mps": speed,
      "speed_kph": speed * 3.6,
      "velocity_mps": {"x": float(velocity[0]), "y": float(velocity[1])},
      "extent_m": {"x": float(extent[0]), "y": float(extent[1])},
  }


def estimate_route(
    route_dir: Path,
    fps: float,
    prediction_horizon: float,
    prediction_step: float,
    confidence_threshold: float,
    max_match_distance: float,
    velocity_smoothing: float,
    pixels_per_meter: float,
    image_size: float,
    ego_extent: np.ndarray,
) -> dict[str, Any]:
  sensor_dir = route_dir / "sensor_data"
  metadata_dir = sensor_dir / "metadata"
  detection_dir = sensor_dir / "vision_tasks" / "detection"
  if not metadata_dir.is_dir() or not detection_dir.is_dir():
    raise FileNotFoundError(
        f"{route_dir} must contain sensor_data/metadata and "
        "sensor_data/vision_tasks/detection"
    )

  frame_ids = sorted(
      path.stem
      for path in metadata_dir.glob("*.json")
      if (detection_dir / path.name).is_file()
  )
  if not frame_ids:
    raise RuntimeError(f"No matching metadata/detection JSON frames found in {route_dir}")

  times = prediction_times(prediction_horizon, prediction_step)
  next_track_id = 1
  previous_frame_id: int | None = None
  previous_detections: list[dict[str, Any]] = []
  frames = []

  for frame_name in frame_ids:
    frame_id = int(frame_name)
    metadata = load_json(metadata_dir / f"{frame_name}.json")
    detection = load_json(detection_dir / f"{frame_name}.json")

    # 1. Read the ego state in the current ego-fixed coordinate frame.
    ego_speed = float(metadata.get("speed", 0.0))
    ego_position = np.zeros(2, dtype=np.float64)
    ego_yaw = 0.0
    ego_velocity = np.array([ego_speed, 0.0], dtype=np.float64)

    # 2. Read surrounding position, yaw, speed, and extent.
    current_detections = []
    for item in normalize_boxes(detection, pixels_per_meter, image_size):
      if item["class"] not in VEHICLE_CLASSES or item["confidence"] < confidence_threshold:
        continue
      box = item["box"]
      item.update({
          "position": box[:2].astype(np.float64),
          "extent": box[2:4].astype(np.float64),
          "yaw": float(box[4]),
          "detected_speed": max(0.0, float(box[5])),
          "absolute_velocity": velocity_from_box(box),
          "relative_velocity": None,
          "track_id": None,
      })
      current_detections.append(item)

    if previous_frame_id is not None:
      dt = (frame_id - previous_frame_id) / fps
      if dt > 0.0:
        matches = match_detections(previous_detections, current_detections, max_match_distance)
        for prev_idx, curr_idx in matches:
          previous = previous_detections[prev_idx]
          current = current_detections[curr_idx]
          relative_velocity = (current["position"] - previous["position"]) / dt
          if previous["relative_velocity"] is not None:
            relative_velocity = (
                velocity_smoothing * previous["relative_velocity"]
                + (1.0 - velocity_smoothing) * relative_velocity
            )
          current["relative_velocity"] = relative_velocity
          current["absolute_velocity"] = relative_velocity + ego_velocity
          current["track_id"] = previous["track_id"]

    for current in current_detections:
      if current["track_id"] is None:
        current["track_id"] = next_track_id
        next_track_id += 1
      if current["relative_velocity"] is None:
        current["relative_velocity"] = current["absolute_velocity"] - ego_velocity

    current_ego_state = state_dict(ego_position, ego_yaw, ego_velocity, ego_extent)

    # 3. Predict ego motion with constant velocity.
    ego_timeline = []
    for time_s in times:
      future_position = ego_position + ego_velocity * time_s
      ego_timeline.append({
          "time_s": time_s,
          **state_dict(future_position, ego_yaw, ego_velocity, ego_extent),
      })

    vehicle_results = []
    for current in current_detections:
      position = current["position"]
      extent = current["extent"]
      yaw = current["yaw"]
      velocity = current["absolute_velocity"]
      future_timeline = []
      collision_times = []

      # 3-4. Predict surrounding motion and test oriented-box overlap.
      for time_s in times:
        ego_future_position = ego_position + ego_velocity * time_s
        vehicle_future_position = position + velocity * time_s
        ego_box = bounding_box_corners(ego_future_position, ego_yaw, ego_extent)
        vehicle_box = bounding_box_corners(vehicle_future_position, yaw, extent)
        collision = bounding_boxes_overlap(ego_box, vehicle_box)
        if collision:
          collision_times.append(time_s)
        future_timeline.append({
            "time_s": time_s,
            **state_dict(vehicle_future_position, yaw, velocity, extent),
            "position_relative_to_ego_m": {
                "x": float(vehicle_future_position[0] - ego_future_position[0]),
                "y": float(vehicle_future_position[1] - ego_future_position[1]),
            },
            "bounding_box_overlap": collision,
        })

      # 5. First overlap in the sampled timeline is the discrete TTC.
      time_to_collision = collision_times[0] if collision_times else None
      vehicle_results.append({
          "track_id": current["track_id"],
          "class": current["class"],
          "confidence": current["confidence"],
          "current_state": state_dict(position, yaw, velocity, extent),
          "detected_speed_mps": current["detected_speed"],
          "velocity_relative_mps": {
              "x": float(current["relative_velocity"][0]),
              "y": float(current["relative_velocity"][1]),
              "speed": float(np.linalg.norm(current["relative_velocity"])),
          },
          "collision_assessment": {
              "collision_risk": time_to_collision is not None,
              "time_to_collision_s": time_to_collision,
              "first_collision_frame_offset": (
                  None if time_to_collision is None else int(round(time_to_collision * fps))
              ),
          },
          "future_timeline": future_timeline,
      })

    collision_ttcs = [
        vehicle["collision_assessment"]["time_to_collision_s"]
        for vehicle in vehicle_results
        if vehicle["collision_assessment"]["time_to_collision_s"] is not None
    ]
    frames.append({
        "frame": frame_id,
        "timeline": {
            "1_current_ego_state": current_ego_state,
            "2_current_surrounding_vehicles": [
                {
                    "track_id": vehicle["track_id"],
                    "class": vehicle["class"],
                    "confidence": vehicle["confidence"],
                    **vehicle["current_state"],
                    "detected_speed_mps": vehicle["detected_speed_mps"],
                }
                for vehicle in vehicle_results
            ],
            "3_ego_constant_velocity_prediction": ego_timeline,
            "4_and_5_vehicle_predictions_and_collision_risk": vehicle_results,
        },
        "frame_collision_risk": bool(collision_ttcs),
        "minimum_time_to_collision_s": min(collision_ttcs, default=None),
    })
    previous_frame_id = frame_id
    previous_detections = current_detections

  return {
      "route": str(route_dir),
      "coordinate_system": {
          "origin": "ego vehicle at the current frame",
          "x": "forward",
          "y": "right",
          "unit": "meter",
      },
      "model": {
          "tracking": "nearest-neighbor association between consecutive frames",
          "motion": "constant velocity and constant yaw",
          "collision": "oriented bounding box overlap using separating axis theorem",
          "ttc": "first overlapping sample in the prediction timeline",
      },
      "fps": fps,
      "prediction_horizon_s": prediction_horizon,
      "prediction_step_s": prediction_step,
      "ego_extent_m": {"x": float(ego_extent[0]), "y": float(ego_extent[1])},
      "frames": frames,
  }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Predict vehicle motion, bounding-box overlap, and time-to-collision."
  )
  parser.add_argument("route_dir", type=Path, help="Route result directory containing sensor_data")
  parser.add_argument(
      "-o", "--output", type=Path, help="Output JSON path (default: ROUTE/motion_predictions.json)"
  )
  parser.add_argument("--fps", type=float, default=20.0, help="Simulation frame rate (default: 20)")
  parser.add_argument(
      "--prediction-horizon", type=float, default=3.0,
      help="Seconds to predict into the future (default: 3.0)",
  )
  parser.add_argument(
      "--prediction-step", type=float, default=0.1,
      help="Collision-check interval in seconds (default: 0.1)",
  )
  parser.add_argument("--confidence", type=float, default=0.5, help="Minimum box confidence")
  parser.add_argument(
      "--max-match-distance", type=float, default=4.0,
      help="Maximum inter-frame association distance in meters",
  )
  parser.add_argument(
      "--velocity-smoothing", type=float, default=0.6,
      help="Previous velocity weight in [0, 1)",
  )
  parser.add_argument(
      "--pixels-per-meter", type=float, default=16.0,
      help="Scale used by the saved 1024x1024 detection visualization",
  )
  parser.add_argument("--image-size", type=float, default=1024.0, help="Saved BEV image size")
  parser.add_argument(
      "--ego-extent-x", type=float, default=2.4508416652679443,
      help="Ego bounding-box half-length in meters",
  )
  parser.add_argument(
      "--ego-extent-y", type=float, default=1.0641621351242065,
      help="Ego bounding-box half-width in meters",
  )
  return parser


def main() -> None:
  parser = build_parser()
  args = parser.parse_args()
  if args.fps <= 0.0:
    parser.error("--fps must be positive")
  if args.prediction_horizon < 0.0 or args.prediction_step <= 0.0:
    parser.error("--prediction-horizon must be non-negative and --prediction-step must be positive")
  if not 0.0 <= args.velocity_smoothing < 1.0:
    parser.error("--velocity-smoothing must be in [0, 1)")
  if args.pixels_per_meter <= 0.0 or args.image_size <= 0.0:
    parser.error("--pixels-per-meter and --image-size must be positive")
  if args.ego_extent_x <= 0.0 or args.ego_extent_y <= 0.0:
    parser.error("ego extents must be positive")

  result = estimate_route(
      route_dir=args.route_dir,
      fps=args.fps,
      prediction_horizon=args.prediction_horizon,
      prediction_step=args.prediction_step,
      confidence_threshold=args.confidence,
      max_match_distance=args.max_match_distance,
      velocity_smoothing=args.velocity_smoothing,
      pixels_per_meter=args.pixels_per_meter,
      image_size=args.image_size,
      ego_extent=np.array([args.ego_extent_x, args.ego_extent_y], dtype=np.float64),
  )
  output_path = args.output or args.route_dir / "motion_predictions.json"
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as outfile:
    json.dump(result, outfile, indent=2)
  print(f"Wrote {len(result['frames'])} frames to {output_path}")


if __name__ == "__main__":
  main()
