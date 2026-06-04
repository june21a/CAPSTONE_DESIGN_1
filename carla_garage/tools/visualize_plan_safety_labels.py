#!/usr/bin/env python3
"""Visualize plan-safety labels on saved LiDAR BEV frames."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_plan_safety_labels import path_positions_for_speed


HIST_MAX_PER_PIXEL = 5
LIDAR_SPLIT_HEIGHT = 0.2
MAX_HEIGHT_LIDAR = 4.0

CLASS_COLORS = [
    np.array([255, 165, 0]),  # car
    np.array([0, 255, 0]),  # walker
    np.array([255, 0, 0]),  # traffic light
    np.array([250, 160, 160]),  # stop sign
    np.array([16, 133, 133]),  # emergency vehicle
]
EGO_COLOR = (0, 220, 255)
FUTURE_EGO_COLOR = (255, 215, 0)
TARGET_POINT_COLOR = (255, 0, 255)
NEXT_TARGET_POINT_COLOR = (120, 0, 255)
DEFAULT_EGO_EXTENT = np.array([2.45, 1.06], dtype=np.float32)
DEFAULT_DT = 5.0 / 20.0


def load_json_gz(path: Path):
  with gzip.open(path, "rt", encoding="utf-8") as file:
    return json.load(file)


def lidar_frame_candidates(route_dir: Path, frame: str) -> dict[str, list[Path]]:
  frame_int = int(frame)
  frame_names = [frame, f"{frame_int:04}", f"{frame_int:05}"]
  lidar_dirs = [route_dir / "sensor_data" / "lidar", route_dir / "lidar"]
  return {
      "bev": [lidar_dir / f"{name}_bev.npz" for lidar_dir in lidar_dirs for name in frame_names],
      "points_npz": [lidar_dir / f"{name}_points.npz" for lidar_dir in lidar_dirs for name in frame_names],
      "laz": [lidar_dir / f"{name}.laz" for lidar_dir in lidar_dirs for name in frame_names],
  }


def find_existing(paths: list[Path]) -> Path | None:
  for path in paths:
    if path.is_file():
      return path
  return None


def lidar_points_to_bev(lidar: np.ndarray, pixels_per_meter: float, min_x: float, max_x: float, min_y: float,
                        max_y: float, use_ground_plane: bool) -> np.ndarray:
  def splat_points(point_cloud: np.ndarray) -> np.ndarray:
    xbins = np.linspace(min_x, max_x, int((max_x - min_x) * pixels_per_meter) + 1)
    ybins = np.linspace(min_y, max_y, int((max_y - min_y) * pixels_per_meter) + 1)
    hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
    hist[hist > HIST_MAX_PER_PIXEL] = HIST_MAX_PER_PIXEL
    return (hist / HIST_MAX_PER_PIXEL).T

  lidar = np.asarray(lidar, dtype=np.float32)
  if lidar.ndim != 2 or lidar.shape[1] < 3:
    raise ValueError(f"Expected LiDAR point cloud with shape (N, >=3), got {lidar.shape}")

  lidar = lidar[lidar[:, 2] < MAX_HEIGHT_LIDAR]
  below = lidar[lidar[:, 2] <= LIDAR_SPLIT_HEIGHT]
  above = lidar[lidar[:, 2] > LIDAR_SPLIT_HEIGHT]
  channels = [splat_points(below), splat_points(above)] if use_ground_plane else [splat_points(above)]
  return np.stack(channels, axis=0).astype(np.float32)


def load_laz_points(path: Path) -> np.ndarray:
  try:
    import laspy
  except ImportError as exc:
    raise ImportError("Reading .laz LiDAR requires laspy. Install it in this environment, for example: "
                      "pip install 'laspy[lazrs]'") from exc

  return np.asarray(laspy.read(path).xyz, dtype=np.float32)


def lidar_bev_to_image(lidar_bev: np.ndarray, source_path: Path) -> np.ndarray:
  lidar_bev = np.asarray(lidar_bev)
  if lidar_bev.ndim == 4:
    lidar_map = lidar_bev[0, 0]
  elif lidar_bev.ndim == 3:
    lidar_map = lidar_bev[0]
  elif lidar_bev.ndim == 2:
    lidar_map = lidar_bev
  else:
    raise ValueError(f"Unsupported LiDAR BEV shape {lidar_bev.shape} in {source_path}")

  lidar_map = lidar_map.astype(np.float32)
  lidar_map = lidar_map - float(np.min(lidar_map))
  max_value = float(np.max(lidar_map))
  if max_value > 1e-6:
    lidar_map = lidar_map / max_value

  image = 255 - (lidar_map * 255).astype(np.uint8)
  return np.stack([image, image, image], axis=-1)


def load_npz_points(path: Path) -> np.ndarray:
  with np.load(path) as data:
    if "lidar" in data:
      return np.asarray(data["lidar"], dtype=np.float32)
    return np.asarray(data[data.files[0]], dtype=np.float32)


def load_lidar_bev(route_dir: Path, frame: str, pixels_per_meter: float, min_x: float, max_x: float, min_y: float,
                   max_y: float, use_ground_plane: bool) -> np.ndarray:
  candidates = lidar_frame_candidates(route_dir, frame)
  lidar_path = find_existing(candidates["bev"])
  if lidar_path is not None:
    with np.load(lidar_path) as data:
      if "lidar_bev" in data:
        lidar_bev = data["lidar_bev"]
      else:
        lidar_bev = data[data.files[0]]
    return lidar_bev_to_image(lidar_bev, lidar_path)

  points_path = find_existing(candidates["points_npz"])
  if points_path is not None:
    lidar_bev = lidar_points_to_bev(load_npz_points(points_path), pixels_per_meter, min_x, max_x, min_y, max_y,
                                    use_ground_plane)
    return lidar_bev_to_image(lidar_bev, points_path)

  laz_path = find_existing(candidates["laz"])
  if laz_path is not None:
    lidar_bev = lidar_points_to_bev(load_laz_points(laz_path), pixels_per_meter, min_x, max_x, min_y, max_y,
                                    use_ground_plane)
    return lidar_bev_to_image(lidar_bev, laz_path)

  searched = [str(path) for paths in candidates.values() for path in paths]
  raise FileNotFoundError("Missing LiDAR for frame "
                          f"{frame}. Tried:\n  " + "\n  ".join(searched))


def load_ego_extent(route_dir: Path, frame: str) -> np.ndarray:
  boxes_path = route_dir / "boxes" / f"{int(frame):04}.json.gz"
  if boxes_path.is_file():
    for box in load_json_gz(boxes_path):
      if box.get("class") == "ego_car":
        extent = box["extent"]
        return np.array([float(extent[0]), float(extent[1])], dtype=np.float32)
  return DEFAULT_EGO_EXTENT.copy()


def load_measurement(route_dir: Path, frame: str) -> dict | None:
  frame_int = int(frame)
  for name in (frame, f"{frame_int:04}", f"{frame_int:05}"):
    measurement_path = route_dir / "measurements" / f"{name}.json.gz"
    if measurement_path.is_file():
      return load_json_gz(measurement_path)
  return None


def ego_to_image_point(x: float, y: float, pixels_per_meter: float, min_y: float,
                       max_x: float) -> tuple[int, int]:
  col = int((y - min_y) * pixels_per_meter)
  row = int((max_x - x) * pixels_per_meter)
  return col, row


def draw_waypoints(image: np.ndarray, waypoints, color, pixels_per_meter: float, min_y: float,
                   max_x: float) -> None:
  pil_image = Image.fromarray(image)
  draw = ImageDraw.Draw(pil_image)
  points: list[tuple[int, int]] = []
  for waypoint in waypoints:
    x, y = float(waypoint[0]), float(waypoint[1])
    col, row = ego_to_image_point(x, y, pixels_per_meter, min_y, max_x)
    points.append((col, row))
    if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
      draw.ellipse((col - 4, row - 4, col + 4, row + 4), fill=tuple(color))

  for start, end in zip(points, points[1:]):
    draw.line((start, end), fill=tuple(color), width=2)
  image[...] = np.array(pil_image)


def draw_target_point(image: np.ndarray, target_point, color: tuple[int, int, int], pixels_per_meter: float,
                      min_y: float, max_x: float) -> None:
  if target_point is None or len(target_point) < 2:
    return

  x, y = float(target_point[0]), float(target_point[1])
  col, row = ego_to_image_point(x, y, pixels_per_meter, min_y, max_x)
  if not (0 <= row < image.shape[0] and 0 <= col < image.shape[1]):
    return

  pil_image = Image.fromarray(image)
  draw = ImageDraw.Draw(pil_image)
  radius = 9
  draw.line((col - radius, row, col + radius, row), fill=(0, 0, 0), width=5)
  draw.line((col, row - radius, col, row + radius), fill=(0, 0, 0), width=5)
  draw.line((col - radius, row, col + radius, row), fill=color, width=3)
  draw.line((col, row - radius, col, row + radius), fill=color, width=3)
  draw.ellipse((col - 4, row - 4, col + 4, row + 4), fill=color, outline=(0, 0, 0))
  image[...] = np.array(pil_image)


def draw_rollout_path(image: np.ndarray, positions: np.ndarray, color: tuple[int, int, int],
                      pixels_per_meter: float, min_y: float, max_x: float) -> None:
  points = [
      ego_to_image_point(float(position[0]), float(position[1]), pixels_per_meter, min_y, max_x)
      for position in positions
  ]
  if len(points) < 2:
    return

  pil_image = Image.fromarray(image)
  draw = ImageDraw.Draw(pil_image)
  draw.line(points, fill=(0, 0, 0), width=5)
  draw.line(points, fill=color, width=3)
  image[...] = np.array(pil_image)


def draw_future_ego_rollout(image: np.ndarray, candidate: dict, measurement: dict | None, ego_extent: np.ndarray,
                            dt: float, pixels_per_meter: float, min_y: float, max_x: float) -> None:
  waypoints = np.asarray(candidate.get("waypoints", []), dtype=np.float32)
  if waypoints.size == 0:
    return
  waypoints = waypoints.reshape(-1, 2)

  target_speed = float(candidate.get("target_speed", 0.0))
  current_speed = float(measurement.get("speed", 0.0)) if measurement is not None else 0.0
  positions, yaws = path_positions_for_speed(waypoints, target_speed, dt, current_speed)
  if len(positions) == 0:
    return

  draw_rollout_path(image, positions, FUTURE_EGO_COLOR, pixels_per_meter, min_y, max_x)
  for index, position in enumerate(positions):
    color_weight = 0.45 + 0.55 * float(index + 1) / len(positions)
    color = tuple(int(channel * color_weight) for channel in FUTURE_EGO_COLOR)
    box = np.array([position[0], position[1], ego_extent[0], ego_extent[1], yaws[index]], dtype=np.float32)
    draw_box(image, box, color=color, thickness=2, pixels_per_meter=pixels_per_meter, min_y=min_y, max_x=max_x)


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], thickness: int,
             pixels_per_meter: float, min_y: float, max_x: float) -> None:
  center = np.array([box[0], box[1]], dtype=np.float32)
  extent_x = float(box[2])
  extent_y = float(box[3])
  yaw = float(box[4])
  rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
  local_corners = np.array(
      [[extent_x, extent_y], [extent_x, -extent_y], [-extent_x, -extent_y], [-extent_x, extent_y]],
      dtype=np.float32,
  )
  corners = (rotation @ local_corners.T).T + center
  points = [ego_to_image_point(float(point[0]), float(point[1]), pixels_per_meter, min_y, max_x) for point in corners]
  pil_image = Image.fromarray(image)
  draw = ImageDraw.Draw(pil_image)
  draw.line(points + [points[0]], fill=color, width=thickness)
  if len(box) > 5 and float(box[5]) > 0.0:
    speed = float(box[5])
    speed_point = center + np.array([np.cos(yaw) * speed, np.sin(yaw) * speed], dtype=np.float32)
    center_point = ego_to_image_point(float(center[0]), float(center[1]), pixels_per_meter, min_y, max_x)
    speed_image_point = ego_to_image_point(float(speed_point[0]), float(speed_point[1]), pixels_per_meter, min_y, max_x)
    draw.line((center_point, speed_image_point), fill=color, width=thickness)
  image[...] = np.array(pil_image)


def render_frame(route_dir: Path, labels: dict, frame: str, candidate_index: int, output_dir: Path,
                 scale_factor: int, pixels_per_meter: float, min_x: float, max_x: float, min_y: float,
                 max_y: float, use_ground_plane: bool) -> Path:
  candidates = labels["frames"][frame]
  candidate = candidates[candidate_index]

  image = load_lidar_bev(route_dir, frame, pixels_per_meter, min_x, max_x, min_y, max_y, use_ground_plane)
  image = np.array(
      Image.fromarray(image).resize((image.shape[1] * scale_factor, image.shape[0] * scale_factor),
                                    Image.Resampling.NEAREST))
  image = np.ascontiguousarray(np.rot90(image, k=1), dtype=np.uint8)

  loc_pixels_per_meter = pixels_per_meter * scale_factor
  ego_extent = load_ego_extent(route_dir, frame)
  ego_box = np.array([0.0, 0.0, ego_extent[0], ego_extent[1], 0.0], dtype=np.float32)
  draw_box(image, ego_box, color=EGO_COLOR, thickness=4, pixels_per_meter=loc_pixels_per_meter, min_y=min_y,
           max_x=max_x)

  for box in candidate.get("other_boxes", []):
    box = np.asarray(box, dtype=np.float32).copy()
    class_id = int(box[7]) if len(box) > 7 else 0
    color = CLASS_COLORS[class_id % len(CLASS_COLORS)].copy()
    if len(box) > 6:
      color[1] = color[1] * (1.0 - float(box[6]))
    draw_box(image, box, color=tuple(int(channel) for channel in color), thickness=3,
             pixels_per_meter=loc_pixels_per_meter, min_y=min_y, max_x=max_x)

  unsafe = is_unsafe_candidate(candidate, labels)
  path_color = (0, 0, 255) if unsafe else (0, 200, 0)
  draw_waypoints(image, candidate.get("waypoints", []), path_color, loc_pixels_per_meter, min_y, max_x)

  measurement = load_measurement(route_dir, frame)
  draw_future_ego_rollout(image, candidate, measurement, ego_extent, float(labels.get("dt", DEFAULT_DT)),
                          loc_pixels_per_meter, min_y, max_x)
  if measurement is not None:
    draw_target_point(image, measurement.get("target_point"), TARGET_POINT_COLOR, loc_pixels_per_meter, min_y, max_x)
    draw_target_point(image, measurement.get("target_point_next"), NEXT_TARGET_POINT_COLOR, loc_pixels_per_meter, min_y,
                      max_x)

  label = "unsafe" if unsafe else "safe"
  pil_image = Image.fromarray(image)
  draw = ImageDraw.Draw(pil_image)
  draw.text((10, 10), f"{frame} candidate {candidate_index} {candidate.get('variant', '')} {label}", fill=path_color)
  target_speed = float(candidate.get("target_speed", 0.0))
  target_speed_text = f"target speed {target_speed:.2f} m/s"
  text_bbox = draw.textbbox((0, 0), target_speed_text)
  text_width = text_bbox[2] - text_bbox[0]
  draw.text((max(10, image.shape[1] - text_width - 10), 10), target_speed_text, fill=(0, 0, 0))
  image[...] = np.array(pil_image)

  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / f"{frame}_candidate{candidate_index}_{label}.png"
  Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(output_path)
  return output_path


def iter_route_dirs(data_root: Path):
  for labels_path in sorted(data_root.glob("**/plan_safety_labels.json.gz")):
    yield labels_path.parent


def is_unsafe_candidate(candidate: dict, labels: dict) -> bool:
  label_map = labels.get("label_map", {})
  unsafe_label = label_map.get("unsafe_sim_collision", label_map.get("unsafe_will_collide", 1))
  return int(candidate.get("will_collide", 0)) == int(unsafe_label)


def unsafe_frames(labels: dict) -> list[str]:
  frames = []
  for frame, candidates in labels.get("frames", {}).items():
    if any(is_unsafe_candidate(candidate, labels) for candidate in candidates):
      frames.append(frame)
  return sorted(frames)


def collision_window_frames(labels: dict, before: int, after: int) -> list[str]:
  collision_frame = labels.get("collision_data_frame")
  if collision_frame is None:
    return []

  available_frames = set(labels.get("frames", {}))
  start = max(0, int(collision_frame) - before)
  end = int(collision_frame) + after
  return [f"{frame:04}" for frame in range(start, end + 1) if f"{frame:04}" in available_frames]


def selected_frames(labels: dict, args: argparse.Namespace) -> list[str]:
  frames = sorted(labels.get("frames", {}))
  if not frames:
    return []

  if args.frame:
    return [args.frame]
  if args.collision_window:
    frames = collision_window_frames(labels, args.frames_before_collision, args.frames_after_collision)
  elif args.unsafe_only:
    frames = unsafe_frames(labels)
  elif not args.all_labeled:
    frames = unsafe_frames(labels) or frames[:1]

  if args.limit is not None:
    frames = frames[:args.limit]
  return frames


def render_route(route_dir: Path, args: argparse.Namespace) -> int:
  labels_path = route_dir / "plan_safety_labels.json.gz"
  labels = load_json_gz(labels_path)
  frames = selected_frames(labels, args)
  if not frames:
    print(f"No selected frames in {route_dir}")
    return 0

  output_dir = args.output_dir or route_dir / "plan_safety_visualizations"
  rendered = 0
  for frame in frames:
    if frame not in labels["frames"]:
      raise KeyError(f"Frame {frame} not found in {labels_path}. Available examples: {sorted(labels['frames'])[:10]}")

    candidate_indices = range(len(labels["frames"][frame])) if args.all_candidates else [args.candidate]
    for candidate_index in candidate_indices:
      if args.list_frames:
        unsafe = any(is_unsafe_candidate(candidate, labels) for candidate in labels["frames"][frame])
        label = "unsafe" if unsafe else "safe"
        print(f"{route_dir} frame={frame} candidate={candidate_index} {label}")
        rendered += 1
        continue

      output_path = render_frame(
          route_dir=route_dir,
          labels=labels,
          frame=frame,
          candidate_index=candidate_index,
          output_dir=output_dir,
          scale_factor=args.scale_factor,
          pixels_per_meter=args.pixels_per_meter,
          min_x=args.min_x,
          max_x=args.max_x,
          min_y=args.min_y,
          max_y=args.max_y,
          use_ground_plane=args.use_ground_plane,
      )
      print(output_path)
      rendered += 1
  return rendered


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("route_dir", type=Path, nargs="?", help="Route folder containing plan_safety_labels.json.gz")
  parser.add_argument("--data-root", type=Path, help="Dataset root; renders routes containing unsafe labels.")
  parser.add_argument("--frame", help="Frame key, e.g. 0005. Defaults to the first labeled frame.")
  parser.add_argument("--candidate", type=int, default=0, help="Candidate index within the frame.")
  parser.add_argument("--all-candidates", action="store_true", help="Render every candidate for the selected frame.")
  parser.add_argument("--unsafe-only", action="store_true", help="Render labeled unsafe/collision-imminent frames only.")
  parser.add_argument("--collision-window", action="store_true", help="Render labeled frames around collision_data_frame.")
  parser.add_argument("--frames-before-collision", type=int, default=10)
  parser.add_argument("--frames-after-collision", type=int, default=0)
  parser.add_argument("--all-labeled", action="store_true", help="Render all labeled frames.")
  parser.add_argument("--limit", type=int, help="Maximum number of frames/routes to render.")
  parser.add_argument("--list-frames", action="store_true", help="Print selected frames without rendering PNGs.")
  parser.add_argument("--output-dir", type=Path, help="Where to write PNGs.")
  parser.add_argument("--scale-factor", type=int, default=4)
  parser.add_argument("--pixels-per-meter", type=float, default=4.0)
  parser.add_argument("--min-x", type=float, default=-32.0)
  parser.add_argument("--max-x", type=float, default=32.0)
  parser.add_argument("--min-y", type=float, default=-32.0)
  parser.add_argument("--max-y", type=float, default=32.0)
  parser.add_argument("--use-ground-plane", action="store_true", help="Use below+above channels when building BEV.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.data_root is not None:
    rendered_routes = 0
    for route_dir in iter_route_dirs(args.data_root):
      labels = load_json_gz(route_dir / "plan_safety_labels.json.gz")
      if not unsafe_frames(labels):
        continue
      render_route(route_dir, args)
      rendered_routes += 1
      if args.limit is not None and rendered_routes >= args.limit:
        break
    if rendered_routes == 0:
      raise ValueError(f"No routes with unsafe labels found under {args.data_root}")
    return

  if args.route_dir is None:
    raise ValueError("Provide either route_dir or --data-root")

  render_route(args.route_dir, args)


if __name__ == "__main__":
  main()
