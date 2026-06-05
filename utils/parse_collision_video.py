#!/usr/bin/env python3
"""Create videos for collision routes in a results/<exp_name> folder."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import os
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


COLLISION_KEYS = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")
COLLISION_PATTERN = re.compile(
    r"type=(?P<object_type>.+?) and id=(?P<object_id>\d+) at "
    r"\(x=(?P<x>-?\d+(?:\.\d+)?), y=(?P<y>-?\d+(?:\.\d+)?), z=(?P<z>-?\d+(?:\.\d+)?)\)"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make collision videos from a CARLA results/<exp_name> folder."
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="Experiment folder path, or experiment name under carla_garage/results.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/results",
        help="Root used when exp is only a name. Defaults to carla_garage/results.",
    )
    parser.add_argument(
        "--debug-json",
        type=Path,
        default=None,
        help="Path to debug_results.json. Defaults to <exp>/debug_results.json.",
    )
    parser.add_argument(
        "--folder1",
        default="model_results",
        help="First frame folder relative to each route folder. Defaults to model_results.",
    )
    parser.add_argument(
        "--folder2",
        default='none',
        help=(
            "Optional second frame folder relative to each route folder. "
            "Use 'none' to make single-folder videos. Defaults to sensor_data/attention_overlay."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <exp>/collision_videos.",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Output FPS. Defaults to 20.")
    parser.add_argument("--left", type=int, default=60, help="Frames before collision. Defaults to 60.")
    parser.add_argument("--right", type=int, default=60, help="Frames after collision. Defaults to 60.")
    parser.add_argument(
        "--full-route",
        action="store_true",
        help="Use the full route video instead of a collision-centered window.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing videos.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def parse_collision_message(message: str) -> dict:
    match = COLLISION_PATTERN.search(message)
    if not match:
        return {
            "object_type": None,
            "object_id": None,
            "collision_x": None,
            "collision_y": None,
            "collision_z": None,
            "collision_message": message,
        }

    parsed = match.groupdict()
    return {
        "object_type": parsed["object_type"],
        "object_id": int(parsed["object_id"]),
        "collision_x": float(parsed["x"]),
        "collision_y": float(parsed["y"]),
        "collision_z": float(parsed["z"]),
        "collision_message": message,
    }


def load_collision_events(debug_json_path: Path) -> list[dict]:
    debug_results = load_json(debug_json_path)
    records = debug_results.get("_checkpoint", {}).get("records", [])
    events = []

    for record in records:
        infractions = record.get("infractions", {})
        for collision_kind in COLLISION_KEYS:
            for collision_idx, message in enumerate(infractions.get(collision_kind, [])):
                event = parse_collision_message(message)
                event.update(
                    {
                        "route": record.get("timestamp"),
                        "route_index": record.get("index"),
                        "route_id": record.get("route_id"),
                        "collision_kind": collision_kind,
                        "collision_index": collision_idx,
                        "score_composed": record.get("scores", {}).get("score_composed"),
                    }
                )
                events.append(event)

    return events


def flatten_one(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def load_route_records(route_dir: Path) -> dict | None:
    records_path = route_dir / "records.json.gz"
    if not records_path.exists():
        return None
    with gzip.open(records_path, "rt", encoding="utf-8") as infile:
        return json.load(infile)


def estimate_collision_state_index(route_dir: Path, event: dict) -> tuple[int | None, str]:
    route_records = load_route_records(route_dir)
    if not route_records:
        return None, "missing_records"

    states = route_records.get("states", [])
    if not states:
        return None, "empty_records"

    object_id = event.get("object_id")
    collision_x = event.get("collision_x")
    collision_y = event.get("collision_y")
    best_idx = None
    best_distance = math.inf
    best_source = "not_found"

    for state_idx, state in enumerate(states):
        ids = flatten_one(state.get("id", []))
        positions = flatten_one(state.get("pos", []))
        if not isinstance(ids, list) or not isinstance(positions, list):
            continue

        for actor_idx, raw_actor_id in enumerate(ids):
            actor_id = flatten_one(raw_actor_id)
            pos = positions[actor_idx] if actor_idx < len(positions) else None
            if not isinstance(pos, list) or len(pos) < 2:
                continue

            actor_matches = object_id is not None and int(actor_id) == int(object_id)
            if actor_matches:
                if collision_x is None or collision_y is None:
                    return state_idx, "object_id"
                distance = math.hypot(float(pos[0]) - collision_x, float(pos[1]) - collision_y)
                if distance < best_distance:
                    best_idx = state_idx
                    best_distance = distance
                    best_source = "object_id_and_position"
            elif collision_x is not None and collision_y is not None:
                distance = math.hypot(float(pos[0]) - collision_x, float(pos[1]) - collision_y)
                if distance < best_distance:
                    best_idx = state_idx
                    best_distance = distance
                    best_source = "nearest_position"

    return best_idx, best_source


def frame_sort_key(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def get_frames(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        return {}
    return {path.stem: path for path in sorted(folder.glob("*.png"), key=frame_sort_key)}


def frame_ids_for_video(route_dir: Path, folder1: str, folder2: str | None) -> tuple[list[str], Path, Path | None]:
    frame_dir1 = route_dir / folder1
    frames1 = get_frames(frame_dir1)
    if not frames1:
        return [], frame_dir1, None

    if folder2 is None:
        return sorted(frames1, key=lambda frame: int(frame) if frame.isdigit() else frame), frame_dir1, None

    frame_dir2 = route_dir / folder2
    frames2 = get_frames(frame_dir2)
    if not frames2:
        return sorted(frames1, key=lambda frame: int(frame) if frame.isdigit() else frame), frame_dir1, None

    common = sorted(set(frames1) & set(frames2), key=lambda frame: int(frame) if frame.isdigit() else frame)
    return common, frame_dir1, frame_dir2


def estimate_collision_frame_id(route_dir: Path, event: dict, frame_ids: list[str]) -> tuple[str | None, str]:
    if not frame_ids:
        return None, "no_frames"

    state_idx, source = estimate_collision_state_index(route_dir, event)
    if state_idx is None:
        return None, source

    route_records = load_route_records(route_dir)
    n_states = len(route_records.get("states", [])) if route_records else 0
    if n_states <= 1:
        return frame_ids[-1], source

    frame_pos = round((state_idx / (n_states - 1)) * (len(frame_ids) - 1))
    frame_pos = max(0, min(len(frame_ids) - 1, frame_pos))
    return frame_ids[frame_pos], source


def select_window(frame_ids: list[str], center_frame: str | None, left: int, right: int, full_route: bool) -> list[str]:
    if full_route or center_frame is None or center_frame not in frame_ids:
        return frame_ids

    center_idx = frame_ids.index(center_frame)
    start = max(0, center_idx - left)
    end = min(len(frame_ids), center_idx + right + 1)
    return frame_ids[start:end]


def read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def resize_keep_ratio(image, target_height: int):
    height, width = image.shape[:2]
    new_width = int(width * target_height / height)
    return cv2.resize(image, (new_width, target_height))


def make_video(route_dir: Path, frame_ids: list[str], frame_dir1: Path, frame_dir2: Path | None, output_path: Path,
               fps: float, title: str) -> None:
    first1 = read_image(frame_dir1 / f"{frame_ids[0]}.png")
    if frame_dir2 is not None:
        first2 = read_image(frame_dir2 / f"{frame_ids[0]}.png")
        video_height = min(first1.shape[0], first2.shape[0])
        preview1 = resize_keep_ratio(first1, video_height)
        preview2 = resize_keep_ratio(first2, video_height)
        video_width = preview1.shape[1] + preview2.shape[1]
    else:
        video_height, video_width = first1.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (video_width, video_height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for frame_id in frame_ids:
            frame1 = read_image(frame_dir1 / f"{frame_id}.png")
            if frame_dir2 is not None:
                frame2 = read_image(frame_dir2 / f"{frame_id}.png")
                frame1 = resize_keep_ratio(frame1, video_height)
                frame2 = resize_keep_ratio(frame2, video_height)
                frame = np.hstack((frame1, frame2))
            else:
                frame = cv2.resize(frame1, (video_width, video_height))

            cv2.putText(frame, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"frame {frame_id}", (12, video_height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"frame {frame_id}", (12, video_height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()


def safe_name(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def main() -> int:
    args = parse_arguments()
    if cv2 is None:
        raise SystemExit("OpenCV is required. Run this with the same Python environment used for CARLA evaluation.")

    exp_dir = Path(os.path.join(args.results_root, args.exp))
    debug_json_path = args.debug_json.expanduser().resolve() if args.debug_json else exp_dir / "debug_results.json"
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else exp_dir / "collision_videos"
    folder2 = None if args.folder2.lower() in ("", "none", "null") else args.folder2

    if not exp_dir.is_dir():
        raise SystemExit(f"Experiment folder does not exist: {exp_dir}")
    if not debug_json_path.exists():
        raise SystemExit(f"debug_results.json does not exist: {debug_json_path}")

    events = load_collision_events(debug_json_path)
    if not events:
        print(f"No collision events found in {debug_json_path}")
        return 0

    made = 0
    skipped = 0
    for event_idx, event in enumerate(events):
        route_name = event.get("route")
        route_dir = exp_dir / str(route_name)
        if not route_dir.is_dir():
            print(f"Skip event {event_idx}: missing route folder {route_dir}")
            skipped += 1
            continue

        all_frame_ids, frame_dir1, frame_dir2 = frame_ids_for_video(route_dir, args.folder1, folder2)
        if not all_frame_ids:
            print(f"Skip event {event_idx}: no PNG frames in {frame_dir1}")
            skipped += 1
            continue

        center_frame, frame_source = estimate_collision_frame_id(route_dir, event, all_frame_ids)
        frame_ids = select_window(all_frame_ids, center_frame, args.left, args.right, args.full_route)
        if not frame_ids:
            print(f"Skip event {event_idx}: empty frame window for {route_name}")
            skipped += 1
            continue

        output_name = (
            f"route{event.get('route_index')}_event{event_idx}_"
            f"{safe_name(event.get('collision_kind'))}_{safe_name(event.get('object_id'))}.mp4"
        )
        output_path = output_dir / output_name
        if output_path.exists() and not args.overwrite:
            print(f"Skip existing: {output_path}")
            skipped += 1
            continue

        title = (
            f"{event.get('collision_kind')} id={event.get('object_id')} "
            f"center={center_frame or 'unknown'} source={frame_source}"
        )
        make_video(route_dir, frame_ids, frame_dir1, frame_dir2, output_path, args.fps, title)
        print(f"Wrote {output_path} ({len(frame_ids)} frames from {route_name})")
        made += 1

    print(f"Done. Wrote {made} collision video(s), skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
