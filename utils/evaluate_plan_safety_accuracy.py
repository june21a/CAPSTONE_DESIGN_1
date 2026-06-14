#!/usr/bin/env python3
"""Evaluate plan-safety prediction accuracy for CARLA Garage result folders."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_EXP = Path(
    "/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/"
    "carla_garage/results/pretrained_dynamic_plan_safety_head"
)
COLLISION_KEYS = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")
COLLISION_PATTERN = re.compile(
    r"type=(?P<object_type>.+?) and id=(?P<object_id>\d+) at "
    r"\(x=(?P<x>-?\d+(?:\.\d+)?), y=(?P<y>-?\d+(?:\.\d+)?), z=(?P<z>-?\d+(?:\.\d+)?)\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate safety prediction accuracy. Correct means predict safe when no collision occurs within "
            "the next horizon seconds, or predict unsafe when a collision occurs within that horizon. "
            "By default, frames after the first collision in a route are skipped."
        )
    )
    parser.add_argument(
        "--exp",
        type=Path,
        default=DEFAULT_EXP,
        help=f"Results experiment folder. Defaults to {DEFAULT_EXP}.",
    )
    parser.add_argument(
        "--debug-json",
        type=Path,
        default=None,
        help="Optional debug_results.json path. Defaults to <exp>/debug_results.json.",
    )
    parser.add_argument(
        "--horizon-sec",
        type=float,
        default=1.25,
        help="Future collision horizon in seconds. Defaults to 2.5.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Saved prediction frame rate used to convert seconds to frames. Defaults to 20.",
    )
    parser.add_argument(
        "--max-location-distance",
        type=float,
        default=15.0,
        help="Maximum meters allowed when inferring a collision frame from an infraction location. Defaults to 15.",
    )
    parser.add_argument(
        "--include-after-collision",
        action="store_true",
        help="Evaluate frames after the first collision. By default those frames are skipped.",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=None,
        help="Evaluate only route folder(s) with this name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write per-frame evaluation rows.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write aggregate and per-route metrics as JSON.",
    )
    parser.add_argument(
        "--show-routes",
        action="store_true",
        help="Print one summary line per route.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as infile:
        return json.load(infile)


def frame_sort_key(path: Path) -> int | str:
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def flatten_one(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def parse_collision_message(message: str) -> dict[str, Any]:
    match = COLLISION_PATTERN.search(message)
    if not match:
        return {
            "object_id": None,
            "collision_x": None,
            "collision_y": None,
            "collision_z": None,
            "collision_message": message,
        }

    parsed = match.groupdict()
    return {
        "object_id": int(parsed["object_id"]),
        "collision_x": float(parsed["x"]),
        "collision_y": float(parsed["y"]),
        "collision_z": float(parsed["z"]),
        "collision_message": message,
    }


def load_debug_records(debug_json_path: Path | None) -> dict[str, dict[str, Any]]:
    if debug_json_path is None or not debug_json_path.exists():
        return {}
    data = load_json(debug_json_path)
    records = data.get("_checkpoint", {}).get("records", [])
    return {str(record.get("timestamp")): record for record in records if record.get("timestamp")}


def route_result_record(route_dir: Path, debug_records: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    result_path = route_dir / "results.json.gz"
    if result_path.exists():
        return load_json_gz(result_path)
    return debug_records.get(route_dir.name)


def collision_events_from_record(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not record:
        return []

    events: list[dict[str, Any]] = []
    infractions = record.get("infractions", {})
    for collision_kind in COLLISION_KEYS:
        for collision_index, message in enumerate(infractions.get(collision_kind, [])):
            event = parse_collision_message(message)
            event.update({"collision_kind": collision_kind, "collision_index": collision_index})
            events.append(event)
    return events


def prediction_paths(route_dir: Path) -> list[Path]:
    return sorted((route_dir / "sensor_data" / "mode_prediction").glob("*.json"), key=frame_sort_key)


def prediction_frame_ids(route_dir: Path) -> list[str]:
    return [path.stem for path in prediction_paths(route_dir)]


def load_route_records(route_dir: Path) -> dict[str, Any] | None:
    records_path = route_dir / "records.json.gz"
    if not records_path.exists():
        return None
    return load_json_gz(records_path)


def estimate_collision_state_index(route_dir: Path, event: dict[str, Any]) -> tuple[int | None, str, float | None]:
    route_records = load_route_records(route_dir)
    if not route_records:
        return None, "missing_records", None

    states = route_records.get("states", [])
    if not states:
        return None, "empty_records", None

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

            try:
                actor_matches = object_id is not None and int(actor_id) == int(object_id)
            except (TypeError, ValueError):
                actor_matches = False

            if collision_x is None or collision_y is None:
                if actor_matches:
                    return state_idx, "object_id", None
                continue

            distance = math.hypot(float(pos[0]) - float(collision_x), float(pos[1]) - float(collision_y))
            if actor_matches:
                if distance < best_distance:
                    best_idx = state_idx
                    best_distance = distance
                    best_source = "object_id_and_position"
            elif best_source != "object_id_and_position" and distance < best_distance:
                best_idx = state_idx
                best_distance = distance
                best_source = "nearest_position"

    if best_idx is None:
        return None, best_source, None
    return best_idx, best_source, best_distance if math.isfinite(best_distance) else None


def state_index_to_frame(state_idx: int, n_states: int, frame_ids: list[str]) -> int | None:
    if not frame_ids:
        return None
    if n_states <= 1:
        return int(frame_ids[-1])

    frame_pos = round((state_idx / (n_states - 1)) * (len(frame_ids) - 1))
    frame_pos = max(0, min(len(frame_ids) - 1, frame_pos))
    return int(frame_ids[frame_pos])


def collision_frames(
    route_dir: Path,
    record: dict[str, Any] | None,
    frame_ids: list[str],
    max_location_distance: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    if not record:
        return [], []

    resolved: list[dict[str, Any]] = []
    meta = record.get("meta", {})

    data_frame = meta.get("collision_data_frame")
    if data_frame is not None:
        resolved.append({"frame": int(data_frame), "source": "meta.collision_data_frame", "location_distance": None})

    collision_frame = meta.get("collision_frame")
    if collision_frame is not None:
        collision_frame = int(collision_frame)
        frame_ids_set = set(frame_ids)
        if f"{collision_frame:04}" in frame_ids_set:
            resolved.append({"frame": collision_frame, "source": "meta.collision_frame", "location_distance": None})

    route_records = load_route_records(route_dir)
    n_states = len(route_records.get("states", [])) if route_records else 0
    for event in collision_events_from_record(record):
        state_idx, source, distance = estimate_collision_state_index(route_dir, event)
        if state_idx is None:
            resolved.append({"frame": None, "source": source, "location_distance": distance})
            continue
        if distance is not None and distance > max_location_distance:
            resolved.append({"frame": None, "source": f"{source}_too_far", "location_distance": distance})
            continue

        frame = state_index_to_frame(state_idx, n_states, frame_ids)
        resolved.append({"frame": frame, "source": source, "location_distance": distance})

    by_frame: dict[int, dict[str, Any]] = {}
    unresolved = []
    for event in resolved:
        frame = event.get("frame")
        if frame is None:
            unresolved.append(event)
            continue
        by_frame[int(frame)] = event

    return sorted(by_frame), sorted(by_frame.values(), key=lambda item: int(item["frame"])) + unresolved


def prediction_label(prediction: dict[str, Any]) -> str:
    label = prediction.get("predicted_label")
    if isinstance(label, str):
        label = label.lower()
        if label in {"safe", "unsafe"}:
            return label

    class_names = prediction.get("class_names", [])
    predicted_class = prediction.get("predicted_class")
    if isinstance(class_names, list) and predicted_class is not None:
        try:
            return str(class_names[int(predicted_class)]).lower()
        except (IndexError, TypeError, ValueError):
            pass

    unsafe_prob = prediction.get("unsafe_probability")
    safe_prob = prediction.get("safe_probability")
    if unsafe_prob is not None and safe_prob is not None:
        return "unsafe" if float(unsafe_prob) >= float(safe_prob) else "safe"

    raise ValueError(f"Cannot determine safety label from prediction: {prediction}")


def empty_counts() -> dict[str, int]:
    return {
        "total": 0,
        "correct": 0,
        "safe_correct": 0,
        "unsafe_correct": 0,
        "false_safe": 0,
        "false_unsafe": 0,
        "pred_safe": 0,
        "pred_unsafe": 0,
        "collision_within_horizon": 0,
        "no_collision_within_horizon": 0,
        "skipped_after_collision": 0,
    }


def add_counts(counts: dict[str, int], pred: str, collision_in_horizon: bool) -> bool:
    counts["total"] += 1
    counts["pred_safe" if pred == "safe" else "pred_unsafe"] += 1
    counts["collision_within_horizon" if collision_in_horizon else "no_collision_within_horizon"] += 1

    correct = (pred == "unsafe" and collision_in_horizon) or (pred == "safe" and not collision_in_horizon)
    if correct:
        counts["correct"] += 1
        counts["unsafe_correct" if pred == "unsafe" else "safe_correct"] += 1
    elif pred == "safe":
        counts["false_safe"] += 1
    else:
        counts["false_unsafe"] += 1
    return correct


def accuracy(counts: dict[str, int]) -> float | None:
    return None if counts["total"] == 0 else counts["correct"] / counts["total"]


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def class_metrics(counts: dict[str, int]) -> dict[str, float | None]:
    return {
        "safe_precision": ratio(counts["safe_correct"], counts["pred_safe"]),
        "safe_recall": ratio(counts["safe_correct"], counts["no_collision_within_horizon"]),
        "unsafe_precision": ratio(counts["unsafe_correct"], counts["pred_unsafe"]),
        "unsafe_recall": ratio(counts["unsafe_correct"], counts["collision_within_horizon"]),
    }


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100.0:.2f}%"


def route_dirs(exp: Path, selected_routes: list[str] | None) -> list[Path]:
    selected = set(selected_routes or [])
    dirs = []
    for path in sorted(exp.iterdir()):
        if not path.is_dir():
            continue
        if selected and path.name not in selected:
            continue
        if (path / "sensor_data" / "mode_prediction").is_dir():
            dirs.append(path)
    return dirs


def evaluate_route(
    route_dir: Path,
    record: dict[str, Any] | None,
    horizon_frames: int,
    max_location_distance: float,
    skip_after_collision: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame_ids = prediction_frame_ids(route_dir)
    frames, collision_resolution = collision_frames(route_dir, record, frame_ids, max_location_distance)
    first_collision_frame = min(frames) if frames else None
    counts = empty_counts()
    rows = []

    for path in prediction_paths(route_dir):
        frame = int(path.stem)
        if skip_after_collision and first_collision_frame is not None and frame > first_collision_frame:
            counts["skipped_after_collision"] += 1
            continue

        pred = prediction_label(load_json(path))
        collision_in_horizon = any(0 <= collision_frame - frame <= horizon_frames for collision_frame in frames)
        correct = add_counts(counts, pred, collision_in_horizon)
        next_collision_frame = min(
            (collision_frame for collision_frame in frames if collision_frame >= frame),
            default=None,
        )
        rows.append({
            "route": route_dir.name,
            "frame": frame,
            "predicted_label": pred,
            "collision_within_horizon": collision_in_horizon,
            "correct": correct,
            "next_collision_frame": next_collision_frame,
            "frames_to_collision": None if next_collision_frame is None else next_collision_frame - frame,
        })

    summary = {
        "route": route_dir.name,
        "accuracy": accuracy(counts),
        **class_metrics(counts),
        "collision_frames": frames,
        "first_collision_frame": first_collision_frame,
        "collision_resolution": collision_resolution,
        **counts,
    }
    return summary, rows


def print_summary(overall: dict[str, int], horizon_sec: float, fps: float, horizon_frames: int) -> None:
    acc = accuracy(overall)
    metrics = class_metrics(overall)
    print(f"Horizon: {horizon_sec:.3g}s ({horizon_frames} frames at {fps:.3g} FPS)")
    print(f"Frames evaluated: {overall['total']}")
    print(f"Frames skipped after collision: {overall['skipped_after_collision']}")
    print(f"Accuracy: {0.0 if acc is None else acc * 100.0:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"Safe precision: {format_percent(metrics['safe_precision'])}")
    print(f"Safe recall: {format_percent(metrics['safe_recall'])}")
    print(f"Unsafe precision: {format_percent(metrics['unsafe_precision'])}")
    print(f"Unsafe recall: {format_percent(metrics['unsafe_recall'])}")
    print(f"Predicted safe correct, no collision within horizon: {overall['safe_correct']}")
    print(f"Predicted unsafe correct, collision within horizon: {overall['unsafe_correct']}")
    print(f"False safe (missed collision in horizon): {overall['false_safe']}")
    print(f"False unsafe (predicted collision, none in horizon): {overall['false_unsafe']}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    exp = args.exp.expanduser().resolve()
    debug_json = args.debug_json.expanduser().resolve() if args.debug_json else exp / "debug_results.json"

    if not exp.is_dir():
        print(f"Experiment folder does not exist: {exp}")
        return 1
    if args.fps <= 0:
        print("--fps must be positive")
        return 1
    if args.horizon_sec < 0:
        print("--horizon-sec must be non-negative")
        return 1

    horizon_frames = int(math.ceil(args.horizon_sec * args.fps))
    debug_records = load_debug_records(debug_json)
    summaries = []
    all_rows = []
    overall = empty_counts()
    skip_after_collision = not args.include_after_collision

    for route_dir in route_dirs(exp, args.route):
        record = route_result_record(route_dir, debug_records)
        summary, rows = evaluate_route(
            route_dir,
            record,
            horizon_frames,
            args.max_location_distance,
            skip_after_collision,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        for key in overall:
            overall[key] += summary[key]

    if not summaries:
        print(f"No route folders with sensor_data/mode_prediction found under {exp}")
        return 1

    print_summary(overall, args.horizon_sec, args.fps, horizon_frames)
    print(f"Routes evaluated: {len(summaries)}")

    if args.show_routes:
        for summary in summaries:
            route_acc = summary["accuracy"]
            print(
                f"{summary['route']}: {format_percent(route_acc)} "
                f"({summary['correct']}/{summary['total']}), "
                f"safe P/R={format_percent(summary['safe_precision'])}/{format_percent(summary['safe_recall'])}, "
                f"unsafe P/R={format_percent(summary['unsafe_precision'])}/{format_percent(summary['unsafe_recall'])}, "
                f"collisions={summary['collision_frames']}, "
                f"skipped_after_collision={summary['skipped_after_collision']}"
            )

    if args.csv:
        write_csv(args.csv.expanduser().resolve(), all_rows)
        print(f"Wrote per-frame CSV: {args.csv}")

    if args.json:
        output = {
            "experiment": str(exp),
            "horizon_sec": args.horizon_sec,
            "fps": args.fps,
            "horizon_frames": horizon_frames,
            "skip_after_collision": skip_after_collision,
            "overall": {"accuracy": accuracy(overall), **class_metrics(overall), **overall},
            "routes": summaries,
        }
        json_path = args.json.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as outfile:
            json.dump(output, outfile, indent=2)
        print(f"Wrote JSON summary: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
