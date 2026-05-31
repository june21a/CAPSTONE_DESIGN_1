#!/usr/bin/env python3
"""Restore visualization PNGs from saved NPZ files for one route folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit("OpenCV is required. Run this with the same Python environment used for CARLA evaluation.") from exc


SEMANTIC_PALETTE_BGR = np.array(
    [
        [0, 0, 0],
        [250, 170, 30],
        [200, 200, 200],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)

PRED_SEMANTIC_PALETTE_RGB = np.array(
    [
        [0, 0, 0],
        [70, 70, 70],
        [100, 40, 40],
        [55, 90, 80],
        [220, 20, 60],
        [153, 153, 153],
        [157, 234, 50],
    ],
    dtype=np.uint8,
)

BEV_PALETTE_BGR = np.array(
    [
        [0, 0, 0],
        [200, 200, 200],
        [255, 255, 255],
        [255, 255, 0],
        [50, 234, 157],
        [160, 160, 0],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0],
        [250, 170, 30],
        [0, 255, 0],
    ],
    dtype=np.uint8,
)

DETECTION_COLORS_BGR = [
    np.array([0, 165, 255], dtype=np.uint8),
    np.array([0, 255, 0], dtype=np.uint8),
    np.array([0, 0, 255], dtype=np.uint8),
    np.array([160, 160, 250], dtype=np.uint8),
    np.array([133, 133, 16], dtype=np.uint8),
]

MIN_X = -32.0
MAX_X = 32.0
MIN_Y = -32.0
MAX_Y = 32.0
PIXELS_PER_METER = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate visualization PNGs from NPZ files in a CARLA route result folder."
    )
    parser.add_argument(
        "route_folder",
        type=Path,
        help="Route folder, or its sensor_data folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files. By default only missing PNGs are restored.",
    )
    return parser.parse_args()


def sensor_data_root(route_folder: Path) -> Path:
    route_folder = route_folder.expanduser().resolve()
    if route_folder.name == "sensor_data":
        return route_folder
    return route_folder / "sensor_data"


def should_write(path: Path, overwrite: bool) -> bool:
    return overwrite or not path.exists()


def write_image(path: Path, image: np.ndarray, overwrite: bool) -> int:
    if not should_write(path, overwrite):
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    return 1


def load_npz_array(path: Path, key: str) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            return None
        return data[key]


def colorize_classes(indices: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[np.clip(indices.astype(np.int64), 0, len(palette) - 1)]


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - float(np.min(values))
    max_value = float(np.max(values))
    if max_value > 1e-6:
        values = values / max_value
    return np.clip(values * 255.0, 0, 255).astype(np.uint8)


def read_rgb(sensor_root: Path, frame_id: str) -> np.ndarray | None:
    rgb_path = sensor_root / "rgb" / f"{frame_id}.png"
    if not rgb_path.exists():
        return None
    return cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)


def restore_semantic_predictions(sensor_root: Path, overwrite: bool) -> int:
    count = 0
    semantic_dir = sensor_root / "vision_tasks" / "semantic"
    for npz_path in sorted(semantic_dir.glob("*.npz")):
        frame_id = npz_path.stem
        output_path = semantic_dir / f"{frame_id}.png"
        if not should_write(output_path, overwrite):
            continue
        semantic = load_npz_array(npz_path, "semantic")
        if semantic is None:
            continue
        semantic_rgb = colorize_classes(semantic, PRED_SEMANTIC_PALETTE_RGB)
        semantic_bgr = cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR)
        rgb = read_rgb(sensor_root, frame_id)
        if rgb is not None:
            semantic_bgr = cv2.resize(semantic_bgr, dsize=(rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            semantic_bgr = cv2.addWeighted(rgb, 0.55, semantic_bgr, 0.45, 0)
        count += write_image(output_path, semantic_bgr, overwrite)
    return count


def restore_gt_semantics(sensor_root: Path, overwrite: bool) -> int:
    count = 0
    semantic_dir = sensor_root / "vision_tasks_gt" / "semantic"
    for npz_path in sorted(semantic_dir.glob("*.npz")):
        frame_id = npz_path.stem
        semantic = load_npz_array(npz_path, "semantic")
        if semantic is None:
            continue
        count += write_image(semantic_dir / f"{frame_id}.png", colorize_classes(semantic, SEMANTIC_PALETTE_BGR), overwrite)
    return count


def restore_bev_semantics(sensor_root: Path, overwrite: bool) -> int:
    count = 0
    for bev_dir in (sensor_root / "vision_tasks" / "bev_semantic", sensor_root / "vision_tasks_gt" / "bev_semantic"):
        for npz_path in sorted(bev_dir.glob("*.npz")):
            frame_id = npz_path.stem
            bev = load_npz_array(npz_path, "bev_semantic")
            if bev is None:
                continue
            count += write_image(bev_dir / f"{frame_id}.png", colorize_classes(bev, BEV_PALETTE_BGR), overwrite)
    return count


def restore_depth(sensor_root: Path, overwrite: bool) -> int:
    count = 0
    for depth_dir in (sensor_root / "vision_tasks" / "depth", sensor_root / "vision_tasks_gt" / "depth"):
        for npz_path in sorted(depth_dir.glob("*.npz")):
            frame_id = npz_path.stem
            depth = load_npz_array(npz_path, "depth")
            if depth is None:
                continue
            if depth.dtype == np.uint8:
                depth_uint8 = depth
            else:
                depth_uint8 = normalize_to_uint8(depth)
            image = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
            count += write_image(depth_dir / f"{frame_id}.png", image, overwrite)
    return count


def restore_rgb_attention(sensor_root: Path, overwrite: bool) -> int:
    count = 0
    attention_dir = sensor_root / "attention"
    overlay_dir = sensor_root / "attention_overlay"
    for npz_path in sorted(attention_dir.glob("*.npz")):
        frame_id = npz_path.stem
        output_path = overlay_dir / f"{frame_id}.png"
        if not should_write(output_path, overwrite):
            continue
        attention = load_npz_array(npz_path, "attention")
        rgb = read_rgb(sensor_root, frame_id)
        if attention is None or rgb is None:
            continue
        attention_uint8 = normalize_to_uint8(attention)
        attention_uint8 = cv2.resize(attention_uint8, dsize=(rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
        count += write_image(output_path, cv2.addWeighted(rgb, 0.7, heatmap, 0.3, 0), overwrite)
    return count


def lidar_bev_to_image(lidar_bev: np.ndarray) -> np.ndarray:
    lidar_map = np.asarray(lidar_bev)
    lidar_map = np.squeeze(lidar_map)
    if lidar_map.ndim == 3:
        lidar_map = lidar_map[0]
    lidar_image = 255 - np.clip(lidar_map * 255.0, 0, 255).astype(np.uint8)
    return np.stack([lidar_image, lidar_image, lidar_image], axis=-1)


def restore_lidar_attention(sensor_root: Path, overwrite: bool) -> int:
    count = 0
    attention_dir = sensor_root / "lidar_attention"
    overlay_dir = sensor_root / "lidar_attention_overlay"
    for npz_path in sorted(attention_dir.glob("*.npz")):
        frame_id = npz_path.stem
        overlay_path = overlay_dir / f"{frame_id}.png"
        bev_png_path = overlay_dir / f"{frame_id}_bev.png"
        if not (should_write(overlay_path, overwrite) or should_write(bev_png_path, overwrite)):
            continue
        attention = load_npz_array(npz_path, "attention")
        bev = load_npz_array(sensor_root / "lidar" / f"{frame_id}_bev.npz", "lidar_bev")
        if attention is None or bev is None:
            continue

        lidar_image = lidar_bev_to_image(bev)
        attention_uint8 = normalize_to_uint8(attention)
        attention_uint8 = cv2.resize(
            attention_uint8,
            dsize=(lidar_image.shape[1], lidar_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(lidar_image, 0.65, heatmap, 0.35, 0)
        lidar_image = cv2.resize(lidar_image, dsize=(lidar_image.shape[1] * 4, lidar_image.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.resize(overlay, dsize=(overlay.shape[1] * 4, overlay.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
        lidar_image = np.ascontiguousarray(np.rot90(lidar_image, k=1), dtype=np.uint8)
        overlay = np.ascontiguousarray(np.rot90(overlay, k=1), dtype=np.uint8)
        count += write_image(bev_png_path, lidar_image, overwrite)
        count += write_image(overlay_path, overlay, overwrite)
    return count


def box_to_pixels(box: np.ndarray, image_shape: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float], float]:
    height, width = image_shape
    x, y, extent_x, extent_y, yaw = [float(value) for value in box[:5]]
    metric_limit = max(abs(MIN_X), abs(MAX_X), abs(MIN_Y), abs(MAX_Y))
    if abs(x) <= metric_limit * 2.0 and abs(y) <= metric_limit * 2.0:
        center_x = (y - MIN_Y) * PIXELS_PER_METER
        center_y = (MAX_X - x) * PIXELS_PER_METER
        size = (max(1.0, extent_y * 2.0 * PIXELS_PER_METER), max(1.0, extent_x * 2.0 * PIXELS_PER_METER))
        angle = -np.degrees(yaw)
    else:
        center_x = x * width / 1024.0
        center_y = y * height / 1024.0
        size = (max(1.0, extent_x * width / 1024.0), max(1.0, extent_y * height / 1024.0))
        angle = np.degrees(yaw)
    return (center_x, center_y), size, angle


def restore_detection(sensor_root: Path, overwrite: bool) -> int:
    count = 0
    detection_dir = sensor_root / "vision_tasks" / "detection"
    for npz_path in sorted(detection_dir.glob("*.npz")):
        frame_id = npz_path.stem
        output_path = detection_dir / f"{frame_id}.png"
        if not should_write(output_path, overwrite):
            continue
        boxes = load_npz_array(npz_path, "boxes")
        bev = load_npz_array(sensor_root / "lidar" / f"{frame_id}_bev.npz", "lidar_bev")
        if boxes is None or bev is None:
            continue

        image = lidar_bev_to_image(bev)
        image = cv2.resize(image, dsize=(image.shape[1] * 4, image.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
        for box in np.asarray(boxes):
            if len(box) < 8:
                continue
            class_id = int(box[7])
            color = DETECTION_COLORS_BGR[class_id % len(DETECTION_COLORS_BGR)].tolist()
            rect = box_to_pixels(box, image.shape[:2])
            points = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        image = np.ascontiguousarray(np.rot90(image, k=1), dtype=np.uint8)
        count += write_image(output_path, image, overwrite)
    return count


def restore_2d_box_overlays(sensor_root: Path, overwrite: bool) -> int:
    count = 0
    box_dir = sensor_root / "vision_tasks_gt" / "2d_box"
    overlay_dir = sensor_root / "vision_tasks_gt" / "2d_box_overlay"
    colors = {
        "car": (0, 165, 255),
        "walker": (0, 255, 0),
        "traffic_light": (0, 0, 255),
        "stop_sign": (160, 160, 250),
        "emergency_vehicle": (133, 133, 16),
    }
    for json_path in sorted(box_dir.glob("*.json")):
        frame_id = json_path.stem
        output_path = overlay_dir / f"{frame_id}.png"
        if not should_write(output_path, overwrite):
            continue
        image = read_rgb(sensor_root, frame_id)
        if image is None:
            continue
        with open(json_path, "r", encoding="utf-8") as infile:
            payload = json.load(infile)
        for box in payload.get("boxes", []):
            x_min = int(box.get("x_min", box.get("xmin", 0)))
            y_min = int(box.get("y_min", box.get("ymin", 0)))
            x_max = int(box.get("x_max", box.get("xmax", 0)))
            y_max = int(box.get("y_max", box.get("ymax", 0)))
            color = colors.get(box.get("class"), (255, 255, 255))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"{box.get('class', 'box')}:{box.get('id', '')}"
            cv2.putText(image, label, (x_min, max(0, y_min - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        count += write_image(output_path, image, overwrite)
    return count


def main() -> int:
    args = parse_args()
    sensor_root = sensor_data_root(args.route_folder)
    if not sensor_root.is_dir():
        print(f"sensor_data folder does not exist: {sensor_root}")
        return 1

    counts = {
        "semantic": restore_semantic_predictions(sensor_root, args.overwrite),
        "gt_semantic": restore_gt_semantics(sensor_root, args.overwrite),
        "bev_semantic": restore_bev_semantics(sensor_root, args.overwrite),
        "depth": restore_depth(sensor_root, args.overwrite),
        "rgb_attention": restore_rgb_attention(sensor_root, args.overwrite),
        "lidar_attention": restore_lidar_attention(sensor_root, args.overwrite),
        "detection": restore_detection(sensor_root, args.overwrite),
        "2d_box_overlay": restore_2d_box_overlays(sensor_root, args.overwrite),
    }

    for name, count in counts.items():
        print(f"{name}: restored {count} image(s)")
    print(f"Total restored: {sum(counts.values())} image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
