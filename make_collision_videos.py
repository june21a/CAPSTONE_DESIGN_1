#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


BASE = Path("/workspace/CAPSTONE_DESIGN_1/carla_garage/results/temp")
DEBUG_JSON = BASE / "debug_results.json"
OUT_ROOT = BASE / "temp_collisions_file"
FPS = 10

COLLISION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
)


def first_png_number(directory: Path) -> int:
    pngs = sorted(directory.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {directory}")
    return int(pngs[0].stem)


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def make_sequence_video(src_dir: Path, start: int, out_file: Path) -> None:
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-start_number",
            str(start),
            "-i",
            str(src_dir / "%04d.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_file),
        ]
    )


def make_side_by_side_video(model_dir: Path, attention_dir: Path, start: int, out_file: Path) -> None:
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-start_number",
            str(start),
            "-i",
            str(model_dir / "%04d.png"),
            "-framerate",
            str(FPS),
            "-start_number",
            str(start),
            "-i",
            str(attention_dir / "%04d.png"),
            "-filter_complex",
            (
                "[0:v]scale=-2:720[left];"
                "[1:v]scale=-2:720[right];"
                "[left][right]hstack=inputs=2,format=yuv420p[v]"
            ),
            "-map",
            "[v]",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_file),
        ]
    )


def main() -> None:
    data = json.loads(DEBUG_JSON.read_text())
    records = data["_checkpoint"]["records"]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    collision_records = []
    for record in records:
        infractions = record.get("infractions", {})
        if any(infractions.get(key) for key in COLLISION_KEYS):
            collision_records.append(record)

    print(f"collision routes: {len(collision_records)}")

    summary = []
    for record in collision_records:
        timestamp = record["timestamp"]
        route_dir = BASE / timestamp
        model_dir = route_dir / "model_results"
        attention_dir = route_dir / "sensor_data" / "attention_overlay"
        out_dir = OUT_ROOT / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)

        model_start = first_png_number(model_dir)
        attention_start = first_png_number(attention_dir)
        start = max(model_start, attention_start)

        model_out = out_dir / "model_results.mp4"
        attention_out = out_dir / "attention_overlay.mp4"
        combined_out = out_dir / "model_attention_side_by_side.mp4"

        print(f"[route {record['index']}] {timestamp}")
        make_sequence_video(model_dir, start, model_out)
        make_sequence_video(attention_dir, start, attention_out)
        make_side_by_side_video(model_dir, attention_dir, start, combined_out)

        summary.append(
            {
                "index": record["index"],
                "timestamp": timestamp,
                "route_id": record.get("route_id"),
                "collision_counts": {
                    key: len(record.get("infractions", {}).get(key, []))
                    for key in COLLISION_KEYS
                },
                "outputs": {
                    "model_results": str(model_out),
                    "attention_overlay": str(attention_out),
                    "side_by_side": str(combined_out),
                },
            }
        )

    (OUT_ROOT / "collision_videos_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved outputs under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
