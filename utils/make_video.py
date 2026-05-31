#!/usr/bin/env python3
"""Create a video from one image folder, or a side-by-side video from two folders."""

import argparse
import os
from glob import glob

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create an MP4 from PNG frames.")
    parser.add_argument("folder1", help="First image folder.")
    parser.add_argument(
        "folder2",
        nargs="?",
        default=None,
        help="Optional second image folder. If provided, frames are concatenated side by side.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.mp4",
        help="Output video path. Defaults to output.mp4.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frames per second. Defaults to 20.",
    )
    return parser.parse_args()


def frame_sort_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return stem


def get_sorted_images(folder):
    if folder is None:
        return []
    return sorted(glob(os.path.join(folder, "*.png")), key=frame_sort_key)


def resize_keep_ratio(img, target_height):
    h, w = img.shape[:2]
    new_w = int(w * target_height / h)
    return cv2.resize(img, (new_w, target_height))


def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def make_video(folder1, folder2=None, output_video="output.mp4", fps=20.0):
    if cv2 is None:
        raise SystemExit("OpenCV is required. Run this with the same Python environment used for CARLA evaluation.")

    images1 = get_sorted_images(folder1)
    images2 = get_sorted_images(folder2)

    if not images1:
        raise ValueError(f"No PNG images found in folder1: {folder1}")

    side_by_side = folder2 is not None
    if side_by_side and not images2:
        raise ValueError(f"No PNG images found in folder2: {folder2}")

    num_frames = min(len(images1), len(images2)) if side_by_side else len(images1)
    if num_frames == 0:
        raise ValueError("No frames to write.")

    first_frame1 = read_image(images1[0])
    if side_by_side:
        first_frame2 = read_image(images2[0])
        height = min(first_frame1.shape[0], first_frame2.shape[0])
        frame1 = resize_keep_ratio(first_frame1, height)
        frame2 = resize_keep_ratio(first_frame2, height)
        video_width = frame1.shape[1] + frame2.shape[1]
        video_height = height
    else:
        video_height, video_width = first_frame1.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (video_width, video_height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_video}")

    try:
        for i in range(num_frames):
            frame1 = read_image(images1[i])

            if side_by_side:
                frame2 = read_image(images2[i])
                frame1 = resize_keep_ratio(frame1, video_height)
                frame2 = resize_keep_ratio(frame2, video_height)
                frame = np.hstack((frame1, frame2))
            else:
                if frame1.shape[1] != video_width or frame1.shape[0] != video_height:
                    frame = cv2.resize(frame1, (video_width, video_height))
                else:
                    frame = frame1

            writer.write(frame)
            print(f"Processed frame {i + 1}/{num_frames}")
    finally:
        writer.release()

    print(f"Saved video: {output_video}")


def main():
    args = parse_arguments()
    make_video(args.folder1, args.folder2, args.output, args.fps)


if __name__ == "__main__":
    main()
