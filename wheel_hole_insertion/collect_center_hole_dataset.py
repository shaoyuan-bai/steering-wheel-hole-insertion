# -*- coding: utf-8 -*-
"""Collect wrist-camera images for center-hole segmentation labeling.

Interactive mode:
  - Shows the RGB stream.
  - Press p to save the current frame.
  - Press q or Esc to quit.

Headless mode:
  - Use --headless --count N --interval S to save images over SSH.
  - Use --headless --count 1 for one capture.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SERIAL = "405622075930"


PROFILE_CANDIDATES = [
    (1920, 1080, 30),
    (1280, 720, 30),
    (848, 480, 30),
    (640, 480, 30),
]


def intrinsics_to_dict(intr):
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "model": str(intr.model),
        "coeffs": [float(x) for x in intr.coeffs],
    }


def list_realsense_devices():
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        devices.append({
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
        })
    return devices


def make_config(serial, width, height, fps, save_depth):
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if save_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    return config


def start_pipeline(serial, requested_width, requested_height, requested_fps, save_depth):
    candidates = [(requested_width, requested_height, requested_fps)]
    for profile in PROFILE_CANDIDATES:
        if profile not in candidates:
            candidates.append(profile)

    last_error = None
    for width, height, fps in candidates:
        pipeline = rs.pipeline()
        config = make_config(serial, width, height, fps, save_depth)
        try:
            profile = pipeline.start(config)
            return pipeline, profile, width, height, fps
        except Exception as exc:
            last_error = exc
            try:
                pipeline.stop()
            except Exception:
                pass
    raise RuntimeError(f"Failed to start RealSense stream. Last error: {last_error}")


def set_sensor_options(profile, args):
    device = profile.get_device()
    try:
        color_sensor = device.query_sensors()[1]
    except Exception:
        return

    options = [
        (rs.option.enable_auto_exposure, 1.0 if args.auto_exposure else 0.0),
        (rs.option.sharpness, float(args.sharpness)),
        (rs.option.contrast, float(args.contrast)),
        (rs.option.saturation, float(args.saturation)),
    ]
    for option, value in options:
        try:
            if color_sensor.supports(option):
                color_sensor.set_option(option, value)
        except Exception:
            pass


def ensure_dirs(root, save_depth):
    root = Path(root).expanduser().resolve()
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = root / "depth_m"
    meta_dir = root / "meta"
    if save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
    return root, images_dir, depth_dir, meta_dir


def make_frame_name(prefix, index):
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{index:05d}"


def save_sample(color_frame, depth_frame, profile, devices, out_dirs, args, index):
    root, images_dir, depth_dir, meta_dir = out_dirs
    stem = make_frame_name(args.prefix, index)
    color = np.asanyarray(color_frame.get_data())
    color_path = images_dir / f"{stem}.png"
    cv2.imwrite(str(color_path), color)

    saved = {"image": str(color_path)}
    if args.save_depth and depth_frame:
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        depth_z16 = np.asanyarray(depth_frame.get_data())
        depth_m = depth_z16.astype(np.float32) * depth_scale
        depth_path = depth_dir / f"{stem}.npy"
        np.save(str(depth_path), depth_m)
        color_intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        depth_intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        meta = {
            "capture_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "devices": devices,
            "selected_serial": args.serial,
            "image": str(color_path.relative_to(root)),
            "depth_m": str(depth_path.relative_to(root)),
            "depth_scale_m_per_unit": depth_scale,
            "streams_aligned_to": "color",
            "color_intrinsics": intrinsics_to_dict(color_intr),
            "depth_intrinsics_after_align": intrinsics_to_dict(depth_intr),
        }
        meta_path = meta_dir / f"{stem}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        saved["depth"] = str(depth_path)
        saved["meta"] = str(meta_path)
    print(f"[SAVE] {saved['image']}")
    return saved


def draw_status(color, saved_count, args, width, height, fps):
    view = color.copy()
    lines = [
        f"{width}x{height}@{fps}  saved={saved_count}",
        "p: save   q/esc: quit",
        f"out: {Path(args.out).expanduser()}",
    ]
    for idx, line in enumerate(lines):
        y = 28 + idx * 28
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 1, cv2.LINE_AA)
    return view


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--out", default=str(SCRIPT_DIR / "label_dataset"))
    parser.add_argument("--prefix", default="center_hole")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--save-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--count", type=int, default=0, help="Headless captures. 0 means run until Ctrl-C.")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--auto-exposure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sharpness", type=float, default=80.0)
    parser.add_argument("--contrast", type=float, default=50.0)
    parser.add_argument("--saturation", type=float, default=64.0)
    return parser.parse_args()


def main():
    args = parse_args()
    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError("No RealSense device found.")

    out_dirs = ensure_dirs(args.out, args.save_depth)
    pipeline, profile, width, height, fps = start_pipeline(
        serial=args.serial,
        requested_width=args.width,
        requested_height=args.height,
        requested_fps=args.fps,
        save_depth=args.save_depth,
    )
    set_sensor_options(profile, args)
    align = rs.align(rs.stream.color) if args.save_depth else None

    saved_count = 0
    last_save = 0.0
    try:
        for _ in range(max(0, int(args.warmup))):
            pipeline.wait_for_frames()

        print(f"[INFO] stream: {width}x{height}@{fps}, headless={args.headless}")
        print(f"[INFO] output: {out_dirs[0]}")
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if args.save_depth else None
            if not color_frame:
                continue

            if args.headless:
                now = time.time()
                if now - last_save >= max(0.0, float(args.interval)):
                    saved_count += 1
                    save_sample(color_frame, depth_frame, profile, devices, out_dirs, args, saved_count)
                    last_save = now
                    if int(args.count) > 0 and saved_count >= int(args.count):
                        break
                continue

            color = np.asanyarray(color_frame.get_data())
            view = draw_status(color, saved_count, args, width, height, fps)
            cv2.imshow("center_hole_dataset_capture", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                saved_count += 1
                save_sample(color_frame, depth_frame, profile, devices, out_dirs, args, saved_count)
    finally:
        pipeline.stop()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
