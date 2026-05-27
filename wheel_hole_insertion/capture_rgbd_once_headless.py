# -*- coding: utf-8 -*-
"""Capture one aligned RealSense RGBD sample without any GUI windows."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


SCRIPT_DIR = Path(__file__).resolve().parent


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


def make_output_dir(root):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root).expanduser().resolve() / f"wrist_rgbd_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "captures"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()

    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError("No RealSense device found.")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(max(0, int(args.warmup))):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to get aligned color/depth frames.")

        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        color = np.asanyarray(color_frame.get_data())
        depth_z16 = np.asanyarray(depth_frame.get_data())
        depth_m = depth_z16.astype(np.float32) * depth_scale
        depth_mm = np.rint(depth_m * 1000.0).astype(np.uint16)

        depth_vis = cv2.convertScaleAbs(depth_z16, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        alpha = min(max(float(args.alpha), 0.0), 1.0)
        overlay = cv2.addWeighted(color, alpha, depth_colormap, 1.0 - alpha, 0)

        out_dir = make_output_dir(args.out)
        cv2.imwrite(str(out_dir / "color.png"), color)
        cv2.imwrite(str(out_dir / "depth_mm.png"), depth_mm)
        np.save(str(out_dir / "depth_raw.npy"), depth_m)
        cv2.imwrite(str(out_dir / "depth_colormap.png"), depth_colormap)
        cv2.imwrite(str(out_dir / "depth_overlay.png"), overlay)

        color_intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        depth_intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        meta = {
            "capture_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "selected_serial": args.serial,
            "devices": devices,
            "depth_scale_m_per_unit": depth_scale,
            "streams_aligned_to": "color",
            "color_intrinsics": intrinsics_to_dict(color_intr),
            "depth_intrinsics_after_align": intrinsics_to_dict(depth_intr),
            "files": {
                "color": "color.png",
                "depth_mm": "depth_mm.png",
                "depth_raw_m": "depth_raw.npy",
                "depth_colormap": "depth_colormap.png",
                "depth_overlay": "depth_overlay.png",
            },
        }
        with open(out_dir / "intrinsics.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(out_dir)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
