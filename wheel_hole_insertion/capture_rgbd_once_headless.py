# -*- coding: utf-8 -*-
"""Capture one aligned RealSense RGBD sample without any GUI windows."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_loader import CONFIG, cfg_get, relative_path  # noqa: E402


def post_json(url, payload, timeout_s=5.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


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
    parser.add_argument("--serial", default=cfg_get(CONFIG, "camera", "serial", default=""))
    parser.add_argument("--out", default=str(relative_path(CONFIG, "paths", "captures_dir", default="captures")))
    parser.add_argument("--width", type=int, default=int(cfg_get(CONFIG, "camera", "width", default=1280)))
    parser.add_argument("--height", type=int, default=int(cfg_get(CONFIG, "camera", "height", default=720)))
    parser.add_argument("--fps", type=int, default=int(cfg_get(CONFIG, "camera", "fps", default=30)))
    parser.add_argument("--warmup", type=int, default=int(cfg_get(CONFIG, "camera", "warmup", default=30)))
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--camera-service-url", default=str(cfg_get(CONFIG, "camera_service", "url", default="http://127.0.0.1:8099")))
    parser.add_argument("--camera-role", default=str(cfg_get(CONFIG, "camera_service", "default_role", default="right_arm")))
    parser.add_argument("--camera-service-timeout-s", type=float, default=float(cfg_get(CONFIG, "camera_service", "timeout_s", default=5.0)))
    parser.add_argument("--direct-realsense", action="store_true",
                        help="Bypass camera_service and open RealSense directly. Use only for debugging.")
    args = parser.parse_args()

    if not args.direct_realsense:
        out_root = Path(args.out).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        response = post_json(
            args.camera_service_url.rstrip("/") + "/camera/capture",
            {
                "role": args.camera_role,
                "out": str(out_root),
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "warmup": args.warmup,
            },
            timeout_s=max(5.0, float(args.camera_service_timeout_s)),
        )
        capture_dir = response.get("capture_dir", "")
        if not capture_dir and isinstance(response.get("body"), dict):
            capture_dir = response["body"].get("capture_dir", "")
        if not capture_dir:
            raise RuntimeError(f"camera_service did not return capture_dir: {response}")
        print(Path(capture_dir).expanduser().resolve())
        return

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
