# -*- coding: utf-8 -*-
"""Check whether J6 rotation changes detected hole pose in base frame."""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_loader import CONFIG, cfg_get, relative_path  # noqa: E402
from detect_center_hole_yolo_rgbd import run_detection  # noqa: E402
from insert_center_hole import T_EE_CAM, get_current_state, normalize, pose_to_matrix  # noqa: E402


SCRIPT_VERSION = "2026-06-01-j6-rotation-effect-v1"
DEFAULT_PYTHON = "/home/wooshrobot/miniconda3/envs/cyy/bin/python"


def angle_deg(a, b):
    a = normalize(np.asarray(a, dtype=np.float64), "a")
    b = normalize(np.asarray(b, dtype=np.float64), "b")
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def make_detect_args(capture_dir, out_dir, out_stem, args):
    return argparse.Namespace(
        input=str(capture_dir),
        model=str(args.model),
        out_dir=str(out_dir),
        out_stem=str(out_stem),
        imgsz=int(args.imgsz),
        conf=float(args.yolo_conf),
        iou=float(args.yolo_iou),
        mask_threshold=float(args.mask_threshold),
        min_depth=0.18,
        max_depth=1.20,
        plane_inner_radius_px=20,
        plane_outer_radius_px=115,
        plane_inner_scale=1.55,
        plane_outer_scale=4.2,
        min_plane_points=500,
        plane_iters=6,
        plane_inlier_m=0.006,
        metal_max_saturation=145,
        metal_min_gray=35,
        metal_min_value=35,
        min_confidence=float(args.min_confidence),
        min_mask_area_px=60,
        max_mask_area_px=20000,
        max_plane_rmse_m=0.004,
        min_abs_normal_z=0.55,
        min_plane_inliers=1200,
        min_plane_inlier_ratio=0.10,
        normal_draw_length_m=0.08,
        overlay_alpha=0.38,
    )


def capture_once(out_root, args):
    cmd = [
        args.python,
        str(SCRIPT_DIR / "capture_rgbd_once_headless.py"),
        "--out",
        str(out_root),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--warmup",
        str(args.warmup),
    ]
    if args.serial:
        cmd.extend(["--serial", args.serial])

    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Capture command did not print output directory.")
    return Path(lines[-1]).expanduser().resolve()


def detection_to_base(pose_now, detection):
    t_base_cam = pose_to_matrix(pose_now) @ T_EE_CAM
    center_cam = np.asarray(detection["center_headcam_m"], dtype=np.float64)
    normal_cam = normalize(np.asarray(detection["normal_headcam"], dtype=np.float64), "normal_headcam")
    center_base = (t_base_cam @ np.r_[center_cam, 1.0])[:3]
    normal_base = normalize(t_base_cam[:3, :3] @ normal_cam, "normal_base")
    return center_base, normal_base


def collect_sample(label, index, sock, capture_root, detect_root, args):
    print(f"\n[{index}] {label}")
    if not args.auto:
        input("确认方向盘不动、当前姿态已就位后按 Enter 采集...")

    state = get_current_state(sock)
    if not state:
        raise RuntimeError("Failed to read current robot state.")
    pose_now, joint_now = state
    print(f"[{index}] pose: {[round(float(x), 5) for x in pose_now]}")
    print(f"[{index}] joint_deg: {[round(float(x), 3) for x in joint_now]}")

    capture_dir = capture_once(capture_root, args)
    stem = f"j6_{index:02d}_{label}"
    detect_args = make_detect_args(capture_dir, detect_root, stem, args)
    detection, json_path, overlay_path = run_detection(capture_dir, detect_args)
    center_base, normal_base = detection_to_base(pose_now, detection)

    sample = {
        "index": int(index),
        "label": label,
        "ok": True,
        "capture_dir": str(capture_dir),
        "detection_path": str(json_path),
        "overlay_path": str(overlay_path),
        "pose_now": [float(x) for x in pose_now],
        "joint_now": [float(x) for x in joint_now],
        "quality": detection.get("quality", "unknown"),
        "quality_reasons": detection.get("quality_reasons", []),
        "score": float(detection["source"]["score"]),
        "center_px": [float(x) for x in detection["center_px"]],
        "center_headcam_m": [float(x) for x in detection["center_headcam_m"]],
        "normal_headcam": [float(x) for x in detection["normal_headcam"]],
        "center_base_m": [float(x) for x in center_base],
        "normal_base": [float(x) for x in normal_base],
        "plane_rmse_m": float(detection["plane_rmse_m"]),
        "plane_points": int(detection["plane_points"]),
        "plane_inliers": int(detection["plane_inliers"]),
    }
    print(
        f"[{index}] score={sample['score']:.3f}, quality={sample['quality']}, "
        f"center_px={[round(x, 2) for x in sample['center_px']]}"
    )
    print(f"[{index}] center_base_m: {[round(float(x), 5) for x in center_base]}")
    print(f"[{index}] normal_base: {[round(float(x), 5) for x in normal_base]}")
    print(f"[{index}] overlay: {overlay_path}")
    return sample


def summarize(samples, out_dir):
    valid = [s for s in samples if s.get("ok")]
    summary = {
        "script_version": SCRIPT_VERSION,
        "samples": samples,
        "valid_count": len(valid),
    }
    if len(valid) >= 2:
        ref = valid[0]
        comparisons = []
        for sample in valid[1:]:
            dc = np.asarray(sample["center_base_m"], dtype=np.float64) - np.asarray(ref["center_base_m"], dtype=np.float64)
            normal_angle = angle_deg(sample["normal_base"], ref["normal_base"])
            j6_delta = float(sample["joint_now"][5] - ref["joint_now"][5])
            comp = {
                "from": ref["label"],
                "to": sample["label"],
                "j6_delta_deg_raw": j6_delta,
                "center_delta_m": dc.tolist(),
                "center_delta_mm": (dc * 1000.0).tolist(),
                "center_distance_mm": float(np.linalg.norm(dc) * 1000.0),
                "normal_angle_deg": float(normal_angle),
                "center_px_from": ref["center_px"],
                "center_px_to": sample["center_px"],
            }
            comparisons.append(comp)
            print("\n" + "=" * 72)
            print(f"对比: {ref['label']} -> {sample['label']}")
            print(f"J6 raw delta: {j6_delta:.2f} deg")
            print(f"center delta xyz: {[round(float(x), 2) for x in comp['center_delta_mm']]} mm")
            print(f"center distance: {comp['center_distance_mm']:.2f} mm")
            print(f"normal angle: {comp['normal_angle_deg']:.2f} deg")
            print("=" * 72)
        summary["comparisons"] = comparisons

        max_center = max(c["center_distance_mm"] for c in comparisons)
        max_normal = max(c["normal_angle_deg"] for c in comparisons)
        if max_center <= 3.0 and max_normal <= 3.0:
            verdict = "base pose stable; insertion offset is more likely tool tip/roll/fixture geometry."
        elif max_center > 6.0:
            verdict = "base center changes a lot; suspect hand-eye/RGBD/YOLO coordinate chain, especially image-edge capture."
        else:
            verdict = "borderline; repeat with hole closer to image center and compare overlays."
        summary["verdict"] = verdict
        print(f"\n判定: {verdict}")

    out_dir = Path(out_dir)
    summary_path = out_dir / "j6_rotation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] summary: {summary_path}")
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "j6_rotation_verify"))
    parser.add_argument("--robot-ip", default=cfg_get(CONFIG, "robot", "ip", default="169.254.128.21"))
    parser.add_argument("--robot-port", type=int, default=int(cfg_get(CONFIG, "robot", "port", default=8080)))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--serial", default=str(cfg_get(CONFIG, "camera", "serial", default="")))
    parser.add_argument("--width", type=int, default=int(cfg_get(CONFIG, "camera", "width", default=1280)))
    parser.add_argument("--height", type=int, default=int(cfg_get(CONFIG, "camera", "height", default=720)))
    parser.add_argument("--fps", type=int, default=int(cfg_get(CONFIG, "camera", "fps", default=30)))
    parser.add_argument("--warmup", type=int, default=int(cfg_get(CONFIG, "camera", "warmup", default=30)))
    parser.add_argument("--model", default=str(relative_path(CONFIG, "detection", "model", default="label_dataset/best.onnx")))
    parser.add_argument("--imgsz", type=int, default=int(cfg_get(CONFIG, "detection", "imgsz", default=960)))
    parser.add_argument("--yolo-conf", type=float, default=float(cfg_get(CONFIG, "detection", "yolo_conf", default=0.05)))
    parser.add_argument("--yolo-iou", type=float, default=float(cfg_get(CONFIG, "detection", "yolo_iou", default=0.45)))
    parser.add_argument("--mask-threshold", type=float, default=float(cfg_get(CONFIG, "detection", "mask_threshold", default=0.5)))
    parser.add_argument("--min-confidence", type=float, default=float(cfg_get(CONFIG, "detection", "min_confidence", default=0.35)))
    parser.add_argument("--keep-latest", action="store_true")
    parser.add_argument("--auto", action="store_true", help="Capture two samples without waiting for Enter.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and not args.keep_latest:
        shutil.rmtree(out_dir)
    capture_root = out_dir / "captures"
    detect_root = out_dir / "detections"
    capture_root.mkdir(parents=True, exist_ok=True)
    detect_root.mkdir(parents=True, exist_ok=True)

    print("J6 旋转影响验证")
    print("操作要求：方向盘/架子全程不要动；只改变机械臂末端 J6 角度或恢复原角度。")
    print("样本 1：使用插入较准的姿态，例如摄像头在上方。")
    print("样本 2：将末端关节旋转约 90 度，保持孔仍可见。")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.robot_ip, int(args.robot_port)))
    samples = []
    try:
        samples.append(collect_sample("normal_view", 1, sock, capture_root, detect_root, args))
        samples.append(collect_sample("j6_rotated_90", 2, sock, capture_root, detect_root, args))
    finally:
        sock.close()
    summarize(samples, out_dir)


if __name__ == "__main__":
    main()
