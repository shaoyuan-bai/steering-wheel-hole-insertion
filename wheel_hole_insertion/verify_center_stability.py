# -*- coding: utf-8 -*-
"""Verify whether repeated RGBD detections map to a stable base-frame hole pose."""

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

from detect_center_hole_yolo_rgbd import DEFAULT_MODEL, run_detection  # noqa: E402
from insert_center_hole import T_EE_CAM, get_current_state, normalize, pose_to_matrix  # noqa: E402
from config_loader import CONFIG, cfg_get  # noqa: E402


SCRIPT_VERSION = "2026-05-26-center-stability-v1"
DEFAULT_PYTHON = "/home/wooshrobot/miniconda3/envs/cyy/bin/python"
DEFAULT_ROBOT_IP = cfg_get(CONFIG, "robot", "ip", default="169.254.128.21")
DEFAULT_ROBOT_PORT = int(cfg_get(CONFIG, "robot", "port", default=8080))


def make_detect_args(capture_dir, out_dir, out_stem, cli_args):
    return argparse.Namespace(
        input=str(capture_dir),
        model=str(cli_args.model),
        out_dir=str(out_dir),
        out_stem=str(out_stem),
        imgsz=960,
        conf=0.35,
        iou=0.45,
        mask_threshold=0.5,
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
        min_confidence=float(cli_args.min_confidence),
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
        "--out", str(out_root),
        "--width", str(args.width),
        "--height", str(args.height),
        "--fps", str(args.fps),
        "--warmup", str(args.warmup),
    ]
    if args.serial:
        cmd.extend(["--serial", args.serial])

    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Capture script did not print output directory.")
    return Path(lines[-1]).expanduser().resolve()


def transform_detection_to_base(pose_now, detection):
    t_base_cam = pose_to_matrix(pose_now) @ T_EE_CAM
    center_cam = np.asarray(detection["center_headcam_m"], dtype=np.float64)
    normal_cam = normalize(np.asarray(detection["normal_headcam"], dtype=np.float64), "normal_headcam")
    center_base = (t_base_cam @ np.r_[center_cam, 1.0])[:3]
    normal_base = normalize(t_base_cam[:3, :3] @ normal_cam, "normal_base")
    insert_axis_base = -normal_base
    return center_base, normal_base, insert_axis_base


def angle_deg(a, b):
    a = normalize(np.asarray(a, dtype=np.float64), "a")
    b = normalize(np.asarray(b, dtype=np.float64), "b")
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def write_summary(out_dir, samples):
    out_dir = Path(out_dir)
    summary_path = out_dir / "stability_summary.json"
    csv_path = out_dir / "stability_samples.csv"

    valid = [s for s in samples if s.get("ok")]
    summary = {
        "script_version": SCRIPT_VERSION,
        "sample_count": len(samples),
        "valid_count": len(valid),
        "samples": samples,
    }

    if valid:
        centers = np.asarray([s["center_base_m"] for s in valid], dtype=np.float64)
        normals = np.asarray([s["normal_base"] for s in valid], dtype=np.float64)
        mean_center = centers.mean(axis=0)
        offsets = centers - mean_center
        dists = np.linalg.norm(offsets, axis=1)
        mean_normal = normalize(normals.mean(axis=0), "mean_normal")
        normal_angles = np.asarray([angle_deg(n, mean_normal) for n in normals], dtype=np.float64)
        summary["center_base_mean_m"] = mean_center.tolist()
        summary["center_error_mm"] = {
            "rms": float(np.sqrt(np.mean(dists * dists)) * 1000.0),
            "max": float(np.max(dists) * 1000.0),
            "std_xyz": (centers.std(axis=0) * 1000.0).tolist(),
        }
        summary["normal_base_mean"] = mean_normal.tolist()
        summary["normal_angle_deg"] = {
            "mean": float(np.mean(normal_angles)),
            "max": float(np.max(normal_angles)),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,ok,quality,score,center_x,center_y,center_z,normal_x,normal_y,normal_z,overlay\n")
        for s in samples:
            score = s.get("score", "")
            center = s.get("center_base_m", ["", "", ""])
            normal = s.get("normal_base", ["", "", ""])
            f.write(
                f"{s['index']},{int(bool(s.get('ok')))},{s.get('quality', '')},{score},"
                f"{center[0]},{center[1]},{center[2]},"
                f"{normal[0]},{normal[1]},{normal[2]},{s.get('overlay_path', '')}\n"
            )
    return summary_path, csv_path, summary


def process_existing_samples(summary_file, args):
    summary_file = Path(summary_file).expanduser().resolve()
    with open(summary_file, "r", encoding="utf-8") as f:
        old = json.load(f)

    out_dir = Path(args.out_dir).expanduser().resolve()
    detect_root = out_dir / "detections"
    detect_root.mkdir(parents=True, exist_ok=True)
    samples = []
    for old_sample in old.get("samples", []):
        idx = int(old_sample["index"])
        capture_dir = Path(old_sample["capture_dir"]).expanduser().resolve()
        pose_now = old_sample["pose_now"]
        joint_now = old_sample["joint_now"]
        stem = f"sample_{idx:02d}"
        print(f"[{idx}] reprocessing {capture_dir}")
        try:
            detect_args = make_detect_args(capture_dir, detect_root, stem, args)
            detection, json_path, overlay_path = run_detection(capture_dir, detect_args)
            center_base, normal_base, insert_axis_base = transform_detection_to_base(pose_now, detection)
            sample = {
                "index": idx,
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
                "insert_axis_base": [float(x) for x in insert_axis_base],
                "plane_rmse_m": float(detection["plane_rmse_m"]),
                "plane_points": int(detection["plane_points"]),
                "plane_inliers": int(detection["plane_inliers"]),
            }
            print(
                f"[{idx}] center_base={np.round(center_base, 5).tolist()}, "
                f"normal_base={np.round(normal_base, 5).tolist()}, "
                f"score={sample['score']:.3f}, quality={sample['quality']}"
            )
        except Exception as exc:
            sample = {
                "index": idx,
                "ok": False,
                "capture_dir": str(capture_dir),
                "pose_now": [float(x) for x in pose_now],
                "joint_now": [float(x) for x in joint_now],
                "error": str(exc),
            }
            print(f"[{idx}] ERROR: {exc}")
        samples.append(sample)
    return samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3, help="Number of samples to collect.")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "stability_verify"))
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--serial", default="")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--min-confidence", type=float, default=0.35,
                        help="Lower threshold for this diagnostic; quality is still recorded.")
    parser.add_argument("--auto", action="store_true",
                        help="Do not wait for Enter between samples.")
    parser.add_argument("--keep-latest", action="store_true",
                        help="Do not delete the previous contents of out-dir.")
    parser.add_argument("--reprocess-summary", default="",
                        help="Re-run detection from an existing stability_summary.json without capturing again.")
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

    if args.reprocess_summary:
        samples = process_existing_samples(args.reprocess_summary, args)
        summary_path, csv_path, summary = write_summary(out_dir, samples)
        print(f"\n[OK] summary: {summary_path}")
        print(f"[OK] csv: {csv_path}")
        if summary.get("valid_count", 0) > 0:
            err = summary["center_error_mm"]
            ang = summary["normal_angle_deg"]
            print(f"valid samples: {summary['valid_count']}/{summary['sample_count']}")
            print(f"center RMS error: {err['rms']:.2f} mm, max: {err['max']:.2f} mm")
            print(f"center std xyz: {[round(float(x), 2) for x in err['std_xyz']]} mm")
            print(f"normal angle mean/max: {ang['mean']:.2f}/{ang['max']:.2f} deg")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.robot_ip, int(args.robot_port)))
    samples = []
    try:
        for idx in range(1, int(args.count) + 1):
            if not args.auto:
                input(f"\n[{idx}/{args.count}] Move the wrist camera to a new view, keep wheel fixed, then press Enter...")

            state = get_current_state(sock)
            if not state:
                raise RuntimeError("Failed to read current robot state.")
            pose_now, joint_now = state

            print(f"[{idx}] capturing RGBD...")
            capture_dir = capture_once(capture_root, args)
            print(f"[{idx}] capture: {capture_dir}")

            stem = f"sample_{idx:02d}"
            try:
                detect_args = make_detect_args(capture_dir, detect_root, stem, args)
                detection, json_path, overlay_path = run_detection(capture_dir, detect_args)
                center_base, normal_base, insert_axis_base = transform_detection_to_base(pose_now, detection)
                sample = {
                    "index": idx,
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
                    "insert_axis_base": [float(x) for x in insert_axis_base],
                    "plane_rmse_m": float(detection["plane_rmse_m"]),
                    "plane_points": int(detection["plane_points"]),
                    "plane_inliers": int(detection["plane_inliers"]),
                }
                print(
                    f"[{idx}] center_base={np.round(center_base, 5).tolist()}, "
                    f"normal_base={np.round(normal_base, 5).tolist()}, "
                    f"score={sample['score']:.3f}, quality={sample['quality']}"
                )
            except Exception as exc:
                sample = {
                    "index": idx,
                    "ok": False,
                    "capture_dir": str(capture_dir),
                    "pose_now": [float(x) for x in pose_now],
                    "joint_now": [float(x) for x in joint_now],
                    "error": str(exc),
                }
                print(f"[{idx}] ERROR: {exc}")
            samples.append(sample)
    finally:
        sock.close()

    summary_path, csv_path, summary = write_summary(out_dir, samples)
    print(f"\n[OK] summary: {summary_path}")
    print(f"[OK] csv: {csv_path}")
    if summary.get("valid_count", 0) > 0:
        err = summary["center_error_mm"]
        ang = summary["normal_angle_deg"]
        print(f"valid samples: {summary['valid_count']}/{summary['sample_count']}")
        print(f"center RMS error: {err['rms']:.2f} mm, max: {err['max']:.2f} mm")
        print(f"center std xyz: {[round(float(x), 2) for x in err['std_xyz']]} mm")
        print(f"normal angle mean/max: {ang['mean']:.2f}/{ang['max']:.2f} deg")


if __name__ == "__main__":
    main()
