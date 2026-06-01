# -*- coding: utf-8 -*-
"""Headless pivot calibration for the insertion rod tip.

Keep the physical rod tip fixed on the same base-frame point, change the robot
TCP orientation, and press Enter for each sample. The first N samples solve:

    R_i * p_tip_tcp + t_i = p_fixed_base

The optional next sample verifies that T_5 * p_tip_tcp lands on the same base
point.
"""

import argparse
import json
import socket
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

from insert_center_hole import get_current_state, pose_to_matrix  # noqa: E402
from config_loader import CONFIG, cfg_get  # noqa: E402


DEFAULT_ROBOT_IP = cfg_get(CONFIG, "robot", "ip", default="169.254.128.21")
DEFAULT_ROBOT_PORT = int(cfg_get(CONFIG, "robot", "port", default=8080))


def parse_args():
    parser = argparse.ArgumentParser(
        description="4-point pivot calibration for insertion rod tip in TCP frame."
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--fit-samples", type=int, default=4,
                        help="Number of samples used for fitting. Four is the minimum.")
    parser.add_argument("--verify-samples", type=int, default=1,
                        help="Additional held-out samples used only for validation.")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "insert_tip_pivot_calibration.json"))
    parser.add_argument("--load-samples", default="",
                        help="Load sampled poses from a previous JSON instead of collecting from robot.")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Collect samples immediately without waiting for Enter.")
    return parser.parse_args()


def read_robot_state(robot_ip, robot_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((robot_ip, int(robot_port)))
        state = get_current_state(sock)
    finally:
        sock.close()
    if not state:
        raise RuntimeError("Failed to read current robot state.")
    pose, joint = state
    return [float(x) for x in pose], [float(x) for x in joint]


def collect_samples(args):
    total = int(args.fit_samples) + int(args.verify_samples)
    if int(args.fit_samples) < 4:
        raise RuntimeError("--fit-samples must be at least 4.")

    samples = []
    print("=" * 72)
    print("Insertion rod tip pivot calibration")
    print("Keep the rod tip touching the same fixed physical point.")
    print("For each sample, change only the robot orientation as much as practical.")
    print("Use the first 4 samples for fitting; sample 5 is validation by default.")
    print("=" * 72)

    for idx in range(total):
        role = "FIT" if idx < int(args.fit_samples) else "VERIFY"
        if not args.no_prompt:
            input(f"\n[{idx + 1}/{total} {role}] Move to a new pose, keep tip fixed, then press Enter...")
        pose, joint = read_robot_state(args.robot_ip, args.robot_port)
        sample = {
            "index": idx + 1,
            "role": role.lower(),
            "pose": pose,
            "joint_deg": joint,
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        samples.append(sample)
        print(f"[{idx + 1}] pose: {[round(x, 6) for x in pose]}")
        print(f"[{idx + 1}] joint_deg: {[round(x, 3) for x in joint]}")
    return samples


def load_samples(path):
    with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as f:
        data = json.load(f)
    if "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise RuntimeError(f"Cannot find samples in {path}")


def fit_pivot(samples):
    if len(samples) < 4:
        raise RuntimeError("Need at least 4 fit samples.")

    rows = []
    rhs = []
    for sample in samples:
        t_tcp = pose_to_matrix(sample["pose"])
        rot = t_tcp[:3, :3]
        trans = t_tcp[:3, 3]
        rows.append(np.hstack([rot, -np.eye(3)]))
        rhs.append(-trans)

    a = np.vstack(rows)
    b = np.concatenate(rhs)
    solution, residuals, rank, singular_values = np.linalg.lstsq(a, b, rcond=None)
    tip_tcp = solution[:3]
    fixed_base = solution[3:]

    predicted = []
    errors = []
    for sample in samples:
        t_tcp = pose_to_matrix(sample["pose"])
        tip_base = (t_tcp @ np.r_[tip_tcp, 1.0])[:3]
        err = tip_base - fixed_base
        predicted.append(tip_base)
        errors.append(err)

    errors = np.asarray(errors, dtype=np.float64)
    norms = np.linalg.norm(errors, axis=1)
    return {
        "tip_tcp_m": tip_tcp,
        "fixed_base_m": fixed_base,
        "rank": int(rank),
        "singular_values": singular_values,
        "residuals": residuals,
        "fit_tip_base_m": predicted,
        "fit_errors_m": errors,
        "fit_error_norms_m": norms,
    }


def validate_samples(samples, tip_tcp, fixed_base):
    results = []
    for sample in samples:
        t_tcp = pose_to_matrix(sample["pose"])
        tip_base = (t_tcp @ np.r_[tip_tcp, 1.0])[:3]
        err = tip_base - fixed_base
        results.append({
            "index": sample["index"],
            "tip_base_m": tip_base.tolist(),
            "error_m": err.tolist(),
            "error_mm": float(np.linalg.norm(err) * 1000.0),
        })
    return results


def main():
    args = parse_args()
    if args.load_samples:
        samples = load_samples(args.load_samples)
    else:
        samples = collect_samples(args)

    fit_samples = [s for s in samples if s.get("role") == "fit"]
    verify_samples = [s for s in samples if s.get("role") == "verify"]
    if not fit_samples:
        fit_samples = samples[:int(args.fit_samples)]
        verify_samples = samples[int(args.fit_samples):]

    fit = fit_pivot(fit_samples)
    verify = validate_samples(verify_samples, fit["tip_tcp_m"], fit["fixed_base_m"])

    fit_error_mm = fit["fit_error_norms_m"] * 1000.0
    result = {
        "script": "calibrate_insert_tip_pivot.py",
        "method": "pivot-calibration",
        "fit_sample_count": len(fit_samples),
        "verify_sample_count": len(verify_samples),
        "tip_tcp_m": fit["tip_tcp_m"].tolist(),
        "fixed_base_m": fit["fixed_base_m"].tolist(),
        "fit_error_mm": [float(x) for x in fit_error_mm],
        "fit_rmse_mm": float(np.sqrt(np.mean(fit_error_mm ** 2))) if len(fit_error_mm) else None,
        "fit_max_error_mm": float(np.max(fit_error_mm)) if len(fit_error_mm) else None,
        "verify": verify,
        "rank": fit["rank"],
        "singular_values": [float(x) for x in fit["singular_values"]],
        "samples": samples,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(f"tip_tcp_m: {[round(float(x), 6) for x in fit['tip_tcp_m']]}")
    print(f"fixed_base_m: {[round(float(x), 6) for x in fit['fixed_base_m']]}")
    print(f"fit_rmse_mm: {result['fit_rmse_mm']:.3f}")
    print(f"fit_max_error_mm: {result['fit_max_error_mm']:.3f}")
    for item in verify:
        print(
            f"verify sample {item['index']}: tip_base="
            f"{[round(float(x), 6) for x in item['tip_base_m']]}, "
            f"error={item['error_mm']:.3f} mm"
        )
    print(f"[OK] saved: {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
