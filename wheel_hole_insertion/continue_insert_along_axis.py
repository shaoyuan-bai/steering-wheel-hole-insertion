# -*- coding: utf-8 -*-
"""Continue insertion from current pose along a saved plan's insertion axis."""

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from insert_center_hole import get_current_state, normalize  # noqa: E402
from move_to_center_hole import run_motion_with_keyboard_stop, run_movel  # noqa: E402
from rm65_sdk_safe_ik import Rm65SafeIkMover  # noqa: E402


DEFAULT_ROBOT_IP = "169.254.128.21"
DEFAULT_ROBOT_PORT = 8080


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--distance-m", type=float, default=0.03)
    parser.add_argument("--max-distance-m", type=float, default=0.03)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--watch-stop-key", action="store_true")
    parser.add_argument("--slow-stop", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    distance = float(args.distance_m)
    if distance <= 0.0:
        raise RuntimeError("--distance-m must be positive.")
    if distance > float(args.max_distance_m):
        raise RuntimeError(
            f"distance {distance * 1000.0:.1f} mm exceeds "
            f"max {args.max_distance_m * 1000.0:.1f} mm"
        )

    with open(Path(args.plan).expanduser().resolve(), "r", encoding="utf-8") as f:
        plan = json.load(f)
    axis = normalize(np.asarray(plan["insert_axis_base"], dtype=np.float64), "insert_axis_base")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.robot_ip, int(args.robot_port)))
        state = get_current_state(sock)
        if not state:
            raise RuntimeError("Failed to read current robot state.")
        pose_now, joint_now = state
    finally:
        sock.close()

    current_xyz = np.asarray(pose_now[:3], dtype=np.float64)
    target_xyz = current_xyz + axis * distance
    target_pose = [
        float(round(target_xyz[0], 3)),
        float(round(target_xyz[1], 3)),
        float(round(target_xyz[2], 3)),
        float(pose_now[3]),
        float(pose_now[4]),
        float(pose_now[5]),
    ]

    print("=" * 72)
    print(f"current pose: {[round(float(x), 5) for x in pose_now]}")
    print(f"current joint: {[round(float(x), 3) for x in joint_now]}")
    print(f"insert_axis_base: {[round(float(x), 5) for x in axis]}")
    print(f"continue distance: {distance * 1000.0:.1f} mm")
    print(f"target pose: {[round(float(x), 5) for x in target_pose]}")
    print("=" * 72)

    mover = Rm65SafeIkMover(robot_ip=args.robot_ip, robot_port=args.robot_port, speed=args.speed)
    try:
        ok = run_motion_with_keyboard_stop(
            "continue-insert",
            lambda: run_movel(mover, target_pose, args.speed),
            args,
        )
        if not ok:
            raise RuntimeError("Continue insertion failed.")
    finally:
        mover.close()


if __name__ == "__main__":
    main()
