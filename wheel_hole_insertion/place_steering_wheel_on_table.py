# -*- coding: utf-8 -*-
"""Execute a taught table placement sequence for the steering wheel."""

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

from config_loader import CONFIG, cfg_get, relative_path  # noqa: E402
from insert_center_hole import get_current_state  # noqa: E402
from move_to_center_hole import run_motion_with_keyboard_stop  # noqa: E402
from rm65_sdk_safe_ik import Rm65SafeIkMover  # noqa: E402


SCRIPT_VERSION = "2026-06-01-table-place-execute-v1"
DEFAULT_ROBOT_IP = cfg_get(CONFIG, "robot", "ip", default="169.254.128.21")
DEFAULT_ROBOT_PORT = int(cfg_get(CONFIG, "robot", "port", default=8080))
DEFAULT_POSES = relative_path(CONFIG, "placement", "poses_file", default="table_place_poses.json")


def set_right_gripper_modbus(robot_ip, robot_port, position, force, speed, device_id, timeout_s):
    commands = [
        '{"command":"set_tool_voltage","voltage_type":3}\r\n',
        '{"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout ":2}\r\n',
        '{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,0], "device":%d}\r\n' % device_id,
        '{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,1], "device":%d}\r\n' % device_id,
        '{"command":"write_registers","port":1,"address":1002,"num":1,"data":[%d,%d], "device":%d}\r\n' % (force, speed, device_id),
        '{"command":"write_registers","port":1,"address":1001,"num":1,"data":[%d,%d], "device":%d}\r\n' % (position, position, device_id),
        '{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,9], "device":%d}\r\n' % device_id,
    ]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as gripper_sock:
        gripper_sock.settimeout(max(1, int(timeout_s)))
        gripper_sock.connect((robot_ip, int(robot_port)))
        time.sleep(0.2)
        for command in commands:
            gripper_sock.sendall(command.encode("utf-8"))
            time.sleep(0.25)
        time.sleep(0.8)


def load_poses(path):
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = data.get("poses", {})
    for key in ("pre_place", "place", "retreat"):
        if key not in poses:
            raise RuntimeError(f"Missing pose {key!r} in {path}")
        if "pose" not in poses[key] or "joint_deg" not in poses[key]:
            raise RuntimeError(f"Pose {key!r} must contain pose and joint_deg in {path}")
    return path, data


def pose_distance_m(a, b):
    return float(np.linalg.norm(np.asarray(a[:3], dtype=np.float64) - np.asarray(b[:3], dtype=np.float64)))


def read_current_pose(robot_ip, robot_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((robot_ip, int(robot_port)))
        state = get_current_state(sock)
        if not state:
            raise RuntimeError("Failed to read current robot state.")
        return state
    finally:
        sock.close()


def movej_recorded(mover, joint_deg, speed):
    old_speed = mover.speed
    mover.speed = int(speed)
    try:
        return mover.movej([float(x) for x in joint_deg[:6]])
    finally:
        mover.speed = old_speed


def movel_pose(mover, pose, speed):
    return mover.arm.rm_movel([float(x) for x in pose[:6]], int(speed), 0, 0, 1)


def print_plan(current_pose, poses, args):
    pre = poses["pre_place"]["pose"]
    place = poses["place"]["pose"]
    retreat = poses["retreat"]["pose"]
    print("\n" + "=" * 72)
    print(f"script: {SCRIPT_VERSION}")
    print(f"current pose: {[round(float(x), 5) for x in current_pose]}")
    print(f"pre_place pose: {[round(float(x), 5) for x in pre]}")
    print(f"place pose: {[round(float(x), 5) for x in place]}")
    print(f"retreat pose: {[round(float(x), 5) for x in retreat]}")
    print(f"current -> pre_place: {pose_distance_m(current_pose, pre) * 1000.0:.1f} mm")
    print(f"pre_place -> place: {pose_distance_m(pre, place) * 1000.0:.1f} mm")
    print(f"place -> retreat: {pose_distance_m(place, retreat) * 1000.0:.1f} mm")
    print(f"movej speed: {args.movej_speed}")
    print(f"descend movel speed: {args.descend_speed}")
    print(f"retreat movel speed: {args.retreat_speed}")
    print(f"open gripper: {args.open_gripper}")
    print("=" * 72)


def check_plan(current_pose, poses, args):
    pre = poses["pre_place"]["pose"]
    place = poses["place"]["pose"]
    retreat = poses["retreat"]["pose"]
    current_to_pre = pose_distance_m(current_pose, pre)
    pre_to_place = pose_distance_m(pre, place)
    place_to_retreat = pose_distance_m(place, retreat)
    if current_to_pre > float(args.max_current_to_pre_m):
        raise RuntimeError(
            f"current->pre_place {current_to_pre * 1000.0:.1f} mm exceeds "
            f"{args.max_current_to_pre_m * 1000.0:.1f} mm"
        )
    if pre_to_place > float(args.max_descend_m):
        raise RuntimeError(
            f"pre_place->place {pre_to_place * 1000.0:.1f} mm exceeds "
            f"{args.max_descend_m * 1000.0:.1f} mm"
        )
    if place_to_retreat > float(args.max_retreat_m):
        raise RuntimeError(
            f"place->retreat {place_to_retreat * 1000.0:.1f} mm exceeds "
            f"{args.max_retreat_m * 1000.0:.1f} mm"
        )


def confirm_or_raise(args):
    if args.yes:
        return
    answer = input("确认执行桌面放置流程？输入 y 继续: ").strip().lower()
    if answer != "y":
        raise RuntimeError("User cancelled.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", default=str(DEFAULT_POSES))
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--movej-speed", type=int, default=int(cfg_get(CONFIG, "placement", "movej_speed", default=30)))
    parser.add_argument("--descend-speed", type=int, default=int(cfg_get(CONFIG, "placement", "descend_speed", default=5)))
    parser.add_argument("--retreat-speed", type=int, default=int(cfg_get(CONFIG, "placement", "retreat_speed", default=10)))
    parser.add_argument("--max-current-to-pre-m", type=float, default=float(cfg_get(CONFIG, "placement", "max_current_to_pre_m", default=0.80)))
    parser.add_argument("--max-descend-m", type=float, default=float(cfg_get(CONFIG, "placement", "max_descend_m", default=0.20)))
    parser.add_argument("--max-retreat-m", type=float, default=float(cfg_get(CONFIG, "placement", "max_retreat_m", default=0.20)))
    parser.add_argument("--open-gripper", action="store_true", default=bool(cfg_get(CONFIG, "placement", "open_gripper", default=True)))
    parser.add_argument("--no-open-gripper", action="store_false", dest="open_gripper")
    parser.add_argument("--gripper-open-position", type=int, default=int(cfg_get(CONFIG, "gripper", "open_position", default=255)))
    parser.add_argument("--gripper-speed", type=int, default=int(cfg_get(CONFIG, "gripper", "speed", default=255)))
    parser.add_argument("--gripper-force", type=int, default=int(cfg_get(CONFIG, "gripper", "force", default=255)))
    parser.add_argument("--gripper-device-id", type=int, default=int(cfg_get(CONFIG, "gripper", "device_id", default=9)))
    parser.add_argument("--gripper-timeout-s", type=int, default=int(cfg_get(CONFIG, "gripper", "timeout_s", default=5)))
    parser.add_argument("--watch-stop-key", action="store_true", default=True)
    parser.add_argument("--no-watch-stop-key", action="store_false", dest="watch_stop_key")
    parser.add_argument("--slow-stop", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually move the robot. Default is dry-run.")
    parser.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation before execution.")
    return parser.parse_args()


def main():
    args = parse_args()
    poses_path, data = load_poses(args.poses)
    poses = data["poses"]
    current_pose, current_joint = read_current_pose(args.robot_ip, args.robot_port)
    print(f"[POSES] loaded: {poses_path}")
    print_plan(current_pose, poses, args)
    check_plan(current_pose, poses, args)

    if not args.execute:
        print("[DRY-RUN] 未执行运动。添加 --execute 才会真实放置。")
        return
    confirm_or_raise(args)

    mover = Rm65SafeIkMover(robot_ip=args.robot_ip, robot_port=args.robot_port, speed=args.movej_speed)
    try:
        mover.connect()
        print("[1/4] movej to pre_place")
        ok = run_motion_with_keyboard_stop(
            "place-pre",
            lambda: movej_recorded(mover, poses["pre_place"]["joint_deg"], args.movej_speed) == 0,
            args,
        )
        if not ok:
            raise RuntimeError("movej to pre_place failed.")

        print("[2/4] slow movel to place")
        ok = run_motion_with_keyboard_stop(
            "place-descend",
            lambda: movel_pose(mover, poses["place"]["pose"], args.descend_speed) == 0,
            args,
        )
        if not ok:
            raise RuntimeError("movel to place failed.")

        if args.open_gripper:
            print("[3/4] open gripper")
            set_right_gripper_modbus(
                robot_ip=args.robot_ip,
                robot_port=args.robot_port,
                position=args.gripper_open_position,
                force=args.gripper_force,
                speed=args.gripper_speed,
                device_id=args.gripper_device_id,
                timeout_s=args.gripper_timeout_s,
            )
        else:
            print("[3/4] open gripper skipped")

        print("[4/4] movel to retreat")
        ok = run_motion_with_keyboard_stop(
            "place-retreat",
            lambda: movel_pose(mover, poses["retreat"]["pose"], args.retreat_speed) == 0,
            args,
        )
        if not ok:
            raise RuntimeError("movel to retreat failed.")
        print("[OK] table placement complete")
    finally:
        mover.close()


if __name__ == "__main__":
    main()
