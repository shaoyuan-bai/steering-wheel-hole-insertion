# -*- coding: utf-8 -*-
"""Place the steering wheel by holding the arm pose and moving the lift."""

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
import socket as socket_error
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


SCRIPT_VERSION = "2026-06-01-table-place-with-lift-v1"
DEFAULT_ROBOT_IP = cfg_get(CONFIG, "robot", "ip", default="169.254.128.21")
DEFAULT_ROBOT_PORT = int(cfg_get(CONFIG, "robot", "port", default=8080))
DEFAULT_POSES = relative_path(CONFIG, "placement", "poses_file", default="table_place_poses.json")


def post_json(url, payload, timeout_s=10.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8", "ignore")
            try:
                parsed = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {"raw": body}
            return int(resp.status), parsed
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except (TimeoutError, socket_error.timeout) as exc:
        raise RuntimeError(
            f"Lift API timed out after {timeout_s:.1f}s. The command may still have been accepted; "
            "check lift state before retrying or opening the gripper."
        ) from exc


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


def load_place_poses(path):
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pre = data.get("poses", {}).get("pre_place")
    if not pre or "joint_deg" not in pre or "pose" not in pre:
        raise RuntimeError(f"{path} missing poses.pre_place.pose/joint_deg")
    vertical = data.get("poses", {}).get("vertical_pre_release")
    if vertical and ("joint_deg" not in vertical or "pose" not in vertical):
        raise RuntimeError(f"{path} has invalid poses.vertical_pre_release; expected pose/joint_deg")
    return path, vertical, pre


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


def distance_m(a, b):
    return float(np.linalg.norm(np.asarray(a[:3], dtype=np.float64) - np.asarray(b[:3], dtype=np.float64)))


def lift_move(args, height, label):
    payload = {
        "execMode": int(args.lift_exec_mode),
        "speed": float(args.lift_speed),
        "height": float(height),
    }
    url = args.mobile_base_url.rstrip("/") + "/lift_control3"
    print(f"[LIFT] {label}: POST {url} {payload}")
    status, body = post_json(url, payload, timeout_s=args.mobile_timeout_s)
    print(f"[LIFT] {label}: status={status}, body={body}")


def query_lift_state(args, label):
    url = args.mobile_base_url.rstrip("/") + "/lift_control3"
    print(f"[LIFT] {label}: query {url} {{'execMode': 0}}")
    try:
        status, body = post_json(url, {"execMode": 0}, timeout_s=args.mobile_timeout_s)
        print(f"[LIFT] {label}: status={status}, body={body}")
        return body
    except Exception as exc:
        print(f"[LIFT WARN] {label}: query failed: {exc}")
        return None


def print_plan(current_pose, vertical, pre, args):
    pre_pose = pre["pose"]
    print("\n" + "=" * 72)
    print(f"script: {SCRIPT_VERSION}")
    print(f"current pose: {[round(float(x), 5) for x in current_pose]}")
    if vertical is not None:
        vertical_pose = vertical["pose"]
        print(f"vertical_pre_release pose: {[round(float(x), 5) for x in vertical_pose]}")
        print(f"current -> vertical_pre_release: {distance_m(current_pose, vertical_pose) * 1000.0:.1f} mm")
        print(f"vertical_pre_release -> pre_place: {distance_m(vertical_pose, pre_pose) * 1000.0:.1f} mm")
    else:
        print("vertical_pre_release pose: not found, skipped")
    print(f"pre_place pose: {[round(float(x), 5) for x in pre_pose]}")
    print(f"current -> pre_place: {distance_m(current_pose, pre_pose) * 1000.0:.1f} mm")
    print(f"move arm to pre_place: {not args.skip_arm_move}")
    print(f"arm movej speed: {args.movej_speed}")
    print(f"lift lower height: {args.lift_lower_height_m} m")
    print(f"lift raise height: {args.lift_raise_height_m} m")
    print(f"lift speed: {args.lift_speed}")
    print(f"pre-place only: {args.pre_place_only}")
    print(f"gripper action: {args.gripper_action}")
    lift_state = pre.get("lift_state")
    if lift_state is not None:
        print(f"taught pre_place lift state: {lift_state}")
    print("=" * 72)


def confirm_or_raise(args):
    if args.yes:
        return
    answer = input("确认执行升降机放置流程？输入 y 继续: ").strip().lower()
    if answer != "y":
        raise RuntimeError("User cancelled.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", default=str(DEFAULT_POSES))
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--movej-speed", type=int, default=int(cfg_get(CONFIG, "placement", "movej_speed", default=30)))
    parser.add_argument("--max-current-to-pre-m", type=float, default=float(cfg_get(CONFIG, "placement", "max_current_to_pre_m", default=0.80)))
    parser.add_argument("--skip-arm-move", action="store_true", help="Do not move arm; start lift sequence from current pose.")
    parser.add_argument("--mobile-base-url", default=str(cfg_get(CONFIG, "mobile_base", "base_url", default="http://192.168.2.228:5001")))
    parser.add_argument("--mobile-timeout-s", type=float, default=float(cfg_get(CONFIG, "mobile_base", "timeout_s", default=30.0)))
    parser.add_argument("--lift-exec-mode", type=int, default=int(cfg_get(CONFIG, "placement", "lift_exec_mode", default=2)))
    parser.add_argument("--lift-speed", type=float, default=float(cfg_get(CONFIG, "placement", "lift_speed", default=0.03)))
    parser.add_argument("--lift-lower-height-m", type=float, default=float(cfg_get(CONFIG, "placement", "lift_lower_height_m", default=-0.10)))
    parser.add_argument("--lift-raise-height-m", type=float, default=float(cfg_get(CONFIG, "placement", "lift_raise_height_m", default=0.10)))
    parser.add_argument("--settle-s", type=float, default=float(cfg_get(CONFIG, "placement", "settle_s", default=0.5)))
    parser.add_argument("--pre-place-only", action="store_true",
                        help="Only move to pre_place and stop. No lift or gripper action.")
    parser.add_argument("--gripper-action", choices=["open", "close", "none"],
                        default=str(cfg_get(CONFIG, "placement", "gripper_action", default="close")))
    parser.add_argument("--open-gripper", action="store_const", const="open", dest="gripper_action",
                        help="Compatibility alias: use open gripper action.")
    parser.add_argument("--no-open-gripper", action="store_const", const="none", dest="gripper_action",
                        help="Compatibility alias: skip gripper action.")
    parser.add_argument("--gripper-open-position", type=int, default=int(cfg_get(CONFIG, "gripper", "open_position", default=255)))
    parser.add_argument("--gripper-close-position", type=int, default=int(cfg_get(CONFIG, "gripper", "close_position", default=0)))
    parser.add_argument("--gripper-speed", type=int, default=int(cfg_get(CONFIG, "gripper", "speed", default=255)))
    parser.add_argument("--gripper-force", type=int, default=int(cfg_get(CONFIG, "gripper", "force", default=255)))
    parser.add_argument("--gripper-device-id", type=int, default=int(cfg_get(CONFIG, "gripper", "device_id", default=9)))
    parser.add_argument("--gripper-timeout-s", type=int, default=int(cfg_get(CONFIG, "gripper", "timeout_s", default=5)))
    parser.add_argument("--watch-stop-key", action="store_true", default=True)
    parser.add_argument("--no-watch-stop-key", action="store_false", dest="watch_stop_key")
    parser.add_argument("--slow-stop", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually move lift/robot. Default is dry-run.")
    parser.add_argument("-y", "--yes", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    poses_path, vertical, pre = load_place_poses(args.poses)
    current_pose, _ = read_current_pose(args.robot_ip, args.robot_port)
    print(f"[POSES] loaded: {poses_path}")
    print_plan(current_pose, vertical, pre, args)
    safety_target = vertical if vertical is not None else pre
    if not args.skip_arm_move and distance_m(current_pose, safety_target["pose"]) > float(args.max_current_to_pre_m):
        raise RuntimeError(
            f"current->{safety_target['name']} {distance_m(current_pose, safety_target['pose']) * 1000.0:.1f} mm exceeds "
            f"{args.max_current_to_pre_m * 1000.0:.1f} mm"
        )
    if not args.execute:
        print("[DRY-RUN] 未执行。添加 --execute 才会真实放置。")
        return
    confirm_or_raise(args)

    mover = None
    try:
        if not args.skip_arm_move:
            mover = Rm65SafeIkMover(robot_ip=args.robot_ip, robot_port=args.robot_port, speed=args.movej_speed)
            mover.connect()
            step_label = "vertical_pre_release" if vertical is not None else "pre_place"
            print(f"[1/5] movej to {step_label}")
            ok = run_motion_with_keyboard_stop(
                f"lift-place-{step_label}",
                lambda: movej_recorded(mover, (vertical or pre)["joint_deg"], args.movej_speed) == 0,
                args,
            )
            if not ok:
                raise RuntimeError(f"movej to {step_label} failed.")
            if vertical is not None:
                print("[2/5] movej to pre_place")
                ok = run_motion_with_keyboard_stop(
                    "lift-place-pre",
                    lambda: movej_recorded(mover, pre["joint_deg"], args.movej_speed) == 0,
                    args,
                )
                if not ok:
                    raise RuntimeError("movej to pre_place failed.")
        else:
            print("[1/5] arm move skipped")

        if args.pre_place_only:
            query_lift_state(args, "pre_place_only")
            print("[OK] pre_place reached. Lift and gripper actions skipped.")
            return

        query_lift_state(args, "before_lower")
        print("[3/5] lower lift")
        lift_move(args, args.lift_lower_height_m, "lower")
        query_lift_state(args, "after_lower")
        time.sleep(max(0.0, float(args.settle_s)))

        if args.gripper_action != "none":
            position = args.gripper_open_position if args.gripper_action == "open" else args.gripper_close_position
            print(f"[4/5] {args.gripper_action} gripper, position={position}")
            set_right_gripper_modbus(
                robot_ip=args.robot_ip,
                robot_port=args.robot_port,
                position=position,
                force=args.gripper_force,
                speed=args.gripper_speed,
                device_id=args.gripper_device_id,
                timeout_s=args.gripper_timeout_s,
            )
        else:
            print("[4/5] gripper action skipped")
        time.sleep(max(0.0, float(args.settle_s)))

        print("[5/5] raise lift")
        lift_move(args, args.lift_raise_height_m, "raise")
        query_lift_state(args, "after_raise")
        print("[OK] lift-based table placement complete")
    finally:
        if mover is not None:
            mover.close()


if __name__ == "__main__":
    main()
