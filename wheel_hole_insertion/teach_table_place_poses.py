# -*- coding: utf-8 -*-
"""Teach fixed table placement poses for placing the steering wheel down."""

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_loader import CONFIG, cfg_get, relative_path  # noqa: E402
from insert_center_hole import get_current_state  # noqa: E402


SCRIPT_VERSION = "2026-06-01-table-place-teach-v1"
DEFAULT_ROBOT_IP = cfg_get(CONFIG, "robot", "ip", default="169.254.128.21")
DEFAULT_ROBOT_PORT = int(cfg_get(CONFIG, "robot", "port", default=8080))
DEFAULT_OUT = relative_path(CONFIG, "placement", "poses_file", default="table_place_poses.json")
DEFAULT_MOBILE_BASE_URL = cfg_get(CONFIG, "mobile_base", "base_url", default="http://192.168.2.228:5001")
DEFAULT_MOBILE_TIMEOUT_S = float(cfg_get(CONFIG, "mobile_base", "timeout_s", default=30.0))


STEPS = [
    (
        "pre_place",
        "移动到桌面上方安全位。建议方向盘已经在桌面目标点正上方，离桌面/障碍物 80-150mm。",
    ),
    (
        "place",
        "缓慢移动到最终放置位。方向盘应刚接触或接近桌面，姿态是希望释放时的统一状态。",
    ),
    (
        "retreat",
        "移动到释放后退出位。建议从放置位沿桌面法向上抬 50-100mm，避免横向拖拽方向盘。",
    ),
]


def read_state(sock):
    state = get_current_state(sock)
    if not state:
        raise RuntimeError("Failed to read current robot state.")
    pose, joint = state
    return [float(x) for x in pose], [float(x) for x in joint]


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
            return {"ok": True, "status": int(resp.status), "body": parsed}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def query_lift_state(args):
    url = args.mobile_base_url.rstrip("/") + "/lift_control3"
    return post_json(url, {"execMode": 0}, timeout_s=args.mobile_timeout_s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--mobile-base-url", default=str(DEFAULT_MOBILE_BASE_URL))
    parser.add_argument("--mobile-timeout-s", type=float, default=DEFAULT_MOBILE_TIMEOUT_S)
    parser.add_argument("--force", action="store_true", help="Overwrite existing pose file without asking.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out).expanduser().resolve()
    if out_path.exists() and not args.force:
        answer = input(f"{out_path} 已存在，是否覆盖？输入 y 覆盖: ").strip().lower()
        if answer != "y":
            print("[CANCEL] not overwritten")
            return

    print("桌面放置位姿示教")
    print("单位说明：pose xyz 为米，rpy 为弧度，joint 为度。")
    print("示教过程中只读取当前机械臂状态，不会主动运动。")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.robot_ip, int(args.robot_port)))
    samples = {}
    try:
        for key, instruction in STEPS:
            print("\n" + "=" * 72)
            print(f"[{key}] {instruction}")
            input("手动调整到该位姿后按 Enter 记录...")
            pose, joint = read_state(sock)
            lift_state = query_lift_state(args)
            sample = {
                "name": key,
                "pose": pose,
                "joint_deg": joint,
                "lift_state": lift_state,
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            samples[key] = sample
            print(f"[{key}] pose: {[round(x, 6) for x in pose]}")
            print(f"[{key}] joint_deg: {[round(x, 3) for x in joint]}")
            print(f"[{key}] lift_state: {lift_state}")
    finally:
        sock.close()

    payload = {
        "script_version": SCRIPT_VERSION,
        "robot_ip": args.robot_ip,
        "robot_port": int(args.robot_port),
        "mobile_base_url": args.mobile_base_url,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "pre_place: safe pose above the table target.",
            "place: final release pose near/on table.",
            "retreat: safe pose after releasing and lifting away.",
        ],
        "poses": samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 72)
    print(f"[OK] saved: {out_path}")
    print("下一步可先 dry-run：")
    print(f"python wheel_hole_insertion/place_steering_wheel_on_table.py --poses {out_path}")


if __name__ == "__main__":
    main()
