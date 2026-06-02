# -*- coding: utf-8 -*-
"""
Plan and optionally move the right arm toward the detected steering-wheel
center metal ring.

Default behavior is dry-run: read the latest detection JSON, read current robot
state, compute pre-insertion/final TCP poses, and print the plan. It will not
move unless --move-preinsert or --insert is explicitly provided.
"""

import argparse
import json
import select
import socket
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from insert_center_hole import (  # noqa: E402
    CONFIG,
    HAND_EYE_VALID,
    SDK_FORCE_TYPE_NAME,
    T_EE_CAM,
    build_rotation_aligning_tool_axis,
    cfg_get,
    get_current_state,
    matrix_to_rpy_xyz,
    normalize,
    parse_tool_axis,
    pose_to_matrix,
)
from rm65_sdk_safe_ik import Rm65SafeIkMover  # noqa: E402


SCRIPT_VERSION = "2026-05-21-move-to-center-hole-v1"
DEFAULT_ROBOT_IP = cfg_get(CONFIG, "robot", "ip", default="169.254.128.21")
DEFAULT_ROBOT_PORT = int(cfg_get(CONFIG, "robot", "port", default=8080))


def load_detection(path):
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = ["center_headcam_m", "normal_headcam"]
    for key in required:
        if key not in data:
            raise RuntimeError(f"Detection JSON missing {key}: {path}")
    return path, data


def find_latest_detection():
    candidates = sorted(
        SCRIPT_DIR.glob("*_detection.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"No *_detection.json found in {SCRIPT_DIR}")
    return candidates[0]


def transform_center_axis_to_base(pose_now, detection, reverse_insert_axis):
    t_base_ee = pose_to_matrix(pose_now)
    t_base_cam = t_base_ee @ T_EE_CAM

    center_cam = np.asarray(detection["center_headcam_m"], dtype=np.float64)
    normal_cam = normalize(np.asarray(detection["normal_headcam"], dtype=np.float64), "normal_headcam")

    center_base = (t_base_cam @ np.r_[center_cam, 1.0])[:3]
    normal_base = normalize(t_base_cam[:3, :3] @ normal_cam, "normal_base")

    # The fitted normal usually points toward the camera. Insertion goes into
    # the hole, so default to the opposite direction.
    insert_axis_base = -normal_base
    if reverse_insert_axis:
        insert_axis_base = -insert_axis_base
    return center_base, normal_base, normalize(insert_axis_base, "insert_axis_base")


def build_plan_from_base(pose_now, center_base, normal_base, args, quality="base-target"):
    center_base = np.asarray(center_base, dtype=np.float64)
    normal_base = normalize(np.asarray(normal_base, dtype=np.float64), "normal_base")
    insert_axis_base = -normal_base
    if args.reverse_insert_axis:
        insert_axis_base = -insert_axis_base
    insert_axis_base = normalize(insert_axis_base, "insert_axis_base")

    offset_frame = str(getattr(args, "offset_frame", "camera")).lower()
    if offset_frame == "camera":
        t_base_cam = pose_to_matrix(pose_now) @ T_EE_CAM
        view_right_base = t_base_cam[:3, 0]
        view_up_base = -t_base_cam[:3, 1]
        view_right_base = view_right_base - insert_axis_base * float(np.dot(view_right_base, insert_axis_base))
        view_up_base = view_up_base - insert_axis_base * float(np.dot(view_up_base, insert_axis_base))
        if np.linalg.norm(view_right_base) < 1e-6 or np.linalg.norm(view_up_base) < 1e-6:
            raise RuntimeError("Camera offset axes are nearly parallel to insertion axis.")
        view_right_base = normalize(view_right_base, "camera_view_right_base")
        view_up_base = normalize(view_up_base, "camera_view_up_base")
    else:
        view_up_base = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        view_up_base = view_up_base - insert_axis_base * float(np.dot(view_up_base, insert_axis_base))
        if np.linalg.norm(view_up_base) < 1e-6:
            view_up_base = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            view_up_base = view_up_base - insert_axis_base * float(np.dot(view_up_base, insert_axis_base))
        view_up_base = normalize(view_up_base, "view_up_base")
        view_right_base = normalize(np.cross(view_up_base, insert_axis_base), "view_right_base")

    # Positive observed offsets mean "the rod appears right/up from the hole"
    # when looking from the robot toward the hole. Compensate by moving the
    # target left/down in that same view plane.
    visual_correction_base = (
        -view_right_base * float(args.observed_right_offset_m)
        -view_up_base * float(args.observed_up_offset_m)
    )
    center_base = center_base + visual_correction_base

    current_rot = pose_to_matrix(pose_now)[:3, :3]
    tool_axis_local = parse_tool_axis(args.tool_axis)
    if getattr(args, "keep_current_orientation", False):
        target_rot = current_rot
        target_rpy = np.asarray(pose_now[3:6], dtype=np.float64)
    else:
        target_rot = build_rotation_aligning_tool_axis(current_rot, tool_axis_local, insert_axis_base)
        target_rpy = matrix_to_rpy_xyz(target_rot)

    tip_tcp = get_tip_tcp_vector(args, tool_axis_local)
    pre_tip = center_base - insert_axis_base * float(args.preinsert_distance_m)
    final_tip = center_base + insert_axis_base * float(args.insert_depth_m)
    tcp_offset_base = target_rot @ tip_tcp
    pre_tcp = pre_tip - tcp_offset_base
    final_tcp = final_tip - tcp_offset_base

    pre_pose = [
        float(round(pre_tcp[0], 3)),
        float(round(pre_tcp[1], 3)),
        float(round(pre_tcp[2], 3)),
        float(target_rpy[0]),
        float(target_rpy[1]),
        float(target_rpy[2]),
    ]
    final_pose = [
        float(round(final_tcp[0], 3)),
        float(round(final_tcp[1], 3)),
        float(round(final_tcp[2], 3)),
        float(target_rpy[0]),
        float(target_rpy[1]),
        float(target_rpy[2]),
    ]

    current_xyz = np.asarray(pose_now[:3], dtype=np.float64)
    return {
        "script_version": SCRIPT_VERSION,
        "quality": quality,
        "center_base_m": center_base.tolist(),
        "normal_base": normal_base.tolist(),
        "insert_axis_base": insert_axis_base.tolist(),
        "view_right_base": view_right_base.tolist(),
        "view_up_base": view_up_base.tolist(),
        "offset_frame": offset_frame,
        "visual_correction_base_m": visual_correction_base.tolist(),
        "preinsert_pose": pre_pose,
        "final_pose": final_pose,
        "pre_tip_base_m": pre_tip.tolist(),
        "final_tip_base_m": final_tip.tolist(),
        "preinsert_move_m": float(np.linalg.norm(pre_tcp - current_xyz)),
        "insert_move_m": float(np.linalg.norm(final_tcp - pre_tcp)),
        "tcp_to_tip_m": float(args.tcp_to_tip_m),
        "tip_tcp_m": tip_tcp.tolist(),
        "preinsert_distance_m": float(args.preinsert_distance_m),
        "insert_depth_m": float(args.insert_depth_m),
        "tool_axis": args.tool_axis,
        "keep_current_orientation": bool(getattr(args, "keep_current_orientation", False)),
    }


def build_plan(pose_now, detection, args):
    center_base, normal_base, _ = transform_center_axis_to_base(
        pose_now=pose_now,
        detection=detection,
        reverse_insert_axis=False,
    )
    return build_plan_from_base(
        pose_now=pose_now,
        center_base=center_base,
        normal_base=normal_base,
        args=args,
        quality=detection.get("quality", "unknown"),
    )


def parse_vec3(text, name):
    parts = [p.strip() for p in str(text).replace(",", " ").split() if p.strip()]
    if len(parts) != 3:
        raise RuntimeError(f"{name} needs 3 numbers, got: {text!r}")
    return [float(p) for p in parts]


def get_tip_tcp_vector(args, tool_axis_local):
    if getattr(args, "tip_tcp_m", ""):
        return np.asarray(parse_vec3(args.tip_tcp_m, "--tip-tcp-m"), dtype=np.float64)
    return np.asarray(tool_axis_local, dtype=np.float64) * float(args.tcp_to_tip_m)


def print_plan(detection_path, pose_now, joint_now, plan):
    print("\n" + "=" * 72)
    print(f"script: {SCRIPT_VERSION}")
    print(f"detection: {detection_path}")
    print(f"detection quality: {plan['quality']}")
    print(f"current pose: {[round(float(x), 4) for x in pose_now]}")
    print(f"current joint: {[round(float(x), 3) for x in joint_now]}")
    print(f"center_base_m: {[round(float(x), 5) for x in plan['center_base_m']]}")
    print(f"normal_base: {[round(float(x), 5) for x in plan['normal_base']]}")
    print(f"insert_axis_base: {[round(float(x), 5) for x in plan['insert_axis_base']]}")
    if any(abs(float(x)) > 1e-9 for x in plan.get("visual_correction_base_m", [0.0, 0.0, 0.0])):
        print(f"visual_correction_base_m: {[round(float(x), 5) for x in plan['visual_correction_base_m']]}")
    print(f"preinsert_pose: {[round(float(x), 5) for x in plan['preinsert_pose']]}")
    print(f"final_pose: {[round(float(x), 5) for x in plan['final_pose']]}")
    print(f"preinsert move: {plan['preinsert_move_m'] * 1000.0:.1f} mm")
    print(f"insert move: {plan['insert_move_m'] * 1000.0:.1f} mm")
    print(f"tcp_to_tip: {plan['tcp_to_tip_m'] * 1000.0:.1f} mm")
    if "tip_tcp_m" in plan:
        print(f"tip_tcp_m: {[round(float(x), 6) for x in plan['tip_tcp_m']]}")
    print("=" * 72)


def save_plan(path, detection_path, pose_now, joint_now, plan):
    path = Path(path).expanduser().resolve()
    payload = dict(plan)
    payload["detection_path"] = str(detection_path)
    payload["planned_from_pose"] = [float(x) for x in pose_now]
    payload["planned_from_joint"] = [float(x) for x in joint_now]
    payload["saved_at_unix"] = time.time()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[PLAN] saved: {path}")


def load_plan(path):
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    for key in ("preinsert_pose", "final_pose"):
        if key not in plan:
            raise RuntimeError(f"Plan JSON missing {key}: {path}")
    return path, plan


def check_plan(plan, args, allow_insert=False, allow_preinsert=True):
    if args.require_quality_ok and plan["quality"] != "ok":
        raise RuntimeError(f"Detection quality is {plan['quality']!r}, refusing to move.")
    min_tcp_z = float(args.min_tcp_z_m)
    if plan["preinsert_pose"][2] < min_tcp_z:
        raise RuntimeError(
            f"Preinsert TCP z {plan['preinsert_pose'][2]:.3f} m is below "
            f"minimum {min_tcp_z:.3f} m"
        )
    if allow_insert and plan["final_pose"][2] < min_tcp_z:
        raise RuntimeError(
            f"Final TCP z {plan['final_pose'][2]:.3f} m is below "
            f"minimum {min_tcp_z:.3f} m"
        )
    if allow_preinsert and plan["preinsert_move_m"] > float(args.max_preinsert_move_m):
        raise RuntimeError(
            f"Preinsert move {plan['preinsert_move_m'] * 1000.0:.1f} mm exceeds "
            f"{args.max_preinsert_move_m * 1000.0:.1f} mm"
        )
    planned_insert_move_m = float(plan["insert_move_m"])
    if allow_insert and float(args.insert_step_m) > 0.0:
        planned_insert_move_m = min(planned_insert_move_m, float(args.insert_step_m))
    if allow_insert and planned_insert_move_m > float(args.max_insert_move_m):
        raise RuntimeError(
            f"Insert move {planned_insert_move_m * 1000.0:.1f} mm exceeds "
            f"{args.max_insert_move_m * 1000.0:.1f} mm"
        )
    planned_insert_depth_m = float(plan.get("insert_depth_m", args.insert_depth_m))
    if allow_insert and planned_insert_depth_m > float(args.max_insert_depth_m):
        raise RuntimeError(
            f"Insert depth {planned_insert_depth_m * 1000.0:.1f} mm exceeds "
            f"{args.max_insert_depth_m * 1000.0:.1f} mm"
        )


def solve_and_movej(sdk_mover, current_joint, target_pose, speed):
    old_speed = sdk_mover.speed
    sdk_mover.speed = int(speed)
    try:
        best_joint, diagnostics = sdk_mover.solve_best_joint(current_joint, target_pose)
        if best_joint is None:
            print(f"[IK] no safe solution. accepted={len(diagnostics.get('accepted', []))}, "
                  f"rejected={len(diagnostics.get('rejected', []))}")
            for item in diagnostics.get("rejected", [])[:8]:
                print(f"[IK] reject sample: {item}")
            return False
        print(f"[MOVEJ] target joint: {[round(float(x), 3) for x in best_joint]}")
        ret = sdk_mover.movej(best_joint)
        print(f"[MOVEJ] ret={ret}")
        return ret == 0
    finally:
        sdk_mover.speed = old_speed


def rotation_about_axis(axis, angle_rad):
    axis = normalize(np.asarray(axis, dtype=np.float64), "roll axis")
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    cc = 1.0 - c
    return np.array([
        [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
        [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
        [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
    ], dtype=np.float64)


def parse_roll_search_degs(text):
    if not str(text).strip():
        return []
    values = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def make_roll_search_poses(target_pose, tool_axis_local, roll_degs):
    if not roll_degs:
        return [target_pose]
    base_rot = pose_to_matrix(target_pose)[:3, :3]
    local_roll_axis = parse_tool_axis(tool_axis_local)
    poses = []
    seen = set()
    for deg in roll_degs:
        key = round(float(deg), 6)
        if key in seen:
            continue
        seen.add(key)
        rot = base_rot @ rotation_about_axis(local_roll_axis, np.deg2rad(float(deg)))
        rpy = matrix_to_rpy_xyz(rot)
        poses.append([
            float(target_pose[0]),
            float(target_pose[1]),
            float(target_pose[2]),
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        ])
    return poses


def solve_roll_search_and_movej(sdk_mover, current_joint, target_pose, args):
    roll_degs = parse_roll_search_degs(args.roll_search_deg)
    target_poses = make_roll_search_poses(target_pose, args.tool_axis, roll_degs)
    print(f"[IK] roll-search candidates deg: {roll_degs if roll_degs else [0.0]}")
    old_speed = sdk_mover.speed
    sdk_mover.speed = int(args.movej_speed)
    try:
        best_joint, best_pose, diagnostics = sdk_mover.solve_best_joint_for_poses(
            current_joint, target_poses
        )
        if best_joint is None:
            print(f"[IK] no safe roll-search solution. accepted={len(diagnostics.get('accepted', []))}, "
                  f"rejected={len(diagnostics.get('rejected', []))}")
            for item in diagnostics.get("rejected", [])[:12]:
                print(f"[IK] reject sample: {item}")
            return False
        chosen_pose_index = diagnostics.get("accepted", [(None, 0)])[0][1]
        chosen_roll = roll_degs[chosen_pose_index] if chosen_pose_index is not None and roll_degs else 0.0
        print(f"[IK] selected roll deg: {chosen_roll:.1f}")
        print(f"[IK] selected pose: {[round(float(x), 5) for x in best_pose]}")
        print(f"[MOVEJ] target joint: {[round(float(x), 3) for x in best_joint]}")
        ret = sdk_mover.movej(best_joint)
        print(f"[MOVEJ] ret={ret}")
        return ret == 0
    finally:
        sdk_mover.speed = old_speed


def run_movel(sdk_mover, target_pose, speed):
    sdk_mover.connect()
    ret = sdk_mover.arm.rm_movel(target_pose, int(speed), 0, 0, 1)
    print(f"[MOVEL] ret={ret}")
    return ret == 0


def run_movej_p(sdk_mover, target_pose, speed):
    sdk_mover.connect()
    print(f"[MOVEJ_P] target pose: {[round(float(x), 5) for x in target_pose]}")
    ret = sdk_mover.arm.rm_movej_p(target_pose, int(speed), 0, 0, 1)
    print(f"[MOVEJ_P] ret={ret}")
    return ret == 0


def build_orientation_align_pose(pose_now, plan):
    return [
        float(pose_now[0]),
        float(pose_now[1]),
        float(pose_now[2]),
        float(plan["preinsert_pose"][3]),
        float(plan["preinsert_pose"][4]),
        float(plan["preinsert_pose"][5]),
    ]


def build_insert_target_from_current(pose_now, plan, args):
    final_pose = [float(x) for x in plan["final_pose"][:6]]
    step = float(args.insert_step_m)
    if step <= 0.0:
        return final_pose

    current_xyz = np.asarray(pose_now[:3], dtype=np.float64)
    final_xyz = np.asarray(final_pose[:3], dtype=np.float64)
    delta = final_xyz - current_xyz
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return final_pose

    move = min(step, dist)
    target_xyz = current_xyz + delta / dist * move
    return [
        float(round(target_xyz[0], 3)),
        float(round(target_xyz[1], 3)),
        float(round(target_xyz[2], 3)),
        final_pose[3],
        final_pose[4],
        final_pose[5],
    ]


def send_arm_stop(robot_ip, robot_port, slow_stop=False):
    stopper = Rm65SafeIkMover(robot_ip=robot_ip, robot_port=robot_port, speed=1)
    try:
        stopper.connect()
        if slow_stop:
            ret = stopper.arm.rm_set_arm_slow_stop()
            print(f"[STOP] rm_set_arm_slow_stop ret={ret}")
        else:
            ret = stopper.arm.rm_set_arm_stop()
            print(f"[STOP] rm_set_arm_stop ret={ret}")
        return ret == 0
    finally:
        stopper.close()


def run_motion_with_keyboard_stop(label, motion_fn, args):
    if not args.watch_stop_key:
        return motion_fn()

    done_event = threading.Event()
    stop_event = threading.Event()
    result = {"ok": False, "error": None}

    def _worker():
        try:
            result["ok"] = bool(motion_fn())
        except Exception as exc:
            result["error"] = exc
        finally:
            done_event.set()

    thread = threading.Thread(target=_worker, name=f"{label}-motion", daemon=True)
    thread.start()

    print("[STOP] 运动中按 s / 空格 / 回车 / q 会发送轨迹急停。")
    if not sys.stdin.isatty():
        print("[STOP WARN] 当前 stdin 不是交互终端，按键停止不可用；可另开 SSH 执行 robot_stop.py。")
        thread.join()
    else:
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not done_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if readable:
                    ch = sys.stdin.read(1)
                    if ch in ("s", "S", " ", "\r", "\n", "q", "Q"):
                        print("\n[STOP] 收到按键，正在发送停止指令...")
                        stop_event.set()
                        send_arm_stop(args.robot_ip, args.robot_port, slow_stop=args.slow_stop)
                        break
            thread.join()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    if result["error"] is not None:
        raise result["error"]
    if stop_event.is_set():
        return False
    return bool(result["ok"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", default="", help="Detection JSON. Default: latest *_detection.json.")
    parser.add_argument("--plan-out", default="", help="Save the computed plan to this JSON file.")
    parser.add_argument("--plan-in", default="", help="Load a saved plan JSON and execute its fixed poses.")
    parser.add_argument("--base-center", default="",
                        help="Fixed base-frame center x,y,z in meters. Bypasses camera detection transform.")
    parser.add_argument("--base-normal", default="",
                        help="Fixed base-frame normal x,y,z. Usually the plane normal pointing toward camera.")
    parser.add_argument("--observed-right-offset-m", type=float,
                        default=float(cfg_get(CONFIG, "insertion", "observed_right_offset_m", default=0.0)),
                        help="Observed rod offset to the right of the hole, from robot-to-hole view. Target is compensated left.")
    parser.add_argument("--observed-up-offset-m", type=float,
                        default=float(cfg_get(CONFIG, "insertion", "observed_up_offset_m", default=0.0)),
                        help="Observed rod offset above the hole, from robot-to-hole view. Target is compensated down.")
    parser.add_argument("--offset-frame", choices=["camera", "base"], default=cfg_get(CONFIG, "insertion", "offset_frame", default="camera"),
                        help="Frame used by observed right/up offsets. camera means image right/up projected to the hole plane.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--tool-axis", default=cfg_get(CONFIG, "tool", "tool_axis", default="+z"),
                        choices=["+x", "-x", "+y", "-y", "+z", "-z"])
    parser.add_argument("--reverse-insert-axis", action="store_true",
                        help="Flip the default insertion axis.")
    parser.add_argument("--tcp-to-tip-m", type=float,
                        default=float(cfg_get(CONFIG, "tool", "tcp_to_tip_m", default=0.150)))
    parser.add_argument("--tip-tcp-m", default=",".join(str(x) for x in cfg_get(CONFIG, "tool", "tip_tcp_m", default=[])),
                        help="Full rod tip vector in TCP frame, x,y,z meters. Overrides --tcp-to-tip-m.")
    parser.add_argument("--preinsert-distance-m", type=float,
                        default=float(cfg_get(CONFIG, "insertion", "preinsert_distance_m", default=0.080)))
    parser.add_argument("--insert-depth-m", type=float,
                        default=float(cfg_get(CONFIG, "insertion", "insert_depth_m", default=0.005)))
    parser.add_argument("--max-preinsert-move-m", type=float,
                        default=float(cfg_get(CONFIG, "insertion", "max_preinsert_move_m", default=0.50)))
    parser.add_argument("--max-insert-move-m", type=float,
                        default=float(cfg_get(CONFIG, "insertion", "max_insert_move_m", default=0.12)))
    parser.add_argument("--max-insert-depth-m", type=float,
                        default=float(cfg_get(CONFIG, "insertion", "max_insert_depth_m", default=0.015)))
    parser.add_argument("--min-tcp-z-m", type=float,
                        default=float(cfg_get(CONFIG, "insertion", "min_tcp_z_m", default=-0.05)),
                        help="Refuse motion if planned TCP z is below this base-frame height.")
    parser.add_argument("--insert-step-m", type=float, default=0.0,
                        help="For --insert, move only this distance from current pose toward final_pose. 0 means full insert.")
    parser.add_argument("--movej-speed", type=int,
                        default=int(cfg_get(CONFIG, "motion", "movej_speed", default=8)))
    parser.add_argument("--movel-speed", type=int,
                        default=int(cfg_get(CONFIG, "motion", "movel_speed", default=3)))
    parser.add_argument("--max-joint-step-deg", type=float,
                        default=float(cfg_get(CONFIG, "motion", "max_joint_step_deg", default=90.0)))
    parser.add_argument("--max-j6-step-deg", type=float,
                        default=float(cfg_get(CONFIG, "motion", "max_j6_step_deg", default=60.0)))
    parser.add_argument("--roll-search-deg", default="",
                        help="Comma-separated wrist roll candidates around the tool insertion axis, in degrees.")
    parser.add_argument("--controller-pose-fallback", action="store_true",
                        help="If local SDK IK fails for preinsert, use controller rm_movej_p to plan to the checked pose.")
    parser.add_argument("--keep-current-orientation", action="store_true",
                        help="Keep the current TCP orientation while planning the target tip position.")
    parser.add_argument("--align-orientation-first", action="store_true",
                        help="Before moving to preinsert, rotate the current TCP pose to the planned insertion orientation.")
    parser.add_argument("--align-orientation-only", action="store_true",
                        help="Only rotate the current TCP pose to the planned insertion orientation, then stop.")
    parser.add_argument("--require-quality-ok", action="store_true", default=True)
    parser.add_argument("--allow-non-ok-quality", action="store_true")
    parser.add_argument("--allow-stale-hand-eye", action="store_true",
                        help="Allow motion even if config hand_eye.valid is false.")
    parser.add_argument("--move-preinsert", action="store_true",
                        help="Actually movej to pre-insertion pose.")
    parser.add_argument("--insert", action="store_true",
                        help="Actually execute short movel to final pose. Run only after preinsert is reached.")
    parser.add_argument("--watch-stop-key", action="store_true",
                        help="During motion, press s/space/enter/q in this terminal to stop the arm.")
    parser.add_argument("--slow-stop", action="store_true",
                        help="Use rm_set_arm_slow_stop instead of rm_set_arm_stop when the stop key is pressed.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.allow_non_ok_quality:
        args.require_quality_ok = False
    wants_motion = args.move_preinsert or args.insert or args.align_orientation_only
    if wants_motion and not HAND_EYE_VALID and not args.allow_stale_hand_eye:
        raise RuntimeError(
            "Hand-eye calibration is marked invalid in wheel_hole_insertion/config.yaml. "
            "Update hand_eye.matrix and set hand_eye.valid=true after recalibration, "
            "or pass --allow-stale-hand-eye for controlled debugging."
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.robot_ip, int(args.robot_port)))
        state = get_current_state(sock)
        if not state:
            raise RuntimeError("Failed to read current robot state.")
        pose_now, joint_now = state

        if args.plan_in:
            detection_path, plan = load_plan(args.plan_in)
            print_plan(detection_path, pose_now, joint_now, plan)
        elif args.base_center or args.base_normal:
            if not args.base_center or not args.base_normal:
                raise RuntimeError("--base-center and --base-normal must be provided together.")
            detection_path = Path("<base-target>")
            plan = build_plan_from_base(
                pose_now=pose_now,
                center_base=parse_vec3(args.base_center, "--base-center"),
                normal_base=parse_vec3(args.base_normal, "--base-normal"),
                args=args,
            )
            print_plan(detection_path, pose_now, joint_now, plan)
            if args.plan_out:
                save_plan(args.plan_out, detection_path, pose_now, joint_now, plan)
        else:
            detection_path = Path(args.detection).expanduser().resolve() if args.detection else find_latest_detection()
            detection_path, detection = load_detection(detection_path)
            plan = build_plan(pose_now, detection, args)
            print_plan(detection_path, pose_now, joint_now, plan)
            if args.plan_out:
                save_plan(args.plan_out, detection_path, pose_now, joint_now, plan)

        check_plan(plan, args, allow_insert=args.insert, allow_preinsert=(args.move_preinsert or not args.insert))

        if not args.move_preinsert and not args.insert and not args.align_orientation_only:
            print("[DRY-RUN] No motion sent. Use --move-preinsert first.")
            return

        sdk_mover = Rm65SafeIkMover(
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            speed=args.movej_speed,
            force_type_name=SDK_FORCE_TYPE_NAME,
            max_joint_step_deg=args.max_joint_step_deg,
            max_j6_step_deg=args.max_j6_step_deg,
        )
        try:
            if args.align_orientation_only:
                align_pose = build_orientation_align_pose(pose_now, plan)
                print("[EXEC] align TCP orientation only")
                print(f"[ALIGN] target pose: {[round(float(x), 5) for x in align_pose]}")
                ok = run_motion_with_keyboard_stop(
                    "align-orientation-only",
                    lambda: solve_and_movej(sdk_mover, joint_now, align_pose, args.movej_speed),
                    args,
                )
                if not ok and args.controller_pose_fallback:
                    print("[EXEC] local IK failed; trying controller rm_movej_p orientation fallback")
                    ok = run_motion_with_keyboard_stop(
                        "align-orientation-only-controller-fallback",
                        lambda: run_movej_p(sdk_mover, align_pose, args.movej_speed),
                        args,
                    )
                if not ok:
                    raise RuntimeError("Orientation alignment failed.")
                return

            if args.move_preinsert:
                if args.align_orientation_first:
                    align_pose = build_orientation_align_pose(pose_now, plan)
                    print("[EXEC] align TCP orientation to insertion normal")
                    print(f"[ALIGN] target pose: {[round(float(x), 5) for x in align_pose]}")
                    ok = run_motion_with_keyboard_stop(
                        "align-orientation",
                        lambda: solve_and_movej(sdk_mover, joint_now, align_pose, args.movej_speed),
                        args,
                    )
                    if not ok and args.controller_pose_fallback:
                        print("[EXEC] local IK failed; trying controller rm_movej_p orientation fallback")
                        ok = run_motion_with_keyboard_stop(
                            "align-orientation-controller-fallback",
                            lambda: run_movej_p(sdk_mover, align_pose, args.movej_speed),
                            args,
                        )
                    if not ok:
                        raise RuntimeError("Orientation alignment failed.")
                    time.sleep(0.2)
                    state = get_current_state(sock)
                    if not state:
                        raise RuntimeError("Failed to read current robot state after orientation alignment.")
                    pose_now, joint_now = state
                    print(f"[ALIGN] current pose after align: {[round(float(x), 5) for x in pose_now]}")
                    print(f"[ALIGN] current joint after align: {[round(float(x), 3) for x in joint_now]}")

                print("[EXEC] movej to pre-insertion pose")
                if args.roll_search_deg:
                    motion_fn = lambda: solve_roll_search_and_movej(
                        sdk_mover, joint_now, plan["preinsert_pose"], args
                    )
                else:
                    motion_fn = lambda: solve_and_movej(
                        sdk_mover, joint_now, plan["preinsert_pose"], args.movej_speed
                    )
                ok = run_motion_with_keyboard_stop(
                    "preinsert",
                    motion_fn,
                    args,
                )
                if not ok and args.controller_pose_fallback:
                    print("[EXEC] local IK failed; trying controller rm_movej_p fallback")
                    ok = run_motion_with_keyboard_stop(
                        "preinsert-controller-fallback",
                        lambda: run_movej_p(sdk_mover, plan["preinsert_pose"], args.movej_speed),
                        args,
                    )
                if not ok:
                    raise RuntimeError("Pre-insertion move failed.")
                time.sleep(0.2)

            if args.insert:
                print("[EXEC] short movel insertion")
                insert_target_pose = build_insert_target_from_current(pose_now, plan, args)
                print(f"[MOVEL] target pose: {[round(float(x), 5) for x in insert_target_pose]}")
                ok = run_motion_with_keyboard_stop(
                    "insert",
                    lambda: run_movel(sdk_mover, insert_target_pose, args.movel_speed),
                    args,
                )
                if not ok:
                    raise RuntimeError("Insertion movel failed.")
        finally:
            sdk_mover.close()
    finally:
        sock.close()


if __name__ == "__main__":
    main()
