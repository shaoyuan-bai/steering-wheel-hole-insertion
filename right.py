# -*- coding: utf-8 -*-
"""
RM65 right-arm RGBD click-to-move helper.

Controls:
- Left-click a target in the RealSense color image.
- Press "t": keep the current TCP orientation and movej to pregrasp only.
- Press "g": after pregrasp, re-click/refine the pinch point and movel to grasp.
- Press "y": experimental auto orientation, filter IK branches, then movej.
- Press "c": clear the clicked point.
- Press "q": quit.

Coordinate chain:
    P_base = T_base_ee * T_ee_cam * P_cam

The clicked point is treated as the desired gripper pinch point, not the TCP.
The t path subtracts the measured TCP-to-pinch offset, keeps the current/taught
RPY, and movej's the TCP to a far pregrasp pose. From there, press g to perform
the short final movel where the gripper pinch point should align with the
clicked target.
"""

import json
import socket
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from rm65_sdk_safe_ik import Rm65SafeIkMover


SCRIPT_DIR = Path(__file__).resolve().parent
WHEEL_INSERTION_DIR = SCRIPT_DIR / "wheel_hole_insertion"
if str(WHEEL_INSERTION_DIR) not in sys.path:
    sys.path.insert(0, str(WHEEL_INSERTION_DIR))

from config_loader import CONFIG, cfg_get  # noqa: E402


SCRIPT_VERSION = "2026-05-13-right-click-t-pregrasp-g-grasp-v6"


# -------------------- Hardware and motion config --------------------
ROBOT_IP = str(cfg_get(CONFIG, "robot", "ip", default="169.254.128.21"))
ROBOT_PORT = int(cfg_get(CONFIG, "robot", "port", default=8080))

MOVE_SPEED = int(cfg_get(CONFIG, "motion", "movej_speed", default=20))
T_MOVEJ_SPEED = int(cfg_get(CONFIG, "motion", "movej_speed", default=10))
T_MOVEL_SPEED = 8
STOP_DISTANCE_M = 0.15
MIN_MOVE_DISTANCE_M = 0.005

# Tool geometry for the conservative t path.
# The clicked target is the desired gripper pinch point. The robot command pose
# is still the TCP pose, so the TCP must stay this far behind the target along
# the approach direction. Measure and tune this on your actual gripper/adapter.
TCP_TO_PINCH_M = float(cfg_get(CONFIG, "tool", "tcp_to_tip_m", default=0.15))
PREGRASP_EXTRA_DISTANCE_M = 0.10

# Safety limits for the conservative t path. Keep these tight while testing.
MAX_PREGRASP_MOVEJ_M = 0.45
MAX_FINAL_MOVEL_M = 0.12
MAX_T_SEGMENT_M = 0.03
TARGET_REACHED_TOLERANCE_M = 0.008
PATH_OVERSHOOT_MARGIN_M = 0.04
RPY_KEEP_TOLERANCE_RAD = 0.12
ENABLE_SEGMENT_IK_CHECK = True
SEGMENT_MIN_ABS_J3_DEG = 12.0
SEGMENT_MIN_ABS_J5_DEG = 12.0

# Right-arm base workspace bounds for this click test.
# Tune these only after the coordinate chain is confirmed in your real setup.
# Set False to disable this coarse software workspace guard.
ENABLE_SAFE_BOUNDS = False
# Keep these as a coarse workspace guard, not a precision reachability check.
# The earlier x min -0.35 was too tight for reachable bottle-grasp poses.
SAFE_X_RANGE_M = (-0.45, 0.35)
SAFE_Y_RANGE_M = (-0.62, 0.10)
SAFE_Z_RANGE_M = (0.18, 0.75)

# Depth sampling.
DEPTH_WINDOW = 3
MIN_VALID_DEPTH_M = 0.10
MAX_VALID_DEPTH_M = 2.00

# Auto-orientation mode for y.
TOOL_Z_TARGET_SIGN = 1
SDK_FORCE_TYPE_NAME = str(cfg_get(CONFIG, "robot", "force_type_name", default="RM_MODEL_RM_SF_E"))
AUTO_ROLL_CANDIDATES_DEG = [
    0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 180
]


HAND_EYE_VALID = bool(cfg_get(CONFIG, "hand_eye", "valid", default=False))
T_EE_CAM = np.array(cfg_get(CONFIG, "hand_eye", "matrix"), dtype=np.float64)
if T_EE_CAM.shape != (4, 4):
    raise ValueError("wheel_hole_insertion/config.yaml hand_eye.matrix must be 4x4.")


clicked_pixel = None


def on_mouse(event, x, y, flags, param):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)
        print(f"[CLICK] pixel=({x}, {y})")


def decode_json_stream(text):
    """Decode one or more JSON objects from an RM socket reply."""
    decoder = json.JSONDecoder()
    objects = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            next_line = text.find("\n", idx)
            if next_line < 0:
                break
            idx = next_line + 1
            continue
        objects.append(obj)
        idx += end
    return objects


def get_current_state(sock):
    """Return (pose, joint), pose in meters/radians, joint in degrees."""
    try:
        cmd = {"command": "get_current_arm_state"}
        sock.sendall((json.dumps(cmd) + "\r\n").encode("utf-8"))
        time.sleep(0.05)
        data = sock.recv(8192).decode("utf-8", "ignore")
        for obj in reversed(decode_json_stream(data)):
            if "arm_state" not in obj:
                continue
            arm_state = obj["arm_state"]
            p = arm_state["pose"]
            j = arm_state["joint"]
            pose = [
                float(p[0] / 1e6),
                float(p[1] / 1e6),
                float(p[2] / 1e6),
                float(p[3] / 1000),
                float(p[4] / 1000),
                float(p[5] / 1000),
            ]
            joint = [float(x / 1000) for x in j[:6]]
            return pose, joint
    except Exception as exc:
        print(f"[ERROR] get_current_state failed: {exc}")
    return None


def pose_to_matrix(pose_xyzrpy):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = R.from_euler("xyz", pose_xyzrpy[3:], degrees=False).as_matrix()
    t[:3, 3] = pose_xyzrpy[:3]
    return t


def cam_to_base(pose_now, p_cam):
    p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
    p_base_h = pose_to_matrix(pose_now) @ T_EE_CAM @ p_cam_h
    return p_base_h[:3].astype(np.float64)


def get_depth_median(depth_img, u, v, depth_scale):
    h, w = depth_img.shape
    u0, u1 = max(0, u - DEPTH_WINDOW), min(w, u + DEPTH_WINDOW + 1)
    v0, v1 = max(0, v - DEPTH_WINDOW), min(h, v + DEPTH_WINDOW + 1)
    patch = depth_img[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    depth_m = float(np.median(valid) * depth_scale)
    if not (MIN_VALID_DEPTH_M <= depth_m <= MAX_VALID_DEPTH_M):
        return None
    return depth_m


def clicked_point_to_base(depth_img, intr, depth_scale, pose_now, pixel):
    u, v = pixel
    depth_m = get_depth_median(depth_img, u, v, depth_scale)
    if depth_m is None:
        return None, None
    p_cam = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], depth_m)
    return cam_to_base(pose_now, p_cam), depth_m


def require_valid_hand_eye_for_motion():
    if HAND_EYE_VALID:
        return True
    print(
        "[安全] wheel_hole_insertion/config.yaml 中 hand_eye.valid=false，"
        "right.py 不会发送运动指令。重新手眼标定并更新配置后再运行。"
    )
    return False


def compute_standoff_point(current_pos, target_pos):
    current = np.asarray(current_pos, dtype=np.float64)
    target = np.asarray(target_pos, dtype=np.float64)
    direction = target - current
    distance_to_target = float(np.linalg.norm(direction))
    if distance_to_target < 1e-9 or distance_to_target <= STOP_DISTANCE_M:
        return current, distance_to_target, 0.0

    unit_dir = direction / distance_to_target
    move_distance = distance_to_target - STOP_DISTANCE_M
    stop_pos = current + unit_dir * move_distance
    return stop_pos.astype(np.float64), distance_to_target, float(move_distance)


def build_keep_orientation_pose(pose_now, stop_pos):
    return [
        float(round(stop_pos[0], 3)),
        float(round(stop_pos[1], 3)),
        float(round(stop_pos[2], 3)),
        float(pose_now[3]),
        float(pose_now[4]),
        float(pose_now[5]),
    ]


def unit_vector(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return None
    return vec / norm


def build_keep_rpy_pregrasp_and_final(pose_now, target_pos):
    """
    Build the conservative bottle-cap approach:
    - target_pos is the desired gripper pinch point, not the TCP
    - final TCP pose: TCP_TO_PINCH_M behind the target along approach direction
    - pregrasp TCP pose: an extra PREGRASP_EXTRA_DISTANCE_M behind final

    The approach direction is current TCP -> detected target. This matches the
    operator's taught parallel gripper pose and the wrist-camera viewing line.
    """
    current = np.asarray(pose_now[:3], dtype=np.float64)
    target = np.asarray(target_pos, dtype=np.float64)
    approach_dir = unit_vector(target - current)
    if approach_dir is None:
        return None, "target too close to current TCP"

    if TCP_TO_PINCH_M <= 0:
        return None, "TCP_TO_PINCH_M must be positive"
    if PREGRASP_EXTRA_DISTANCE_M <= 0:
        return None, "PREGRASP_EXTRA_DISTANCE_M must be positive"

    final_xyz = target - approach_dir * TCP_TO_PINCH_M
    pregrasp_xyz = target - approach_dir * (TCP_TO_PINCH_M + PREGRASP_EXTRA_DISTANCE_M)
    rpy = [float(pose_now[3]), float(pose_now[4]), float(pose_now[5])]

    pregrasp_pose = [
        float(round(pregrasp_xyz[0], 3)),
        float(round(pregrasp_xyz[1], 3)),
        float(round(pregrasp_xyz[2], 3)),
        *rpy,
    ]
    final_pose = [
        float(round(final_xyz[0], 3)),
        float(round(final_xyz[1], 3)),
        float(round(final_xyz[2], 3)),
        *rpy,
    ]
    return {
        "approach_dir": approach_dir,
        "pinch_target_xyz": target.tolist(),
        "pregrasp_pose": pregrasp_pose,
        "final_pose": final_pose,
        "movej_distance_m": float(np.linalg.norm(pregrasp_xyz - current)),
        "movel_distance_m": float(np.linalg.norm(final_xyz - pregrasp_xyz)),
        "tcp_to_pinch_m": float(TCP_TO_PINCH_M),
        "pregrasp_extra_m": float(PREGRASP_EXTRA_DISTANCE_M),
    }, None


def build_keep_rpy_final(pose_now, target_pos):
    """
    Build only the final grasp TCP pose from the current pregrasp pose.

    target_pos is the desired gripper pinch point. The TCP final pose stays
    TCP_TO_PINCH_M behind that point along the current TCP -> target direction.
    """
    current = np.asarray(pose_now[:3], dtype=np.float64)
    target = np.asarray(target_pos, dtype=np.float64)
    approach_dir = unit_vector(target - current)
    if approach_dir is None:
        return None, "target too close to current TCP"

    final_xyz = target - approach_dir * TCP_TO_PINCH_M
    final_pose = [
        float(round(final_xyz[0], 3)),
        float(round(final_xyz[1], 3)),
        float(round(final_xyz[2], 3)),
        float(pose_now[3]),
        float(pose_now[4]),
        float(pose_now[5]),
    ]
    return {
        "approach_dir": approach_dir,
        "pinch_target_xyz": target.tolist(),
        "final_pose": final_pose,
        "movel_distance_m": float(np.linalg.norm(final_xyz - current)),
        "tcp_to_pinch_m": float(TCP_TO_PINCH_M),
    }, None


def rpy_distance(a, b):
    ra = R.from_euler("xyz", a, degrees=False)
    rb = R.from_euler("xyz", b, degrees=False)
    return float((ra.inv() * rb).magnitude())


def solve_auto_orientation(pose_now, target_pos, reference_pos, roll_deg=0.0):
    reference = np.asarray(reference_pos, dtype=np.float64)
    target = np.asarray(target_pos, dtype=np.float64)
    approach = target - reference
    norm = np.linalg.norm(approach)
    if norm < 1e-9:
        return pose_now[3:]

    z_axis = float(TOOL_Z_TARGET_SIGN) * approach / norm
    current_rot = R.from_euler("xyz", pose_now[3:], degrees=False).as_matrix()

    x_axis = current_rot[:, 0] - np.dot(current_rot[:, 0], z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = current_rot[:, 1] - np.dot(current_rot[:, 1], z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    rot = np.column_stack([x_axis, y_axis, z_axis])
    if abs(float(roll_deg)) > 1e-9:
        rot = rot @ R.from_euler("z", float(roll_deg), degrees=True).as_matrix()
    return R.from_matrix(rot).as_euler("xyz", degrees=False).tolist()


def build_auto_target_poses(pose_now, stop_pos, target_pos):
    poses = []
    for roll_deg in AUTO_ROLL_CANDIDATES_DEG:
        rpy = solve_auto_orientation(
            pose_now=pose_now,
            target_pos=target_pos,
            reference_pos=stop_pos,
            roll_deg=roll_deg,
        )
        poses.append([
            float(round(stop_pos[0], 3)),
            float(round(stop_pos[1], 3)),
            float(round(stop_pos[2], 3)),
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        ])
    return poses


def draw_text(img, lines):
    for idx, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (10, 22 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def print_motion_summary(mode, pose_now, joint_now, p_base, stop_pos, distance_to_target, move_distance):
    print("\n" + "=" * 58)
    print(f"mode: {mode}")
    print(f"current pose: {[round(x, 4) for x in pose_now]}")
    print(f"current joint: {[round(x, 3) for x in joint_now]}")
    print(f"clicked target base xyz: {[round(float(x), 4) for x in p_base]}")
    print(f"stop xyz: {[round(float(x), 4) for x in stop_pos]}")
    print(f"distance to target: {distance_to_target * 1000:.1f} mm")
    print(f"move distance: {move_distance * 1000:.1f} mm")
    print(f"remaining standoff: {STOP_DISTANCE_M * 1000:.1f} mm")
    print("=" * 58)


def print_two_stage_summary(pose_now, joint_now, p_base, plan):
    print("\n" + "=" * 58)
    print("mode: keep RPY (t) -> movej pregrasp, then short movel final")
    print(f"current pose: {[round(x, 4) for x in pose_now]}")
    print(f"current joint: {[round(x, 3) for x in joint_now]}")
    print(f"clicked pinch target xyz: {[round(float(x), 4) for x in p_base]}")
    print(f"pregrasp TCP pose: {[round(x, 4) for x in plan['pregrasp_pose']]}")
    print(f"final TCP pose: {[round(x, 4) for x in plan['final_pose']]}")
    print(f"movej distance: {plan['movej_distance_m'] * 1000:.1f} mm")
    print(f"final movel distance: {plan['movel_distance_m'] * 1000:.1f} mm")
    print(f"TCP-to-pinch offset: {plan['tcp_to_pinch_m'] * 1000:.1f} mm")
    print(f"pregrasp extra distance: {plan['pregrasp_extra_m'] * 1000:.1f} mm")
    print("=" * 58)


def is_xyz_inside_safe_bounds(xyz):
    if not ENABLE_SAFE_BOUNDS:
        return True
    x, y, z = [float(v) for v in xyz[:3]]
    return (
        SAFE_X_RANGE_M[0] <= x <= SAFE_X_RANGE_M[1]
        and SAFE_Y_RANGE_M[0] <= y <= SAFE_Y_RANGE_M[1]
        and SAFE_Z_RANGE_M[0] <= z <= SAFE_Z_RANGE_M[1]
    )


def make_segment_pose(current_pose, final_pose, segment_m):
    current_xyz = np.asarray(current_pose[:3], dtype=np.float64)
    final_xyz = np.asarray(final_pose[:3], dtype=np.float64)
    delta = final_xyz - current_xyz
    dist = float(np.linalg.norm(delta))
    if dist <= segment_m:
        next_xyz = final_xyz
    else:
        next_xyz = current_xyz + delta / dist * segment_m
    return [
        float(round(next_xyz[0], 3)),
        float(round(next_xyz[1], 3)),
        float(round(next_xyz[2], 3)),
        float(final_pose[3]),
        float(final_pose[4]),
        float(final_pose[5]),
    ]


def joint_margin_ok(joint_deg):
    if joint_deg is None or len(joint_deg) < 6:
        return False, "关节角不可用"
    if abs(float(joint_deg[2])) < SEGMENT_MIN_ABS_J3_DEG:
        return False, f"J3 接近 0 度，肘部奇异风险：J3={joint_deg[2]:.3f} deg"
    if abs(float(joint_deg[4])) < SEGMENT_MIN_ABS_J5_DEG:
        return False, f"J5 接近 0 度，腕部奇异风险：J5={joint_deg[4]:.3f} deg"
    return True, "ok"


def print_joint_singularity_report(sdk_mover, joint_deg, label):
    print(f"[奇异诊断] {label}")
    if joint_deg is None or len(joint_deg) < 6:
        print("[奇异诊断]   关节角不可用")
        return False

    joint = [float(v) for v in joint_deg[:6]]
    print(
        "[奇异诊断]   关节角(deg): "
        f"J1={joint[0]:.2f}, J2={joint[1]:.2f}, J3={joint[2]:.2f}, "
        f"J4={joint[3]:.2f}, J5={joint[4]:.2f}, J6={joint[5]:.2f}"
    )
    print(
        "[奇异诊断]   规则检查: "
        f"|J3|={abs(joint[2]):.2f} deg, |J5|={abs(joint[4]):.2f} deg, "
        f"阈值={SEGMENT_MIN_ABS_J3_DEG:.1f}/{SEGMENT_MIN_ABS_J5_DEG:.1f} deg"
    )

    ok, reason = joint_margin_ok(joint)
    if ok:
        print("[奇异诊断]   J3/J5 规则检查：通过")
    else:
        print(f"[奇异诊断]   J3/J5 规则检查：不通过，{reason}")

    try:
        sdk_mover.connect()
        universal_ret = sdk_mover._universal_singularity_analyse(joint)
        print(f"[奇异诊断]   SDK 通用奇异分析返回码: {universal_ret}")
    except Exception as exc:
        universal_ret = None
        print(f"[奇异诊断]   SDK 通用奇异分析失败: {exc}")

    try:
        if hasattr(sdk_mover.algo, "rm_algo_kin_robot_singularity_analyse"):
            analytic_ret, distance = sdk_mover._robot_singularity_analyse(joint)
            print(f"[奇异诊断]   SDK 解析奇异分析返回码: {analytic_ret}, 距离/裕量: {distance:.6f}")
        else:
            analytic_ret = None
            print("[奇异诊断]   SDK 解析奇异分析接口不可用")
    except Exception as exc:
        analytic_ret = None
        print(f"[奇异诊断]   SDK 解析奇异分析失败: {exc}")

    if ok and universal_ret in (None, 0) and analytic_ret in (None, 0):
        print("[奇异诊断]   结论：未发现明显奇异风险")
        return True

    print("[奇异诊断]   结论：存在奇异风险或 SDK 检查未通过")
    return False


def print_ik_reject_summary(diagnostics, prefix="[IK诊断]", limit=8):
    accepted = diagnostics.get("accepted", [])
    rejected = diagnostics.get("rejected", [])
    print(f"{prefix}   可接受解数量: {len(accepted)}, 拒绝解数量: {len(rejected)}")
    for item in rejected[:limit]:
        print(f"{prefix}   拒绝原因样例: {item}")


def run_segmented_movel(sdk_mover, sock, start_pose, target_pose, max_total_m):
    print(f"[MOVEL] 最终目标 TCP 位姿: {[round(x, 4) for x in target_pose]}")
    planned_move = float(np.linalg.norm(np.asarray(target_pose[:3]) - np.asarray(start_pose[:3])))
    if planned_move > max_total_m:
        print(
            f"[安全] 笛卡尔移动被拦截：计划距离 {planned_move * 1000:.1f} mm > "
            f"允许上限 {max_total_m * 1000:.1f} mm。"
        )
        return False
    if not is_xyz_inside_safe_bounds(target_pose[:3]):
        print(f"[安全] 笛卡尔移动被拦截：目标点超出软件边界 {target_pose[:3]}")
        return False

    sdk_mover.connect()
    max_steps = int(np.ceil(max(planned_move, 1e-9) / MAX_T_SEGMENT_M)) + 2
    start_xyz = np.asarray(start_pose[:3], dtype=np.float64)
    final_xyz = np.asarray(target_pose[:3], dtype=np.float64)

    for step in range(1, max_steps + 1):
        state_now = get_current_state(sock)
        if not state_now:
            print("[安全] 笛卡尔移动停止：读取当前位姿失败。")
            return False
        current_pose, current_joint = state_now
        current_xyz = np.asarray(current_pose[:3], dtype=np.float64)
        dist_to_goal = float(np.linalg.norm(final_xyz - current_xyz))
        dist_from_start = float(np.linalg.norm(current_xyz - start_xyz))

        print(
            f"[MOVEL] 第 {step} 段，距目标 {dist_to_goal * 1000:.1f} mm，"
            f"距起点 {dist_from_start * 1000:.1f} mm"
        )

        if dist_to_goal <= TARGET_REACHED_TOLERANCE_M:
            print("[MOVEL] 已到达目标容差范围，不再继续发送运动指令。")
            return True
        if dist_from_start > planned_move + PATH_OVERSHOOT_MARGIN_M:
            print("[安全] 笛卡尔移动停止：检测到路径超出预期。")
            return False
        if not is_xyz_inside_safe_bounds(current_xyz):
            print(f"[安全] 笛卡尔移动停止：当前位置超出软件边界 {current_pose[:3]}")
            return False

        segment_pose = make_segment_pose(current_pose, target_pose, MAX_T_SEGMENT_M)
        if not is_xyz_inside_safe_bounds(segment_pose[:3]):
            print(f"[安全] 笛卡尔移动停止：下一段目标超出软件边界 {segment_pose[:3]}")
            return False

        if ENABLE_SEGMENT_IK_CHECK:
            print_joint_singularity_report(sdk_mover, current_joint, f"g/movel 第 {step} 段之前的当前关节")
            margin_ok, reason = joint_margin_ok(current_joint)
            if not margin_ok:
                print(f"[安全] 笛卡尔移动停止：当前关节接近奇异，{reason}")
                return False

            best_joint, diagnostics = sdk_mover.solve_best_joint(current_joint, segment_pose)
            if best_joint is None:
                print("[安全] 笛卡尔移动停止：下一段目标没有安全 IK 解。")
                print_ik_reject_summary(diagnostics, prefix="[IK诊断]", limit=6)
                return False
            print_joint_singularity_report(sdk_mover, best_joint, f"g/movel 第 {step} 段下一目标的预测关节")
            margin_ok, reason = joint_margin_ok(best_joint)
            if not margin_ok:
                print(f"[安全] 笛卡尔移动停止：下一段预测关节接近奇异，{reason}")
                return False

        print(f"[MOVEL] 分段目标 TCP 位姿: {[round(x, 4) for x in segment_pose]}")
        ret = sdk_mover.arm.rm_movel(segment_pose, T_MOVEL_SPEED, 0, 0, 1)
        print(f"[MOVEL] rm_movel ret={ret}")
        if ret != 0:
            print("[安全] 笛卡尔移动停止：rm_movel 返回非 0。")
            return False

    print("[安全] 笛卡尔移动停止：达到最大分段次数。")
    return False


def run_keep_rpy_pregrasp_movej(sdk_mover, sock, joint_now, plan):
    pregrasp_pose = plan["pregrasp_pose"]

    if plan["movej_distance_m"] > MAX_PREGRASP_MOVEJ_M:
        print(
            f"[安全] 预夹取 movej 被拦截：距离 {plan['movej_distance_m'] * 1000:.1f} mm > "
            f"{MAX_PREGRASP_MOVEJ_M * 1000:.1f} mm."
        )
        return False
    if not is_xyz_inside_safe_bounds(pregrasp_pose[:3]):
        print(f"[安全] 预夹取点超出软件边界：{pregrasp_pose[:3]}")
        return False

    print_joint_singularity_report(sdk_mover, joint_now, "t/movej 前的当前关节")
    print("[MOVEJ] 正在为预夹取点求 IK，姿态使用当前/示教 RPY...")
    best_joint, diagnostics = sdk_mover.solve_best_joint(joint_now, pregrasp_pose)
    if best_joint is None:
        print("[MOVEJ] 预夹取点没有安全 IK 解，机械臂不会移动。")
        print_ik_reject_summary(diagnostics, prefix="[MOVEJ诊断]", limit=8)
        return False

    print(f"[MOVEJ] 选中的预夹取关节角: {[round(x, 3) for x in best_joint]}")
    print_joint_singularity_report(sdk_mover, best_joint, "t/movej 预夹取目标预测关节")
    sdk_mover.connect()
    old_speed = sdk_mover.speed
    sdk_mover.speed = T_MOVEJ_SPEED
    try:
        ret = sdk_mover.movej(best_joint)
    finally:
        sdk_mover.speed = old_speed
    print(f"[MOVEJ] ret={ret}")
    if ret != 0:
        print("[安全] 预夹取 movej 失败：movej 返回非 0。")
        return False

    time.sleep(0.15)
    state_after = get_current_state(sock)
    if not state_after:
        print("[安全] 预夹取 movej 后读取位姿失败。")
        return False
    pre_pose_now, pre_joint_now = state_after
    print_joint_singularity_report(sdk_mover, pre_joint_now, "t/movej 到达预夹取后的实际关节")
    rpy_err = rpy_distance(pregrasp_pose[3:], pre_pose_now[3:])
    print(f"[检查] movej 后 RPY 漂移: {rpy_err:.4f} rad")
    if rpy_err > RPY_KEEP_TOLERANCE_RAD:
        print(
            f"[安全] 虽然到达预夹取附近，但夹爪姿态漂移超过 "
            f"{RPY_KEEP_TOLERANCE_RAD:.3f} rad."
        )
        return False

    pregrasp_error = float(np.linalg.norm(np.asarray(pre_pose_now[:3]) - np.asarray(pregrasp_pose[:3])))
    print(f"[检查] 预夹取位置误差: {pregrasp_error * 1000:.1f} mm")
    return pregrasp_error <= 0.03


def run_cached_final_grasp_movel(sdk_mover, sock, plan):
    final_pose = plan["final_pose"]
    state_now = get_current_state(sock)
    if not state_now:
        print("[安全] g 中止：读取当前位姿失败。")
        return False
    start_pose, start_joint = state_now
    print_joint_singularity_report(sdk_mover, start_joint, "g/movel 前的当前关节")

    movel_distance = float(np.linalg.norm(np.asarray(final_pose[:3]) - np.asarray(start_pose[:3])))
    print(f"[夹取] 缓存的最终 TCP 位姿: {[round(x, 4) for x in final_pose]}")
    print(f"[夹取] 当前到最终点 movel 距离: {movel_distance * 1000:.1f} mm")
    if movel_distance > MAX_FINAL_MOVEL_M:
        print(
            f"[安全] g 被拦截：当前到最终点距离 {movel_distance * 1000:.1f} mm > "
            f"{MAX_FINAL_MOVEL_M * 1000:.1f} mm."
        )
        return False

    rpy_err = rpy_distance(final_pose[3:], start_pose[3:])
    print(f"[检查] 当前 RPY 与缓存最终 RPY 差值: {rpy_err:.4f} rad")
    if rpy_err > RPY_KEEP_TOLERANCE_RAD:
        print("[安全] g 被拦截：当前夹爪姿态与缓存最终姿态差异过大。")
        return False

    print("[MOVEL] 开始短距离最终靠近，保持固定 RPY。")
    return run_segmented_movel(
        sdk_mover=sdk_mover,
        sock=sock,
        start_pose=start_pose,
        target_pose=final_pose,
        max_total_m=MAX_FINAL_MOVEL_M,
    )


def run_auto_orientation_movej(sdk_mover, joint_now, target_poses):
    print(f"[SDK] trying {len(target_poses)} auto-orientation pose candidates...")
    best_joint, best_pose, diagnostics, ret = sdk_mover.solve_poses_and_movej(
        joint_now, target_poses
    )
    accepted_count = len(diagnostics.get("accepted", []))
    rejected_count = len(diagnostics.get("rejected", []))
    if best_joint is None:
        print("[SDK] no safe IK solution found; robot will not move.")
        print(f"[SDK] accepted={accepted_count}, rejected={rejected_count}")
        for item in diagnostics.get("rejected", [])[:8]:
            print(f"[SDK] reject sample: {item}")
        return

    print(f"[SDK] best target pose: {[round(x, 4) for x in best_pose]}")
    print(f"[SDK] best joint: {[round(x, 3) for x in best_joint]}")
    print(f"[SDK] accepted={accepted_count}, rejected={rejected_count}")
    print(f"[SDK] movej ret={ret}")


def main():
    global clicked_pixel
    cached_grasp_plan = None

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((ROBOT_IP, ROBOT_PORT))
        print(f"[OK] right arm socket connected: {ROBOT_IP}:{ROBOT_PORT}")
        print(f"[INFO] script version: {SCRIPT_VERSION}")
        print(f"[INFO] config: {CONFIG.get('_config_path')}")
        print(f"[INFO] hand_eye.valid={HAND_EYE_VALID}")
    except Exception as exc:
        print(f"[FAIL] socket connect failed: {exc}")
        return

    sdk_mover = Rm65SafeIkMover(
        robot_ip=ROBOT_IP,
        robot_port=ROBOT_PORT,
        speed=MOVE_SPEED,
        force_type_name=SDK_FORCE_TYPE_NAME,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
    except Exception as exc:
        sock.close()
        sdk_mover.close()
        print(f"[FAIL] RealSense start failed: {exc}")
        return

    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    cv2.namedWindow("RM65 Right Arm Vision Control")
    cv2.setMouseCallback("RM65 Right Arm Vision Control", on_mouse)

    print(">>> Ready.")
    print(">>> Left-click target.")
    print(">>> t: keep current orientation and MOVEJ to pregrasp only.")
    print(">>> g: use cached target from t and short MOVEL to grasp.")
    print(">>> y: experimental auto orientation + SDK safe IK + MOVEJ.")
    print(">>> c: clear point, q: quit.")
    if not HAND_EYE_VALID:
        print(">>> SAFETY: hand_eye.valid=false, t/g/y motion commands are blocked.")
    print(f">>> y stop distance: {STOP_DISTANCE_M * 1000:.0f} mm.")
    print(
        f">>> t plan: clicked point is pinch target, TCP-to-pinch={TCP_TO_PINCH_M * 1000:.0f} mm, "
        f"pregrasp extra={PREGRASP_EXTRA_DISTANCE_M * 1000:.0f} mm, "
        f"final movel max={MAX_FINAL_MOVEL_M * 1000:.0f} mm."
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

            vis_img = color_img.copy()
            overlay = [
                "L-click pinch target | t: pregrasp only | g: cached final grasp | y: experimental | c/q",
                f"script: {SCRIPT_VERSION}",
            ]
            if cached_grasp_plan is not None:
                overlay.append("cached final grasp: READY")
            if clicked_pixel:
                cv2.circle(vis_img, clicked_pixel, 5, (0, 0, 255), -1)
                overlay.append(f"clicked pixel: {clicked_pixel}")
            draw_text(vis_img, overlay)
            cv2.imshow("RM65 Right Arm Vision Control", vis_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                clicked_pixel = None
                cached_grasp_plan = None
                print("[RESET] clicked point cleared")
                print("[RESET] cached grasp plan cleared")
                continue
            if key == ord("g"):
                if not require_valid_hand_eye_for_motion():
                    continue
                if cached_grasp_plan is None:
                    print("[WARN] No cached grasp plan. Click target and press 't' first.")
                    continue
                try:
                    ok = run_cached_final_grasp_movel(sdk_mover, sock, cached_grasp_plan)
                    print(f"[GRASP] final movel {'completed' if ok else 'not completed'}")
                except Exception as exc:
                    print(f"[G MOVE ERROR] {exc}")
                    traceback.print_exc()
                continue
            if key not in (ord("t"), ord("y")):
                continue
            if clicked_pixel is None:
                print("[WARN] No clicked point. Left-click a target first.")
                continue
            if not require_valid_hand_eye_for_motion():
                continue

            state_now = get_current_state(sock)
            if not state_now:
                print("[ERROR] Failed to get current right-arm state.")
                continue
            pose_now, joint_now = state_now

            p_base, depth_m = clicked_point_to_base(
                depth_img=depth_img,
                intr=intr,
                depth_scale=depth_scale,
                pose_now=pose_now,
                pixel=clicked_pixel,
            )
            if p_base is None:
                print("[WARN] Invalid depth near clicked point; click a visible surface again.")
                continue

            if key == ord("t"):
                plan, plan_error = build_keep_rpy_pregrasp_and_final(pose_now, p_base)
                if plan is None:
                    print(f"[PLAN] t unavailable: {plan_error}")
                    continue
                if plan["movej_distance_m"] < MIN_MOVE_DISTANCE_M:
                    print("[PLAN] t skipped: pregrasp is already too close to current pose.")
                    continue
                print_two_stage_summary(pose_now, joint_now, p_base, plan)
                try:
                    ok = run_keep_rpy_pregrasp_movej(sdk_mover, sock, joint_now, plan)
                    if ok:
                        cached_grasp_plan = plan
                        print("[CACHE] pregrasp reached. Press 'g' to execute cached final grasp movel.")
                    else:
                        cached_grasp_plan = None
                        print("[CACHE] pregrasp failed or uncertain; cached final grasp cleared.")
                except Exception as exc:
                    print(f"[T MOVE ERROR] {exc}")
                    traceback.print_exc()

            elif key == ord("y"):
                stop_pos, distance_to_target, move_distance = compute_standoff_point(
                    current_pos=pose_now[:3],
                    target_pos=p_base,
                )
                if move_distance < MIN_MOVE_DISTANCE_M:
                    print(
                        f"[INFO] target depth={depth_m:.3f} m; distance to target is "
                        f"{distance_to_target * 1000:.1f} mm, already within stop distance."
                    )
                    continue
                target_poses = build_auto_target_poses(pose_now, stop_pos, p_base)
                print_motion_summary(
                    "auto orientation (y) -> SDK safe IK / movej",
                    pose_now,
                    joint_now,
                    p_base,
                    stop_pos,
                    distance_to_target,
                    move_distance,
                )
                try:
                    run_auto_orientation_movej(sdk_mover, joint_now, target_poses)
                except Exception as exc:
                    print(f"[SDK ERROR] safe IK/movej failed: {exc}")
                    traceback.print_exc()

    finally:
        sdk_mover.close()
        pipeline.stop()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
