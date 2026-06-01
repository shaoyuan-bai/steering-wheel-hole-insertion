# -*- coding: utf-8 -*-
"""
Safe RM65 IK + movej helper for click-to-move scripts.

The old click scripts can keep using HTTP /movel for the known-good "t" path.
For "y", use this helper so the RM SDK computes all IK branches, filters unsafe
solutions, then executes a joint-space move with the selected branch.
"""

import math
import os
import sys
from ctypes import POINTER, c_float, cast
from pathlib import Path


def _add_rm_sdk_search_paths():
    """Add likely RM_API2 locations to sys.path before importing Robotic_Arm."""
    here = Path(__file__).resolve().parent
    home = Path.home()

    candidates = []
    env_path = os.environ.get("RM_API2_PATH")
    if not env_path:
        default_path = str(home / "RM_API2")
        os.environ["RM_API2_PATH"] = default_path
        env_path = default_path

    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        here,
        here / "RM_API2",
        here.parent / "RM_API2",
        home / "RM_API2",
        home / "bai" / "RM_API2",
        home / "bai" / "hand_eye_calibration" / "RM_API2",
        Path("/opt/RM_API2"),
    ])

    for base in candidates:
        try:
            base = base.resolve()
        except Exception:
            continue
        if (base / "Python" / "Robotic_Arm" / "rm_robot_interface.py").exists():
            sdk_python = base / "Python"
            sdk_python_str = str(sdk_python)
            if sdk_python_str not in sys.path:
                sys.path.insert(0, sdk_python_str)
            return sdk_python_str
        if (base / "Robotic_Arm" / "rm_robot_interface.py").exists():
            base_str = str(base)
            if base_str not in sys.path:
                sys.path.insert(0, base_str)
            return base_str

    return None


RM_SDK_PATH = _add_rm_sdk_search_paths()
SAFE_IK_HELPER_VERSION = "2026-05-07-sdk-ctypes-v5"

try:
    from Robotic_Arm.rm_robot_interface import *  # noqa: F401,F403
    RM_SDK_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on robot SDK install.
    RM_SDK_IMPORT_ERROR = exc


def _wrap_delta_deg(a, b):
    """Smallest signed delta a-b in degrees."""
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def _first_six(values):
    return [float(x) for x in list(values)[:6]]


def _float_array(values, size=6):
    vals = [float(x) for x in list(values)[:size]]
    if len(vals) < size:
        vals.extend([0.0] * (size - len(vals)))
    return (c_float * size)(*vals)


def _joint_algo_array(values):
    """RM_API2 v1.1.5 algorithm structs use 7 floats even for 6-DOF RM65."""
    return _float_array(values, size=7)


def _pose_to_rm_pose(pose_values):
    pose = rm_pose_t()  # noqa: F405
    pose.position.x = float(pose_values[0])
    pose.position.y = float(pose_values[1])
    pose.position.z = float(pose_values[2])
    pose.euler.rx = float(pose_values[3])
    pose.euler.ry = float(pose_values[4])
    pose.euler.rz = float(pose_values[5])
    return pose


def _make_ik_params(q_now, target_pose):
    """
    Build rm_inverse_kinematics_params_t across SDK versions.

    Some RM_API2 Python builds accept plain lists; v1.1.5 on the user's robot
    exposes q_in as LP_c_float and raises "expected LP_c_float instead of list".
    """
    q = _first_six(q_now)
    pose = [float(x) for x in target_pose[:6]]

    params = rm_inverse_kinematics_params_t()  # noqa: F405

    q_arr = _joint_algo_array(q)
    pose_arr = _float_array(pose)

    # Different RM_API2 Python builds expose q_in/q_pose as either C arrays,
    # pointers, or wrapped properties. Try array assignment first, then pointer.
    try:
        params.q_in = q_arr
    except TypeError:
        params.q_in = cast(q_arr, POINTER(c_float))

    try:
        params.q_pose = pose_arr
    except TypeError:
        try:
            params.q_pose = cast(pose_arr, POINTER(c_float))
        except TypeError as exc:
            raise TypeError(
                "Unsupported rm_inverse_kinematics_params_t.q_pose type. "
                "Expected a float array or float pointer; refusing to assign rm_pose_t."
            ) from exc

    params.flag = 1
    params._q_in_array = q_arr
    params._q_pose_array = pose_arr
    return params


def _nearest_command_joint(q, q_ref):
    """
    Convert IK output to the nearest equivalent command around current joints.

    This matters most for J6: e.g. current -331 deg and IK +28 deg are nearly
    the same wrist angle mathematically, but commanding +28 can spin the gripper
    almost one full turn on some controllers.
    """
    cmd = [float(v) for v in q[:6]]
    for i in range(6):
        delta = _wrap_delta_deg(cmd[i], q_ref[i])
        if i == 5 or abs(q_ref[i]) > 180.0:
            cmd[i] = float(q_ref[i] + delta)
    return cmd


class Rm65SafeIkMover:
    """RM65 SDK wrapper: full IK solution, safety filtering, and movej."""

    def __init__(
        self,
        robot_ip,
        robot_port=8080,
        speed=20,
        force_type_name="RM_MODEL_RM_SF_E",
        singularity_limit=0.01,
        min_abs_j3_deg=12.0,
        min_abs_j5_deg=12.0,
        max_joint_step_deg=90.0,
        max_j6_step_deg=60.0,
    ):
        self.robot_ip = robot_ip
        self.robot_port = int(robot_port)
        self.speed = int(speed)
        self.force_type_name = force_type_name
        self.singularity_limit = float(singularity_limit)
        self.min_abs_j3_deg = float(min_abs_j3_deg)
        self.min_abs_j5_deg = float(min_abs_j5_deg)
        self.max_joint_step_deg = float(max_joint_step_deg)
        self.max_j6_step_deg = float(max_j6_step_deg)

        self.arm = None
        self.algo = None
        self.handle = None

    def connect(self):
        if RM_SDK_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Cannot import RealMan SDK: "
                f"{RM_SDK_IMPORT_ERROR}. Put RM_API2 in ~/RM_API2 or set "
                "RM_API2_PATH=/path/to/RM_API2. The folder must contain "
                "Robotic_Arm/rm_robot_interface.py."
            )

        if self.arm is not None:
            return

        if RM_SDK_PATH:
            print(f"[SDK] using RM_API2 path: {RM_SDK_PATH}")
        print(f"[SDK] safe IK helper version: {SAFE_IK_HELPER_VERSION}")

        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)  # noqa: F405
        self.handle = self.arm.rm_create_robot_arm(self.robot_ip, self.robot_port)
        if not getattr(self.handle, "id", None):
            raise RuntimeError(f"SDK connection failed: {self.robot_ip}:{self.robot_port}")

        self.arm.rm_set_timeout(3000)

        arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E  # noqa: F405
        try:
            force_type = getattr(rm_force_type_e, self.force_type_name)  # noqa: F405
        except AttributeError as exc:
            raise RuntimeError(
                f"SDK force type {self.force_type_name!r} not found. "
                "Refusing to fall back to another robot force model; check "
                "wheel_hole_insertion/config.yaml or the installed RM_API2 enum."
            ) from exc
        self.algo = Algo(arm_model, force_type)  # noqa: F405

        if hasattr(self.algo, "rm_algo_set_redundant_parameter_traversal_mode"):
            self.algo.rm_algo_set_redundant_parameter_traversal_mode(True)
        if hasattr(self.algo, "rm_algo_kin_set_singularity_thresholds"):
            self.algo.rm_algo_kin_set_singularity_thresholds(12.0, 12.0, 0.05)

        ret, info = self.arm.rm_get_robot_info()
        if ret == 0:
            print(f"[SDK] connected: {self.robot_ip}:{self.robot_port}, robot_info={info}")
        else:
            print(f"[SDK] connected: {self.robot_ip}:{self.robot_port}, robot_info ret={ret}")

    def close(self):
        if self.arm is not None:
            try:
                self.arm.rm_delete_robot_arm()
            finally:
                self.arm = None
                self.algo = None
                self.handle = None

    def reconnect(self):
        self.close()
        self.connect()

    def solve_best_joint(self, current_joint_deg, target_pose):
        """
        Return (best_joint, diagnostics).

        current_joint_deg: current J1..J6 in degrees.
        target_pose: [x, y, z, rx, ry, rz], position in meters, Euler in radians.
        """
        self.connect()

        q_now = _first_six(current_joint_deg)
        pose = [float(x) for x in target_pose[:6]]
        params = _make_ik_params(q_now, pose)

        result = self.algo.rm_algo_inverse_kinematics_all(params)
        diagnostics = {
            "ik_result": getattr(result, "result", None),
            "ik_num": getattr(result, "num", None),
            "accepted": [],
            "rejected": [],
        }

        candidates = []
        if getattr(result, "result", 1) == 0:
            candidates.extend(list(getattr(result, "q_solve", [])))
        else:
            diagnostics["rejected"].append(("all-ik", f"result-{getattr(result, 'result', None)}", []))

        single_ret = None
        single_q = None
        if hasattr(self.algo, "rm_algo_inverse_kinematics"):
            try:
                single_ret, single_q = self.algo.rm_algo_inverse_kinematics(params)
            except Exception as exc:
                diagnostics["single_ik_error"] = str(exc)
        diagnostics["single_ik_ret"] = single_ret
        if single_ret == 0 and single_q is not None:
            candidates.append(single_q)

        for idx, raw_q in enumerate(candidates):
            q = _first_six(raw_q)
            if len(q) < 6 or any(not math.isfinite(v) for v in q):
                diagnostics["rejected"].append((idx, "invalid-number", q))
                continue

            limit_ret = self._check_joint_position_limit(q)
            if limit_ret != 0:
                diagnostics["rejected"].append((idx, f"joint-limit-{limit_ret}", q))
                continue

            universal_ret = self._universal_singularity_analyse(q)
            if universal_ret != 0:
                diagnostics["rejected"].append((idx, f"universal-singular-{universal_ret}", q))
                continue

            if hasattr(self.algo, "rm_algo_kin_robot_singularity_analyse"):
                singular_ret, distance = self._robot_singularity_analyse(q)
                if singular_ret != 0:
                    diagnostics["rejected"].append(
                        (idx, f"analytic-singular-{singular_ret}-d={distance:.4f}", q)
                    )
                    continue

            if abs(q[2]) < self.min_abs_j3_deg:
                diagnostics["rejected"].append((idx, "j3-too-close-zero", q))
                continue
            if abs(q[4]) < self.min_abs_j5_deg:
                diagnostics["rejected"].append((idx, "j5-too-close-zero", q))
                continue

            q_command = _nearest_command_joint(q, q_now)
            command_limit_ret = self._check_joint_position_limit(q_command)
            if command_limit_ret != 0:
                diagnostics["rejected"].append(
                    (idx, f"command-joint-limit-{command_limit_ret}", q_command)
                )
                continue

            deltas = [q_command[i] - q_now[i] for i in range(6)]
            if max(abs(d) for d in deltas) > self.max_joint_step_deg:
                diagnostics["rejected"].append((idx, "joint-step-too-large", q_command))
                continue
            if abs(deltas[5]) > self.max_j6_step_deg:
                diagnostics["rejected"].append((idx, "j6-step-too-large", q_command))
                continue

            # Weighted continuity score. J6 is expensive because it rotates the gripper.
            score = (
                1.0 * sum(abs(d) for d in deltas[:3])
                + 1.5 * sum(abs(d) for d in deltas[3:5])
                + 3.0 * abs(deltas[5])
                - 0.2 * max(0.0, abs(q[2]) - self.min_abs_j3_deg)
                - 0.2 * max(0.0, abs(q[4]) - self.min_abs_j5_deg)
            )
            diagnostics["accepted"].append((score, idx, q_command, deltas))

        diagnostics["accepted"].sort(key=lambda item: item[0])
        if not diagnostics["accepted"]:
            return None, diagnostics

        return diagnostics["accepted"][0][2], diagnostics

    def solve_best_joint_for_poses(self, current_joint_deg, target_poses):
        """Try multiple Cartesian pose candidates and return the best joint branch."""
        best = None
        merged = {
            "pose_count": len(target_poses),
            "accepted": [],
            "rejected": [],
            "pose_diagnostics": [],
        }

        for pose_index, pose in enumerate(target_poses):
            joint, diagnostics = self.solve_best_joint(current_joint_deg, pose)
            merged["pose_diagnostics"].append(diagnostics)

            for item in diagnostics.get("rejected", []):
                merged["rejected"].append((pose_index,) + tuple(item))

            for score, branch_index, q, deltas in diagnostics.get("accepted", []):
                merged["accepted"].append((score, pose_index, branch_index, q, deltas, pose))
                if best is None or score < best[0]:
                    best = (score, pose_index, branch_index, q, deltas, pose)

        merged["accepted"].sort(key=lambda item: item[0])
        if best is None:
            return None, None, merged
        return best[3], best[5], merged

    def solve_poses_and_movej(self, current_joint_deg, target_poses):
        best_joint, best_pose, diagnostics = self.solve_best_joint_for_poses(
            current_joint_deg, target_poses
        )
        if best_joint is None:
            return None, best_pose, diagnostics, None

        ret = self.movej(best_joint)
        return best_joint, best_pose, diagnostics, ret

    def movej(self, target_joint_deg):
        self.connect()
        joint = _first_six(target_joint_deg)
        try:
            return self.arm.rm_movej(joint, self.speed, 0, 0, 1)
        except TypeError:
            return self.arm.rm_movej(_float_array(joint), self.speed, 0, 0, 1)

    def solve_and_movej(self, current_joint_deg, target_pose):
        best_joint, diagnostics = self.solve_best_joint(current_joint_deg, target_pose)
        if best_joint is None:
            return None, diagnostics, None

        ret = self.movej(best_joint)
        return best_joint, diagnostics, ret

    def _check_joint_position_limit(self, joint):
        try:
            return self.algo.rm_algo_ikine_check_joint_position_limit(_joint_algo_array(joint))
        except Exception:
            return self.algo.rm_algo_ikine_check_joint_position_limit(joint)

    def _universal_singularity_analyse(self, joint):
        try:
            return self.algo.rm_algo_universal_singularity_analyse(
                _joint_algo_array(joint), self.singularity_limit
            )
        except Exception:
            return self.algo.rm_algo_universal_singularity_analyse(
                joint, self.singularity_limit
            )

    def _robot_singularity_analyse(self, joint):
        try:
            return self.algo.rm_algo_kin_robot_singularity_analyse(_joint_algo_array(joint))
        except Exception:
            return self.algo.rm_algo_kin_robot_singularity_analyse(joint)
