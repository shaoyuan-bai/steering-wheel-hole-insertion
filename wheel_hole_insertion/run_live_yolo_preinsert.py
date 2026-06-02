# -*- coding: utf-8 -*-
"""Live workflow: YOLO detect, preinsert, move to YOLO final, then extra insert."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_loader import CONFIG, cfg_get, relative_path  # noqa: E402

DEFAULT_PYTHON = "/home/wooshrobot/miniconda3/envs/cyy/bin/python"
INITIAL_JOINT_DEG = [float(x) for x in cfg_get(CONFIG, "robot", "initial_joint_deg", default=[])]


def run(cmd, capture_stdout=False):
    print("\n[CMD]", " ".join(str(x) for x in cmd))
    if capture_stdout:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
        return proc.stdout
    subprocess.run(cmd, check=True)
    return ""


def confirm_or_skip(prompt, yes):
    if yes:
        print(f"[AUTO] {prompt}")
        return True
    answer = input(f"{prompt} 输入 y 继续，其他键跳过/停止: ").strip().lower()
    return answer == "y"


def format_vec3_arg(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        return ",".join(str(float(x)) for x in value[:3])
    text = str(value).strip()
    if not text:
        return ""
    return text.replace(" ", "")


def move_to_initial(args):
    sys.path.insert(0, str(ROOT))
    from rm65_sdk_safe_ik import Rm65SafeIkMover

    print("\n[INIT] move to saved initial joint before capture")
    print("[INIT] target joint:", INITIAL_JOINT_DEG)
    mover = Rm65SafeIkMover(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        speed=args.initial_movej_speed,
        max_joint_step_deg=120,
        max_j6_step_deg=120,
    )
    try:
        mover.connect()
        ret = mover.movej(INITIAL_JOINT_DEG)
        print(f"[INIT] movej ret={ret}")
        if ret != 0:
            raise RuntimeError(f"Initial move failed, ret={ret}")
    finally:
        mover.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--serial", default=cfg_get(CONFIG, "camera", "serial", default=""))
    parser.add_argument("--width", type=int, default=int(cfg_get(CONFIG, "camera", "width", default=1280)))
    parser.add_argument("--height", type=int, default=int(cfg_get(CONFIG, "camera", "height", default=720)))
    parser.add_argument("--fps", type=int, default=int(cfg_get(CONFIG, "camera", "fps", default=30)))
    parser.add_argument("--warmup", type=int, default=int(cfg_get(CONFIG, "camera", "warmup", default=30)))
    parser.add_argument("--out-root", default=str(relative_path(CONFIG, "paths", "live_run_dir", default="live_runs")))
    parser.add_argument("--yolo-conf", type=float, default=float(cfg_get(CONFIG, "detection", "yolo_conf", default=0.05)),
                        help="Low-level YOLO candidate threshold. Lower values help produce an overlay for review.")
    parser.add_argument("--min-confidence", type=float, default=float(cfg_get(CONFIG, "detection", "min_confidence", default=0.35)),
                        help="Quality threshold used by detection JSON before motion planning.")
    parser.add_argument("--robot-ip", default=cfg_get(CONFIG, "robot", "ip", default="169.254.128.21"))
    parser.add_argument("--robot-port", type=int, default=int(cfg_get(CONFIG, "robot", "port", default=8080)))
    parser.add_argument("--initial-movej-speed", type=int, default=int(cfg_get(CONFIG, "motion", "initial_movej_speed", default=5)))
    parser.add_argument("--skip-initial", action="store_true",
                        help="Do not move to the saved initial joint before capture.")
    parser.add_argument("--right-offset-m", type=float, default=float(cfg_get(CONFIG, "insertion", "observed_right_offset_m", default=0.0025)),
                        help="Observed rod offset to the right of hole; target compensates left.")
    parser.add_argument("--up-offset-m", type=float, default=float(cfg_get(CONFIG, "insertion", "observed_up_offset_m", default=0.0030)),
                        help="Observed rod offset above hole; target compensates down.")
    parser.add_argument("--tcp-to-tip-m", type=float, default=float(cfg_get(CONFIG, "tool", "tcp_to_tip_m", default=0.20)))
    parser.add_argument("--tip-tcp-m", default=",".join(str(x) for x in cfg_get(CONFIG, "tool", "tip_tcp_m", default=[])))
    parser.add_argument("--preinsert-distance-m", type=float, default=float(cfg_get(CONFIG, "insertion", "preinsert_distance_m", default=0.08)))
    parser.add_argument("--insert-depth-m", type=float, default=float(cfg_get(CONFIG, "insertion", "insert_depth_m", default=0.002)))
    parser.add_argument("--max-preinsert-move-m", type=float, default=float(cfg_get(CONFIG, "insertion", "max_preinsert_move_m", default=0.35)))
    parser.add_argument("--max-insert-move-m", type=float, default=float(cfg_get(CONFIG, "insertion", "max_insert_move_m", default=0.10)))
    parser.add_argument("--max-insert-depth-m", type=float, default=float(cfg_get(CONFIG, "insertion", "max_insert_depth_m", default=0.003)))
    parser.add_argument("--extra-insert-m", type=float, default=0.030,
                        help="Extra distance to continue past the YOLO final pose along insertion axis.")
    parser.add_argument("--movej-speed", type=int, default=int(cfg_get(CONFIG, "motion", "movej_speed", default=2)))
    parser.add_argument("--movel-speed", type=int, default=int(cfg_get(CONFIG, "motion", "movel_speed", default=1)))
    parser.add_argument("--max-j6-step-deg", type=float, default=float(cfg_get(CONFIG, "motion", "max_j6_step_deg", default=90.0)))
    parser.add_argument("--allow-non-ok-quality", action="store_true")
    parser.add_argument("--require-quality-ok", action="store_true",
                        help="Require detector quality ok before moving.")
    parser.add_argument("--no-move", action="store_true",
                        help="Only capture, detect, and plan. Do not move.")
    parser.add_argument("--no-stop-key", action="store_true",
                        help="Disable keyboard stop watcher in the movement subprocess.")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Automatically execute all motion stages without confirmation prompts.")
    return parser.parse_args()


def main():
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.out_root).expanduser().resolve() / f"run_{ts}"
    capture_root = run_root / "capture"
    result_root = run_root / "result"
    capture_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_initial:
        move_to_initial(args)

    capture_cmd = [
        args.python,
        str(SCRIPT_DIR / "capture_rgbd_once_headless.py"),
        "--out", str(capture_root),
        "--width", str(args.width),
        "--height", str(args.height),
        "--fps", str(args.fps),
        "--warmup", str(args.warmup),
    ]
    if args.serial:
        capture_cmd.extend(["--serial", args.serial])
    stdout = run(capture_cmd, capture_stdout=True)
    capture_lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not capture_lines:
        raise RuntimeError("Capture command did not print capture directory.")
    capture_dir = Path(capture_lines[-1]).expanduser().resolve()

    stem = f"live_yolo_{ts}"
    detect_cmd = [
        args.python,
        str(SCRIPT_DIR / "detect_center_hole_yolo_rgbd.py"),
        str(capture_dir),
        "--out-dir", str(result_root),
        "--out-stem", stem,
        "--conf", str(args.yolo_conf),
        "--min-confidence", str(args.min_confidence),
    ]
    run(detect_cmd)
    detection_json = result_root / f"{stem}_detection.json"
    plan_json = result_root / f"{stem}_plan.json"

    plan_cmd = [
        args.python,
        str(SCRIPT_DIR / "move_to_center_hole.py"),
        "--detection", str(detection_json),
        "--observed-right-offset-m", str(args.right_offset_m),
        "--observed-up-offset-m", str(args.up_offset_m),
        "--tcp-to-tip-m", str(args.tcp_to_tip_m),
        "--preinsert-distance-m", str(args.preinsert_distance_m),
        "--insert-depth-m", str(args.insert_depth_m),
        "--max-preinsert-move-m", str(args.max_preinsert_move_m),
        "--movej-speed", str(args.movej_speed),
        "--movel-speed", str(args.movel_speed),
        "--max-j6-step-deg", str(args.max_j6_step_deg),
        "--plan-out", str(plan_json),
    ]
    tip_tcp_arg = format_vec3_arg(args.tip_tcp_m)
    if tip_tcp_arg:
        plan_cmd.append(f"--tip-tcp-m={tip_tcp_arg}")
    if args.require_quality_ok or not args.allow_non_ok_quality:
        plan_cmd.append("--require-quality-ok")
    else:
        plan_cmd.append("--allow-non-ok-quality")

    if args.no_move:
        run(plan_cmd)
        print("\n[OK] live run directory:", run_root)
        print("[OK] detection:", detection_json)
        print("[OK] plan:", plan_json)
        return

    if confirm_or_skip("步骤 1/3：移动到预插入位置", args.yes):
        preinsert_cmd = list(plan_cmd)
        preinsert_cmd.append("--move-preinsert")
        if not args.no_stop_key:
            preinsert_cmd.append("--watch-stop-key")
        run(preinsert_cmd)
    else:
        print("[STOP] 用户未确认预插入，流程停止。")
        return

    if confirm_or_skip("步骤 2/3：沿 YOLO 计划移动到 final_pose", args.yes):
        final_cmd = [
            args.python,
            str(SCRIPT_DIR / "move_to_center_hole.py"),
            "--plan-in", str(plan_json),
            "--insert",
            "--max-insert-move-m", str(args.max_insert_move_m),
            "--max-insert-depth-m", str(args.max_insert_depth_m),
            "--movel-speed", str(args.movel_speed),
        ]
        if args.allow_non_ok_quality and not args.require_quality_ok:
            final_cmd.append("--allow-non-ok-quality")
        if not args.no_stop_key:
            final_cmd.append("--watch-stop-key")
        run(final_cmd)
    else:
        print("[STOP] 用户未确认移动到 YOLO 终点，流程停止。")
        return

    if confirm_or_skip(f"步骤 3/3：无视 YOLO 终点继续前伸 {args.extra_insert_m * 1000.0:.1f} mm", args.yes):
        extra_cmd = [
            args.python,
            str(SCRIPT_DIR / "continue_insert_along_axis.py"),
            "--plan", str(plan_json),
            "--distance-m", str(args.extra_insert_m),
            "--max-distance-m", str(args.extra_insert_m),
            "--speed", str(args.movel_speed),
        ]
        if not args.no_stop_key:
            extra_cmd.append("--watch-stop-key")
        run(extra_cmd)
    else:
        print("[STOP] 用户未确认额外前伸，流程停止。")
        return

    print("\n[OK] live run directory:", run_root)
    print("[OK] detection:", detection_json)
    print("[OK] plan:", plan_json)


if __name__ == "__main__":
    main()
