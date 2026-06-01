# -*- coding: utf-8 -*-
"""Send an immediate trajectory stop to the RM65 arm."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rm65_sdk_safe_ik import Rm65SafeIkMover  # noqa: E402
from config_loader import CONFIG, cfg_get  # noqa: E402


DEFAULT_ROBOT_IP = cfg_get(CONFIG, "robot", "ip", default="169.254.128.21")
DEFAULT_ROBOT_PORT = int(cfg_get(CONFIG, "robot", "port", default=8080))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--slow-stop", action="store_true",
                        help="Use trajectory slow stop instead of immediate trajectory stop.")
    return parser.parse_args()


def main():
    args = parse_args()
    mover = Rm65SafeIkMover(robot_ip=args.robot_ip, robot_port=args.robot_port, speed=1)
    try:
        mover.connect()
        if args.slow_stop:
            ret = mover.arm.rm_set_arm_slow_stop()
            print(f"rm_set_arm_slow_stop ret={ret}")
        else:
            ret = mover.arm.rm_set_arm_stop()
            print(f"rm_set_arm_stop ret={ret}")
        if ret != 0:
            raise SystemExit(ret)
    finally:
        mover.close()


if __name__ == "__main__":
    main()
