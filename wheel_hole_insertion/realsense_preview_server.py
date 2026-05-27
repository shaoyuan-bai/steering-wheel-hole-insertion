# -*- coding: utf-8 -*-
"""LAN-accessible RealSense preview server for SSH-only operation."""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, jsonify, render_template_string

from rm65_sdk_safe_ik import Rm65SafeIkMover


SCRIPT_DIR = Path(__file__).resolve().parent
INITIAL_JOINT_DEG = [-9.144, 72.947, 94.574, -99.437, -88.530, -154.118]


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RealSense Preview</title>
  <style>
    :root { color-scheme: dark; font-family: Arial, sans-serif; }
    body { margin: 0; background: #111; color: #eee; }
    header { height: 48px; display: flex; align-items: center; gap: 18px; padding: 0 16px; background: #1f1f1f; border-bottom: 1px solid #333; }
    main { display: grid; grid-template-columns: 1fr 300px; gap: 14px; padding: 14px; }
    img { width: 100%; max-height: calc(100vh - 86px); object-fit: contain; background: #000; border: 1px solid #333; }
    aside { border: 1px solid #333; padding: 12px; background: #181818; }
    .row { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; border-bottom: 1px solid #2b2b2b; }
    .k { color: #aaa; }
    button { background: #2d6cdf; color: white; border: 0; padding: 9px 12px; cursor: pointer; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <strong>RealSense Preview</strong>
    <span id="status">connecting...</span>
    <button onclick="snapshot()">保存截图</button>
    <button onclick="moveInitial()">回初始位置</button>
  </header>
  <main>
    <img src="/color.mjpg" alt="RealSense color stream">
    <aside>
      <div class="row"><span class="k">Color</span><span id="color"></span></div>
      <div class="row"><span class="k">Depth</span><span id="depth"></span></div>
      <div class="row"><span class="k">FPS</span><span id="fps"></span></div>
      <div class="row"><span class="k">Last frame</span><span id="last"></span></div>
      <div class="row"><span class="k">Snapshot</span><span id="shot"></span></div>
      <div class="row"><span class="k">Robot</span><span id="robot"></span></div>
    </aside>
  </main>
  <script>
    async function refresh() {
      try {
        const r = await fetch('/status');
        const s = await r.json();
        document.getElementById('status').textContent = s.running ? 'running' : 'stopped';
        document.getElementById('color').textContent = `${s.width}x${s.height}`;
        document.getElementById('depth').textContent = s.depth_ready ? 'ready' : 'none';
        document.getElementById('fps').textContent = s.actual_fps.toFixed(1);
        document.getElementById('last').textContent = s.last_frame_time || '';
      } catch (e) {
        document.getElementById('status').textContent = 'offline';
      }
    }
    async function snapshot() {
      const r = await fetch('/snapshot', {method: 'POST'});
      const s = await r.json();
      document.getElementById('shot').textContent = s.path || s.error || '';
    }
    async function moveInitial() {
      if (!confirm('确认让机械臂低速回到初始位置？')) return;
      document.getElementById('robot').textContent = 'sending...';
      const r = await fetch('/robot/move_initial', {method: 'POST'});
      const s = await r.json();
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


class RealSenseStreamer:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.color = None
        self.depth_m = None
        self.running = False
        self.last_frame_time = ""
        self.actual_fps = 0.0
        self._thread = None
        self._pipeline = None

    def start(self):
        if self._thread is not None:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        pipeline = rs.pipeline()
        config = rs.config()
        if self.args.serial:
            config.enable_device(self.args.serial)
        config.enable_stream(rs.stream.color, self.args.width, self.args.height, rs.format.bgr8, self.args.fps)
        config.enable_stream(rs.stream.depth, self.args.width, self.args.height, rs.format.z16, self.args.fps)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        self._pipeline = pipeline
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        last_t = time.time()
        try:
            for _ in range(max(0, int(self.args.warmup))):
                pipeline.wait_for_frames()
            while self.running:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame:
                    continue
                color = np.asanyarray(color_frame.get_data()).copy()
                depth_m = None
                if depth_frame:
                    depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
                now = time.time()
                dt = max(now - last_t, 1e-6)
                last_t = now
                with self.lock:
                    self.color = color
                    self.depth_m = depth_m
                    self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1.0 / dt) if self.actual_fps else 1.0 / dt
                    self.last_frame_time = time.strftime("%H:%M:%S")
        finally:
            pipeline.stop()
            self.running = False

    def get_jpeg(self):
        with self.lock:
            if self.color is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "waiting for RealSense...", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                frame = self.color.copy()
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)])
        if not ok:
            return None
        return buf.tobytes()

    def snapshot(self):
        with self.lock:
            if self.color is None:
                return None
            color = self.color.copy()
            depth_m = None if self.depth_m is None else self.depth_m.copy()
        out_dir = Path(self.args.snapshot_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("preview_%Y%m%d_%H%M%S")
        color_path = out_dir / f"{stem}_color.png"
        cv2.imwrite(str(color_path), color)
        if depth_m is not None:
            np.save(str(out_dir / f"{stem}_depth_m.npy"), depth_m)
        return color_path

    def status(self):
        with self.lock:
            h, w = self.color.shape[:2] if self.color is not None else (self.args.height, self.args.width)
            return {
                "running": bool(self.running),
                "width": int(w),
                "height": int(h),
                "depth_ready": self.depth_m is not None,
                "actual_fps": float(self.actual_fps),
                "last_frame_time": self.last_frame_time,
            }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--serial", default="")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--snapshot-dir", default=str(SCRIPT_DIR / "preview_snapshots"))
    parser.add_argument("--robot-ip", default="169.254.128.21")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--initial-movej-speed", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    streamer = RealSenseStreamer(args)
    streamer.start()
    robot_lock = threading.Lock()
    robot_state = {"busy": False, "message": ""}

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(PAGE)

    @app.route("/status")
    def status():
        return jsonify(streamer.status())

    @app.route("/snapshot", methods=["POST"])
    def snapshot():
        path = streamer.snapshot()
        if path is None:
            return jsonify({"ok": False, "error": "no frame yet"}), 503
        return jsonify({"ok": True, "path": str(path)})

    @app.route("/robot/move_initial", methods=["POST"])
    def move_initial():
        with robot_lock:
            if robot_state["busy"]:
                return jsonify({"ok": False, "error": "robot command already running"}), 409
            robot_state["busy"] = True
            robot_state["message"] = "moving to initial pose"

        def worker():
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
                msg = f"move_initial ret={ret}"
            except Exception as exc:
                msg = f"move_initial error: {exc}"
            finally:
                try:
                    mover.close()
                finally:
                    with robot_lock:
                        robot_state["busy"] = False
                        robot_state["message"] = msg

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "message": "moving to initial pose"})

    @app.route("/robot/status")
    def robot_status():
        with robot_lock:
            return jsonify(dict(robot_state))

    @app.route("/color.mjpg")
    def color_mjpg():
        def gen():
            while True:
                jpg = streamer.get_jpeg()
                if jpg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.03)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"[INFO] open: http://<robot-ip>:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
