# -*- coding: utf-8 -*-
"""Independent RealSense camera service.

Only this process should open RealSense devices. Other projects should call its
HTTP API or read captures saved by it.
"""

import argparse
import base64
import asyncio
import json
import threading
import time
from pathlib import Path

from aiohttp import web
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, jsonify, request


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "cameras.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "captures"


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {"roles": {}, "defaults": {"width": 1280, "height": 720, "fps": 30, "warmup": 30}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config, path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def intrinsics_to_dict(intr):
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "model": str(intr.model),
        "coeffs": [float(x) for x in intr.coeffs],
    }


def list_realsense_devices():
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        def info(key, default=""):
            try:
                return dev.get_info(key)
            except Exception:
                return default

        devices.append({
            "name": info(rs.camera_info.name),
            "serial": info(rs.camera_info.serial_number),
            "firmware": info(rs.camera_info.firmware_version),
            "product_id": info(rs.camera_info.product_id),
            "usb_type": info(rs.camera_info.usb_type_descriptor),
        })
    return devices


def resolve_camera(config, role_or_serial):
    roles = config.get("roles", {})
    if role_or_serial in roles:
        item = roles[role_or_serial]
        if isinstance(item, dict):
            serial = str(item.get("serial", ""))
        else:
            serial = str(item)
        if not serial:
            raise RuntimeError(f"Role {role_or_serial!r} has no serial in config.")
        return role_or_serial, serial
    return "", str(role_or_serial)


def make_output_dir(root, serial, role=""):
    label = role or serial
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    out_dir = Path(root).expanduser().resolve() / f"{safe}_{serial}_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_rgbd_capture(out_dir, color, depth_m, depth_scale, color_intrinsics, depth_intrinsics, meta_extra=None):
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_mm = np.rint(depth_m * 1000.0).astype(np.uint16)
    depth_z16 = np.rint(depth_m / float(depth_scale)).astype(np.uint16) if depth_scale else depth_mm
    depth_vis = cv2.convertScaleAbs(depth_z16, alpha=0.03)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(color, 0.55, depth_colormap, 0.45, 0)
    cv2.imwrite(str(out_dir / "color.png"), color)
    cv2.imwrite(str(out_dir / "depth_mm.png"), depth_mm)
    cv2.imwrite(str(out_dir / "depth_colormap.png"), depth_colormap)
    cv2.imwrite(str(out_dir / "depth_overlay.png"), overlay)
    np.save(str(out_dir / "depth_raw.npy"), depth_m)
    meta = {
        "capture_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "depth_scale_m_per_unit": float(depth_scale) if depth_scale is not None else None,
        "streams_aligned_to": "color",
        "color_intrinsics": color_intrinsics,
        "depth_intrinsics_after_align": depth_intrinsics,
        "files": {
            "color": "color.png",
            "depth_mm": "depth_mm.png",
            "depth_raw_m": "depth_raw.npy",
            "depth_colormap": "depth_colormap.png",
            "depth_overlay": "depth_overlay.png"
        },
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(out_dir / "intrinsics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def capture_rgbd(serial, out_root, width=1280, height=720, fps=30, warmup=30, role=""):
    pipeline = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(str(serial))
    cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
    cfg.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))

    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(max(0, int(warmup))):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to get aligned color/depth frames.")

        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        color = np.asanyarray(color_frame.get_data())
        depth_z16 = np.asanyarray(depth_frame.get_data())
        depth_m = depth_z16.astype(np.float32) * depth_scale

        out_dir = make_output_dir(out_root, serial or "default", role=role)

        color_intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        depth_intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        meta = save_rgbd_capture(
            out_dir,
            color,
            depth_m,
            depth_scale,
            intrinsics_to_dict(color_intr),
            intrinsics_to_dict(depth_intr),
            {
                "role": role,
                "serial": serial,
                "width": int(width),
                "height": int(height),
                "fps": int(fps),
            },
        )
        return out_dir, meta
    finally:
        pipeline.stop()


def capture_all(config, out_root, width=None, height=None, fps=None, warmup=None):
    defaults = config.get("defaults", {})
    width = int(width or defaults.get("width", 1280))
    height = int(height or defaults.get("height", 720))
    fps = int(fps or defaults.get("fps", 30))
    warmup = int(warmup if warmup is not None else defaults.get("warmup", 30))
    devices = list_realsense_devices()
    role_by_serial = {}
    for role, item in config.get("roles", {}).items():
        if isinstance(item, dict):
            serial = str(item.get("serial", ""))
        else:
            serial = str(item)
        if serial:
            role_by_serial[serial] = role

    results = []
    for dev in devices:
        serial = dev["serial"]
        role = role_by_serial.get(serial, "")
        print(f"[CAPTURE] serial={serial}, role={role or '-'}, name={dev['name']}")
        try:
            out_dir, meta = capture_rgbd(
                serial=serial,
                out_root=out_root,
                width=width,
                height=height,
                fps=fps,
                warmup=warmup,
                role=role,
            )
            item = {"ok": True, "device": dev, "role": role, "capture_dir": str(out_dir), "meta": meta}
            print(f"[OK] {serial}: {out_dir}")
        except Exception as exc:
            item = {"ok": False, "device": dev, "role": role, "error": str(exc)}
            print(f"[FAIL] {serial}: {exc}")
        results.append(item)

    root = Path(out_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / f"capture_all_summary_{now_stamp()}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"devices": devices, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"[SUMMARY] {summary_path}")
    return results


class LiveCamera:
    def __init__(self, role, serial, out_root, width=1280, height=720, fps=30, warmup=30):
        self.role = role
        self.serial = str(serial)
        self.out_root = out_root
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.warmup = int(warmup)
        self.lock = threading.Lock()
        self.running = False
        self.error = ""
        self.last_frame_time = ""
        self.actual_fps = 0.0
        self.color = None
        self.depth_m = None
        self.depth_scale = None
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self._thread = None
        self._pipeline = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self.running = True
        self.error = ""
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.role}", daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        pipeline = self._pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def _run(self):
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        try:
            profile = pipeline.start(cfg)
            self._pipeline = pipeline
            align = rs.align(rs.stream.color)
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            for _ in range(max(0, self.warmup)):
                if not self.running:
                    break
                pipeline.wait_for_frames()
            last_t = time.time()
            while self.running:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color = np.asanyarray(color_frame.get_data()).copy()
                depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
                color_intr = intrinsics_to_dict(color_frame.profile.as_video_stream_profile().get_intrinsics())
                depth_intr = intrinsics_to_dict(depth_frame.profile.as_video_stream_profile().get_intrinsics())
                now = time.time()
                dt = max(now - last_t, 1e-6)
                last_t = now
                with self.lock:
                    self.color = color
                    self.depth_m = depth_m
                    self.depth_scale = depth_scale
                    self.color_intrinsics = color_intr
                    self.depth_intrinsics = depth_intr
                    self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1.0 / dt) if self.actual_fps else 1.0 / dt
                    self.last_frame_time = time.strftime("%H:%M:%S")
                    self.error = ""
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            with self.lock:
                self.running = False
            self._pipeline = None

    def status(self):
        with self.lock:
            return {
                "role": self.role,
                "serial": self.serial,
                "running": bool(self.running),
                "ready": self.color is not None and self.depth_m is not None,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "actual_fps": float(self.actual_fps),
                "last_frame_time": self.last_frame_time,
                "error": self.error,
            }

    def latest(self):
        with self.lock:
            if self.color is None:
                return None
            return {
                "color": self.color.copy(),
                "depth_m": None if self.depth_m is None else self.depth_m.copy(),
                "depth_scale": self.depth_scale,
                "color_intrinsics": None if self.color_intrinsics is None else dict(self.color_intrinsics),
                "depth_intrinsics": None if self.depth_intrinsics is None else dict(self.depth_intrinsics),
                "last_frame_time": self.last_frame_time,
            }

    def jpeg(self, quality=85):
        latest = self.latest()
        if latest is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, f"{self.role}: waiting", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            frame = latest["color"]
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError(f"Failed to encode JPEG for {self.role}.")
        return buf.tobytes()

    def save_current(self, out_root):
        latest = self.latest()
        if latest is None or latest["depth_m"] is None or latest["color_intrinsics"] is None:
            raise RuntimeError(f"Camera {self.role} has no RGBD frame yet.")
        out_dir = make_output_dir(out_root, self.serial, self.role)
        meta = save_rgbd_capture(
            out_dir,
            latest["color"],
            latest["depth_m"],
            latest["depth_scale"],
            latest["color_intrinsics"],
            latest["depth_intrinsics"],
            {
                "source": "camera_service_live",
                "role": self.role,
                "serial": self.serial,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "last_frame_time": latest["last_frame_time"],
            },
        )
        return out_dir, meta

    def latest_json(self, quality=85, include_image=True):
        latest = self.latest()
        if latest is None:
            raise RuntimeError(f"Camera {self.role} has no frame yet.")
        data = {
            "role": self.role,
            "serial": self.serial,
            "width": int(latest["color"].shape[1]),
            "height": int(latest["color"].shape[0]),
            "last_frame_time": latest["last_frame_time"],
            "color_intrinsics": latest["color_intrinsics"],
            "depth_ready": latest["depth_m"] is not None,
        }
        if include_image:
            ok, buf = cv2.imencode(".jpg", latest["color"], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                raise RuntimeError(f"Failed to encode JPEG for {self.role}.")
            data["jpeg_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
        return data


class CameraService:
    def __init__(self, args):
        self.args = args
        self.config = load_config(args.config)
        self.lock = threading.Lock()
        self.live = {}

    def refresh_config(self):
        self.config = load_config(self.args.config)

    def role_defaults(self, role):
        defaults = dict(self.config.get("defaults", {}))
        item = self.config.get("roles", {}).get(role)
        if isinstance(item, dict):
            defaults.update({k: v for k, v in item.items() if k != "serial"})
        return defaults

    def _role_item(self, role):
        item = self.config.get("roles", {}).get(role)
        if isinstance(item, dict):
            serial = str(item.get("serial", ""))
            params = {**self.config.get("defaults", {}), **item}
        else:
            serial = str(item or "")
            params = dict(self.config.get("defaults", {}))
        if not serial:
            raise RuntimeError(f"Role {role!r} has no serial in config.")
        return serial, params

    def start_live_role(self, role):
        self.refresh_config()
        serial, params = self._role_item(role)
        existing = self.live.get(role)
        if existing is not None and existing.serial == serial:
            existing.start()
            return existing
        if existing is not None:
            existing.stop()
        cam = LiveCamera(
            role=role,
            serial=serial,
            out_root=self.args.output,
            width=int(params.get("width", self.config.get("defaults", {}).get("width", 1280))),
            height=int(params.get("height", self.config.get("defaults", {}).get("height", 720))),
            fps=int(params.get("fps", self.config.get("defaults", {}).get("fps", 30))),
            warmup=int(params.get("warmup", self.config.get("defaults", {}).get("warmup", 30))),
        )
        self.live[role] = cam
        cam.start()
        return cam

    def start_live_roles(self, mode):
        mode = str(mode or "none").strip()
        if mode in ("", "none"):
            return []
        self.refresh_config()
        if mode == "roles":
            roles = list(self.config.get("roles", {}).keys())
        elif mode == "all":
            serial_to_role = {}
            for role, item in self.config.get("roles", {}).items():
                serial = str(item.get("serial", "")) if isinstance(item, dict) else str(item or "")
                if serial:
                    serial_to_role[serial] = role
            roles = []
            for dev in list_realsense_devices():
                serial = dev.get("serial", "")
                role = serial_to_role.get(serial, serial)
                if role not in self.config.get("roles", {}):
                    self.config.setdefault("roles", {})[role] = {"serial": serial}
                roles.append(role)
        else:
            roles = [x.strip() for x in mode.split(",") if x.strip()]
        for role in roles:
            self.start_live_role(role)
        return roles

    def live_camera_for(self, role_or_serial):
        self.refresh_config()
        role, serial = resolve_camera(self.config, role_or_serial)
        if role and role in self.live:
            return role, self.live[role]
        for live_role, cam in self.live.items():
            if cam.serial == serial:
                return live_role, cam
        if role:
            return role, self.start_live_role(role)
        return "", None

    def app(self):
        app = Flask(__name__)

        @app.route("/camera/list")
        def camera_list():
            self.refresh_config()
            return jsonify({
                "devices": list_realsense_devices(),
                "roles": self.config.get("roles", {}),
                "defaults": self.config.get("defaults", {}),
            })

        @app.route("/camera/status")
        def camera_status():
            self.refresh_config()
            return jsonify({
                "roles": self.config.get("roles", {}),
                "live": {role: cam.status() for role, cam in self.live.items()},
            })

        @app.route("/camera/roles")
        def camera_roles():
            self.refresh_config()
            return jsonify(self.config.get("roles", {}))

        @app.route("/camera/start", methods=["POST"])
        def camera_start():
            payload = request.get_json(silent=True) or {}
            role = str(payload.get("role", "")).strip()
            mode = str(payload.get("mode", "")).strip()
            if role:
                roles = [self.start_live_role(role).role]
            else:
                roles = self.start_live_roles(mode or "roles")
            return jsonify({"ok": True, "roles": roles, "live": {r: self.live[r].status() for r in roles}})

        @app.route("/camera/stop", methods=["POST"])
        def camera_stop():
            payload = request.get_json(silent=True) or {}
            role = str(payload.get("role", "")).strip()
            if role:
                cam = self.live.get(role)
                if cam:
                    cam.stop()
                return jsonify({"ok": True, "role": role})
            for cam in self.live.values():
                cam.stop()
            return jsonify({"ok": True, "role": "all"})

        @app.route("/camera/config", methods=["POST"])
        def camera_config():
            payload = request.get_json(silent=True) or {}
            role = str(payload.get("role", "")).strip()
            serial = str(payload.get("serial", "")).strip()
            if not role or not serial:
                return jsonify({"ok": False, "error": "role and serial are required"}), 400
            self.refresh_config()
            item = {"serial": serial}
            for key in ("width", "height", "fps", "warmup"):
                if key in payload:
                    item[key] = payload[key]
            self.config.setdefault("roles", {})[role] = item
            save_config(self.config, self.args.config)
            return jsonify({"ok": True, "role": role, "config": item})

        @app.route("/camera/capture", methods=["POST"])
        def camera_capture():
            payload = request.get_json(silent=True) or {}
            self.refresh_config()
            role_or_serial = str(payload.get("role", "") or payload.get("serial", "")).strip()
            if not role_or_serial:
                return jsonify({"ok": False, "error": "role or serial is required"}), 400
            role, serial = resolve_camera(self.config, role_or_serial)
            d = self.role_defaults(role) if role else self.config.get("defaults", {})
            width = int(payload.get("width", d.get("width", 1280)))
            height = int(payload.get("height", d.get("height", 720)))
            fps = int(payload.get("fps", d.get("fps", 30)))
            warmup = int(payload.get("warmup", d.get("warmup", 30)))
            out_root = payload.get("out", self.args.output)
            target_dir = str(payload.get("target_dir", "")).strip()
            with self.lock:
                live_role, cam = self.live_camera_for(role_or_serial)
                if cam is not None:
                    if target_dir:
                        latest = cam.latest()
                        if latest is None or latest["depth_m"] is None or latest["color_intrinsics"] is None:
                            raise RuntimeError(f"Camera {live_role} has no RGBD frame yet.")
                        out_dir = Path(target_dir).expanduser().resolve()
                        meta = save_rgbd_capture(
                            out_dir,
                            latest["color"],
                            latest["depth_m"],
                            latest["depth_scale"],
                            latest["color_intrinsics"],
                            latest["depth_intrinsics"],
                            {
                                "source": "camera_service_live",
                                "role": live_role,
                                "serial": cam.serial,
                                "width": cam.width,
                                "height": cam.height,
                                "fps": cam.fps,
                                "last_frame_time": latest["last_frame_time"],
                            },
                        )
                    else:
                        out_dir, meta = cam.save_current(out_root)
                else:
                    if target_dir:
                        raise RuntimeError("target_dir is only supported for live camera roles.")
                    out_dir, meta = capture_rgbd(serial, out_root, width, height, fps, warmup, role=role)
            return jsonify({"ok": True, "capture_dir": str(out_dir), "meta": meta})

        @app.route("/camera/capture_all", methods=["POST"])
        def camera_capture_all():
            payload = request.get_json(silent=True) or {}
            self.refresh_config()
            with self.lock:
                results = capture_all(
                    self.config,
                    payload.get("out", self.args.output),
                    width=payload.get("width"),
                    height=payload.get("height"),
                    fps=payload.get("fps"),
                    warmup=payload.get("warmup"),
                )
            return jsonify({"ok": True, "results": results})

        @app.route("/camera/frame/<role>.jpg")
        def camera_frame(role):
            quality = int(request.args.get("quality", 85))
            _, cam = self.live_camera_for(role)
            if cam is None:
                return jsonify({"ok": False, "error": f"Camera {role!r} is not configured for live streaming."}), 404
            return Response(cam.jpeg(quality), mimetype="image/jpeg")

        @app.route("/camera/latest/<role>")
        def camera_latest(role):
            quality = int(request.args.get("quality", 85))
            include_image = request.args.get("image", "1") not in ("0", "false", "False")
            _, cam = self.live_camera_for(role)
            if cam is None:
                return jsonify({"ok": False, "error": f"Camera {role!r} is not configured for live streaming."}), 404
            return jsonify({"ok": True, "frame": cam.latest_json(quality=quality, include_image=include_image)})

        @app.route("/camera/mjpeg/<role>")
        def camera_mjpeg(role):
            quality = int(request.args.get("quality", 85))
            _, cam = self.live_camera_for(role)
            if cam is None:
                return jsonify({"ok": False, "error": f"Camera {role!r} is not configured for live streaming."}), 404

            def gen():
                while True:
                    try:
                        jpg = cam.jpeg(quality)
                    except Exception:
                        time.sleep(0.2)
                        continue
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    time.sleep(0.03)

            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        return app


class CameraWebSocketServer:
    def __init__(self, camera_service, host="0.0.0.0", port=8100):
        self.camera_service = camera_service
        self.host = host
        self.port = int(port)
        self._thread = None
        self._loop = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="camera-ws", daemon=True)
        self._thread.start()

    def _run(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        app = web.Application()
        app.router.add_get("/camera/ws/{role}", self.handle_camera_ws)
        app.router.add_get("/camera/ws", self.handle_camera_ws_query)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, self.host, self.port)
        loop.run_until_complete(site.start())
        print(f"[WS] camera websocket listening on ws://{self.host}:{self.port}/camera/ws/<role>")
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(runner.cleanup())
            loop.close()

    async def handle_camera_ws_query(self, request):
        role = request.query.get("role", "")
        if not role:
            return web.json_response({"ok": False, "error": "role is required"}, status=400)
        return await self._stream_camera_ws(request, role)

    async def handle_camera_ws(self, request):
        role = request.match_info.get("role") or request.query.get("role", "")
        return await self._stream_camera_ws(request, role)

    async def _stream_camera_ws(self, request, role):
        quality = int(request.query.get("quality", 85))
        fps = float(request.query.get("fps", 30))
        interval = max(0.005, 1.0 / max(fps, 1.0))
        ws = web.WebSocketResponse(heartbeat=10.0, max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        try:
            _, cam = self.camera_service.live_camera_for(role)
            if cam is None:
                await ws.send_str(json.dumps({"ok": False, "error": f"Camera {role!r} is not configured"}))
                await ws.close()
                return ws
            await ws.send_str(json.dumps({"ok": True, "role": cam.role, "serial": cam.serial, "type": "camera_info"}))
            while not ws.closed:
                try:
                    jpg = cam.jpeg(quality)
                    await ws.send_bytes(jpg)
                except Exception as exc:
                    await ws.send_str(json.dumps({"ok": False, "error": str(exc), "type": "error"}))
                    await asyncio.sleep(0.2)
                    continue
                await asyncio.sleep(interval)
        finally:
            await ws.close()
        return ws


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    p_capture = sub.add_parser("capture")
    p_capture.add_argument("role_or_serial")
    p_capture.add_argument("--width", type=int, default=0)
    p_capture.add_argument("--height", type=int, default=0)
    p_capture.add_argument("--fps", type=int, default=0)
    p_capture.add_argument("--warmup", type=int, default=-1)

    p_all = sub.add_parser("capture-all")
    p_all.add_argument("--width", type=int, default=0)
    p_all.add_argument("--height", type=int, default=0)
    p_all.add_argument("--fps", type=int, default=0)
    p_all.add_argument("--warmup", type=int, default=-1)

    p_test = sub.add_parser("test-open-all")
    p_test.add_argument("--seconds", type=float, default=8.0)
    p_test.add_argument("--mode", default="roles", help="roles, all, or comma-separated roles.")

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8099)
    p_serve.add_argument("--ws-port", type=int, default=8100)
    p_serve.add_argument("--no-websocket", action="store_true")
    p_serve.add_argument("--open-on-start", default="roles",
                         help="none, roles, all, or comma-separated roles. Default: roles.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.cmd == "list":
        print(json.dumps({"devices": list_realsense_devices(), "config": config}, indent=2, ensure_ascii=False))
        return
    if args.cmd == "capture-all":
        capture_all(
            config,
            args.output,
            width=args.width or None,
            height=args.height or None,
            fps=args.fps or None,
            warmup=None if args.warmup < 0 else args.warmup,
        )
        return
    if args.cmd == "capture":
        role, serial = resolve_camera(config, args.role_or_serial)
        d = config.get("defaults", {})
        if role:
            item = config.get("roles", {}).get(role, {})
            if isinstance(item, dict):
                d = {**d, **item}
        out_dir, _ = capture_rgbd(
            serial=serial,
            out_root=args.output,
            width=args.width or int(d.get("width", 1280)),
            height=args.height or int(d.get("height", 720)),
            fps=args.fps or int(d.get("fps", 30)),
            warmup=int(d.get("warmup", 30)) if args.warmup < 0 else args.warmup,
            role=role,
        )
        print(out_dir)
        return
    if args.cmd == "test-open-all":
        class TestArgs:
            config = args.config
            output = args.output
        service = CameraService(TestArgs())
        roles = service.start_live_roles(args.mode)
        deadline = time.time() + float(args.seconds)
        while time.time() < deadline:
            statuses = {role: service.live[role].status() for role in roles}
            print(json.dumps(statuses, ensure_ascii=False))
            time.sleep(1.0)
        final_statuses = {role: service.live[role].status() for role in roles}
        for cam in service.live.values():
            cam.stop()
        ok = all(item.get("ready") and not item.get("error") for item in final_statuses.values())
        print(json.dumps({"ok": ok, "final": final_statuses}, indent=2, ensure_ascii=False))
        if not ok:
            raise SystemExit(1)
        return
    if args.cmd == "serve":
        service = CameraService(args)
        roles = service.start_live_roles(args.open_on_start)
        if roles:
            print(f"[LIVE] started roles: {', '.join(roles)}")
        if not args.no_websocket:
            ws_server = CameraWebSocketServer(service, host=args.host, port=args.ws_port)
            ws_server.start()
        service.app().run(host=args.host, port=int(args.port), threaded=True)


if __name__ == "__main__":
    main()
