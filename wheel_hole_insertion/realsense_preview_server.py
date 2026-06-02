# -*- coding: utf-8 -*-
"""LAN-accessible RealSense preview server for SSH-only operation."""

import argparse
import base64
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import websocket
from flask import Flask, Response, jsonify, render_template_string, request

if str(SCRIPT_DIR := Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from insert_center_hole import get_current_state  # noqa: E402
from config_loader import CONFIG, cfg_get, load_config, relative_path  # noqa: E402
from infer_center_hole_onnx import crop_mask_to_box, letterbox, nms, sigmoid  # noqa: E402
from rm65_sdk_safe_ik import Rm65SafeIkMover  # noqa: E402


INITIAL_JOINT_DEG = [float(x) for x in cfg_get(CONFIG, "robot", "initial_joint_deg", default=[])]
DEFAULT_MODEL = relative_path(CONFIG, "detection", "model", default="label_dataset/best.onnx")


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


def post_external_json(url, payload, timeout_s=5.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8", "ignore")
            try:
                parsed = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {"raw": body}
            return {
                "status": int(resp.status),
                "body": parsed,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def get_external_json(url, timeout_s=5.0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


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


def find_latest_frontend_plan(frontend_run_dir):
    root = Path(frontend_run_dir).expanduser().resolve()
    candidates = sorted(
        root.glob("frontend_*/result/*_plan.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


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


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RM65 方向盘插孔控制台</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif;
      background: #0d1117;
      color: #e6edf3;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: #0d1117; color: #e6edf3; }
    header {
      min-height: 64px; display: flex; align-items: center; justify-content: space-between;
      gap: 16px; padding: 12px 18px; background: #151b23; border-bottom: 1px solid #30363d;
    }
    .brand { display: flex; flex-direction: column; gap: 4px; min-width: 260px; }
    .brand strong { font-size: 18px; font-weight: 700; letter-spacing: 0; }
    .brand span { color: #8b949e; font-size: 12px; }
    .top-status { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; justify-content: flex-end; }
    .pill { border: 1px solid #30363d; background: #0d1117; border-radius: 999px; padding: 6px 10px; color: #c9d1d9; font-size: 12px; }
    .pill.ok { color: #7ee787; border-color: #238636; }
    .pill.warn { color: #f2cc60; border-color: #9e6a03; }
    main { display: grid; grid-template-columns: minmax(520px, 1fr) 360px; gap: 16px; padding: 16px; }
    .panel { background: #151b23; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
    .panel-head { min-height: 44px; padding: 12px 14px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; }
    .panel-title { font-weight: 700; font-size: 14px; }
    .camera-wrap { position: relative; background: #05070a; }
    img { display: block; width: 100%; height: calc(100vh - 174px); min-height: 520px; object-fit: contain; background: #05070a; }
    .toolbar { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; padding: 12px; border-top: 1px solid #30363d; background: #111820; }
    aside { display: flex; flex-direction: column; gap: 14px; }
    section { background: #151b23; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
    section h2 { margin: 0; padding: 11px 12px; border-bottom: 1px solid #30363d; font-size: 14px; }
    .rows { padding: 8px 12px; }
    .row { display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid #262c36; font-size: 13px; }
    .row:last-child { border-bottom: 0; }
    .k { color: #8b949e; white-space: nowrap; }
    .v { color: #e6edf3; text-align: right; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
    button {
      min-height: 38px; border: 1px solid #3b4654; border-radius: 8px; background: #1f6feb;
      color: #fff; padding: 8px 12px; cursor: pointer; font-weight: 650; font-size: 13px;
    }
    button.secondary { background: #21262d; color: #c9d1d9; }
    button.success { background: #238636; border-color: #2ea043; }
    button.warn { background: #9e6a03; border-color: #bb8009; }
    button.danger { background: #da3633; border-color: #f85149; }
    button.record { background: #a371f7; border-color: #bc8cff; }
    .stack { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 12px; }
    .speed-grid { display: grid; grid-template-columns: 1fr; gap: 10px; padding: 12px; }
    .speed-field { display: grid; grid-template-columns: 1fr 92px; gap: 10px; align-items: center; }
    .speed-field label { color: #8b949e; font-size: 13px; }
    .speed-field input {
      width: 100%; min-height: 34px; border: 1px solid #3b4654; border-radius: 8px;
      background: #0d1117; color: #e6edf3; padding: 6px 8px; font-size: 13px;
    }
    .robot-message { min-height: 54px; padding: 10px 12px; color: #c9d1d9; background: #0d1117; border-top: 1px solid #30363d; font-size: 12px; line-height: 1.5; overflow-wrap: anywhere; }
    @media (max-width: 1100px) { main { grid-template-columns: 1fr; } img { height: auto; } .toolbar { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <strong>RM65 方向盘中心孔插入控制台</strong>
      <span>视觉识别 · 法向对齐 · 预插入 · VLA 数据采集</span>
    </div>
    <div class="top-status">
      <span id="status" class="pill warn">连接中</span>
      <span id="fpsTop" class="pill">FPS --</span>
      <span id="recordTop" class="pill">未录制</span>
      <button class="danger" onclick="robotStop()">急停</button>
    </div>
  </header>
  <main>
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title">相机实时画面</div>
        <button id="yoloBtn" class="secondary" onclick="toggleYolo()">显示 YOLO</button>
      </div>
      <div class="camera-wrap"><img src="/color.mjpg" alt="RealSense color stream"></div>
      <div class="toolbar">
        <button class="secondary" onclick="snapshot()">保存截图</button>
        <button class="secondary" onclick="moveInitial()">回初始位</button>
        <button onclick="runNormalAlignPreinsert()">法向对齐姿态</button>
        <button onclick="runLastPlanPreinsert()">按计划预插入</button>
        <button class="success" onclick="runLastPlanInsert()">插入 10mm</button>
        <button class="success" onclick="runAutoAlignInsertOpen()">抓取方向盘上料</button>
      </div>
    </div>
    <aside>
      <section>
        <h2>识别状态</h2>
        <div class="rows">
          <div class="row"><span class="k">彩色分辨率</span><span id="color" class="v"></span></div>
          <div class="row"><span class="k">深度图</span><span id="depth" class="v"></span></div>
          <div class="row"><span class="k">最近帧</span><span id="last" class="v"></span></div>
          <div class="row"><span class="k">YOLO</span><span id="yolo" class="v"></span></div>
          <div class="row"><span class="k">置信度</span><span id="score" class="v"></span></div>
          <div class="row"><span class="k">中心像素</span><span id="center" class="v"></span></div>
          <div class="row"><span class="k">手眼标定</span><span id="handEye" class="v"></span></div>
        </div>
      </section>
      <section>
        <h2>运动流程</h2>
        <div class="stack">
          <button onclick="runLivePreinsert()">实时预插入</button>
          <button onclick="runRollSearchPreinsert()">保持姿态预插入</button>
          <button onclick="runNormalAlignPreinsert()">法向对齐姿态</button>
          <button onclick="runLastPlanPreinsert()">按计划预插入</button>
          <button class="success" onclick="runAutoAlignInsertOpen()">抓取方向盘上料</button>
          <button class="success" onclick="runPostInsertPlace()">插入后放置流程</button>
          <button class="warn" onclick="runPostInsertReturn()">原路返回</button>
        </div>
      </section>
      <section>
        <h2>速度配置</h2>
        <div class="speed-grid">
          <div class="speed-field">
            <label for="initialSpeed">回初始位速度</label>
            <input id="initialSpeed" type="number" min="1" max="100" step="1" value="50">
          </div>
          <div class="speed-field">
            <label for="movejSpeed">MoveJ 速度</label>
            <input id="movejSpeed" type="number" min="1" max="100" step="1" value="20">
          </div>
          <div class="speed-field">
            <label for="insertSpeed">插入速度</label>
            <input id="insertSpeed" type="number" min="1" max="100" step="1" value="10">
          </div>
        </div>
      </section>
      <section>
        <h2>夹爪与保存</h2>
        <div class="stack">
          <button class="secondary" onclick="gripperOpen()">打开夹爪</button>
          <button class="secondary" onclick="gripperClose()">关闭夹爪</button>
          <button id="recordStartBtn" class="record" onclick="recordStart()">开始保存</button>
          <button id="recordStopBtn" class="warn" onclick="recordStop()">停止保存</button>
        </div>
        <div class="rows">
          <div class="row"><span class="k">保存状态</span><span id="recording" class="v"></span></div>
          <div class="row"><span class="k">帧数</span><span id="recordFrames" class="v"></span></div>
          <div class="row"><span class="k">输出目录</span><span id="recordPath" class="v"></span></div>
          <div class="row"><span class="k">截图</span><span id="shot" class="v"></span></div>
        </div>
      </section>
      <section>
        <h2>底盘与升降</h2>
        <div class="speed-grid">
          <div class="speed-field">
            <label for="liftHeight">升降高度 m</label>
            <input id="liftHeight" type="number" step="0.01" value="0.05">
          </div>
          <div class="speed-field">
            <label for="liftSpeed">升降速度</label>
            <input id="liftSpeed" type="number" step="0.01" value="0.03">
          </div>
          <div class="speed-field">
            <label for="baseDistance">前后距离</label>
            <input id="baseDistance" type="number" step="0.1" value="-2.5">
          </div>
          <div class="speed-field">
            <label for="baseSpeed">前后速度</label>
            <input id="baseSpeed" type="number" step="0.01" value="0.1">
          </div>
          <div class="speed-field">
            <label for="rotateAngleDeg">旋转角度 deg</label>
            <input id="rotateAngleDeg" type="number" step="1" value="-90">
          </div>
          <div class="speed-field">
            <label for="rotateSpeed">旋转速度</label>
            <input id="rotateSpeed" type="number" step="0.01" value="0.2">
          </div>
        </div>
        <div class="stack">
          <button class="secondary" onclick="liftMove()">升降机移动</button>
          <button class="secondary" onclick="baseStepForward()">整体前后移动</button>
          <button class="secondary" onclick="baseStepRotate()">整体旋转</button>
        </div>
      </section>
      <section>
        <h2>机器人状态</h2>
        <div id="robot" class="robot-message"></div>
      </section>
    </aside>
  </main>
  <script>
    let speedDefaultsLoaded = false;
    function speedPayload() {
      return {
        initial_movej_speed: Number(document.getElementById('initialSpeed').value || 50),
        movej_speed: Number(document.getElementById('movejSpeed').value || 20),
        frontend_insert_speed: Number(document.getElementById('insertSpeed').value || 10),
      };
    }
    async function postJson(url, payload = null) {
      const options = {method: 'POST'};
      if (payload !== null) {
        options.headers = {'Content-Type': 'application/json'};
        options.body = JSON.stringify(payload);
      }
      const r = await fetch(url, options);
      return await r.json();
    }
    async function refresh() {
      try {
        const r = await fetch('/status');
        const s = await r.json();
        document.getElementById('status').textContent = s.running ? '相机运行' : '相机停止';
        document.getElementById('status').className = s.running ? 'pill ok' : 'pill warn';
        document.getElementById('color').textContent = `${s.width} x ${s.height}`;
        document.getElementById('depth').textContent = s.depth_ready ? '正常' : '无数据';
        document.getElementById('fpsTop').textContent = `FPS ${s.actual_fps.toFixed(1)}`;
        document.getElementById('last').textContent = s.last_frame_time || '';
        document.getElementById('yolo').textContent = s.yolo_enabled ? (s.yolo_found ? '已识别' : '开启') : '关闭';
        document.getElementById('score').textContent = s.yolo_score == null ? '' : s.yolo_score.toFixed(3);
        document.getElementById('center').textContent = s.yolo_center ? `[${s.yolo_center[0].toFixed(1)}, ${s.yolo_center[1].toFixed(1)}]` : '';
        document.getElementById('handEye').textContent = s.hand_eye_valid ? '有效' : '无效';
        document.getElementById('yoloBtn').textContent = s.yolo_enabled ? '隐藏 YOLO' : '显示 YOLO';
        document.getElementById('recording').textContent = s.recording ? '录制中' : '未录制';
        document.getElementById('recordTop').textContent = s.recording ? '录制中' : '未录制';
        document.getElementById('recordTop').className = s.recording ? 'pill ok' : 'pill';
        document.getElementById('recordFrames').textContent = String(s.record_frames || 0);
        document.getElementById('recordPath').textContent = s.record_path || '';
        if (!speedDefaultsLoaded && s.speed_defaults) {
          document.getElementById('initialSpeed').value = s.speed_defaults.initial_movej_speed;
          document.getElementById('movejSpeed').value = s.speed_defaults.movej_speed;
          document.getElementById('insertSpeed').value = s.speed_defaults.frontend_insert_speed;
          speedDefaultsLoaded = true;
        }
        const rr = await fetch('/robot/status');
        const rs = await rr.json();
        document.getElementById('robot').textContent = rs.busy ? `执行中：${rs.message || ''}` : (rs.message || '空闲');
      } catch (e) {
        document.getElementById('status').textContent = '离线';
        document.getElementById('status').className = 'pill warn';
      }
    }
    async function snapshot() {
      const s = await postJson('/snapshot');
      document.getElementById('shot').textContent = s.path || s.error || '';
    }
    async function moveInitial() {
      if (!confirm(`确认让机械臂回到初始位置？速度=${speedPayload().initial_movej_speed}`)) return;
      document.getElementById('robot').textContent = '发送回初始位指令...';
      const s = await postJson('/robot/move_initial', speedPayload());
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function runLivePreinsert() {
      if (!confirm(`确认使用当前 RGBD 画面识别并移动到预插入位置？MoveJ速度=${speedPayload().movej_speed}`)) return;
      const s = await postJson('/robot/live_preinsert', speedPayload());
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function runRollSearchPreinsert() {
      if (!confirm(`确认使用当前 RGBD 画面识别，并保持当前工具姿态移动到预插入位置？MoveJ速度=${speedPayload().movej_speed}`)) return;
      const s = await postJson('/robot/live_preinsert_roll_search', speedPayload());
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function runNormalAlignPreinsert() {
      if (!confirm(`确认使用当前 RGBD 画面识别，只旋转姿态对齐中间圆环法向？MoveJ速度=${speedPayload().movej_speed}`)) return;
      const s = await postJson('/robot/live_preinsert_normal_align', speedPayload());
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function runLastPlanPreinsert() {
      if (!confirm(`确认不重新拍照，读取上次法向对齐保存的计划，并移动到预插入位置？MoveJ速度=${speedPayload().movej_speed}`)) return;
      const s = await postJson('/robot/move_last_plan_preinsert', speedPayload());
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function runLastPlanInsert() {
      if (!confirm(`确认不重新拍照，读取上次预插入计划，并沿当前工具轴小步前伸 10mm？插入速度=${speedPayload().frontend_insert_speed}`)) return;
      const s = await postJson('/robot/insert_last_plan', speedPayload());
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function runAutoAlignInsertOpen() {
      if (!confirm(`确认执行抓取方向盘上料？将依次执行：闭合夹爪、法向对齐姿态、按计划预插入、4次插入10mm、打开夹爪、升降机上升0.02m、底盘后退0.3m、机械臂到垂直预释放姿态、升降机上升到0.8、底盘旋转90°、机械臂到释放姿势、底盘前进0.3m并关闭避障、升降机高度到0.66。MoveJ速度=${speedPayload().movej_speed}，插入速度=${speedPayload().frontend_insert_speed}`)) return;
      const s = await postJson('/robot/auto_align_insert_open', speedPayload());
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function runPostInsertPlace() {
      if (!confirm('确认执行插入后放置流程？将依次执行升降机抬升、底盘后退、旋转、前进、机械臂到下降姿态、升降机下降、闭合夹爪、升降机抬起。')) return;
      const s = await postJson('/robot/post_insert_place_sequence', speedPayload());
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function runPostInsertReturn() {
      if (!confirm('确认执行原路返回？将依次执行升降机回落、底盘后退、旋转回正、前进回到原路位置。')) return;
      const s = await postJson('/robot/post_insert_return_sequence', speedPayload());
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function robotStop() {
      const s = await postJson('/robot/stop');
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function gripperOpen() {
      if (!confirm('确认打开夹爪？')) return;
      const s = await postJson('/gripper/open');
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function gripperClose() {
      if (!confirm('确认关闭夹爪？')) return;
      const s = await postJson('/gripper/close');
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function toggleYolo() {
      const s = await postJson('/yolo/toggle');
      document.getElementById('yolo').textContent = s.enabled ? '开启' : '关闭';
    }
    async function recordStart() {
      const s = await postJson('/recording/start');
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function recordStop() {
      const s = await postJson('/recording/stop');
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function liftMove() {
      const payload = {
        execMode: 2,
        speed: Number(document.getElementById('liftSpeed').value || 0.03),
        height: Number(document.getElementById('liftHeight').value || 0.05),
      };
      if (!confirm(`确认升降机移动？height=${payload.height}, speed=${payload.speed}`)) return;
      const s = await postJson('/mobile/lift', payload);
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function baseStepForward() {
      const payload = {
        distance: Number(document.getElementById('baseDistance').value || -2.5),
        speed: Number(document.getElementById('baseSpeed').value || 0.1),
      };
      if (!confirm(`确认整体前后移动？distance=${payload.distance}, speed=${payload.speed}`)) return;
      const s = await postJson('/mobile/step_forward', payload);
      document.getElementById('robot').textContent = s.message || s.error || '';
    }
    async function baseStepRotate() {
      const angleDeg = Number(document.getElementById('rotateAngleDeg').value || -90);
      const payload = {
        angle: angleDeg * Math.PI / 180.0,
        speed: Number(document.getElementById('rotateSpeed').value || 0.2),
      };
      if (!confirm(`确认整体旋转？angle=${angleDeg}° (${payload.angle.toFixed(4)} rad), speed=${payload.speed}`)) return;
      const s = await postJson('/mobile/step_rotate', payload);
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
        self.color_intrinsics = None
        self.depth_scale = None
        self.display = None
        self.running = False
        self.last_frame_time = ""
        self.actual_fps = 0.0
        self.yolo_enabled = False
        self.yolo_result = None
        self.yolo_last_time = 0.0
        self.yolo_net = None
        self._thread = None
        self._pipeline = None

    def start(self):
        if self._thread is not None:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        service_url = self.args.camera_service_url.rstrip("/")
        ws_url = self.args.camera_service_ws_url.rstrip("/")
        role = self.args.camera_role
        try:
            post_external_json(
                f"{service_url}/camera/start",
                {"role": role},
                timeout_s=self.args.camera_service_timeout_s,
            )
        except Exception as exc:
            with self.lock:
                self.last_frame_time = f"camera service error: {exc}"
        with self.lock:
            self.depth_scale = None
            self.color_intrinsics = None
            self.depth_m = None
        last_t = time.time()
        interval = max(0.02, 1.0 / max(float(self.args.fps), 1.0))

        def update_from_color(color, color_intrinsics=None, frame_time=None):
            nonlocal last_t
            now = time.time()
            dt = max(now - last_t, 1e-6)
            last_t = now
            with self.lock:
                self.color = color
                self.depth_m = None
                if color_intrinsics is not None:
                    self.color_intrinsics = color_intrinsics
                self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1.0 / dt) if self.actual_fps else 1.0 / dt
                self.last_frame_time = frame_time or time.strftime("%H:%M:%S")
                yolo_enabled = self.yolo_enabled
                yolo_due = now - self.yolo_last_time >= float(self.args.yolo_interval)
            if yolo_enabled and yolo_due:
                self._update_yolo(color, now)

        def poll_http_once():
            latest = get_external_json(
                f"{service_url}/camera/latest/{role}?quality={int(self.args.jpeg_quality)}&image=1",
                timeout_s=self.args.camera_service_timeout_s,
            )
            frame = latest.get("frame", {})
            jpeg_b64 = frame.get("jpeg_base64")
            if not jpeg_b64:
                raise RuntimeError("camera service returned no jpeg_base64")
            arr = np.frombuffer(base64.b64decode(jpeg_b64), dtype=np.uint8)
            color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if color is None:
                raise RuntimeError("failed to decode camera service JPEG")
            update_from_color(color, frame.get("color_intrinsics"), frame.get("last_frame_time"))

        def run_websocket_loop():
            url = f"{ws_url}/camera/ws/{role}?quality={int(self.args.jpeg_quality)}&fps={int(self.args.fps)}"
            ws = websocket.create_connection(
                url,
                timeout=float(self.args.camera_service_timeout_s),
                http_proxy_host=None,
                http_proxy_port=None,
                http_no_proxy=["127.0.0.1", "localhost", "*"],
            )
            try:
                while self.running:
                    msg = ws.recv()
                    if isinstance(msg, str):
                        try:
                            payload = json.loads(msg)
                        except json.JSONDecodeError:
                            payload = {}
                        if payload.get("type") == "error":
                            raise RuntimeError(payload.get("error", "camera websocket error"))
                        continue
                    arr = np.frombuffer(msg, dtype=np.uint8)
                    color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if color is None:
                        continue
                    update_from_color(color)
            finally:
                try:
                    ws.close()
                except Exception:
                    pass

        try:
            while self.running:
                try:
                    if self.args.disable_camera_websocket:
                        poll_http_once()
                        time.sleep(interval)
                    else:
                        run_websocket_loop()
                except Exception as exc:
                    with self.lock:
                        self.last_frame_time = f"camera service error: {exc}"
                    try:
                        poll_http_once()
                    except Exception:
                        time.sleep(0.5)
        finally:
            self.running = False

    def _load_yolo_net(self):
        if self.yolo_net is None:
            self.yolo_net = cv2.dnn.readNetFromONNX(str(Path(self.args.model).expanduser().resolve()))
        return self.yolo_net

    def _infer_yolo_mask(self, image):
        orig_h, orig_w = image.shape[:2]
        imgsz = int(self.args.imgsz)
        inp, scale, pad_x, pad_y = letterbox(image, imgsz)
        blob = cv2.dnn.blobFromImage(inp, 1.0 / 255.0, (imgsz, imgsz), swapRB=True, crop=False)
        net = self._load_yolo_net()
        net.setInput(blob)
        pred, proto = net.forward(net.getUnconnectedOutLayersNames())
        pred = pred[0].transpose(1, 0)
        proto = proto[0]

        boxes = []
        scores = []
        coeffs = []
        for row in pred:
            score = float(row[4])
            if score < float(self.args.yolo_conf):
                continue
            cx, cy, bw, bh = [float(v) for v in row[:4]]
            boxes.append([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0])
            scores.append(score)
            coeffs.append(row[5:].astype(np.float32))
        if not boxes:
            return None

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        coeffs = np.asarray(coeffs, dtype=np.float32)
        keep = nms(boxes, scores, float(self.args.yolo_iou))
        if not keep:
            return None
        best = max(keep, key=lambda idx: float(scores[idx]))

        proto_flat = proto.reshape(proto.shape[0], -1)
        mask_logits = coeffs[best] @ proto_flat
        mask = sigmoid(mask_logits).reshape(proto.shape[1], proto.shape[2])
        mask = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        mask_u8 = (mask >= float(self.args.mask_threshold)).astype(np.uint8)
        mask_u8 = crop_mask_to_box(mask_u8, boxes[best])

        unpad_h = int(round(orig_h * scale))
        unpad_w = int(round(orig_w * scale))
        unpad = mask_u8[pad_y:pad_y + unpad_h, pad_x:pad_x + unpad_w]
        mask_orig = cv2.resize(unpad, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours((mask_orig * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        center = None
        if abs(float(moments["m00"])) > 1e-6:
            center = [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]

        box = boxes[best].copy()
        box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
        box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, orig_w - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, orig_h - 1)
        return {
            "mask": mask_orig.astype(bool),
            "contour": contour,
            "score": float(scores[best]),
            "center": center,
            "box": [float(v) for v in box],
        }

    def _draw_yolo_overlay(self, image, result):
        overlay = image.copy()
        if result is None:
            cv2.putText(overlay, "YOLO: no center_hole", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return overlay
        green = np.zeros_like(image)
        green[:, :, 1] = 255
        mask = result["mask"]
        overlay = np.where(mask[:, :, None], cv2.addWeighted(image, 0.62, green, 0.38, 0), overlay)
        cv2.drawContours(overlay, [result["contour"]], -1, (0, 255, 0), 2)
        x1, y1, x2, y2 = [int(round(v)) for v in result["box"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if result["center"] is not None:
            cx, cy = [int(round(v)) for v in result["center"]]
            cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
        cv2.putText(overlay, f"YOLO center_hole score={result['score']:.3f}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)
        cv2.putText(overlay, f"YOLO center_hole score={result['score']:.3f}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1)
        return overlay

    def _update_yolo(self, color, timestamp):
        try:
            result = self._infer_yolo_mask(color)
            display = self._draw_yolo_overlay(color, result)
        except Exception as exc:
            result = {"error": str(exc), "score": None, "center": None}
            display = color.copy()
            cv2.putText(display, f"YOLO error: {exc}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        with self.lock:
            self.yolo_result = result
            self.display = display
            self.yolo_last_time = timestamp

    def set_yolo_enabled(self, enabled):
        with self.lock:
            self.yolo_enabled = bool(enabled)
            if not self.yolo_enabled:
                self.display = None
                self.yolo_result = None
        return self.yolo_enabled

    def toggle_yolo(self):
        with self.lock:
            enabled = not self.yolo_enabled
        return self.set_yolo_enabled(enabled)

    def get_jpeg(self):
        with self.lock:
            if self.color is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "waiting for RealSense...", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif self.yolo_enabled and self.display is not None:
                frame = self.display.copy()
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

    def get_record_frame(self):
        with self.lock:
            if self.color is None:
                return None
            color = self.color.copy()
            intr = None if self.color_intrinsics is None else dict(self.color_intrinsics)
            frame_time = self.last_frame_time
        return color, intr, frame_time

    def save_current_capture(self, root):
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        service_url = self.args.camera_service_url.rstrip("/")
        response = post_external_json(
            f"{service_url}/camera/capture",
            {"role": self.args.camera_role, "target_dir": str(root)},
            timeout_s=max(5.0, float(self.args.camera_service_timeout_s)),
        )
        capture_dir = Path(response["body"].get("capture_dir", "")).expanduser().resolve()
        if not capture_dir.exists():
            return None
        if capture_dir != root:
            for name in ("color.png", "depth_raw.npy", "depth_mm.png", "depth_colormap.png", "depth_overlay.png", "intrinsics.json"):
                src = capture_dir / name
                dst = root / name
                if src.exists():
                    dst.write_bytes(src.read_bytes())
        return root

    def status(self):
        with self.lock:
            h, w = self.color.shape[:2] if self.color is not None else (self.args.height, self.args.width)
            yolo_result = self.yolo_result or {}
            return {
                "running": bool(self.running),
                "width": int(w),
                "height": int(h),
                "depth_ready": self.depth_m is not None,
                "actual_fps": float(self.actual_fps),
                "last_frame_time": self.last_frame_time,
                "source": "camera_service",
                "camera_role": self.args.camera_role,
                "camera_transport": "http" if self.args.disable_camera_websocket else "websocket",
                "yolo_enabled": bool(self.yolo_enabled),
                "yolo_found": bool(yolo_result and yolo_result.get("center") is not None),
                "yolo_score": yolo_result.get("score"),
                "yolo_center": yolo_result.get("center"),
            }


class EpisodeRecorder:
    def __init__(self, streamer, args):
        self.streamer = streamer
        self.args = args
        self.lock = threading.Lock()
        self.recording = False
        self.frame_count = 0
        self.episode_dir = None
        self.message = ""
        self._thread = None
        self._stop_event = threading.Event()

    def status(self):
        with self.lock:
            return {
                "recording": bool(self.recording),
                "record_frames": int(self.frame_count),
                "record_path": "" if self.episode_dir is None else str(self.episode_dir),
                "record_message": self.message,
            }

    def start(self):
        with self.lock:
            if self.recording:
                return False, f"已经在保存: {self.episode_dir}"
            self._stop_event.clear()
            root = Path(self.args.recording_dir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            existing = sorted(root.glob("episode_*"))
            next_index = len(existing)
            episode_dir = root / f"episode_{next_index:06d}"
            episode_dir.mkdir(parents=True, exist_ok=False)
            (episode_dir / "videos").mkdir()
            (episode_dir / "data").mkdir()
            (episode_dir / "meta").mkdir()
            self.episode_dir = episode_dir
            self.frame_count = 0
            self.message = "保存中"
            self.recording = True
            self._thread = threading.Thread(target=self._run, args=(episode_dir, next_index), daemon=True)
            self._thread.start()
            return True, f"开始保存: {episode_dir}"

    def stop(self):
        with self.lock:
            if not self.recording:
                return False, "当前没有保存任务"
            episode_dir = self.episode_dir
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
        with self.lock:
            msg = f"保存完成: {episode_dir}"
            self.message = msg
            return True, msg

    def _read_robot_state(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(float(self.args.recording_robot_timeout_s))
            sock.connect((self.args.robot_ip, int(self.args.robot_port)))
            state = get_current_state(sock)
        finally:
            sock.close()
        if not state:
            return None, None
        pose, joint = state
        return [float(x) for x in pose], [float(x) for x in joint]

    def _write_meta(self, episode_dir, episode_index, frame_count, fps, intrinsics):
        meta_dir = episode_dir / "meta"
        info = {
            "codebase_version": "lerobot-like-v0",
            "robot_type": "rm65b",
            "fps": float(fps),
            "total_episodes": 1,
            "total_frames": int(frame_count),
            "data_path": "data/frames.jsonl",
            "video_path": "videos/camera_rgb.mp4",
            "features": {
                "observation.images.camera_rgb": {"dtype": "video", "shape": [int(self.args.height), int(self.args.width), 3]},
                "observation.state": {"dtype": "float32", "shape": [12], "names": ["x", "y", "z", "rx", "ry", "rz", "j1", "j2", "j3", "j4", "j5", "j6"]},
                "timestamp": {"dtype": "float64", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
            },
        }
        episode = {
            "episode_index": int(episode_index),
            "length": int(frame_count),
            "task": "方向盘中心孔插入",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "color_intrinsics": intrinsics,
        }
        tasks = [{"task_index": 0, "task": "方向盘中心孔插入"}]
        with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        with open(meta_dir / "episode.json", "w", encoding="utf-8") as f:
            json.dump(episode, f, indent=2, ensure_ascii=False)
        with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
            for item in tasks:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _try_write_parquet(self, rows, out_path):
        try:
            import pandas as pd  # noqa: WPS433
            df = pd.DataFrame(rows)
            df.to_parquet(out_path, index=False)
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _run(self, episode_dir, episode_index):
        fps = float(self.args.recording_fps)
        period = 1.0 / max(fps, 0.1)
        frames_path = episode_dir / "data" / "frames.jsonl"
        video_path = episode_dir / "videos" / "camera_rgb.mp4"
        rows = []
        writer = None
        intrinsics = None
        start_mono = time.monotonic()
        start_unix = time.time()

        try:
            with open(frames_path, "w", encoding="utf-8") as frames_file:
                next_t = time.monotonic()
                while not self._stop_event.is_set():
                    now = time.monotonic()
                    if now < next_t:
                        time.sleep(min(next_t - now, 0.02))
                        continue
                    next_t += period

                    frame_pack = self.streamer.get_record_frame()
                    if frame_pack is None:
                        continue
                    color, intr, camera_frame_time = frame_pack
                    intrinsics = intrinsics or intr
                    pose, joint = self._read_robot_state()
                    if pose is None or joint is None:
                        continue

                    if writer is None:
                        h, w = color.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot open video writer: {video_path}")

                    frame_index = len(rows)
                    timestamp = time.time() - start_unix
                    writer.write(color)
                    row = {
                        "episode_index": int(episode_index),
                        "frame_index": int(frame_index),
                        "timestamp": float(timestamp),
                        "monotonic_time": float(time.monotonic() - start_mono),
                        "camera_frame_time": camera_frame_time,
                        "observation.state": [float(x) for x in (pose + joint)],
                        "observation.robot_pose": pose,
                        "observation.robot_joint_deg": joint,
                        "observation.images.camera_rgb": "videos/camera_rgb.mp4",
                    }
                    rows.append(row)
                    frames_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    frames_file.flush()
                    with self.lock:
                        self.frame_count = len(rows)

            if writer is not None:
                writer.release()
                writer = None
            parquet_ok, parquet_error = self._try_write_parquet(rows, episode_dir / "data" / "frames.parquet")
            self._write_meta(episode_dir, episode_index, len(rows), fps, intrinsics)
            with open(episode_dir / "README.md", "w", encoding="utf-8") as f:
                f.write("# RM65 steering wheel insertion episode\n\n")
                f.write("LeRobot-like layout: video in `videos/`, synchronized robot state rows in `data/frames.jsonl`.\n")
                if not parquet_ok:
                    f.write(f"\nParquet was not written: {parquet_error}\n")
            with self.lock:
                self.message = f"保存完成，帧数 {len(rows)}"
        except Exception as exc:
            if writer is not None:
                writer.release()
            with self.lock:
                self.message = f"保存失败: {exc}"
        finally:
            with self.lock:
                self.recording = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--serial", default=cfg_get(CONFIG, "camera", "serial", default=""))
    parser.add_argument("--width", type=int, default=int(cfg_get(CONFIG, "camera", "width", default=1280)))
    parser.add_argument("--height", type=int, default=int(cfg_get(CONFIG, "camera", "height", default=720)))
    parser.add_argument("--fps", type=int, default=int(cfg_get(CONFIG, "camera", "fps", default=30)))
    parser.add_argument("--warmup", type=int, default=int(cfg_get(CONFIG, "camera", "warmup", default=15)))
    parser.add_argument("--camera-service-url", default=str(cfg_get(CONFIG, "camera_service", "url", default="http://127.0.0.1:8099")))
    parser.add_argument("--camera-service-ws-url", default=str(cfg_get(CONFIG, "camera_service", "ws_url", default="ws://127.0.0.1:8100")))
    parser.add_argument("--camera-role", default=str(cfg_get(CONFIG, "camera_service", "default_role", default="right_arm")))
    parser.add_argument("--camera-service-timeout-s", type=float, default=float(cfg_get(CONFIG, "camera_service", "timeout_s", default=3.0)))
    parser.add_argument("--disable-camera-websocket", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--snapshot-dir", default=str(relative_path(CONFIG, "paths", "snapshot_dir", default="preview_snapshots")))
    parser.add_argument("--frontend-run-dir", default=str(relative_path(CONFIG, "paths", "frontend_run_dir", default="frontend_runs")))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--imgsz", type=int, default=int(cfg_get(CONFIG, "detection", "imgsz", default=960)))
    parser.add_argument("--yolo-conf", type=float, default=float(cfg_get(CONFIG, "detection", "yolo_conf", default=0.05)))
    parser.add_argument("--yolo-iou", type=float, default=float(cfg_get(CONFIG, "detection", "yolo_iou", default=0.45)))
    parser.add_argument("--mask-threshold", type=float, default=float(cfg_get(CONFIG, "detection", "mask_threshold", default=0.5)))
    parser.add_argument("--yolo-interval", type=float, default=0.7,
                        help="Seconds between YOLO overlay updates when enabled.")
    parser.add_argument("--robot-ip", default=cfg_get(CONFIG, "robot", "ip", default="169.254.128.21"))
    parser.add_argument("--robot-port", type=int, default=int(cfg_get(CONFIG, "robot", "port", default=8080)))
    parser.add_argument("--initial-movej-speed", type=int, default=int(cfg_get(CONFIG, "motion", "initial_movej_speed", default=50)))
    parser.add_argument("--right-offset-m", type=float, default=float(cfg_get(CONFIG, "insertion", "observed_right_offset_m", default=0.0025)))
    parser.add_argument("--up-offset-m", type=float, default=float(cfg_get(CONFIG, "insertion", "observed_up_offset_m", default=0.0030)))
    parser.add_argument("--tcp-to-tip-m", type=float, default=float(cfg_get(CONFIG, "tool", "tcp_to_tip_m", default=0.1613)))
    parser.add_argument("--tip-tcp-m", default=",".join(str(x) for x in cfg_get(CONFIG, "tool", "tip_tcp_m", default=[])))
    parser.add_argument("--preinsert-distance-m", type=float, default=float(cfg_get(CONFIG, "insertion", "preinsert_distance_m", default=0.08)))
    parser.add_argument("--insert-depth-m", type=float, default=float(cfg_get(CONFIG, "insertion", "insert_depth_m", default=0.002)))
    parser.add_argument("--max-preinsert-move-m", type=float, default=float(cfg_get(CONFIG, "insertion", "max_preinsert_move_m", default=0.35)))
    parser.add_argument("--movej-speed", type=int, default=int(cfg_get(CONFIG, "motion", "movej_speed", default=20)))
    parser.add_argument("--max-j6-step-deg", type=float, default=float(cfg_get(CONFIG, "motion", "max_j6_step_deg", default=90.0)))
    parser.add_argument("--roll-search-deg", default="0,15,-15,30,-30,45,-45,60,-60,90,-90,120,-120,150,-150,180",
                        help="Comma-separated wrist roll candidates for the roll-search preinsert button.")
    parser.add_argument("--gripper-device-id", type=int, default=int(cfg_get(CONFIG, "gripper", "device_id", default=9)))
    parser.add_argument("--gripper-open-position", type=int, default=int(cfg_get(CONFIG, "gripper", "open_position", default=255)),
                        help="Frontend corrected value for opening the reversed gripper.")
    parser.add_argument("--gripper-close-position", type=int, default=int(cfg_get(CONFIG, "gripper", "close_position", default=0)),
                        help="Frontend corrected value for closing the reversed gripper.")
    parser.add_argument("--gripper-speed", type=int, default=int(cfg_get(CONFIG, "gripper", "speed", default=255)))
    parser.add_argument("--gripper-force", type=int, default=int(cfg_get(CONFIG, "gripper", "force", default=255)))
    parser.add_argument("--gripper-timeout-s", type=int, default=int(cfg_get(CONFIG, "gripper", "timeout_s", default=5)))
    parser.add_argument("--frontend-insert-distance-m", type=float, default=float(cfg_get(CONFIG, "insertion", "frontend_insert_distance_m", default=0.01)))
    parser.add_argument("--frontend-max-insert-distance-m", type=float, default=float(cfg_get(CONFIG, "insertion", "frontend_max_insert_distance_m", default=0.01)))
    parser.add_argument("--frontend-insert-speed", type=int, default=int(cfg_get(CONFIG, "motion", "frontend_insert_speed", default=10)))
    parser.add_argument("--recording-dir", default=str(relative_path(CONFIG, "paths", "recording_dir", default="vla_recordings")))
    parser.add_argument("--recording-fps", type=float, default=float(cfg_get(CONFIG, "recording", "fps", default=10.0)))
    parser.add_argument("--recording-robot-timeout-s", type=float, default=float(cfg_get(CONFIG, "recording", "robot_timeout_s", default=1.0)))
    parser.add_argument("--mobile-base-url", default=str(cfg_get(CONFIG, "mobile_base", "base_url", default="http://192.168.2.228:5001")))
    parser.add_argument("--mobile-timeout-s", type=float, default=float(cfg_get(CONFIG, "mobile_base", "timeout_s", default=5.0)))
    parser.add_argument("--mobile-stop-timeout-s", type=float, default=float(cfg_get(CONFIG, "mobile_base", "stop_timeout_s", default=0.8)))
    parser.add_argument("--post-insert-poses", default=str(relative_path(CONFIG, "post_insert_sequence", "poses_file", default="table_place_poses.json")))
    return parser.parse_args()


def main():
    args = parse_args()
    streamer = RealSenseStreamer(args)
    streamer.start()
    recorder = EpisodeRecorder(streamer, args)
    robot_lock = threading.Lock()
    robot_state = {"busy": False, "message": "", "last_plan": ""}
    emergency_event = threading.Event()

    def clear_emergency_for_new_motion():
        emergency_event.clear()

    def assert_not_emergency(label=""):
        if emergency_event.is_set():
            suffix = f" at {label}" if label else ""
            raise RuntimeError(f"emergency stop requested{suffix}")

    def configured_mobile_stop_requests():
        runtime_config = load_config()
        requests_cfg = cfg_get(runtime_config, "mobile_base", "stop_requests", default=[]) or []
        cleaned = []
        for item in requests_cfg:
            if not isinstance(item, dict):
                continue
            endpoint = str(item.get("endpoint", "")).strip()
            if not endpoint:
                continue
            if not endpoint.startswith("/"):
                endpoint = "/" + endpoint
            payload = item.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            cleaned.append((endpoint, payload))
        return cleaned

    def send_arm_stop_now():
        mover = Rm65SafeIkMover(robot_ip=args.robot_ip, robot_port=args.robot_port, speed=1)
        try:
            mover.connect()
            ret = mover.arm.rm_set_arm_stop()
            return f"arm_stop ret={ret}"
        finally:
            try:
                mover.close()
            except Exception:
                pass

    def send_mobile_stop_now():
        results = []
        for endpoint, payload in configured_mobile_stop_requests():
            url = args.mobile_base_url.rstrip("/") + endpoint
            try:
                result = post_external_json(url, payload, timeout_s=args.mobile_stop_timeout_s)
                results.append(f"{endpoint}: ok {result}")
            except Exception as exc:
                results.append(f"{endpoint}: {exc}")
        if not results:
            results.append("no mobile stop_requests configured")
        return results

    app = Flask(__name__)

    def speed_from_request(name, default_value):
        payload = request.get_json(silent=True) or {}
        value = payload.get(name, default_value)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = int(default_value)
        return max(1, min(100, value))

    def request_speed_config():
        return {
            "initial_movej_speed": speed_from_request("initial_movej_speed", args.initial_movej_speed),
            "movej_speed": speed_from_request("movej_speed", args.movej_speed),
            "frontend_insert_speed": speed_from_request("frontend_insert_speed", args.frontend_insert_speed),
        }

    @app.route("/")
    def index():
        return render_template_string(PAGE)

    @app.route("/status")
    def status():
        data = streamer.status()
        data.update(recorder.status())
        data["speed_defaults"] = {
            "initial_movej_speed": int(args.initial_movej_speed),
            "movej_speed": int(args.movej_speed),
            "frontend_insert_speed": int(args.frontend_insert_speed),
        }
        data["hand_eye_valid"] = bool(cfg_get(CONFIG, "hand_eye", "valid", default=False))
        return jsonify(data)

    @app.route("/snapshot", methods=["POST"])
    def snapshot():
        path = streamer.snapshot()
        if path is None:
            return jsonify({"ok": False, "error": "no frame yet"}), 503
        return jsonify({"ok": True, "path": str(path)})

    @app.route("/recording/start", methods=["POST"])
    def recording_start():
        ok, message = recorder.start()
        status_code = 200 if ok else 409
        return jsonify({"ok": ok, "message": message, "error": "" if ok else message}), status_code

    @app.route("/recording/stop", methods=["POST"])
    def recording_stop():
        ok, message = recorder.stop()
        status_code = 200 if ok else 409
        return jsonify({"ok": ok, "message": message, "error": "" if ok else message}), status_code

    @app.route("/robot/move_initial", methods=["POST"])
    def move_initial():
        speed_config = request_speed_config()
        with robot_lock:
            if robot_state["busy"]:
                return jsonify({"ok": False, "error": "robot command already running"}), 409
            clear_emergency_for_new_motion()
            robot_state["busy"] = True
            robot_state["message"] = f"moving to initial pose, speed={speed_config['initial_movej_speed']}"

        def worker():
            mover = Rm65SafeIkMover(
                robot_ip=args.robot_ip,
                robot_port=args.robot_port,
                speed=speed_config["initial_movej_speed"],
                max_joint_step_deg=120,
                max_j6_step_deg=120,
            )
            try:
                mover.connect()
                ret = mover.movej(INITIAL_JOINT_DEG)
                msg = f"move_initial ret={ret}, speed={speed_config['initial_movej_speed']}"
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
        return jsonify({"ok": True, "message": f"moving to initial pose, speed={speed_config['initial_movej_speed']}"})

    @app.route("/robot/stop", methods=["POST"])
    def robot_stop():
        def worker():
            results = []

            def run_stop(label, fn):
                try:
                    value = fn()
                    if isinstance(value, list):
                        results.extend([f"{label}: {x}" for x in value])
                    else:
                        results.append(f"{label}: {value}")
                except Exception as exc:
                    results.append(f"{label}: error {exc}")

            threads = [
                threading.Thread(target=run_stop, args=("arm", send_arm_stop_now), daemon=True),
                threading.Thread(target=run_stop, args=("mobile", send_mobile_stop_now), daemon=True),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=max(0.2, float(args.mobile_stop_timeout_s) + 0.5))
            msg = "EMERGENCY STOP sent: " + " | ".join(results)
            with robot_lock:
                robot_state["busy"] = False
                robot_state["message"] = msg

        emergency_event.set()
        with robot_lock:
            robot_state["busy"] = False
            robot_state["message"] = "EMERGENCY STOP requested: stopping arm/mobile"
        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "message": "EMERGENCY STOP command sent"})

    def start_gripper_action(action_name, position):
        with robot_lock:
            if robot_state["busy"]:
                return jsonify({"ok": False, "error": "robot command already running"}), 409
            clear_emergency_for_new_motion()
            robot_state["busy"] = True
            robot_state["message"] = f"gripper {action_name} running"

        def worker():
            try:
                set_right_gripper_modbus(
                    robot_ip=args.robot_ip,
                    robot_port=args.robot_port,
                    position=int(position),
                    force=int(args.gripper_force),
                    speed=int(args.gripper_speed),
                    device_id=int(args.gripper_device_id),
                    timeout_s=int(args.gripper_timeout_s),
                )
                msg = f"gripper {action_name} ok, position={int(position)}"
            except Exception as exc:
                msg = f"gripper {action_name} error: {exc}"
            finally:
                with robot_lock:
                    robot_state["busy"] = False
                    robot_state["message"] = msg

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "message": f"gripper {action_name} started"})

    @app.route("/gripper/open", methods=["POST"])
    def gripper_open():
        return start_gripper_action("open", args.gripper_open_position)

    @app.route("/gripper/close", methods=["POST"])
    def gripper_close():
        return start_gripper_action("close", args.gripper_close_position)

    def mobile_post(endpoint, payload, action_name):
        url = args.mobile_base_url.rstrip("/") + endpoint
        with robot_lock:
            if robot_state["busy"]:
                return jsonify({"ok": False, "error": "robot command already running"}), 409
            clear_emergency_for_new_motion()
            robot_state["busy"] = True
            robot_state["message"] = f"{action_name} running: {payload}"

        def worker():
            try:
                result = post_external_json(url, payload, timeout_s=args.mobile_timeout_s)
                msg = f"{action_name} ok: {result}"
            except Exception as exc:
                msg = f"{action_name} error: {exc}"
            finally:
                with robot_lock:
                    robot_state["busy"] = False
                    robot_state["message"] = msg

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "message": f"{action_name} started: {payload}"})

    @app.route("/mobile/lift", methods=["POST"])
    def mobile_lift():
        payload = request.get_json(silent=True) or {}
        clean = {
            "execMode": int(payload.get("execMode", 2)),
            "speed": float(payload.get("speed", 0.03)),
            "height": float(payload.get("height", 0.05)),
        }
        return mobile_post("/lift_control3", clean, "lift")

    @app.route("/mobile/step_forward", methods=["POST"])
    def mobile_step_forward():
        payload = request.get_json(silent=True) or {}
        clean = {
            "distance": float(payload.get("distance", -2.5)),
            "speed": float(payload.get("speed", 0.1)),
        }
        return mobile_post("/step_forward", clean, "step_forward")

    @app.route("/mobile/step_rotate", methods=["POST"])
    def mobile_step_rotate():
        payload = request.get_json(silent=True) or {}
        clean = {
            "angle": float(payload.get("angle", -1.57)),
            "speed": float(payload.get("speed", 0.2)),
        }
        return mobile_post("/step_rotate", clean, "step_rotate")

    def load_post_insert_place_poses():
        path = Path(args.post_insert_poses).expanduser().resolve()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pre = data.get("poses", {}).get("pre_place")
        if not pre or "joint_deg" not in pre or "pose" not in pre:
            raise RuntimeError(f"{path} missing poses.pre_place.pose/joint_deg")
        vertical = data.get("poses", {}).get("vertical_pre_release")
        if vertical and ("joint_deg" not in vertical or "pose" not in vertical):
            raise RuntimeError(f"{path} has invalid poses.vertical_pre_release; expected pose/joint_deg")
        release = data.get("poses", {}).get("place")
        if not release or "joint_deg" not in release or "pose" not in release:
            raise RuntimeError(f"{path} missing poses.place.pose/joint_deg")
        return path, vertical, pre, release

    def mobile_post_sync(endpoint, payload, label):
        assert_not_emergency(label)
        url = args.mobile_base_url.rstrip("/") + endpoint
        print(f"[POST-INSERT] {label}: POST {url} {payload}")
        result = post_external_json(url, payload, timeout_s=args.mobile_timeout_s)
        print(f"[POST-INSERT] {label}: {result}")
        return result

    def post_insert_lift(height, speed, label):
        return mobile_post_sync(
            "/lift_control3",
            {"execMode": 2, "speed": float(speed), "height": float(height)},
            label,
        )

    def post_insert_lift_command(exec_mode, height, speed, label):
        return mobile_post_sync(
            "/lift_control3",
            {"execMode": int(exec_mode), "speed": float(speed), "height": float(height)},
            label,
        )

    def post_insert_lift_query(label):
        return mobile_post_sync("/lift_control3", {"execMode": 0}, label)

    def post_insert_step_forward(distance, speed, label, avoid=None):
        payload = {"distance": float(distance), "speed": float(speed)}
        if avoid is not None:
            payload["avoid"] = int(avoid)
        return mobile_post_sync(
            "/step_forward",
            payload,
            label,
        )

    def post_insert_step_rotate(angle_rad, speed, label):
        return mobile_post_sync(
            "/step_rotate",
            {"angle": float(angle_rad), "speed": float(speed)},
            label,
        )

    @app.route("/robot/post_insert_place_sequence", methods=["POST"])
    def post_insert_place_sequence():
        speed_config = request_speed_config()
        with robot_lock:
            if robot_state["busy"]:
                return jsonify({"ok": False, "error": "robot command already running"}), 409
            clear_emergency_for_new_motion()
            robot_state["busy"] = True
            robot_state["message"] = "post-insert place sequence starting"

        def worker():
            mover = None
            try:
                runtime_config = load_config()
                seq = cfg_get(runtime_config, "post_insert_sequence", default={}) or {}
                first_lift = float(seq.get("first_lift_height_m", 0.01))
                backward = float(seq.get("backward_distance_m", -0.20))
                rotate_deg = float(seq.get("rotate_angle_deg", -90.0))
                forward = float(seq.get("forward_distance_m", 1.00))
                lower = float(seq.get("place_lift_lower_height_m", -0.10))
                raise_h = float(seq.get("place_lift_raise_height_m", 0.10))
                lift_speed = float(seq.get("lift_speed", 0.03))
                base_speed = float(seq.get("base_speed", 0.10))
                rotate_speed = float(seq.get("rotate_speed", 0.20))
                settle_s = float(seq.get("settle_s", 1.0))
                movej_speed = int(seq.get("movej_speed", speed_config["movej_speed"]))
                max_current_to_pre = float(seq.get("max_current_to_pre_m", 1.20))

                poses_path, vertical, pre, _ = load_post_insert_place_poses()
                print(f"[POST-INSERT] loaded place poses: {poses_path}")
                steps = [
                    "1 lift +0.01",
                    "2 base backward",
                    "3 base rotate",
                    "4 arm to vertical_pre_release",
                    "5 arm to pre_place",
                    "6 base forward",
                    "7 lift lower",
                    "8 close gripper",
                    "9 lift raise",
                ]
                if vertical is None:
                    steps[3] = "4 vertical_pre_release missing, skip"
                print(f"[POST-INSERT] steps: {steps}")

                assert_not_emergency("post-insert lift up")
                with robot_lock:
                    robot_state["message"] = "post-insert 1/9 lift up"
                post_insert_lift(first_lift, lift_speed, "1/9 lift_up")

                assert_not_emergency("post-insert base backward")
                with robot_lock:
                    robot_state["message"] = "post-insert 2/9 base backward"
                post_insert_step_forward(backward, base_speed, "2/9 base_backward")

                assert_not_emergency("post-insert base rotate")
                with robot_lock:
                    robot_state["message"] = "post-insert 3/9 base rotate"
                post_insert_step_rotate(np.deg2rad(rotate_deg), rotate_speed, "3/9 base_rotate")

                current_pose = None
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((args.robot_ip, int(args.robot_port)))
                    state = get_current_state(sock)
                    if state:
                        current_pose, _ = state
                finally:
                    try:
                        sock.close()
                    except Exception:
                        pass
                if current_pose is not None:
                    safety_target = vertical if vertical is not None else pre
                    dist = float(np.linalg.norm(np.asarray(current_pose[:3]) - np.asarray(safety_target["pose"][:3])))
                    if dist > max_current_to_pre:
                        raise RuntimeError(
                            f"current->{safety_target['name']} {dist * 1000.0:.1f} mm exceeds "
                            f"{max_current_to_pre * 1000.0:.1f} mm"
                        )

                assert_not_emergency("post-insert vertical_pre_release")
                with robot_lock:
                    robot_state["message"] = "post-insert 4/9 arm to vertical_pre_release"
                mover = Rm65SafeIkMover(
                    robot_ip=args.robot_ip,
                    robot_port=args.robot_port,
                    speed=movej_speed,
                    max_joint_step_deg=120,
                    max_j6_step_deg=120,
                )
                mover.connect()
                if vertical is not None:
                    ret = mover.movej([float(x) for x in vertical["joint_deg"][:6]])
                    print(f"[POST-INSERT] 4/9 movej vertical_pre_release ret={ret}")
                    if ret != 0:
                        raise RuntimeError(f"movej vertical_pre_release failed, ret={ret}")
                else:
                    print("[POST-INSERT] 4/9 vertical_pre_release missing, skipped")

                assert_not_emergency("post-insert pre_place")
                with robot_lock:
                    robot_state["message"] = "post-insert 5/9 arm to pre_place"
                ret = mover.movej([float(x) for x in pre["joint_deg"][:6]])
                print(f"[POST-INSERT] 5/9 movej pre_place ret={ret}")
                if ret != 0:
                    raise RuntimeError(f"movej pre_place failed, ret={ret}")

                assert_not_emergency("post-insert base forward")
                with robot_lock:
                    robot_state["message"] = "post-insert 6/9 base forward"
                post_insert_step_forward(forward, base_speed, "6/9 base_forward")

                assert_not_emergency("post-insert lift query")
                with robot_lock:
                    robot_state["message"] = "post-insert query lift pre-place"
                post_insert_lift_query("pre_place_lift_state")

                assert_not_emergency("post-insert lift lower")
                with robot_lock:
                    robot_state["message"] = "post-insert 7/9 lift lower"
                post_insert_lift(lower, lift_speed, "7/9 lift_lower")

                assert_not_emergency("post-insert close gripper")
                with robot_lock:
                    robot_state["message"] = "post-insert 8/9 close gripper"
                set_right_gripper_modbus(
                    robot_ip=args.robot_ip,
                    robot_port=args.robot_port,
                    position=int(args.gripper_close_position),
                    force=int(args.gripper_force),
                    speed=int(args.gripper_speed),
                    device_id=int(args.gripper_device_id),
                    timeout_s=int(args.gripper_timeout_s),
                )
                time.sleep(max(0.0, settle_s))

                assert_not_emergency("post-insert lift raise")
                with robot_lock:
                    robot_state["message"] = "post-insert 9/9 lift raise"
                post_insert_lift(raise_h, lift_speed, "9/9 lift_raise")
                msg = "post-insert place sequence complete"
            except Exception as exc:
                msg = f"post-insert place sequence error: {exc}"
            finally:
                if mover is not None:
                    try:
                        mover.close()
                    except Exception:
                        pass
                with robot_lock:
                    robot_state["busy"] = False
                    robot_state["message"] = msg

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "message": "post-insert place sequence started"})

    @app.route("/robot/post_insert_return_sequence", methods=["POST"])
    def post_insert_return_sequence():
        with robot_lock:
            if robot_state["busy"]:
                return jsonify({"ok": False, "error": "robot command already running"}), 409
            clear_emergency_for_new_motion()
            robot_state["busy"] = True
            robot_state["message"] = "post-insert return sequence starting"

        def worker():
            try:
                runtime_config = load_config()
                seq = cfg_get(runtime_config, "post_insert_return_sequence", default={}) or {}
                lift_exec_mode = int(seq.get("lift_exec_mode", 1))
                lift_h = float(seq.get("lift_height_m", 0.70))
                backward = float(seq.get("backward_distance_m", -0.30))
                rotate_deg = float(seq.get("rotate_angle_deg", -90.0))
                forward = float(seq.get("forward_distance_m", 0.30))
                lift_speed = float(seq.get("lift_speed", 0.03))
                base_speed = float(seq.get("base_speed", 0.10))
                rotate_speed = float(seq.get("rotate_speed", 0.20))
                initial_movej_speed = int(seq.get("initial_movej_speed", args.initial_movej_speed))

                steps = [
                    "1 lift to return height",
                    "2 base backward",
                    "3 arm to initial",
                    "4 base rotate",
                    "5 base forward",
                ]
                print(f"[POST-RETURN] steps: {steps}")

                assert_not_emergency("post-return lift")
                with robot_lock:
                    robot_state["message"] = f"post-return 1/5 lift to {lift_h:.3f}"
                post_insert_lift_command(lift_exec_mode, lift_h, lift_speed, "return 1/5 lift_to_height")

                assert_not_emergency("post-return base backward")
                with robot_lock:
                    robot_state["message"] = "post-return 2/5 base backward"
                post_insert_step_forward(backward, base_speed, "return 2/5 base_backward")

                assert_not_emergency("post-return arm initial")
                with robot_lock:
                    robot_state["message"] = f"post-return 3/5 arm to initial, speed={initial_movej_speed}"
                if not INITIAL_JOINT_DEG:
                    raise RuntimeError("robot.initial_joint_deg is empty in config.yaml")
                mover = Rm65SafeIkMover(
                    robot_ip=args.robot_ip,
                    robot_port=args.robot_port,
                    speed=initial_movej_speed,
                    max_joint_step_deg=180,
                    max_j6_step_deg=180,
                )
                try:
                    mover.connect()
                    ret = mover.movej(INITIAL_JOINT_DEG)
                    print(f"[POST-RETURN] 3/5 movej initial ret={ret}")
                    if ret != 0:
                        raise RuntimeError(f"movej initial failed, ret={ret}")
                finally:
                    try:
                        mover.close()
                    except Exception:
                        pass

                assert_not_emergency("post-return base rotate")
                with robot_lock:
                    robot_state["message"] = f"post-return 4/5 base rotate {rotate_deg:.1f}deg"
                post_insert_step_rotate(np.deg2rad(rotate_deg), rotate_speed, "return 4/5 base_rotate")

                assert_not_emergency("post-return base forward")
                with robot_lock:
                    robot_state["message"] = "post-return 5/5 base forward"
                post_insert_step_forward(forward, base_speed, "return 5/5 base_forward")

                msg = "post-insert return sequence complete"
            except Exception as exc:
                msg = f"post-insert return sequence error: {exc}"
            finally:
                with robot_lock:
                    robot_state["busy"] = False
                    robot_state["message"] = msg

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "message": "post-insert return sequence started"})

    def start_live_preinsert(mode="normal"):
        speed_config = request_speed_config()
        with robot_lock:
            if robot_state["busy"]:
                return jsonify({"ok": False, "error": "robot command already running"}), 409
            clear_emergency_for_new_motion()
            robot_state["busy"] = True
            robot_state["message"] = f"{mode} preinsert running, movej_speed={speed_config['movej_speed']}"
            robot_state["last_plan"] = ""

        def worker():
            ts = time.strftime("%Y%m%d_%H%M%S")
            suffix = mode
            run_root = Path(args.frontend_run_dir).expanduser().resolve() / f"frontend_{ts}_{suffix}"
            capture_dir = run_root / "capture"
            result_dir = run_root / "result"
            result_dir.mkdir(parents=True, exist_ok=True)
            try:
                if mode == "normal_align":
                    with robot_lock:
                        robot_state["message"] = "normal_align: closing gripper before alignment"
                    set_right_gripper_modbus(
                        robot_ip=args.robot_ip,
                        robot_port=args.robot_port,
                        position=int(args.gripper_close_position),
                        force=int(args.gripper_force),
                        speed=int(args.gripper_speed),
                        device_id=int(args.gripper_device_id),
                        timeout_s=int(args.gripper_timeout_s),
                    )
                    time.sleep(0.2)

                runtime_config = load_config()
                right_offset_m = float(cfg_get(runtime_config, "insertion", "observed_right_offset_m", default=args.right_offset_m))
                up_offset_m = float(cfg_get(runtime_config, "insertion", "observed_up_offset_m", default=args.up_offset_m))
                offset_frame = str(cfg_get(runtime_config, "insertion", "offset_frame", default="camera"))
                tcp_to_tip_m = float(cfg_get(runtime_config, "tool", "tcp_to_tip_m", default=args.tcp_to_tip_m))
                tip_tcp_m = cfg_get(runtime_config, "tool", "tip_tcp_m", default=args.tip_tcp_m)
                preinsert_distance_m = float(cfg_get(runtime_config, "insertion", "preinsert_distance_m", default=args.preinsert_distance_m))
                insert_depth_m = float(cfg_get(runtime_config, "insertion", "insert_depth_m", default=args.insert_depth_m))
                max_preinsert_move_m = float(cfg_get(runtime_config, "insertion", "max_preinsert_move_m", default=args.max_preinsert_move_m))

                saved_capture = streamer.save_current_capture(capture_dir)
                if saved_capture is None:
                    raise RuntimeError("no RGBD frame ready")
                stem = f"frontend_yolo_{ts}"
                detection_json = result_dir / f"{stem}_detection.json"
                plan_json = result_dir / f"{stem}_plan.json"
                detect_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "detect_center_hole_yolo_rgbd.py"),
                    str(saved_capture),
                    "--out-dir", str(result_dir),
                    "--out-stem", stem,
                    "--conf", str(args.yolo_conf),
                    "--min-confidence", "0.35",
                ]
                move_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "move_to_center_hole.py"),
                    "--detection", str(detection_json),
                    "--observed-right-offset-m", str(right_offset_m),
                    "--observed-up-offset-m", str(up_offset_m),
                    "--offset-frame", offset_frame,
                    "--tcp-to-tip-m", str(tcp_to_tip_m),
                    "--preinsert-distance-m", str(preinsert_distance_m),
                    "--insert-depth-m", str(insert_depth_m),
                    "--max-preinsert-move-m", str(max_preinsert_move_m),
                    "--movej-speed", str(speed_config["movej_speed"]),
                    "--max-j6-step-deg", str(args.max_j6_step_deg),
                    "--plan-out", str(plan_json),
                    "--require-quality-ok",
                    "--move-preinsert",
                ]
                tip_tcp_arg = format_vec3_arg(tip_tcp_m)
                if tip_tcp_arg:
                    move_cmd.append(f"--tip-tcp-m={tip_tcp_arg}")
                if mode == "roll_search":
                    move_cmd.extend([
                        "--keep-current-orientation",
                        "--roll-search-deg", str(args.roll_search_deg),
                        "--controller-pose-fallback",
                    ])
                elif mode == "normal_align":
                    move_cmd.extend([
                        "--align-orientation-only",
                        "--controller-pose-fallback",
                    ])
                print(
                    f"[FRONTEND PLAN] offsets right/up={right_offset_m:.4f}/{up_offset_m:.4f} m, "
                    f"offset_frame={offset_frame}, "
                    f"tcp_to_tip={tcp_to_tip_m:.4f} m, tip_tcp={tip_tcp_arg}"
                )
                subprocess.run(detect_cmd, check=True)
                try:
                    with open(detection_json, "r", encoding="utf-8") as f:
                        detection_payload = json.load(f)
                    if detection_payload.get("quality") != "ok":
                        reasons = ", ".join(detection_payload.get("quality_reasons", [])) or "unknown"
                        raise RuntimeError(
                            f"detection quality={detection_payload.get('quality')}: {reasons}"
                        )
                except RuntimeError:
                    raise
                except Exception as exc:
                    raise RuntimeError(f"failed to read detection quality: {exc}") from exc
                print("[FRONTEND MOVE CMD]", " ".join(str(x) for x in move_cmd))
                subprocess.run(move_cmd, check=True)
                msg = f"{suffix} ok: {run_root}"
                last_plan = str(plan_json)
            except Exception as exc:
                msg = f"{suffix} error: {exc}"
                last_plan = ""
            finally:
                with robot_lock:
                    robot_state["busy"] = False
                    robot_state["message"] = msg
                    robot_state["last_plan"] = last_plan

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "message": f"{mode} preinsert started, movej_speed={speed_config['movej_speed']}"})

    @app.route("/robot/live_preinsert", methods=["POST"])
    def live_preinsert():
        return start_live_preinsert(mode="preinsert")

    @app.route("/robot/live_preinsert_roll_search", methods=["POST"])
    def live_preinsert_roll_search():
        return start_live_preinsert(mode="roll_search")

    @app.route("/robot/live_preinsert_normal_align", methods=["POST"])
    def live_preinsert_normal_align():
        return start_live_preinsert(mode="normal_align")

    @app.route("/robot/move_last_plan_preinsert", methods=["POST"])
    def move_last_plan_preinsert():
        speed_config = request_speed_config()
        with robot_lock:
            if robot_state["busy"]:
                return jsonify({"ok": False, "error": "robot command already running"}), 409
            plan_path = robot_state.get("last_plan") or ""
            if not plan_path:
                return jsonify({"ok": False, "error": "no current valid plan; run YOLO plan first"}), 404
            if not Path(plan_path).exists():
                robot_state["last_plan"] = ""
                return jsonify({"ok": False, "error": f"current plan missing: {plan_path}"}), 404
            clear_emergency_for_new_motion()
            robot_state["busy"] = True
            robot_state["message"] = f"last-plan preinsert running, movej_speed={speed_config['movej_speed']}: {plan_path}"

        def worker():
            try:
                preinsert_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "move_to_center_hole.py"),
                    "--plan-in", str(plan_path),
                    "--max-preinsert-move-m", str(args.max_preinsert_move_m),
                    "--movej-speed", str(speed_config["movej_speed"]),
                    "--max-j6-step-deg", str(args.max_j6_step_deg),
                    "--require-quality-ok",
                    "--move-preinsert",
                    "--controller-pose-fallback",
                ]
                subprocess.run(preinsert_cmd, check=True)
                msg = f"last-plan preinsert ok: {plan_path}"
            except Exception as exc:
                msg = f"last-plan preinsert error: {exc}"
            finally:
                with robot_lock:
                    robot_state["busy"] = False
                    robot_state["message"] = msg

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "message": f"last-plan preinsert started, movej_speed={speed_config['movej_speed']}: {plan_path}"})

    @app.route("/robot/insert_last_plan", methods=["POST"])
    def insert_last_plan():
        speed_config = request_speed_config()
        with robot_lock:
            if robot_state["busy"]:
                return jsonify({"ok": False, "error": "robot command already running"}), 409
            plan_path = robot_state.get("last_plan") or ""
            if not plan_path:
                return jsonify({"ok": False, "error": "no current valid plan; run YOLO plan first"}), 404
            if not Path(plan_path).exists():
                robot_state["last_plan"] = ""
                return jsonify({"ok": False, "error": f"current plan missing: {plan_path}"}), 404
            clear_emergency_for_new_motion()
            robot_state["busy"] = True
            robot_state["message"] = f"insert running, speed={speed_config['frontend_insert_speed']}: {plan_path}"

        def worker():
            try:
                insert_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "continue_insert_along_axis.py"),
                    "--plan", str(plan_path),
                    "--distance-m", str(args.frontend_insert_distance_m),
                    "--max-distance-m", str(args.frontend_max_insert_distance_m),
                    "--axis-source", "current-tool",
                    "--tool-axis", "+z",
                    "--speed", str(speed_config["frontend_insert_speed"]),
                    "--robot-ip", str(args.robot_ip),
                    "--robot-port", str(args.robot_port),
                ]
                subprocess.run(insert_cmd, check=True)
                msg = f"insert ok: {plan_path}"
            except Exception as exc:
                msg = f"insert error: {exc}"
            finally:
                with robot_lock:
                    robot_state["busy"] = False
                    robot_state["message"] = msg

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "message": f"insert started, speed={speed_config['frontend_insert_speed']}: {plan_path}"})

    @app.route("/robot/auto_align_insert_open", methods=["POST"])
    def auto_align_insert_open():
        speed_config = request_speed_config()
        with robot_lock:
            if robot_state["busy"]:
                return jsonify({"ok": False, "error": "robot command already running"}), 409
            clear_emergency_for_new_motion()
            robot_state["busy"] = True
            robot_state["message"] = "抓取方向盘上料 starting: close -> align -> preinsert -> insert -> open -> lift/back -> vertical -> lift/rotate -> release -> forward -> lift height"
            robot_state["last_plan"] = ""

        def worker():
            plan_json = None
            mover = None
            try:
                ts = time.strftime("%Y%m%d_%H%M%S")
                run_root = Path(args.frontend_run_dir).expanduser().resolve() / f"frontend_{ts}_auto_align_insert_open"
                capture_dir = run_root / "capture"
                result_dir = run_root / "result"
                result_dir.mkdir(parents=True, exist_ok=True)

                runtime_config = load_config()
                right_offset_m = float(cfg_get(runtime_config, "insertion", "observed_right_offset_m", default=args.right_offset_m))
                up_offset_m = float(cfg_get(runtime_config, "insertion", "observed_up_offset_m", default=args.up_offset_m))
                offset_frame = str(cfg_get(runtime_config, "insertion", "offset_frame", default="camera"))
                tcp_to_tip_m = float(cfg_get(runtime_config, "tool", "tcp_to_tip_m", default=args.tcp_to_tip_m))
                tip_tcp_m = cfg_get(runtime_config, "tool", "tip_tcp_m", default=args.tip_tcp_m)
                preinsert_distance_m = float(cfg_get(runtime_config, "insertion", "preinsert_distance_m", default=args.preinsert_distance_m))
                insert_depth_m = float(cfg_get(runtime_config, "insertion", "insert_depth_m", default=args.insert_depth_m))
                max_preinsert_move_m = float(cfg_get(runtime_config, "insertion", "max_preinsert_move_m", default=args.max_preinsert_move_m))
                post_seq = cfg_get(runtime_config, "post_insert_sequence", default={}) or {}
                lift_speed = float(post_seq.get("lift_speed", 0.03))
                base_speed = float(post_seq.get("base_speed", 0.10))
                rotate_speed = float(post_seq.get("rotate_speed", 0.20))
                movej_speed = int(post_seq.get("movej_speed", speed_config["movej_speed"]))
                max_current_to_vertical = float(post_seq.get("max_current_to_vertical_m", post_seq.get("max_current_to_pre_m", 1.20)))
                max_current_to_release = float(post_seq.get("max_current_to_release_m", post_seq.get("max_current_to_pre_m", 1.20)))
                onload_lift_after_vertical_exec_mode = int(post_seq.get("onload_lift_after_vertical_exec_mode", 1))
                onload_lift_after_vertical_height = float(post_seq.get("onload_lift_after_vertical_height_m", 0.80))
                onload_rotate_after_vertical_deg = float(post_seq.get("onload_rotate_after_vertical_deg", 90.0))
                onload_forward_before_release_height = float(post_seq.get("onload_forward_before_release_height_m", 0.30))
                onload_forward_avoid = int(post_seq.get("onload_forward_avoid", 1))
                onload_release_lift_exec_mode = int(post_seq.get("onload_release_lift_exec_mode", 1))
                onload_release_lift_height = float(post_seq.get("onload_release_lift_height_m", 0.66))
                tip_tcp_arg = format_vec3_arg(tip_tcp_m)

                assert_not_emergency("auto close gripper")
                with robot_lock:
                    robot_state["message"] = "抓取方向盘上料 1/17 close gripper"
                set_right_gripper_modbus(
                    robot_ip=args.robot_ip,
                    robot_port=args.robot_port,
                    position=int(args.gripper_close_position),
                    force=int(args.gripper_force),
                    speed=int(args.gripper_speed),
                    device_id=int(args.gripper_device_id),
                    timeout_s=int(args.gripper_timeout_s),
                )

                assert_not_emergency("auto capture")
                with robot_lock:
                    robot_state["message"] = "抓取方向盘上料 2/17 capture RGBD and detect center hole"
                saved_capture = streamer.save_current_capture(capture_dir)
                if saved_capture is None:
                    raise RuntimeError("no RGBD frame ready")

                stem = f"frontend_yolo_{ts}"
                detection_json = result_dir / f"{stem}_detection.json"
                plan_json = result_dir / f"{stem}_plan.json"
                detect_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "detect_center_hole_yolo_rgbd.py"),
                    str(saved_capture),
                    "--out-dir", str(result_dir),
                    "--out-stem", stem,
                    "--conf", str(args.yolo_conf),
                    "--min-confidence", "0.35",
                ]
                subprocess.run(detect_cmd, check=True)

                assert_not_emergency("auto detection quality check")
                with open(detection_json, "r", encoding="utf-8") as f:
                    detection_payload = json.load(f)
                if detection_payload.get("quality") != "ok":
                    reasons = ", ".join(detection_payload.get("quality_reasons", [])) or "unknown"
                    raise RuntimeError(f"detection quality={detection_payload.get('quality')}: {reasons}")

                assert_not_emergency("auto align orientation")
                with robot_lock:
                    robot_state["message"] = "抓取方向盘上料 3/17 align orientation to normal"
                align_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "move_to_center_hole.py"),
                    "--detection", str(detection_json),
                    "--observed-right-offset-m", str(right_offset_m),
                    "--observed-up-offset-m", str(up_offset_m),
                    "--offset-frame", offset_frame,
                    "--tcp-to-tip-m", str(tcp_to_tip_m),
                    "--preinsert-distance-m", str(preinsert_distance_m),
                    "--insert-depth-m", str(insert_depth_m),
                    "--max-preinsert-move-m", str(max_preinsert_move_m),
                    "--movej-speed", str(speed_config["movej_speed"]),
                    "--max-j6-step-deg", str(args.max_j6_step_deg),
                    "--plan-out", str(plan_json),
                    "--require-quality-ok",
                    "--move-preinsert",
                    "--align-orientation-only",
                    "--controller-pose-fallback",
                ]
                if tip_tcp_arg:
                    align_cmd.append(f"--tip-tcp-m={tip_tcp_arg}")
                print("[AUTO ALIGN CMD]", " ".join(str(x) for x in align_cmd))
                subprocess.run(align_cmd, check=True)

                assert_not_emergency("auto planned preinsert")
                with robot_lock:
                    robot_state["last_plan"] = str(plan_json)
                    robot_state["message"] = "抓取方向盘上料 4/17 move to planned preinsert"
                preinsert_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "move_to_center_hole.py"),
                    "--plan-in", str(plan_json),
                    "--max-preinsert-move-m", str(args.max_preinsert_move_m),
                    "--movej-speed", str(speed_config["movej_speed"]),
                    "--max-j6-step-deg", str(args.max_j6_step_deg),
                    "--require-quality-ok",
                    "--move-preinsert",
                    "--controller-pose-fallback",
                ]
                subprocess.run(preinsert_cmd, check=True)

                for idx in range(4):
                    assert_not_emergency(f"auto insert step {idx + 1}")
                    with robot_lock:
                        robot_state["message"] = f"抓取方向盘上料 {idx + 5}/17 insert 10mm step {idx + 1}/4"
                    insert_cmd = [
                        sys.executable,
                        str(SCRIPT_DIR / "continue_insert_along_axis.py"),
                        "--plan", str(plan_json),
                        "--distance-m", str(args.frontend_insert_distance_m),
                        "--max-distance-m", str(args.frontend_max_insert_distance_m),
                        "--axis-source", "current-tool",
                        "--tool-axis", "+z",
                        "--speed", str(speed_config["frontend_insert_speed"]),
                        "--robot-ip", str(args.robot_ip),
                        "--robot-port", str(args.robot_port),
                    ]
                    subprocess.run(insert_cmd, check=True)
                    if idx < 3:
                        time.sleep(1.0)

                assert_not_emergency("auto open gripper")
                with robot_lock:
                    robot_state["message"] = "抓取方向盘上料 9/17 open gripper"
                set_right_gripper_modbus(
                    robot_ip=args.robot_ip,
                    robot_port=args.robot_port,
                    position=int(args.gripper_open_position),
                    force=int(args.gripper_force),
                    speed=int(args.gripper_speed),
                    device_id=int(args.gripper_device_id),
                    timeout_s=int(args.gripper_timeout_s),
                )

                assert_not_emergency("auto lift up")
                with robot_lock:
                    robot_state["message"] = "抓取方向盘上料 10/17 lift up 0.02m"
                post_insert_lift(0.02, lift_speed, "抓取方向盘上料 10/17 lift_up")

                assert_not_emergency("auto base backward")
                with robot_lock:
                    robot_state["message"] = "抓取方向盘上料 11/17 base backward -0.3m"
                post_insert_step_forward(-0.3, base_speed, "抓取方向盘上料 11/17 base_backward")

                assert_not_emergency("auto load vertical_pre_release")
                poses_path, vertical, _, release = load_post_insert_place_poses()
                if vertical is None:
                    raise RuntimeError(f"{poses_path} missing poses.vertical_pre_release")
                current_pose = None
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((args.robot_ip, int(args.robot_port)))
                    state = get_current_state(sock)
                    if state:
                        current_pose, _ = state
                finally:
                    try:
                        sock.close()
                    except Exception:
                        pass
                if current_pose is not None:
                    dist = float(np.linalg.norm(np.asarray(current_pose[:3]) - np.asarray(vertical["pose"][:3])))
                    if dist > max_current_to_vertical:
                        raise RuntimeError(
                            f"current->vertical_pre_release {dist * 1000.0:.1f} mm exceeds "
                            f"{max_current_to_vertical * 1000.0:.1f} mm"
                        )

                assert_not_emergency("auto vertical_pre_release")
                with robot_lock:
                    robot_state["message"] = "抓取方向盘上料 12/17 arm to vertical_pre_release"
                mover = Rm65SafeIkMover(
                    robot_ip=args.robot_ip,
                    robot_port=args.robot_port,
                    speed=movej_speed,
                    max_joint_step_deg=120,
                    max_j6_step_deg=120,
                )
                mover.connect()
                ret = mover.movej([float(x) for x in vertical["joint_deg"][:6]])
                print(f"[AUTO] 12/17 movej vertical_pre_release ret={ret}, poses={poses_path}")
                if ret != 0:
                    raise RuntimeError(f"movej vertical_pre_release failed, ret={ret}")

                assert_not_emergency("auto lift after vertical_pre_release")
                with robot_lock:
                    robot_state["message"] = f"抓取方向盘上料 13/17 lift height to {onload_lift_after_vertical_height:.3f}"
                post_insert_lift_command(
                    onload_lift_after_vertical_exec_mode,
                    onload_lift_after_vertical_height,
                    lift_speed,
                    "抓取方向盘上料 13/17 lift_after_vertical_to_height",
                )

                assert_not_emergency("auto base rotate after vertical_pre_release")
                with robot_lock:
                    robot_state["message"] = f"抓取方向盘上料 14/17 base rotate {onload_rotate_after_vertical_deg:.1f}deg"
                post_insert_step_rotate(
                    np.deg2rad(onload_rotate_after_vertical_deg),
                    rotate_speed,
                    "抓取方向盘上料 14/17 base_rotate_after_vertical",
                )

                assert_not_emergency("auto release pose")
                current_pose = None
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((args.robot_ip, int(args.robot_port)))
                    state = get_current_state(sock)
                    if state:
                        current_pose, _ = state
                finally:
                    try:
                        sock.close()
                    except Exception:
                        pass
                if current_pose is not None:
                    dist = float(np.linalg.norm(np.asarray(current_pose[:3]) - np.asarray(release["pose"][:3])))
                    if dist > max_current_to_release:
                        raise RuntimeError(
                            f"current->release {dist * 1000.0:.1f} mm exceeds "
                            f"{max_current_to_release * 1000.0:.1f} mm"
                        )
                with robot_lock:
                    robot_state["message"] = "抓取方向盘上料 15/17 arm to release pose"
                ret = mover.movej([float(x) for x in release["joint_deg"][:6]])
                print(f"[AUTO] 15/17 movej release ret={ret}, poses={poses_path}")
                if ret != 0:
                    raise RuntimeError(f"movej release pose failed, ret={ret}")

                assert_not_emergency("auto base forward before release height")
                with robot_lock:
                    robot_state["message"] = f"抓取方向盘上料 16/17 base forward {onload_forward_before_release_height:.3f}m avoid={onload_forward_avoid}"
                post_insert_step_forward(
                    onload_forward_before_release_height,
                    base_speed,
                    "抓取方向盘上料 16/17 base_forward_before_release_height",
                    avoid=onload_forward_avoid,
                )

                assert_not_emergency("auto lift to release height")
                with robot_lock:
                    robot_state["message"] = f"抓取方向盘上料 17/17 lift height to {onload_release_lift_height:.3f}"
                post_insert_lift_command(
                    onload_release_lift_exec_mode,
                    onload_release_lift_height,
                    lift_speed,
                    "抓取方向盘上料 17/17 lift_to_release_height",
                )

                msg = f"抓取方向盘上料 complete: {run_root}"
            except Exception as exc:
                msg = f"抓取方向盘上料 error: {exc}"
            finally:
                if mover is not None:
                    try:
                        mover.close()
                    except Exception:
                        pass
                with robot_lock:
                    robot_state["busy"] = False
                    robot_state["message"] = msg
                    if plan_json is not None and Path(plan_json).exists():
                        robot_state["last_plan"] = str(plan_json)

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "message": "抓取方向盘上料 started"})

    @app.route("/robot/status")
    def robot_status():
        with robot_lock:
            return jsonify(dict(robot_state))

    @app.route("/yolo/toggle", methods=["POST"])
    def yolo_toggle():
        enabled = streamer.toggle_yolo()
        return jsonify({"ok": True, "enabled": enabled})

    @app.route("/yolo/on", methods=["POST"])
    def yolo_on():
        return jsonify({"ok": True, "enabled": streamer.set_yolo_enabled(True)})

    @app.route("/yolo/off", methods=["POST"])
    def yolo_off():
        return jsonify({"ok": True, "enabled": streamer.set_yolo_enabled(False)})

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
