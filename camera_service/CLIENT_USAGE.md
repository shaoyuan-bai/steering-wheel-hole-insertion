# Camera Service Client Usage

这个服务是全项目唯一允许直接打开 RealSense 的进程。其他项目不要再使用 `pyrealsense2.pipeline.start()`，统一通过 HTTP/WebSocket 调用。

## 服务地址

HTTP:

```text
http://127.0.0.1:8099
http://192.168.2.228:8099
```

WebSocket:

```text
ws://127.0.0.1:8100
ws://192.168.2.228:8100
```

## 相机角色

当前角色映射：

```text
right_arm -> 405622075930
left_arm  -> 244222070594
head      -> 250122076328
```

业务代码只使用角色名，不要写死序列号。

## 低延迟 WebSocket 取图

连接地址：

```text
ws://127.0.0.1:8100/camera/ws/right_arm?quality=85&fps=30
```

参数：

```text
quality: JPEG 质量，建议 70-90
fps: 服务端推送上限
```

消息格式：

```text
第 1 条消息: JSON 文本，包含 role/serial 等元信息
后续消息: JPEG 二进制帧
```

Python 示例：

```python
import cv2
import numpy as np
import websocket

url = "ws://127.0.0.1:8100/camera/ws/right_arm?quality=85&fps=30"
ws = websocket.create_connection(
    url,
    timeout=3,
    http_proxy_host=None,
    http_proxy_port=None,
    http_no_proxy=["127.0.0.1", "localhost", "*"],
)

try:
    info = ws.recv()  # JSON text
    print(info)

    while True:
        msg = ws.recv()
        if isinstance(msg, str):
            print("control/error:", msg)
            continue
        frame = cv2.imdecode(np.frombuffer(msg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        # use frame here
finally:
    ws.close()
```

## 保存一帧 RGBD

需要深度数据时走 HTTP 保存离线目录：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:8099/camera/capture \
  -H 'Content-Type: application/json' \
  -d '{"role":"right_arm","target_dir":"/tmp/right_arm_rgbd"}'
```

输出目录包含：

```text
color.png
depth_raw.npy
depth_mm.png
depth_colormap.png
depth_overlay.png
intrinsics.json
```

## 查询服务状态

```bash
curl --noproxy '*' http://127.0.0.1:8099/camera/status
```

## 前端项目

`wheel_hole_insertion/realsense_preview_server.py` 默认使用：

```text
camera_service.url:    http://127.0.0.1:8099
camera_service.ws_url: ws://127.0.0.1:8100
camera_service.default_role: right_arm
```

配置位置：

```text
wheel_hole_insertion/config.yaml
```

前端启动后 `/status` 中 `camera_transport` 应为：

```text
websocket
```

## 启停服务

```bash
systemctl --user status woosh-camera-service.service
systemctl --user restart woosh-camera-service.service
```

## 规则

- 其他项目只用角色名：`right_arm` / `left_arm` / `head`。
- 实时预览/算法输入优先用 WebSocket。
- 需要深度和内参落盘时用 HTTP `/camera/capture`。
- 不要在其他项目里直接打开 RealSense。
