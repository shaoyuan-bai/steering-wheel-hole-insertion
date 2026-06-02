# RealSense Camera Service

这个目录是独立摄像头服务。长期约定：只有 `camera_service.py serve` 直接打开 RealSense，其他项目通过 HTTP API 调用，不再自己 `rs.pipeline.start()` 抢设备。

## 1. 枚举摄像头

```bash
cd /home/wooshrobot/bai/hand_eye_calibration
conda activate cyy
python camera_service/camera_service.py list
```

## 2. 给所有摄像头拍样张

```bash
python camera_service/camera_service.py capture-all
```

输出在：

```text
camera_service/captures/
```

每个摄像头会保存：

- `color.png`
- `depth_raw.npy`
- `depth_mm.png`
- `depth_colormap.png`
- `depth_overlay.png`
- `intrinsics.json`

你看 `color.png` 后，把序列号对应到名称，例如左臂、右臂、头部。

## 3. 配置角色

编辑：

```text
camera_service/cameras.json
```

示例：

```json
{
  "roles": {
    "left_arm": {"serial": "123456789", "width": 1280, "height": 720, "fps": 30},
    "right_arm": {"serial": "234567890", "width": 1280, "height": 720, "fps": 30},
    "head": {"serial": "345678901", "width": 1280, "height": 720, "fps": 30}
  },
  "defaults": {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "warmup": 30
  }
}
```

也可以用 HTTP 写入：

```bash
curl -X POST http://127.0.0.1:8099/camera/config \
  -H 'Content-Type: application/json' \
  -d '{"role":"right_arm","serial":"234567890","width":1280,"height":720,"fps":30}'
```

## 4. 启动服务

```bash
python camera_service/camera_service.py serve --host 0.0.0.0 --port 8099 --open-on-start roles
```

`--open-on-start roles` 会按照 `cameras.json` 同时打开 `right_arm`、`head`、`left_arm` 三路 RGBD，并在内存中维护最新帧。

当前设备已验证三路 `1280x720@30` 彩色+深度可以同时打开。

查看状态：

```bash
curl --noproxy '*' http://127.0.0.1:8099/camera/status
```

查看某个角色的 MJPEG：

```text
http://<robot-ip>:8099/camera/mjpeg/right_arm
```

WebSocket 低延迟预览：

```text
ws://<robot-ip>:8100/camera/ws/right_arm?quality=85&fps=30
```

WebSocket 第一条消息是 JSON 文本元信息，后续消息是 JPEG 二进制帧。

## 5. 其他项目如何调用

列出设备和角色：

```bash
curl http://127.0.0.1:8099/camera/list
```

按角色保存一帧 RGBD：

```bash
curl -X POST http://127.0.0.1:8099/camera/capture \
  -H 'Content-Type: application/json' \
  -d '{"role":"right_arm"}'
```

返回：

```json
{
  "ok": true,
  "capture_dir": "...",
  "meta": {
    "role": "right_arm",
    "serial": "...",
    "color_intrinsics": {}
  }
}
```

项目拿到 `capture_dir` 后读取：

```text
color.png
depth_raw.npy
intrinsics.json
```

这和当前方向盘识别脚本使用的 RGBD 离线目录格式一致。

获取最新 JPEG 和内参：

```bash
curl --noproxy '*' http://127.0.0.1:8099/camera/latest/right_arm
```

直接拿单帧 JPEG：

```text
http://127.0.0.1:8099/camera/frame/right_arm.jpg
```

## 6. 开机自启

已提供用户级 systemd 服务：

```bash
cd /home/wooshrobot/bai/hand_eye_calibration
camera_service/install_user_service.sh
```

状态查看：

```bash
systemctl --user status woosh-camera-service.service
```

重启：

```bash
systemctl --user restart woosh-camera-service.service
```

如果要无人登录也在开机后启动，需要管理员执行一次：

```bash
sudo loginctl enable-linger wooshrobot
```

## 调用原则

- 其他项目不要直接打开 RealSense。
- 所有项目只请求角色名：`left_arm` / `right_arm` / `head`。
- 角色和序列号只在 `camera_service/cameras.json` 维护。
- 如果摄像头更换或 USB 顺序变化，只改 `cameras.json`，不要改业务代码。
