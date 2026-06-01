# 方向盘中心孔视觉插入项目

本项目用于 RM65 机械臂在眼在手上的 RGBD 相机引导下，将夹爪末端插入方向盘中心金属环孔。方向盘挂在架子上时会有不同倾斜角，因此流程不是使用固定世界姿态插入，而是通过视觉识别中心孔位置，并结合 RGBD 深度拟合中心金属环附近的局部法向，让插入杆沿孔轴方向接近和插入。

手眼标定只是本项目中的一个必要坐标转换环节。完整流程还包括中心孔实例分割、RGBD 法向估计、杆尖 TCP 标定、运动规划、夹爪控制、急停保护和 VLA 数据采集。

## 当前能力

- 使用 YOLO11n-seg 单类模型识别 `center_hole`。
- 使用 RGBD 点云在中心孔周围拟合局部平面，得到孔中心和法向。
- 根据手眼矩阵将相机坐标转换到机械臂 base 坐标。
- 根据杆尖 TCP 偏置生成预插入位姿和插入位姿。
- 提供内网 Web 前端，可查看相机画面、YOLO 叠加、执行运动、控制夹爪、保存训练数据。
- 提供命令行脚本用于采集、离线检测、稳定性验证、杆尖 pivot 标定。
- 支持将 RGB 视频和机械臂关节/位姿按时间对齐保存，便于后续 VLA 数据集整理。

## 目录结构

核心代码位于：

```text
wheel_hole_insertion/
```

主要文件：

```text
wheel_hole_insertion/config.yaml                    # 设备、标定、补偿、速度、安全参数
wheel_hole_insertion/realsense_preview_server.py    # 内网前端服务
wheel_hole_insertion/detect_center_hole_yolo_rgbd.py # YOLO + RGBD 中心孔识别
wheel_hole_insertion/move_to_center_hole.py         # 运动规划与预插入执行
wheel_hole_insertion/continue_insert_along_axis.py  # 沿当前工具轴小步插入
wheel_hole_insertion/calibrate_insert_tip_pivot.py  # 插入杆尖 pivot 标定
wheel_hole_insertion/capture_rgbd_once_headless.py  # 无界面采集一帧 RGBD
wheel_hole_insertion/label_dataset/best.onnx        # 当前 YOLO-seg 推理权重
```

更详细的运行说明见：

```text
wheel_hole_insertion/README.md
```

## 配置

所有设备相关参数集中在：

```text
wheel_hole_insertion/config.yaml
```

其中包括：

- 机器人 IP、端口、初始关节位姿。
- 手眼矩阵 `hand_eye.matrix`。
- 杆尖 TCP 偏置 `tool.tip_tcp_m`。
- 视觉补偿 `observed_right_offset_m` / `observed_up_offset_m`。
- 预插入距离、插入距离、安全阈值。
- 相机分辨率、YOLO 权重路径、置信度阈值。
- 前端速度、夹爪参数、VLA 数据保存路径。

当前摄像头曾被撞歪，因此配置中默认：

```yaml
hand_eye:
  valid: false
```

在重新完成手眼标定前，真实运动脚本会拒绝执行，避免继续使用旧矩阵。重新标定后，将新的矩阵写入 `hand_eye.matrix`，并设置：

```yaml
hand_eye:
  valid: true
```

## 启动前端

推荐现场使用前端：

```bash
cd /home/wooshrobot/bai/hand_eye_calibration
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/realsense_preview_server.py
```

浏览器访问：

```text
http://<设备IP>:8090
```

前端功能：

- 实时查看 RealSense 彩色画面。
- 切换 YOLO 识别叠加图。
- 回到初始位。
- 执行法向对齐、预插入、小步插入。
- 控制夹爪开合。
- 调整运动速度。
- 保存 RGB 视频和机械臂状态数据。

## 典型流程

1. 固定相机和插入杆，完成手眼标定并更新 `config.yaml`。
2. 使用 `calibrate_insert_tip_pivot.py` 标定插入杆尖相对 TCP 的偏置。
3. 启动前端，确认中心孔在画面中且 YOLO 识别稳定。
4. 执行法向对齐姿态调试。
5. 执行预插入，人工确认杆尖是否对准中心孔。
6. 沿当前工具轴小步插入。
7. 如需训练 VLA，点击前端保存按钮记录视频和机械臂状态。

## 离线检测

采集一帧 RGBD：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/capture_rgbd_once_headless.py
```

对采集结果运行识别：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/detect_center_hole_yolo_rgbd.py <RGBD采集目录>
```

输出包含：

- `*_detection.json`
- `*_overlay.png`

## 数据保存

前端“开始保存 / 停止保存”会按 episode 保存数据，默认目录：

```text
wheel_hole_insertion/vla_recordings/
```

单个 episode 结构：

```text
episode_000000/
  videos/camera_rgb.mp4
  data/frames.jsonl
  meta/info.json
  meta/episode.json
  meta/tasks.jsonl
```

视频帧和机械臂状态通过同一个 `frame_index` 对齐。

## 安全说明

- 当前手眼矩阵无效时，运动脚本默认拒绝执行。
- 运动过程中可使用急停按钮，或在命令行模式按 `s`、空格、回车、`q` 停止。
- 插入前必须人工确认 overlay、中心孔位置、法向和杆尖实际位置。
- 如果相机、夹爪、插入杆发生碰撞或松动，需要重新确认手眼标定和杆尖 TCP 标定。
