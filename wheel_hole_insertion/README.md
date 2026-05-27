# 方向盘中心孔插入

这个目录用于“眼在手上”的 RGBD 相机引导夹爪末端插入方向盘中心孔。方向盘相对地面倾斜时，不应该用固定世界姿态插入；应当从 RGBD 深度拟合中心孔附近的方向盘平面，并让夹爪插入轴沿孔轴方向进入。

## 当前进度

- 已接入 YOLO11n-seg 单类 `center_hole` 的 ONNX 推理，默认权重为 `label_dataset/best.onnx`。
- 已实现 YOLO mask + RGBD 金属环局部平面拟合，输出中心孔位置、法向和质量评估。
- 已实现基于手眼外参的 base 坐标规划，支持保存/加载固定计划，避免移动后用旧相机检测重新换算。
- 已加入按键急停和独立停止脚本：运动时按 `s`、空格、回车或 `q` 会发送 `rm_set_arm_stop()`。
- 已加入 `observed-right-offset-m` / `observed-up-offset-m` 视觉偏差补偿，当前保留两版计划：
  - `stability_verify_A_reprocessed/base_mean_preinsert_plan_corrected.json`：右 2.5 mm，上 3.0 mm。
  - `stability_verify_A_reprocessed/base_mean_preinsert_plan_corrected_r45_u30.json`：右 4.5 mm，上 3.0 mm。
- 已实现内网 RealSense 前端 `realsense_preview_server.py`，支持彩色预览、截图和“回初始位置”按钮。
- 已实现现场一键流程 `run_live_yolo_preinsert.py`：每次重新拍照、YOLO 识别、移动到预插入、移动到 YOLO 终点、再沿插入轴额外前伸。
- 现场验证 A 的有效样本显示中心点约 3 mm 级稳定，法向约 0.22 度稳定；后续重点是 TCP/插入杆偏置补偿和避开奇异位姿。

## 思路

1. 让腕部 RGBD 相机看到方向盘中心孔。
2. 程序默认先用方向盘外圈拟合整轮中心，必要时再用暗区/深度缺失的小圆作为辅助候选。
3. 程序在孔周围取一个深度环带，避开孔洞中心，拟合方向盘局部平面。
4. 平面法向先转到机器人 base 坐标系。
5. 插入方向默认取“远离相机、进入方向盘”的方向，也就是拟合法向的反方向。
6. 生成两个 TCP 位姿：
   - `preinsert_pose`：孔外预插入点。
   - `final_pose`：沿孔轴短直线插入后的点。

## 运行

当前推荐现场启动方式：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/run_live_yolo_preinsert.py
```

默认每一步都会询问是否继续：

1. 实时拍照 + YOLO 识别。
2. 默认先自动回到保存的初始关节位姿，不需要确认。
3. 移动到预插入位置。
4. 移动到 YOLO 计划终点。
5. 无视 YOLO 终点，沿插入轴继续前伸 30 mm。

如果需要自动执行完整流程：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/run_live_yolo_preinsert.py -y
```

脚本默认会用较低的 YOLO 候选阈值生成检测图，但仍要求 `quality: ok` 才运动。如果现场 overlay 人工确认无误但质量被拒绝，可以显式加：

```bash
--allow-non-ok-quality
```

如果只是调试当前相机位姿，不想先回初始位置：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/run_live_yolo_preinsert.py --skip-initial
```

当前默认补偿是“从机械臂朝孔看，右 2.5 mm、上 3.0 mm”：

```bash
--right-offset-m 0.0025 --up-offset-m 0.0030
```

额外前伸默认 30 mm，可调整：

```bash
--extra-insert-m 0.030
```

默认只规划，不发运动：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/insert_center_hole.py
```

允许运动：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/insert_center_hole.py --execute
```

如果有多台 RealSense，指定序列号：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/insert_center_hole.py --serial 你的相机序列号
```

## 按键

- `p`：使用当前自动识别到的中心孔，拟合平面并打印插入计划。
- `m`：运动到预插入位姿，只在 `--execute` 下有效。
- `i`：从预插入点短直线插入，只在 `--execute` 下有效。
- `b`：直线退回预插入点，只在 `--execute` 下有效。
- `c`：清除点击点和计划。
- `q`：退出。

如果自动识别不稳定，可以临时改回手动点击：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/insert_center_hole.py --manual-click
```

## 自动识别参数

无显示器采集一帧 RGBD：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/capture_rgbd_once_headless.py --serial 405622075930
```

默认采集分辨率是 `1280x720@30fps`。D435I 的 RGB 最高可到 `1920x1080@30fps`，深度最高是 `1280x720@30fps`；为了 RGBD 对齐和法向计算，默认使用 `1280x720`。

可以先离线只跑视觉识别，不连接机械臂：

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/detect_center_hole_pose.py head_wheel_hybrid/wheel_hybrid_20260509_094055
```

输出：

- `center_hole_detection.json`
- `center_hole_overlay.png`

默认输出的是“中间金属环内圆位置 + 金属环 RGBD 法向”。具体做法：

1. 先用方向盘外圈只得到一个粗中心先验；它不参与法向计算。
2. 在粗中心附近找银色金属环的内圆。
3. 取内圆外侧、金属环上的 RGBD 点。
4. 对金属环点拟合平面，得到插入中心孔所需的法向。

整轮外圈结果只作为调试/先验字段写进 JSON，不能作为插入法向使用。

默认方法是：

```bash
--auto-method metal-ring
--normal-source metal-ring
```

overlay 中：

- 绿色圆/十字：识别到的金属环中心孔。
- 黄色圆：实际用于金属环 RGBD 平面拟合的外边界。
- 红色箭头：拟合出的金属环法向。

脚本会输出 `quality`。只有 `quality: ok` 才建议进入运动规划；如果是 `reject`，看 `quality_reason` 和 overlay 调整相机位置或参数。

当前生产视角默认使用局部边缘椭圆优先检测中心孔，因为远距离倾斜视角下中心孔更像椭圆而不是正圆。默认先验约为画面 `(0.53W, 0.67H)`，如果相机安装后中心孔长期落在别的位置，调：

```bash
--metal-prior-x 0.53 --metal-prior-y 0.67
```

也可以强制使用某一种：

```bash
--auto-method outer-wheel
--auto-method hole
--auto-method center-circle
--normal-source outer-wheel
--normal-source local
```

如果图像里方向盘外圈不完整，金属环又大致在画面中心，可以不用外圈先验：

```bash
--metal-prior-source image-center
```

中心孔通常表现为黑色圆形区域，或者深度无效的圆形区域。小孔候选可调参数：

```bash
--hole-dark-threshold 70
--min-hole-radius-px 12
--max-hole-radius-px 90
--auto-roi-x0 0.12 --auto-roi-x1 0.88 --auto-roi-y0 0.10 --auto-roi-y1 0.90
```

如果背景黑色区域干扰识别，先收窄 ROI；如果孔比较亮或曝光变化大，再调高 `--hole-dark-threshold`。

方向盘外圈候选可调参数：

```bash
--outer-min-radius-px 150
--outer-max-radius-px 280
--outer-dark-threshold 80
--outer-roi-x0 0.08 --outer-roi-x1 0.96 --outer-roi-y0 0.02 --outer-roi-y1 0.98
```

## 现场必须确认的参数

最重要的是这两个：

```bash
--tool-axis +z
--tcp-to-tip-m 0.150
```

- `--tool-axis`：TCP 坐标系里哪根轴指向“夹爪末端插入方向”。默认 `+z`，如果实际是 `-z` 或 `+x`，必须改。
- `--tcp-to-tip-m`：TCP 到真正插入尖端的距离，单位米。默认 150 mm，必须按实物测量。

如果发现规划方向在孔外反了，用：

```bash
--reverse-insert-axis
```

常用保守参数：

```bash
--preinsert-distance-m 0.080 --insert-depth-m 0.020 --movel-speed 5
```

## 调试建议

先不加 `--execute`，让绿色圆圈稳定套住中心孔后按 `p`，看终端输出：

- `center_base` 是否在合理工作空间。
- `insertion_axis_base` 是否指向孔内。
- `plane rmse` 是否小于 3-6 mm。
- `straight insert move` 是否等于 `preinsert_distance + insert_depth` 附近。

确认方向和 TCP 到尖端距离后，再加 `--execute`，并先把 `--insert-depth-m` 设小，例如 0.005 到 0.010。
