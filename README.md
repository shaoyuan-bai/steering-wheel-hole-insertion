# Steering Wheel Center-Hole Insertion

This repository contains the vision-guided insertion and transfer workflow for an RM65 robot arm, a gripper-mounted RGBD camera, a custom insertion tool, a mobile base, and a lift platform.

The main task is to insert the tool into the metal center hole of a steering wheel. The wheel is hanging on a rack and may be tilted differently each time, so the system does not rely on a fixed world pose. Instead, it detects the center hole, estimates the local normal from RGBD depth, transforms the target into the robot base frame through hand-eye calibration, and then executes guarded insertion and transfer motions.

Chinese documentation is kept in [README.zh-CN.md](README.zh-CN.md).

## Current Capabilities

- Single-class YOLO segmentation for `center_hole`.
- RGBD-based local plane fitting around the detected metal ring to estimate hole center and normal.
- Eye-in-hand camera to robot-base transform using the hand-eye matrix in configuration.
- Tool-tip offset compensation through pivot calibration.
- Web UI for camera preview, YOLO overlay, robot motion, gripper control, lift/base control, emergency stop, and VLA-style data recording.
- RealSense camera service support to avoid multiple projects competing for the same camera device.
- Automated insertion and loading flows, including linear pre-release and return-to-grasp sequences.

## Main Directory

Most project-specific code is under:

```text
wheel_hole_insertion/
```

Important files:

```text
wheel_hole_insertion/config.yaml                     # Robot, camera, calibration, offsets, speed, and safety parameters
wheel_hole_insertion/realsense_preview_server.py     # LAN web UI and workflow server
wheel_hole_insertion/detect_center_hole_yolo_rgbd.py # YOLO + RGBD center-hole detection
wheel_hole_insertion/move_to_center_hole.py          # Motion planning and pre-insertion execution
wheel_hole_insertion/continue_insert_along_axis.py   # Continue insertion along the current tool axis
wheel_hole_insertion/calibrate_insert_tip_pivot.py   # Pivot calibration for the insertion tip
wheel_hole_insertion/capture_rgbd_once_headless.py   # Headless RGBD capture
wheel_hole_insertion/table_place_poses.json          # Taught placement and release poses
wheel_hole_insertion/label_dataset/best.onnx         # Current YOLO segmentation model
```

Camera service files:

```text
camera_service/camera_service.py
camera_service/cameras.json
```

## Configuration

Runtime parameters are centralized in:

```text
wheel_hole_insertion/config.yaml
```

The file includes:

- Robot IP, port, and initial joint pose.
- Hand-eye transform under `hand_eye.matrix`.
- Tool-tip offset under `tool.tip_tcp_m`.
- Visual correction offsets under `insertion`.
- Pre-insertion, insertion, and safety limits.
- Camera resolution and YOLO model path.
- Gripper parameters.
- Mobile base and lift platform endpoints and speeds.
- Automated workflow distances, heights, and delay values.

Hand-eye calibration validity is controlled by:

```yaml
hand_eye:
  valid: true
```

If the camera, gripper, or tool mount is moved or hit, redo or verify the hand-eye calibration and tool-tip pivot calibration before running insertion motions.

## Start the Web UI

Use the `cyy` environment:

```bash
cd /home/wooshrobot/bai/hand_eye_calibration
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/realsense_preview_server.py
```

Open in a browser:

```text
http://<device-ip>:8090
```

The web UI provides:

- RealSense RGB preview.
- Optional YOLO overlay.
- Initial-pose return.
- Normal alignment.
- Planned pre-insertion.
- Manual 10 mm insertion step.
- Gripper open/close.
- Lift and mobile-base controls.
- Emergency stop for robot and mobile platform requests.
- VLA-style recording of synchronized video and robot state.

## Current Automated Workflows

### Steering Wheel Loading

The `抓取方向盘上料` button runs the full loading sequence. The current implementation performs:

1. Close gripper.
2. Capture RGBD and detect the steering-wheel center hole.
3. Align tool orientation to the estimated hole normal.
4. Move to the planned pre-insertion pose.
5. Insert once by 40 mm along the current tool axis.
6. Open gripper.
7. Lift up by 0.02 m.
8. Move the base backward.
9. Move the arm to the vertical pre-release pose.
10. Move lift and base/arm through the configured release sequence.

Exact distances, heights, speeds, and safety checks are configured in `wheel_hole_insertion/config.yaml`.

### Linear Pre-Release Loading

The `直线待释放上料` button runs a variant that uses the taught pose:

```text
poses.linear_pre_release
```

The current sequence is:

1. Close gripper.
2. Capture RGBD and detect the center hole.
3. Align to the estimated hole normal.
4. Move to planned pre-insertion.
5. Insert once by 40 mm.
6. Open gripper.
7. Lift up by 0.02 m.
8. Start base backward motion and lift-to-0.7 motion, then start arm motion to `linear_pre_release` after `post_insert_sequence.linear_release_base_to_arm_delay_s`.
9. Lower lift to 0.545.
10. Close gripper.

### Return to Linear Grasp

The `返回直线夹取` button runs:

1. Lift up by 0.03 m.
2. Send the mobile-base forward 2 m request.
3. After `post_insert_return_sequence.linear_grasp_base_to_arm_lift_delay_s`, move the arm to the initial pose while moving the lift to 0.7.
4. Wait for the base request to finish.

## Offline RGBD Capture and Detection

Capture one RGBD sample:

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/capture_rgbd_once_headless.py
```

Run detection:

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/detect_center_hole_yolo_rgbd.py <rgbd_capture_dir>
```

Outputs include:

```text
*_detection.json
*_overlay.png
```

## Tool-Tip Pivot Calibration

Run:

```bash
/home/wooshrobot/miniconda3/envs/cyy/bin/python wheel_hole_insertion/calibrate_insert_tip_pivot.py
```

Keep the physical insertion point fixed while moving the arm through multiple orientations. The script writes the calibrated tip offset to:

```text
wheel_hole_insertion/insert_tip_pivot_calibration.json
```

Copy the validated value into `tool.tip_tcp_m` in `config.yaml`.

## VLA-Style Data Recording

The web UI can record synchronized RGB video and robot state. Episodes are saved under:

```text
wheel_hole_insertion/vla_recordings/
```

Typical structure:

```text
episode_000000/
  videos/camera_rgb.mp4
  data/frames.jsonl
  meta/info.json
  meta/episode.json
  meta/tasks.jsonl
```

Video frames and robot state records are aligned by `frame_index`.

## Safety Notes

- Confirm the YOLO overlay, center-hole location, normal direction, and physical tip alignment before insertion.
- Recalibrate hand-eye and tool-tip offset after any camera/tool collision or mount change.
- Use the web emergency stop button if motion becomes unsafe.
- Do not run automated insertion if `hand_eye.valid` is false or if the current setup does not match the saved calibration.
- The mobile-base and lift APIs are external services; if concurrent behavior looks serialized, inspect the service at `mobile_base.base_url`.
