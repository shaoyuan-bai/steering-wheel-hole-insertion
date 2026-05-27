# -*- coding: utf-8 -*-
"""
Eye-in-hand RGBD steering-wheel center-hole insertion helper.

Workflow:
- Move the right wrist camera so the center hole is visible.
- Press "p" to auto-detect the center hole, fit the local wheel plane, and
  build an insertion plan.
- Press "m" to movej to the pre-insertion pose.
- Press "i" to execute the short straight insertion.
- Press "b" to back out to the pre-insertion pose.
- Press "q" to quit.

The script is deliberately dry-run by default. Add --execute to allow robot
motion commands.
"""

import argparse
import json
import math
import socket
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rm65_sdk_safe_ik import Rm65SafeIkMover  # noqa: E402
import head_rgbd_detect_wheel as wheel_rgbd  # noqa: E402


SCRIPT_VERSION = "2026-05-20-eye-in-hand-wheel-hole-insertion-v1"

ROBOT_IP = "169.254.128.21"
ROBOT_PORT = 8080
SDK_FORCE_TYPE_NAME = "RM_MODEL_RM_SF_E"

MOVEJ_SPEED = 8
MOVEL_SPEED = 5

T_EE_CAM = np.array([
    [0.83343679, -0.55261378, 0.00106433, 0.02416865],
    [0.55261480, 0.83343547, -0.00148271, -0.10373028],
    [-0.00006769, 0.00182391, 0.99999833, 0.04847300],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

clicked_pixel = None
auto_hole = None
cached_plan = None


def on_mouse(event, x, y, flags, param):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (int(x), int(y))
        print(f"[CLICK] center-hole pixel=({x}, {y})")


def decode_json_stream(text):
    decoder = json.JSONDecoder()
    objects = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            next_line = text.find("\n", idx)
            if next_line < 0:
                break
            idx = next_line + 1
            continue
        objects.append(obj)
        idx += end
    return objects


def get_current_state(sock):
    try:
        cmd = {"command": "get_current_arm_state"}
        sock.sendall((json.dumps(cmd) + "\r\n").encode("utf-8"))
        time.sleep(0.05)
        data = sock.recv(8192).decode("utf-8", "ignore")
        for obj in reversed(decode_json_stream(data)):
            if "arm_state" not in obj:
                continue
            arm_state = obj["arm_state"]
            p = arm_state["pose"]
            j = arm_state["joint"]
            pose = [
                float(p[0] / 1e6),
                float(p[1] / 1e6),
                float(p[2] / 1e6),
                float(p[3] / 1000),
                float(p[4] / 1000),
                float(p[5] / 1000),
            ]
            joint = [float(x / 1000) for x in j[:6]]
            return pose, joint
    except Exception as exc:
        print(f"[ERROR] get_current_state failed: {exc}")
    return None


def pose_to_matrix(pose_xyzrpy):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rpy_xyz_to_matrix(pose_xyzrpy[3:])
    t[:3, 3] = pose_xyzrpy[:3]
    return t


def rpy_xyz_to_matrix(rpy):
    """Return rotation matrix for RM-style rx, ry, rz in radians."""
    rx, ry, rz = [float(v) for v in rpy[:3]]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ], dtype=np.float64)


def matrix_to_rpy_xyz(rot):
    """Inverse of rpy_xyz_to_matrix for non-gimbal-lock poses."""
    rot = np.asarray(rot, dtype=np.float64)
    sy = -float(rot[2, 0])
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-8:
        rx = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        rz = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        rx = 0.0
        rz = math.atan2(-float(rot[0, 1]), float(rot[1, 1]))
    return np.array([rx, ry, rz], dtype=np.float64)


def normalize(vec, name="vector"):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise RuntimeError(f"Cannot normalize near-zero {name}.")
    return vec / norm


def backproject_pixels(xs, ys, zs, intr):
    x = (xs.astype(np.float64) - intr_value(intr, "ppx")) * zs / intr_value(intr, "fx")
    y = (ys.astype(np.float64) - intr_value(intr, "ppy")) * zs / intr_value(intr, "fy")
    return np.column_stack([x, y, zs.astype(np.float64)])


def intr_value(intr, key):
    if isinstance(intr, dict):
        return float(intr[key])
    return float(getattr(intr, key))


def intersect_pixel_ray_with_plane(u, v, plane_point, normal, intr):
    ray = np.array([
        (float(u) - intr_value(intr, "ppx")) / intr_value(intr, "fx"),
        (float(v) - intr_value(intr, "ppy")) / intr_value(intr, "fy"),
        1.0,
    ], dtype=np.float64)
    denom = float(np.dot(normal, ray))
    if abs(denom) < 1e-9:
        raise RuntimeError("Clicked center ray is nearly parallel to the fitted wheel plane.")
    scale = float(np.dot(normal, plane_point) / denom)
    if scale <= 0:
        raise RuntimeError("Plane intersection is behind the camera; check the clicked point.")
    return ray * scale


def fit_plane_pca(points):
    center = points.mean(axis=0)
    demeaned = points - center
    _, _, vh = np.linalg.svd(demeaned, full_matrices=False)
    normal = normalize(vh[-1], "plane normal")
    if normal[2] > 0:
        normal = -normal
    distances = (points - center) @ normal
    return center, normal, distances


def robust_plane(points, iterations, threshold_m):
    selected = np.asarray(points, dtype=np.float64)
    if len(selected) < 80:
        raise RuntimeError(f"Not enough depth points for local plane fit: {len(selected)}")
    for _ in range(max(1, int(iterations))):
        center, normal, distances = fit_plane_pca(selected)
        keep = np.abs(distances) <= float(threshold_m)
        if keep.sum() < 80 or keep.sum() == len(selected):
            break
        selected = selected[keep]
    center, normal, distances = fit_plane_pca(selected)
    rmse = float(np.sqrt(np.mean(distances * distances)))
    return center, normal, selected, rmse


def fit_hole_plane_from_click(depth_m, intr, pixel, args):
    u, v = pixel
    h, w = depth_m.shape
    yy, xx = np.indices((h, w))
    radius_px = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
    mask = (
        (radius_px >= float(args.plane_inner_radius_px))
        & (radius_px <= float(args.plane_outer_radius_px))
        & (depth_m >= float(args.min_depth))
        & (depth_m <= float(args.max_depth))
    )
    ys, xs = np.nonzero(mask)
    if len(xs) < int(args.min_plane_points):
        raise RuntimeError(
            f"Only {len(xs)} valid depth points in annulus. "
            "Increase radii, improve view, or tune depth limits."
        )

    zs = depth_m[ys, xs].astype(np.float64)
    points = backproject_pixels(xs, ys, zs, intr)
    plane_center, normal_cam, inliers, rmse = robust_plane(
        points,
        iterations=args.plane_iters,
        threshold_m=args.plane_inlier_m,
    )
    center_cam = intersect_pixel_ray_with_plane(u, v, plane_center, normal_cam, intr)
    return {
        "center_cam": center_cam,
        "normal_cam": normal_cam,
        "plane_center_cam": plane_center,
        "plane_points": int(len(points)),
        "plane_inliers": int(len(inliers)),
        "plane_rmse_m": rmse,
    }


def fit_hole_plane_from_auto(depth_m, intr, hole, args):
    center = hole["center_px"]
    radius = float(hole["radius_px"])
    local_args = argparse.Namespace(**vars(args))
    local_args.plane_inner_radius_px = max(
        float(args.plane_inner_radius_px),
        radius * float(args.auto_plane_inner_scale),
    )
    local_args.plane_outer_radius_px = max(
        float(args.plane_outer_radius_px),
        radius * float(args.auto_plane_outer_scale),
    )
    detection = fit_hole_plane_from_click(
        depth_m=depth_m,
        intr=intr,
        pixel=(int(round(center[0])), int(round(center[1]))),
        args=local_args,
    )
    detection["hole_center_px"] = [float(center[0]), float(center[1])]
    detection["hole_radius_px"] = float(radius)
    detection["hole_detect_score"] = float(hole["score"])
    detection["auto_plane_inner_radius_px"] = float(local_args.plane_inner_radius_px)
    detection["auto_plane_outer_radius_px"] = float(local_args.plane_outer_radius_px)
    return detection


def contour_circle_metrics(contour):
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 1.0 or perimeter <= 1e-6:
        return None
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = math.pi * radius * radius
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    fill_ratio = area / max(circle_area, 1e-6)
    return {
        "center_px": np.array([float(x), float(y)], dtype=np.float64),
        "radius_px": float(radius),
        "area": area,
        "circularity": float(circularity),
        "fill_ratio": float(fill_ratio),
    }


def annulus_valid_depth_count(depth_m, center, inner_radius, outer_radius, args):
    h, w = depth_m.shape
    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = (
        (dist >= inner_radius)
        & (dist <= outer_radius)
        & (depth_m >= float(args.min_depth))
        & (depth_m <= float(args.max_depth))
    )
    return int(mask.sum())


def evaluate_hole_candidate(center, radius, depth_m, gray, image_center, args, source):
    if not (args.min_hole_radius_px <= radius <= args.max_hole_radius_px):
        return None, ("radius", float(radius), source)

    h, w = depth_m.shape
    if not (0 <= center[0] < w and 0 <= center[1] < h):
        return None, ("center-outside", [float(center[0]), float(center[1])], source)

    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    disk = dist <= max(radius * 0.85, 2.0)
    if int(disk.sum()) < 8:
        return None, ("tiny-disk", float(radius), source)

    dark_ratio = float((gray[disk] <= int(args.hole_dark_threshold)).sum()) / float(disk.sum())
    invalid_ratio = float(
        ((depth_m[disk] < float(args.min_depth)) | (depth_m[disk] > float(args.max_depth))).sum()
    ) / float(disk.sum())
    if max(dark_ratio, invalid_ratio) < float(args.min_hole_inside_ratio):
        return None, ("inside-ratio", max(dark_ratio, invalid_ratio), source)

    plane_points = annulus_valid_depth_count(
        depth_m=depth_m,
        center=center,
        inner_radius=max(args.plane_inner_radius_px, radius * args.auto_plane_inner_scale),
        outer_radius=max(args.plane_outer_radius_px, radius * args.auto_plane_outer_scale),
        args=args,
    )
    if plane_points < int(args.min_plane_points):
        return None, ("few-plane-points", plane_points, source)

    center_penalty = args.auto_center_penalty * float(np.linalg.norm(center - image_center))
    score = (
        2.5 * dark_ratio
        + 2.0 * invalid_ratio
        + min(2.0, plane_points / max(float(args.min_plane_points), 1.0))
        + 0.25 * (1.0 if source == "hough" else 0.0)
        - center_penalty
    )
    return {
        "center_px": np.asarray(center, dtype=np.float64),
        "radius_px": float(radius),
        "area": float(math.pi * radius * radius),
        "circularity": 1.0,
        "fill_ratio": float(max(dark_ratio, invalid_ratio)),
        "inside_dark_ratio": dark_ratio,
        "inside_invalid_ratio": invalid_ratio,
        "score": float(score),
        "plane_points_est": int(plane_points),
        "source": source,
    }, None


def auto_detect_center_hole(color, depth_m, args):
    h, w = depth_m.shape
    yy, xx = np.indices((h, w))
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    roi = (
        (xx >= int(w * args.auto_roi_x0))
        & (xx <= int(w * args.auto_roi_x1))
        & (yy >= int(h * args.auto_roi_y0))
        & (yy <= int(h * args.auto_roi_y1))
    )

    valid_depth = (depth_m >= float(args.min_depth)) & (depth_m <= float(args.max_depth))
    invalid_depth = roi & (~valid_depth)
    dark = roi & (gray <= int(args.hole_dark_threshold))
    image_center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    best = None
    rejected = []

    if args.auto_method in ("outer-wheel", "hybrid"):
        wheel_candidate, wheel_reason = auto_detect_outer_wheel_center(color, depth_m, args)
        if wheel_candidate is not None:
            best = wheel_candidate
        else:
            rejected.append(wheel_reason)

    if args.auto_method == "outer-wheel" and best is not None:
        best["mask"] = np.zeros_like(gray, dtype=np.uint8)
        return best, {
            "mask": best["mask"],
            "contours": 0,
            "rejected": rejected[:12],
        }

    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=float(args.hough_dp),
        minDist=float(args.hough_min_dist_px),
        param1=float(args.hough_param1),
        param2=float(args.hough_param2),
        minRadius=int(args.min_hole_radius_px),
        maxRadius=int(args.max_hole_radius_px),
    )
    if circles is not None:
        for x, y, radius in np.round(circles[0, :]).astype(np.float64):
            if not (
                int(w * args.auto_roi_x0) <= x <= int(w * args.auto_roi_x1)
                and int(h * args.auto_roi_y0) <= y <= int(h * args.auto_roi_y1)
            ):
                rejected.append(("hough-roi", [float(x), float(y)]))
                continue
            candidate, reason = evaluate_hole_candidate(
                center=np.array([x, y], dtype=np.float64),
                radius=float(radius),
                depth_m=depth_m,
                gray=gray,
                image_center=image_center,
                args=args,
                source="hough",
            )
            if candidate is None:
                rejected.append(reason)
                continue
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = np.zeros_like(gray, dtype=np.uint8)
    contour_count = 0
    for source, mask in [("dark-contour", dark), ("invalid-depth-contour", invalid_depth)]:
        current_u8 = (mask.astype(np.uint8) * 255)
        current_u8 = cv2.morphologyEx(current_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        current_u8 = cv2.morphologyEx(current_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_u8 = cv2.bitwise_or(mask_u8, current_u8)
        contours, _ = cv2.findContours(current_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count += len(contours)
        for contour in contours:
            metrics = contour_circle_metrics(contour)
            if metrics is None:
                continue
            if metrics["area"] < args.min_hole_area_px:
                rejected.append(("area", metrics["area"], source))
                continue
            if metrics["circularity"] < args.min_hole_circularity:
                rejected.append(("circularity", metrics["circularity"], source))
                continue
            candidate, reason = evaluate_hole_candidate(
                center=metrics["center_px"],
                radius=metrics["radius_px"],
                depth_m=depth_m,
                gray=gray,
                image_center=image_center,
                args=args,
                source=source,
            )
            if candidate is None:
                rejected.append(reason)
                continue
            candidate["area"] = metrics["area"]
            candidate["circularity"] = metrics["circularity"]
            candidate["fill_ratio"] = metrics["fill_ratio"]
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is not None:
        best["mask"] = mask_u8
        return best, {
            "mask": mask_u8,
            "contours": contour_count,
            "rejected": rejected[:12],
        }

    return None, {
        "mask": mask_u8,
        "contours": contour_count,
        "rejected": rejected[:12],
    }


def auto_detect_outer_wheel_center(color, depth_m, args):
    h, w = depth_m.shape
    yy, xx = np.indices((h, w))
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    roi = (
        (xx >= int(w * args.outer_roi_x0))
        & (xx <= int(w * args.outer_roi_x1))
        & (yy >= int(h * args.outer_roi_y0))
        & (yy <= int(h * args.outer_roi_y1))
    )
    valid_depth = (depth_m > args.min_depth) & (depth_m < args.max_depth)
    dark = gray < int(args.outer_dark_threshold)
    wheel_mask = roi & valid_depth & dark
    wheel_u8 = cv2.morphologyEx(
        (wheel_mask.astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    eroded = cv2.erode(wheel_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
    boundary = (wheel_u8 > 0) & (~eroded)
    b_ys, b_xs = np.nonzero(boundary)
    boundary_points = np.column_stack([b_xs, b_ys]).astype(np.float64)
    if len(boundary_points) < int(args.outer_min_boundary_points):
        return None, ("outer-wheel-boundary-points", int(len(boundary_points)))

    ransac_args = argparse.Namespace(
        ransac_seed=args.outer_ransac_seed,
        ransac_iters=args.outer_ransac_iters,
        min_radius_px=args.outer_min_radius_px,
        max_radius_px=args.outer_max_radius_px,
        circle_inlier_px=args.outer_circle_inlier_px,
        center_penalty=args.outer_center_penalty,
    )
    try:
        center, radius, inliers = wheel_rgbd.ransac_outer_circle(
            boundary_points,
            depth_m.shape,
            ransac_args,
        )
    except Exception as exc:
        return None, ("outer-wheel-ransac", str(exc))

    if int(inliers.sum()) < int(args.outer_min_inliers):
        return None, ("outer-wheel-inliers", int(inliers.sum()))

    plane_points = annulus_valid_depth_count(
        depth_m=depth_m,
        center=center,
        inner_radius=float(args.plane_inner_radius_px),
        outer_radius=float(args.plane_outer_radius_px),
        args=args,
    )
    if plane_points < int(args.min_plane_points):
        return None, ("outer-wheel-center-plane-points", plane_points)

    score = 8.0 + min(2.0, int(inliers.sum()) / max(float(args.outer_min_inliers), 1.0))
    return {
        "center_px": np.asarray(center, dtype=np.float64),
        "radius_px": float(args.plane_inner_radius_px),
        "outer_wheel_radius_px": float(radius),
        "area": float(math.pi * radius * radius),
        "circularity": 1.0,
        "fill_ratio": 1.0,
        "inside_dark_ratio": 0.0,
        "inside_invalid_ratio": 0.0,
        "score": float(score),
        "plane_points_est": int(plane_points),
        "source": "outer-wheel-center",
    }, None


def parse_tool_axis(text):
    axes = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    key = str(text).strip().lower()
    if key not in axes:
        raise RuntimeError(f"Unsupported --tool-axis {text!r}. Use one of {sorted(axes)}.")
    return axes[key]


def axis_index_and_sign(axis):
    idx = int(np.argmax(np.abs(axis)))
    sign = 1.0 if axis[idx] >= 0 else -1.0
    return idx, sign


def build_rotation_aligning_tool_axis(current_rot, tool_axis_local, target_axis_base):
    target = normalize(target_axis_base, "target insertion axis")
    axis_idx, axis_sign = axis_index_and_sign(tool_axis_local)
    columns = [None, None, None]
    columns[axis_idx] = axis_sign * target

    preferred_order = [0, 1, 2]
    preferred_order.remove(axis_idx)
    for candidate_idx in preferred_order:
        hint = current_rot[:, candidate_idx]
        projected = hint - np.dot(hint, columns[axis_idx]) * columns[axis_idx]
        if np.linalg.norm(projected) > 1e-6:
            columns[candidate_idx] = normalize(projected, "projected roll hint")
            break

    if columns[preferred_order[0]] is None and columns[preferred_order[1]] is None:
        hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(hint, columns[axis_idx])) > 0.95:
            hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        candidate_idx = preferred_order[0]
        projected = hint - np.dot(hint, columns[axis_idx]) * columns[axis_idx]
        columns[candidate_idx] = normalize(projected, "fallback roll hint")

    missing_idx = [i for i, col in enumerate(columns) if col is None][0]
    known_idx = [i for i, col in enumerate(columns) if col is not None and i != axis_idx][0]

    if (axis_idx, known_idx, missing_idx) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        columns[missing_idx] = normalize(np.cross(columns[axis_idx], columns[known_idx]), "cross axis")
    else:
        columns[missing_idx] = normalize(np.cross(columns[known_idx], columns[axis_idx]), "cross axis")

    rot = np.column_stack(columns)
    if np.linalg.det(rot) < 0:
        columns[missing_idx] = -columns[missing_idx]
        rot = np.column_stack(columns)
    return rot


def transform_detection_to_base(pose_now, detection, args):
    t_base_ee = pose_to_matrix(pose_now)
    t_base_cam = t_base_ee @ T_EE_CAM
    center_cam_h = np.r_[detection["center_cam"], 1.0]
    center_base = (t_base_cam @ center_cam_h)[:3]
    normal_base = normalize(t_base_cam[:3, :3] @ detection["normal_cam"], "base normal")
    insertion_axis_base = -normal_base
    if args.reverse_insert_axis:
        insertion_axis_base = -insertion_axis_base
    return center_base, normal_base, normalize(insertion_axis_base, "base insertion axis")


def build_insertion_plan(pose_now, detection, args):
    center_base, normal_base, insertion_axis_base = transform_detection_to_base(pose_now, detection, args)
    current_rot = rpy_xyz_to_matrix(pose_now[3:])
    tool_axis_local = parse_tool_axis(args.tool_axis)
    target_rot = build_rotation_aligning_tool_axis(current_rot, tool_axis_local, insertion_axis_base)
    rpy = matrix_to_rpy_xyz(target_rot)

    final_tip = center_base + insertion_axis_base * float(args.insert_depth_m)
    pre_tip = center_base - insertion_axis_base * float(args.preinsert_distance_m)
    tcp_offset_base = target_rot @ (tool_axis_local * float(args.tcp_to_tip_m))
    final_tcp = final_tip - tcp_offset_base
    pre_tcp = pre_tip - tcp_offset_base

    pre_pose = [
        float(round(pre_tcp[0], 3)),
        float(round(pre_tcp[1], 3)),
        float(round(pre_tcp[2], 3)),
        float(rpy[0]),
        float(rpy[1]),
        float(rpy[2]),
    ]
    final_pose = [
        float(round(final_tcp[0], 3)),
        float(round(final_tcp[1], 3)),
        float(round(final_tcp[2], 3)),
        float(rpy[0]),
        float(rpy[1]),
        float(rpy[2]),
    ]
    return {
        "hole_center_px": detection.get("hole_center_px"),
        "hole_radius_px": detection.get("hole_radius_px"),
        "hole_detect_score": detection.get("hole_detect_score"),
        "center_base": center_base.tolist(),
        "normal_base_toward_camera": normal_base.tolist(),
        "insertion_axis_base": insertion_axis_base.tolist(),
        "preinsert_pose": pre_pose,
        "final_pose": final_pose,
        "preinsert_move_m": float(np.linalg.norm(pre_tcp - np.asarray(pose_now[:3]))),
        "insert_move_m": float(np.linalg.norm(final_tcp - pre_tcp)),
        "tcp_to_tip_m": float(args.tcp_to_tip_m),
        "preinsert_distance_m": float(args.preinsert_distance_m),
        "insert_depth_m": float(args.insert_depth_m),
        "tool_axis": args.tool_axis,
        "plane_rmse_m": detection["plane_rmse_m"],
        "plane_points": detection["plane_points"],
        "plane_inliers": detection["plane_inliers"],
    }


def print_plan(plan):
    print("\n" + "=" * 64)
    print(f"script: {SCRIPT_VERSION}")
    if plan.get("hole_center_px") is not None:
        print(f"hole_center_px: {[round(x, 1) for x in plan['hole_center_px']]}")
        print(f"hole_radius_px: {float(plan.get('hole_radius_px', 0.0)):.1f}")
        print(f"hole_detect_score: {float(plan.get('hole_detect_score', 0.0)):.3f}")
    print(f"center_base: {[round(x, 4) for x in plan['center_base']]}")
    print(f"normal_base_toward_camera: {[round(x, 4) for x in plan['normal_base_toward_camera']]}")
    print(f"insertion_axis_base: {[round(x, 4) for x in plan['insertion_axis_base']]}")
    print(f"preinsert_pose: {[round(x, 4) for x in plan['preinsert_pose']]}")
    print(f"final_pose: {[round(x, 4) for x in plan['final_pose']]}")
    print(f"preinsert move: {plan['preinsert_move_m'] * 1000:.1f} mm")
    print(f"straight insert move: {plan['insert_move_m'] * 1000:.1f} mm")
    print(f"tcp_to_tip: {plan['tcp_to_tip_m'] * 1000:.1f} mm")
    print(f"plane rmse: {plan['plane_rmse_m'] * 1000:.2f} mm")
    print(f"plane points/inliers: {plan['plane_points']}/{plan['plane_inliers']}")
    print("=" * 64)


def check_plan_limits(plan, args):
    if plan["preinsert_move_m"] > args.max_preinsert_move_m:
        raise RuntimeError(
            f"Preinsert move {plan['preinsert_move_m'] * 1000:.1f} mm exceeds "
            f"{args.max_preinsert_move_m * 1000:.1f} mm."
        )
    if plan["insert_move_m"] > args.max_insert_move_m:
        raise RuntimeError(
            f"Insert move {plan['insert_move_m'] * 1000:.1f} mm exceeds "
            f"{args.max_insert_move_m * 1000:.1f} mm."
        )
    if plan["plane_rmse_m"] > args.max_plane_rmse_m:
        raise RuntimeError(
            f"Plane fit rmse {plan['plane_rmse_m'] * 1000:.2f} mm exceeds "
            f"{args.max_plane_rmse_m * 1000:.2f} mm."
        )


def solve_and_movej(sdk_mover, current_joint, target_pose, speed):
    old_speed = sdk_mover.speed
    sdk_mover.speed = int(speed)
    try:
        best_joint, diagnostics = sdk_mover.solve_best_joint(current_joint, target_pose)
        if best_joint is None:
            print(f"[IK] no safe solution. accepted={len(diagnostics.get('accepted', []))}, "
                  f"rejected={len(diagnostics.get('rejected', []))}")
            for item in diagnostics.get("rejected", [])[:8]:
                print(f"[IK] reject sample: {item}")
            return False
        print(f"[MOVEJ] target joint: {[round(x, 3) for x in best_joint]}")
        ret = sdk_mover.movej(best_joint)
        print(f"[MOVEJ] ret={ret}")
        return ret == 0
    finally:
        sdk_mover.speed = old_speed


def run_movel(sdk_mover, target_pose, speed):
    sdk_mover.connect()
    ret = sdk_mover.arm.rm_movel(target_pose, int(speed), 0, 0, 1)
    print(f"[MOVEL] ret={ret}")
    return ret == 0


def draw_overlay(image, plan, detected_hole=None, detect_debug=None, manual_click=False):
    lines = [
        "auto hole | p: plan | m: preinsert | i: insert | b: backout | c: clear | q: quit",
        f"script: {SCRIPT_VERSION}",
    ]
    if detected_hole is not None:
        center = detected_hole["center_px"]
        radius = detected_hole["radius_px"]
        px = (int(round(center[0])), int(round(center[1])))
        cv2.circle(image, px, int(round(radius)), (0, 255, 0), 2)
        cv2.drawMarker(image, px, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        lines.append(
            f"auto center=({px[0]}, {px[1]}) r={radius:.1f} score={detected_hole['score']:.2f}"
        )
    elif detect_debug is not None:
        lines.append(f"auto center: not found, contours={detect_debug.get('contours', 0)}")
    if clicked_pixel is not None and manual_click:
        cv2.circle(image, clicked_pixel, 5, (0, 0, 255), -1)
        lines.append(f"manual clicked: {clicked_pixel}")
    if plan is not None:
        lines.append("plan: READY")
    for idx, line in enumerate(lines):
        cv2.putText(image, line, (10, 24 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 255, 255), 1, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Allow robot motion. Without this, planning only.")
    parser.add_argument("--manual-click", action="store_true",
                        help="Disable automatic hole selection and use left-click as fallback.")
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=ROBOT_PORT)
    parser.add_argument("--serial", default="", help="Optional RealSense serial number.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tool-axis", default="+z", choices=["+x", "-x", "+y", "-y", "+z", "-z"],
                        help="TCP-frame axis that points along the physical insertion direction.")
    parser.add_argument("--reverse-insert-axis", action="store_true",
                        help="Use the opposite side of the fitted plane normal.")
    parser.add_argument("--tcp-to-tip-m", type=float, default=0.150,
                        help="Distance from TCP to insertion tip along --tool-axis.")
    parser.add_argument("--preinsert-distance-m", type=float, default=0.080)
    parser.add_argument("--insert-depth-m", type=float, default=0.020)
    parser.add_argument("--max-preinsert-move-m", type=float, default=0.45)
    parser.add_argument("--max-insert-move-m", type=float, default=0.12)
    parser.add_argument("--plane-inner-radius-px", type=float, default=28)
    parser.add_argument("--plane-outer-radius-px", type=float, default=95)
    parser.add_argument("--min-plane-points", type=int, default=500)
    parser.add_argument("--plane-iters", type=int, default=5)
    parser.add_argument("--plane-inlier-m", type=float, default=0.006)
    parser.add_argument("--max-plane-rmse-m", type=float, default=0.006)
    parser.add_argument("--min-depth", type=float, default=0.18)
    parser.add_argument("--max-depth", type=float, default=1.20)
    parser.add_argument("--hole-dark-threshold", type=int, default=70)
    parser.add_argument("--auto-method", choices=["outer-wheel", "hole", "hybrid"], default="hybrid",
                        help="outer-wheel uses the steering-wheel outer circle center; hole uses only small center-hole cues.")
    parser.add_argument("--min-hole-radius-px", type=float, default=12)
    parser.add_argument("--max-hole-radius-px", type=float, default=90)
    parser.add_argument("--min-hole-area-px", type=float, default=250)
    parser.add_argument("--min-hole-circularity", type=float, default=0.45)
    parser.add_argument("--min-hole-inside-ratio", type=float, default=0.35)
    parser.add_argument("--auto-center-penalty", type=float, default=0.006)
    parser.add_argument("--auto-plane-inner-scale", type=float, default=1.4)
    parser.add_argument("--auto-plane-outer-scale", type=float, default=3.2)
    parser.add_argument("--auto-roi-x0", type=float, default=0.12)
    parser.add_argument("--auto-roi-x1", type=float, default=0.88)
    parser.add_argument("--auto-roi-y0", type=float, default=0.10)
    parser.add_argument("--auto-roi-y1", type=float, default=0.90)
    parser.add_argument("--hough-dp", type=float, default=1.2)
    parser.add_argument("--hough-min-dist-px", type=float, default=45)
    parser.add_argument("--hough-param1", type=float, default=90)
    parser.add_argument("--hough-param2", type=float, default=18)
    parser.add_argument("--outer-dark-threshold", type=int, default=80)
    parser.add_argument("--outer-min-radius-px", type=float, default=150)
    parser.add_argument("--outer-max-radius-px", type=float, default=280)
    parser.add_argument("--outer-circle-inlier-px", type=float, default=8.0)
    parser.add_argument("--outer-ransac-iters", type=int, default=2500)
    parser.add_argument("--outer-ransac-seed", type=int, default=2)
    parser.add_argument("--outer-center-penalty", type=float, default=0.02)
    parser.add_argument("--outer-min-boundary-points", type=int, default=300)
    parser.add_argument("--outer-min-inliers", type=int, default=120)
    parser.add_argument("--outer-roi-x0", type=float, default=0.08)
    parser.add_argument("--outer-roi-x1", type=float, default=0.96)
    parser.add_argument("--outer-roi-y0", type=float, default=0.02)
    parser.add_argument("--outer-roi-y1", type=float, default=0.98)
    parser.add_argument("--movej-speed", type=int, default=MOVEJ_SPEED)
    parser.add_argument("--movel-speed", type=int, default=MOVEL_SPEED)
    return parser.parse_args()


def main():
    global clicked_pixel, auto_hole, cached_plan
    args = parse_args()

    try:
        import pyrealsense2 as rs
    except Exception as exc:
        print(
            "[FAIL] Cannot import pyrealsense2. Install/fix the RealSense Python binding "
            f"for this Python environment first: {exc}"
        )
        return

    print(f"[INFO] script version: {SCRIPT_VERSION}")
    print("[INFO] Direction: auto-detect center hole, fit local wheel plane, then insert along -normal.")
    if args.manual_click:
        print("[INFO] --manual-click enabled: p will use the clicked point instead of auto detection.")
    if not args.execute:
        print("[DRY-RUN] Robot motion is disabled. Add --execute to allow m/i/b movement keys.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.robot_ip, args.robot_port))
        print(f"[OK] robot socket connected: {args.robot_ip}:{args.robot_port}")
    except Exception as exc:
        print(f"[FAIL] socket connect failed: {exc}")
        return

    sdk_mover = Rm65SafeIkMover(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        speed=args.movej_speed,
        force_type_name=SDK_FORCE_TYPE_NAME,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    try:
        profile = pipeline.start(config)
    except Exception as exc:
        sock.close()
        sdk_mover.close()
        print(f"[FAIL] RealSense start failed: {exc}")
        return

    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    cv2.namedWindow("Wheel Center Hole Insertion", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Wheel Center Hole Insertion", on_mouse)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_z16 = np.asanyarray(depth_frame.get_data())
            depth_m = depth_z16.astype(np.float32) * depth_scale
            if args.manual_click:
                auto_hole, detect_debug = None, None
            else:
                auto_hole, detect_debug = auto_detect_center_hole(color, depth_m, args)

            vis = color.copy()
            draw_overlay(
                vis,
                cached_plan,
                detected_hole=auto_hole,
                detect_debug=detect_debug,
                manual_click=args.manual_click,
            )
            cv2.imshow("Wheel Center Hole Insertion", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                clicked_pixel = None
                auto_hole = None
                cached_plan = None
                print("[RESET] cleared click and cached plan")
                continue
            if key == ord("p"):
                if args.manual_click and clicked_pixel is None:
                    print("[WARN] click the center hole first, or run without --manual-click.")
                    continue
                if not args.manual_click and auto_hole is None:
                    print(f"[WARN] auto center-hole detection failed: {detect_debug}")
                    continue
                state = get_current_state(sock)
                if not state:
                    print("[ERROR] cannot read robot state.")
                    continue
                pose_now, _joint_now = state
                try:
                    if args.manual_click:
                        detection = fit_hole_plane_from_click(depth_m, intr, clicked_pixel, args)
                    else:
                        detection = fit_hole_plane_from_auto(depth_m, intr, auto_hole, args)
                    plan = build_insertion_plan(pose_now, detection, args)
                    check_plan_limits(plan, args)
                    cached_plan = plan
                    print_plan(plan)
                except Exception as exc:
                    cached_plan = None
                    print(f"[PLAN ERROR] {exc}")
                    traceback.print_exc()
                continue
            if key in (ord("m"), ord("i"), ord("b")):
                if cached_plan is None:
                    print("[WARN] no cached plan. Press p first.")
                    continue
                if not args.execute:
                    print("[DRY-RUN] motion blocked. Restart with --execute to move the robot.")
                    continue
                state = get_current_state(sock)
                if not state:
                    print("[ERROR] cannot read robot state.")
                    continue
                pose_now, joint_now = state
                try:
                    if key == ord("m"):
                        print("[MOVE] movej to pre-insertion pose")
                        solve_and_movej(sdk_mover, joint_now, cached_plan["preinsert_pose"], args.movej_speed)
                    elif key == ord("i"):
                        current_to_final = float(np.linalg.norm(
                            np.asarray(cached_plan["final_pose"][:3]) - np.asarray(pose_now[:3])
                        ))
                        if current_to_final > args.max_insert_move_m:
                            print(
                                f"[SAFETY] current-to-final {current_to_final * 1000:.1f} mm exceeds "
                                f"{args.max_insert_move_m * 1000:.1f} mm. Move to preinsert first."
                            )
                            continue
                        print("[MOVE] straight insertion movel")
                        run_movel(sdk_mover, cached_plan["final_pose"], args.movel_speed)
                    elif key == ord("b"):
                        print("[MOVE] back out to pre-insertion pose")
                        run_movel(sdk_mover, cached_plan["preinsert_pose"], args.movel_speed)
                except Exception as exc:
                    print(f"[MOVE ERROR] {exc}")
                    traceback.print_exc()
    finally:
        sdk_mover.close()
        pipeline.stop()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
