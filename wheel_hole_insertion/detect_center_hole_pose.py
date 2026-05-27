# -*- coding: utf-8 -*-
"""
Detect steering-wheel center-hole pose from one saved RGBD capture.

Input folder must contain:
- color.png
- depth_raw.npy or depth_mm.png
- intrinsics.json

Outputs by default are written to this script directory:
- center_hole_detection_<capture-folder-name>.json
- center_hole_overlay_<capture-folder-name>.png

The detector estimates:
- center-hole pixel
- center-hole 3D point in camera frame
- local wheel-plane normal in camera frame
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insert_center_hole import (  # noqa: E402
    auto_detect_center_hole,
    backproject_pixels,
    contour_circle_metrics,
    fit_hole_plane_from_auto,
    fit_hole_plane_from_click,
    intersect_pixel_ray_with_plane,
    intr_value,
    robust_plane,
)
import head_rgbd_detect_wheel as wheel_rgbd  # noqa: E402


SCRIPT_VERSION = "2026-05-21-rgbd-plane-hole-detector-v2"


def load_capture(folder):
    folder = Path(folder).expanduser().resolve()
    color = cv2.imread(str(folder / "color.png"))
    if color is None:
        raise RuntimeError(f"Cannot read {folder / 'color.png'}")

    if (folder / "depth_raw.npy").exists():
        depth_m = np.load(str(folder / "depth_raw.npy")).astype(np.float32)
    else:
        depth_mm = cv2.imread(str(folder / "depth_mm.png"), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise RuntimeError("Cannot read depth_raw.npy or depth_mm.png")
        depth_m = depth_mm.astype(np.float32) / 1000.0

    with open(folder / "intrinsics.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    intr = meta["color_intrinsics"]
    return folder, color, depth_m, intr


def point_to_pixel(point, intr):
    point = np.asarray(point, dtype=np.float64)
    if point[2] <= 1e-6:
        return None
    u = int(round(point[0] * intr_value(intr, "fx") / point[2] + intr_value(intr, "ppx")))
    v = int(round(point[1] * intr_value(intr, "fy") / point[2] + intr_value(intr, "ppy")))
    return u, v


def draw_detection_overlay(color, depth_m, detection, intr, args, source_debug=None):
    overlay = color.copy()
    center_px = detection.get("hole_center_px")
    radius_px = float(detection.get("hole_radius_px", args.plane_inner_radius_px))
    if center_px is not None:
        center_i = (int(round(center_px[0])), int(round(center_px[1])))
        fit_outer_px = detection.get("metal_ring_result", {}).get("fit_outer_radius_px")
        if fit_outer_px is None:
            fit_outer_px = detection.get("metal_outer_radius_px")
        cv2.circle(overlay, center_i, int(round(radius_px)), (0, 255, 0), 2)
        cv2.drawMarker(overlay, center_i, (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        if fit_outer_px is not None:
            cv2.circle(overlay, center_i, int(round(float(fit_outer_px))), (0, 255, 255), 1)

    center_cam = np.asarray(detection["center_cam"], dtype=np.float64)
    normal_cam = np.asarray(detection["normal_cam"], dtype=np.float64)
    local_normal_cam = np.asarray(detection.get("local_normal_cam", normal_cam), dtype=np.float64)
    center_projected = point_to_pixel(center_cam, intr)
    if center_projected is not None:
        tip = center_cam + normal_cam * float(args.normal_draw_length_m)
        tip_px = point_to_pixel(tip, intr)
        cv2.circle(overlay, center_projected, 6, (0, 0, 255), -1)
        if tip_px is not None:
            cv2.arrowedLine(overlay, center_projected, tip_px, (0, 0, 255), 2, tipLength=0.22)
            cv2.putText(overlay, "+N", (tip_px[0] + 5, tip_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    lines = [
        f"version: {SCRIPT_VERSION}",
        f"center_px: {format_vec(center_px, 1) if center_px is not None else 'manual/unknown'}",
        f"center_cam_m: {format_vec(center_cam, 4)}",
        f"normal_cam: {format_vec(normal_cam, 4)}",
        f"local_normal: {format_vec(local_normal_cam, 4)}",
        f"plane_rmse_mm: {detection['plane_rmse_m'] * 1000.0:.2f}",
        f"points/inliers: {detection['plane_points']}/{detection['plane_inliers']}",
        f"quality: {detection.get('quality', 'unknown')}",
    ]
    if source_debug:
        lines.append(f"source: {source_debug.get('source', 'manual')}")
        if source_debug.get("outer_wheel_radius_px") is not None:
            lines.append(f"outer_radius_px: {source_debug['outer_wheel_radius_px']:.1f}")

    for idx, line in enumerate(lines):
        cv2.putText(overlay, line, (12, 24 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, line, (12, 24 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


def format_vec(values, precision):
    if values is None:
        return "None"
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def normalize_vec(vec, name="vector"):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise RuntimeError(f"Cannot normalize near-zero {name}.")
    return vec / norm


def fit_plane_pca(points):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    demeaned = points - center
    _, _, vh = np.linalg.svd(demeaned, full_matrices=False)
    normal = normalize_vec(vh[-1], "plane normal")
    if normal[2] > 0:
        normal = -normal
    distances = (points - center) @ normal
    return center, normal, distances


def run_detection(folder, args):
    folder, color, depth_m, intr = load_capture(folder)
    wheel_result = None
    if args.normal_source in ("outer-wheel", "center-ring-gated", "hybrid") or args.auto_method in (
        "outer-wheel",
        "center-circle",
        "hybrid",
        "metal-ring",
    ):
        try:
            wheel_result = detect_outer_wheel_pose(folder, args)
        except Exception as exc:
            if args.normal_source in ("outer-wheel", "center-ring-gated") or args.auto_method in (
                "outer-wheel",
                "center-circle",
            ):
                raise
            print(f"[WARN] outer wheel pose unavailable: {exc}")

    if args.manual_center:
        parts = [float(x) for x in args.manual_center.split(",")]
        if len(parts) != 2:
            raise RuntimeError("--manual-center must be 'u,v'")
        detection = fit_hole_plane_from_click(
            depth_m=depth_m,
            intr=intr,
            pixel=(int(round(parts[0])), int(round(parts[1]))),
            args=args,
        )
        detection["hole_center_px"] = [parts[0], parts[1]]
        detection["hole_radius_px"] = float(args.plane_inner_radius_px)
        source_debug = {"source": "manual-center"}
    elif args.auto_method == "metal-ring":
        detection = detect_metal_ring_pose(color, depth_m, intr, args, wheel_result=wheel_result)
        source_debug = {
            "source": detection.get("source", "metal-ring"),
            "score": float(detection.get("score", 0.0)),
            "inner_radius_px": detection.get("hole_radius_px"),
            "metal_outer_radius_px": detection.get("metal_outer_radius_px"),
            "metal_pixels": detection.get("metal_pixels"),
        }
    elif args.auto_method == "center-circle":
        if wheel_result is None:
            raise RuntimeError("center-circle mode needs an outer-wheel center prior.")
        hole = detect_center_circle_near_prior(color, depth_m, wheel_result, args)
        if hole is None:
            if not args.allow_outer_center_fallback:
                raise RuntimeError("Center circle was not found near the outer-wheel center prior.")
            hole = outer_result_to_center_candidate(wheel_result, args)
        detection = fit_hole_plane_from_auto(depth_m, intr, hole, args)
        source_debug = {
            "source": hole.get("source", "unknown"),
            "score": float(hole.get("score", 0.0)),
            "outer_wheel_radius_px": hole.get("outer_wheel_radius_px"),
        }
    else:
        hole, debug = auto_detect_center_hole(color, depth_m, args)
        if hole is None:
            raise RuntimeError(f"Center-hole auto detection failed: {debug}")
        detection = fit_hole_plane_from_auto(depth_m, intr, hole, args)
        source_debug = {
            "source": hole.get("source", "unknown"),
            "score": float(hole.get("score", 0.0)),
            "outer_wheel_radius_px": hole.get("outer_wheel_radius_px"),
            "auto_debug": {
                "contours": debug.get("contours"),
                "rejected": debug.get("rejected", []),
            },
        }

    detection["local_center_cam"] = np.asarray(detection["center_cam"], dtype=np.float64).tolist()
    detection["local_normal_cam"] = np.asarray(detection["normal_cam"], dtype=np.float64).tolist()
    detection["local_plane_rmse_m"] = float(detection["plane_rmse_m"])

    if args.normal_source == "metal-ring":
        detection["normal_source"] = "metal-ring-rgbd-plane"
    elif args.normal_source == "center-ring-gated":
        if wheel_result is None:
            raise RuntimeError("center-ring-gated normal needs an outer-wheel plane.")
        gated = fit_center_ring_normal_gated_by_outer_plane(
            depth_m=depth_m,
            intr=intr,
            center_px=detection["hole_center_px"],
            hole_radius_px=float(detection.get("hole_radius_px") or args.plane_inner_radius_px),
            wheel_result=wheel_result,
            args=args,
        )
        detection["normal_cam"] = gated["normal_cam"]
        detection["center_cam"] = gated["center_cam"]
        detection["plane_center_cam"] = gated["plane_center_cam"]
        detection["plane_points"] = gated["plane_points"]
        detection["plane_inliers"] = gated["plane_inliers"]
        detection["plane_rmse_m"] = gated["plane_rmse_m"]
        detection["normal_source"] = "center-ring-gated-by-outer-plane"
        detection["center_ring_gated_result"] = gated["debug"]
    elif args.normal_source in ("outer-wheel", "hybrid"):
        try:
            if wheel_result is None:
                wheel_result = detect_outer_wheel_pose(folder, args)
            outer_normal = np.asarray(wheel_result["normal_headcam"], dtype=np.float64)
            outer_normal /= max(float(np.linalg.norm(outer_normal)), 1e-9)
            local_normal = np.asarray(detection["normal_cam"], dtype=np.float64)
            local_normal /= max(float(np.linalg.norm(local_normal)), 1e-9)
            if np.dot(outer_normal, local_normal) < 0:
                outer_normal = -outer_normal
            detection["normal_cam"] = outer_normal
            detection["normal_source"] = "outer-wheel-plane"
            detection["outer_wheel_result"] = {
                "center_headcam_m": wheel_result["center_headcam_m"],
                "normal_headcam": outer_normal.tolist(),
                "radius_m": wheel_result["radius_m"],
                "circle_center_px": wheel_result["debug"]["circle_center_px"],
                "circle_radius_px": wheel_result["debug"]["circle_radius_px"],
                "plane_points": wheel_result["debug"]["plane_points"],
            }
        except Exception as exc:
            if args.normal_source == "outer-wheel":
                raise
            detection["normal_source"] = f"local-plane-fallback: {exc}"
    else:
        detection["normal_source"] = "local-center-annulus"

    output = {
        "folder": str(folder),
        "script_version": SCRIPT_VERSION,
        "method": "auto-center-hole-local-plane-v1",
        "params": vars(args),
        "source": source_debug,
        "normal_source": detection.get("normal_source", "local-center-annulus"),
        "center_px": detection.get("hole_center_px"),
        "hole_radius_px": detection.get("hole_radius_px"),
        "center_headcam_m": np.asarray(detection["center_cam"], dtype=np.float64).tolist(),
        "normal_headcam": np.asarray(detection["normal_cam"], dtype=np.float64).tolist(),
        "local_center_headcam_m": detection["local_center_cam"],
        "local_normal_headcam": detection["local_normal_cam"],
        "local_plane_rmse_m": detection["local_plane_rmse_m"],
        "outer_wheel_result": detection.get("outer_wheel_result"),
        "center_ring_gated_result": detection.get("center_ring_gated_result"),
        "metal_ring_result": detection.get("metal_ring_result"),
        "quality": detection.get("quality", "unknown"),
        "quality_reasons": detection.get("quality_reasons", []),
        "plane_center_headcam_m": np.asarray(detection["plane_center_cam"], dtype=np.float64).tolist(),
        "plane_rmse_m": float(detection["plane_rmse_m"]),
        "plane_points": int(detection["plane_points"]),
        "plane_inliers": int(detection["plane_inliers"]),
    }

    overlay = draw_detection_overlay(color, depth_m, detection, intr, args, source_debug)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_stem if args.out_stem else f"center_hole_{folder.name}"
    json_path = out_dir / f"{stem}_detection.json"
    overlay_path = out_dir / f"{stem}_overlay.png"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    cv2.imwrite(str(overlay_path), overlay)
    return output, json_path, overlay_path


def detect_outer_wheel_pose(folder, args):
    wheel_args = argparse.Namespace(
        min_depth=args.outer_min_depth,
        max_depth=args.outer_max_depth,
        dark_threshold=args.outer_dark_threshold,
        plane_dark_threshold=args.outer_plane_dark_threshold,
        min_area=args.outer_min_area,
        min_radius_px=args.outer_min_radius_px,
        max_radius_px=args.outer_max_radius_px,
        circle_inlier_px=args.outer_circle_inlier_px,
        plane_band_px=args.outer_plane_band_px,
        ransac_iters=args.outer_ransac_iters,
        ransac_seed=args.outer_ransac_seed,
        center_penalty=args.outer_center_penalty,
        roi_x0=args.outer_roi_x0,
        roi_x1=args.outer_roi_x1,
        roi_y0=args.outer_roi_y0,
        roi_y1=args.outer_roi_y1,
        flip_x=False,
        flip_y=False,
    )
    return wheel_rgbd.detect_wheel(folder, wheel_args)


def outer_result_to_center_candidate(wheel_result, args):
    center = np.asarray(wheel_result["debug"]["circle_center_px"], dtype=np.float64)
    return {
        "center_px": center,
        "radius_px": float(args.plane_inner_radius_px),
        "outer_wheel_radius_px": float(wheel_result["debug"]["circle_radius_px"]),
        "score": 0.0,
        "source": "outer-wheel-center-fallback",
    }


def detect_center_circle_near_prior(color, depth_m, wheel_result, args):
    prior = np.asarray(wheel_result["debug"]["circle_center_px"], dtype=np.float64)
    outer_radius = float(wheel_result["debug"]["circle_radius_px"])
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
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
    if circles is None:
        return None

    best = None
    for x, y, radius in np.round(circles[0, :]).astype(np.float64):
        center = np.array([x, y], dtype=np.float64)
        center_error = float(np.linalg.norm(center - prior))
        if center_error > float(args.center_circle_prior_max_px):
            continue

        h, w = depth_m.shape
        yy, xx = np.indices((h, w))
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        disk = dist <= max(radius * 0.85, 2.0)
        if int(disk.sum()) < 8:
            continue
        dark_ratio = float((gray[disk] <= int(args.hole_dark_threshold)).sum()) / float(disk.sum())
        invalid_ratio = float(
            ((depth_m[disk] < float(args.min_depth)) | (depth_m[disk] > float(args.max_depth))).sum()
        ) / float(disk.sum())
        if max(dark_ratio, invalid_ratio) < float(args.min_hole_inside_ratio):
            continue

        score = (
            3.0 * max(dark_ratio, invalid_ratio)
            + 1.0 * (1.0 - min(1.0, center_error / max(float(args.center_circle_prior_max_px), 1.0)))
            - 0.005 * abs(radius - args.expected_hole_radius_px)
        )
        candidate = {
            "center_px": center,
            "radius_px": float(radius),
            "outer_wheel_radius_px": outer_radius,
            "score": float(score),
            "source": "center-circle-hough-prior",
            "center_error_px": center_error,
            "inside_dark_ratio": dark_ratio,
            "inside_invalid_ratio": invalid_ratio,
        }
        if best is None or score > best["score"]:
            best = candidate
    return best


def detect_metal_ring_pose_from_rgbd_plane(color, depth_m, intr, args, wheel_result=None):
    h, w = depth_m.shape
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    if wheel_result is not None and args.metal_prior_source == "outer-wheel":
        prior = np.asarray(wheel_result["debug"]["circle_center_px"], dtype=np.float64)
        prior_source = "outer-wheel-center"
    else:
        prior = np.array([w * float(args.metal_prior_x), h * float(args.metal_prior_y)], dtype=np.float64)
        prior_source = "image-ratio-prior"

    yy, xx = np.indices((h, w))
    roi = ((xx - prior[0]) ** 2 + (yy - prior[1]) ** 2) <= float(args.metal_search_radius_px) ** 2
    valid_depth = (depth_m >= float(args.min_depth)) & (depth_m <= float(args.max_depth))
    metal_color = (
        roi
        & valid_depth
        & (saturation <= int(args.rgbd_plane_max_saturation))
        & (value >= int(args.rgbd_plane_min_value))
        & (gray >= int(args.rgbd_plane_min_gray))
    )
    metal_color_u8 = (metal_color.astype(np.uint8) * 255)
    metal_color_u8 = cv2.morphologyEx(metal_color_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    metal_color_u8 = cv2.morphologyEx(metal_color_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    metal_color = metal_color_u8 > 0

    plane_seed = metal_color & valid_depth
    seed_source = "metal-color"
    if int(plane_seed.sum()) < int(args.rgbd_plane_min_seed_points):
        plane_seed = roi & valid_depth
        seed_source = "roi-depth-fallback"

    seed_ys, seed_xs = np.nonzero(plane_seed)
    if len(seed_xs) < int(args.min_plane_points):
        raise RuntimeError(f"RGBD plane detector has too few seed points: {len(seed_xs)}")
    seed_points = backproject_pixels(seed_xs, seed_ys, depth_m[seed_ys, seed_xs].astype(np.float64), intr)
    plane_center, normal_cam, plane_inlier_points, plane_rmse, seed_inlier_idx = ransac_plane(
        seed_points,
        iterations=int(args.rgbd_plane_ransac_iters),
        threshold_m=float(args.rgbd_plane_inlier_m),
        seed=int(args.rgbd_plane_ransac_seed),
    )

    all_ys, all_xs = np.nonzero(roi & valid_depth)
    all_points = backproject_pixels(all_xs, all_ys, depth_m[all_ys, all_xs].astype(np.float64), intr)
    all_dist = np.abs((all_points - plane_center) @ normal_cam)
    all_plane_keep = all_dist <= float(args.rgbd_plane_inlier_m)
    plane_mask = np.zeros((h, w), dtype=bool)
    plane_mask[all_ys[all_plane_keep], all_xs[all_plane_keep]] = True
    metal_plane_mask = plane_mask & (metal_color | (seed_source == "roi-depth-fallback"))
    metal_plane_u8 = (metal_plane_mask.astype(np.uint8) * 255)
    metal_plane_u8 = cv2.morphologyEx(metal_plane_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    metal_plane_u8 = cv2.morphologyEx(metal_plane_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    metal_plane_mask = metal_plane_u8 > 0
    mp_ys, mp_xs = np.nonzero(metal_plane_mask)
    metal_centroid_px = np.array([
        float(mp_xs.mean()) if len(mp_xs) else float(prior[0]),
        float(mp_ys.mean()) if len(mp_ys) else float(prior[1]),
    ], dtype=np.float64)

    candidates = []
    if args.rectified_plane_hole:
        rectified_candidate = detect_hole_in_rectified_plane(
            color=color,
            depth_m=depth_m,
            intr=intr,
            plane_point=plane_center,
            normal=normal_cam,
            metal_plane_mask=metal_plane_mask,
            metal_centroid_px=metal_centroid_px,
            args=args,
        )
        if rectified_candidate is not None:
            candidates.append(rectified_candidate)
    candidates.extend(detect_rgbd_void_candidates(metal_plane_u8, prior, args))
    candidates.extend(detect_inner_ellipse_from_edges(gray, prior, args))
    if not args.disable_hough:
        candidates.extend(detect_hough_circle_candidates(gray, prior, args))

    scored = []
    for candidate in candidates:
        scored_candidate = score_rgbd_plane_hole_candidate(
            candidate=candidate,
            depth_m=depth_m,
            plane_mask=plane_mask,
            metal_plane_mask=metal_plane_mask,
            metal_centroid_px=metal_centroid_px,
            prior=prior,
            args=args,
        )
        if scored_candidate is not None:
            scored.append(scored_candidate)

    if args.rgbd_grid_fallback and not scored:
        scored.extend(grid_search_rgbd_plane_hole(
            depth_m=depth_m,
            plane_mask=plane_mask,
            metal_plane_mask=metal_plane_mask,
            metal_centroid_px=metal_centroid_px,
            prior=prior,
            args=args,
        ))
    scored.extend(local_refine_rgbd_candidates(
        candidates=scored,
        depth_m=depth_m,
        plane_mask=plane_mask,
        metal_plane_mask=metal_plane_mask,
        metal_centroid_px=metal_centroid_px,
        prior=prior,
        args=args,
    ))

    if not scored:
        raise RuntimeError("RGBD plane-hole detector found no valid center-hole candidate.")

    scored.sort(key=lambda item: item["score"], reverse=True)
    hole = scored[0]
    refined_hole = refine_hole_with_radial_edges(gray, hole, args)
    if refined_hole is not None:
        hole = refined_hole
    depth_refined_hole = refine_hole_with_depth_void(depth_m, hole, args)
    if depth_refined_hole is not None:
        hole = depth_refined_hole
    center = np.asarray(hole["center_px"], dtype=np.float64)
    inner_radius = float(hole["radius_px"])

    fit_inner = max(inner_radius * float(args.metal_fit_inner_scale), inner_radius + 2.0)
    fit_outer = max(inner_radius * float(args.metal_outer_radius_scale), fit_inner + 8.0)
    radius_px = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    fit_mask = metal_plane_mask & (radius_px >= fit_inner) & (radius_px <= fit_outer) & valid_depth
    fit_ys, fit_xs = np.nonzero(fit_mask)
    if len(fit_xs) >= int(args.min_plane_points):
        fit_points = backproject_pixels(fit_xs, fit_ys, depth_m[fit_ys, fit_xs].astype(np.float64), intr)
        fit_center, fit_normal, fit_inliers, fit_rmse = robust_plane(
            fit_points,
            iterations=args.plane_iters,
            threshold_m=args.plane_inlier_m,
        )
        if np.dot(fit_normal, normal_cam) < 0:
            fit_normal = -fit_normal
        plane_center = fit_center
        normal_cam = fit_normal
        plane_inlier_points = fit_inliers
        plane_rmse = fit_rmse
        fit_source = "local-metal-ring-annulus"
    else:
        fit_source = "global-metal-plane"

    center_cam = intersect_pixel_ray_with_plane(center[0], center[1], plane_center, normal_cam, intr)
    quality_prior = np.array([
        center[0],
        metal_centroid_px[1],
    ], dtype=np.float64)
    quality, quality_reasons = assess_detection_quality(
        center=center,
        radius=inner_radius,
        prior=quality_prior,
        rmse=plane_rmse,
        points=int(len(plane_inlier_points)),
        inliers=int(len(plane_inlier_points)),
        hole=hole,
        args=args,
    )
    if hole.get("sector_coverage", 0.0) < float(args.rgbd_min_sector_coverage):
        quality = "reject"
        quality_reasons.append(
            f"sector_coverage={hole.get('sector_coverage', 0.0):.2f}<{args.rgbd_min_sector_coverage:.2f}"
        )

    return {
        "source": hole["source"],
        "score": float(hole.get("score", 0.0)),
        "center_cam": center_cam,
        "normal_cam": normal_cam,
        "plane_center_cam": plane_center,
        "plane_points": int(len(seed_points)),
        "plane_inliers": int(len(plane_inlier_points)),
        "plane_rmse_m": float(plane_rmse),
        "hole_center_px": [float(center[0]), float(center[1])],
        "hole_radius_px": float(inner_radius),
        "metal_outer_radius_px": float(fit_outer),
        "metal_pixels": int(metal_plane_mask.sum()),
        "metal_ring_result": {
            "detector": "rgbd-plane-hole",
            "prior_px": prior.tolist(),
            "prior_source": prior_source,
            "seed_source": seed_source,
            "seed_points": int(len(seed_points)),
            "seed_plane_inliers": int(len(seed_inlier_idx)),
            "plane_pixels": int(plane_mask.sum()),
            "metal_plane_pixels": int(metal_plane_mask.sum()),
            "metal_centroid_px": metal_centroid_px.tolist(),
            "candidate_count": int(len(candidates)),
            "valid_candidate_count": int(len(scored)),
            "fit_source": fit_source,
            "fit_inner_radius_px": float(fit_inner),
            "fit_outer_radius_px": float(fit_outer),
            "center_error_px": float(np.linalg.norm(center - prior)),
            "quality_prior_px": quality_prior.tolist(),
            "hole_candidate": {
                key: value
                for key, value in hole.items()
                if key not in ("center_px",)
            },
        },
        "quality": quality,
        "quality_reasons": quality_reasons,
    }


def ransac_plane(points, iterations, threshold_m, seed):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 80:
        raise RuntimeError(f"Not enough points for RANSAC plane fit: {len(points)}")
    rng = np.random.default_rng(int(seed))
    sample_count = min(len(points), 6000)
    if len(points) > sample_count:
        sample_idx = rng.choice(len(points), size=sample_count, replace=False)
        sample = points[sample_idx]
    else:
        sample_idx = np.arange(len(points))
        sample = points

    best_keep = None
    best_count = -1
    for _ in range(max(1, int(iterations))):
        ids = rng.choice(len(sample), size=3, replace=False)
        p0, p1, p2 = sample[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        distances = np.abs((sample - p0) @ normal)
        keep = distances <= float(threshold_m)
        count = int(keep.sum())
        if count > best_count:
            best_count = count
            best_keep = keep

    if best_keep is None or best_count < 80:
        return robust_plane(points, iterations=5, threshold_m=threshold_m) + (np.arange(len(points)),)

    inlier_sample_points = sample[best_keep]
    center, normal, _distances = fit_plane_pca(inlier_sample_points)
    all_distances = np.abs((points - center) @ normal)
    all_keep = all_distances <= float(threshold_m)
    if int(all_keep.sum()) < 80:
        all_keep = np.zeros(len(points), dtype=bool)
        all_keep[sample_idx[best_keep]] = True
    center, normal, inliers, rmse = robust_plane(
        points[all_keep],
        iterations=5,
        threshold_m=threshold_m,
    )
    if normal[2] > 0:
        normal = -normal
    return center, normal, inliers, rmse, np.nonzero(all_keep)[0]


def detect_rgbd_void_candidates(metal_plane_u8, prior, args):
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(metal_plane_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    search = np.zeros_like(closed)
    cv2.circle(
        search,
        (int(round(prior[0])), int(round(prior[1]))),
        int(round(float(args.metal_prior_max_px))),
        255,
        -1,
    )
    void_u8 = cv2.bitwise_and(cv2.bitwise_not(closed), search)
    contours, _ = cv2.findContours(void_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(args.min_hole_area_px):
            continue
        if len(contour) >= 5:
            (cx, cy), (axis_a, axis_b), _angle = cv2.fitEllipse(contour)
            major = max(float(axis_a), float(axis_b))
            minor = min(float(axis_a), float(axis_b))
            radius = 0.25 * (major + minor)
            axis_ratio = minor / max(major, 1e-6)
        else:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            axis_ratio = 1.0
        center = np.array([float(cx), float(cy)], dtype=np.float64)
        center_error = float(np.linalg.norm(center - prior))
        if center_error > float(args.metal_prior_max_px):
            continue
        if not (float(args.min_hole_radius_px) <= radius <= float(args.max_hole_radius_px)):
            continue
        candidates.append({
            "center_px": center,
            "radius_px": float(radius),
            "source": "rgbd-plane-void",
            "score": 0.0,
            "center_error_px": center_error,
            "axis_ratio": float(axis_ratio),
            "area": area,
        })
    return candidates


def detect_hough_circle_candidates(gray, prior, args):
    enhanced = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(enhanced, (7, 7), 1.5)
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
    if circles is None:
        return []
    candidates = []
    for x, y, radius in np.round(circles[0, :]).astype(np.float64):
        center = np.array([x, y], dtype=np.float64)
        center_error = float(np.linalg.norm(center - prior))
        if center_error > float(args.metal_prior_max_px):
            continue
        candidates.append({
            "center_px": center,
            "radius_px": float(radius),
            "source": "hough-rgbd-scored",
            "score": 0.0,
            "center_error_px": center_error,
        })
    return candidates


def detect_hole_in_rectified_plane(color, depth_m, intr, plane_point, normal, metal_plane_mask, metal_centroid_px, args):
    mp_ys, mp_xs = np.nonzero(metal_plane_mask)
    if len(mp_xs) < int(args.min_plane_points):
        return None

    normal = normalize_vec(normal, "rectified plane normal")
    x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_hint - np.dot(x_hint, normal) * normal
    if np.linalg.norm(x_axis) < 1e-6:
        x_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = x_hint - np.dot(x_hint, normal) * normal
    x_axis = normalize_vec(x_axis, "rectified x axis")
    y_axis = normalize_vec(np.cross(normal, x_axis), "rectified y axis")

    metal_depth = depth_m[mp_ys, mp_xs].astype(np.float64)
    metal_points = backproject_pixels(mp_xs, mp_ys, metal_depth, intr)
    metal_uv = np.column_stack([
        (metal_points - plane_point) @ x_axis,
        (metal_points - plane_point) @ y_axis,
    ])
    centroid_depth = float(depth_m[int(round(metal_centroid_px[1])), int(round(metal_centroid_px[0]))])
    if not (float(args.min_depth) <= centroid_depth <= float(args.max_depth)):
        centroid_3d = metal_points.mean(axis=0)
    else:
        centroid_3d = backproject_pixels(
            np.array([metal_centroid_px[0]], dtype=np.float64),
            np.array([metal_centroid_px[1]], dtype=np.float64),
            np.array([centroid_depth], dtype=np.float64),
            intr,
        )[0]
    centroid_uv = np.array([
        float((centroid_3d - plane_point) @ x_axis),
        float((centroid_3d - plane_point) @ y_axis),
    ], dtype=np.float64)

    margin = float(args.rectified_margin_m)
    min_u = float(np.percentile(metal_uv[:, 0], 1.0) - margin)
    max_u = float(np.percentile(metal_uv[:, 0], 99.0) + margin)
    min_v = float(np.percentile(metal_uv[:, 1], 1.0) - margin)
    max_v = float(np.percentile(metal_uv[:, 1], 99.0) + margin)
    resolution = float(args.rectified_resolution_m)
    rect_w = int(np.clip(math.ceil((max_u - min_u) / resolution), 80, int(args.rectified_max_size_px)))
    rect_h = int(np.clip(math.ceil((max_v - min_v) / resolution), 80, int(args.rectified_max_size_px)))
    if rect_w <= 20 or rect_h <= 20:
        return None

    grid_x, grid_y = np.meshgrid(np.arange(rect_w, dtype=np.float64), np.arange(rect_h, dtype=np.float64))
    plane_u = min_u + grid_x * resolution
    plane_v = min_v + grid_y * resolution
    points = (
        plane_point.reshape(1, 1, 3)
        + plane_u[:, :, None] * x_axis.reshape(1, 1, 3)
        + plane_v[:, :, None] * y_axis.reshape(1, 1, 3)
    )
    z = points[:, :, 2]
    map_x = (points[:, :, 0] * intr_value(intr, "fx") / z + intr_value(intr, "ppx")).astype(np.float32)
    map_y = (points[:, :, 1] * intr_value(intr, "fy") / z + intr_value(intr, "ppy")).astype(np.float32)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    rect_gray = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    rect_depth = cv2.remap(depth_m.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rect_metal = cv2.remap(
        (metal_plane_mask.astype(np.uint8) * 255),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rect_valid = (rect_depth >= float(args.min_depth)) & (rect_depth <= float(args.max_depth))

    centroid_rx = int(round((centroid_uv[0] - min_u) / resolution))
    centroid_ry = int(round((centroid_uv[1] - min_v) / resolution))
    yy, xx = np.indices((rect_h, rect_w))
    search_r = float(args.rectified_search_radius_m) / resolution
    search = ((xx - centroid_rx) ** 2 + (yy - centroid_ry) ** 2) <= search_r * search_r
    metal_u8 = cv2.morphologyEx(rect_metal, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    metal_u8 = cv2.morphologyEx(metal_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    outer_contours, _ = cv2.findContours(metal_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not outer_contours:
        return None
    component = np.zeros_like(metal_u8)
    outer_contours = sorted(outer_contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(component, outer_contours[: max(1, int(args.rectified_fill_components))], -1, 255, -1)
    void_u8 = cv2.bitwise_and(component, cv2.bitwise_not(metal_u8))
    void_u8 = cv2.bitwise_and(void_u8, (search.astype(np.uint8) * 255))
    void_u8 = cv2.morphologyEx(void_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    void_u8 = cv2.morphologyEx(void_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(void_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for contour in contours:
        area_px = float(cv2.contourArea(contour))
        if area_px < float(args.rectified_min_area_px):
            continue
        if len(contour) >= 5:
            (cx, cy), (axis_a, axis_b), _angle = cv2.fitEllipse(contour)
            major = max(float(axis_a), float(axis_b))
            minor = min(float(axis_a), float(axis_b))
            axis_ratio = minor / max(major, 1e-6)
            radius_px = 0.25 * (major + minor)
        else:
            (cx, cy), radius_px = cv2.minEnclosingCircle(contour)
            axis_ratio = 1.0
        radius_m = float(radius_px) * resolution
        if not (float(args.rectified_min_radius_m) <= radius_m <= float(args.rectified_max_radius_m)):
            continue
        if axis_ratio < float(args.rectified_min_axis_ratio):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * math.pi * area_px / (perimeter * perimeter) if perimeter > 1e-6 else 0.0
        if circularity < float(args.rectified_min_circularity):
            continue
        center_uv = np.array([min_u + float(cx) * resolution, min_v + float(cy) * resolution], dtype=np.float64)
        center_error_m = float(np.linalg.norm(center_uv - centroid_uv))
        if center_error_m > float(args.rectified_search_radius_m):
            continue
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        if not (0 <= cx_i < rect_w and 0 <= cy_i < rect_h and search[cy_i, cx_i]):
            continue
        radius_for_probe = max(2, int(round(radius_px * 0.75)))
        probe = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius_for_probe * radius_for_probe
        dark_or_void_ratio = float(((rect_gray <= int(args.rectified_dark_threshold)) | (~rect_valid))[probe].sum()) / max(float(probe.sum()), 1.0)
        if dark_or_void_ratio < float(args.rectified_min_dark_or_void_ratio):
            continue
        score = (
            3.0 * axis_ratio
            + 2.0 * circularity
            + min(2.0, area_px / max(float(args.rectified_good_area_px), 1.0))
            + 2.0 * dark_or_void_ratio
            - 8.0 * center_error_m
        )
        candidate = {
            "rect_center_px": np.array([float(cx), float(cy)], dtype=np.float64),
            "rect_radius_px": float(radius_px),
            "center_uv_m": center_uv,
            "radius_m": radius_m,
            "score": float(score),
            "axis_ratio": float(axis_ratio),
            "circularity": float(circularity),
            "dark_or_void_ratio": float(dark_or_void_ratio),
            "area": area_px,
            "center_error_m": center_error_m,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        return None

    center_3d = plane_point + best["center_uv_m"][0] * x_axis + best["center_uv_m"][1] * y_axis
    center_px = np.array([
        center_3d[0] * intr_value(intr, "fx") / center_3d[2] + intr_value(intr, "ppx"),
        center_3d[1] * intr_value(intr, "fy") / center_3d[2] + intr_value(intr, "ppy"),
    ], dtype=np.float64)
    radius_px_image = best["radius_m"] * intr_value(intr, "fx") / max(float(center_3d[2]), 1e-6)
    return {
        "center_px": center_px,
        "radius_px": float(radius_px_image),
        "source": "rectified-plane-hole",
        "score": float(best["score"] + float(args.rectified_score_bonus)),
        "axis_ratio": float(best["axis_ratio"]),
        "circularity": float(best["circularity"]),
        "rectified_result": {
            "rect_size_px": [int(rect_w), int(rect_h)],
            "resolution_m": resolution,
            "rect_center_px": best["rect_center_px"].tolist(),
            "rect_radius_px": float(best["rect_radius_px"]),
            "radius_m": float(best["radius_m"]),
            "center_uv_m": best["center_uv_m"].tolist(),
            "metal_centroid_uv_m": centroid_uv.tolist(),
            "center_error_m": float(best["center_error_m"]),
            "dark_or_void_ratio": float(best["dark_or_void_ratio"]),
            "area_px": float(best["area"]),
        },
    }


def refine_hole_with_radial_edges(gray, hole, args):
    axis_ratio = float(hole.get("axis_ratio", 1.0))
    if axis_ratio >= float(args.radial_refine_max_axis_ratio):
        return None
    init = np.asarray(hole["center_px"], dtype=np.float64)
    points = []
    h, w = gray.shape[:2]
    gray_f = gray.astype(np.float32)
    for angle in np.linspace(0.0, 2.0 * math.pi, int(args.radial_refine_rays), endpoint=False):
        radii = np.arange(
            float(args.radial_refine_min_radius_px),
            float(args.radial_refine_max_radius_px),
            1.0,
            dtype=np.float64,
        )
        xs = init[0] + radii * math.cos(angle)
        ys = init[1] + radii * math.sin(angle)
        values = []
        for x, y in zip(xs, ys):
            if x < 1 or y < 1 or x >= w - 2 or y >= h - 2:
                values.append(np.nan)
            else:
                values.append(float(cv2.getRectSubPix(gray_f, (1, 1), (float(x), float(y)))[0, 0]))
        values = np.asarray(values, dtype=np.float64)
        if np.isnan(values).all():
            continue
        gradients = np.abs(np.gradient(values))
        if np.isnan(gradients).all():
            continue
        idx = int(np.nanargmax(gradients))
        if float(gradients[idx]) < float(args.radial_refine_min_gradient):
            continue
        points.append([float(xs[idx]), float(ys[idx])])

    if len(points) < int(args.radial_refine_min_points):
        return None
    points = np.asarray(points, dtype=np.float64)
    center, radius, inlier_count, rmse = robust_fit_circle_2d(
        points,
        max_residual_px=float(args.radial_refine_max_residual_px),
        iterations=int(args.radial_refine_fit_iters),
    )
    shift = float(np.linalg.norm(center - init))
    if not (float(args.min_hole_radius_px) <= radius <= float(args.max_hole_radius_px)):
        return None
    if shift > float(args.radial_refine_max_shift_px):
        return None
    if inlier_count < int(args.radial_refine_min_inliers):
        return None
    if rmse > float(args.radial_refine_max_rmse_px):
        return None
    refined = dict(hole)
    refined["center_px"] = center
    refined["radius_px"] = float(radius)
    refined["source"] = f"{hole.get('source', 'hole')}-radial-edge-refine"
    refined["radial_refine"] = {
        "initial_center_px": init.tolist(),
        "shift_px": shift,
        "points": int(len(points)),
        "inliers": int(inlier_count),
        "rmse_px": float(rmse),
    }
    return refined


def robust_fit_circle_2d(points, max_residual_px, iterations):
    points = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    center = np.array([float(points[:, 0].mean()), float(points[:, 1].mean())], dtype=np.float64)
    radius = 0.0
    residuals = np.zeros(len(points), dtype=np.float64)
    for _ in range(max(1, int(iterations))):
        selected = points[keep]
        if len(selected) < 3:
            break
        x = selected[:, 0]
        y = selected[:, 1]
        a = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(selected))])
        b = x * x + y * y
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy, c = [float(v) for v in sol]
        radius = math.sqrt(max(0.0, c + cx * cx + cy * cy))
        center = np.array([cx, cy], dtype=np.float64)
        residuals = np.sqrt(((points - center) ** 2).sum(axis=1)) - radius
        new_keep = np.abs(residuals) <= float(max_residual_px)
        if int(new_keep.sum()) == int(keep.sum()):
            keep = new_keep
            break
        keep = new_keep
    rmse = float(np.sqrt(np.mean(residuals[keep] * residuals[keep]))) if int(keep.sum()) else float("inf")
    return center, float(radius), int(keep.sum()), rmse


def refine_hole_with_depth_void(depth_m, hole, args):
    center = np.asarray(hole["center_px"], dtype=np.float64)
    radius = float(hole["radius_px"])
    h, w = depth_m.shape[:2]
    yy, xx = np.indices((h, w))
    roi_radius = max(radius * float(args.depth_void_roi_scale), float(args.depth_void_min_roi_radius_px))
    roi = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= roi_radius * roi_radius
    valid = (depth_m >= float(args.min_depth)) & (depth_m <= float(args.max_depth))
    void = roi & (~valid)
    void_u8 = (void.astype(np.uint8) * 255)
    void_u8 = cv2.morphologyEx(void_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    void_u8 = cv2.morphologyEx(void_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(void_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(args.depth_void_min_area_px):
            continue
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) < 1e-6:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        shift = float(np.linalg.norm(np.array([cx, cy], dtype=np.float64) - center))
        if shift > float(args.depth_void_max_shift_px):
            continue
        (_x, _y), enclosing_radius = cv2.minEnclosingCircle(contour)
        if not (float(args.depth_void_min_radius_px) <= enclosing_radius <= float(args.depth_void_max_radius_px)):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 1e-6 else 0.0
        if circularity < float(args.depth_void_min_circularity):
            continue
        score = area + 200.0 * circularity - 4.0 * shift
        candidate = {
            "center": np.array([cx, cy], dtype=np.float64),
            "radius": float(enclosing_radius),
            "area": area,
            "circularity": float(circularity),
            "shift": shift,
            "score": float(score),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        return None
    refined = dict(hole)
    refined["center_px"] = best["center"]
    refined["radius_px"] = float(max(radius, best["radius"]))
    refined["source"] = f"{hole.get('source', 'hole')}-depth-void-refine"
    refined["depth_void_refine"] = {
        "initial_center_px": center.tolist(),
        "shift_px": best["shift"],
        "void_radius_px": best["radius"],
        "void_area_px": best["area"],
        "void_circularity": best["circularity"],
    }
    return refined


def score_rgbd_plane_hole_candidate(candidate, depth_m, plane_mask, metal_plane_mask, metal_centroid_px, prior, args):
    center = np.asarray(candidate["center_px"], dtype=np.float64)
    radius = float(candidate["radius_px"])
    if not (float(args.min_hole_radius_px) <= radius <= float(args.max_hole_radius_px)):
        return None
    if radius < float(args.rgbd_min_candidate_radius_px):
        return None
    h, w = depth_m.shape
    if center[0] < 0 or center[0] >= w or center[1] < 0 or center[1] >= h:
        return None
    centroid = np.asarray(metal_centroid_px, dtype=np.float64)
    min_center_y = float(centroid[1]) - float(args.rgbd_min_center_below_metal_centroid_px)
    if center[1] < min_center_y:
        return None
    centroid_dx = abs(float(center[0] - centroid[0]))
    if centroid_dx > float(args.rgbd_max_center_x_from_metal_centroid_px):
        return None
    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    inner = dist <= max(radius * float(args.rgbd_inner_scale), 2.0)
    edge_band = (dist >= radius * float(args.rgbd_edge_inner_scale)) & (dist <= radius * float(args.rgbd_edge_outer_scale))
    outer = (dist >= radius * float(args.rgbd_outer_inner_scale)) & (dist <= radius * float(args.rgbd_outer_outer_scale))
    if int(inner.sum()) < 20 or int(edge_band.sum()) < 20 or int(outer.sum()) < 20:
        return None

    inner_plane_ratio = float(metal_plane_mask[inner].sum()) / float(inner.sum())
    edge_plane_ratio = float(metal_plane_mask[edge_band].sum()) / float(edge_band.sum())
    outer_plane_ratio = float(metal_plane_mask[outer].sum()) / float(outer.sum())
    if inner_plane_ratio > float(args.rgbd_max_inner_plane_ratio):
        return None
    if outer_plane_ratio < float(args.rgbd_min_outer_plane_ratio):
        return None

    angles = np.arctan2(yy[outer] - center[1], xx[outer] - center[0])
    sector_ids = np.floor((angles + math.pi) / (2.0 * math.pi) * int(args.rgbd_sector_count)).astype(np.int32)
    sector_ids = np.clip(sector_ids, 0, int(args.rgbd_sector_count) - 1)
    metal_outer_flat = metal_plane_mask[outer]
    covered = set(int(idx) for idx in sector_ids[metal_outer_flat])
    sector_coverage = len(covered) / max(float(args.rgbd_sector_count), 1.0)
    if sector_coverage < float(args.rgbd_min_sector_coverage) * 0.8:
        return None

    center_error = float(np.linalg.norm(center - prior))
    prior_term = 1.0 - min(1.0, center_error / max(float(args.metal_prior_max_px), 1.0))
    radius_term = 1.0 - min(1.0, abs(radius - float(args.expected_hole_radius_px)) / max(float(args.expected_hole_radius_px), 1.0))
    axis_ratio = float(candidate.get("axis_ratio", 1.0))
    score = (
        4.0 * outer_plane_ratio
        + 2.0 * edge_plane_ratio
        + 2.0 * sector_coverage
        + 1.0 * prior_term
        + 0.8 * radius_term
        + 0.4 * axis_ratio
        - 4.0 * inner_plane_ratio
        - 0.8 * min(1.0, centroid_dx / max(float(args.rgbd_max_center_x_from_metal_centroid_px), 1.0))
    )
    result = dict(candidate)
    result["score"] = float(score)
    result["inner_plane_ratio"] = float(inner_plane_ratio)
    result["edge_plane_ratio"] = float(edge_plane_ratio)
    result["outer_plane_ratio"] = float(outer_plane_ratio)
    result["sector_coverage"] = float(sector_coverage)
    result["center_error_px"] = center_error
    result["metal_centroid_dx_px"] = float(centroid_dx)
    result["metal_centroid_y_px"] = float(centroid[1])
    result["radius_error_px"] = float(abs(radius - float(args.expected_hole_radius_px)))
    return result


def local_refine_rgbd_candidates(candidates, depth_m, plane_mask, metal_plane_mask, metal_centroid_px, prior, args):
    refined = []
    if not candidates:
        return refined
    offsets = [-float(args.rgbd_refine_offset_px), 0.0, float(args.rgbd_refine_offset_px)]
    radius_offsets = [
        -float(args.rgbd_refine_radius_px),
        0.0,
        float(args.rgbd_refine_radius_px),
        2.0 * float(args.rgbd_refine_radius_px),
    ]
    seen = set()
    for base in sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[: int(args.rgbd_refine_topk)]:
        base_center = np.asarray(base["center_px"], dtype=np.float64)
        base_radius = float(base["radius_px"])
        for dy in offsets:
            for dx in offsets:
                for dr in radius_offsets:
                    center = base_center + np.array([dx, dy], dtype=np.float64)
                    radius = base_radius + dr
                    key = (int(round(center[0])), int(round(center[1])), int(round(radius)))
                    if key in seen:
                        continue
                    seen.add(key)
                    item = score_rgbd_plane_hole_candidate(
                        candidate={
                            "center_px": center,
                            "radius_px": float(radius),
                            "source": f"{base.get('source', 'candidate')}-local-refine",
                            "axis_ratio": float(base.get("axis_ratio", 1.0)),
                        },
                        depth_m=depth_m,
                        plane_mask=plane_mask,
                        metal_plane_mask=metal_plane_mask,
                        metal_centroid_px=metal_centroid_px,
                        prior=prior,
                        args=args,
                    )
                    if item is not None:
                        refined.append(item)
    return refined


def grid_search_rgbd_plane_hole(depth_m, plane_mask, metal_plane_mask, metal_centroid_px, prior, args):
    candidates = []
    centroid = np.asarray(metal_centroid_px, dtype=np.float64)
    step = max(4, int(args.rgbd_grid_step_px))
    radii = np.arange(
        max(float(args.min_hole_radius_px), float(args.rgbd_min_candidate_radius_px)),
        float(args.max_hole_radius_px) + 1.0,
        float(args.rgbd_grid_radius_step_px),
    )
    x0 = int(max(0, centroid[0] - float(args.rgbd_max_center_x_from_metal_centroid_px)))
    x1 = int(min(depth_m.shape[1] - 1, centroid[0] + float(args.rgbd_max_center_x_from_metal_centroid_px)))
    y0 = int(max(0, centroid[1] - float(args.rgbd_min_center_below_metal_centroid_px)))
    y1 = int(min(depth_m.shape[0] - 1, centroid[1] + float(args.rgbd_max_center_below_metal_centroid_px)))
    for y in range(y0, y1 + 1, step):
        for x in range(x0, x1 + 1, step):
            if np.linalg.norm(np.array([x, y], dtype=np.float64) - prior) > float(args.metal_prior_max_px):
                continue
            for radius in radii:
                item = score_rgbd_plane_hole_candidate(
                    candidate={
                        "center_px": np.array([float(x), float(y)], dtype=np.float64),
                        "radius_px": float(radius),
                        "source": "rgbd-grid-search",
                    },
                    depth_m=depth_m,
                    plane_mask=plane_mask,
                    metal_plane_mask=metal_plane_mask,
                    metal_centroid_px=metal_centroid_px,
                    prior=prior,
                    args=args,
                )
                if item is not None:
                    candidates.append(item)
    return candidates


def detect_metal_ring_pose(color, depth_m, intr, args, wheel_result=None):
    if args.rgbd_plane_hole:
        return detect_metal_ring_pose_from_rgbd_plane(color, depth_m, intr, args, wheel_result=wheel_result)

    h, w = depth_m.shape
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    if wheel_result is not None and args.metal_prior_source == "outer-wheel":
        prior = np.asarray(wheel_result["debug"]["circle_center_px"], dtype=np.float64)
        prior_source = "outer-wheel-center"
    else:
        prior = np.array([
            w * float(args.metal_prior_x),
            h * float(args.metal_prior_y),
        ], dtype=np.float64)
        prior_source = "image-ratio-prior"

    yy, xx = np.indices((h, w))
    roi = (
        ((xx - prior[0]) ** 2 + (yy - prior[1]) ** 2)
        <= float(args.metal_search_radius_px) ** 2
    )
    metal_like = (
        roi
        & (saturation <= int(args.metal_max_saturation))
        & (value >= int(args.metal_min_value))
        & (gray >= int(args.metal_min_gray))
    )
    metal_u8 = (metal_like.astype(np.uint8) * 255)
    metal_u8 = cv2.morphologyEx(metal_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    metal_u8 = cv2.morphologyEx(metal_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    hole = detect_metal_inner_circle(color, depth_m, metal_u8, prior, args)
    if hole is None:
        raise RuntimeError(
            "Metal ring inner circle was not found. Tune --metal-search-radius-px, "
            "--min-hole-radius-px/--max-hole-radius-px, or --hough-param2."
        )

    center = np.asarray(hole["center_px"], dtype=np.float64)
    inner_radius = float(hole["radius_px"])
    outer_radius = (
        float(args.metal_outer_radius_px)
        if float(args.metal_outer_radius_px) > 0
        else inner_radius * float(args.metal_outer_radius_scale)
    )
    ring_inner = max(inner_radius * float(args.metal_fit_inner_scale), inner_radius + 2.0)
    ring_outer = max(outer_radius, ring_inner + 6.0)
    radius_px = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    valid_depth = (depth_m >= float(args.min_depth)) & (depth_m <= float(args.max_depth))
    annulus = (radius_px >= ring_inner) & (radius_px <= ring_outer) & valid_depth
    metal_annulus = annulus & (metal_u8 > 0)
    use_mask = metal_annulus
    if int(use_mask.sum()) < int(args.min_plane_points):
        # RealSense exposure can make the metal darker than the color threshold.
        # Use geometry-only annulus as fallback, but record that in the output.
        use_mask = annulus
        mask_source = "annulus-depth-fallback"
    else:
        mask_source = "metal-color-annulus"

    ys, xs = np.nonzero(use_mask)
    if len(xs) < int(args.min_plane_points):
        raise RuntimeError(f"Metal ring has too few RGBD points for plane fit: {len(xs)}")

    points = backproject_pixels(xs, ys, depth_m[ys, xs].astype(np.float64), intr)
    plane_center, normal_cam, inliers, rmse = robust_plane(
        points,
        iterations=args.plane_iters,
        threshold_m=args.plane_inlier_m,
    )
    center_cam = intersect_pixel_ray_with_plane(center[0], center[1], plane_center, normal_cam, intr)
    quality, quality_reasons = assess_detection_quality(
        center=center,
        radius=inner_radius,
        prior=prior,
        rmse=rmse,
        points=len(points),
        inliers=len(inliers),
        hole=hole,
        args=args,
    )
    return {
        "source": hole["source"],
        "score": float(hole.get("score", 0.0)),
        "center_cam": center_cam,
        "normal_cam": normal_cam,
        "plane_center_cam": plane_center,
        "plane_points": int(len(points)),
        "plane_inliers": int(len(inliers)),
        "plane_rmse_m": float(rmse),
        "hole_center_px": [float(center[0]), float(center[1])],
        "hole_radius_px": float(inner_radius),
        "metal_outer_radius_px": float(ring_outer),
        "metal_pixels": int((metal_u8 > 0).sum()),
        "metal_ring_result": {
            "prior_px": prior.tolist(),
            "prior_source": prior_source,
            "center_error_px": float(np.linalg.norm(center - prior)),
            "inner_radius_px": float(inner_radius),
            "fit_inner_radius_px": float(ring_inner),
            "fit_outer_radius_px": float(ring_outer),
            "mask_source": mask_source,
            "metal_annulus_points": int(metal_annulus.sum()),
            "annulus_depth_points": int(annulus.sum()),
            "hole_candidate": {
                key: value
                for key, value in hole.items()
                if key not in ("center_px",)
            },
        },
        "quality": quality,
        "quality_reasons": quality_reasons,
    }


def assess_detection_quality(center, radius, prior, rmse, points, inliers, hole, args):
    reasons = []
    center_error = float(np.linalg.norm(np.asarray(center) - np.asarray(prior)))
    inlier_ratio = float(inliers) / max(float(points), 1.0)
    if center_error > float(args.max_quality_center_error_px):
        reasons.append(f"center_error_px={center_error:.1f}>{args.max_quality_center_error_px:.1f}")
    if not (float(args.quality_min_radius_px) <= radius <= float(args.quality_max_radius_px)):
        reasons.append(
            f"radius_px={radius:.1f} not in [{args.quality_min_radius_px:.1f},{args.quality_max_radius_px:.1f}]"
        )
    if rmse > float(args.max_quality_rmse_m):
        reasons.append(f"rmse_mm={rmse * 1000.0:.2f}>{args.max_quality_rmse_m * 1000.0:.2f}")
    if points < int(args.min_quality_points):
        reasons.append(f"points={points}<{args.min_quality_points}")
    if inlier_ratio < float(args.min_quality_inlier_ratio):
        reasons.append(f"inlier_ratio={inlier_ratio:.2f}<{args.min_quality_inlier_ratio:.2f}")
    if float(hole.get("score", 0.0)) < float(args.min_quality_score):
        reasons.append(f"score={float(hole.get('score', 0.0)):.2f}<{args.min_quality_score:.2f}")
    return ("ok" if not reasons else "reject"), reasons


def detect_metal_inner_circle(color, depth_m, metal_u8, prior, args):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(enhanced, (7, 7), 1.5)
    candidates = []
    candidates.extend(detect_inner_ellipse_from_edges(gray, prior, args))
    candidates.extend(detect_inner_holes_from_metal_mask(metal_u8, prior, args))
    if not args.disable_hough:
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
                center = np.array([x, y], dtype=np.float64)
                center_error = float(np.linalg.norm(center - prior))
                if center_error > float(args.metal_prior_max_px):
                    continue
                score = score_metal_circle_candidate(center, float(radius), depth_m, metal_u8, prior, args)
                if score is None:
                    continue
                candidates.append({
                    "center_px": center,
                    "radius_px": float(radius),
                    "score": float(score),
                    "source": "metal-inner-circle-hough",
                    "center_error_px": center_error,
                })

    if candidates:
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return choose_stable_candidate(candidates, prior, args)

    # Fallback: use the largest non-metal hole inside the metal region. This is
    # less precise than Hough, but useful when the inner edge is partially broken.
    inv = cv2.bitwise_not(metal_u8)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        metrics = contour_circle_metrics(contour)
        if metrics is None:
            continue
        center = metrics["center_px"]
        radius = metrics["radius_px"]
        if not (args.min_hole_radius_px <= radius <= args.max_hole_radius_px):
            continue
        center_error = float(np.linalg.norm(center - prior))
        if center_error > float(args.metal_prior_max_px):
            continue
        if metrics["circularity"] < float(args.min_hole_circularity):
            continue
        score = score_metal_circle_candidate(center, radius, depth_m, metal_u8, prior, args)
        if score is None:
            continue
        candidate = {
            "center_px": center,
            "radius_px": float(radius),
            "score": float(score),
            "source": "metal-inner-circle-contour",
            "center_error_px": center_error,
        }
        if best is None or score > best["score"]:
            best = candidate
    return best


def detect_inner_ellipse_from_edges(gray, prior, args):
    h, w = gray.shape[:2]
    roi = int(args.edge_roi_radius_px)
    x0 = max(0, int(round(prior[0] - roi)))
    x1 = min(w, int(round(prior[0] + roi)))
    y0 = max(0, int(round(prior[1] - roi)))
    y1 = min(h, int(round(prior[1] + roi)))
    if x1 <= x0 + 20 or y1 <= y0 + 20:
        return []

    crop = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(crop, (5, 5), 1.0)
    edges = cv2.Canny(blur, int(args.edge_canny1), int(args.edge_canny2))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        if len(contour) < int(args.edge_min_contour_points):
            continue
        area = float(cv2.contourArea(contour))
        if area < float(args.edge_min_area_px):
            continue
        try:
            (cx, cy), (axis_a, axis_b), _angle = cv2.fitEllipse(contour)
        except Exception:
            continue
        cx += x0
        cy += y0
        major = max(float(axis_a), float(axis_b))
        minor = min(float(axis_a), float(axis_b))
        if not (float(args.edge_min_minor_axis_px) <= minor <= float(args.edge_max_minor_axis_px)):
            continue
        if not (float(args.edge_min_major_axis_px) <= major <= float(args.edge_max_major_axis_px)):
            continue
        axis_ratio = minor / max(major, 1e-6)
        if axis_ratio < float(args.edge_min_axis_ratio):
            continue
        center = np.array([float(cx), float(cy)], dtype=np.float64)
        center_error = float(np.linalg.norm(center - prior))
        if center_error > float(args.metal_prior_max_px):
            continue
        radius = 0.25 * (major + minor)
        score = (
            3.0 * axis_ratio
            + min(1.0, area / max(float(args.edge_good_area_px), 1.0))
            + 1.0 * (1.0 - min(1.0, center_error / max(float(args.metal_prior_max_px), 1.0)))
            - 0.004 * abs(radius - float(args.expected_hole_radius_px))
        )
        candidates.append({
            "center_px": center,
            "radius_px": float(radius),
            "score": float(score),
            "source": "center-hole-edge-ellipse",
            "center_error_px": center_error,
            "axis_ratio": float(axis_ratio),
            "major_axis_px": float(major),
            "minor_axis_px": float(minor),
            "area": area,
        })
    return candidates


def detect_inner_holes_from_metal_mask(metal_u8, prior, args):
    contours, hierarchy = cv2.findContours(metal_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    candidates = []
    for idx, contour in enumerate(contours):
        parent = int(hierarchy[idx][3])
        if parent < 0:
            continue
        area = float(cv2.contourArea(contour))
        if area < float(args.min_hole_area_px):
            continue
        if len(contour) >= 5:
            (cx, cy), (major, minor), angle = cv2.fitEllipse(contour)
            radius = 0.25 * (float(major) + float(minor))
            axis_ratio = min(float(major), float(minor)) / max(float(major), float(minor), 1e-6)
        else:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            axis_ratio = 1.0
        center = np.array([float(cx), float(cy)], dtype=np.float64)
        center_error = float(np.linalg.norm(center - prior))
        if center_error > float(args.metal_prior_max_px):
            continue
        if not (float(args.min_hole_radius_px) <= radius <= float(args.max_hole_radius_px)):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < float(args.min_hole_circularity) * 0.6:
            continue
        score = (
            2.0 * circularity
            + 1.0 * axis_ratio
            + 1.0 * (1.0 - min(1.0, center_error / max(float(args.metal_prior_max_px), 1.0)))
            - 0.006 * abs(radius - float(args.expected_hole_radius_px))
        )
        candidates.append({
            "center_px": center,
            "radius_px": float(radius),
            "score": float(score),
            "source": "metal-mask-inner-hole",
            "center_error_px": center_error,
            "axis_ratio": float(axis_ratio),
            "circularity": float(circularity),
            "area": area,
        })
    return candidates


def choose_stable_candidate(candidates, prior, args):
    if not candidates:
        return None
    filtered = []
    for item in candidates:
        radius = float(item["radius_px"])
        center_error = float(np.linalg.norm(np.asarray(item["center_px"]) - np.asarray(prior)))
        if radius < float(args.reject_small_side_hole_radius_px):
            continue
        if center_error > float(args.metal_prior_max_px):
            continue
        filtered.append(item)
    if not filtered:
        filtered = candidates

    expected = float(args.expected_hole_radius_px)
    for item in filtered:
        radius = float(item["radius_px"])
        center_error = float(np.linalg.norm(np.asarray(item["center_px"]) - np.asarray(prior)))
        item["selection_score"] = (
            float(item["score"])
            - float(args.radius_selection_weight) * abs(radius - expected)
            - float(args.center_selection_weight) * center_error
        )
    filtered.sort(key=lambda item: item["selection_score"], reverse=True)
    return filtered[0]


def score_metal_circle_candidate(center, radius, depth_m, metal_u8, prior, args):
    h, w = depth_m.shape
    yy, xx = np.indices((h, w))
    radius_px = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    inner_disk = radius_px <= max(radius * 0.80, 2.0)
    outer_ring = (radius_px >= radius * 1.15) & (radius_px <= radius * float(args.metal_outer_radius_scale))
    if int(inner_disk.sum()) < 8 or int(outer_ring.sum()) < 8:
        return None
    inner_metal_ratio = float((metal_u8[inner_disk] > 0).sum()) / float(inner_disk.sum())
    outer_metal_ratio = float((metal_u8[outer_ring] > 0).sum()) / float(outer_ring.sum())
    valid_depth_ratio = float(
        (
            (depth_m[outer_ring] >= float(args.min_depth))
            & (depth_m[outer_ring] <= float(args.max_depth))
        ).sum()
    ) / float(outer_ring.sum())
    if outer_metal_ratio < float(args.min_metal_ring_ratio):
        return None
    if valid_depth_ratio < float(args.min_metal_depth_ratio):
        return None
    center_error = float(np.linalg.norm(np.asarray(center) - np.asarray(prior)))
    return (
        3.0 * outer_metal_ratio
        + 1.5 * valid_depth_ratio
        + 1.0 * (1.0 - min(1.0, center_error / max(float(args.metal_prior_max_px), 1.0)))
        - 1.0 * inner_metal_ratio
        - 0.004 * abs(float(radius) - float(args.expected_hole_radius_px))
    )


def fit_center_ring_normal_gated_by_outer_plane(depth_m, intr, center_px, hole_radius_px, wheel_result, args):
    center_px = np.asarray(center_px, dtype=np.float64)
    plane_point = np.asarray(wheel_result["center_headcam_m"], dtype=np.float64)
    plane_normal = np.asarray(wheel_result["normal_headcam"], dtype=np.float64)
    plane_normal /= max(float(np.linalg.norm(plane_normal)), 1e-9)

    inner = max(float(args.plane_inner_radius_px), float(hole_radius_px) * float(args.auto_plane_inner_scale))
    outer = max(float(args.plane_outer_radius_px), float(hole_radius_px) * float(args.auto_plane_outer_scale))
    h, w = depth_m.shape
    yy, xx = np.indices((h, w))
    radius_px = np.sqrt((xx - center_px[0]) ** 2 + (yy - center_px[1]) ** 2)
    base_mask = (
        (radius_px >= inner)
        & (radius_px <= outer)
        & (depth_m >= float(args.min_depth))
        & (depth_m <= float(args.max_depth))
    )
    ys, xs = np.nonzero(base_mask)
    if len(xs) < int(args.min_plane_points):
        raise RuntimeError(f"Center ring has too few valid depth points before gating: {len(xs)}")

    points = backproject_pixels(xs, ys, depth_m[ys, xs].astype(np.float64), intr)
    distances = (points - plane_point) @ plane_normal
    keep = np.abs(distances) <= float(args.center_ring_plane_gate_m)
    gated_points = points[keep]
    if len(gated_points) < int(args.min_plane_points):
        raise RuntimeError(
            f"Center ring has too few points near outer plane: {len(gated_points)} "
            f"of {len(points)} with gate {args.center_ring_plane_gate_m * 1000.0:.1f} mm"
        )

    plane_center, normal_cam, inliers, rmse = robust_plane(
        gated_points,
        iterations=args.plane_iters,
        threshold_m=args.plane_inlier_m,
    )
    if np.dot(normal_cam, plane_normal) < 0:
        normal_cam = -normal_cam
    center_cam = intersect_pixel_ray_with_plane(
        center_px[0],
        center_px[1],
        plane_center,
        normal_cam,
        intr,
    )
    return {
        "center_cam": center_cam,
        "normal_cam": normal_cam,
        "plane_center_cam": plane_center,
        "plane_points": int(len(gated_points)),
        "plane_inliers": int(len(inliers)),
        "plane_rmse_m": float(rmse),
        "debug": {
            "inner_radius_px": float(inner),
            "outer_radius_px": float(outer),
            "points_before_gate": int(len(points)),
            "points_after_gate": int(len(gated_points)),
            "gate_m": float(args.center_ring_plane_gate_m),
            "outer_plane_normal_headcam": plane_normal.tolist(),
        },
    }


def print_result(result, json_path, overlay_path):
    print(f"[OK] saved: {json_path}")
    print(f"[OK] saved: {overlay_path}")
    print(f"source: {result['source'].get('source')}")
    print(f"center_px: {format_vec(result['center_px'], 2)}")
    print(f"center_headcam_m: {format_vec(result['center_headcam_m'], 5)}")
    print(f"normal_headcam: {format_vec(result['normal_headcam'], 5)}")
    print(f"normal_source: {result['normal_source']}")
    print(f"local_normal_headcam: {format_vec(result['local_normal_headcam'], 5)}")
    print(f"plane_rmse_mm: {result['plane_rmse_m'] * 1000.0:.3f}")
    print(f"plane_points/inliers: {result['plane_points']}/{result['plane_inliers']}")
    print(f"quality: {result.get('quality', 'unknown')}")
    for reason in result.get("quality_reasons", []):
        print(f"quality_reason: {reason}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Saved RGBD capture folder.")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR), help="Output directory.")
    parser.add_argument("--out-stem", default="", help="Output filename stem without suffix.")
    parser.add_argument("--manual-center", default="", help="Optional fallback center pixel: 'u,v'.")
    parser.add_argument("--auto-method", choices=["outer-wheel", "hole", "hybrid", "center-circle", "metal-ring"],
                        default="metal-ring")
    parser.add_argument("--normal-source", choices=["outer-wheel", "local", "hybrid", "center-ring-gated", "metal-ring"],
                        default="metal-ring",
                        help="Which normal to use for insertion direction.")
    parser.add_argument("--min-depth", type=float, default=0.18)
    parser.add_argument("--max-depth", type=float, default=1.20)
    parser.add_argument("--plane-inner-radius-px", type=float, default=28)
    parser.add_argument("--plane-outer-radius-px", type=float, default=95)
    parser.add_argument("--min-plane-points", type=int, default=500)
    parser.add_argument("--plane-iters", type=int, default=5)
    parser.add_argument("--plane-inlier-m", type=float, default=0.006)
    parser.add_argument("--normal-draw-length-m", type=float, default=0.08)

    parser.add_argument("--hole-dark-threshold", type=int, default=70)
    parser.add_argument("--min-hole-radius-px", type=float, default=25)
    parser.add_argument("--max-hole-radius-px", type=float, default=100)
    parser.add_argument("--min-hole-area-px", type=float, default=250)
    parser.add_argument("--min-hole-circularity", type=float, default=0.45)
    parser.add_argument("--min-hole-inside-ratio", type=float, default=0.35)
    parser.add_argument("--expected-hole-radius-px", type=float, default=55)
    parser.add_argument("--center-circle-prior-max-px", type=float, default=45)
    parser.add_argument("--allow-outer-center-fallback", action="store_true",
                        help="If center circle is not found, use outer-wheel center as fallback.")
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
    parser.add_argument("--hough-param2", type=float, default=12)
    parser.add_argument("--disable-hough", action="store_true")

    parser.add_argument("--outer-dark-threshold", type=int, default=80)
    parser.add_argument("--outer-min-depth", type=float, default=0.30)
    parser.add_argument("--outer-max-depth", type=float, default=0.90)
    parser.add_argument("--outer-plane-dark-threshold", type=int, default=115)
    parser.add_argument("--outer-min-area", type=int, default=3000)
    parser.add_argument("--outer-min-radius-px", type=float, default=150)
    parser.add_argument("--outer-max-radius-px", type=float, default=280)
    parser.add_argument("--outer-circle-inlier-px", type=float, default=8.0)
    parser.add_argument("--outer-plane-band-px", type=float, default=24.0)
    parser.add_argument("--outer-ransac-iters", type=int, default=2500)
    parser.add_argument("--outer-ransac-seed", type=int, default=2)
    parser.add_argument("--outer-center-penalty", type=float, default=0.02)
    parser.add_argument("--outer-min-boundary-points", type=int, default=300)
    parser.add_argument("--outer-min-inliers", type=int, default=120)
    parser.add_argument("--outer-roi-x0", type=float, default=0.08)
    parser.add_argument("--outer-roi-x1", type=float, default=0.96)
    parser.add_argument("--outer-roi-y0", type=float, default=0.02)
    parser.add_argument("--outer-roi-y1", type=float, default=0.98)
    parser.add_argument("--center-ring-plane-gate-m", type=float, default=0.018)
    parser.add_argument("--metal-prior-source", choices=["outer-wheel", "image-center"], default="image-center")
    parser.add_argument("--metal-prior-x", type=float, default=0.53)
    parser.add_argument("--metal-prior-y", type=float, default=0.67)
    parser.add_argument("--metal-search-radius-px", type=float, default=180)
    parser.add_argument("--metal-prior-max-px", type=float, default=130)
    parser.add_argument("--metal-max-saturation", type=int, default=95)
    parser.add_argument("--metal-min-value", type=int, default=85)
    parser.add_argument("--metal-min-gray", type=int, default=85)
    parser.add_argument("--metal-outer-radius-px", type=float, default=0.0)
    parser.add_argument("--metal-outer-radius-scale", type=float, default=1.65)
    parser.add_argument("--metal-fit-inner-scale", type=float, default=1.12)
    parser.add_argument("--min-metal-ring-ratio", type=float, default=0.04)
    parser.add_argument("--min-metal-depth-ratio", type=float, default=0.12)
    parser.add_argument("--rgbd-plane-hole", action=argparse.BooleanOptionalAction, default=True,
                        help="Use RGBD plane occupancy to choose the center hole in metal-ring mode.")
    parser.add_argument("--rgbd-plane-max-saturation", type=int, default=135)
    parser.add_argument("--rgbd-plane-min-value", type=int, default=45)
    parser.add_argument("--rgbd-plane-min-gray", type=int, default=45)
    parser.add_argument("--rgbd-plane-min-seed-points", type=int, default=450)
    parser.add_argument("--rgbd-plane-ransac-iters", type=int, default=500)
    parser.add_argument("--rgbd-plane-ransac-seed", type=int, default=7)
    parser.add_argument("--rgbd-plane-inlier-m", type=float, default=0.008)
    parser.add_argument("--rgbd-inner-scale", type=float, default=0.72)
    parser.add_argument("--rgbd-edge-inner-scale", type=float, default=0.82)
    parser.add_argument("--rgbd-edge-outer-scale", type=float, default=1.18)
    parser.add_argument("--rgbd-outer-inner-scale", type=float, default=1.10)
    parser.add_argument("--rgbd-outer-outer-scale", type=float, default=1.85)
    parser.add_argument("--rgbd-min-candidate-radius-px", type=float, default=40)
    parser.add_argument("--rgbd-min-center-below-metal-centroid-px", type=float, default=25)
    parser.add_argument("--rgbd-max-center-below-metal-centroid-px", type=float, default=150)
    parser.add_argument("--rgbd-max-center-x-from-metal-centroid-px", type=float, default=90)
    parser.add_argument("--rgbd-max-inner-plane-ratio", type=float, default=0.36)
    parser.add_argument("--rgbd-min-outer-plane-ratio", type=float, default=0.12)
    parser.add_argument("--rgbd-sector-count", type=int, default=16)
    parser.add_argument("--rgbd-min-sector-coverage", type=float, default=0.50)
    parser.add_argument("--rgbd-grid-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgbd-grid-step-px", type=int, default=8)
    parser.add_argument("--rgbd-grid-radius-step-px", type=float, default=6)
    parser.add_argument("--rgbd-refine-topk", type=int, default=8)
    parser.add_argument("--rgbd-refine-offset-px", type=float, default=10)
    parser.add_argument("--rgbd-refine-radius-px", type=float, default=8)
    parser.add_argument("--rectified-plane-hole", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rectified-resolution-m", type=float, default=0.0007)
    parser.add_argument("--rectified-margin-m", type=float, default=0.025)
    parser.add_argument("--rectified-max-size-px", type=int, default=520)
    parser.add_argument("--rectified-search-radius-m", type=float, default=0.075)
    parser.add_argument("--rectified-dark-threshold", type=int, default=115)
    parser.add_argument("--rectified-min-area-px", type=float, default=180)
    parser.add_argument("--rectified-good-area-px", type=float, default=1800)
    parser.add_argument("--rectified-min-radius-m", type=float, default=0.014)
    parser.add_argument("--rectified-max-radius-m", type=float, default=0.050)
    parser.add_argument("--rectified-min-axis-ratio", type=float, default=0.45)
    parser.add_argument("--rectified-min-circularity", type=float, default=0.18)
    parser.add_argument("--rectified-min-dark-or-void-ratio", type=float, default=0.25)
    parser.add_argument("--rectified-fill-components", type=int, default=2)
    parser.add_argument("--rectified-score-bonus", type=float, default=1.5)
    parser.add_argument("--radial-refine-max-axis-ratio", type=float, default=0.72)
    parser.add_argument("--radial-refine-rays", type=int, default=180)
    parser.add_argument("--radial-refine-min-radius-px", type=float, default=25)
    parser.add_argument("--radial-refine-max-radius-px", type=float, default=65)
    parser.add_argument("--radial-refine-min-gradient", type=float, default=8)
    parser.add_argument("--radial-refine-min-points", type=int, default=45)
    parser.add_argument("--radial-refine-min-inliers", type=int, default=45)
    parser.add_argument("--radial-refine-max-residual-px", type=float, default=6)
    parser.add_argument("--radial-refine-max-rmse-px", type=float, default=5)
    parser.add_argument("--radial-refine-max-shift-px", type=float, default=30)
    parser.add_argument("--radial-refine-fit-iters", type=int, default=5)
    parser.add_argument("--depth-void-roi-scale", type=float, default=1.45)
    parser.add_argument("--depth-void-min-roi-radius-px", type=float, default=70)
    parser.add_argument("--depth-void-min-area-px", type=float, default=250)
    parser.add_argument("--depth-void-min-radius-px", type=float, default=18)
    parser.add_argument("--depth-void-max-radius-px", type=float, default=65)
    parser.add_argument("--depth-void-max-shift-px", type=float, default=45)
    parser.add_argument("--depth-void-min-circularity", type=float, default=0.25)
    parser.add_argument("--reject-small-side-hole-radius-px", type=float, default=18)
    parser.add_argument("--radius-selection-weight", type=float, default=0.010)
    parser.add_argument("--center-selection-weight", type=float, default=0.006)
    parser.add_argument("--quality-min-radius-px", type=float, default=20)
    parser.add_argument("--quality-max-radius-px", type=float, default=90)
    parser.add_argument("--max-quality-center-error-px", type=float, default=90)
    parser.add_argument("--max-quality-rmse-m", type=float, default=0.004)
    parser.add_argument("--min-quality-points", type=int, default=1200)
    parser.add_argument("--min-quality-inlier-ratio", type=float, default=0.30)
    parser.add_argument("--min-quality-score", type=float, default=1.0)
    parser.add_argument("--edge-roi-radius-px", type=float, default=170)
    parser.add_argument("--edge-canny1", type=int, default=50)
    parser.add_argument("--edge-canny2", type=int, default=140)
    parser.add_argument("--edge-min-contour-points", type=int, default=20)
    parser.add_argument("--edge-min-area-px", type=float, default=100)
    parser.add_argument("--edge-good-area-px", type=float, default=900)
    parser.add_argument("--edge-min-minor-axis-px", type=float, default=35)
    parser.add_argument("--edge-max-minor-axis-px", type=float, default=180)
    parser.add_argument("--edge-min-major-axis-px", type=float, default=45)
    parser.add_argument("--edge-max-major-axis-px", type=float, default=220)
    parser.add_argument("--edge-min-axis-ratio", type=float, default=0.45)
    return parser.parse_args()


def main():
    args = parse_args()
    result, json_path, overlay_path = run_detection(args.folder, args)
    print_result(result, json_path, overlay_path)


if __name__ == "__main__":
    main()
