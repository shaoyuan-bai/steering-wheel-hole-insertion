# -*- coding: utf-8 -*-
"""Detect steering-wheel center hole pose using YOLO-seg mask + RGBD plane fit."""

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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infer_center_hole_onnx import crop_mask_to_box, letterbox, nms, sigmoid  # noqa: E402
from insert_center_hole import backproject_pixels, intersect_pixel_ray_with_plane, intr_value, robust_plane  # noqa: E402
from config_loader import CONFIG, cfg_get, relative_path  # noqa: E402


SCRIPT_VERSION = "2026-05-25-yolo-rgbd-center-hole-v1"
DEFAULT_MODEL = relative_path(CONFIG, "detection", "model", default="label_dataset/best.onnx")


def load_rgbd(input_path):
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        if not (path / "color.png").exists():
            capture_dirs = sorted(
                [p for p in path.iterdir() if p.is_dir() and (p / "color.png").exists()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if capture_dirs:
                path = capture_dirs[0]

        color_path = path / "color.png"
        depth_path = path / "depth_raw.npy"
        meta_path = path / "intrinsics.json"
        color = cv2.imread(str(color_path))
        if color is None:
            raise RuntimeError(f"Cannot read {color_path}")
        if not depth_path.exists():
            raise RuntimeError(f"Missing {depth_path}")
        depth_m = np.load(str(depth_path)).astype(np.float32)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        intr = meta["color_intrinsics"]
        return path, color_path, color, depth_m, intr, meta

    color_path = path
    color = cv2.imread(str(color_path))
    if color is None:
        raise RuntimeError(f"Cannot read {color_path}")

    dataset_root = color_path.parents[1]
    depth_path = dataset_root / "depth_m" / f"{color_path.stem}.npy"
    meta_path = dataset_root / "meta" / f"{color_path.stem}.json"
    if not depth_path.exists() or not meta_path.exists():
        raise RuntimeError(
            "Image input needs matching depth/meta files. Expected "
            f"{depth_path} and {meta_path}"
        )
    depth_m = np.load(str(depth_path)).astype(np.float32)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    intr = meta["color_intrinsics"]
    return dataset_root, color_path, color, depth_m, intr, meta


def yolo_mask_onnx(net, image, args):
    orig_h, orig_w = image.shape[:2]
    inp, scale, pad_x, pad_y = letterbox(image, int(args.imgsz))
    blob = cv2.dnn.blobFromImage(inp, 1.0 / 255.0, (int(args.imgsz), int(args.imgsz)), swapRB=True, crop=False)
    net.setInput(blob)
    pred, proto = net.forward(net.getUnconnectedOutLayersNames())
    pred = pred[0].transpose(1, 0)
    proto = proto[0]

    boxes = []
    scores = []
    coeffs = []
    for row in pred:
        score = float(row[4])
        if score < float(args.conf):
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
    keep = nms(boxes, scores, float(args.iou))
    if not keep:
        return None
    best = max(keep, key=lambda idx: float(scores[idx]))

    proto_flat = proto.reshape(proto.shape[0], -1)
    mask_logits = coeffs[best] @ proto_flat
    mask = sigmoid(mask_logits).reshape(proto.shape[1], proto.shape[2])
    mask = cv2.resize(mask, (int(args.imgsz), int(args.imgsz)), interpolation=cv2.INTER_LINEAR)
    mask_u8 = (mask >= float(args.mask_threshold)).astype(np.uint8)
    mask_u8 = crop_mask_to_box(mask_u8, boxes[best])

    unpad_h = int(round(orig_h * scale))
    unpad_w = int(round(orig_w * scale))
    unpad = mask_u8[pad_y:pad_y + unpad_h, pad_x:pad_x + unpad_w]
    mask_orig = cv2.resize(unpad, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours((mask_orig * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    moments = cv2.moments(contour)
    if area < float(args.min_mask_area_px) or abs(float(moments["m00"])) < 1e-6:
        return None
    center = np.array([float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])], dtype=np.float64)

    box = boxes[best].copy()
    box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
    box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
    box[[0, 2]] = np.clip(box[[0, 2]], 0, orig_w - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, orig_h - 1)

    return {
        "mask": mask_orig.astype(bool),
        "contour": contour,
        "center_px": center,
        "score": float(scores[best]),
        "box_xyxy": [float(v) for v in box],
        "mask_area_px": area,
        "equiv_radius_px": float(math.sqrt(area / math.pi)),
    }


def fit_metal_plane_around_mask(color, depth_m, intr, mask_result, args):
    h, w = depth_m.shape[:2]
    center = np.asarray(mask_result["center_px"], dtype=np.float64)
    radius = float(mask_result["equiv_radius_px"])
    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)

    inner = max(float(args.plane_inner_radius_px), radius * float(args.plane_inner_scale))
    outer = max(float(args.plane_outer_radius_px), radius * float(args.plane_outer_scale))
    valid_depth = (depth_m >= float(args.min_depth)) & (depth_m <= float(args.max_depth))
    mask_dilated = cv2.dilate(mask_result["mask"].astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    metal_like = (
        (hsv[:, :, 1] <= int(args.metal_max_saturation))
        & (gray >= int(args.metal_min_gray))
        & (hsv[:, :, 2] >= int(args.metal_min_value))
    )
    annulus = (
        (dist >= inner)
        & (dist <= outer)
        & valid_depth
        & (~mask_dilated)
        & metal_like
    )
    ys, xs = np.nonzero(annulus)
    mask_source = "metal-color-annulus"
    if len(xs) < int(args.min_plane_points):
        annulus = (dist >= inner) & (dist <= outer) & valid_depth & (~mask_dilated)
        ys, xs = np.nonzero(annulus)
        mask_source = "depth-annulus-fallback"
    if len(xs) < int(args.min_plane_points):
        raise RuntimeError(f"Too few plane points around YOLO mask: {len(xs)}")

    points = backproject_pixels(xs, ys, depth_m[ys, xs].astype(np.float64), intr)
    plane_center, normal_cam, inliers, rmse = robust_plane(
        points,
        iterations=int(args.plane_iters),
        threshold_m=float(args.plane_inlier_m),
    )
    center_cam = intersect_pixel_ray_with_plane(center[0], center[1], plane_center, normal_cam, intr)
    return {
        "center_cam": center_cam,
        "normal_cam": normal_cam,
        "plane_center_cam": plane_center,
        "plane_points": int(len(points)),
        "plane_inliers": int(len(inliers)),
        "plane_rmse_m": float(rmse),
        "annulus_inner_px": float(inner),
        "annulus_outer_px": float(outer),
        "mask_source": mask_source,
    }


def point_to_pixel(point, intr):
    point = np.asarray(point, dtype=np.float64)
    if point[2] <= 1e-6:
        return None
    return (
        int(round(point[0] * intr_value(intr, "fx") / point[2] + intr_value(intr, "ppx"))),
        int(round(point[1] * intr_value(intr, "fy") / point[2] + intr_value(intr, "ppy"))),
    )


def draw_overlay(color, mask_result, plane_result, intr, args, quality, reasons):
    overlay = color.copy()
    mask = mask_result["mask"]
    green = np.zeros_like(color)
    green[:, :, 1] = 255
    overlay = np.where(mask[:, :, None], cv2.addWeighted(color, 1.0 - float(args.overlay_alpha), green, float(args.overlay_alpha), 0), overlay)
    cv2.drawContours(overlay, [mask_result["contour"]], -1, (0, 255, 0), 2)

    center = mask_result["center_px"]
    center_i = (int(round(center[0])), int(round(center[1])))
    cv2.drawMarker(overlay, center_i, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
    cv2.circle(overlay, center_i, int(round(mask_result["equiv_radius_px"])), (0, 255, 0), 1)
    cv2.circle(overlay, center_i, int(round(plane_result["annulus_outer_px"])), (0, 255, 255), 1)

    center_cam = np.asarray(plane_result["center_cam"], dtype=np.float64)
    normal_cam = np.asarray(plane_result["normal_cam"], dtype=np.float64)
    projected = point_to_pixel(center_cam, intr)
    if projected is not None:
        tip_px = point_to_pixel(center_cam + normal_cam * float(args.normal_draw_length_m), intr)
        if tip_px is not None:
            cv2.arrowedLine(overlay, projected, tip_px, (0, 0, 255), 2, tipLength=0.22)
            cv2.putText(overlay, "+N", (tip_px[0] + 5, tip_px[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    lines = [
        f"version: {SCRIPT_VERSION}",
        f"score: {mask_result['score']:.3f}",
        f"center_px: [{center[0]:.1f}, {center[1]:.1f}]",
        f"center_cam_m: [{center_cam[0]:.4f}, {center_cam[1]:.4f}, {center_cam[2]:.4f}]",
        f"normal_cam: [{normal_cam[0]:.4f}, {normal_cam[1]:.4f}, {normal_cam[2]:.4f}]",
        f"plane_rmse_mm: {plane_result['plane_rmse_m'] * 1000.0:.2f}",
        f"points/inliers: {plane_result['plane_points']}/{plane_result['plane_inliers']}",
        f"quality: {quality}",
    ]
    for reason in reasons[:2]:
        lines.append(reason)
    for idx, line in enumerate(lines):
        y = 24 + idx * 24
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


def assess_quality(mask_result, plane_result, args):
    reasons = []
    inlier_ratio = float(plane_result["plane_inliers"]) / max(float(plane_result["plane_points"]), 1.0)
    if float(mask_result["score"]) < float(args.min_confidence):
        reasons.append(f"score={mask_result['score']:.2f}<{args.min_confidence:.2f}")
    if float(mask_result["mask_area_px"]) < float(args.min_mask_area_px):
        reasons.append(f"mask_area={mask_result['mask_area_px']:.1f}<{args.min_mask_area_px:.1f}")
    if float(mask_result["mask_area_px"]) > float(args.max_mask_area_px):
        reasons.append(f"mask_area={mask_result['mask_area_px']:.1f}>{args.max_mask_area_px:.1f}")
    if float(plane_result["plane_rmse_m"]) > float(args.max_plane_rmse_m):
        reasons.append(f"rmse_mm={plane_result['plane_rmse_m'] * 1000.0:.2f}>{args.max_plane_rmse_m * 1000.0:.2f}")
    normal_z_abs = abs(float(np.asarray(plane_result["normal_cam"], dtype=np.float64)[2]))
    if normal_z_abs < float(args.min_abs_normal_z):
        reasons.append(f"abs_normal_z={normal_z_abs:.2f}<{args.min_abs_normal_z:.2f}")
    if int(plane_result["plane_inliers"]) < int(args.min_plane_inliers):
        reasons.append(f"plane_inliers={plane_result['plane_inliers']}<{args.min_plane_inliers}")
    if inlier_ratio < float(args.min_plane_inlier_ratio):
        reasons.append(f"inlier_ratio={inlier_ratio:.2f}<{args.min_plane_inlier_ratio:.2f}")
    return ("ok" if not reasons else "reject"), reasons


def run_detection(input_path, args):
    source_root, color_path, color, depth_m, intr, meta = load_rgbd(input_path)
    model_path = Path(args.model).expanduser().resolve()
    net = cv2.dnn.readNetFromONNX(str(model_path))
    mask_result = yolo_mask_onnx(net, color, args)
    if mask_result is None:
        raise RuntimeError("YOLO center_hole mask was not found.")
    plane_result = fit_metal_plane_around_mask(color, depth_m, intr, mask_result, args)
    quality, reasons = assess_quality(mask_result, plane_result, args)

    result = {
        "script_version": SCRIPT_VERSION,
        "method": "yolo-seg-mask-rgbd-plane",
        "input": str(Path(input_path).expanduser().resolve()),
        "image": str(color_path),
        "model": str(model_path),
        "params": vars(args),
        "source": {
            "source": "yolo-seg-onnx",
            "score": float(mask_result["score"]),
            "mask_area_px": float(mask_result["mask_area_px"]),
            "box_xyxy": mask_result["box_xyxy"],
            "equiv_radius_px": float(mask_result["equiv_radius_px"]),
        },
        "center_px": [float(v) for v in mask_result["center_px"]],
        "hole_radius_px": float(mask_result["equiv_radius_px"]),
        "center_headcam_m": [float(v) for v in plane_result["center_cam"]],
        "normal_headcam": [float(v) for v in plane_result["normal_cam"]],
        "plane_center_headcam_m": [float(v) for v in plane_result["plane_center_cam"]],
        "plane_rmse_m": float(plane_result["plane_rmse_m"]),
        "plane_points": int(plane_result["plane_points"]),
        "plane_inliers": int(plane_result["plane_inliers"]),
        "quality": quality,
        "quality_reasons": reasons,
        "yolo_rgbd_result": {
            "annulus_inner_px": float(plane_result["annulus_inner_px"]),
            "annulus_outer_px": float(plane_result["annulus_outer_px"]),
            "mask_source": plane_result["mask_source"],
        },
    }

    overlay = draw_overlay(color, mask_result, plane_result, intr, args, quality, reasons)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_stem if args.out_stem else f"yolo_rgbd_{Path(color_path).stem}"
    overlay_path = out_dir / f"{stem}_overlay.png"
    json_path = out_dir / f"{stem}_detection.json"
    cv2.imwrite(str(overlay_path), overlay)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result, json_path, overlay_path


def print_result(result, json_path, overlay_path):
    print(f"[OK] saved: {json_path}")
    print(f"[OK] saved: {overlay_path}")
    print(f"score: {result['source']['score']:.3f}")
    print(f"center_px: {[round(float(v), 2) for v in result['center_px']]}")
    print(f"center_headcam_m: {[round(float(v), 5) for v in result['center_headcam_m']]}")
    print(f"normal_headcam: {[round(float(v), 5) for v in result['normal_headcam']]}")
    print(f"plane_rmse_mm: {result['plane_rmse_m'] * 1000.0:.3f}")
    print(f"plane_points/inliers: {result['plane_points']}/{result['plane_inliers']}")
    print(f"quality: {result['quality']}")
    for reason in result.get("quality_reasons", []):
        print(f"quality_reason: {reason}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="RGBD capture folder, or label_dataset/images/*.png with matching depth/meta.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--out-stem", default="")
    parser.add_argument("--imgsz", type=int, default=int(cfg_get(CONFIG, "detection", "imgsz", default=960)))
    parser.add_argument("--conf", type=float, default=float(cfg_get(CONFIG, "detection", "min_confidence", default=0.35)))
    parser.add_argument("--iou", type=float, default=float(cfg_get(CONFIG, "detection", "yolo_iou", default=0.45)))
    parser.add_argument("--mask-threshold", type=float, default=float(cfg_get(CONFIG, "detection", "mask_threshold", default=0.5)))
    parser.add_argument("--min-depth", type=float, default=0.18)
    parser.add_argument("--max-depth", type=float, default=1.20)
    parser.add_argument("--plane-inner-radius-px", type=float, default=20)
    parser.add_argument("--plane-outer-radius-px", type=float, default=115)
    parser.add_argument("--plane-inner-scale", type=float, default=1.55)
    parser.add_argument("--plane-outer-scale", type=float, default=4.2)
    parser.add_argument("--min-plane-points", type=int, default=500)
    parser.add_argument("--plane-iters", type=int, default=6)
    parser.add_argument("--plane-inlier-m", type=float, default=0.006)
    parser.add_argument("--metal-max-saturation", type=int, default=145)
    parser.add_argument("--metal-min-gray", type=int, default=35)
    parser.add_argument("--metal-min-value", type=int, default=35)
    parser.add_argument("--min-confidence", type=float, default=float(cfg_get(CONFIG, "detection", "min_confidence", default=0.45)))
    parser.add_argument("--min-mask-area-px", type=float, default=60)
    parser.add_argument("--max-mask-area-px", type=float, default=20000)
    parser.add_argument("--max-plane-rmse-m", type=float, default=0.004)
    parser.add_argument("--min-abs-normal-z", type=float, default=float(cfg_get(CONFIG, "detection", "min_abs_normal_z", default=0.55)))
    parser.add_argument("--min-plane-inliers", type=int, default=1200)
    parser.add_argument("--min-plane-inlier-ratio", type=float, default=0.10)
    parser.add_argument("--normal-draw-length-m", type=float, default=float(cfg_get(CONFIG, "detection", "normal_draw_length_m", default=0.08)))
    parser.add_argument("--overlay-alpha", type=float, default=0.38)
    return parser.parse_args()


def main():
    args = parse_args()
    result, json_path, overlay_path = run_detection(args.input, args)
    print_result(result, json_path, overlay_path)


if __name__ == "__main__":
    main()
