# -*- coding: utf-8 -*-
"""
RGB-only debug detector for the steering-wheel center metal ring.

This script is for visual validation only. It detects the inner circle of the
silver center metal ring from a normal RGB image and writes an overlay. It
cannot estimate the 3D normal because depth is not available.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_VERSION = "2026-05-20-metal-ring-rgb-debug-v1"


def normalize_u8(gray):
    lo, hi = np.percentile(gray, [1.0, 99.0])
    if hi <= lo:
        return gray.copy()
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def make_metal_mask(image, args):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    metal = (
        (saturation <= int(args.metal_max_saturation))
        & (value >= int(args.metal_min_value))
        & (gray >= int(args.metal_min_gray))
    )
    metal_u8 = (metal.astype(np.uint8) * 255)
    metal_u8 = cv2.morphologyEx(metal_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    metal_u8 = cv2.morphologyEx(metal_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    return metal_u8


def score_candidate(center, radius, image, metal_u8, prior, args):
    h, w = image.shape[:2]
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    inner = rr <= max(radius * 0.75, 2.0)
    rim = (rr >= radius * 0.95) & (rr <= radius * float(args.rim_outer_scale))
    ring = (rr >= radius * float(args.fit_inner_scale)) & (rr <= radius * float(args.fit_outer_scale))
    if int(inner.sum()) < 10 or int(rim.sum()) < 10 or int(ring.sum()) < 10:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inner_dark = float((gray[inner] <= int(args.hole_dark_threshold)).sum()) / float(inner.sum())
    inner_not_metal = 1.0 - float((metal_u8[inner] > 0).sum()) / float(inner.sum())
    rim_metal = float((metal_u8[rim] > 0).sum()) / float(rim.sum())
    ring_metal = float((metal_u8[ring] > 0).sum()) / float(ring.sum())
    if rim_metal < float(args.min_rim_metal_ratio):
        return None
    center_error = float(np.linalg.norm(np.asarray(center) - np.asarray(prior)))
    if center_error > float(args.prior_max_px):
        return None
    return (
        3.0 * rim_metal
        + 2.0 * ring_metal
        + 0.7 * inner_not_metal
        + 0.4 * inner_dark
        + 1.0 * (1.0 - min(1.0, center_error / max(float(args.prior_max_px), 1.0)))
        - 0.003 * abs(float(radius) - float(args.expected_radius_px))
    )


def detect_metal_ring(image, args):
    h, w = image.shape[:2]
    prior = np.array([
        w * float(args.prior_x),
        h * float(args.prior_y),
    ], dtype=np.float64)
    metal_u8 = make_metal_mask(image, args)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = normalize_u8(gray)
    blur = cv2.GaussianBlur(enhanced, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=float(args.hough_dp),
        minDist=float(args.hough_min_dist_px),
        param1=float(args.hough_param1),
        param2=float(args.hough_param2),
        minRadius=int(args.min_radius_px),
        maxRadius=int(args.max_radius_px),
    )

    candidates = []
    if circles is not None:
        for x, y, radius in np.round(circles[0, :]).astype(np.float64):
            center = np.array([x, y], dtype=np.float64)
            score = score_candidate(center, float(radius), image, metal_u8, prior, args)
            if score is None:
                continue
            candidates.append({
                "center_px": center,
                "radius_px": float(radius),
                "score": float(score),
                "source": "hough",
            })

    if not candidates:
        raise RuntimeError("No metal ring inner circle found. Tune radius/prior/metal thresholds.")

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    best["prior_px"] = prior
    best["candidate_count"] = len(candidates)
    best["metal_mask"] = metal_u8
    return best


def draw_overlay(image, result, args):
    overlay = image.copy()
    center = result["center_px"]
    radius = float(result["radius_px"])
    prior = result["prior_px"]
    center_i = (int(round(center[0])), int(round(center[1])))
    prior_i = (int(round(prior[0])), int(round(prior[1])))

    cv2.circle(overlay, center_i, int(round(radius)), (0, 255, 0), 3)
    cv2.drawMarker(overlay, center_i, (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
    cv2.circle(overlay, center_i, int(round(radius * args.fit_outer_scale)), (0, 255, 255), 2)
    cv2.drawMarker(overlay, prior_i, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)

    lines = [
        f"version: {SCRIPT_VERSION}",
        f"center_px: [{center[0]:.1f}, {center[1]:.1f}]",
        f"radius_px: {radius:.1f}",
        f"score: {result['score']:.3f}",
        f"source: {result['source']}, candidates={result['candidate_count']}",
        "RGB-only: no 3D normal",
    ]
    for idx, line in enumerate(lines):
        y = 28 + idx * 28
        cv2.putText(overlay, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


def run(args):
    image_path = Path(args.image).expanduser().resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    result = detect_metal_ring(image, args)
    overlay = draw_overlay(image, result, args)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_stem if args.out_stem else f"{image_path.stem}_metal_ring_rgb"
    overlay_path = out_dir / f"{stem}_overlay.png"
    json_path = out_dir / f"{stem}_detection.json"

    output = {
        "image": str(image_path),
        "script_version": SCRIPT_VERSION,
        "method": "rgb-metal-ring-inner-circle-v1",
        "center_px": [float(result["center_px"][0]), float(result["center_px"][1])],
        "radius_px": float(result["radius_px"]),
        "score": float(result["score"]),
        "source": result["source"],
        "candidate_count": int(result["candidate_count"]),
        "note": "RGB-only result. Use RGBD for normal estimation.",
        "params": vars(args),
    }
    cv2.imwrite(str(overlay_path), overlay)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return output, json_path, overlay_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="RGB image path.")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--out-stem", default="")
    parser.add_argument("--prior-x", type=float, default=0.53, help="Prior center x as image width ratio.")
    parser.add_argument("--prior-y", type=float, default=0.45, help="Prior center y as image height ratio.")
    parser.add_argument("--prior-max-px", type=float, default=220)
    parser.add_argument("--min-radius-px", type=float, default=25)
    parser.add_argument("--max-radius-px", type=float, default=95)
    parser.add_argument("--expected-radius-px", type=float, default=55)
    parser.add_argument("--hough-dp", type=float, default=1.2)
    parser.add_argument("--hough-min-dist-px", type=float, default=80)
    parser.add_argument("--hough-param1", type=float, default=100)
    parser.add_argument("--hough-param2", type=float, default=24)
    parser.add_argument("--metal-max-saturation", type=int, default=105)
    parser.add_argument("--metal-min-value", type=int, default=90)
    parser.add_argument("--metal-min-gray", type=int, default=85)
    parser.add_argument("--hole-dark-threshold", type=int, default=150)
    parser.add_argument("--rim-outer-scale", type=float, default=1.35)
    parser.add_argument("--fit-inner-scale", type=float, default=1.15)
    parser.add_argument("--fit-outer-scale", type=float, default=2.20)
    parser.add_argument("--min-rim-metal-ratio", type=float, default=0.18)
    return parser.parse_args()


def main():
    result, json_path, overlay_path = run(parse_args())
    print(f"[OK] saved: {json_path}")
    print(f"[OK] saved: {overlay_path}")
    print(f"center_px: {[round(x, 2) for x in result['center_px']]}")
    print(f"radius_px: {result['radius_px']:.2f}")
    print(f"score: {result['score']:.3f}")
    print("note: RGB-only, cannot estimate 3D normal")


if __name__ == "__main__":
    main()
