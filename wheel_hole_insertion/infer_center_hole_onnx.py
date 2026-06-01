# -*- coding: utf-8 -*-
"""Run center-hole YOLO segmentation ONNX and draw the actual mask."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_loader import CONFIG, relative_path  # noqa: E402

DEFAULT_MODEL = relative_path(CONFIG, "detection", "model", default="label_dataset/best.onnx")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(image, size):
    h, w = image.shape[:2]
    scale = min(size / float(w), size / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    boxes_xywh = []
    for x1, y1, x2, y2 in boxes:
        boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), score_threshold=0.0, nms_threshold=float(iou_threshold))
    if len(indices) == 0:
        return []
    return np.asarray(indices).reshape(-1).tolist()


def crop_mask_to_box(mask, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    keep = np.zeros_like(mask, dtype=np.uint8)
    h, w = mask.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 > x1 and y2 > y1:
        keep[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return keep


def infer(model_path, image_path, args):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    orig_h, orig_w = image.shape[:2]
    inp, scale, pad_x, pad_y = letterbox(image, int(args.imgsz))

    net = cv2.dnn.readNetFromONNX(str(model_path))
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
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        coeffs.append(row[5:].astype(np.float32))

    if not boxes:
        return image, None

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    coeffs = np.asarray(coeffs, dtype=np.float32)
    keep = nms(boxes, scores, float(args.iou))
    if not keep:
        return image, None
    best = max(keep, key=lambda idx: float(scores[idx]))

    proto_flat = proto.reshape(proto.shape[0], -1)
    mask_logits = coeffs[best] @ proto_flat
    mask = sigmoid(mask_logits).reshape(proto.shape[1], proto.shape[2])
    mask = cv2.resize(mask, (int(args.imgsz), int(args.imgsz)), interpolation=cv2.INTER_LINEAR)
    mask_u8 = (mask >= float(args.mask_threshold)).astype(np.uint8)
    mask_u8 = crop_mask_to_box(mask_u8, boxes[best])

    unpad = mask_u8[pad_y:pad_y + int(round(orig_h * scale)), pad_x:pad_x + int(round(orig_w * scale))]
    mask_orig = cv2.resize(unpad, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours((mask_orig * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    moments = cv2.moments(contour)
    if abs(float(moments["m00"])) < 1e-6:
        center = None
    else:
        center = [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]

    box = boxes[best].copy()
    box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
    box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
    box[[0, 2]] = np.clip(box[[0, 2]], 0, orig_w - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, orig_h - 1)

    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[:, :, 1] = 255
    alpha = float(args.alpha)
    overlay = np.where(mask_orig[:, :, None].astype(bool), cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0), overlay)
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    if args.draw_box:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if center is not None:
        cv2.drawMarker(overlay, (int(round(center[0])), int(round(center[1]))), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    result = {
        "image": str(image_path),
        "model": str(model_path),
        "score": float(scores[best]),
        "box_xyxy": [float(v) for v in box],
        "mask_area_px": area,
        "center_px": center,
        "mask_threshold": float(args.mask_threshold),
    }
    return overlay, result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--out", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.38)
    parser.add_argument("--draw-box", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    overlay, result = infer(model_path, image_path, args)

    out_path = Path(args.out).expanduser().resolve() if args.out else image_path.with_name(image_path.stem + "_onnx_mask.png")
    cv2.imwrite(str(out_path), overlay)
    print(f"[OK] saved: {out_path}")
    if result is None:
        print("[WARN] no mask")
        return
    json_path = Path(args.json_out).expanduser().resolve() if args.json_out else out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[OK] saved: {json_path}")
    print(f"score: {result['score']:.3f}")
    print(f"center_px: {result['center_px']}")
    print(f"mask_area_px: {result['mask_area_px']:.1f}")


if __name__ == "__main__":
    main()
