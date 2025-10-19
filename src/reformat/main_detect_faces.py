#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face-Detection + Mouth-Openness (YOLOv8-face + MediaPipe)
- liest Rohclips aus RAW_CLIPS_DIR
- schreibt pro Video eine faces.json in FACE_COMBINED_DIR
- optionaler Fortschrittsbalken (tqdm)
"""

from __future__ import annotations
import argparse
import logging
import json
import time
from pathlib import Path
from contextlib import nullcontext
from typing import List, Dict, Any
from src.reformat.speaking import get_mouth_openness

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp
import sys

# ── Projekt-Root + zentrale Pfade laden
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import RAW_CLIPS_DIR, FACE_COMBINED_DIR  # zentrale Verzeichnisse

# Fortschritt hübsch, wenn verfügbar
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ---------- Performance Tweaks ----------
torch.set_float32_matmul_precision("high")
cv2.setUseOptimized(True)

# ---------- Hilfsfunktionen ----------
def make_square_crop(x1, y1, x2, y2, W, H, margin_scale, min_crop):
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w  = (x2 - x1) * (1.0 + 2.0 * margin_scale)
    h  = (y2 - y1) * (1.0 + 2.0 * margin_scale)
    side = max(w, h, float(min_crop))
    half = side * 0.5

    sx1 = int(max(0, round(cx - half)))
    sy1 = int(max(0, round(cy - half)))
    sx2 = int(min(W, round(cx + half)))
    sy2 = int(min(H, round(cy + half)))

    side_w = max(0, sx2 - sx1)
    side_h = max(0, sy2 - sy1)
    side   = max(2, min(side_w, side_h))
    sx2 = sx1 + side
    sy2 = sy1 + side
    return sx1, sy1, sx2, sy2


def pick_landmarks_near_crop_center(lm_lists, crop_w, crop_h):
    if not lm_lists:
        return None
    cx_t, cy_t = crop_w * 0.5, crop_h * 0.5
    best, best_d = None, 1e12
    for lms in lm_lists:
        xs = [p.x * crop_w for p in lms.landmark]
        ys = [p.y * crop_h for p in lms.landmark]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        d = (cx - cx_t) ** 2 + (cy - cy_t) ** 2
        if d < best_d:
            best, best_d = lms, d
    return best


def run_mesh(face_mesh, crop_bgr, upscale_if_small):
    if crop_bgr.size == 0:
        return None, 0.0
    ch, cw = crop_bgr.shape[:2]
    if max(ch, cw) < upscale_if_small:
        scale = float(upscale_if_small) / max(ch, cw)
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        ch, cw = new_h, new_w
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None, 0.0
    chosen = pick_landmarks_near_crop_center(res.multi_face_landmarks, cw, ch)
    if chosen is None:
        return None, 0.0
    mo = get_mouth_openness(chosen.landmark, ch)
    return chosen, float(mo)

# ---------- Kernprozess ----------
def process_video(video_path: Path,
                  output_path: Path,
                  model: YOLO,
                  face_mesh,
                  conf_thresh: float,
                  frame_skip: int,
                  downscale: float,
                  expansion_1: float,
                  expansion_2: float,
                  min_crop: int,
                  faces_upscale: int,
                  imgsz: int,
                  device: str,
                  max_det: int):
    print(f"🎬 Starte Detection: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"❌ Kann Video nicht öffnen: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_to_process = None
    if total_frames_raw > 0:
        total_to_process = (total_frames_raw + (frame_skip - 1)) // max(1, frame_skip)

    scaled_w = max(1, int(round(orig_w * downscale)))
    scaled_h = max(1, int(round(orig_h * downscale)))

    data: List[Dict[str, Any]] = []
    frame_idx = 0
    processed_frames = 0

    sx = (orig_w / scaled_w) if downscale != 1.0 else 1.0
    sy = (orig_h / scaled_h) if downscale != 1.0 else 1.0

    autocast_ctx = (
        torch.autocast(device_type=device, dtype=torch.float16)
        if device in ("mps", "cuda") else nullcontext()
    )

    bar = None
    start_t = time.time()
    if _HAS_TQDM and total_to_process:
        bar = tqdm(total=total_to_process, desc=f"{video_path.name}", unit="f", leave=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_skip > 1 and (frame_idx % frame_skip != 0):
            frame_idx += 1
            continue

        frame_infer = frame if downscale == 1.0 else cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        with torch.no_grad():
            with autocast_ctx:
                # Ultralytics 8 API: __call__ statt .predict() (beide funktionieren)
                result = model(frame_infer, imgsz=imgsz, device=device, verbose=False,
                               conf=conf_thresh, iou=0.5, max_det=max_det)
                detections = result[0]

        faces = []
        for i in range(len(detections.boxes)):
            box = detections.boxes[i]
            conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            if downscale != 1.0:
                x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy
            x1 = max(0.0, min(x1, orig_w - 1))
            y1 = max(0.0, min(y1, orig_h - 1))
            x2 = max(0.0, min(x2, orig_w - 1))
            y2 = max(0.0, min(y2, orig_h - 1))

            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            # Pass 1
            sx1, sy1, sx2, sy2 = make_square_crop(x1, y1, x2, y2, orig_w, orig_h, expansion_1, min_crop)
            if sx2 - sx1 < 4 or sy2 - sy1 < 4:
                continue
            face_crop = frame[sy1:sy2, sx1:sx2]
            _, mouth_open = run_mesh(face_mesh, face_crop, faces_upscale)

            # Pass 2 nur wenn nötig
            if mouth_open == 0.0:
                sx1b, sy1b, sx2b, sy2b = make_square_crop(x1, y1, x2, y2, orig_w, orig_h, expansion_2, min_crop)
                if (sx2b - sx1b) >= 4 and (sy2b - sy1b) >= 4:
                    face_crop_b = frame[sy1b:sy2b, sx1b:sx2b]
                    _, mouth_open = run_mesh(face_mesh, face_crop_b, faces_upscale)

            faces.append({
                "bbox": [int(round(x1)), int(round(y1)), int(round(w)), int(round(h))],
                "conf": round(conf, 3),
                "center": [round(cx, 1), round(cy, 1)],
                "mouth_openness": round(float(mouth_open), 3)
            })

        data.append({
            "frame": frame_idx,
            "timestamp": round(frame_idx / fps, 3),
            "W": orig_w,
            "H": orig_h,
            "faces": faces
        })
        frame_idx += 1
        processed_frames += 1

        # Fortschritt
        if bar is not None:
            bar.update(1)
        else:
            if processed_frames % 30 == 0:
                elapsed = time.time() - start_t
                rate = processed_frames / max(1e-6, elapsed)  # frames/sec
                if total_to_process:
                    remaining = max(0, total_to_process - processed_frames)
                    eta_sec = remaining / max(1e-6, rate)
                    print(f"[{video_path.name}] {processed_frames}/{total_to_process} "
                          f"({processed_frames/total_to_process*100:.1f}%) "
                          f"— {rate:.1f} f/s — ETA {eta_sec/60:.1f} min")
                else:
                    print(f"[{video_path.name}] {processed_frames} frames — {rate:.1f} f/s")

    cap.release()
    if bar is not None:
        bar.close()

    # schön formatiertes JSON
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Faces gespeichert: {output_path.name}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8-face + MediaPipe FaceMesh → faces.json pro Clip")
    # Verzeichnisse (Default aus config.py)
    p.add_argument("--input-dir", type=Path, default=RAW_CLIPS_DIR, help=f"Rohclips (Default: {RAW_CLIPS_DIR})")
    p.add_argument("--output-dir", type=Path, default=FACE_COMBINED_DIR, help=f"Zielordner (Default: {FACE_COMBINED_DIR})")
    # Modell
    p.add_argument("--model", type=Path, default=ROOT / "models" / "yolov8n-face.pt",
                   help="Pfad zum YOLOv8-face Modell (.pt)")
    # Optimierte Defaults
    p.add_argument("--conf-thresh", type=float, default=0.35)
    p.add_argument("--frame-skip", type=int, default=1, help="Nur jeden n-ten Frame verarbeiten")
    p.add_argument("--downscale", type=float, default=0.5, help="Eingangsframe auf Faktor verkleinern (0..1)")
    p.add_argument("--expansion", type=float, default=0.4, help="Crop-Margin Pass 1 (relativ)")
    p.add_argument("--expansion2", type=float, default=0.8, help="Crop-Margin Pass 2 (relativ)")
    p.add_argument("--min-crop", type=int, default=160, help="Minimaler Croprand in Pixeln (quadratisch)")
    p.add_argument("--faces-upscale", type=int, default=192, help="Minimale Kantenlänge für FaceMesh (bei kleineren Crops upscalen)")
    p.add_argument("--imgsz", type=int, default=448)
    p.add_argument("--max-det", type=int, default=20)
    p.add_argument("--use-refine", action="store_true", default=False, help="MediaPipe mit refine_landmarks")
    return p.parse_args()

def main():
    args = parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # YOLO Modell & Device
    yolo = YOLO(str(args.model))
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    yolo.to(device)
    print(f"🖥️  Inference-Device: {device}")

    # Warmup
    try:
        with torch.no_grad():
            dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
            _ = yolo(source=[dummy], imgsz=args.imgsz, verbose=False, device=device)
    except Exception:
        pass

    # Eingabedateien anzeigen
    videos = sorted([*args.input_dir.glob("*.mp4"), *args.input_dir.glob("*.mov"), *args.input_dir.glob("*.mkv")])
    print(f"🔍 Input-Ordner: {args.input_dir.resolve()}")
    if not videos:
        print("⚠️  Keine passenden Videos gefunden.")
        return
    print("📁 Dateien:")
    for p in videos:
        print("  →", p.name)

    outer = None
    if _HAS_TQDM:
        outer = tqdm(total=len(videos), desc="Gesamt", unit="vid", leave=False)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=10,
        refine_landmarks=args.use_refine,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        for vid in videos:
            out = args.output_dir / f"{vid.stem}_faces.json"
            process_video(
                video_path=vid,
                output_path=out,
                model=yolo,
                face_mesh=face_mesh,
                conf_thresh=args.conf_thresh,
                frame_skip=args.frame_skip,
                downscale=args.downscale,
                expansion_1=args.expansion,
                expansion_2=args.expansion2,
                min_crop=args.min_crop,
                faces_upscale=args.faces_upscale,
                imgsz=args.imgsz,
                device=device,
                max_det=args.max_det
            )
            if outer is not None:
                outer.update(1)

    if outer is not None:
        outer.close()

if __name__ == "__main__":
    main()
