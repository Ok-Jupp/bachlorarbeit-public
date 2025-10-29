#!/usr/bin/env python3
# src/reformat/new/main_apply_crop.py
from __future__ import annotations
import logging, json, math, subprocess, argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import sys

import cv2
import numpy as np

# ── Projektwurzel importierbar machen
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import RAW_CLIPS_DIR, FACE_COMBINED_DIR, SEGMENTS_DIR, CROPPED_DIR

# ==== Defaults (per CLI überschreibbar) ======================================
OUT_W_DEFAULT, OUT_H_DEFAULT = 1080, 1920  # 9:16
DEBUG_SCALE_DEFAULT          = 0.6
MEDIAN_WIN_DEFAULT           = 5
EMA_ALPHA_DEFAULT            = 0.22
DEADBAND_PX_DEFAULT          = 8.0
SWITCH_COOLDOWN_FR_DEFAULT   = 19
ZOOM_PAD_FRAC_DEFAULT        = 0.10

USE_CUT_DETECT_DEFAULT       = True
CUT_CORR_THRESH_DEFAULT      = 0.65
CUT_COOLDOWN_DEFAULT         = 6

MUX_AUDIO_DEFAULT            = True
FFMPEG_BIN                   = "ffmpeg"
# ============================================================================

def clamp(v, lo, hi): return max(lo, min(hi, v))

def compute_crop_rect(cx: float, cy: float, src_w: int, src_h: int,
                      out_w: int, out_h: int, zoom_pad_frac: float) -> tuple[int,int,int,int]:
    """9:16 (out_w:out_h) Crop um (cx,cy) — ohne Squeeze, mit Zoom-Pad, im Bild gehalten."""
    target_ar = out_w / out_h
    src_ar = src_w / src_h
    if src_ar >= target_ar:
        base_h = src_h
        base_w = int(round(base_h * target_ar))
    else:
        base_w = src_w
        base_h = int(round(base_w / target_ar))

    desired_scale = 1.0 + zoom_pad_frac
    s = min(desired_scale, src_w / base_w, src_h / base_h)
    w = int(round(base_w * s))
    h = int(round(base_h * s))
    half_w, half_h = w // 2, h // 2

    cx = clamp(cx, half_w, src_w - half_w)
    cy = clamp(cy, half_h, src_h - half_h)
    x = int(round(cx - half_w))
    y = int(round(cy - half_h))
    return x, y, w, h

def draw_center(img, pt, color, label=None):
    if pt is None: return
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(img, (x, y), 6, color, -1)
    if label:
        cv2.putText(img, label, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

def scene_corr(a_small: np.ndarray, b_small: np.ndarray) -> float:
    a_hsv = cv2.cvtColor(a_small, cv2.COLOR_BGR2HSV)
    b_hsv = cv2.cvtColor(b_small, cv2.COLOR_BGR2HSV)
    ha = cv2.calcHist([a_hsv],[0,1],None,[50,50],[0,180,0,256])
    hb = cv2.calcHist([b_hsv],[0,1],None,[50,50],[0,180,0,256])
    cv2.normalize(ha,ha,0,1,cv2.NORM_MINMAX); cv2.normalize(hb,hb,0,1,cv2.NORM_MINMAX)
    return float((cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL) + 1.0)/2.0)

def mux_audio_from_source(src_video: Path, silent_video: Path, out_video: Path):
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(src_video),
        "-i", str(silent_video),
        "-map", "1:v:0",
        "-map", "0:a:0?",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_video),
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def load_faces(name: str) -> List[Dict[str,Any]]:
    p = FACE_COMBINED_DIR / f"{name}_faces.json"
    if not p.exists(): return []
    return json.loads(p.read_text(encoding="utf-8"))

def load_target_map_or_segments(name: str, total_frames: int) -> List[Optional[int] | Dict]:
    """
    Bevorzugt *_target_by_frame.json (Liste Dicts mit t,cx,cy,w,h).
    Fallback: *_segments.json (pro Frame Track-ID).
    Gibt Liste gleicher Länge wie total_frames zurück.
    """
    map_p = SEGMENTS_DIR / f"{name}_target_by_frame.json"
    if map_p.exists():
        target = json.loads(map_p.read_text(encoding="utf-8"))
        # Falls es Dicts sind (cx,cy,w,h pro frame), einfach zurückgeben:
        if target and isinstance(target[0], dict):
            if len(target) < total_frames:
                last = target[-1] if target else {"t":0,"cx":0.5,"cy":0.5,"w":0.6,"h":0.6}
                target += [last] * (total_frames - len(target))
            return target[:total_frames]
        # Falls numerische IDs drin wären, fällt es unten durch auf segs-Logik
    seg_p = SEGMENTS_DIR / f"{name}_segments.json"
    if seg_p.exists():
        segs = json.loads(seg_p.read_text(encoding="utf-8"))
        target_tid = [None]*total_frames
        for s in segs:
            a, b, tid = int(s["start_f"]), int(s["end_f"]), s["track_id"]
            for t in range(max(0,a), min(total_frames, b+1)):
                target_tid[t] = tid
        return target_tid
    return [None]*total_frames

def find_center_for_track(faces_frame: Dict[str,Any], target_tid: Optional[int], fallback: Tuple[float,float]) -> Tuple[float,float]:
    if target_tid is None:
        return fallback
    faces = faces_frame.get("faces", [])
    for f in faces:
        if int(f.get("track_id", -1)) == int(target_tid):
            x,y,w,h = f.get("bbox", [None,None,None,None])
            if None not in (x,y,w,h):
                return (float(x + w/2), float(y + h/2))
    return fallback

def parse_args():
    p = argparse.ArgumentParser(description="Apply 9:16 Auto-Crop auf Rohclips mit Face-/Target-Daten.")
    p.add_argument("--pattern", type=str, default="*.mp4", help="Dateimuster in RAW_CLIPS_DIR (Default: *.mp4)")
    p.add_argument("--out_w", type=int, default=OUT_W_DEFAULT, help="Output-Breite (Default: 1080)")
    p.add_argument("--out_h", type=int, default=OUT_H_DEFAULT, help="Output-Höhe (Default: 1920)")
    p.add_argument("--zoom_pad", type=float, default=ZOOM_PAD_FRAC_DEFAULT, help="Zoom-Pad (0..1, Default 0.10)")
    p.add_argument("--median", type=int, default=MEDIAN_WIN_DEFAULT, help="Median-Fenster (ungerade, >=3)")
    p.add_argument("--ema", type=float, default=EMA_ALPHA_DEFAULT, help="EMA-Alpha (0..1)")
    p.add_argument("--deadband", type=float, default=DEADBAND_PX_DEFAULT, help="Totband in Pixel")
    p.add_argument("--switch_cd", type=int, default=SWITCH_COOLDOWN_FR_DEFAULT, help="Cooldown-Frames nach Trackwechsel")
    p.add_argument("--cut_detect", action="store_true", default=USE_CUT_DETECT_DEFAULT, help="Szenenschnitt-Erkennung aktivieren")
    p.add_argument("--cut_corr", type=float, default=CUT_CORR_THRESH_DEFAULT, help="Korrelation-Schwelle (0..1)")
    p.add_argument("--cut_cd", type=int, default=CUT_COOLDOWN_DEFAULT, help="Cooldown-Frames nach Cut")
    p.add_argument("--mux_audio", action="store_true", default=MUX_AUDIO_DEFAULT, help="Audio vom Original muxen")
    p.add_argument("--debug", action="store_true", help="Debug-Overlay anzeigen (langsam)")
    p.add_argument("--debug_scale", type=float, default=DEBUG_SCALE_DEFAULT, help="Skalierung Debug-Preview")
    p.add_argument("--overwrite", action="store_true", help="Existierende Outputs überschreiben")
    return p.parse_args()

def main():
    args = parse_args()
    OUT_DIR = CROPPED_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
    clips = sorted(list(RAW_CLIPS_DIR.glob(args.pattern)))
    if not clips:
        print(f"⚠️  Keine Clips in {RAW_CLIPS_DIR} mit Pattern '{args.pattern}'")
        return

    print(f"🔎 {len(clips)} Clips gefunden …")
    for video_path in clips:
        name = video_path.stem
        out_path = OUT_DIR / f"{name}_9x16.mp4"
        if out_path.exists() and not args.overwrite:
            print(f"⏭️  Skip (existiert): {out_path.name}")
            continue

        # Video öffnen
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Kann Video nicht öffnen: {video_path.name}")
            continue
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Face/Target laden
        faces_all = load_faces(name)
        if faces_all and len(faces_all) < total:
            faces_all += [ {"faces": [], "W": width, "H": height} ] * (total - len(faces_all))
        target_by_frame = load_target_map_or_segments(name, total)

        # Writer vorbereiten
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (args.out_w, args.out_h))

        median_buf = deque(maxlen=max(3, args.median if args.median % 2 else args.median+1))
        ema_center: Optional[Tuple[float,float]] = None
        last_center: Optional[Tuple[float,float]] = (width/2, height/2)
        switch_cooldown = 0

        prev_small = None
        cut_cd = 0

        print(f"🎞️  Apply: {name}  src={width}x{height}  fps={fps:.2f}  frames={total}")

        for t in range(total):
            ret, frame = cap.read()
            if not ret: break

            # Ziel bestimmen:
            desired = None
            tgt = target_by_frame[t] if t < len(target_by_frame) else None

            # Fall A: target_by_frame.json mit direkten Zentren (Dict)
            if isinstance(tgt, dict) and all(k in tgt for k in ("cx","cy","w","h")):
                desired = (float(tgt["cx"])*width, float(tgt["cy"])*height)
            else:
                # Fall B: numerische Track-ID
                target_tid = tgt if tgt is None or isinstance(tgt, (int, float)) else None
                faces_fr = faces_all[t] if (faces_all and t < len(faces_all)) else {"faces":[]}
                desired = find_center_for_track(faces_fr, target_tid, last_center or (width/2, height/2))

            # Szenenschnitt?
            if args.cut_detect:
                small = cv2.resize(frame, (128, 72))
                if prev_small is not None:
                    corr = scene_corr(prev_small, small)
                    if corr < args.cut_corr:
                        ema_center = desired
                        last_center = desired
                        switch_cooldown = args.switch_cd
                        cut_cd = args.cut_cd
                prev_small = small

            # Median-Filter
            median_buf.append(desired)
            if len(median_buf) >= 3:
                xs = sorted(p[0] for p in median_buf)
                ys = sorted(p[1] for p in median_buf)
                m  = len(median_buf)//2
                desired_f = (xs[m], ys[m])
            else:
                desired_f = desired

            # Trackwechsel erkennen (nur bei Track-IDs sauber möglich)
            if t > 0:
                prev_tgt = target_by_frame[t-1] if t-1 < len(target_by_frame) else None
            else:
                prev_tgt = tgt
            is_switch = (not isinstance(tgt, dict)) and (tgt != prev_tgt)

            if ema_center is None:
                ema_center = desired_f
            if last_center is None:
                last_center = desired_f

            if is_switch:
                ema_center  = desired_f
                last_center = desired_f
                switch_cooldown = args.switch_cd
            else:
                dx = desired_f[0] - ema_center[0]
                dy = desired_f[1] - ema_center[1]
                dist = math.hypot(dx, dy)
                if cut_cd > 0:
                    ema_center = desired_f
                    cut_cd -= 1
                else:
                    if dist > args.deadband:
                        ema_center = (ema_center[0] + dx*args.ema,
                                      ema_center[1] + dy*args.ema)

            last_center = desired_f

            # 9:16 Crop anwenden
            x, y, w, h = compute_crop_rect(ema_center[0], ema_center[1], width, height,
                                           args.out_w, args.out_h, args.zoom_pad)
            cropped = frame[y:y+h, x:x+w]
            if cropped.size == 0: cropped = frame
            final = cv2.resize(cropped, (args.out_w, args.out_h), interpolation=cv2.INTER_AREA)
            writer.write(final)

            if args.debug:
                dbg = frame.copy()
                cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 0, 255), 2)
                #draw_center(dbg, desired,    (128,128,255), "desired")
                #draw_center(dbg, desired_f,  (255,255,  0), "median")
                #draw_center(dbg, ema_center, (  0,255,255), "ema")
                cv2.putText(dbg, f"t={t+1}/{total}", (12, height-14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,220,20), 2, cv2.LINE_AA)
                disp = cv2.resize(dbg, (int(width*args.debug_scale), int(height*args.debug_scale)))
                cv2.imshow("Apply Debug", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("🛑 Abgebrochen (q).")
                    break

        writer.release()
        cap.release()

        # Audio muxen?
        if args.mux_audio:
            tmp = out_path.with_suffix(".tmp.mp4")
            try:
                out_path.rename(tmp)
                mux_audio_from_source(video_path, tmp, out_path)
            finally:
                if tmp.exists():
                    try: tmp.unlink()
                    except: pass
            print(f"✅ Fertig (mit Audio): {out_path.name}")
        else:
            print(f"✅ Fertig: {out_path.name}")

    if args.debug:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
