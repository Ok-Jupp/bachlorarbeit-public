#!/usr/bin/env python3
# viz_crop_debug_reasons_lean.py — Lean-Scorer: nur EMA-Vergleich + minimalistisches Debug-Overlay

from __future__ import annotations
import json, cv2
from pathlib import Path
import sys
from typing import Dict, List, Optional

# ---- zentrale Pfade aus config.py laden ----
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import RAW_CLIPS_DIR, FACE_COMBINED_DIR, SEGMENTS_DIR

# Zielordner für Debug-Videos
DEBUG_VIZ_DIR = RAW_CLIPS_DIR / "debug_viz"
DEBUG_VIZ_DIR.mkdir(parents=True, exist_ok=True)

# === Lean-Parameter (müssen zu deinem build_target_by_frame(lean) passen) ===
MOUTH_MIN    = 5.0     # unterhalb = Rauschen
ALPHA_EMA    = 0.60    # EMA-Gewicht (höher = schneller)
DELTA_RATIO  = 0.30    # relativer Vorsprung (z. B. +30%)
COOLDOWN_S   = 0.40    # fixer Cooldown nach Wechsel
DOMINANCE_S  = 0.25    # so lange muss "best" vorne liegen
TAIL_LOCK_S  = 0.80    # am Ende keine Wechsel mehr

# Anzeige-Optionen (minimal halten)
SHOW_OTHERS  = False   # False = nur gewählten Sprecher labeln (andere dünn grau)
THICK_CHOSEN = 3       # Linienstärke für gewählten Sprecher
THIN_OTHER   = 1       # Linienstärke für andere Gesichter

def load_json_safe(p: Path):
    if not p.exists():
        print(f"⚠️  Datei fehlt: {p.name}")
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Fehler beim Lesen {p.name}: {e}")
        return None

def draw_box(img, box, color, label="", thickness=2):
    x, y, w, h = [int(round(v)) for v in box[:4]]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        y1 = max(0, y - th - 6)
        cv2.rectangle(img, (x, y1), (x + tw + 10, y), color, -1)
        cv2.putText(img, label, (x + 5, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

def draw_crop_overlay(frame, cx, cy, w, h, color=(0, 255, 255)):
    """Zeichnet die Crop-Box aus target_by_frame (cx,cy,w,h auf 0..1)."""
    H, W = frame.shape[:2]
    box_w = int(w * W)
    box_h = int(h * H)
    cx_px = int(cx * W)
    cy_px = int(cy * H)
    x1 = max(0, cx_px - box_w // 2)
    y1 = max(0, cy_px - box_h // 2)
    x2 = min(W - 1, x1 + box_w)
    y2 = min(H - 1, y1 + box_h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Crop-Mitte markieren (kleiner Punkt)
    cv2.circle(frame, (cx_px, cy_px), 4, color, -1)
    return (x1, y1, x2 - x1, y2 - y1)

def process_video(video_path: Path):
    stem = video_path.stem
    faces_path  = FACE_COMBINED_DIR / f"{stem}_faces.json"
    target_path = SEGMENTS_DIR      / f"{stem}_target_by_frame.json"
    out_path = DEBUG_VIZ_DIR / f"{stem}_debug_viz.mp4"

    faces_data  = load_json_safe(faces_path)
    target_data = load_json_safe(target_path)
    if not isinstance(faces_data, list) or not isinstance(target_data, list):
        print(f"⚠️  Fehlen Daten für {stem}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Kann Video {video_path.name} nicht öffnen.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    print(f"🎥 Visualisierung: {out_path.name}")

    # Zustände wie im Lean-Scorer
    ema_mouth: Dict[int, float] = {}
    last_track: Optional[int]   = None
    last_switch_i: int          = -10_000_000
    dom_tid: Optional[int]      = None
    dom_count: int              = 0

    tail_lock_from = max(0, total - int(round(TAIL_LOCK_S * fps)))
    NEED_DOM_FRAMES = max(2, int(round(DOMINANCE_S * fps)))
    COOLDOWN_FR = int(round(COOLDOWN_S * fps))

    for i in range(min(total, len(faces_data), len(target_data))):
        ret, frame = cap.read()
        if not ret:
            break

        fr = faces_data[i]
        faces = fr.get("faces", [])

        # Kandidatenliste aufbauen & EMA aktualisieren
        candidates = []
        for f in faces:
            x, y, w, h = f.get("bbox", [0, 0, 0, 0])
            tid = int(f.get("track_id", -1))
            mouth_raw = float(f.get("mouth_prob", f.get("mouth_openness", 0.0)))
            raw = 0.0 if mouth_raw < MOUTH_MIN else mouth_raw
            prev = ema_mouth.get(tid, raw)
            ema = ALPHA_EMA * raw + (1.0 - ALPHA_EMA) * prev
            ema_mouth[tid] = ema
            candidates.append({
                "tid": tid, "bbox": [x, y, w, h],
                "mouth_raw": mouth_raw,
                "mouth_ema": ema
            })

        cur_tid = last_track if last_track is not None else -1

        # "best" NUR nach EMA; bei Gleichstand: Current bevorzugen, sonst kleinster tid
        best = None
        for item in sorted(candidates, key=lambda z: (-z["mouth_ema"], z["tid"])):
            if best is None:
                best = item
            elif item["mouth_ema"] == best["mouth_ema"]:
                # Wenn Current mit gleichem EMA dabei ist -> bevorzugen
                if item["tid"] == cur_tid:
                    best = item
        # Dominanzfenster zählen, falls Best != Current
        if best and best["tid"] != cur_tid:
            if dom_tid == best["tid"]:
                dom_count += 1
            else:
                dom_tid = best["tid"]
                dom_count = 1
        else:
            dom_tid = None
            dom_count = 0

        # Wechselbedingungen (Lean): nur EMA-REL-Vorsprung + Dominanz + Cooldown + Tail-Lock
        can_switch = (i - last_switch_i) >= COOLDOWN_FR and (i < tail_lock_from)
        best_ema = best["mouth_ema"] if best else 0.0
        cur_ema  = ema_mouth.get(cur_tid, 0.0)
        rel_factor = (best_ema / max(1e-6, cur_ema)) if cur_tid != -1 else float("inf")
        rel_ok   = (cur_ema > 0.0) and (rel_factor >= (1.0 + DELTA_RATIO))
        dom_ok   = (dom_count >= NEED_DOM_FRAMES)

        chosen_tid = cur_tid
        switched = False
        if best:
            if cur_tid == -1:
                chosen_tid = best["tid"]
                last_switch_i = i
                switched = True
            elif best["tid"] == cur_tid:
                chosen_tid = cur_tid
            else:
                if can_switch and rel_ok and dom_ok:
                    chosen_tid = best["tid"]
                    last_switch_i = i
                    switched = True
                else:
                    chosen_tid = cur_tid

        last_track = chosen_tid if chosen_tid != -1 else last_track

        # Crop aus target_by_frame zeichnen (gelb)
        tgt = target_data[i]
        cx = float(tgt.get("cx", 0.5))
        cy = float(tgt.get("cy", 0.5))
        ww = float(tgt.get("w", 0.6))
        hh = float(tgt.get("h", 0.6))
        draw_crop_overlay(frame, cx, cy, ww, hh)

        # Gesichter zeichnen: gewählter Sprecher grün + Label; andere optional dünn grau, ohne Label
        for item in candidates:
            tid = item["tid"]
            x, y, w, h = item["bbox"]
            if tid == chosen_tid:
                color = (0, 255, 0)
                lbl = f"id:{tid}  raw:{item['mouth_raw']:.1f}  ema:{item['mouth_ema']:.1f}"
                draw_box(frame, [x, y, w, h], color, lbl, thickness=THICK_CHOSEN)
            elif SHOW_OTHERS:
                draw_box(frame, [x, y, w, h], (180, 180, 180), "", thickness=THIN_OTHER)

        # Debug-Header (knapp & entscheidungsrelevant)
        cooldown_left = max(0, COOLDOWN_FR - (i - last_switch_i))
        header = (f"chosen:{chosen_tid} | best:{best['tid'] if best else '-'} "
                  f"EMA(best:{best_ema:.1f}  cur:{cur_ema:.1f}  x{rel_factor:.2f}) | "
                  f"cooldown:{cooldown_left} | dom:{dom_count}/{NEED_DOM_FRAMES} | "
                  f"switch:{int(switched)}")
        cv2.putText(frame, header, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 220, 40), 2, cv2.LINE_AA)

        # Fortschritt
        cv2.putText(frame, f"{i+1}/{total}", (12, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"✅ Fertig: {out_path.name}")

def main():
    vids = sorted(RAW_CLIPS_DIR.glob("highlight_*.mp4"))
    if not vids:
        print(f"⚠️ Keine Videos in {RAW_CLIPS_DIR} gefunden.")
        return
    print(f"🔎 {len(vids)} Videos gefunden.")
    for v in vids:
        process_video(v)
    print("\n🎉 Alle Lean-Debug-Videos erstellt!")

if __name__ == "__main__":
    main()