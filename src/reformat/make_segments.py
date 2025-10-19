#!/usr/bin/env python3
# make_segments.py — erzeugt pro Highlight eine Zielspur (target_by_frame.json) fürs Cropping

from __future__ import annotations
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

# ── Projektwurzel in sys.path aufnehmen (dieses Skript liegt z. B. unter src/reformat/)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import RAW_CLIPS_DIR, FACE_COMBINED_DIR, FACE_CROP_CENTERS, SEGMENTS_DIR

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_OK = True
except Exception:
    MOVIEPY_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsstrukturen
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceDet:
    t: float             # Sekunden
    cx: float            # Zentrum x (0..1)
    cy: float            # Zentrum y (0..1)
    w: float             # Breite rel. (0..1)
    h: float             # Höhe rel. (0..1)
    track_id: Optional[int] = None
    mouth_prob: Optional[float] = None

def moving_average(xs: List[float], win: int) -> List[float]:
    if win <= 1 or len(xs) <= 2:
        return xs[:]
    # ungerade Fensterbreite erzwingen
    win = win if win % 2 == 1 else win + 1
    r = win // 2
    out = []
    for i in range(len(xs)):
        a = max(0, i - r)
        b = min(len(xs), i + r + 1)
        out.append(sum(xs[a:b]) / (b - a))
    return out

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ──────────────────────────────────────────────────────────────────────────────
# Lesen möglicher Eingabeformate (robust, schema-tolerant)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_face_like(obj: Dict, t: float, W: float | None = None, H: float | None = None) -> FaceDet:
    """
    Erwartet entweder:
      - bbox=[x,y,w,h] in Pixel → wird via W,H auf 0..1 normiert
      - oder bereits normierte Felder cx,cy,w,h in 0..1
    Optional: track_id, mouth_prob / mouth_open / talking_prob
    """
    if "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) >= 4:
        x, y, w, h = [float(v) for v in obj["bbox"][:4]]
        if W and H and W > 0 and H > 0:
            cx = (x + w * 0.5) / W
            cy = (y + h * 0.5) / H
            w  = w / W
            h  = h / H
        else:
            # Falls Maße fehlen: best effort, danach clampen
            cx = x + w * 0.5
            cy = y + h * 0.5
        cx, cy = clamp01(cx), clamp01(cy)
        w, h = max(0.0, min(1.0, w)), max(0.0, min(1.0, h))
    else:
        cx = float(obj.get("cx", 0.5))
        cy = float(obj.get("cy", 0.5))
        w  = float(obj.get("w",  0.3))
        h  = float(obj.get("h",  0.3))
        cx, cy = clamp01(cx), clamp01(cy)
        w, h = max(0.0, min(1.0, w)), max(0.0, min(1.0, h))

    track_id   = obj.get("track_id")
    mouth_prob = obj.get("mouth_prob") or obj.get("mouth_open") or obj.get("talking_prob")
    mouth_prob = None if mouth_prob is None else float(mouth_prob)

    return FaceDet(t=t, cx=cx, cy=cy, w=w, h=h, track_id=track_id, mouth_prob=mouth_prob)


def load_faces_or_centers(stem: str, fps_hint: float | None = None) -> List[FaceDet]:
    """
    Lädt die beste verfügbare Gesichts/Center-Quelle für ein Highlight.
    Suchreihenfolge:
      1) FACE_COMBINED_DIR/{stem}_faces.json  (Liste von Frames mit 'faces')
      2) FACE_CROP_CENTERS/{stem}_centers.json
         - akzeptiert entweder [[cx,cy], ...] oder [{t,cx,cy,w,h}, ...]
    """
    candidates = [
        (FACE_COMBINED_DIR / f"{stem}_faces.json", "faces"),
        (FACE_CROP_CENTERS / f"{stem}_centers.json", "centers"),
    ]
    path = kind = None
    for p, k in candidates:
        if p.exists():
            path, kind = p, k
            break

    if path is None:
        print(f"⚠️ Keine Face/Centers-Datei gefunden für {stem}. Fallback später → (0.5,0.5).")
        return []

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:
        print(f"❌ Konnte {path.name} nicht lesen: {e}")
        return []

    dets: List[FaceDet] = []

    # 1) Liste von Frames: [{ "W":..,"H":..,"timestamp"/"t":.., "faces":[...] }, ...]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "faces" in data[0]:
        for fr in data:
            W = float(fr.get("W") or 0.0)
            H = float(fr.get("H") or 0.0)
            t = float(fr.get("t") or fr.get("timestamp") or fr.get("time") or 0.0)
            for f in fr.get("faces", []):
                dets.append(_parse_face_like(f, t, W, H))

    # 2) Dict mit "frames": [...]
    elif isinstance(data, dict) and "frames" in data:
        for fr in data["frames"]:
            W = float(fr.get("W") or 0.0)
            H = float(fr.get("H") or 0.0)
            t = float(fr.get("t") or fr.get("timestamp") or fr.get("time") or 0.0)
            for f in fr.get("faces", []):
                dets.append(_parse_face_like(f, t, W, H))

    # 3) centers.json als Liste von Listen: [[cx,cy], ...]
    elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
        fps = float(fps_hint or 25.0)
        for i, pair in enumerate(data):
            cx, cy = float(pair[0]), float(pair[1])
            dets.append(FaceDet(t=i / fps, cx=clamp01(cx), cy=clamp01(cy), w=0.6, h=0.6))

    # 4) Liste von Dicts mit evtl. bereits normierten Feldern
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        for item in data:
            t = float(item.get("t") or item.get("time") or 0.0)
            dets.append(_parse_face_like(item, t))

    else:
        print(f"⚠️ Unbekanntes JSON-Format in {path.name}.")
        return []

    # filtern & sortieren
    dets = [d for d in dets if 0.0 <= d.cx <= 1.0 and 0.0 <= d.cy <= 1.0]
    dets.sort(key=lambda d: d.t)
    print(f"✅ {len(dets)} Detektionen aus {path.name} ({kind}).")
    return dets


# ──────────────────────────────────────────────────────────────────────────────
# Zielspur berechnen
# ──────────────────────────────────────────────────────────────────────────────

def build_target_by_frame(
    faces: List[FaceDet],
    duration: float,
    fps: float,
    smooth_win: int = 7
) -> List[Dict]:
    """
    Wählt pro Frame eine Zielposition (cx,cy,w,h).
    Heuristik:
      - bevorzuge Gesicht mit höchster mouth_prob (wenn vorhanden),
      - sonst größtes Bounding-Box-Areal (w*h),
      - halte IDs stabil (nicht zu häufige Sprünge).
    Anschließend leichte Glättung (Moving Average) der Zentren/Größen.
    """
    if fps <= 0:
        fps = 25.0
    total_frames = max(1, int(round(duration * fps)))
    if not faces:
        # Fallback: center track
        return [{"frame": i, "t": round(i / fps, 4), "cx": 0.5, "cy": 0.5, "w": 0.6, "h": 0.6} for i in range(total_frames)]

    frame_targets: List[Tuple[float, float, float, float]] = []  # (cx, cy, w, h)
    last_track: Optional[int] = None

    # lineare Suche über faces (bei Bedarf später bucketisieren)
    for i in range(total_frames):
        t = i / fps
        lo, hi = t - 1.0 / fps, t + 1.0 / fps

        cand: List[FaceDet] = [d for d in faces if lo <= d.t <= hi]
        if not cand:
            # Nimm den zeitlich nächsten
            nearest = min(faces, key=lambda d: abs(d.t - t))
            cand = [nearest]

        def score(d: FaceDet) -> Tuple[float, float, float]:
            mouth = -1.0 if d.mouth_prob is None else float(d.mouth_prob)  # None schlechter als 0
            area  = float(d.w) * float(d.h)
            stable = 1.0 if (last_track is not None and d.track_id == last_track) else 0.0
            return (mouth, area, stable)

        cand.sort(key=score, reverse=True)
        best = cand[0]
        if best.track_id is not None:
            last_track = best.track_id
        frame_targets.append((best.cx, best.cy, best.w, best.h))

    # Glätten
    cxs = moving_average([c for c, _, _, _ in frame_targets], smooth_win)
    cys = moving_average([c for _, c, _, _ in frame_targets], smooth_win)
    ws  = moving_average([w for *_, w, _ in frame_targets], max(3, smooth_win // 2))
    hs  = moving_average([h for *_, _, h in frame_targets], max(3, smooth_win // 2))

    out = []
    for i, (cx, cy, w, h) in enumerate(zip(cxs, cys, ws, hs)):
        t = i / fps
        out.append({
            "frame": i,
            "t": round(t, 4),
            "cx": round(clamp01(cx), 4),
            "cy": round(clamp01(cy), 4),
            "w": round(max(0.05, min(1.0, w)), 4),
            "h": round(max(0.05, min(1.0, h)), 4),
        })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def write_target_json(stem: str, target: List[Dict]) -> Path:
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SEGMENTS_DIR / f"{stem}_target_by_frame.json"
    out_path.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI / Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Erzeugt target_by_frame.json aus Face/Center-Detektionen für Cropping.")
    p.add_argument("--pattern", type=str, default="highlight_*.mp4", help="Dateimuster in RAW_CLIPS_DIR (Default: highlight_*.mp4)")
    p.add_argument("--fps", type=float, default=0.0, help="FPS erzwingen (0 = aus Video lesen).")
    p.add_argument("--smooth", type=int, default=7, help="Fensterbreite für Moving-Average-Glättung (ungerade).")
    p.add_argument("--overwrite", action="store_true", help="Existierende target_by_frame.json überschreiben.")
    return p.parse_args()


def main():
    if not MOVIEPY_OK:
        raise RuntimeError("moviepy ist nicht installiert. Bitte `pip install moviepy` ausführen.")

    args = parse_args()

    vids = sorted(RAW_CLIPS_DIR.glob(args.pattern))
    if not vids:
        print(f"⚠️ Keine Rohclips gefunden in {RAW_CLIPS_DIR} mit Pattern '{args.pattern}'.")
        return

    print(f"🔎 Finde {len(vids)} Clips …")

    for vid in vids:
        stem = vid.stem  # z. B. highlight_3
        out_json = SEGMENTS_DIR / f"{stem}_target_by_frame.json"
        if out_json.exists() and not args.overwrite:
            print(f"⏭️  {out_json.name} existiert bereits – überspringe (nutze --overwrite zum Ersetzen).")
            continue

        # Video-Metadaten
        try:
            with VideoFileClip(str(vid)) as V:
                duration = float(V.duration or 0.0)
                fps = float(args.fps or (V.fps or 25.0))
        except Exception as e:
            print(f"❌ Kann Video {vid.name} nicht öffnen: {e} – Fallback duration/fps (10s/25fps).")
            duration, fps = 10.0, (args.fps or 25.0)

        # Face/Centers laden (fps_hint durchreichen, wichtig für centers-Listen)
        faces = load_faces_or_centers(stem, fps_hint=fps)

        # Zielspur bauen
        target = build_target_by_frame(faces, duration=duration, fps=fps, smooth_win=args.smooth)

        # Schreiben
        out = write_target_json(stem, target)
        print(f"💾 geschrieben: {out}")

    print("🎉 Fertig.")


if __name__ == "__main__":
    main()
