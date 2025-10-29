#!/usr/bin/env python3
# make_segments.py — erzeugt pro Highlight eine Zielspur (target_by_frame.json) fürs Cropping
# Lean, stabil, ohne Overkill:
# - Score = EMA(mouth_prob)
# - Wechsel nur bei REL-Vorsprung + Dominanzfenster + Cooldown + kein Tail-Switch
# - Fi-exaktes Matching (falls vorhanden), sonst Zeit-Fallback
# - Ausgabe enthält "tid" pro Frame

from __future__ import annotations
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterable
from pathlib import Path
import bisect
import sys

# ── Projektwurzel in sys.path aufnehmen
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import RAW_CLIPS_DIR, FACE_COMBINED_DIR, FACE_CROP_CENTERS, SEGMENTS_DIR

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_OK = True
except Exception:
    MOVIEPY_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# TUNING (schlank & zuverlässig)
# ──────────────────────────────────────────────────────────────────────────────
MOUTH_MIN       = 5.0    # unterhalb = Rauschen raus
ALPHA_EMA       = 0.65   # EMA-Gewicht (höher = schneller)
ZERO_DECAY      = 0.50   # bei Ausfall: ema = decay*ema_alt (stabilisiert)
DELTA_REL       = 0.30   # relativer Vorsprung: best_ema >= cur_ema*(1+DELTA_REL)
DOMINANCE_S     = 0.4   # best muss so lange vorne liegen (Sek.)
COOLDOWN_S      = 0.40   # fester Cooldown nach Wechsel (Sek.)
TAIL_LOCK_S     = 0.80   # letzte s ohne Wechsel
SMOOTH_WIN_DEF  = 7      # Moving-Average-Fenster für (cx,cy), w/h halb so stark

# Zeit-Fallback-Fenster, wenn kein frame-exakter Treffer existiert
FALLBACK_WIN_S  = 1.0/25.0   # ~±1 Frame bei 25 fps


# ──────────────────────────────────────────────────────────────────────────────
# Datenstrukturen
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FaceDet:
    t: float                 # Sekunden
    cx: float                # Zentrum x (0..1)
    cy: float                # Zentrum y (0..1)
    w: float                 # Breite rel. (0..1)
    h: float                 # Höhe rel. (0..1)
    track_id: Optional[int] = None
    mouth_prob: Optional[float] = None
    fi: Optional[int] = None  # Frameindex, falls im JSON vorhanden


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def moving_average(xs: List[float], win: int) -> List[float]:
    if win <= 1 or len(xs) <= 2:
        return xs[:]
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
# Robustes Parsen
# ──────────────────────────────────────────────────────────────────────────────
def _parse_face_like(obj: Dict, t: float, W: float | None = None, H: float | None = None, fi: Optional[int] = None) -> FaceDet:
    """
    Erwartet entweder:
      - bbox=[x,y,w,h] in Pixel → normiert via W,H auf 0..1
      - oder bereits normierte Felder cx,cy,w,h in 0..1
    Optional: track_id, mouth_prob / mouth_open / talking_prob / mouth_openness
    Optional: fi (frame index)
    """
    if "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) >= 4:
        x, y, w, h = [float(v) for v in obj["bbox"][:4]]
        if W and H and W > 0 and H > 0:
            cx = (x + w * 0.5) / W
            cy = (y + h * 0.5) / H
            w  = w / W
            h  = h / H
        else:
            cx = x + w * 0.5
            cy = y + h * 0.5
        cx, cy = clamp01(cx), clamp01(cy)
        w, h = max(0.0, min(1.0, w)), max(0.0, min(1.0, h))
    else:
        cx = float(obj.get("cx", 0.5)); cy = float(obj.get("cy", 0.5))
        w  = float(obj.get("w",  0.3)); h  = float(obj.get("h",  0.3))
        cx, cy = clamp01(cx), clamp01(cy)
        w, h = max(0.0, min(1.0, w)), max(0.0, min(1.0, h))

    track_id   = obj.get("track_id")
    mouth_prob = obj.get("mouth_prob") or obj.get("mouth_open") or obj.get("talking_prob") or obj.get("mouth_openness")
    mouth_prob = None if mouth_prob is None else float(mouth_prob)
    return FaceDet(t=t, cx=cx, cy=cy, w=w, h=h, track_id=track_id, mouth_prob=mouth_prob, fi=fi)


def load_faces_or_centers(stem: str, fps_hint: float | None = None) -> List[FaceDet]:
    """
    Lädt die beste verfügbare Gesichts/Center-Quelle für ein Highlight.
    Reihenfolge:
      1) FACE_COMBINED_DIR/{stem}_faces.json  (Liste von Frames mit 'faces')
      2) FACE_CROP_CENTERS/{stem}_centers.json
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
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Konnte {path.name} nicht lesen: {e}")
        return []

    dets: List[FaceDet] = []

    # 1) Liste von Frames: [{ "W","H","frame","timestamp"/"t", "faces":[...] }, ...]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "faces" in data[0]:
        for fr in data:
            W = float(fr.get("W") or 0.0)
            H = float(fr.get("H") or 0.0)
            t = float(fr.get("t") or fr.get("timestamp") or fr.get("time") or 0.0)
            fi = fr.get("frame")
            for f in fr.get("faces", []):
                dets.append(_parse_face_like(f, t, W, H, fi=fi))

    # 2) Dict mit "frames": [...]
    elif isinstance(data, dict) and "frames" in data:
        for fr in data["frames"]:
            W = float(fr.get("W") or 0.0)
            H = float(fr.get("H") or 0.0)
            t = float(fr.get("t") or fr.get("timestamp") or fr.get("time") or 0.0)
            fi = fr.get("frame")
            for f in fr.get("faces", []):
                dets.append(_parse_face_like(f, t, W, H, fi=fi))

    # 3) centers.json als Liste von Listen: [[cx,cy], ...]
    elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
        fps = float(fps_hint or 25.0)
        for i, pair in enumerate(data):
            cx, cy = float(pair[0]), float(pair[1])
            dets.append(FaceDet(t=i / fps, cx=clamp01(cx), cy=clamp01(cy), w=0.6, h=0.6, fi=i))

    # 4) Liste von Dicts mit evtl. bereits normierten Feldern
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        for i, item in enumerate(data):
            t = float(item.get("t") or item.get("time") or 0.0)
            fi = item.get("frame", i)
            dets.append(_parse_face_like(item, t, None, None, fi=fi))

    else:
        print(f"⚠️ Unbekanntes JSON-Format in {path.name}.")
        return []

    # filtern & sortieren
    dets = [d for d in dets if 0.0 <= d.cx <= 1.0 and 0.0 <= d.cy <= 1.0]
    dets.sort(key=lambda d: d.t)
    print(f"✅ {len(dets)} Detektionen aus {path.name} ({kind}).")
    return dets


# ──────────────────────────────────────────────────────────────────────────────
# Zielspur berechnen (LEAN, stabil)
# ──────────────────────────────────────────────────────────────────────────────
def build_target_by_frame(
    faces: List[FaceDet],
    duration: float,
    fps: float,
    smooth_win: int = SMOOTH_WIN_DEF
) -> List[Dict]:
    if fps <= 0:
        fps = 25.0
    total_frames = max(1, int(round(duration * fps)))

    # Fallback: kein Gesicht
    if not faces:
        return [{"frame": i, "t": round(i/fps, 4), "cx": 0.5, "cy": 0.5, "w": 0.6, "h": 0.6, "tid": -1}
                for i in range(total_frames)]

    # ---- Indexe für frame-exaktes Matching / Fallback
    by_fi: Dict[int, List[FaceDet]] = {}
    times_sorted = sorted(d.t for d in faces)
    for d in faces:
        if d.fi is not None:
            by_fi.setdefault(int(d.fi), []).append(d)

    def candidates_for_frame(i: int) -> List[FaceDet]:
        if i in by_fi:
            return by_fi[i]
        t = i / fps
        lo, hi = t - FALLBACK_WIN_S, t + FALLBACK_WIN_S
        cand = [d for d in faces if lo <= d.t <= hi]
        return cand if cand else [min(faces, key=lambda d: abs(d.t - t))]

    # ---- Parameter in Frames
    NEED_DOM_FR = max(2, int(round(DOMINANCE_S * fps)))
    COOL_FR     = max(1, int(round(COOLDOWN_S  * fps)))
    TAIL_LOCK_I = max(0, total_frames - int(round(TAIL_LOCK_S * fps)))
    PRE_ROLL_S  = 0.20   # <<<<< HIER anpassen: um wie viele Sekunden rückdatieren?
    PRE_ROLL_FR = max(0, int(round(PRE_ROLL_S * fps)))

    # ---- Zustände (Lean-Logik)
    ema: Dict[int, float] = {}
    last_tid: Optional[int] = None
    last_switch_i = -10_000_000
    dom_tid: Optional[int] = None
    dom_count = 0

    # Wir sammeln nur die *entscheidungs*-Wechselpunkte, backdaten später:
    switch_points: List[Tuple[int, int]] = []  # (frame_i, new_tid)
    # Für die Boxen brauchen wir pro Frame die Kandidaten (für spätere Rekonstruktion):
    cand_per_frame: List[List[FaceDet]] = [None] * total_frames  # type: ignore

    def raw_mouth(d: FaceDet) -> float:
        v = float(d.mouth_prob or 0.0)
        return v if v >= MOUTH_MIN else 0.0

    for i in range(total_frames):
        cand = candidates_for_frame(i)
        cand_per_frame[i] = cand

        # EMA update pro Kandidat
        for d in cand:
            tid = d.track_id if d.track_id is not None else -1
            r = raw_mouth(d)
            prev = ema.get(tid, r)
            m_in = (ZERO_DECAY * prev) if r == 0.0 else r
            ema[tid] = ALPHA_EMA * m_in + (1.0 - ALPHA_EMA) * prev

        cur_tid = last_tid if last_tid is not None else -1

        # best nach EMA, Tie-Breaker: Fläche
        def key_best(x: FaceDet):
            tid = x.track_id if x.track_id is not None else -1
            return (ema.get(tid, 0.0), float(x.w) * float(x.h))

        best = max(cand, key=key_best)
        best_tid = best.track_id if best.track_id is not None else -1
        best_ema = float(ema.get(best_tid, 0.0))
        cur_ema  = float(ema.get(cur_tid, 0.0))

        # Dominanzfenster zählen
        if best_tid != cur_tid:
            if dom_tid == best_tid:
                dom_count += 1
            else:
                dom_tid = best_tid
                dom_count = 1
        else:
            dom_tid = None
            dom_count = 0

        can_switch   = (i - last_switch_i) >= COOL_FR and (i < TAIL_LOCK_I)
        rel_ok       = (cur_ema > 0.0) and (best_ema >= cur_ema * (1.0 + DELTA_REL))
        dom_ok       = (dom_count >= NEED_DOM_FR)
        cur_in_view  = any((d.track_id if d.track_id is not None else -1) == cur_tid for d in cand)

        if last_tid is None:
            last_tid = best_tid
            last_switch_i = i
            switch_points.append((i, best_tid))
        else:
            if best_tid == cur_tid:
                continue  # kein Wechsel nötig
            # Wechsel zulassen?
            if (can_switch and rel_ok and dom_ok) or not cur_in_view:
                last_tid = best_tid
                last_switch_i = i
                switch_points.append((i, best_tid))

    # ---- Backdating (Pre-Roll): Schiebe jeden Switch um PRE_ROLL_FR zurück, aber:
    # - nicht vor 0
    # - nicht vor das Ende des vorherigen Cooldowns
    shifted: List[Tuple[int, int]] = []
    prev_lock_end = -1
    for idx, (i_sw, tid_sw) in enumerate(switch_points):
        j = max(0, i_sw - PRE_ROLL_FR)
        j = max(j, prev_lock_end + 1)
        shifted.append((j, tid_sw))
        prev_lock_end = j + COOL_FR - 1  # nächster Switch darf erst danach beginnen

    # ---- finalen tid pro Frame aus den (verschobenen) Switches ableiten
    tid_per_frame = [-1] * total_frames
    if shifted:
        # Vor dem ersten Switch: nichts gewählt → wir nehmen den ersten tid
        first_i, first_tid = shifted[0]
        for k in range(0, first_i):
            tid_per_frame[k] = first_tid
        # Intervalle
        for (a_i, a_tid), nxt in zip(shifted, shifted[1:]):
            b_i = nxt[0]
            for k in range(a_i, min(b_i, total_frames)):
                tid_per_frame[k] = a_tid
        # ab letztem Switch
        last_i, last_t = shifted[-1]
        for k in range(last_i, total_frames):
            tid_per_frame[k] = last_t
    else:
        # Falls nie geswitched wurde, nimm den letzten last_tid (oder -1)
        fill_tid = last_tid if last_tid is not None else -1
        tid_per_frame = [fill_tid] * total_frames

    # ---- Boxen pro Frame zur gewählten tid wählen (frame-exakt, sonst Fallback)
    def pick_for_tid(i: int, want_tid: int) -> FaceDet:
        cand = cand_per_frame[i] or candidates_for_frame(i)
        same = [d for d in cand if (d.track_id if d.track_id is not None else -1) == want_tid]
        if same:
            return same[0]
        # Fallback: nächstliegende Detektion mit gleicher tid in Zeit
        t = i / fps
        near = [d for d in faces if (d.track_id if d.track_id is not None else -1) == want_tid]
        if near:
            return min(near, key=lambda d: abs(d.t - t))
        # worst-case: nimm best in diesem Frame
        return max(cand, key=lambda x: (float(x.w) * float(x.h)))

    raw = []
    for i in range(total_frames):
        tid = tid_per_frame[i]
        d = pick_for_tid(i, tid) if tid != -1 else pick_for_tid(i, tid)
        raw.append((d.cx, d.cy, d.w, d.h, tid))

    # ---- Nachglätten (nur (cx,cy,w,h))
    def _ma(xs: List[float], win: int) -> List[float]:
        if win <= 1 or len(xs) <= 2:
            return xs[:]
        win = win if win % 2 == 1 else win + 1
        r = win // 2
        out = []
        for k in range(len(xs)):
            a = max(0, k - r); b = min(len(xs), k + r + 1)
            out.append(sum(xs[a:b]) / (b - a))
        return out

    cxs = _ma([c for c,_,_,_,_ in raw], smooth_win)
    cys = _ma([c for _,c,_,_,_ in raw], smooth_win)
    ws  = _ma([w for *_,w,_ in raw], max(3, smooth_win//2))
    hs  = _ma([h for *_,_,h,_ in raw], max(3, smooth_win//2))
    tids= [tid for *_, tid in raw]

    out: List[Dict] = []
    for i, (cx, cy, w, h, tid) in enumerate(zip(cxs, cys, ws, hs, tids)):
        t = i / fps
        out.append({
            "frame": i,
            "t": round(t, 4),
            "cx": round(max(0.0, min(1.0, cx)), 4),
            "cy": round(max(0.0, min(1.0, cy)), 4),
            "w":  round(max(0.05, min(1.0, w)), 4),
            "h":  round(max(0.05, min(1.0, h)), 4),
            "tid": int(tid)
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
    p.add_argument("--pattern", type=str, default="highlight_*.mp4",
                   help="Dateimuster in RAW_CLIPS_DIR (Default: highlight_*.mp4)")
    p.add_argument("--fps", type=float, default=0.0, help="FPS erzwingen (0 = aus Video lesen).")
    p.add_argument("--smooth", type=int, default=SMOOTH_WIN_DEF,
                   help="Fensterbreite für Moving-Average-Glättung (ungerade).")
    p.add_argument("--overwrite", action="store_true",
                   help="Existierende target_by_frame.json überschreiben.")
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

        # Face/Centers laden
        faces = load_faces_or_centers(stem, fps_hint=fps)

        # Zielspur bauen
        target = build_target_by_frame(faces, duration=duration, fps=fps, smooth_win=args.smooth)

        # Schreiben
        out = write_target_json(stem, target)
        print(f"💾 geschrieben: {out}")

    print("🎉 Fertig.")


if __name__ == "__main__":
    main()