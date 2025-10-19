#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_subtitles.py — TikTok-Word-Caps mit OpenAI Whisper (CPU)
- läuft Ordner-weise über 9:16-Kurzclips
- transkribiert mit word_timestamps=True
- erzeugt ASS (ein Wort pro Zeile, Pop-Animation, Bottom-Center)
- brennt via ffmpeg in *_subtitled.mp4
"""

import os
import re
import glob
import json
import subprocess
import tempfile
import traceback
import argparse
from typing import List, Tuple, Optional
from pathlib import Path
import sys

# ── Projektwurzel in sys.path aufnehmen (dieses Skript liegt z. B. unter src/subtitles/)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import CROPPED_DIR, SUBTITLED_DIR  # zentrale Pfade

# --- Stabil auf CPU (vermeidet MPS-Sparse-Fehler) ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def log(*a): print("[LOG]", *a)
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def has_audio_stream(video_path: str) -> bool:
    cmd = ["ffprobe","-v","error","-select_streams","a","-show_entries","stream=index","-of","json",video_path]
    try:
        out = subprocess.check_output(cmd).decode("utf-8")
        data = json.loads(out)
        return bool(data.get("streams"))
    except Exception:
        return False

def load_whisper_cpu(model_name: str):
    import whisper  # openai-whisper
    device = "cpu"
    model = whisper.load_model(model_name, device=device)
    fp16 = False
    return model, device, fp16

def transcribe_words_whisper(model, media_path: str, language: Optional[str], fp16: bool) -> List[Tuple[float,float,str]]:
    """
    Nutzt 'openai-whisper' mit word_timestamps=True.
    Fallback: wenn 'words' fehlen, werden Segmenttexte approx. auf Wörter verteilt.
    """
    result = model.transcribe(
        media_path,
        language=language,
        task="transcribe",
        word_timestamps=True,
        condition_on_previous_text=False,
        verbose=False,
        fp16=fp16
    )
    words: List[Tuple[float,float,str]] = []
    segs = result.get("segments", []) or []
    for seg in segs:
        wlist = seg.get("words")
        if isinstance(wlist, list) and wlist and all(isinstance(w, dict) for w in wlist):
            for w in wlist:
                t = (w.get("word") or w.get("text") or "").strip()
                if not t:
                    continue
                ws = w.get("start"); we = w.get("end")
                if ws is None or we is None:
                    continue
                t = re.sub(r"\s+", " ", t)
                if t:
                    words.append((float(ws), float(we), t))
        else:
            # Fallback: Segment auf Wörter aufteilen & Zeiten gleichmäßig verteilen
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            seg_start = float(seg.get("start", 0.0))
            seg_end   = float(seg.get("end", seg_start))
            toks = [w for w in re.split(r"(\s+)", text) if w.strip()]
            if not toks or seg_end <= seg_start:
                continue
            dur = seg_end - seg_start
            step = dur / len(toks)
            for i, tok in enumerate(toks):
                ws = seg_start + i * step
                we = seg_start + (i+1) * step
                words.append((ws, we, tok))
    return words

def ass_time(t: float) -> str:
    if t < 0: t = 0
    h = int(t // 3600); m = int((t % 3600)//60); s = int(t % 60); cs = int(round((t - int(t))*100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def write_ass_words(words: List[Tuple[float,float,str]], ass_path: Path, font_size: int, margin_v: int, uppercase: bool):
    """
    Ein Wort pro Zeile, ohne Überlappung:
    - Ende = min(eigene Endzeit, Start nächstes Wort - 0.02)
    - Pop-Animation 150ms, fette Outline, Bottom-Center (PlayResY=1920)
    """
    header = f"""[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: WordCap,Inter,{font_size},&H00FFFFFF,&H00FFFFFF,&H00101010,&H64000000,1,0,0,0,100,100,0,0,1,6,0.8,2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Zeiten glätten, damit immer nur ein Wort sichtbar ist
    adjusted = []
    for i, (s, e, t) in enumerate(words):
        nstart = words[i+1][0] if i+1 < len(words) else e
        new_end = min(e, nstart - 0.02) if nstart > s else e
        if new_end <= s:
            new_end = s + 0.06
        adjusted.append((s, new_end, t))

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for s, e, t in adjusted:
            st, en = ass_time(s), ass_time(e)
            txt = t.upper() if uppercase else t
            # \fs sichere Größe, \blur für weiche Outline, \fad Ein/Aus,
            # \fscx135\fscy135 → Start groß, \t(...) schrumpft in 150ms auf 100% = Pop
            overrides = r"\blur1\bord8\1c&H0000FFFF&\3c&H000000&\4c&H000000&\fad(50,20)\fscx135\fscy135\t(0,150,\fscx100\fscy100)"
            f.write(f"Dialogue: 0,{st},{en},WordCap,,0,0,0,,{{{overrides}}}{txt}\n")

def ffmpeg_escape_for_subtitles(path: Path) -> str:
    """
    Pfad für -vf subtitles=… escapen (für Leerzeichen, Doppelpunkte etc.).
    ffmpeg erwartet Backslash-escaping für Filter-Argumente.
    """
    s = str(path)
    s = s.replace("\\", "\\\\")
    s = s.replace(":", "\\:")
    s = s.replace("'", "\\'")
    s = s.replace(",", "\\,")
    s = s.replace("[", "\\[")
    s = s.replace("]", "\\]")
    s = s.replace(";", "\\;")
    s = s.replace("=", "\\=")
    return s

def burn(video_in: Path, ass_file: Path, out_path: Path, crf=18, preset="medium") -> int:
    vf = f"subtitles={ffmpeg_escape_for_subtitles(ass_file)}"
    cmd = [
        "ffmpeg","-y","-i",str(video_in),
        "-vf", vf,
        "-c:v","libx264","-preset",preset,"-crf",str(crf),
        "-c:a","copy",
        str(out_path)
    ]
    log("FFmpeg:", " ".join(cmd))
    return subprocess.call(cmd)

def parse_args():
    p = argparse.ArgumentParser(description="Brennt Word-Caps (ASS) via Whisper-Transkription in 9:16-Clips.")
    p.add_argument("--clips_dir", type=Path, default=CROPPED_DIR, help=f"Quellordner (Default: {CROPPED_DIR})")
    p.add_argument("--out_dir",   type=Path, default=SUBTITLED_DIR, help=f"Zielordner (Default: {SUBTITLED_DIR})")
    p.add_argument("--pattern",   type=str,  default="*.mp4",       help="Dateimuster (Default: *.mp4)")
    p.add_argument("--limit",     type=int,  default=None,          help="Nur die ersten N Clips verarbeiten")
    p.add_argument("--model",     type=str,  default=os.getenv("WHISPER_MODEL", "medium"), help="Whisper-Modell")
    p.add_argument("--lang",      type=str,  default=os.getenv("LANGUAGE", "none"),          help="Sprachcode (z. B. de, en, None=Auto)")
    p.add_argument("--uppercase", action="store_true", help="Text in Großbuchstaben rendern")
    p.add_argument("--font_size", type=int,  default=108,  help="ASS-Fontgröße")
    p.add_argument("--margin_v",  type=int,  default=320,  help="ASS-MarginV (Abstand vom unteren Rand)")
    p.add_argument("--crf",       type=int,  default=18,   help="ffmpeg CRF (Qualität)")
    p.add_argument("--preset",    type=str,  default="medium", help="ffmpeg Preset")
    return p.parse_args()

def main():
    args = parse_args()

    clips_dir  = args.clips_dir
    output_dir = args.out_dir
    ensure_dir(output_dir)

    log("Starte TikTok Word-Caps (Whisper)")
    log("CLIPS_DIR =", clips_dir)
    log("OUTPUT_DIR =", output_dir)

    clips: List[str] = []
    for pat in (args.pattern,):
        clips += glob.glob(str(clips_dir / pat))
    clips.sort()
    log(f"{len(clips)} Clips gefunden.")
    if args.limit:
        clips = clips[:args.limit]
        log(f"LIMIT aktiv: {args.limit}")

    if not clips:
        log("Keine Clips gefunden. Pfad/Pattern checken.")
        return

    # Whisper laden (CPU)
    try:
        model, device, fp16 = load_whisper_cpu(args.model)
        log(f"Whisper geladen: {args.model} auf {device} (fp16={fp16})")
        log("Hinweis: Beim ersten Lauf kann das Modell nachgeladen werden.")
    except Exception as e:
        print("[ERROR] Whisper konnte nicht geladen werden:", e)
        traceback.print_exc()
        return

    lang = None if str(args.lang).strip().lower() in {"", "none", "null"} else args.lang

    for clip in clips:
        base = os.path.basename(clip)
        stem, _ = os.path.splitext(base)
        log("="*60)
        log("Clip:", base)

        if not has_audio_stream(clip):
            log("WARN: Keine Audio-Spur → übersprungen.")
            continue

        # Transkription
        try:
            log("Transkription startet …")
            words = transcribe_words_whisper(model, clip, language=lang, fp16=fp16)
            log(f"Transkription fertig. {len(words)} Wörter.")
            if not words:
                log("WARN: 0 Wörter erkannt → übersprungen.")
                continue
        except Exception as e:
            print("[ERROR] Transkription fehlgeschlagen:", e)
            traceback.print_exc()
            continue

        # ASS erzeugen & brennen
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
            ass_path = Path(tmp.name)
        try:
            log("Erzeuge ASS …")
            write_ass_words(words, ass_path, font_size=args.font_size, margin_v=args.margin_v, uppercase=args.uppercase)
            out_path = output_dir / f"{stem}_subtitled.mp4"
            log("Brenne Untertitel …")
            rc = burn(Path(clip), ass_path, out_path, crf=args.crf, preset=args.preset)
            if rc == 0:
                log("OK:", out_path)
            else:
                log("ERROR: ffmpeg fehlgeschlagen, code", rc)
        finally:
            try: ass_path.unlink(missing_ok=True)
            except Exception: pass

    log("Fertig.")

if __name__ == "__main__":
    main()
