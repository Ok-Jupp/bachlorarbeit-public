#!/usr/bin/env python3
"""Run the full Bachelor pipeline end-to-end with timing, errors, and flexible flags.

Steps:
  1) transcription.py         → Whisper transcripts (segments + timed words)
  2) segment_transcript.py    → LLM selects highlight candidates → SQLite
  3) cutClips.py              → export highlight_*.mp4 (raw clips)
  4) main_detect_faces.py     → YOLO + MediaPipe → faces.json per clip
  5) make_segments.py         → *_target_by_frame.json (center+side per frame)
  6) main_apply_crop.py       → 9:16 crop with smoothing & optional audio mux
  7) rateCluster.py           → (optional) LLM scoring (virality, emotion, ...)
  8) add_subtitles.py         → (optional) word-cap subtitles burned in

Usage examples:
  python main.py --input data/input/meinvideo.mp4 --limit 10 --openai-model gpt-4o
  python main.py --no-rate --no-subs
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# --- Import project config ---
try:
    from config import (
        PROJECT_ROOT, INPUT_DIR, RAW_CLIPS_DIR, CROPPED_DIR, SUBTITLED_DIR,
        WHISPER_CACHE_DIR
    )
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from config import (
        PROJECT_ROOT, INPUT_DIR, RAW_CLIPS_DIR, CROPPED_DIR, SUBTITLED_DIR,
        WHISPER_CACHE_DIR
    )

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- korrekte Pfade zu den Skripten ---
SCRIPTS = {
    "transcription":        str(PROJECT_ROOT / "src" / "text" / "transcription.py"),
    "segment_transcript":   str(PROJECT_ROOT / "src" / "text" / "segment_transcript.py"),
    "cutClips":             str(PROJECT_ROOT / "src" / "text" / "cutClips.py"),
    "detect_faces":         str(PROJECT_ROOT / "src" / "reformat" / "main_detect_faces.py"),
    "make_segments":        str(PROJECT_ROOT / "src" / "reformat" / "make_segments.py"),
    "apply_crop":           str(PROJECT_ROOT / "src" / "reformat" / "main_apply_crop.py"),
    "rateCluster":          str(PROJECT_ROOT / "src" / "text" / "rateCluster.py"),
    "add_subtitles":        str(PROJECT_ROOT / "src" / "subtitles" / "add_subtitles.py"),
}

def shlex_join(cmd):
    return " ".join(str(c) for c in cmd)

def run_step(cmd: list[str], name: str, env: dict[str, str] | None = None) -> float:
    """Run a subprocess step, raise on error, return duration in seconds."""
    t0 = time.perf_counter()
    print(f"\n===== {name} =====")
    print(" ", shlex_join(cmd))
    cp = subprocess.run(cmd, env=env)
    dt = time.perf_counter() - t0
    if cp.returncode != 0:
        print(f"❌ Fehler in {name} (Exit {cp.returncode}) nach {dt:.2f}s")
        print("   → Prüfe das Logfile oben für Details und stelle sicher, dass Abhängigkeiten installiert sind:")
        print("     - ffmpeg/ffprobe im PATH")
        print("     - Python-Pakete: openai-whisper, torch, ffmpeg-python, ultralytics, opencv-python, mediapipe, moviepy, tqdm, numpy")
        print("     - OPENAI_API_KEY gesetzt (für LLM-Schritte)")
        raise SystemExit(cp.returncode)
    print(f"✅ {name} in {dt:.2f}s")
    return dt

def infer_base_from_input(input_path: Path) -> str:
    return input_path.stem

def default_input() -> Path | None:
    if not INPUT_DIR.exists():
        return None
    for p in sorted(INPUT_DIR.iterdir()):
        if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".m4v", ".mp3", ".wav"}:
            return p
    return None

def main():
    ap = argparse.ArgumentParser(description="Bachelor Pipeline Runner")
    ap.add_argument("--input", type=str, default=None, help="Pfad zu Eingabedatei (Default: erstes File in data/input)")
    ap.add_argument("--limit", type=int, default=10, help="Anzahl Highlights (cutClips)")
    ap.add_argument("--whisper-model", type=str, default=os.getenv("WHISPER_MODEL", "small"))
    ap.add_argument("--lang", type=str, default=None, help="Sprachcode (z. B. de)")
    ap.add_argument("--openai-model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    ap.add_argument("--pattern", type=str, default="highlight_*.mp4")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-rate", action="store_true")
    ap.add_argument("--no-subs", action="store_true")
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--no-make", action="store_true")
    ap.add_argument("--no-apply", action="store_true")
    ap.add_argument("--logfile", type=str, default=None)
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)

    env = os.environ.copy()
    env.setdefault("OPENAI_MODEL", args.openai_model)
    env.setdefault("XDG_CACHE_HOME", str(WHISPER_CACHE_DIR))

    if not env.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY ist nicht gesetzt – LLM-Schritte könnten fehlschlagen.")

    # Input-Datei bestimmen
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            candidate = INPUT_DIR / args.input
            if candidate.is_file():
                input_path = candidate
            else:
                raise SystemExit(f"Input nicht gefunden: {args.input}")
    else:
        picked = default_input()
        if not picked:
            raise SystemExit(f"Kein Input in {INPUT_DIR} gefunden. Bitte --input setzen.")
        input_path = picked

    base = infer_base_from_input(input_path)
    print(f"📥 Input: {input_path}")
    print(f"🔤 Whisper: {args.whisper_model} | 🌐 LLM: {env.get('OPENAI_MODEL')}")
    print(f"🧩 Base: {base}")

    # Logfile
    if args.logfile:
        log_path = Path(args.logfile)
    else:
        log_path = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Tee: schreibe in Datei UND Konsole
    try:
        log_fh = open(log_path, "w", encoding="utf-8")
        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, data):
                for s in self.streams:
                    try: s.write(data); s.flush()
                    except Exception: pass
            def flush(self):
                for s in self.streams:
                    try: s.flush()
                    except Exception: pass
        sys.stdout = _Tee(sys.__stdout__, log_fh)
        sys.stderr = _Tee(sys.__stderr__, log_fh)
        print(f"📝 Logfile: {log_path}")
    except Exception:
        print(f"⚠️  Konnte Logfile nicht initialisieren: {log_path}")

    durations = []
    started = datetime.now()
    print(f"🚀 Start: {started:%Y-%m-%d %H:%M:%S}")

    try:
        # 1) Transcription
        t_args = [sys.executable, SCRIPTS["transcription"], "--input", str(input_path), "--model", args.whisper_model]
        if args.lang: t_args += ["--lang", args.lang]
        durations.append(("Transcription", run_step(t_args, "Transcription", env=env)))

        # 2) LLM Segmentierung
        st_args = [sys.executable, SCRIPTS["segment_transcript"], "--base", base]
        durations.append(("Segment Transcript", run_step(st_args, "Segment Transcript", env=env)))

        # 3) Highlights schneiden
        cut_filename = input_path.name
        cc_args = [sys.executable, SCRIPTS["cutClips"], "--file", cut_filename, "--limit", str(args.limit)]
        durations.append(("Cut Clips", run_step(cc_args, "Cut Clips", env=env)))

        # 4) Faces
        if not args.no_detect:
            df_args = [sys.executable, SCRIPTS["detect_faces"]]
            durations.append(("Detect Faces", run_step(df_args, "Detect Faces", env=env)))
        else:
            print("⏭️  Detect Faces übersprungen.")

        # 5) Make Targets
        if not args.no_make:
            ms_args = [sys.executable, SCRIPTS["make_segments"], "--pattern", args.pattern]
            durations.append(("Make Targets", run_step(ms_args, "Make Targets", env=env)))
        else:
            print("⏭️  Make Targets übersprungen.")

        # 6) Crop
        if not args.no_apply:
            ac_args = [sys.executable, SCRIPTS["apply_crop"], "--pattern", args.pattern, "--mux_audio"]
            if args.overwrite: ac_args.append("--overwrite")
            durations.append(("Apply Crop", run_step(ac_args, "Apply Crop", env=env)))
        else:
            print("⏭️  Apply Crop übersprungen.")

        # 7) Bewertung
        if not args.no_rate:
            rc_args = [sys.executable, SCRIPTS["rateCluster"]]
            durations.append(("Rate Clusters", run_step(rc_args, "Rate Clusters", env=env)))
        else:
            print("⏭️  Rate Clusters übersprungen.")

        # 8) Untertitel
        if not args.no_subs:
            as_args = [sys.executable, SCRIPTS["add_subtitles"]]
            durations.append(("Subtitles", run_step(as_args, "Subtitles", env=env)))
        else:
            print("⏭️  Subtitles übersprungen.")

    except KeyboardInterrupt:
        print("\n⛔ Abgebrochen (Ctrl+C).")
    finally:
        finished = datetime.now()
        total = sum(dt for _, dt in durations)
        print("\n======================== ZUSAMMENFASSUNG ============================")
        for name, dt in durations:
            print(f"✅ {name:<24} {dt:7.2f}s")
        print("---------------------------------------------------------------------")
        print(f"⏱️  Gesamtdauer: {total:.2f}s")
        print(f"🕒  Start : {started:%Y-%m-%d %H:%M:%S}")
        print(f"🕒  Ende  : {finished:%Y-%m-%d %H:%M:%S}")
        print(f"📂 Output:")
        print(f"    Raw Clips : {RAW_CLIPS_DIR}")
        print(f"    9:16      : {CROPPED_DIR}")
        print(f"    Subbed    : {SUBTITLED_DIR}")
        print("=====================================================================")

if __name__ == "__main__":
    main()