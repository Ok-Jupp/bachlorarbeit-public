#!/usr/bin/env python3
# transcription_chunked_words.py — Whisper mit Wortzeitstempeln, doppler-sicher

import os
import sys
import json
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple, Optional

import ffmpeg
import whisper

# ── Projektwurzel in sys.path aufnehmen (dieses Skript liegt z. B. unter src/text/)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import INPUT_DIR, TRANSCRIPTS_DIR  # zentrale Pfade

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def probe_duration(path: Path) -> float:
    """Ermittle die Videodauer in Sekunden (ffmpeg.probe)."""
    try:
        meta = ffmpeg.probe(str(path))
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg.probe fehlgeschlagen für {path}: {e.stderr.decode('utf-8','ignore') if hasattr(e, 'stderr') else e}") from e

    dur = meta.get("format", {}).get("duration")
    if dur is not None:
        return float(dur)

    cand = 0.0
    for s in meta.get("streams", []) or []:
        d = s.get("duration")
        if d:
            cand = max(cand, float(d))
    if cand > 0:
        return cand
    raise RuntimeError(f"Konnte Videodauer nicht bestimmen: {path}")

def make_chunks(total: float, chunk_seconds: float, overlap: float) -> List[Tuple[float,float]]:
    """Zerteile [0,total] in überlappende Intervalle."""
    if chunk_seconds <= 0:
        return [(0.0, total)]
    s, out = 0.0, []
    while s < total:
        e = min(s + chunk_seconds, total)
        out.append((s, e))
        if e >= total:
            break
        s = max(0.0, e - overlap)
    return out

def extract_audio_segment(src_video: Path, start: float, end: float, out_wav: Path) -> None:
    """Extrahiere [start,end] als Mono-16kHz-WAV."""
    (
        ffmpeg
        .input(str(src_video), ss=start, to=end)
        .output(
            str(out_wav),
            format="wav",
            acodec="pcm_s16le",
            ac=1,
            ar="16000",
            loglevel="error",
        )
        .overwrite_output()
        .run()
    )

def is_suspect(text: str) -> bool:
    """Heuristik: leere/loopende/zweifelhafte Zeilen markieren."""
    t = (text or "").strip().lower()
    if not t:
        return True
    words = t.split()
    if not words:
        return True
    counts = {w: words.count(w) for w in set(words)}
    most_common = max(counts.values())
    return most_common / len(words) > 0.6 or most_common > 20

def merge_overlaps_keep_best(
    segments: List[Dict],
    max_gap: float = 0.15,
    min_dur: float = 0.30
) -> List[Dict]:
    """
    Zeitlich sortieren, kleine Gaps schließen. Bei Überlappung:
    - keine Text-Konkatenation
    - behalte das "bessere" Segment (längere Dauer, dann längerer Text)
    - words: vom "best" übernehmen (falls vorhanden)
    """
    cleaned = []
    for s in segments:
        s0 = float(s["start"]); s1 = float(s["end"])
        txt = (s.get("text") or "").strip()
        if s1 - s0 >= min_dur and txt:
            cleaned.append({
                "start": s0, "end": s1,
                "text": txt,
                "words": s.get("words", [])
            })
    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    out = [cleaned[0]]

    def score(x: Dict) -> tuple:
        return (x["end"] - x["start"], len(x.get("text", "")))

    for s in cleaned[1:]:
        m = out[-1]
        if s["start"] <= m["end"] + max_gap:
            best = s if score(s) > score(m) else m
            out[-1] = {
                "start": min(m["start"], s["start"]),
                "end":   max(m["end"],   s["end"]),
                "text":  best["text"],
                "words": best.get("words", []),
            }
        else:
            out.append(s)
    return out

def write_outputs(base: Path, segments: List[Dict], out_dir: Path, ascii_dash: bool = True):
    """Schreibe _timed.txt, _suspect_lines.txt und _segments.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dash = "-" if ascii_dash else "–"

    out_txt  = out_dir / f"{base.stem}_timed.txt"
    out_sus  = out_dir / f"{base.stem}_suspect_lines.txt"
    out_json = out_dir / f"{base.stem}_segments.json"

    # TXT nur zur Ansicht
    with open(out_txt, "w", encoding="utf-8") as f_txt, open(out_sus, "w", encoding="utf-8") as f_sus:
        for s in segments:
            line = f"[{s['start']:.2f} {dash} {s['end']:.2f}] {s['text']}\n"
            f_txt.write(line)
            if is_suspect(s["text"]):
                f_sus.write(line)

    # JSON für die Weiterverarbeitung (inkl. words)
    with open(out_json, "w", encoding="utf-8") as f_json:
        json.dump(segments, f_json, ensure_ascii=False, indent=2)

    return out_txt, out_sus, out_json

def find_default_input() -> Optional[Path]:
    """Nimm das erste Video aus INPUT_DIR, falls kein --input übergeben wurde."""
    exts = (".mp4", ".mov", ".mkv", ".m4v", ".wav", ".mp3")
    for p in sorted(INPUT_DIR.iterdir()):
        if p.suffix.lower() in exts:
            return p
    return None

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Chunked Whisper Transcription mit Wortzeitstempeln & doppler-sicherem Stitching."
    )
    p.add_argument("--input", type=Path, default=None, help=f"Eingabevideo/-audio. Default: erstes File in {INPUT_DIR}")
    p.add_argument("--outdir", type=Path, default=None, help=f"Ausgabeverzeichnis. Default: {TRANSCRIPTS_DIR}")
    p.add_argument("--model", type=str, default=os.getenv("WHISPER_MODEL", "small"), help="Whisper-Modell (tiny/base/small/medium/large)")
    p.add_argument("--lang", type=str, default=os.getenv("LANGUAGE", "none"), help="Sprachcode (z. B. 'de') oder leer/None für Auto-Detect")
    p.add_argument("--chunk", type=float, default=60.0, help="Chunk-Länge in Sekunden (0 = ganzes File)")
    p.add_argument("--overlap", type=float, default=2.0, help="Overlap in Sekunden")
    p.add_argument("--min-dur", type=float, default=0.30, help="Mindest-Segmentdauer (Sekunden)")
    p.add_argument("--max-gap", type=float, default=0.15, help="Maximaler Zeit-Gap für Merge (Sekunden)")
    p.add_argument("--fp16", action="store_true", help="fp16 aktivieren (nur sinnvoll mit GPU)")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Whisper-Cache (damit Modelle lokal landen)
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "whisper-cache"))

    args = parse_args()
    input_path = args.input or find_default_input()
    out_dir = args.outdir or TRANSCRIPTS_DIR

    print("📁 Projekt-Root:", ROOT)
    print("📄 Input:", input_path if input_path else "—")
    if not input_path or not input_path.exists():
        raise FileNotFoundError(f"Kein gültiges Eingabefile gefunden. Lege ein Video/Audio in {INPUT_DIR} oder nutze --input.")

    out_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(input_path)
    print(f"🎬 Dauer: {duration:.2f}s")

    chunks = make_chunks(duration, args.chunk, args.overlap)
    print(f"🔪 {len(chunks)} Chunks à {args.chunk:.1f}s mit {args.overlap:.1f}s Overlap")

    # Whisper laden
    print(f"🧠 Lade Whisper-Modell: {args.model}")
    try:
        model = whisper.load_model(args.model)
    except Exception as e:
        raise RuntimeError(f"Whisper-Modell '{args.model}' konnte nicht geladen werden. Installiert? (pip install openai-whisper)\n{e}") from e

    all_segments: List[Dict] = []
    with TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for i, (start, end) in enumerate(chunks, 1):
            print(f"🔉 Chunk {i}/{len(chunks)}: {start:.2f}s - {end:.2f}s")
            wav = tmpdir / f"chunk_{i:03d}.wav"
            extract_audio_segment(input_path, start, end, wav)

            # Sprache: ''/none = Auto-Detect
            lang = None if str(args.lang).strip().lower() in {"", "none", "null"} else args.lang

            # Transkribieren mit Wortzeiten, ohne Cross-Chunk-Kontext
            result = model.transcribe(
                str(wav),
                language=lang,
                fp16=args.fp16,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0,
                verbose=False,
            )

            # Center-Cut: nur Mittelteil behalten (verhindert Grenz-Doppler)
            keep_start = start if i == 1 else start + args.overlap / 2.0
            keep_end   = end   if i == len(chunks) else end - args.overlap / 2.0

            for seg in result.get("segments", []) or []:
                s0 = float(seg["start"]) + start
                s1 = float(seg["end"]) + start
                mid = (s0 + s1) / 2.0
                if not (keep_start <= mid < keep_end):
                    continue

                # Wörter mit absoluten Zeiten übernehmen
                words = []
                for w in (seg.get("words") or []):
                    txt = (w.get("word") or w.get("text") or "").strip()
                    if not txt:
                        continue
                    words.append({
                        "start": float(w["start"]) + start,
                        "end":   float(w["end"])   + start,
                        "text":  txt
                    })

                all_segments.append({
                    "start": s0,
                    "end":   s1,
                    "text":  (seg.get("text") or "").strip(),
                    "words": words
                })

    print(f"🧹 Roh-Segmente: {len(all_segments)}  → merge & filter …")
    merged = merge_overlaps_keep_best(all_segments, max_gap=args.max_gap, min_dur=args.min_dur)
    print(f"✅ Gemergte Segmente: {len(merged)}")

    out_txt, out_sus, out_json = write_outputs(input_path, merged, out_dir, ascii_dash=True)
    print(f"📝 TXT: {out_txt}")
    print(f"⚠️  SUSPECT: {out_sus}")
    print(f"💾 JSON: {out_json}")
    print("🎉 Fertig.")

if __name__ == "__main__":
    main()
