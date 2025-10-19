#!/usr/bin/env python3
# cutClips.py — exportiert Clips aus dem ersten gefundenen Video oder aus angegebener Datei

from pathlib import Path
import sqlite3
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
import sys

# ── Projektwurzel in sys.path aufnehmen
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import INPUT_DIR, RAW_CLIPS_DIR, DB_PATH


def parse_args():
    p = argparse.ArgumentParser(description="Exportiert Highlights aus dem Video gemäß SQLite-DB.")
    p.add_argument("--file", type=str, default=None,
                   help="Name der Input-Datei im INPUT_DIR. Wenn leer, wird das erste Video im Ordner verwendet.")
    p.add_argument("--limit", type=int, default=10,
                   help="Anzahl der zu exportierenden Clips (Default: 10)")
    p.add_argument("--order", type=str, choices=["score", "start"], default="score",
                   help="Sortierung: 'score' (score_total absteigend) oder 'start' (zeitlich).")
    return p.parse_args()


def find_first_video(directory: Path):
    """Suche nach der ersten Videodatei im Verzeichnis (mp4, mov, mkv)."""
    for ext in ("*.mp4", "*.mov", "*.mkv"):
        files = sorted(directory.glob(ext))
        if files:
            return files[0]
    return None


def main():
    args = parse_args()

    # === Eingabevideo bestimmen ===
    if args.file:
        input_video = INPUT_DIR / args.file
    else:
        input_video = find_first_video(INPUT_DIR)
        if not input_video:
            raise FileNotFoundError(f"🚫 Kein Video im Eingabeordner {INPUT_DIR} gefunden.")
        print(f"📂 Kein --file angegeben → verwende automatisch: {input_video.name}")

    if not input_video.exists():
        raise FileNotFoundError(f"🚫 Input-Video nicht gefunden: {input_video}")

    output_dir = RAW_CLIPS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # === SQLite DB lesen ===
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    order_clause = "ORDER BY score_total DESC" if args.order == "score" else "ORDER BY start ASC"
    cursor.execute(f"""
        SELECT start, end, text
        FROM highlights
        {order_clause}
        LIMIT ?
    """, (args.limit,))
    highlights = cursor.fetchall()

    if not highlights:
        print("⚠️ Keine Highlights in der Datenbank gefunden.")
        conn.close()
        return

    # === Video laden ===
    video = VideoFileClip(str(input_video))

    # === Clips schneiden ===
    for i, (start, end, text) in enumerate(highlights, start=1):
        if start >= video.duration:
            print(f"⚠️ Clip {i} übersprungen – Startzeit {start:.2f}s liegt außerhalb der Videolänge ({video.duration:.2f}s).")
            continue

        end = min(end, video.duration)
        output_file = output_dir / f"highlight_{i}.mp4"
        print(f"🎬 Exportiere Clip {i}: {start:.2f}s – {end:.2f}s → {output_file.name}")

        try:
            clip = video.subclipped(start, end)
            clip.write_videofile(str(output_file), codec="libx264", audio_codec="aac", logger=None)
            clip.close()
        except Exception as e:
            print(f"❌ Fehler beim Export von Clip {i}: {e}")

    # === Cleanup ===
    conn.close()
    video.close()
    print(f"✅ {len(highlights)} Clips exportiert nach {output_dir}")


if __name__ == "__main__":
    main()