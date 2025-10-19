import os
import tempfile
from add_subtitles import process  # wir nutzen die Logik aus dem großen Skript

# ==== HIER EINSTELLEN ====
VIDEO_PATH = "data/input.mp4"           # Dein Video
TRANSCRIPT_PATH = "data/transcript.srt" # Oder .json (Whisper)
OUTPUT_DIR = "data/output"              # Ordner für Ergebnisse
CLIPS_PATH = None                       # Optional: "data/clips.csv" oder "data/clips.json"
CRF = 18
PRESET = "medium"
STYLE = r"\\bord4\\shad4\\outline3\\fs64\\b1\\1c&HFFFFFF&\\3c&H000000&\\4c&H000000&"
# ==========================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process(
        video_path=VIDEO_PATH,
        transcript_path=TRANSCRIPT_PATH,
        output_dir=OUTPUT_DIR,
        clips_path=CLIPS_PATH,
        crf=CRF,
        preset=PRESET,
        style_overrides=STYLE,
    )
