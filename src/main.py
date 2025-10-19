#!/usr/bin/env python3
"""
Einfaches Master-Skript, das alle Unter-Skripte nacheinander startet – ohne Argumente.
"""
import subprocess
import sys
from pathlib import Path

# Reihenfolge der auszuführenden Skripte (relativer Pfad)
SCRIPTS = [
    "text/transcription.py",
    "text/segment_transcript.py",
    "text/rateCluster.py",
    "text/cutClips.py",
    "reformat/track_faces_Yolo.py",
    "reformat/detect_speaking_faces.py",
    "reformat/crop_to_speaker.py",
]


def run_script(script_path: str):
    """
    Führt ein Python-Skript ohne weitere Argumente aus.
    """
    print(f"🔄 Running: {script_path}")
    full_path = Path(__file__).parent / script_path
    try:
        subprocess.check_call([sys.executable, str(full_path)])
        print(f"✔️  {script_path} erfolgreich abgeschlossen.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler in {script_path}: Rückgabecode {e.returncode}")
        sys.exit(e.returncode)


def main():
    print("\n=== Starte komplette Podcast-Pipeline ===\n")
    for script in SCRIPTS:
        run_script(script)
    print("✅ Alle Schritte erfolgreich abgeschlossen.")


if __name__ == '__main__':
    main()
