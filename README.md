# Bachelorarbeit – Pipeline: Automatisierte Highlight-Erkennung & 9:16-Aufbereitung

Diese Repository enthält eine vollständige, skriptbasierte Pipeline, um aus Langvideos automatisch Social‑Media‑taugliche 9:16‑Highlights zu erzeugen – inkl. Transkription, KI‑gestützter Clip‑Selektion, Gesichts‑/Mundaktivitätsanalyse, Auto‑Cropping, Untertitel (Word‑Caps) und finalem Export.

## Inhaltsverzeichnis
- [Features](#features)
- [Ordnerstruktur](#ordnerstruktur)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Schnellstart (empfohlener Workflow)](#schnellstart-empfohlener-workflow)
- [Skripte & CLI](#skripte--cli)
- [Tipps & Troubleshooting](#tipps--troubleshooting)
- [Reproduzierbarkeit](#reproduzierbarkeit)
- [Lizenz / Danksagung](#lizenz--danksagung)

---

## Features
- **Transkription mit Wort‑Zeitstempeln (Whisper, chunked ohne Grenz‑Doppler)**
- **LLM‑gestützte Clip‑Selektion** (Viralität/Emotionalität etc. in SQLite gespeichert)
- **Face‑Detection (YOLOv8‑face) & Mundaktivität (MediaPipe)**
- **Stabiles 9:16‑Auto‑Cropping** (Median + EMA, Deadband, Szenenschnitt‑Erkennung, Switch‑Cooldown)
- **Word‑Caps Untertitel** (ASS generiert, per ffmpeg eingebrannt)
- **Batch‑Export der Highlights** (MoviePy, Längen‑/Grenz‑Checks)

## Ordnerstruktur
Die Pfade werden zentral in `config.py` definiert:
```
PROJECT_ROOT/
├─ data/
│  ├─ input/                 # Eingabevideo(s)
│  ├─ transkripte/           # Whisper-Outputs (*_segments.json, *_timed.txt ...)
│  ├─ segments/              # LLM-Clip-Auswahl, DB etc.
│  ├─ output/
│  │  └─ raw_clips/          # Roh-Highlight-Clips (aus cutClips.py)
│  ├─ face_data_combined/    # faces.json je Clip (YOLO + MediaPipe)
│  └─ face_crop_centers/     # (optional) Center-Listen
├─ output/
│  ├─ output_9x16_final/         # Auto-cropped 9:16 Videos
│  ├─ output_9x16_final_subbed_word/  # 9:16 mit eingebrannten Word-Caps
│  └─ debug/                     # Debug-Previews/Artefakte
├─ models/                    # YOLO-Weights (z. B. yolov8n-face.pt)
├─ whisper-cache/            # Whisper Modell-Cache
└─ src/... (optional projektspezifisch)
```
> Beim ersten Start legt `config.py` fehlende Verzeichnisse automatisch an.

## Voraussetzungen
**System‑Tools**
- `ffmpeg` (inkl. `ffprobe`) im `PATH`

**Python**
- Python 3.10+ empfohlen
- Pakete (Beispiel):  
  `openai-whisper`, `torch`, `ffmpeg-python`, `ultralytics`, `opencv-python`, `mediapipe`, `moviepy`, `tqdm`, `numpy`, `regex`
- Optional/abhängig vom Codepfad: `pydub`, `scikit-image` (falls in Erweiterungen verwendet)

**Modelle & Keys**
- **Whisper**: lädt Modelle automatisch in `whisper-cache/` (steuerbar via `WHISPER_MODEL`)
- **YOLOv8‑face**: `models/yolov8n-face.pt` (oder größeres Modell)
- **OpenAI API Key** (für `segment_transcript.py` & `rateCluster.py`): `export OPENAI_API_KEY=...`
  - Default‑Modell ggf. per `export OPENAI_MODEL=gpt-4o` setzen

## Installation
```bash
# 1) Python-Umgebung
python3 -m venv .venv
source .venv/bin/activate

# 2) Systemabhängigkeiten
# ffmpeg installieren (Mac: brew install ffmpeg, Ubuntu: apt install ffmpeg)

# 3) Python-Pakete (Beispiel)
pip install --upgrade pip
pip install openai-whisper torch ffmpeg-python ultralytics opencv-python mediapipe moviepy tqdm numpy regex

# 4) Modelle/Dateien
# YOLO-Weights:
#   Download yolov8n-face.pt → ./models/yolov8n-face.pt
# API Key für LLM:
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"
```

## Schnellstart (empfohlener Workflow)
1) **Eingabe ablegen**  
   Lege dein Langvideo in `data/input/` (z. B. `meinvideo.mp4`).

2) **Transkription (Whisper, chunked & doppler-sicher)**  
```bash
python transcription.py --input data/input/meinvideo.mp4 --model small --lang de
```
   → erzeugt `*_segments.json` + `*_timed.txt` in `data/transkripte/`.

3) **Clips mit LLM selektieren & in DB speichern**  
```bash
export OPENAI_API_KEY="..."; export OPENAI_MODEL="gpt-4o"
python segment_transcript.py --base meinvideo --block 60 --min 6.0 --max 30.0
```
   → schreibt Clips in SQLite (`data/clips_openai.db` o. ä.)

4) **Highlights aus dem Originalvideo schneiden**  
```bash
python cutClips.py --file meinvideo.mp4 --limit 10 --order score
```
   → exportiert `highlight_*.mp4` nach `data/output/raw_clips/`

5) **Face‑Detection + Mundaktivität**  
```bash
python main_detect_faces.py --model models/yolov8n-face.pt     --input-dir data/output/raw_clips --output-dir data/face_data_combined     --frame-skip 1 --downscale 0.5
```

6) **Targets je Frame bauen (Zentren/Größe glätten)**  
```bash
python make_segments.py --pattern "highlight_*.mp4" --fps 0 --smooth 7 --overwrite
```

7) **9:16 Auto‑Crop anwenden**  
```bash
python main_apply_crop.py --pattern "highlight_*.mp4" --median 7 --ema 0.5     --deadband 16 --cut_detect --mux_audio --overwrite
```
   → fertige 9:16‑Clips in `output/output_9x16_final/`

8) **Word‑Caps Untertitel einbrennen (optional)**  
```bash
python add_subtitles.py --clips_dir output/output_9x16_final     --out_dir output/output_9x16_final_subbed_word --model small --limit 20
```
   → fertige Videos mit eingebrannten Word‑Caps in `output/output_9x16_final_subbed_word/`

> 💡 Du kannst viele Parameter (Fensterbreiten, Deadband, Erkennungsschwellen, Limits) über die CLI anpassen.

## Skripte & CLI
### `transcription.py`
Chunked‑Transkription mit Wortzeitstempeln.
```
--input PATH        # Eingabevideo/-audio (Default: erstes File in data/input/)
--outdir PATH       # Ausgabeverzeichnis (Default: data/transkripte/)
--model NAME        # Whisper-Modell (tiny/base/small/medium/large; env: WHISPER_MODEL)
--lang CODE         # Sprachcode (z. B. de) oder leer/None für Auto-Detect
--chunk FLOAT       # Chunk-Länge in s (Default 60)
--overlap FLOAT     # Überlappung in s (Default 2.0)
--min-dur FLOAT     # Mindest-Segmentdauer (s)
--max-gap FLOAT     # Max. Zeit-Gap beim Mergen (s)
--fp16              # Nur sinnvoll mit GPU
```

### `segment_transcript.py`
LLM‑Selektion & Speichern in SQLite.
```
--base STR          # Basename der Transkriptdateien (z. B. 'meinvideo')
--block FLOAT       # Blocklänge s für den Prompt
--min FLOAT         # minimale Clip-Länge s
--max FLOAT         # maximale Clip-Länge s
# env: OPENAI_API_KEY, OPENAI_MODEL (z. B. gpt-4o)
```

### `cutClips.py`
Schneidet ausgewählte Highlights als Einzelclips.
```
--file NAME         # Name der Input-Datei in data/input (Default: erstes Video)
--limit INT         # Anzahl zu exportierender Clips (Default 10)
--order {score,start} # Sortierung: Score (absteigend) oder Startzeit
```

### `main_detect_faces.py`
YOLOv8‑face + MediaPipe → `faces.json` pro Clip.
```
--input-dir PATH    # Default: data/output/raw_clips
--output-dir PATH   # Default: data/face_data_combined
--model PATH        # YOLOv8-face Weights (Default: models/yolov8n-face.pt)
--conf-thresh FLOAT # Default 0.35
--frame-skip INT    # z. B. 1 = jeden Frame, 2 = jeden von zwei ...
--downscale FLOAT   # Frame-Downscale vor YOLO (0..1, z. B. 0.5)
--expansion FLOAT   # Margin Pass 1 (relativ)
--expansion2 FLOAT  # Margin Pass 2 (relativ)
--min-crop INT      # minimale Croplänge (px)
--faces-upscale INT # min. Kantenlänge für FaceMesh (kleine Crops hochskalieren)
--imgsz INT         # YOLO input size (Default 448)
--max-det INT       # Max Detects / Frame
--use-refine        # MediaPipe refine_landmarks aktivieren
```

### `make_segments.py`
Erzeugt `*_target_by_frame.json` (Zentren+Side pro Frame) aus Face/Center‑Daten.
```
--pattern STR       # Dateimuster in raw_clips (Default: highlight_*.mp4)
--fps FLOAT         # FPS erzwingen (0 = aus Video lesen)
--smooth INT        # MA-Fensterbreite (ungerade)
--overwrite         # bestehende target_by_frame.json überschreiben
```

### `main_apply_crop.py`
Wendet 9:16‑Crop mit Glättung/Szenenschnitt an.
```
--pattern STR       # Dateimuster in raw_clips (Default: *.mp4)
--out_w INT         # Output-Breite (Default 1080)
--out_h INT         # Output-Höhe (Default 1920)
--zoom_pad FLOAT    # Zoom-Pad (0..1)
--median INT        # Median-Fenster (>=3, ungerade)
--ema FLOAT         # EMA-Alpha (0..1)
--deadband FLOAT    # Totband in Pixel
--switch_cd INT     # Cooldown-Frames nach Trackwechsel
--cut_detect        # Szenenschnitt-Erkennung aktivieren
--cut_corr FLOAT    # Schwellwert Korrelation (0..1)
--cut_cd INT        # Cooldown-Frames nach Cut
--mux_audio         # Original-Audio unterlegen
--debug             # Debug-Overlay anzeigen
--debug_scale FLOAT # Debug-Preview skaliert rendern
--overwrite         # vorhandene Ausgaben überschreiben
```

### `add_subtitles.py`
Generiert Word‑Caps mit Whisper & brennt sie ein.
```
--clips_dir PATH    # Quelle (Default: output/output_9x16_final)
--out_dir PATH      # Ziel   (Default: output/output_9x16_final_subbed_word)
--pattern STR       # z. B. *.mp4
--limit INT         # Nur die ersten N Clips
--model NAME        # Whisper-Modell (tiny/base/small/medium/large)
--lang CODE         # Sprachcode oder Auto
```

### `rateCluster.py` (optional)
Lässt LLM Scores (Viralität, Emotion, Humor, Provokation) nachtragen.
> Modelliere Standard‑Modell via `OPENAI_MODEL` (z. B. `gpt-4o`).

---

## Tipps & Troubleshooting
- **Modelle/Performance**
  - CPU‑only ist möglich (Whisper/YOLO langsamer). Auf Apple Silicon wird automatisch **MPS** genutzt; auf NVIDIA **CUDA**.
  - `--frame-skip` und `--downscale` in `main_detect_faces.py` beschleunigen die Face‑Detection deutlich.
- **ffmpeg‑Muxing prüfen** (`main_apply_crop.py --mux_audio`): Falls Ton fehlt, `ffmpeg`-Installation checken. Rückgabecode im Log prüfen.
- **Fehlende Dateien**
  - Kein Input? → `data/input/` prüfen.
  - Fehlende Transkript‑Paare? → `*_timed.txt` und `*_segments.json` müssen existieren (aus `transcription.py`).
  - Fehlende Faces? → Pfad zu `models/yolov8n-face.pt` korrekt?
- **Datenbank**
  - Highlights liegen in SQLite (siehe `config.py`: `DB_PATH`). Bei Wiederholungen kann ein `DELETE FROM highlights; VACUUM;` sinnvoll sein.
- **Cache/Verzeichnisse**
  - Whisper‑Cache via `XDG_CACHE_HOME` → `whisper-cache/` neben dem Projekt. Speicherplatz beachten.

## Reproduzierbarkeit
- Lege eine `requirements.txt` mit exakten Versionen an (Freeze deiner funktionierenden Umgebung).
- Dokumentiere verwendete **Modell‑Versionsstände** (YOLO Weights, Whisper‑Modellgröße, OPENAI_MODEL).
- Fixiere Random‑Seeds, falls nötig (hier meist deterministisch durch externe Modelle/Bibliotheken).

## Lizenz / Danksagung
- Verwendung von **OpenAI Whisper**, **Ultralytics YOLOv8**, **MediaPipe**, **OpenCV**, **MoviePy**, **ffmpeg**.
- Die jeweiligen Lizenzen der Bibliotheken beachten.
