#!/usr/bin/env python3
# clip_selector_optimized.py — word-based text rebuild (no duplicates)

import os
import re
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime
import argparse
import sys
from typing import List, Dict, Optional

from openai import OpenAI

# ── Projektwurzel in sys.path aufnehmen (dieses Skript kann z. B. unter src/text/ liegen)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import TRANSCRIPTS_DIR, DB_PATH  # zentrale Pfade

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

# === DEFAULTS (per CLI überschreibbar) ===
DEFAULT_BLOCK_DURATION = 300.0  # Sek. pro Block
DEFAULT_MIN_CLIP_LEN   = 30.0   # konsistent mit Prompt
DEFAULT_MAX_CLIP_LEN   = 90.0

# === OPENAI-CLIENT (API-Key aus Env) ===
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("🚫 OPENAI_API_KEY fehlt in der Umgebung")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # bei Bedarf überschreiben
client = OpenAI(api_key=API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def log_text(filename: str, content: str) -> None:
    (LOG_DIR / filename).write_text((content or "").strip(), encoding="utf-8")

def append_error_log(content: str) -> None:
    with (LOG_DIR / "errors.txt").open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} {content}\n\n")

def extract_json(text: str) -> list:
    """Nur für Debug: versucht JSON-Array aus beliebigem Text zu extrahieren."""
    txt = (text or "").strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.IGNORECASE | re.DOTALL)
    m = re.search(r"\[\s*{.*?}\s*\]", txt, re.DOTALL)
    if not m:
        append_error_log(f"❌ Kein JSON-Array gefunden.\n{txt}")
        return []
    try:
        return json.loads(m.group(0))
    except Exception as e:
        append_error_log(f"❌ JSON-Fehler: {e}\n{txt}")
        return []

def get_json_snippets_for_clip(start: float, end: float, segment_json: List[Dict]) -> List[Dict]:
    """halb-offenes Fenster [start, end)"""
    return [s for s in segment_json if not (float(s["end"]) <= start or float(s["start"]) >= end)]

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def explode_segments_to_words(segments: List[Dict]) -> List[Dict]:
    """
    Baut eine globale Wortliste. Bevorzugt echte 'words' aus JSON,
    fällt ansonsten auf lineare Interpolation über Segmentdauer zurück.
    Ausgabe-Items: {idx, mid, text}
    """
    words = []
    idx = 0
    for seg in sorted(segments, key=lambda s: (float(s["start"]), float(s["end"]))):
        s0, s1 = float(seg["start"]), float(seg["end"])
        txt = (seg.get("text") or "").strip()
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                t = (w.get("text") or w.get("word") or "").strip()
                if not t:
                    continue
                w0 = float(w["start"]); w1 = float(w["end"])
                words.append({"idx": idx, "mid": round((w0 + w1) / 2.0, 4), "text": t})
                idx += 1
        else:
            toks = txt.split()
            n = len(toks)
            if n == 0:
                continue
            dur = max(1e-6, s1 - s0)
            for i, tok in enumerate(toks):
                w0 = s0 + (i / n) * dur
                w1 = s0 + ((i + 1) / n) * dur
                words.append({"idx": idx, "mid": round((w0 + w1) / 2.0, 4), "text": tok})
                idx += 1
    return words

def build_text_strict_from_words(clip_start: float, clip_end: float, WORDS: List[Dict]) -> str:
    """Nimmt jedes Wort genau einmal, wenn mid ∈ [start, end)."""
    sel = [w for w in WORDS if clip_start <= w["mid"] < clip_end]
    sel.sort(key=lambda w: w["idx"])
    return _norm_space(" ".join(w["text"] for w in sel))

def find_transcript_pair(base: Optional[str]) -> tuple[Path, Path, str]:
    """
    Finde (timed.txt, segments.json) in TRANSCRIPTS_DIR.
    - Wenn base übergeben: benutzt {base}_timed.txt und {base}_segments.json.
    - Sonst: nimmt das lexikographisch erste *_timed.txt und leitet die JSON davon ab.
    """
    if base:
        txt = TRANSCRIPTS_DIR / f"{base}_timed.txt"
        jsn = TRANSCRIPTS_DIR / f"{base}_segments.json"
        if not txt.exists():
            raise FileNotFoundError(f"Transkript nicht gefunden: {txt}")
        if not jsn.exists():
            raise FileNotFoundError(f"Segment-JSON nicht gefunden: {jsn}")
        return txt, jsn, base

    # auto-detect
    candidates = sorted(TRANSCRIPTS_DIR.glob("*_timed.txt"))
    if not candidates:
        raise FileNotFoundError(f"Keine *_timed.txt in {TRANSCRIPTS_DIR} gefunden.")
    txt = candidates[0]
    stem = txt.stem.replace("_timed", "")
    jsn = TRANSCRIPTS_DIR / f"{stem}_segments.json"
    if not jsn.exists():
        raise FileNotFoundError(f"Gefundenes TXT: {txt.name}, aber JSON fehlt: {jsn.name}")
    return txt, jsn, stem

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Selektiert Social-Media-taugliche Clips aus Transkripten (LLM-gestützt).")
    p.add_argument("--base", type=str, default=None,
                   help="Basename der Transkriptdateien (z. B. 'testVideoShort' für *_timed.txt und *_segments.json).")
    p.add_argument("--block", type=float, default=DEFAULT_BLOCK_DURATION, help="Blocklänge in Sekunden für die Prompt-Bildung.")
    p.add_argument("--min",   type=float, default=DEFAULT_MIN_CLIP_LEN,   help="Minimale Clip-Länge (Sekunden).")
    p.add_argument("--max",   type=float, default=DEFAULT_MAX_CLIP_LEN,   help="Maximale Clip-Länge (Sekunden).")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    BLOCK_DURATION = float(args.block)
    MIN_CLIP_LEN   = float(args.min)
    MAX_CLIP_LEN   = float(args.max)

    # --- Transkriptdateien finden ---
    TRANSCRIPT_PATH, SEGMENT_JSON_PATH, base = find_transcript_pair(args.base)
    print(f"📄 TXT : {TRANSCRIPT_PATH}")
    print(f"🧾 JSON: {SEGMENT_JSON_PATH}")

    # === TRANSKRIPT EINLESEN (TXT) -> NUR für Blockbildung & Promptanzeige ===
    lines = TRANSCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    segments_txt: List[Dict] = []
    for line in lines:
        m = re.match(r"\[(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\]\s*(?:[A-Z_0-9]+:)?\s*(.*)", line)
        if not m:
            continue
        start, end, text = m.groups()
        start, end = float(start), float(end)
        if end - start >= 2.0:
            segments_txt.append({"start": start, "end": end, "text": (text or "").strip()})

    if not segments_txt:
        raise RuntimeError("🚫 Keine gültigen TXT-Segmente gefunden.")
    print(f"✅ {len(segments_txt)} gültige TXT-Segmente geladen.")

    # === TRANSKRIPT EINLESEN (JSON) -> Quelle für DB-Text/Wörter ===
    segment_json_data = json.loads(SEGMENT_JSON_PATH.read_text(encoding="utf-8"))
    if not isinstance(segment_json_data, list) or not segment_json_data:
        raise RuntimeError("🚫 JSON-Segmente leer/ungültig.")
    print(f"✅ {len(segment_json_data)} JSON-Segmente geladen.")

    # Globale Wörterliste einmal berechnen (bevor wir Clips bilden)
    WORDS = explode_segments_to_words(segment_json_data)
    print(f"🔤 Globale Wörter im Korpus: {len(WORDS)}")

    # === BLÖCKE BILDEN (aus TXT) ===
    segments_txt.sort(key=lambda s: (s["start"], s["end"]))
    blocks, current_block, current_start = [], [], 0.0
    for seg in segments_txt:
        if not current_block:
            current_start = seg["start"]
        # Blockwechsel, wenn Dauer überschritten
        if seg["end"] - current_start > BLOCK_DURATION:
            blocks.append(current_block)
            current_block = []
            current_start = seg["start"]
        current_block.append(seg)
    if current_block:
        blocks.append(current_block)
    print(f"🧱 {len(blocks)} Blöcke erstellt (à ~{BLOCK_DURATION:.0f}s).")

    # === KI: CLIP-AUSWAHL ===
    all_clips = []
    t0 = time.perf_counter()

    for i, block in enumerate(blocks, start=1):
        if not block:
            continue
        print(f"\n🤖 Sende Block {i}/{len(blocks)} an {OPENAI_MODEL} …")
        block_text = "\n".join([f"[{s['start']} – {s['end']}] {s['text']}" for s in block])

        prompt = f"""
Du bekommst einen Transkriptblock mit Zeitangaben. Extrahiere daraus 1–3 besonders interessante Abschnitte, die sich als eigenständige Social Media Clips eignen.
Ein guter Clip:
- ist abgeschlossen und verständlich
- enthält eine Pointe, Erkenntnis oder einen emotionalen Moment
- wirkt wie ein Mini-Ausschnitt mit Anfang, Spannungsbogen, Auflösung oder Punchline
- ist mindestens {MIN_CLIP_LEN:.0f} Sekunden lang
Nutze ausschließlich die vorhandenen Start- und Endzeiten – keine neuen erfinden.

Gib ein JSON-Objekt zurück im Format:
{{
  "clips": [
    {{
      "start": float,
      "end": float,
      "summary": "Kurze Beschreibung des Inhalts"
    }}
  ]
}}

TRANSKRIPT:
{block_text}
""".strip()

        log_text(f"block_prompt_{i:02d}.txt", prompt)

        # --- robuster API-Call mit Schema (Root=object) und kleinem Retry ---
        import time as _time
        clips = []
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "clips_payload",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "clips": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "start": {"type": "number"},
                                                "end": {"type": "number"},
                                                "summary": {"type": "string"}
                                            },
                                            "required": ["start", "end", "summary"],
                                            "additionalProperties": False
                                        }
                                    }
                                },
                                "required": ["clips"],
                                "additionalProperties": False
                            }
                        }
                    }
                )
                msg = resp.choices[0].message
                payload = getattr(msg, "parsed", None)
                if payload is None:
                    payload = json.loads(msg.content)

                clips = payload.get("clips", []) or []

                try:
                    log_text(f"block_output_{i:02d}.txt", json.dumps(payload, ensure_ascii=False, indent=2))
                except Exception:
                    pass
                break
            except Exception as e:
                if attempt == 2:
                    append_error_log(f"❌ OpenAI-Fehler Block {i}: {e}")
                    print(f"❌ Fehler bei Block {i}: {e}")
                else:
                    _time.sleep(1.5 * (attempt + 1))

        print(f"✅ {len(clips)} Clips empfangen in Block {i}")

        # --- Clips filtern & clampen ---
        for clip in clips:
            try:
                b_start, b_end = block[0]["start"], block[-1]["end"]
                start = max(b_start, min(float(clip["start"]), b_end))
                end   = max(b_start, min(float(clip["end"]),   b_end))
                dur = end - start
                if MIN_CLIP_LEN <= dur <= MAX_CLIP_LEN:
                    clip["start"] = start
                    clip["end"] = end
                    clip["duration"] = round(dur, 2)
                    all_clips.append(clip)
            except Exception as e:
                append_error_log(f"⛔ Clip-Filterfehler: {clip}\n{e}")

        elapsed = time.perf_counter() - t0
        avg = elapsed / i
        eta = max(0.0, avg * (len(blocks) - i))
        print(f"⏱️ Geschätzte Restzeit: {eta:.1f} s")

    # --- Duplikate entfernen (auf 2 Dezimalen) ---
    dedup, seen = [], set()
    for c in all_clips:
        k = (round(c["start"], 2), round(c["end"], 2))
        if k in seen:
            continue
        seen.add(k)
        dedup.append(c)
    all_clips = dedup

    print(f"\n📈 Gesamtclips vor DB-Insert: {len(all_clips)}")

    # === DB SPEICHERN ===
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS highlights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file TEXT,
        start REAL,
        end REAL,
        duration REAL,
        text TEXT,
        summary TEXT,
        json_raw TEXT,
        viralitaet INTEGER,
        emotionalitaet INTEGER,
        witz INTEGER,
        provokation INTEGER,
        score_total INTEGER,
        UNIQUE(file,start,end)
    )
    """)

    # --- Tabelle vor neuem Lauf komplett leeren ---
    cur.execute("DELETE FROM highlights")
    conn.commit()  # Transaktion schließen, damit VACUUM außerhalb läuft

    # VACUUM separat (optional)
    try:
        conn.execute("VACUUM")  # oder: sqlite3.connect(DB_PATH).execute("VACUUM").close()
        print("🧹 Alte Highlights gelöscht und Datenbank komprimiert.")
    except sqlite3.OperationalError as e:
        print(f"⚠️ VACUUM übersprungen: {e}")

    inserted = 0
    failed = 0

    for clip in all_clips:
        try:
            start = float(clip["start"])
            end = float(clip["end"])
            duration = float(clip["duration"])
            summary = (clip.get("summary") or "").strip()

            if end <= start or start < 0:
                raise ValueError("Ungültige Zeiten")

            # JSON-Segmente (zur Nachvollziehbarkeit) + Wort-basierter Text (dopplerfrei)
            json_snippets = get_json_snippets_for_clip(start, end, segment_json_data)
            json_raw = json.dumps(json_snippets, ensure_ascii=False)

            original_text = build_text_strict_from_words(start, end, WORDS)

            cur.execute("""
                INSERT OR IGNORE INTO highlights (
                    file, start, end, duration, text, summary, json_raw,
                    viralitaet, emotionalitaet, witz, provokation, score_total
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
            """, (
                # 'file' = Basename (z. B. testVideoShort)
                Path(base).name,
                start, end, duration,
                original_text, summary, json_raw
            ))
            if cur.rowcount > 0:
                inserted += 1
        except Exception as e:
            failed += 1
            append_error_log(f"❌ DB-Fehler: {clip}\n{e}")

    conn.commit()
    conn.close()

    print("\n📊 Ergebnisse:")
    print(f"  ✅ Highlights gespeichert:  {inserted}")
    print(f"  ❌ Fehlerhafte Clips:       {failed}")
    print(f"📁 Logs:                     {LOG_DIR.resolve()}")

if __name__ == "__main__":
    main()
