import sqlite3
import re
from openai import OpenAI
from time import sleep
from pathlib import Path
import os

from pathlib import Path
import sys

# Projekt-Root einfügen (2 Ebenen hoch von src/* ausgehend)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import DB_PATH


MAX_CLIPS = 5  # oder "all"

# === OPENAI-CLIENT (API-Key aus Env) ===
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("🚫 OPENAI_API_KEY fehlt in der Umgebung")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === DB-Verbindung
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# === Unbewertete Highlights laden
cursor.execute("""
    SELECT id, start, end, text FROM highlights
    WHERE viralitaet IS NULL OR emotionalitaet IS NULL OR witz IS NULL OR provokation IS NULL
    ORDER BY start
""")
segments = cursor.fetchall()
print(f"📥 {len(segments)} unbewertete Highlights geladen.")

# === Bewertungsfunktion (GPT-4o)
def analyse_segment(clip_id, text, start, end):
    print(f"\n🔎 Bewerte Clip: {start:.2f}s – {end:.2f}s")

    prompt = f"""
Bewerte folgenden Podcast-Ausschnitt mit genau vier Zahlen zwischen 1 und 10. Achte darauf das es abgeschlossene Clips sind und als eigenstaendiger Clip funktionieren kann.

\"\"\"{text}\"\"\"

Dauer: {start:.2f} bis {end:.2f} Sekunden.

Antwortformat (bitte exakt einhalten, keine weiteren Kommentare):
Viralität: [Zahl]
Emotionalität: [Zahl]
Witz: [Zahl]
Provokation: [Zahl]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        output = response.choices[0].message.content.strip()
        print(f"📤 GPT-Antwort:\n{output}")

        values = {
            "viralitaet": None,
            "emotionalitaet": None,
            "witz": None,
            "provokation": None
        }

        for line in output.splitlines():
            line = line.strip().lower().replace("ä", "ae")
            if line.startswith("viralitaet"):
                values["viralitaet"] = int(re.search(r"\d+", line).group())
            elif line.startswith("emotionalitaet"):
                values["emotionalitaet"] = int(re.search(r"\d+", line).group())
            elif line.startswith("witz"):
                values["witz"] = int(re.search(r"\d+", line).group())
            elif line.startswith("provokation"):
                values["provokation"] = int(re.search(r"\d+", line).group())

        if all(v is not None for v in values.values()):
            total_score = sum(values.values())
            cursor.execute("""
                UPDATE highlights SET
                    viralitaet = ?, emotionalitaet = ?, witz = ?, provokation = ?, score_total = ?
                WHERE id = ?
            """, (
                values["viralitaet"], values["emotionalitaet"],
                values["witz"], values["provokation"],
                total_score,
                clip_id
            ))
            conn.commit()

            return {
                "id": clip_id,
                "start": start,
                "end": end,
                "text": text.strip(),
                "score": values,
                "total": total_score
            }
        else:
            raise ValueError("Unvollständige Bewertung")
    except Exception as e:
        print(f"⚠️ Fehler bei GPT-Auswertung: {e}")
        return None

# === Clips bewerten
rated = []
for clip_id, start, end, text in segments:
    result = analyse_segment(clip_id, text, float(start), float(end))
    if result:
        rated.append(result)
    sleep(1.2)  # Anti-Rate-Limit

# === Beste Clips anzeigen
rated.sort(key=lambda x: x["total"], reverse=True)
selected = rated if MAX_CLIPS == "all" else rated[:int(MAX_CLIPS)]

print(f"\n🎬 Beste {len(selected)} Highlights nach Bewertung:")
for clip in selected:
    print(f"\n🚀 {clip['start']:.2f}s – {clip['end']:.2f}s")
    print(f"🎙️  {clip['text'][:200]}...")
    print("📊 Bewertung:")
    for k, v in clip["score"].items():
        print(f"   {k.capitalize()}: {v}")
    print(f"   👉 Gesamt: {clip['total']}")

conn.close()
