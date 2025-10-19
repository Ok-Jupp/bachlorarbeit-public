#!/usr/bin/env python3
import logging, json
from pathlib import Path
from typing import List, Dict, Any
import sys

# Projekt-Root verfügbar machen
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import FACE_COMBINED_DIR, FACE_CROP_CENTERS  # ggf. SEGMENTS_DIR, wenn du dorthin schreibst


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW, interH = max(0, xB-xA), max(0, yB-yA)
    inter = interW * interH
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter/union if union > 0 else 0.0

def track_faces(faces_all: List[Dict[str,Any]], iou_thresh=0.3):
    next_id = 0
    last_boxes = {}  # track_id -> bbox
    for frame in faces_all:
        new_boxes = {}
        for face in frame["faces"]:
            box = face["bbox"]
            # match gegen bestehende
            best_id, best_iou = None, 0.0
            for tid, prev_box in last_boxes.items():
                ov = iou(box, prev_box)
                if ov > best_iou:
                    best_id, best_iou = tid, ov
            if best_iou > iou_thresh:
                face["track_id"] = best_id
                new_boxes[best_id] = box
            else:
                face["track_id"] = next_id
                new_boxes[next_id] = box
                next_id += 1
        last_boxes = new_boxes
    return faces_all

def main():
    # Eingabe: erkannte Gesichter/Tracks
    FACE_DIR = FACE_COMBINED_DIR
    # Ausgabe: z. B. berechnete Center pro Frame
    OUT_DIR = FACE_CROP_CENTERS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in FACE_DIR.glob("*_faces.json"):
        try:
            faces_all = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"❌ Fehler beim Laden {f.name}: {e}")
            continue

        tracked = track_faces(faces_all)
        f.write_text(json.dumps(tracked, ensure_ascii=False), encoding="utf-8")
        print(f"✅ Track-IDs ergänzt: {f.name}")

        # zusätzlich centers.json (dominant = höchster mouth_openness pro Frame)
        centers = []
        for fr in tracked:
            if fr["faces"]:
                best = max(fr["faces"], key=lambda ff: ff.get("mouth_openness", 0.0))
                centers.append([best["center"][0], best["center"][1]])
            else:
                centers.append([fr["W"]/2, fr["H"]/2])
        centers_path = f.with_name(f.stem.replace("_faces","_centers")+".json")
        centers_path.write_text(json.dumps(centers, ensure_ascii=False), encoding="utf-8")
        print(f"📝 Centers gespeichert: {centers_path.name}")

if __name__ == "__main__":
    main()
