# src/speaking.py

TOP_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
BOTTOM_LIPS = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

def get_mouth_openness(landmarks, image_height):
    """
    Berechnet die Mundöffnung basierend auf MediaPipe-Landmarks.
    """
    top_avg = sum(landmarks[i].y for i in TOP_LIPS) / len(TOP_LIPS)
    bottom_avg = sum(landmarks[i].y for i in BOTTOM_LIPS) / len(BOTTOM_LIPS)
    return abs(bottom_avg - top_avg) * image_height
