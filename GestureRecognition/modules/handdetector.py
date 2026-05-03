import cv2
import numpy as np
import mediapipe as mp

from SignalHub import GALY, bgr, Module

mp_hands = mp.solutions.hands

# Breite (w) und Höhe (h) vür Dynamik
def draw_hand_landmarks(hand_landmarks, galy: GALY, w: int, h: int):
    lm = {
        "thumb":         {"color": bgr("#0000FF")},
        "index_finger":  {"color": bgr("#00FF00")},
        "middle_finger": {"color": bgr("#FF0000")},
        "ring_finger":   {"color": bgr("#00FFFF")},
        "pinky_finger":  {"color": bgr("#FF00FF")},
        "palm":          {"color": bgr("#C8C8C8")},
    }
    
    connections = {
        "thumb":         [(0,1), (1,2), (2,3), (3,4)],
        "index_finger":  [(0,5), (5,6), (6,7), (7,8)],
        "middle_finger": [(9,10), (10,11), (11,12)],
        "ring_finger":   [(13,14), (14,15), (15,16)],
        "pinky_finger":  [(17,18), (18,19), (19,20)],
        "palm":          [(0,17), (5,9), (9,13), (13,17)]
    }

    for key in lm.keys():
        pts = set()
        for start_idx, end_idx in connections[key]:
            start = (int(hand_landmarks.landmark[start_idx].x * w),
                     int(hand_landmarks.landmark[start_idx].y * h))
            end = (int(hand_landmarks.landmark[end_idx].x * w),
                   int(hand_landmarks.landmark[end_idx].y * h))
            
            galy.line(start, end, lm[key]["color"], 2)
            pts.update([start_idx, end_idx])
            
        for pt in pts:
            px = int(hand_landmarks.landmark[pt].x * w)
            py = int(hand_landmarks.landmark[pt].y * h)
            galy.circle((px, py), 5, (255,255,255), 1)
            galy.circle((px, py), 4, lm[key]["color"], -1)


class HandDetector(Module):
    def __init__(self, outputSignal="detector"):
        super().__init__(
            inputSignals=["config", "webcam"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="detector",
        )

    def start(self, data):
        self.detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,  # Weniger wackeln, strenger suchen
            min_tracking_confidence=0.75,   # Hand stabilisieren
            model_complexity=1              # Nutzt genaueres Modell
        )
        self.outputSignal = "detector"
        return {}

    def step(self, data):
        img = data.get("webcam")
        if img is None:
            return {}

        img_h, img_w, _ = img.shape

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)

        galy = GALY()
        galy.blit("webcam", (0, 0))
        galy.layer("landmarks")

        result_data = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            draw_hand_landmarks(hand_landmarks, galy, img_w, img_h)
            result_data = hand_landmarks

        # UI-BEREICH OBEN LINKS
        galy.layer("ui")
        # Einen kleinen Hintergrund für besseren Kontrast zeichnen
        galy.line((10, 40), (250, 40), (0, 0, 0), thickness=60)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Status-Texte
        galy.putText(text="System: Online", org=(20, 30), fontFace=font, fontScale=0.7, color=(255, 255, 255), thickness=2)
        
        if result_data:
            galy.putText(text="Tracking: Hand erfasst", org=(20, 60), fontFace=font, fontScale=0.7, color=(0, 255, 0), thickness=2)
        else:
            galy.putText(text="Tracking: Suche...", org=(20, 60), fontFace=font, fontScale=0.7, color=(0, 0, 255), thickness=2)

        return {self.outputSignal: result_data, "galy": galy}

    def stop(self, data):
        if hasattr(self, 'detector'):
            self.detector.close()