from SignalHub import GALY, Module
from collections import deque

class TrailMarker(Module):
    def __init__(self, outputSignal="trailmarker"):
        super().__init__(
            inputSignals=["config", "detector", "webcam"],  # "webcam" ür Größe
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="trailmarker",
        )

    def start(self, data):
        self.finger_idx = data.get("config", {}).get("preprocessor", {}).get("finger_idx", 8)
        self.history = deque(maxlen=100)
        self.outputSignal = "trailmarker"
        return {}

    def step(self, data):
        hand_landmarks = data.get("detector")
        img = data.get("webcam") # Bild abrufen
        
        galy = GALY()
        galy.layer("trail")

        if hand_landmarks and img is not None:
            img_h, img_w, _ = img.shape
            
            x = int(hand_landmarks.landmark[self.finger_idx].x * img_w)
            y = int(hand_landmarks.landmark[self.finger_idx].y * img_h)
            
            self.history.append((x, y))
        else:
            self.history.clear()

        if len(self.history) > 1:
            history_list = list(self.history)
            for i in range(1, len(history_list)):
                pt1 = history_list[i - 1]
                pt2 = history_list[i]
                galy.line(pt1, pt2, (255, 255, 0), thickness=4)

        return {self.outputSignal: {}, "galy": galy}

    def stop(self, data):
        self.history.clear()