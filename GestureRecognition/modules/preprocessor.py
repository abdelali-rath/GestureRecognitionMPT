from SignalHub import Module
from collections import deque
import numpy as np
import os
import time

class Preprocessor(Module):
    """
    Modul zur Vorverarbeitung von Fingertrajektorien.
    """

    def __init__(self, outputSignal="preprocessor"):
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="preprocessor",
        )

    def start(self, data):
        self.outputSignal = "preprocessor"
        
        # Konfiguration sicher laden (wie beim TrailMarker)
        config = data.get("config", {}).get("preprocessor", {})
        
        self.label = config.get("label", "unknown")
        self.dataset_path = config.get("dataset_path", "dataset")

# Klassenordner erstellen
        self.class_path = os.path.join(self.dataset_path, self.label)
        os.makedirs(self.class_path, exist_ok=True)

        self.finger_idx = config.get("finger_idx", 8)
        self.buffer_size = config.get("buffer_size", 140)
        self.max_lost = config.get("max_lost", 10)
        self.min_steps = config.get("min_steps", 15)

        # Interner Speicher für Trajektorie
        self.history = deque(maxlen=self.buffer_size)
        self.lost_frames = 0
        
        self.outputSignal = "preprocessor"
        return {}

    def step(self, data):
        hand_landmarks = data.get("detector")
        result_trajectory = None

        if hand_landmarks:
            # Hand im Bild: Zähler zurücksetzen
            self.lost_frames = 0
            
            # Relative MediaPipe-Werte (0.0 bis 1.0)
            x = hand_landmarks.landmark[self.finger_idx].x
            y = hand_landmarks.landmark[self.finger_idx].y
            self.history.append([x, y])
        else:
            # Keine Hand im Bild: Zähler hochzählen
            self.lost_frames += 1

        # Geste ist beendet wenn die Hand lange genug aus dem Bild ist
        if self.lost_frames > self.max_lost:
            # Prüfen ob wir genug Punkte für eine Geste gesammelt haben
            if len(self.history) >= self.min_steps:
                
                # 1. In np Array umwandeln [cite: 16]
                traj = np.array(self.history)
                
                # 2. Zentrieren (Mittelwert abziehen, Geste rutscht auf Ursprung 0,0)
                center = np.mean(traj, axis=0)
                traj_centered = traj - center
                
                # 3. Skalieren (durch maximale Ausdehnung teilen, Geste passt exakt in eine Box von -1 bis 1)
                max_dist = np.max(np.abs(traj_centered))
                if max_dist > 0:
                    traj_normalized = traj_centered / max_dist
                else:
                    traj_normalized = traj_centered
                    
                result_trajectory = traj_normalized
                                # Dateiname erzeugen
                timestamp = int(time.time() * 1000)
                
                filename = os.path.join(
                    self.class_path,
                    f"{self.label}_{timestamp}.npy"
                )               
                
                # Datei speichern
                np.save(filename, traj_normalized)
                
                print(f" Gespeichert: {filename}")
                
                print(f" Geste erfasst! Länge: {len(traj_normalized)} Punkte")
                
            # Historie leeren
            self.history.clear()

        # Rückgabe an Framework
        return {self.outputSignal: result_trajectory}
    def stop(self, data):
        pass