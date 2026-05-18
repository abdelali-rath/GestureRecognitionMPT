from SignalHub import Module
from collections import deque
import numpy as np
import os
import time
import ctypes

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
        self.last_saved_file = None
        # Konfiguration sicher laden (wie beim TrailMarker)
        config = data.get("config", {}).get("preprocessor", {})
        
        self.label = config.get("label", "A")
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

        # NEU: Nur die Löschen-Taste (Backspace) abfragen
        is_backspace_pressed = (ctypes.windll.user32.GetAsyncKeyState(0x08) & 0x8000) != 0

        # ==========================================
        # UNDO-FUNKTION (Backspace gedrückt)
        # ==========================================
        if is_backspace_pressed and self.last_saved_file is not None:
            if os.path.exists(self.last_saved_file):
                os.remove(self.last_saved_file)
                print(f"🗑️ UNDO: Letzte Aufnahme gelöscht!")
                self.last_saved_file = None # Verhindert doppeltes Löschen
                time.sleep(0.3) # Kurze Pause gegen Tasten-Flackern

        # DEINE ALTE AUTOMATISCHE LOGIK:
        if hand_landmarks:
            # Hand im Bild: Zähler zurücksetzen
            self.lost_frames = 0
            
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
                
                traj = np.array(self.history)
                center = np.mean(traj, axis=0)
                traj_centered = traj - center
                
                max_dist = np.max(np.abs(traj_centered))
                if max_dist > 0:
                    traj_normalized = traj_centered / max_dist
                else:
                    traj_normalized = traj_centered
                    
                result_trajectory = traj_normalized
                
                timestamp = int(time.time() * 1000)
                filename = os.path.join(self.class_path, f"{self.label}_{timestamp}.npy")               
                
                # Datei speichern
                np.save(filename, traj_normalized)
                
                # NEU: Dateiname für den Undo-Button merken
                self.last_saved_file = filename
                
                # Zähler anzeigen
                anzahl_aktuell = len(os.listdir(self.class_path))
                print(f"✅ Gespeichert! (Aufnahme {anzahl_aktuell}/40)")
                
            # Historie leeren
            self.history.clear()

        return {self.outputSignal: result_trajectory}
    def stop(self, data):
        pass