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
        self.is_recording = False
        self.was_space_pressed = False
        # Konfiguration sicher laden (wie beim TrailMarker)
        config = data.get("config", {}).get("preprocessor", {})
        
        self.label = config.get("label", "Q")
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

        # Tasten abfragen
        is_space_pressed = (ctypes.windll.user32.GetAsyncKeyState(0x20) & 0x8000) != 0
        is_backspace_pressed = (ctypes.windll.user32.GetAsyncKeyState(0x08) & 0x8000) != 0

        # ==========================================
        # 1. UNDO-FUNKTION (Backspace)
        # ==========================================
        if is_backspace_pressed and self.last_saved_file is not None:
            if os.path.exists(self.last_saved_file):
                os.remove(self.last_saved_file)
                print(f"🗑️ UNDO: Letzte Aufnahme gelöscht!")
                self.last_saved_file = None
                time.sleep(0.3)

        # ==========================================
        # 2. KIPPSCHALTER: Start / Stopp mit Leertaste
        # ==========================================
        # Wir reagieren nur genau in dem Moment, in dem die Taste neu heruntergedrückt wird
        if is_space_pressed and not self.was_space_pressed:
            if not self.is_recording:
                # MODUS 1: AUFNAHME STARTEN
                self.is_recording = True
                self.history.clear()
                print("🔴 Aufnahme GESTARTET! Zeichne jetzt...")
            else:
                # MODUS 2: AUFNAHME BEENDEN UND SPEICHERN
                self.is_recording = False
                
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
                    self.last_saved_file = filename
                    
                    anzahl_aktuell = len(os.listdir(self.class_path))
                    print(f"✅ Gespeichert! (Aufnahme {anzahl_aktuell}/40) - Aufnahme GESTOPPT.")
                else:
                    print("⚠️ Geste war zu kurz und wurde ignoriert. Aufnahme GESTOPPT.")
                
                self.history.clear()

        # Den Tastenstatus für den nächsten Frame merken
        self.was_space_pressed = is_space_pressed

        # ==========================================
        # 3. PUNKTE SAMMELN (Aber nur, wenn is_recording an ist!)
        # ==========================================
        if self.is_recording and hand_landmarks:
            x = hand_landmarks.landmark[self.finger_idx].x
            y = hand_landmarks.landmark[self.finger_idx].y
            self.history.append([x, y])

        return {self.outputSignal: result_trajectory}

        return {self.outputSignal: result_trajectory}
    def stop(self, data):
        pass