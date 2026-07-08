from SignalHub import Module
from collections import deque
import numpy as np
import os
import time
from pynput import keyboard

class Preprocessor(Module):
    """
    Modul zur Vorverarbeitung und temporären Zwischenspeicherung von Fingertrajektorien.
    """

    def __init__(self, outputSignal="preprocessor"):
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="preprocessor",
        )
        self.pressed_keys = set()
        self.listener = None

    def _on_press(self, key):
        self.pressed_keys.add(key)

    def _on_release(self, key):
        try:
            self.pressed_keys.remove(key)
        except KeyError:
            pass

    def start(self, data):
        self.outputSignal = "preprocessor"
        self.is_recording = False
        self.was_space_pressed = False
        
        # Temporären Ordner für Rohaufnahmen vorbereiten
        self.temp_path = "dataset/P"
        os.makedirs(self.temp_path, exist_ok=True)

        config = data.get("config", {}).get("preprocessor", {})
        self.finger_idx = config.get("finger_idx", 8)
        self.buffer_size = config.get("buffer_size", 140)
        self.min_steps = config.get("min_steps", 15)

        self.history = deque(maxlen=self.buffer_size)
        
        # Tastatur-Listener für die Aufnahme-Steuerung starten
        self.pressed_keys.clear()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        return {}

    def step(self, data):
        hand_landmarks = data.get("detector")
        result_trajectory = None

        is_space_pressed = keyboard.Key.space in self.pressed_keys

        # Kippschalter: Start / Stopp mit Leertaste
        if is_space_pressed and not self.was_space_pressed:
            if not self.is_recording:
                self.is_recording = True
                self.history.clear()
                print("🔴 [Pipeline] Aufnahme GESTARTET...")
            else:
                self.is_recording = False
                if len(self.history) >= self.min_steps:
                    traj = np.array(self.history)
                    
                    # 1. Zentrieren
                    center = np.mean(traj, axis=0)
                    traj_centered = traj - center
                    
                    # 2. Normalisieren
                    max_dist = np.max(np.abs(traj_centered))
                    traj_normalized = traj_centered / max_dist if max_dist > 0 else traj_centered
                    
                    result_trajectory = traj_normalized
                    
                    # Temporär wegspeichern für das spätere Labeling-Skript
                    timestamp = int(time.time() * 1000)
                    filename = os.path.join(self.temp_path, f"{self.temp_path}_{timestamp}.npy")
                    np.save(filename, traj_normalized)
                    print(f"✅ [Pipeline] Temporär gesichert: {filename} ({len(traj)} Frames)")
                else:
                    print("⚠️ [Pipeline] Geste zu kurz, ignoriert.")
                
                self.history.clear()

        self.was_space_pressed = is_space_pressed

        if self.is_recording and hand_landmarks:
            x = hand_landmarks.landmark[self.finger_idx].x
            y = hand_landmarks.landmark[self.finger_idx].y
            self.history.append([x, y])

        return {self.outputSignal: result_trajectory}

    def stop(self, data):
        if self.listener is not None:
            self.listener.stop()