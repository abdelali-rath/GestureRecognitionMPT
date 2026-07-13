from SignalHub import Module
from collections import deque
import numpy as np
import time
import sys
from pathlib import Path
from pynput import keyboard

try:
    from GestureRecognition.paths import resolve_label_dir
except ImportError:
    PACKAGE_DIR = Path(__file__).resolve().parents[1]
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))
    from paths import resolve_label_dir


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
        self.listener = None

    def _on_press(self, key):
        """
        Wird im exakten Moment des Drückens aufgerufen.
        """
        if hasattr(key, "char") and key.char == "r":
            if not self.is_recording:
                self.history.clear()
            self.toggle_recording = True

        elif key in (keyboard.Key.esc, keyboard.Key.backspace):
            self.trigger_delete = True

    def start(self, data):
        self.outputSignal = "preprocessor"
        self.is_recording = False
        
        self.toggle_recording = False
        self.trigger_delete = False
        self.last_saved_file = None  
        
        self.temp_path = resolve_label_dir("P", create=True)

        config = data.get("config", {}).get("preprocessor", {})
        self.finger_idx = config.get("finger_idx", 8)
        self.buffer_size = config.get("buffer_size", 140)
        self.min_steps = config.get("min_steps", 15)

        self.history = deque(maxlen=self.buffer_size)
        
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        return {}

    def step(self, data):
        hand_landmarks = data.get("detector")
        result_trajectory = None

        # -------------------------------------------------------------------
        # LÖSCHEN DER LETZTEN AUFNAHME
        # -------------------------------------------------------------------
        if self.trigger_delete:
            self.trigger_delete = False  
            if self.last_saved_file and self.last_saved_file.exists():
                self.last_saved_file.unlink()
                print(f"🗑️ [Pipeline] Letzte Aufnahme gelöscht: {self.last_saved_file.name}")
                self.last_saved_file = None
            else:
                print("⚠️ [Pipeline] Keine vorherige Aufnahme zum Löschen gefunden.")
            self.history.clear()

        # -------------------------------------------------------------------
        # AUFNAHME STARTEN / STOPPEN 
        # -------------------------------------------------------------------
        if self.toggle_recording:
            self.toggle_recording = False  
            
            if not self.is_recording:
                self.is_recording = True
                self.history.clear()
                print("🔴 [Pipeline] Aufnahme GESTARTET... (Historie steril gereinigt!)")
            else:
                self.is_recording = False
                if len(self.history) >= self.min_steps:
                    traj = np.array(self.history)
                    
                    center = np.mean(traj, axis=0)
                    traj_centered = traj - center
                    
                    max_dist = np.max(np.abs(traj_centered))
                    traj_normalized = traj_centered / max_dist if max_dist > 0 else traj_centered
                    
                    result_trajectory = traj_normalized
                    
                    label = self.temp_path.name
                    timestamp = int(time.time() * 1000)
                    filename = self.temp_path / f"{label}_{timestamp}.npy"
                    
                    np.save(filename, traj_normalized)
                    self.last_saved_file = filename 
                    
                    print(f"✅ [Pipeline] Gesichert: {filename} ({len(traj)} Frames)")
                else:
                    print("⚠️ [Pipeline] Geste zu kurz, ignoriert.")
                
                self.history.clear()

        # -------------------------------------------------------------------
        # KOORDINATEN AUFZEICHNEN
        # -------------------------------------------------------------------
        if self.is_recording and hand_landmarks:
            x = hand_landmarks.landmark[self.finger_idx].x
            y = hand_landmarks.landmark[self.finger_idx].y
            self.history.append([x, y])

        return {self.outputSignal: result_trajectory}

    def stop(self, data):
        if self.listener is not None:
            self.listener.stop()
