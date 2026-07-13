from SignalHub import Module
from collections import deque
import numpy as np
import time
import sys
from pathlib import Path
from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QApplication

try:
    from GestureRecognition.paths import resolve_label_dir
except ImportError:
    PACKAGE_DIR = Path(__file__).resolve().parents[1]
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))
    from paths import resolve_label_dir


class _KeyboardFilter(QObject):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def eventFilter(self, watched, event):
        if event.type() != QEvent.KeyPress or event.isAutoRepeat():
            return False
        if event.key() == Qt.Key_R:
            self.preprocessor.toggle_recording = True
            return True
        if event.key() == Qt.Key_Backspace:
            self.preprocessor.trigger_delete = True
            return True
        return False


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
        self.keyboard_filter = None

    def _install_keyboard_filter(self):
        if self.keyboard_filter is not None:
            return
        application = QApplication.instance()
        if application is not None:
            self.keyboard_filter = _KeyboardFilter(self)
            application.installEventFilter(self.keyboard_filter)
            print("⌨️ [Pipeline] Steuerung aktiv: R = Start/Stopp, Backspace = letzte Aufnahme löschen")

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
        
        return {}

    def step(self, data):
        self._install_keyboard_filter()
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
        application = QApplication.instance()
        if application is not None and self.keyboard_filter is not None:
            application.removeEventFilter(self.keyboard_filter)
        self.keyboard_filter = None
