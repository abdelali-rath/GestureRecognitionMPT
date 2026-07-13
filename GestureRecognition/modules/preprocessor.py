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
        self.listener = None

    def _on_press(self, key):
        """
        Wird im exakten Moment des Drückens aufgerufen.
        """
        try:
            if hasattr(key, 'char') and key.char == 'r':
                # 🔥 BLITZ-LÖSCHUNG: Wenn wir gleich STARTEN, löschen wir die 
                # Historie DIREKT hier im Tastatur-Thread. Keine Millisekunde Verzögerung!
                if not self.is_recording:
                    self.history.clear()
                self.toggle_recording = True
        result_trajectory = None

        # -------------------------------------------------------------------
        # 🗑️ LÖSCHEN DER LETZTEN AUFNAHME
        # LÖSCHEN DER LETZTEN AUFNAHME
        # -------------------------------------------------------------------
        if self.trigger_delete:
            self.trigger_delete = False  
            self.history.clear()

        # -------------------------------------------------------------------
        # 🔴 AUFNAHME STARTEN / STOPPEN 
        # AUFNAHME STARTEN / STOPPEN 
        # -------------------------------------------------------------------
        if self.toggle_recording:
            self.toggle_recording = False  
            self.history.clear()

        # -------------------------------------------------------------------
        # ✍️ KOORDINATEN AUFZEICHNEN
        # KOORDINATEN AUFZEICHNEN
        # -------------------------------------------------------------------
        if self.is_recording and hand_landmarks:
            x = hand_landmarks.landmark[self.finger_idx].x