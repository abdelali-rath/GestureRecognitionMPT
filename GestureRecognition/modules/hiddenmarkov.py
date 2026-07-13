import sys
import numpy as np
from pathlib import Path

# ==============================================================================
# PFAD-RETTER: Erkennt den Hauptordner automatisch für fehlerfreie Imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent  # Geht hoch bis zu GestureRecognitionMPT
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# ==============================================================================

from SignalHub import GALY, bgr, Module
from GestureRecognition.hmmclassifier import HMMClassifier
from GestureRecognition.paths import resolve_project_path


class HMMModule(Module):
    """Modul zur Live-Klassifikation von Gesten mittels Hidden Markov Models."""

    def __init__(self, outputSignal="markov", model_path="dataset/hmm.pkl", **kwargs):
        super().__init__(
            inputSignals=["config", "preprocessor"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="hiddenmarkov",
        )
        self.outputSignal = outputSignal
        self.model_path = resolve_project_path(model_path)
        self.model = None

    def start(self, data):
        if self.model_path.exists():
            self.model = HMMClassifier.load(self.model_path)
            print(f"🤖 [HMM] Modell erfolgreich geladen aus '{self.model_path}'")
        else:
            print(f"❌ [HMM] Modelldatei nicht gefunden: '{self.model_path}'")
        return {}

    def step(self, data):
        trajectory = data.get("preprocessor")
        if trajectory is None or self.model is None:
            return {}

        scores = self.model.decision_function([trajectory])[0]
        best_idx = np.argmax(scores)
        label = self.model.classes_[best_idx]
        score = scores[best_idx]

        galy = GALY()
        layer = galy.add_layer()
        display_text = f"Geste: {label} ({score:.1f})"
        text_color = bgr.GREEN if score > -1000 else bgr.RED
        layer.putText(display_text, x=40, y=60, color=text_color, scale=1.2, thickness=2)

        return {self.outputSignal: label, "galy": galy}

    def stop(self, data):
        pass
