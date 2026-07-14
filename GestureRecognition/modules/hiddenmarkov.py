import pickle
import sys
import time
import numpy as np
from pathlib import Path

# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent  
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
        self.default_model_path = model_path
        self.model_path = resolve_project_path(model_path)
        self.min_margin = 0.5
        self.display_seconds = 4.0
        self.model = None
        self.last_result = None
        self.display_until = 0.0

    def start(self, data):
        config = data.get("config", {}).get("hiddenmarkov", {})
        self.model_path = resolve_project_path(config.get("model_path", self.default_model_path))
        self.min_margin = float(config.get("min_margin", 0.5))
        self.display_seconds = float(config.get("display_seconds", 4.0))
        self.last_result = None
        self.display_until = 0.0

        if not self.model_path.exists():
            print(f"❌ [HMM] Modelldatei nicht gefunden: '{self.model_path}'")
            return {}

        try:
            self.model = HMMClassifier.load(self.model_path)
        except (OSError, ValueError, pickle.UnpicklingError) as error:
            self.model = None
            print(f"❌ [HMM] Modell konnte nicht geladen werden: {error}")
            return {}

        print(f"🤖 [HMM] Modell erfolgreich geladen aus '{self.model_path}'")
        return {}

    def step(self, data):
        trajectory = data.get("preprocessor")
        if self.model is None:
            return {}

        if trajectory is not None:
            trajectory = np.asarray(trajectory, dtype=float)
            scores = self.model.decision_function([trajectory])[0] / max(len(trajectory), 1)
            best_idx = int(np.argmax(scores))
            label = self.model.classes_[best_idx]
            score = float(scores[best_idx])
            if len(scores) > 1:
                second_score = float(np.partition(scores, -2)[-2])
                margin = score - second_score
            else:
                margin = np.inf
            confident = np.isfinite(score) and margin >= self.min_margin
            self.last_result = {
                "label": label,
                "score": score,
                "margin": margin,
                "confident": confident,
            }
            self.display_until = time.monotonic() + self.display_seconds
            confidence_text = "sicher" if confident else "unsicher"
            print(
                f"🔤 [HMM] Erkannt: {label} | Score: {score:.2f} | "
                f"Abstand: {margin:.2f} ({confidence_text})"
            )

        if self.last_result is None or time.monotonic() > self.display_until:
            return {self.outputSignal: None}

        galy = GALY()
        galy.layer("hmm", alwaysVisible=True)
        result = self.last_result
        display_text = (
            f"Geste: {result['label']} "
            f"(Score {result['score']:.2f}, Abstand {result['margin']:.2f})"
        )
        text_color = bgr("#00FF00") if result["confident"] else bgr("#FF0000")
        galy.putText(
            text=display_text,
            org=(40, 90),
            color=text_color,
            fontScale=0.8,
            thickness=2,
        )

        return {self.outputSignal: result, "galy": galy}

    def stop(self, data):
        pass
