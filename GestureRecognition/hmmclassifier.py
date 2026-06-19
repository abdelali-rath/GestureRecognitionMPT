import pickle
from collections import defaultdict
from typing import List, Sequence, Any

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - optional dependency
    GaussianHMM = None


class HMMClassifier:
    """HMM-basierter Klassifikator.

    Einfacher Wrapper um :class:`hmmlearn.hmm.GaussianHMM`.

    API:
    - `fit(sequences, labels, **hmm_kwargs)` trainiert ein HMM pro Klasse
    - `decision_function(sequences)` gibt Log-Likelihoods zurück
    - `predict(sequences)` gibt für jede Sequenz das beste Label zurück

    Hinweis: `hmmlearn` wird nicht zwingend installiert. Installiere es mit
    ``pip install hmmlearn`` wenn du echte HMMs nutzen willst.
    """

    def __init__(self, n_components: int = 4, covariance_type: str = "diag", n_iter: int = 100,
                 random_state: int | None = None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.models: dict[Any, Any] = {}
        self.classes_: List[Any] = []

    def _ensure_hmm(self):
        if GaussianHMM is None:
            raise ImportError("hmmlearn is required for HMMClassifier. Install it with `pip install hmmlearn`.")

    def fit(self, sequences: Sequence[np.ndarray], labels: Sequence[Any], **hmm_kwargs):
        """Trainiere ein HMM pro Klasse.

        Parameters
        ----------
        sequences : sequence of (T_i, n_features) arrays
            Die Trainingssequenzen.
        labels : sequence
            Labels für jede Sequenz (gleiche Länge wie `sequences`).
        **hmm_kwargs : dict
            Zusätzliche Parameter, die an ``GaussianHMM`` weitergegeben werden.

        Returns
        -------
        self
        """
        self._ensure_hmm()

        if len(sequences) != len(labels):
            raise ValueError("`sequences` und `labels` müssen die gleiche Länge haben")

        # Gruppiere Sequenzen pro Label
        grouped = defaultdict(list)
        for seq, lab in zip(sequences, labels):
            arr = np.asarray(seq)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            grouped[lab].append(arr)

        self.models = {}
        self.classes_ = []

        for lab, seqs in grouped.items():
            X = np.vstack(seqs)
            lengths = [s.shape[0] for s in seqs]

            model = GaussianHMM(n_components=self.n_components,
                                covariance_type=self.covariance_type,
                                n_iter=self.n_iter,
                                random_state=self.random_state,
                                **hmm_kwargs)
            model.fit(X, lengths)
            self.models[lab] = model
            self.classes_.append(lab)

        return self

    def decision_function(self, sequences: Sequence[np.ndarray]) -> np.ndarray:
        """Berechne Log-Likelihoods für jede Sequenz und Klasse.

        Parameters
        ----------
        sequences : sequence of (T_i, n_features) arrays

        Returns
        -------
        scores : ndarray, shape (n_sequences, n_classes)
            Log-Likelihoods; größere Werte bedeuten bessere Übereinstimmung.
        """
        if not self.models:
            raise ValueError("Der Klassifikator ist nicht trainiert. Rufe zuerst `fit` auf.")

        n = len(sequences)
        m = len(self.classes_)
        scores = np.full((n, m), -np.inf, dtype=float)

        for i, seq in enumerate(sequences):
            arr = np.asarray(seq)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            for j, lab in enumerate(self.classes_):
                model = self.models[lab]
                try:
                    scores[i, j] = model.score(arr)
                except Exception:
                    scores[i, j] = -np.inf

        return scores

    def predict(self, sequences: Sequence[np.ndarray]) -> List[Any]:
        """Gebe für jede Sequenz das Label mit der höchsten Likelihood zurück."""
        scores = self.decision_function(sequences)
        idx = np.argmax(scores, axis=1)
        return [self.classes_[i] for i in idx]

    def save(self, path: str) -> None:
        """Speichere den Klassifikator inklusive Hyperparametern und Modellen."""
        with open(path, "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "n_iter": self.n_iter,
                "random_state": self.random_state,
                "classes_": self.classes_,
                "models": self.models,
            }, f)

    @classmethod
    def load(cls, path: str) -> "HMMClassifier":
        with open(path, "rb") as f:
            state = pickle.load(f)

        obj = cls(n_components=state.get("n_components", 4),
                  covariance_type=state.get("covariance_type", "diag"),
                  n_iter=state.get("n_iter", 100),
                  random_state=state.get("random_state", None))
        obj.classes_ = state.get("classes_", [])
        obj.models = state.get("models", {})
        return obj
