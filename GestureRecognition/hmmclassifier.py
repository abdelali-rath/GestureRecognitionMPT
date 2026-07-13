import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from hmmlearn.hmm import GaussianHMM

try:
    from .paths import resolve_project_path
except ImportError:
    from paths import resolve_project_path


def _as_sequence(sequence: np.ndarray, n_features: int | None = None) -> np.ndarray:
    array = np.asarray(sequence, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("Jede Sequenz muss die Form (Frames, Features) haben und darf nicht leer sein.")
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError(f"Erwartet wurden {n_features} Features, erhalten wurden {array.shape[1]}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Sequenzen dürfen keine NaN- oder unendlichen Werte enthalten.")
    return array


class HMMClassifier:
    """Trainiert ein :class:`GaussianHMM` pro Gestenklasse."""

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "diag",
        n_iter: int = 100,
        random_state: int | None = 42,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.models: dict[Any, GaussianHMM] = {}
        self.classes_: list[Any] = []
        self.n_features_in_: int | None = None

    def fit(self, sequences: Sequence[np.ndarray], labels: Sequence[Any]) -> "HMMClassifier":
        """Trainiert mit allen Sequenzen eines Labels ein separates HMM."""
        if len(sequences) != len(labels):
            raise ValueError("sequences und labels müssen die gleiche Länge haben.")
        if not sequences:
            raise ValueError("Zum Trainieren wird mindestens eine Sequenz benötigt.")

        prepared = [_as_sequence(sequence) for sequence in sequences]
        self.n_features_in_ = prepared[0].shape[1]
        prepared = [_as_sequence(sequence, self.n_features_in_) for sequence in prepared]

        grouped: dict[Any, list[np.ndarray]] = defaultdict(list)
        for sequence, label in zip(prepared, labels):
            grouped[label].append(sequence)

        self.classes_ = sorted(grouped, key=str)
        self.models = {}
        for label in self.classes_:
            class_sequences = grouped[label]
            observations = np.concatenate(class_sequences)
            lengths = [len(sequence) for sequence in class_sequences]
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            model.fit(observations, lengths)
            self.models[label] = model
        return self

    def decision_function(self, sequences: Sequence[np.ndarray]) -> np.ndarray:
        """Gibt pro Sequenz die Log-Likelihood unter jedem Klassenmodell zurück."""
        if not self.models or self.n_features_in_ is None:
            raise ValueError("Der Klassifikator ist nicht trainiert. Rufe zuerst fit auf.")

        scores = np.empty((len(sequences), len(self.classes_)), dtype=float)
        for row, sequence in enumerate(sequences):
            prepared = _as_sequence(sequence, self.n_features_in_)
            for column, label in enumerate(self.classes_):
                scores[row, column] = self.models[label].score(prepared)
        return scores

    def predict(self, sequences: Sequence[np.ndarray]) -> list[Any]:
        """Wählt für jede Sequenz die Klasse mit der höchsten Log-Likelihood."""
        scores = self.decision_function(sequences)
        return [self.classes_[index] for index in np.argmax(scores, axis=1)]

    def save(self, path: str | Path) -> None:
        """Speichert Hyperparameter, Klassenreihenfolge und trainierte Modelle."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            pickle.dump(
                {
                    "format_version": 1,
                    "n_components": self.n_components,
                    "covariance_type": self.covariance_type,
                    "n_iter": self.n_iter,
                    "random_state": self.random_state,
                    "n_features_in_": self.n_features_in_,
                    "classes_": self.classes_,
                    "models": self.models,
                },
                file,
            )

    @classmethod
    def load(cls, path: str | Path) -> "HMMClassifier":
        with Path(path).open("rb") as file:
            state = pickle.load(file)
        if state.get("format_version") != 1:
            raise ValueError("Unbekanntes Modellformat. Bitte das HMM neu trainieren.")

        classifier = cls(
            n_components=state["n_components"],
            covariance_type=state["covariance_type"],
            n_iter=state["n_iter"],
            random_state=state["random_state"],
        )
        classifier.n_features_in_ = state["n_features_in_"]
        classifier.classes_ = state["classes_"]
        classifier.models = state["models"]
        return classifier


def load_dataset(path: str | Path) -> tuple[list[np.ndarray], list[str]]:
    """Rekonstruiert einzelne Sequenzen aus ``gesamt_dataset.pkl``."""
    dataset_path = resolve_project_path(path)
    with dataset_path.open("rb") as file:
        dataset = pickle.load(file)

    observations = np.asarray(dataset["X"], dtype=float)
    lengths = [int(length) for length in dataset["lengths"]]
    labels = [str(label) for label in dataset["labels"]]
    if len(lengths) != len(labels) or sum(lengths) != len(observations):
        raise ValueError("X, lengths und labels passen nicht zusammen.")

    sequences: list[np.ndarray] = []
    normalized_labels: list[str] = []
    offset = 0
    for length, label in zip(lengths, labels):
        sequence = observations[offset : offset + length]
        offset += length
        normalized_label = label.removesuffix("_ready").upper()
        if normalized_label != "LIVE":
            sequences.append(sequence)
            normalized_labels.append(normalized_label)
    return sequences, normalized_labels


def stratified_split(
    sequences: list[np.ndarray],
    labels: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[np.ndarray], list[str], list[np.ndarray], list[str]]:
    """Teilt jede Klasse separat in Trainings- und Testsequenzen."""
    if not 0 < test_size < 1:
        raise ValueError("test_size muss zwischen 0 und 1 liegen.")

    random = np.random.default_rng(random_state)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)

    test_indices: set[int] = set()
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        shuffled = random.permutation(indices)
        count = min(max(1, round(len(indices) * test_size)), len(indices) - 1)
        test_indices.update(int(index) for index in shuffled[:count])

    train_sequences, train_labels, test_sequences, test_labels = [], [], [], []
    for index, (sequence, label) in enumerate(zip(sequences, labels)):
        target_sequences, target_labels = (
            (test_sequences, test_labels) if index in test_indices else (train_sequences, train_labels)
        )
        target_sequences.append(sequence)
        target_labels.append(label)
    return train_sequences, train_labels, test_sequences, test_labels


def _print_confusion_matrix(expected: list[str], predicted: list[str], classes: list[str]) -> None:
    width = max(5, max(map(len, classes), default=1) + 2)
    print("".ljust(width) + "".join(label.rjust(width) for label in classes))
    for expected_label in classes:
        values = [
            sum(a == expected_label and b == predicted_label for a, b in zip(expected, predicted))
            for predicted_label in classes
        ]
        print(expected_label.ljust(width) + "".join(str(value).rjust(width) for value in values))


def train_and_evaluate(
    dataset_path: str | Path = "dataset/gesamt_dataset.pkl",
    model_path: str | Path = "dataset/hmm.pkl",
    n_components: int = 4,
    n_iter: int = 100,
) -> HMMClassifier:
    """Evaluiert mit einem Testset und trainiert danach das finale Modell."""
    sequences, labels = load_dataset(dataset_path)
    counts = Counter(labels)
    keep = {label for label, count in counts.items() if count >= 2}
    sequences = [sequence for sequence, label in zip(sequences, labels) if label in keep]
    labels = [label for label in labels if label in keep]

    train_sequences, train_labels, test_sequences, test_labels = stratified_split(sequences, labels)
    evaluation_model = HMMClassifier(n_components=n_components, n_iter=n_iter)
    evaluation_model.fit(train_sequences, train_labels)
    predicted = evaluation_model.predict(test_sequences)
    accuracy = np.mean(np.asarray(predicted) == np.asarray(test_labels))
    print(f"\nTestgenauigkeit: {accuracy:.1%} ({len(test_labels)} Sequenzen)")
    print("Confusion-Matrix (Zeilen = echt, Spalten = vorhergesagt)")
    _print_confusion_matrix(test_labels, predicted, evaluation_model.classes_)

    print(f"\nTrainiere finales Modell mit allen {len(sequences)} Sequenzen...")
    classifier = HMMClassifier(n_components=n_components, n_iter=n_iter)
    classifier.fit(sequences, labels)
    target = resolve_project_path(model_path)
    classifier.save(target)
    print(f"Finales Modell gespeichert: {target}")
    return classifier
