import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

try:
    from .paths import iter_npy_files, resolve_label_dir
except ImportError:
    from paths import iter_npy_files, resolve_label_dir


def visualize_dataset(dataset_dir="dataset", label="P", start=0, stop=5):
    """Visualisiert einen Bereich der Aufnahmen eines Labels."""
    label_path = resolve_label_dir(label, dataset_dir)
    files = iter_npy_files(label_path)

    if not files:
        print(f"⚠️ Keine Daten für '{label}' gefunden! Suchpfad: {label_path / '*.npy'}")
        return

    plt.figure(figsize=(10, 6))

    for idx, file in enumerate(files[start:stop]):
        data = np.load(file)
        current_index = start + idx
        plt.plot(
            data[:, 0],
            data[:, 1],
            alpha=0.7,
            label=f"{label} (Aufnahme {current_index})",
        )

    plt.title(f"Trajektorien für Klasse '{label}' (Aufnahmen {start} bis {stop-1})")
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def evaluate_classifier(
    dataset_path="dataset/gesamt_dataset.pkl",
    n_components=4,
    n_iter=100,
    test_size=0.2,
    random_state=42,
    show=True,
):
    """Evaluiert ein HMM auf getrennten Testdaten und zeigt die Confusion Matrix."""
    try:
        from .hmmclassifier import HMMClassifier, load_dataset, stratified_split
    except ImportError:
        from hmmclassifier import HMMClassifier, load_dataset, stratified_split

    sequences, labels = load_dataset(dataset_path)
    counts = Counter(labels)
    valid_labels = {label for label, count in counts.items() if count >= 2}
    filtered = [
        (sequence, label)
        for sequence, label in zip(sequences, labels)
        if label in valid_labels
    ]
    if not filtered:
        raise ValueError("Keine Klassen mit mindestens zwei Sequenzen gefunden.")

    sequences, labels = map(list, zip(*filtered))
    train_sequences, train_labels, test_sequences, test_labels = stratified_split(
        sequences,
        labels,
        test_size=test_size,
        random_state=random_state,
    )

    classifier = HMMClassifier(
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state,
    )
    classifier.fit(train_sequences, train_labels)
    predictions = classifier.predict(test_sequences)
    classes = classifier.classes_
    class_indices = {label: index for index, label in enumerate(classes)}

    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for expected, predicted in zip(test_labels, predictions):
        matrix[class_indices[expected], class_indices[predicted]] += 1
    accuracy = float(np.mean(np.asarray(predictions) == np.asarray(test_labels)))

    figure_size = max(8, min(15, len(classes) * 0.55))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title=f"HMM-Evaluation - Accuracy: {accuracy:.1%}",
        xlabel="Vorhergesagtes Label",
        ylabel="Tatsächliches Label",
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
    )

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = matrix[row, column]
            if value:
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )

    figure.tight_layout()
    print(f"Testgenauigkeit: {accuracy:.1%} ({len(test_labels)} Sequenzen)")
    if show:
        plt.show()

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classes": classes,
        "predictions": predictions,
        "expected": test_labels,
        "figure": figure,
    }


def replay_recordings(dataset_dir="dataset", label="P", count=3):
    """Spielt gespeicherte Aufnahmen eines Labels nacheinander ab."""
    label_path = resolve_label_dir(label, dataset_dir)
    files = iter_npy_files(label_path)

    if not files:
        print(f"⚠️ Keine Aufnahmen für '{label}' zum Abspielen gefunden: {label_path}")
        return

    print(f"🎬 Starte Replay für Klasse '{label}'. Schließe das Fenster für die nächste Geste...")

    for file in files[:count]:
        data = np.load(file)

        _, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"Replay '{label}': {file.name}")

        x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
        y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])

        pad_x = (x_max - x_min) * 0.1 if x_max > x_min else 1
        pad_y = (y_max - y_min) * 0.1 if y_max > y_min else 1

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.5)

        for index in range(1, len(data) + 1):
            ax.plot(
                data[:index, 0],
                data[:index, 1],
                color="blue",
                marker="o",
                markersize=3,
                linestyle="-",
            )
            plt.pause(0.03)

        plt.show()


if __name__ == "__main__":
    print("Starte Datenexploration...")
    visualize_dataset(label="P", start=0, stop=4)
    replay_recordings(label="P", count=2)
