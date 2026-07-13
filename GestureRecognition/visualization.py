import numpy as np
import matplotlib.pyplot as plt

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
