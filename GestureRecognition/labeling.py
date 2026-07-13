import pickle
import numpy as np
import matplotlib.pyplot as plt

try:
    from .paths import iter_npy_files, resolve_dataset_dir, resolve_label_dir, resolve_project_path
except ImportError:
    from paths import iter_npy_files, resolve_dataset_dir, resolve_label_dir, resolve_project_path


def _wait_for_labeling_key(figure):
    pressed = []

    def on_key(event):
        key = (event.key or "").lower()
        if key in {"enter", "space", "escape"}:
            pressed.append(key)

    connection = figure.canvas.mpl_connect("key_press_event", on_key)
    while not pressed and plt.fignum_exists(figure.number):
        plt.pause(0.05)
    figure.canvas.mpl_disconnect(connection)
    return pressed[0] if pressed else "escape"


def data_labeling(label, base_path="dataset"):

    folder = resolve_label_dir(label, base_path)

    if not folder.exists():
        print(f"Ordner {folder} existiert nicht.")
        return

    files = iter_npy_files(folder)

    if not files:
        print("Keine Aufnahmen gefunden.")
        return

    print(f"\n=== Labeling {label} ===")
    print("[ENTER] behalten")
    print("[SPACE] löschen")
    print("[ESC] abbrechen\n")


    for file in files:
        data = np.load(file)

        figure = plt.figure("Vorschau", figsize=(5,5))
        plt.clf()

        plt.plot(data[:,0], data[:,1], "-o", markersize=3)
        plt.scatter(data[0,0], data[0,1], color="green", s=80, label="Start")
        plt.scatter(data[-1,0], data[-1,1], color="red", s=80, label="Ende")

        plt.xlim(-1.2,1.2)
        plt.ylim(-1.2,1.2)
        plt.gca().invert_yaxis()
        plt.grid()
        plt.legend()

        plt.show(block=False)
        plt.pause(0.1)

        print(f"{file.name} ({len(data)} Frames)")
        key = _wait_for_labeling_key(figure)

        if key == "enter":
            print("✔ behalten\n")
            plt.close(figure)

        elif key == "space":
            file.unlink()
            print("✖ gelöscht\n")
            plt.close(figure)

        elif key == "escape":
            plt.close(figure)
            return

def dataset_building(output_path, base_path="dataset"):

    base = resolve_dataset_dir(base_path)
    output_path = resolve_project_path(output_path)

    X = []
    lengths = []
    labels = []

    if not base.exists():
        print(f"Datensatzordner existiert nicht: {base}")
        return

    classes = sorted(d.name for d in base.iterdir() if d.is_dir())

    for label in classes:

        for file in iter_npy_files(base / label):

            traj = np.load(file)

            if len(traj) < 10:
                continue

            X.append(traj)
            lengths.append(len(traj))
            labels.append(label)

    if not X:
        print(f"Keine verwertbaren .npy-Dateien im Datensatz gefunden: {base}")
        return

    dataset = {
        "X": np.concatenate(X),
        "lengths": lengths,
        "labels": labels,
        "classes": classes
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDatensatz gespeichert: {output_path}")
