import pickle
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard

try:
    from .paths import iter_npy_files, resolve_dataset_dir, resolve_label_dir, resolve_project_path
except ImportError:
    from paths import iter_npy_files, resolve_dataset_dir, resolve_label_dir, resolve_project_path


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


    with keyboard.Events() as events:

        for file in files:


            data = np.load(file)

            plt.figure("Vorschau", figsize=(5,5))
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

            while True:

                event = events.get()

                if not isinstance(event, keyboard.Events.Press):
                    continue

                if event.key == keyboard.Key.enter:

                    print("✔ behalten\n")
                    plt.close()
                    break

                elif event.key == keyboard.Key.space:

                    file.unlink()
                    print("✖ gelöscht\n")
                    plt.close()
                    break

                elif event.key == keyboard.Key.esc:

                    plt.close()
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
