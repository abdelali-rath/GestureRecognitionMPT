import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def visualize_dataset(dataset_dir="dataset", label="P", start=0, stop=5):
    """
    Visualisiert gezielt einen Buchstaben und einen bestimmten Bereich von Aufnahmen.
    """
    search_path = os.path.join(dataset_dir, label, "*.npy")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"⚠️ Keine Daten für den Buchstaben '{label}' im Ordner '{dataset_dir}' gefunden!")
        return

    selected_files = files[start:stop]

    plt.figure(figsize=(10, 6))
    
    for idx, f in enumerate(selected_files):
        data = np.load(f)
        
        current_index = start + idx
        plt.plot(data[:, 0], data[:, 1], alpha=0.7, 
                 label=f"{label} (Aufnahme {current_index})")

    plt.title(f"Trajektorien für Klasse '{label}' (Aufnahmen {start} bis {stop-1})")
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    
    plt.gca().invert_yaxis() 
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def replay_recordings(dataset_dir="dataset", label="P", count=3):
    """
    Exploration und Replay der aufgenommenen Rohdaten.
    Passt das Sichtfenster automatisch an die echten Daten an.
    """
    search_path = os.path.join(dataset_dir, label, "*.npy")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"⚠️ Keine Aufnahmen für '{label}' zum Abspielen gefunden.")
        return

    print(f"🎬 Starte Replay für Klasse '{label}'. Schließe das Fenster für die nächste Geste...")
    
    for f in files[:count]:
        data = np.load(f)
            
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"Replay '{label}': {os.path.basename(f)}")
        
        x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
        y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
        
        pad_x = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 1
        pad_y = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 1
        
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        
        for i in range(1, len(data) + 1):
            ax.plot(data[:i, 0], data[:i, 1], color="blue", marker="o", markersize=3, linestyle="-")
            plt.pause(0.03)
            
        plt.show()

    if __name__ == "__main__":
        print("Starte Datenexploration...")

        visualize_dataset(label="P", start=0, stop=4)

        replay_recordings(label="P", count=2)