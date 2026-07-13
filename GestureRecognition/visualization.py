import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def visualize_dataset(base_path="dataset", label="A", start=0, stop=5):
    search_pattern = os.path.join(base_path, label, "*.npy")
    dateien = glob.glob(search_pattern)
    
    if not dateien:
        print(f"⚠️ Keine Daten für '{label}' gefunden! Suchpfad: {search_pattern}")
        return

    plt.figure(figsize=(8, 8))
    plt.title(f"Datenexploration: Trajektorien der Klasse '{label}'")

    for datei_pfad in dateien[start:stop]:
        traj = np.load(datei_pfad)
        plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=3, label=os.path.basename(datei_pfad))

    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper right", fontsize="small")
    plt.xlabel("X-Koordinate (Normalisiert)")
    plt.ylabel("Y-Koordinate (Normalisiert)")
    plt.show()

def replay_recordings(base_path="dataset", label="I"):
    search_pattern = os.path.join(base_path, label, "*.npy")
    dateien = glob.glob(search_pattern)
    
    if not dateien:
        return

    datei_pfad = dateien[0]
    traj = np.load(datei_pfad)
    x, y = traj[:, 0], traj[:, 1]
    
    plt.figure(figsize=(6, 6))
    plt.title(f"Replay-Modus: {os.path.basename(datei_pfad)}")
    plt.gca().invert_yaxis()
    plt.xlim(-1.1, 1.1)
    plt.ylim(1.1, -1.1) 
    plt.grid(True, linestyle='--', alpha=0.5)
    
    for i in range(1, len(x)):
        plt.plot(x[:i], y[:i], color='blue', linewidth=2, marker='o', markersize=4)
        plt.pause(0.05)
        
    plt.show()

# ==========================================
# HIER IST DER START-KNOPF
# ==========================================
if __name__ == "__main__":
    print("Starte Datenexploration...")
    visualize_dataset(label="D", start=0, stop=1)