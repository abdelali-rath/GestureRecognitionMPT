import os
import shutil
import numpy as np
from pathlib import Path
from pynput import keyboard 

# REPARATUR: Da diese Datei im selben Ordner wie labeling.py liegt,
# importieren wir es direkt, ohne "GestureRecognition." davor!
from labeling import dataset_building
if __name__ == "__main__":
    import sys
    # Wir importieren matplotlib für die visuelle Anzeige
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("ℹ️ Installiere kurz matplotlib für die Bildanzeige...")
        os.system(f"{sys.executable} -m pip install matplotlib")
        import matplotlib.pyplot as plt
    
    print("Möchtest du:")
    print("[1] Deine aufgenommenen Gesten interaktiv labeln/sortieren (MIT VORSCHAU)")
    print("[2] Den finalen Datensatz für das HMM-Training erstellen")
    
    auswahl = input("\nBitte wähle (1 oder 2): ").strip()
    
    if auswahl == "1":
        geste = input("Welche Geste möchtest du labeln? (z.B. P oder A): ").strip()
        
        def data_labeling_fixed(times: int, label: str):
            temp_dir = f"dataset/{label}"  
            final_dir = f"dataset/{label}_ready"
            
            os.makedirs(final_dir, exist_ok=True)
            if not os.path.exists(temp_dir) or len(os.listdir(temp_dir)) == 0:
                print(f"❌ Keine Aufnahmen in '{temp_dir}' gefunden.")
                return

            temp_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.npy')])

            print(f"\n=== 🏷️ Labeling-Interface für Geste: '{label}' ===")
            print(f"Steuerung: [ENTER] = Behalten & Verschieben | [SPACE] = Löschen | [ESC] = Abbrechen\n")

            with keyboard.Events() as events:
                for file in temp_files:
                    file_path = os.path.join(temp_dir, file)
                    try: 
                        data = np.load(file_path)
                    except: 
                        continue
                    
                    # ---------------------------------------------------------------
                    # 👁️ VISUELLE VORSCHAU ERZEUGEN
                    # ---------------------------------------------------------------
                    plt.figure("Gesten-Vorschau", figsize=(5, 5))
                    plt.clf() # Fenster leeren für die nächste Geste
                    
                    # data[:, 0] sind alle X-Werte, data[:, 1] alle Y-Werte
                    plt.plot(data[:, 0], data[:, 1], '-o', color='darkcyan', markersize=4)
                    
                    # Start- und Endpunkt markieren, damit man sieht, wie rum gezeichnet wurde
                    plt.scatter(data[0, 0], data[0, 1], color='green', s=100, label='Start', zorder=5)
                    plt.scatter(data[-1, 0], data[-1, 1], color='red', s=100, label='Ende', zorder=5)
                    
                    plt.title(f"Datei: {file}\n[ENTER] = Behalten | [SPACE] = Löschen")
                    plt.grid(True)
                    plt.legend()
                    
                    # Da der Preprocessor die Daten normalisiert hat, fixieren wir das Fenster
                    plt.xlim(-1.2, 1.2)
                    plt.ylim(-1.2, 1.2)
                    
                    # WICHTIG: In der Computer Vision ist Y=0 oben. Wir drehen die Achse um,
                    # damit die Geste nicht auf dem Kopf steht!
                    plt.gca().invert_yaxis() 
                    
                    plt.show(block=False) # Öffnet das Fenster ohne das Terminal zu blockieren
                    plt.pause(0.1)        # Kurz warten, damit Mac das Fenster zeichnen kann
                    # ---------------------------------------------------------------
                    
                    print(f"📋 {file} ({len(data)} Frames). Behalten? [ENTER/SPACE/ESC]: ", end="", flush=True)
                    
                    while True:
                        event = events.get(timeout=None)
                        if isinstance(event, keyboard.Events.Press):
                            if event.key == keyboard.Key.enter:
                                shutil.move(file_path, os.path.join(final_dir, file))
                                print("➡️ BEHALTEN (Verschoben nach _ready)")
                                plt.close() # Fenster für diese Datei schließen
                                break
                            elif event.key == keyboard.Key.space:
                                os.remove(file_path)
                                print("🗑️ GELÖSCHT.")
                                plt.close() # Fenster für diese Datei schließen
                                break
                            elif event.key == keyboard.Key.esc:
                                print("\n🛑 Abgebrochen.")
                                plt.close()
                                return
        
        data_labeling_fixed(times=30, label=geste)
        
    elif auswahl == "2":
        dataset_building("dataset/gesamt_dataset.pkl")
    else:
        print("Ungültige Auswahl.")