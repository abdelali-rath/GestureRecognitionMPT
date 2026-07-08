import os
import shutil
import numpy as np
import pickle
from pathlib import Path
from pynput import keyboard  # Mac-kompatibler Ersatz für msvcrt!

def data_labeling(times: int, label: str):
    """
    Klassifiziert und filtert aufgezeichnete Gesten interaktiv aus dem Temp-Ordner.
    """
    temp_dir = "dataset/A"
    final_dir = f"dataset/{label}"
    
    os.makedirs(final_dir, exist_ok=True)
    
    if not os.path.exists(temp_dir) or len(os.listdir(temp_dir)) == 0:
        print(f"❌ Keine temporären Aufnahmen in '{temp_dir}' gefunden.")
        print("Bitte starte zuerst die SignalHub-Pipeline und nimm Gesten mit [LEERTASTE] auf!")
        return

    saved_count = len(os.listdir(final_dir))
    temp_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.npy')])

    print(f"\n=== 🏷️ Labeling-Interface für Geste: '{label}' ===")
    print(f"Bisher existieren {saved_count} Aufnahmen für dieses Label.")
    print("Steuerung: [ENTER] = Speichern | [SPACE] = Verwerfen | [ESC] = Abbrechen\n")

    # Wir belauschen die Tastatur-Events direkt im Terminal
    with keyboard.Events() as events:
        for file in temp_files:
            if saved_count >= times:
                print(f"🎉 Ziel von {times} Aufnahmen für '{label}' erreicht!")
                break

            file_path = os.path.join(temp_dir, file)
            try:
                data = np.load(file_path)
            except Exception:
                continue
            
            print(f"📋 Datei {file}: {len(data)} Frames lang. Behalten? ", end="", flush=True)
            
            # Warten auf genau einen Tastendruck
            while True:
                event = events.get(timeout=None)  # Blockiert, bis eine Taste gedrückt wird
                if isinstance(event, keyboard.Events.Press):
                    if event.key == keyboard.Key.enter:
                        # Geste akzeptieren und verschieben
                        new_filename = f"{label}_{int(os.path.getmtime(file_path))}.npy"
                        shutil.move(file_path, os.path.join(final_dir, new_filename))
                        saved_count += 1
                        print(f"➡️ SPEICHERN unter {new_filename} ({saved_count}/{times})")
                        break
                        
                    elif event.key == keyboard.Key.space:
                        # Geste löschen
                        os.remove(file_path)
                        print("🗑️ VERWORFEN.")
                        break
                        
                    elif event.key == keyboard.Key.esc:
                        print("\n🛑 Labeling durch Benutzer abgebrochen.")
                        return
    return


def dataset_building(output_path):
    """
    Erstellt den finalen .pkl-Datensatz für das Hidden-Markov-Modell (HMM).
    """
    base_dir = Path("dataset")
    
    X = []        # Sequenzdaten
    lengths = []  # Längen der einzelnen Sequenzen
    labels = []   # Zugehörige Klassenlabels
    
    # Alle Ordner außer 'temp' durchsuchen
    valid_classes = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "temp"]
    
    print(f"\n📦 Erstelle Datensatz aus den Klassen: {valid_classes}")
    
    for gesture_class in valid_classes:
        class_dir = base_dir / gesture_class
        files = list(class_dir.glob("*.npy"))
        
        for file in files:
            trajectory = np.load(file)
            
            # Validierung aus der TODO (Zu kurze Sequenzen aussortieren)
            if len(trajectory) < 10:
                print(f"  ⚠️ Ignoriere {file.name} (zu kurz: {len(trajectory)} Frames)")
                continue
                
            X.append(trajectory)
            lengths.append(len(trajectory))
            labels.append(gesture_class)
            
    if not X:
        print("❌ Fehler: Keine gültigen Daten für den Datensatz gefunden!")
        return

    # Für hmmlearn müssen alle Sequenzen vertikal konkateniert werden (N, 2)
    X_concat = np.concatenate(X)
    
    dataset = {
        "X": X_concat,
        "lengths": lengths,
        "labels": labels,
        "classes": valid_classes
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"\n💾 Datensatz erfolgreich gespeichert unter: {output_file}")
    print(f"   Gesamtpunkte (Features): {len(X_concat)} | Anzahl Gesten: {len(lengths)}")