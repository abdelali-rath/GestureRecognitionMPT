import os
import shutil
import numpy as np
try:
    import msvcrt  # Für Windows (dein MINGW64 Terminal)
except ImportError:
    pass
import pickle
from pathlib import Path

def data_labeling(times: int, label: str):
    """
    TODO: data_labeling: Datenerfassung für Gesten (SignalHub)

    Ziel:
    -----
    Implementiere eine Funktion, mit der Trainingsdaten für eine bestimmte
    Geste aufgenommen und gespeichert werden können.

    Anforderungen / Ideen:
    ----------------------

    1. Aufnahme starten

       - Starte SignalHub über einen Subprocess
       - Übergib einen Dateipfad für die Aufnahme
       - Überlege, welche Module aufgenommen werden sollen
       - Nimm entsprechende Änderungen in der ``config.yaml`` vor

    2. Interaktive Steuerung (optional)

       - Implementiere eine einfache Benutzerinteraktion:
         - Aufnahme speichern
         - Aufnahme verwerfen
         - Programm beenden

    .. tip::

       Die Funktion ``getch()`` (Aus dem Modul Linux :mod:`getch` oder bei Windows :mod:`msvcrt`) ist sehr hilfreich, um einzelne Tastendrücke
       direkt auszulesen (ohne Enter). Damit kannst du dir ein schnelles
       Labeling-Interface bauen.

       Beispiel:

       .. code-block:: text

           ESC → speichern
           andere Taste → verwerfen

    3. Daten sichten und bereinigen

       - Lade die aufgenommenen Daten
       - Überlege:
         - Welche Teile sind relevant?
         - Welche Frames sind leer oder unbrauchbar?
         - Sollten gewisse Sequenzen evtl. gar nicht benutzt werden?
       - Entferne unnötige Anteile (z. B. keine erkannte Hand am Anfang/Ende)

    4. Speicherung

       - Speichere Daten strukturiert nach Labels (z. B. Ordnerstruktur)
       - Jede Aufnahme sollte einzeln gespeichert werden

    .. note::

       Die konkrete Umsetzung (Dateiformat, Struktur, Ablauf) ist bewusst offen.
       Entwickle ein System, das für dich sinnvoll ist und sich gut weiterverarbeiten lässt.

    .. warning::

       Ziel ist nicht nur, dass es „funktioniert“, sondern ein sauberer und
       effizienter Workflow für Datensammlung.

    Parameters
    ----------
    times : int
        Wie viele Aufnahmen gemacht werden sollen.
        Kann frei angepasst werden (z. B. Endlosschleife oder interaktive Steuerung).

    label : str
        Name der Geste / Klasse.
        Kann ebenfalls frei gestaltet werden (z. B. dynamische Labels, mehrere Klassen gleichzeitig).
    """
    pass


    temp_dir = "data/temp"
    final_dir = f"data/{label}"
    
    # Ordner erstellen, falls nicht vorhanden
    os.makedirs(final_dir, exist_ok=True)
    
    if not os.path.exists(temp_dir) or len(os.listdir(temp_dir)) == 0:
        print(f"Keine temporären Aufnahmen in '{temp_dir}' gefunden.")
        print("Bitte starte zuerst die SignalHub-Pipeline, um Aufnahmen zu machen!")
        return

    saved_count = 0
    temp_files = [f for f in os.listdir(temp_dir) if f.endswith('.npy')]

    print(f"\n=== Labeling für Geste: '{label}' gestartet ===")
    print("Steuerung: [ENTER] = Speichern | [SPACE] = Verwerfen | [ESC] = Beenden\n")

    for file in temp_files:
        if saved_count >= times:
            print(f"Ziel von {times} Aufnahmen erreicht!")
            break

        file_path = os.path.join(temp_dir, file)
        data = np.load(file_path)
        
        # Zeige Infos zur aktuellen Geste an
        print(f"Aufnahme {file}: {len(data)} Frames lang. Behalten?")
        
        # Interaktive Eingabe
        while True:
            key = ord(msvcrt.getch())
            if key == 13:  # ENTER -> Behalten
                new_filename = f"{label}_{len(os.listdir(final_dir))}.npy"
                shutil.move(file_path, os.path.join(final_dir, new_filename))
                saved_count += 1
                print(f" -> Gespeichert als {new_filename} ({saved_count}/{times})")
                break
            elif key == 32:  # LEERTASTE -> Verwerfen
                os.remove(file_path)
                print(" -> Verworfen.")
                break
            elif key == 27:  # ESC -> Abbrechen
                print("\nLabeling abgebrochen.")
    return




def dataset_building(output_path):
    """
    TODO: dataset_building: Trainingsdatensatz aus aufgenommenen Gesten erstellen

    Ziel:
    -----
    Implementiere eine Funktion, die alle aufgenommenen Daten lädt,
    verarbeitet und in eine Form bringt, die von eurem
    Hidden-Markov-Modell (HMM) Classifier verwendet werden kann.

    Anforderungen / Ideen:
    ----------------------

    1. Daten laden

       - Durchsuche deinen Trainingsdaten-Ordner
       - Organisiere Daten nach Labels

    2. Feature-Extraktion / Preprocessing

       - Überlege:
         - Welche Features braucht dein Modell?
         - Wie transformierst du die Rohdaten sinnvoll?
       - Wende eine konsistente Verarbeitung auf alle Sequenzen an

    3. Umgang mit Sequenzen

       - Daten sind zeitliche Sequenzen
       - Achte auf:
         - Unterschiedliche Längen
         - Konsistente Struktur

    4. Validierung

       - Entferne unbrauchbare Daten
         (z. B. zu kurze oder fehlerhafte Sequenzen)

    5. Ausgabeformat

       - Baue den Datensatz so, dass dein HMM direkt damit arbeiten kann
       - Das Format sollst du selbst definieren

    .. note::

       Es gibt hier keine vorgegebene „richtige“ Lösung.
       Wichtig ist, dass dein Datensatz konsistent und nutzbar ist.

    .. tip::

       Denke wie ein System-Designer:
       Wie müssen Daten aussehen, damit Training und Inferenz sauber funktionieren?

    .. warning::

       Inkonsistente Datenstrukturen sind eine der häufigsten Fehlerquellen
       beim Training von Sequenzmodellen.

    Erweiterung (optional):
    -----------------------

    - Normalisierung der Daten
    - Datenaugmentation
    - Debug-Ausgaben oder Visualisierung

    Parameters
    ----------
    output_path : Path or str
        Zielpfad für den erzeugten Trainingsdatensatz.
    """
    pass
    base_dir = Path("data")
    
    X = []        # Hier kommen alle Koordinaten rein
    lengths = []  # Hier kommt die Länge jeder Geste rein
    labels = []   # Hier kommt der Name der Geste rein (z.B. "kreis")
    
    valid_classes = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "temp"]
    
    print(f"Erstelle Datensatz aus den Klassen: {valid_classes}")
    
    for gesture_class in valid_classes:
        class_dir = base_dir / gesture_class
        files = list(class_dir.glob("*.npy"))
        
        for file in files:
            trajectory = np.load(file)
            
            # Validierung: Sind genug Datenpunkte da? (z.B. mind. 10 Frames)
            if len(trajectory) < 10:
                print(f"Ignoriere {file.name} (zu kurz: {len(trajectory)} Frames)")
                continue
                
            # Daten hinzufügen
            X.append(trajectory)
            lengths.append(len(trajectory))
            labels.append(gesture_class)
            
    if not X:
        print("Fehler: Keine gültigen Daten gefunden!")
        return

    # hmmlearn erwartet X als ein zusammenhängendes 2D Array
    X_concat = np.concatenate(X)
    
    # Datensatz zusammenbauen
    dataset = {
        "X": X_concat,
        "lengths": lengths,
        "labels": labels,
        "classes": valid_classes
    }
    
    # Speichern als Pickle-Datei (Perfekt für Python-Machine-Learning)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f" Datensatz erfolgreich gespeichert unter: {output_file}")
    print(f"   Gesamtpunkte: {len(X_concat)}, Anzahl Gesten: {len(lengths)}")