# GestureRecognitionMPT

MPT Projekt zur Erkennung von Gesten in Webcam-Daten.

Dafür werden Hand-Landmarks extrahiert und anschließend mit einem [Hidden-Markov-Modell](https://de.wikipedia.org/wiki/Hidden_Markov_Model) (HMM) klassifiziert.

Die Online-Dokumentation zur Bearbeitung des Projekts finden sie [hier](https://jaboll-ai.github.io/GestureRecognitionMPT).

## Installation

Empfohlen wird [uv](https://docs.astral.sh/uv/), damit auf Linux, macOS und
Windows dieselbe Projektkonfiguration verwendet wird. Nach dem Klonen oder auf
einem neuen Gerät genügen:

```bash
git clone git@github.com:abdelali-rath/GestureRecognitionMPT.git
cd GestureRecognitionMPT
uv sync
```

Programme anschließend immer über die Projektumgebung starten:

```bash
uv run python main.py
uv run python GestureRecognition/start_labeling.py
```

`uv sync` erstellt die lokale `.venv` und installiert alle Abhängigkeiten aus
`pyproject.toml` und `uv.lock`. Die `.venv` wird absichtlich nicht mit Git
übertragen und muss auf jedem Gerät neu erzeugt werden.

Alternativ kann eine eigene virtuelle Umgebung mit
`pip install -r requirements.txt` eingerichtet werden.

## HMM trainieren

Über `GestureRecognition/start_labeling.py` können die Aufnahmen geprüft, der
Datensatz erstellt und das HMM trainiert werden:

```bash
uv run python GestureRecognition/start_labeling.py
```

Menüpunkt 2 erzeugt `dataset/gesamt_dataset.pkl`. Menüpunkt 3 reserviert 20 %
jeder Klasse für die Evaluation, zeigt Genauigkeit und Confusion-Matrix an und
trainiert anschließend das finale Modell mit allen Sequenzen. Es wird als
`dataset/hmm.pkl` gespeichert. Menüpunkt 4 trainiert ein separates
Evaluationsmodell auf 80 % der Daten und zeigt die Accuracy sowie eine
Confusion Matrix für die übrigen 20 % an, ohne das finale Modell zu verändern.

Anschließend startet die Live-Erkennung mit:

```bash
uv run python main.py
```

Klicken Sie einmal in das Kamerafenster, damit es den Tastaturfokus besitzt.
Mit `R` oder `Shift+R` wird die Aufnahme gestartet und mit erneutem Drücken
beendet und klassifiziert. Das Ergebnis wird vier Sekunden im Kamerabild und
zusätzlich im Terminal angezeigt. Live-Testgesten werden standardmäßig nicht
als Trainingsdaten gespeichert.

## Pipeline

Die Verarbeitung erfolgt über mehrere Module:
```
Webcam → HandDetector → Preprocessor → HMMModule
```
- **HandDetector**
  Erkennt Hände im Kamerabild und extrahiert deren Landmarken. (optional: Darstellung der Hand)

- **Preprocessor**
  Sammelt und normalisiert Fingertrajektorien über mehrere Frames.

- **HMMModule**
  Klassifiziert Gesten mithilfe eines trainierten Hidden-Markov-Modells.

- **TrailMarker**
  Optionales Modul zur Visualisierung der Fingerbewegung.

<table>
<tr>
<td><img src="https://github.com/user-attachments/assets/f954735c-e8cb-4a82-9c38-4c748eb90dd4" width="250"></td>
<td><img src="https://github.com/user-attachments/assets/1ac89dba-d959-4a57-9ae3-a8db4629e1a3" width="250"></td>
<td><img src="https://github.com/user-attachments/assets/49a4a880-4def-4dc3-b807-c078870aa4f8" width="250"></td>
</tr>
<tr>
<td><img src="https://github.com/user-attachments/assets/c3947875-1300-414a-b939-96889eb490b6" width="250"></td>
<td><img src="https://github.com/user-attachments/assets/2e766180-9ecf-4434-a7a3-f0cf52b9b53e" width="250"></td>
<td><img src="https://github.com/user-attachments/assets/a85aa1e0-fe16-44f6-a180-c443b502a92b" width="250"></td>
</tr>
</table>


<img width="830" height="1430" alt="Dataset" src="https://github.com/user-attachments/assets/dd61fa9d-353a-46ed-adea-7a28238e1f9e" />
