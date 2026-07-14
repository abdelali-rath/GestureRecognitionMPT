Fit/Train HMM Classifier
========================

Der Klassifikator trainiert für jede Gestenklasse ein eigenes
``hmmlearn.hmm.GaussianHMM``. Die Projektlogik gruppiert variable Sequenzen pro
Label, vergleicht ihre Log-Likelihoods und speichert die Klassenmodelle.

Training
--------

Das vorhandene Labeling-Menü enthält den vollständigen Workflow:

.. code-block:: bash

   uv run python GestureRecognition/start_labeling.py

1. Aufnahmen prüfen
2. ``dataset/gesamt_dataset.pkl`` erstellen
3. HMM trainieren und evaluieren
4. Separate Holdout-Evaluation als Confusion Matrix anzeigen

Beim Training werden 20 Prozent jeder Klasse als Testdaten zurückgehalten. Die
Ausgabe enthält Genauigkeit und Confusion-Matrix. Anschließend wird das finale
Modell mit allen Sequenzen trainiert und unter ``dataset/hmm.pkl`` gespeichert.

``C_ready`` und ``P_ready`` werden dabei den Klassen ``C`` und ``P`` zugeordnet;
unbeschriftete ``live``-Aufnahmen werden ignoriert.

Live-Modus
----------

Das ``HMMModule`` lädt das Modell beim Start. Der Pfad und die Mindestdifferenz
zwischen bestem und zweitbestem Score stehen in ``config.yml``:

.. code-block:: yaml

   hiddenmarkov:
     model_path: dataset/hmm.pkl
     min_margin: 0.5
     display_seconds: 4.0

Danach startet die Live-Erkennung mit:

.. code-block:: bash

   uv run python main.py

Das SignalHub-Fenster muss den Tastaturfokus besitzen. ``R`` startet die
Aufzeichnung einer Geste, ein zweites ``R`` beendet sie und übergibt die
normalisierte Trajektorie an das ``HMMModule``. Das erkannte Label bleibt einige
Sekunden im Kamerabild sichtbar. Live-Testgesten werden nicht in den
Trainingsdatensatz geschrieben, solange ``preprocessor.save_recordings`` auf
``false`` steht.

API-Referenz
------------

.. automodule:: GestureRecognition.hmmclassifier
   :members:
   :show-inheritance:
