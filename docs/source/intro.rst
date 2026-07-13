Erste Schritte
==============

Bevor Sie mit der Implementierung beginnen, müssen Sie das Projekt lokal einrichten.

1. Repository klonen:

.. note::

    Sie sollten mit ``Git`` und ``GitHub`` vertraut sein. Hier wird zu Demonstrationszwecken
    das original Repository geklont. Wenn sie ``Git`` als ``Version-Control-System`` verwenden, klonen sie
    selbstverständlich ihren Fork

.. code-block:: bash

    git clone https://github.com/jaboll-ai/GestureRecognitionMPT
    cd GestureRecognitionMPT

2. Abhängigkeiten installieren:

.. code-block:: bash

    uv sync

Starten Sie die Programme anschließend über die Projektumgebung:

.. code-block:: bash

    uv run python main.py
    uv run python GestureRecognition/start_labeling.py

.. note::

    ``uv sync`` erstellt die lokale ``.venv`` aus ``pyproject.toml`` und
    ``uv.lock``. Die Umgebung wird nicht mit Git übertragen. Nach einem
    Gerätewechsel oder einem erneuten Klonen führen Sie deshalb einfach wieder
    ``uv sync`` aus.

.. tip::

    Ohne ``uv`` können Sie eine eigene virtuelle Umgebung verwenden:

    .. code-block:: bash

        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

    Unter Windows wird die Umgebung mit ``.venv\Scripts\activate`` aktiviert.

3. Download der Recording-Dateien

Die bereitgestellten Daten finden Sie
`hier <https://github.com/jaboll-ai/GestureRecognitionMPT/releases/tag/recordings-v1>`_.

Diese können Sie entweder im Browser oder über die Kommandozeile herunterladen:

.. code-block:: bash

    wget https://github.com/jaboll-ai/GestureRecognitionMPT/releases/download/recordings-v1/recordings.zip

Entpacken Sie die ``.7z``- oder ``.zip``-Datei in Ihren geklonten Projektordner.

4. Testlauf im Replay-Modus:

.. code-block:: bash

    uv run python main.py --mode replay --recorder.file <path_to_recording>.pkl

Ersetzen Sie ``<path_to_recording>`` durch eine der bereitgestellten
Recording-Dateien.

.. note::

    Der Replay-Modus ist der einfachste Einstiegspunkt, da er keine Webcam benötigt.


Grundlagen zum Framework
~~~~~~~~~~~~~~~~~~~~~~~~

Um effizient mit der Aufgabe arbeiten zu können, ist es wichtig,
die grundlegenden Konzepte des bereitgestellten Frameworks zu verstehen.

.. note::

    Sie müssen das Framework **nicht vollständig verstehen**, um zu starten.
    Wichtiger ist:

    - Welche Daten bekommt mein Modul?
    - Welche Daten muss ich zurückgeben?

.. toctree::

    signalhub
