"""Evaluiere HMM-Modell direkt ohne Package imports."""
from pathlib import Path
from collections import defaultdict
import sys
import pickle

import numpy as np

# Lade Funktionen direkt aus der Datei
import importlib.util
spec = importlib.util.spec_from_file_location(
    "hmmclassifier",
    Path(__file__).parent / "GestureRecognition" / "hmmclassifier.py"
)
hmmclassifier = importlib.util.module_from_spec(spec)
sys.modules["hmmclassifier"] = hmmclassifier
spec.loader.exec_module(hmmclassifier)

HMMClassifier = hmmclassifier.HMMClassifier
build_dataset_from_data_dir = hmmclassifier.build_dataset_from_data_dir

# Load model
model_path = "dataset/hmm.pkl"
if not Path(model_path).exists():
    print(f"❌ Modell nicht gefunden: {model_path}")
    sys.exit(1)

clf = HMMClassifier.load(model_path)
print(f"✓ Modell geladen: {model_path}")
print(f"  Klassen: {clf.classes_}")
print(f"  n_components: {clf.n_components}\n")

# Load data
try:
    seqs, labs = build_dataset_from_data_dir("dataset", min_length=10)
except Exception as e:
    print(f"❌ Fehler beim Laden der Daten: {e}")
    sys.exit(1)

if not seqs:
    print("❌ Keine Sequenzen gefunden")
    sys.exit(1)

print(f"✓ {len(seqs)} Sequenzen geladen\n")

# Show distribution
dist = defaultdict(int)
for lab in labs:
    dist[lab] += 1
print(f"📊 Klassenverteilung: {dict(dist)}\n")

# Predict
preds = clf.predict(seqs)

# Accuracy
correct = sum(1 for t, p in zip(labs, preds) if t == p)
acc = correct / len(labs)
print(f"✓ Accuracy: {correct}/{len(labs)} = {acc:.2%}\n")

# Confusion matrix
cm = {}
for true_label in clf.classes_:
    cm[true_label] = defaultdict(int)

for true, pred in zip(labs, preds):
    cm[true][pred] += 1

# Print matrix
print("🔥 Confusion Matrix:")
print(f"{'':5s} " + " ".join(f"{c:8s}" for c in clf.classes_))
for true_label in clf.classes_:
    row = " ".join(f"{cm[true_label].get(pred_label, 0):8d}" for pred_label in clf.classes_)
    print(f"{true_label:5s} {row}")
