"""Evaluiere das trainierte HMM-Modell."""
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

from GestureRecognition.hmmclassifier import HMMClassifier, build_dataset_from_data_dir


def compute_confusion_matrix(y_true, y_pred, classes):
    """Einfache Confusion Matrix ohne sklearn."""
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    class_idx = {c: i for i, c in enumerate(classes)}
    for true, pred in zip(y_true, y_pred):
        i, j = class_idx[true], class_idx[pred]
        cm[i, j] += 1
    
    return cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate HMMClassifier model")
    parser.add_argument("--model", default="dataset/hmm.pkl", help="Path to trained model")
    parser.add_argument("--data", default="dataset", help="Base data directory")
    parser.add_argument("--min-length", type=int, default=10)
    args = parser.parse_args()

    # Lade Modell
    if not Path(args.model).exists():
        print(f"Modell nicht gefunden: {args.model}")
        return

    clf = HMMClassifier.load(args.model)
    print(f"✓ Modell geladen: {args.model}")
    print(f"  Klassen: {clf.classes_}")
    print(f"  n_components: {clf.n_components}")
    print()

    # Lade Daten
    seqs, labs = build_dataset_from_data_dir(args.data, min_length=args.min_length)
    if not seqs:
        print("Keine Sequenzen gefunden.")
        return

    print(f"✓ {len(seqs)} Sequenzen geladen aus {args.data}/")
    
    # Zeige Verteilung
    dist = defaultdict(int)
    for lab in labs:
        dist[lab] += 1
    print(f"  Verteilung: {dict(dist)}\n")

    # Predictions
    preds = clf.predict(seqs)

    # Accuracy
    correct = sum(1 for true, pred in zip(labs, preds) if true == pred)
    acc = correct / len(labs) if labs else 0
    print(f"📊 Accuracy: {correct}/{len(labs)} = {acc:.2%}\n")

    # Confusion Matrix
    cm = compute_confusion_matrix(labs, preds, clf.classes_)
    print(f"🔥 Confusion Matrix:")
    print(f"{'':5s} " + " ".join(f"{c:5s}" for c in clf.classes_))
    for i, c in enumerate(clf.classes_):
        print(f"{c:5s} " + " ".join(f"{cm[i, j]:5d}" for j in range(len(clf.classes_))))


if __name__ == "__main__":
    main()
