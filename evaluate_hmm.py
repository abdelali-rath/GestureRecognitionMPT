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


def compute_classification_metrics(cm, classes):
    report = []
    n = len(classes)
    total_support = cm.sum()
    macro_prec = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_prec = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for i, label in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        report.append((label, precision, recall, f1, support))

        macro_prec += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_prec += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    macro_prec /= n
    macro_recall /= n
    macro_f1 /= n
    weighted_prec /= total_support if total_support > 0 else 1
    weighted_recall /= total_support if total_support > 0 else 1
    weighted_f1 /= total_support if total_support > 0 else 1

    summary = {
        "macro_precision": macro_prec,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }
    return report, summary


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
    print(f"{'':10s}" + "".join(f"{c:10s}" for c in clf.classes_))
    for i, c in enumerate(clf.classes_):
        print(f"{c:10s}" + "".join(f"{cm[i, j]:10d}" for j in range(len(clf.classes_))))
    print()

    # Metrics
    report, summary = compute_classification_metrics(cm, clf.classes_)
    print("📈 Klassifikationsmetriken:")
    print(f"{'Klasse':10s} {'Precision':10s} {'Recall':10s} {'F1':10s} {'Support':10s}")
    for label, precision, recall, f1, support in report:
        print(f"{label:10s} {precision:10.2%} {recall:10.2%} {f1:10.2%} {support:10d}")

    print("\n📊 Durchschnittswerte:")
    print(f"Macro Precision:    {summary['macro_precision']:.2%}")
    print(f"Macro Recall:       {summary['macro_recall']:.2%}")
    print(f"Macro F1:           {summary['macro_f1']:.2%}")
    print(f"Weighted Precision: {summary['weighted_precision']:.2%}")
    print(f"Weighted Recall:    {summary['weighted_recall']:.2%}")
    print(f"Weighted F1:        {summary['weighted_f1']:.2%}")


if __name__ == "__main__":
    main()
