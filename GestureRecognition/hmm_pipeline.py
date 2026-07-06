import argparse
import os
import sys
import time
import pickle
from pathlib import Path
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np

# Ensure imports work both when running as a script and as a package module.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from GestureRecognition.hmmclassifier import HMMClassifier, build_dataset_from_data_dir


def train_model(data_dir: str = "dataset", out_path: str = "dataset/hmm.pkl", min_length: int = 10,
                n_components: int = 6, covariance_type: str = "full", n_iter: int = 300,
                random_state: int | None = 42):
    """Trainiere ein HMM-Modell aus `data_dir/<label>/*.npy`-Dateien."""
    sequences, labels = build_dataset_from_data_dir(data_dir, min_length=min_length)
    if not sequences:
        raise ValueError(f"Keine gültigen Sequenzen in {data_dir} gefunden.")

    clf = HMMClassifier(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    clf.fit(sequences, labels)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(out))
    print(f"[OK] Modell trainiert und gespeichert unter {out}")
    return clf


def evaluate_model(model_path: str = "dataset/hmm.pkl", data_dir: str = "dataset", min_length: int = 10):
    """Bewerte das trainierte Modell anhand der vorhandenen Daten."""
    clf = HMMClassifier.load(model_path)
    sequences, labels = build_dataset_from_data_dir(data_dir, min_length=min_length)

    predictions = clf.predict(sequences)
    correct = sum(1 for pred, true in zip(predictions, labels) if pred == true)
    accuracy = correct / len(labels) if labels else 0.0

    print(f"[OK] Accuracy: {accuracy:.2%} ({correct}/{len(labels)})")
    return accuracy, predictions, labels


def run_live_inference_with_pipeline(model_path: str = "dataset/hmm.pkl", buffer_size: int = 140,
                                      min_steps: int = 15, camera_id: int = 0,
                                      save_dir: str = "dataset/live", label: str = "live"):
    """
    Live-Erkennung mit Visualisierung und Save-Workflow.

    - Zeichnet die Spur der Hand wie TrailMarker
    - korrigiert die Spiegelung der Webcam
    - speichert die Aufnahme wie im Preprocessor
    - klassifiziert später mit dem HMM-Modell
    """
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
    except ImportError:
        raise RuntimeError("MediaPipe erforderlich. Installiere: pip install mediapipe")

    clf = HMMClassifier.load(model_path)
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera_id} konnte nicht geöffnet werden.")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        model_complexity=1
    )

    window_name = "HMM Live Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    history = deque(maxlen=buffer_size)
    trail = deque(maxlen=400)
    prediction = None
    confidence = 0.0
    recording = False
    last_saved_file = None

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("[OK] Live-Inference gestartet. Leertaste = Aufnahme starten/stoppen, ESC = beenden.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Webcam spiegeln, damit sie natürlicher wirkt
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_pt = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark = hand_landmarks.landmark[8]
            current_pt = (int(landmark.x * w), int(landmark.y * h))

            # Spur zeichnen wie TrailMarker
            if current_pt is not None:
                trail.append(current_pt)
                history.append(np.array([landmark.x, landmark.y], dtype=np.float32))

            # Hand-Landmarks zeichnen
            for lm in hand_landmarks.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

        # Zeichne Trail-Linie
        if len(trail) > 1:
            pts = list(trail)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 255, 0), thickness=3)

        # Aufnahme-Status über Leertaste
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            recording = not recording
            if recording:
                history.clear()
                trail.clear()
                print("[OK] Aufnahme gestartet")
            else:
                if len(history) >= min_steps:
                    traj = np.array(history, dtype=np.float32)
                    center = np.mean(traj, axis=0)
                    traj_centered = traj - center
                    max_dist = np.max(np.abs(traj_centered))
                    if max_dist > 0:
                        traj_normalized = traj_centered / max_dist
                    else:
                        traj_normalized = traj_centered

                    timestamp = int(time.time() * 1000)
                    filename = save_path / f"{label}_{timestamp}.npy"
                    np.save(filename, traj_normalized)
                    last_saved_file = filename
                    print(f"[OK] Aufnahme gespeichert unter {filename}")
                else:
                    print("[SKIP] Aufnahme zu kurz, nicht gespeichert")

        if key == 27:
            break

        # Klassifiziere nur, wenn aktuell aufgezeichnet wird und genug Frames da sind
        if recording and len(history) >= min_steps:
            traj = np.array(history, dtype=np.float32)
            center = np.mean(traj, axis=0)
            traj_centered = traj - center
            max_dist = np.max(np.abs(traj_centered))
            if max_dist > 0:
                traj_normalized = traj_centered / max_dist
            else:
                traj_normalized = traj_centered

            prediction = clf.predict([traj_normalized])[0]
            scores = clf.decision_function([traj_normalized])[0]
            confidence = np.exp(scores.max()) / (np.exp(scores).sum() + 1e-8)
        else:
            prediction = None
            confidence = 0.0

        # GUI-Overlay
        cv2.putText(frame, "Leertaste = aufnehmen", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if recording:
            cv2.putText(frame, "Aufnahme: AN", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Aufnahme: AUS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if prediction is not None:
            cv2.putText(frame, f"Prediction: {prediction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Warte auf genug Frames...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, "ESC zum Beenden", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow(window_name, frame)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("[OK] Live-Inference beendet.")


def _prompt_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    choice_text = "/".join(choices)
    while True:
        default_text = f" [{default}]" if default else ""
        response = input(f"{prompt} ({choice_text}){default_text}: ").strip().lower()
        if not response and default:
            return default
        if response in choices:
            return response
        print(f"Ungültiger Wert. Bitte eine der folgenden Optionen wählen: {choice_text}")


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate and run HMM inference")
    parser.add_argument("--mode", choices=["train", "evaluate", "live"], required=False)
    parser.add_argument("--model", default="dataset/hmm.pkl")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--cov", default="full")
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=140)
    parser.add_argument("--min-steps", type=int, default=15)
    args = parser.parse_args()

    if args.mode is None:
        print("Kein Modus angegeben.")
        args.mode = _prompt_choice("Wähle den Modus", ["train", "evaluate", "live"], default="live")

    if args.mode in {"evaluate", "train", "live"} and not args.model:
        args.model = input("Modelldatei (Standard: dataset/hmm.pkl): ").strip() or "dataset/hmm.pkl"

    if args.mode == "train":
        train_model(data_dir=args.data, out_path=args.model, min_length=args.min_length,
                    n_components=args.n_components, covariance_type=args.cov, n_iter=args.n_iter,
                    random_state=args.random_state)
    elif args.mode == "evaluate":
        evaluate_model(model_path=args.model, data_dir=args.data, min_length=args.min_length)
    else:
        run_live_inference_with_pipeline(model_path=args.model, buffer_size=args.buffer_size,
                                        min_steps=args.min_steps, camera_id=args.camera_id)


if __name__ == "__main__":
    main()
