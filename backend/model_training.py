import argparse
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


DEFAULT_EMBEDDINGS_PATH = os.path.join("embeddings", "faces_embeddings_done_4classes.npz")
DEFAULT_MODEL_PATH = os.path.join("model", "svm_model_160x160.pkl")
DEFAULT_ENCODER_PATH = os.path.join("model", "label_encoder.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SVM face recognizer from embeddings NPZ.")
    parser.add_argument(
        "--embeddings-path",
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to embeddings npz file (default: embeddings/faces_embeddings_done_4classes.npz).",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Output path for trained SVM model.",
    )
    parser.add_argument(
        "--encoder-path",
        default=DEFAULT_ENCODER_PATH,
        help="Output path for label encoder.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split ratio (default: 0.25).")
    parser.add_argument("--random-state", type=int, default=17, help="Random seed (default: 17).")
    return parser.parse_args()


def load_embeddings(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Embeddings file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if "arr_0" in data and "arr_1" in data:
        x, y = data["arr_0"], data["arr_1"]
    elif "X" in data and "Y" in data:
        x, y = data["X"], data["Y"]
    else:
        raise ValueError("Unsupported embeddings format. Expected keys arr_0/arr_1 or X/Y.")

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=object)
    return x, y


def main():
    args = parse_args()
    x, y_raw = load_embeddings(args.embeddings_path)

    if x.shape[0] == 0:
        raise ValueError("No embeddings found in input file.")

    unique_classes, class_counts = np.unique(y_raw, return_counts=True)
    if unique_classes.shape[0] < 2:
        raise ValueError("Need at least 2 classes to train SVM.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    can_stratify = np.min(class_counts) >= 2
    stratify_target = y if can_stratify else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        shuffle=True,
        random_state=args.random_state,
        stratify=stratify_target,
    )

    model = SVC(kernel="linear", probability=True)
    model.fit(x_train, y_train)

    ypreds_train = model.predict(x_train)
    ypreds_test = model.predict(x_test)

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.encoder_path) or ".", exist_ok=True)

    with open(args.model_path, "wb") as f:
        pickle.dump(model, f)

    with open(args.encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    print(f"[INFO] Classes: {unique_classes.shape[0]}")
    print(f"[INFO] Samples: {x.shape[0]}")
    print(f"[INFO] Stratified split: {'yes' if can_stratify else 'no'}")
    print(f"[INFO] Train accuracy: {accuracy_score(y_train, ypreds_train):.4f}")
    print(f"[INFO] Test accuracy: {accuracy_score(y_test, ypreds_test):.4f}")
    print(f"[INFO] Saved model: {args.model_path}")
    print(f"[INFO] Saved encoder: {args.encoder_path}")


if __name__ == "__main__":
    main()
