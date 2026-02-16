import argparse
import os
import time

import cv2 as cv
import imgaug.augmenters as iaa
import numpy as np
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDINGS_PATH = os.path.join("embeddings", "faces_embeddings_done_4classes.npz")
TARGET_SIZE = (160, 160)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build FaceNet embeddings from data/<person>/*.jpg. "
            "Supports incremental updates for selected people."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing person folders (default: data).",
    )
    parser.add_argument(
        "--embeddings-path",
        default=DEFAULT_EMBEDDINGS_PATH,
        help="NPZ output path used by model_training.py.",
    )
    parser.add_argument(
        "--people",
        default="",
        help='Comma-separated list of person folder names to process, e.g. "Alice,Bob". Empty = all.',
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Before appending, remove existing embeddings for selected people. "
            "Recommended when refreshing a person."
        ),
    )
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Ignore existing NPZ and rebuild from selected folders only (or all if --people is empty).",
    )
    parser.add_argument(
        "--augmentations",
        type=int,
        default=3,
        help="Number of augmented copies per detected face image (default: 3).",
    )
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=0,
        help="Optional cap per person folder. 0 means no cap.",
    )
    return parser.parse_args()


def list_people(data_dir):
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    )


def parse_people_arg(raw_people):
    if not raw_people.strip():
        return []
    return [p.strip() for p in raw_people.split(",") if p.strip()]


def load_existing_embeddings(npz_path):
    if not os.path.exists(npz_path):
        return np.empty((0, 512), dtype=np.float32), np.empty((0,), dtype=object)

    data = np.load(npz_path, allow_pickle=True)
    if "arr_0" in data and "arr_1" in data:
        emb, labels = data["arr_0"], data["arr_1"]
    elif "X" in data and "Y" in data:
        emb, labels = data["X"], data["Y"]
    else:
        raise ValueError(f"Unsupported embeddings format in {npz_path}")

    emb = np.asarray(emb, dtype=np.float32)
    labels = np.asarray(labels, dtype=object)
    return emb, labels


def save_embeddings(npz_path, emb, labels):
    out_dir = os.path.dirname(npz_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(npz_path, arr_0=emb, arr_1=labels, X=emb, Y=labels)


def build_augmenter():
    return iaa.Sequential(
        [
            iaa.Affine(rotate=(-15, 15)),
            iaa.Sometimes(0.3, iaa.Multiply((0.9, 1.1))),
            iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255))),
        ]
    )


def detect_largest_face_rgb(detector, bgr_image):
    rgb = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)
    if not detections:
        return None

    best = max(
        detections,
        key=lambda d: max(0, d["box"][2]) * max(0, d["box"][3]),
    )
    x, y, w, h = best["box"]
    x = max(0, int(x))
    y = max(0, int(y))
    x2 = min(rgb.shape[1], x + int(w))
    y2 = min(rgb.shape[0], y + int(h))
    if x2 <= x or y2 <= y:
        return None

    face = rgb[y:y2, x:x2]
    if face.size == 0:
        return None
    return cv.resize(face, TARGET_SIZE)


def iter_person_images(person_dir, max_images_per_person):
    files = [
        os.path.join(person_dir, fn)
        for fn in sorted(os.listdir(person_dir))
        if fn.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if max_images_per_person and max_images_per_person > 0:
        files = files[:max_images_per_person]
    return files


def embed_person(detector, embedder, augmenter, person_name, person_dir, augmentations, max_images):
    image_paths = iter_person_images(person_dir, max_images)
    if not image_paths:
        return np.empty((0, 512), dtype=np.float32), 0, 0

    person_embs = []
    processed = 0
    skipped = 0

    for img_path in image_paths:
        img = cv.imread(img_path)
        if img is None:
            skipped += 1
            continue

        face = detect_largest_face_rgb(detector, img)
        if face is None:
            skipped += 1
            continue

        batch_faces = [face]
        for _ in range(max(0, augmentations)):
            batch_faces.append(augmenter(image=face))

        batch = np.asarray(batch_faces, dtype=np.float32)
        batch_embs = embedder.embeddings(batch)
        for emb in batch_embs:
            person_embs.append(np.asarray(emb, dtype=np.float32))
        processed += 1

    if not person_embs:
        print(f"[WARN] {person_name}: no valid faces found.")
        return np.empty((0, 512), dtype=np.float32), processed, skipped

    embs = np.vstack(person_embs).astype(np.float32)
    return embs, processed, skipped


def main():
    args = parse_args()
    start_time = time.time()

    all_people = list_people(args.data_dir)
    if not all_people:
        print(f"[ERROR] No person folders found in {args.data_dir}")
        return

    selected_people = parse_people_arg(args.people)
    if not selected_people:
        selected_people = all_people

    missing = [p for p in selected_people if p not in all_people]
    if missing:
        print(f"[ERROR] Person folder(s) not found: {', '.join(missing)}")
        return

    detector = MTCNN()
    embedder = FaceNet()
    augmenter = build_augmenter()

    existing_embs = np.empty((0, 512), dtype=np.float32)
    existing_labels = np.empty((0,), dtype=object)
    if not args.rebuild_all and os.path.exists(args.embeddings_path):
        existing_embs, existing_labels = load_existing_embeddings(args.embeddings_path)

    if args.replace_existing and existing_labels.size > 0:
        to_replace = set(selected_people)
        keep_mask = np.array([label not in to_replace for label in existing_labels], dtype=bool)
        existing_embs = existing_embs[keep_mask]
        existing_labels = existing_labels[keep_mask]

    new_emb_chunks = []
    new_label_chunks = []
    total_processed = 0
    total_skipped = 0
    total_added = 0

    for person in selected_people:
        person_dir = os.path.join(args.data_dir, person)
        embs, processed, skipped = embed_person(
            detector=detector,
            embedder=embedder,
            augmenter=augmenter,
            person_name=person,
            person_dir=person_dir,
            augmentations=args.augmentations,
            max_images=args.max_images_per_person,
        )
        total_processed += processed
        total_skipped += skipped
        if embs.shape[0] == 0:
            continue

        labels = np.full((embs.shape[0],), person, dtype=object)
        new_emb_chunks.append(embs)
        new_label_chunks.append(labels)
        total_added += embs.shape[0]
        print(
            f"[OK] {person}: source_images={processed + skipped}, "
            f"faces_used={processed}, added_embeddings={embs.shape[0]}, skipped={skipped}"
        )

    if new_emb_chunks:
        new_embs = np.vstack(new_emb_chunks).astype(np.float32)
        new_labels = np.concatenate(new_label_chunks).astype(object)
    else:
        new_embs = np.empty((0, 512), dtype=np.float32)
        new_labels = np.empty((0,), dtype=object)

    if args.rebuild_all:
        final_embs = new_embs
        final_labels = new_labels
    else:
        if existing_embs.size == 0:
            final_embs = new_embs
            final_labels = new_labels
        elif new_embs.size == 0:
            final_embs = existing_embs
            final_labels = existing_labels
        else:
            final_embs = np.vstack([existing_embs, new_embs]).astype(np.float32)
            final_labels = np.concatenate([existing_labels, new_labels]).astype(object)

    if final_labels.size == 0:
        print("[ERROR] No embeddings generated. Nothing to save.")
        return

    save_embeddings(args.embeddings_path, final_embs, final_labels)
    elapsed = time.time() - start_time

    print("")
    print(f"[INFO] Saved embeddings to: {args.embeddings_path}")
    print(f"[INFO] People processed: {len(selected_people)}")
    print(f"[INFO] Images with face: {total_processed} | skipped: {total_skipped}")
    print(f"[INFO] New embeddings added: {total_added}")
    print(f"[INFO] Total embeddings in file: {final_embs.shape[0]}")
    print(f"[INFO] Total unique labels: {len(set(final_labels.tolist()))}")
    print(f"[INFO] Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
