import os
import cv2 as cv
import numpy as np
import torch
from insightface.app import FaceAnalysis

DATA_DIR = "data"
OUT_FILE = os.path.join("embeddings", "arcface_gallery.npz")


def main():
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ctx_id = 0 if torch.cuda.is_available() else -1
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    embs = []
    labels = []
    skipped = 0

    image_paths = []
    for name in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append((name, os.path.join(person_dir, img_name)))

    total = len(image_paths)
    if total == 0:
        print("[ERROR] No images found in data/")
        return

    for i, (name, img_path) in enumerate(image_paths, start=1):
        img = cv.imread(img_path)
        if img is None:
            # print(f"[ERROR] Failed to load image: {img_path}")
            skipped += 1
            continue
        faces = app.get(img)
        if not faces:
            skipped += 1
            continue
        # choose largest face
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        embs.append(face.normed_embedding)
        labels.append(name)

        if i % 10 == 0 or i == total:
            pct = (i / total) * 100.0
            print(
                f"\rProcessed {i}/{total} ({pct:.1f}%) | "
                f"Embeddings: {len(embs)} | Skipped: {skipped}",
                end="",
                flush=True,
            )
            

    print()

    if not embs:
        print("[ERROR] No faces found. Check your data images.")
        return

    os.makedirs("embeddings", exist_ok=True)
    np.savez_compressed(OUT_FILE, embs=np.array(embs), labels=np.array(labels))
    print(f"[OK] Saved {len(embs)} embeddings to {OUT_FILE}")


if __name__ == "__main__":
    main()
