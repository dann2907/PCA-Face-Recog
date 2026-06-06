
import base64
import numpy as np
import cv2
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics.pairwise import cosine_similarity
    
app = FastAPI()

# Allow CORS for React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for ESM
esm_model = None
esm_mean_face = None
gallery_embeddings = None  # shape (n_gallery, n_components)
gallery_labels = None      # list of label strings

# =====================================================
# Dataset Loader
# =====================================================

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SIMILARITY_THRESHOLD = 0.80

def load_dataset(dataset_dir: Path = DATASET_DIR):
    """
    Traverse dataset/ folders, read face images, return X and labels.

    Directory structure expected:
        dataset/
        ├── person_1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── img3.jpg
        ├── person_2/
        │   └── ...

    Returns:
        X: numpy array shape (num_images, num_pixels) — flattened grayscale
        labels: list of label strings aligned with X rows
    """
    images = []
    labels = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize to consistent 64x64
            img_resized = cv2.resize(img, (64, 64))

            # Flatten and normalize to [0, 1]
            flat = img_resized.flatten() / 255.0

            images.append(flat)
            labels.append(person_name)

    if not images:
        raise ValueError(f"No valid images found in {dataset_dir}")

    X = np.array(images)
    return X, labels


# =====================================================
# Face Detection & Cropping
# =====================================================

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (64, 64)

def detect_and_crop_face(image_bytes):
    """
    Preprocess face image for PCA pipeline.

    Pipeline:
        Input bytes → decode → face detection (Haar cascade) → crop → grayscale → resize → normalize → flatten

    Args:
        image_bytes: Raw image file bytes

    Returns:
        flat_vector: numpy array shape (4096,) — flattened 64x64 normalized [0,1]
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image — cannot decode")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No face detected in image")

    # Take first face
    x, y, w, h = faces[0]

    # Crop face with 10% padding
    pad_x = int(w * 0.1)
    pad_y = int(h * 0.1)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(gray.shape[1], x + w + pad_x)
    y2 = min(gray.shape[0], y + h + pad_y)
    face_crop = gray[y1:y2, x1:x2]

    # Resize to 64x64
    face_resized = cv2.resize(face_crop, FACE_SIZE)

    # Normalize [0, 1] and flatten
    flat = face_resized.flatten() / 255.0

    return flat


# =====================================================
# PCA Logic (Refactored from PCA.py)
# =====================================================

def pca_compress_2d(X, k):
    X = X.astype(np.float64)
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    k = min(k, len(S))
    X_reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :] + mean_vec
    
    # Variance ratio for EDA
    total_variance = np.sum(S**2)
    explained_variance = np.cumsum(S**2) / total_variance
    
    return X_reconstructed, explained_variance.tolist()

def compress_image(image_bytes, k):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape
    
    reconstructed_channels = []
    variance_ratios = []
    
    for i in range(3):
        channel = img_rgb[:, :, i]
        reconstructed, v_ratio = pca_compress_2d(channel, k)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        reconstructed_channels.append(reconstructed)
        variance_ratios.append(v_ratio)
    
    # Use average variance ratio for simplicity in EDA
    avg_variance_ratio = np.mean(variance_ratios, axis=0).tolist()
    
    reconstructed_img = np.stack(reconstructed_channels, axis=2)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR))
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    
    return encoded_img, avg_variance_ratio

# =====================================================
# Endpoints
# =====================================================

@app.post("/api/compress")
async def api_compress(file: UploadFile = File(...), k: int = Form(...)):
    try:
        contents = await file.read()
        encoded_img, variance_data = compress_image(contents, k)
        return {
            "compressed_image": f"data:image/png;base64,{encoded_img}",
            "variance_data": variance_data[:100] # Limit to top 100 for chart
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/esm/train")
async def train_esm():
    global esm_model, esm_mean_face, gallery_embeddings, gallery_labels
    try:
        data = fetch_olivetti_faces()
        faces = data.data  # shape (400, 4096) - 64x64 faces
        n_per_subject = 10
        n_subjects = 40

        esm_model = SklearnPCA(n_components=100, whiten=True)
        gallery_embeddings = esm_model.fit_transform(faces)
        esm_mean_face = esm_model.mean_

        # Label: "Subject 0", "Subject 1", ...
        gallery_labels = [
            f"Subject {i // n_per_subject}" for i in range(len(faces))
        ]

        return {
            "status": "success",
            "message": "ESM model trained on Olivetti faces dataset.",
            "gallery_size": len(gallery_labels),
            "n_subjects": n_subjects,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/esm/compare")
async def compare_faces(source_file: UploadFile = File(...), target_file: UploadFile = File(...)):
    global esm_model
    if esm_model is None:
        raise HTTPException(status_code=400, detail="ESM model not trained. Call /api/esm/train first.")

    try:
        src_bytes = await source_file.read()
        tgt_bytes = await target_file.read()

        # Phase 2 pipeline: detect → crop → grayscale → resize → normalize → flatten
        src_flat = detect_and_crop_face(src_bytes)
        tgt_flat = detect_and_crop_face(tgt_bytes)

        # Project into PCA subspace
        src_proj = esm_model.transform([src_flat])
        tgt_proj = esm_model.transform([tgt_flat])

        # Cosine similarity [-1, 1]
        sim = cosine_similarity(src_proj, tgt_proj)[0][0]
        sim = round(float(sim), 4)

        # Threshold classification
        
        decision = "Mirip" if sim >= SIMILARITY_THRESHOLD else "Tidak Mirip"

        return {
            "similarity": sim,
            "decision": decision,
            "threshold": SIMILARITY_THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/esm/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Identify a face against the gallery database.

    Preprocesses the uploaded face, projects into PCA subspace, and finds
    the closest match in the gallery via cosine similarity.

    Returns:
        name: best-matching subject label
        similarity: cosine similarity score [-1, 1]
        decision: "Mirip" / "Tidak Mirip" based on threshold
        unknown: true if similarity below SIMILARITY_THRESHOLD
    """
    global esm_model, gallery_embeddings, gallery_labels
    if esm_model is None or gallery_embeddings is None:
        raise HTTPException(status_code=400, detail="ESM model not trained. Call /api/esm/train first.")

    try:
        contents = await file.read()
        flat = detect_and_crop_face(contents)

        # Project into PCA subspace
        query_proj = esm_model.transform([flat])

        # Compare against all gallery embeddings
        sims = cosine_similarity(query_proj, gallery_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_sim = round(float(sims[best_idx]), 4)
        best_label = gallery_labels[best_idx]

        unknown = best_sim < SIMILARITY_THRESHOLD
        decision = "Tidak Mirip" if unknown else "Mirip"

        return {
            "name": "Unknown" if unknown else best_label,
            "similarity": best_sim,
            "decision": decision,
            "threshold": SIMILARITY_THRESHOLD,
            "unknown": unknown,
            "gallery_size": len(gallery_labels),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dataset/load")
async def api_load_dataset():
    """Load face dataset from disk, return metadata."""
    try:
        X, labels = load_dataset()
        return {
            "status": "success",
            "num_images": X.shape[0],
            "num_pixels": X.shape[1],
            "labels": labels,
            "unique_subjects": len(set(labels)),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/face/detect")
async def api_detect_face(file: UploadFile = File(...)):
    """Detect and preprocess single face from uploaded image.

    Returns flattened 64x64 normalized vector for PCA ingestion.
    """
    try:
        contents = await file.read()
        flat = detect_and_crop_face(contents)
        return {
            "status": "success",
            "shape": list(flat.shape),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/debug/similarity-stats")
async def debug_similarity_stats():
    """Benchmark cosine similarity distribution on Olivetti dataset.

    Computes pairwise cosine similarity for same-person and different-person
    projections to surface model discrimination quality.

    Returns:
        same_mean, same_min, same_max — stats for same-person pairs
        diff_mean, diff_max — stats for different-person pairs
    """
    try:
        data = fetch_olivetti_faces()
        faces = data.data
        model = SklearnPCA(n_components=100, whiten=True)
        projections = model.fit_transform(faces)

        n_total = len(faces)
        n_per_person = 10

        same_scores = []
        diff_scores = []

        # Sample different-person pairs (avoid O(n^2): compare first image
        # of each subject against first image of every other subject)
        for i in range(0, n_total, n_per_person):
            for j in range(i + n_per_person, n_total, n_per_person):
                sim = cosine_similarity([projections[i]], [projections[j]])[0][0]
                diff_scores.append(sim)

        # All same-person pairwise within each subject group
        for group_start in range(0, n_total, n_per_person):
            group = projections[group_start:group_start + n_per_person]
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    sim = cosine_similarity([group[a]], [group[b]])[0][0]
                    same_scores.append(sim)

        return {
            "same_pairs": len(same_scores),
            "same_mean": round(float(np.mean(same_scores)), 4),
            "same_min": round(float(np.min(same_scores)), 4),
            "same_max": round(float(np.max(same_scores)), 4),
            "diff_pairs": len(diff_scores),
            "diff_mean": round(float(np.mean(diff_scores)), 4),
            "diff_min": round(float(np.min(diff_scores)), 4),
            "diff_max": round(float(np.max(diff_scores)), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
