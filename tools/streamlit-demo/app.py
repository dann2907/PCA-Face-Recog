# PCA Vision - Streamlit Demo
# Simplified single-file version for easy demo
# For full React UI, see frontend/ + backend/

import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics.pairwise import cosine_similarity
import io
import tempfile
from pathlib import Path

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="PCA Vision",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Constants
# =====================================================
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SIMILARITY_THRESHOLD = 0.80
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (64, 64)

# =====================================================
# Localization
# =====================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

def toggle_lang():
    st.session_state.lang = 'id' if st.session_state.lang == 'en' else 'en'

translations = {
    'en': {
        'title': "PCA Vision",
        'subtitle': "Image Decomposition & Face Subspace",
        'nav_compression': "Compression",
        'nav_recognition': "Recognition",
        'nav_docs': "Documentation",
        'pca_input': "Input Matrix",
        'pca_upload': "Upload Source Image",
        'pca_k': "Components (k)",
        'pca_btn': "Process Decomposition",
        'pca_orig': "Original Matrix",
        'pca_recon': "Reconstructed",
        'pca_variance': "Variance Statistics",
        'pca_stats_note': "The area chart shows cumulative energy captured. Aim for the 'elbow'.",
        'esm_title': "Eigenface Subspace",
        'esm_btn_train': "Initialize Basis (Olivetti)",
        'esm_btn_train_custom': "Train on Custom Dataset",
        'esm_btn_ready': "Subspace Ready",
        'esm_source': "Subject Matrix (Source)",
        'esm_target': "Subject Matrix (Target)",
        'esm_btn_compare': "Compute Subspace Distance",
        'esm_btn_recognize': "Identify Face",
        'esm_sim': "Cosine Similarity",
        'esm_decision': "Decision",
        'esm_match': "Match",
        'esm_no_match': "No Match",
        'esm_unknown': "Unknown",
        'esm_mode_compare': "Compare",
        'esm_mode_recognize': "Recognize",
        'esm_dataset_req_title': "Dataset Requirements",
        'esm_dataset_req_1': "Minimal age gap between photos",
        'esm_dataset_req_2': "Various poses, lighting, conditions",
        'esm_dataset_req_3': "Minimum 5-10 images per person",
        'esm_dataset_req_4': "Consistent image quality",
        'esm_upload_dataset': "Upload Custom Dataset",
        'esm_upload_hint': "Upload images in folders (person_name/image.jpg)",
        'esm_dataset_info': "Dataset Info",
        'esm_persons': "Persons",
        'esm_images': "Images",
        'esm_threshold_ref': "Similarity Threshold Reference",
        'esm_th_high': "High Similarity (≥0.80)",
        'esm_th_moderate': "Moderate (0.60-0.80)",
        'esm_th_low': "Low (0.40-0.60)",
        'esm_th_very_low': "Very Low (<0.40)",
        'esm_th_high_desc': "Likely same person",
        'esm_th_moderate_desc': "Possible match, verify manually",
        'esm_th_low_desc': "Different persons likely",
        'esm_th_very_low_desc': "Definitely different persons",
        'esm_viz_title': "PCA Visualization",
        'esm_mean_face': "Mean Face",
        'esm_eigenfaces': "Top Eigenfaces",
        'docs_title': "Mathematical Foundation",
        'docs_intro': "This platform leverages PCA for optimized representation and pattern recognition.",
        'lang_toggle': "🇮🇩 Bahasa Indonesia"
    },
    'id': {
        'title': "PCA Vision",
        'subtitle': "Dekomposisi Gambar & Subspace Wajah",
        'nav_compression': "Kompresi",
        'nav_recognition': "Pengenalan",
        'nav_docs': "Dokumentasi",
        'pca_input': "Matriks Input",
        'pca_upload': "Unggah Gambar Sumber",
        'pca_k': "Komponen (k)",
        'pca_btn': "Proses Dekomposisi",
        'pca_orig': "Matriks Orisinal",
        'pca_recon': "Rekonstruksi",
        'pca_variance': "Statistik Varians",
        'pca_stats_note': "Grafik area menunjukkan energi kumulatif. Targetkan titik 'siku'.",
        'esm_title': "Subspace Eigenface",
        'esm_btn_train': "Inisialisasi Basis (Olivetti)",
        'esm_btn_train_custom': "Latih pada Dataset Kustom",
        'esm_btn_ready': "Subspace Siap",
        'esm_source': "Matriks Subjek (Sumber)",
        'esm_target': "Matriks Subjek (Target)",
        'esm_btn_compare': "Hitung Jarak Subspace",
        'esm_btn_recognize': "Identifikasi Wajah",
        'esm_sim': "Kemiripan Kosinus",
        'esm_decision': "Keputusan",
        'esm_match': "Mirip",
        'esm_no_match': "Tidak Mirip",
        'esm_unknown': "Tidak Dikenali",
        'esm_mode_compare': "Bandingkan",
        'esm_mode_recognize': "Identifikasi",
        'esm_dataset_req_title': "Persyaratan Dataset",
        'esm_dataset_req_1': "Perbedaan usia minimal antar foto",
        'esm_dataset_req_2': "Berbagai pose, pencahayaan, kondisi",
        'esm_dataset_req_3': "Minimal 5-10 gambar per orang",
        'esm_dataset_req_4': "Kualitas gambar konsisten",
        'esm_upload_dataset': "Unggah Dataset Kustom",
        'esm_upload_hint': "Unggah gambar dalam folder (nama_orang/gambar.jpg)",
        'esm_dataset_info': "Info Dataset",
        'esm_persons': "Orang",
        'esm_images': "Gambar",
        'esm_threshold_ref': "Referensi Ambang Kemiripan",
        'esm_th_high': "Kemiripan Tinggi (≥0.80)",
        'esm_th_moderate': "Sedang (0.60-0.80)",
        'esm_th_low': "Rendah (0.40-0.60)",
        'esm_th_very_low': "Sangat Rendah (<0.40)",
        'esm_th_high_desc': "Kemungkinan orang yang sama",
        'esm_th_moderate_desc': "Kemungkinan cocok, verifikasi manual",
        'esm_th_low_desc': "Kemungkinan orang berbeda",
        'esm_th_very_low_desc': "Pasti orang berbeda",
        'esm_viz_title': "Visualisasi PCA",
        'esm_mean_face': "Wajah Rata-rata",
        'esm_eigenfaces': "Eigenface Teratas",
        'docs_title': "Fondasi Matematika",
        'docs_intro': "Platform ini memanfaatkan PCA untuk representasi dan pengenalan pola yang dioptimalkan.",
        'lang_toggle': "🇺🇸 English"
    }
}

t = translations[st.session_state.lang]

# =====================================================
# PCA Functions
# =====================================================
def pca_compress_2d(X, k):
    X = X.astype(np.float64)
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    k = min(k, len(S))
    X_reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :] + mean_vec
    total_variance = np.sum(S**2)
    explained_variance = np.cumsum(S**2) / total_variance
    return X_reconstructed, explained_variance.tolist()

def process_image_pca(image_bytes, k):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    reconstructed_channels = []
    variance_ratios = []

    for i in range(3):
        channel = img_rgb[:, :, i]
        reconstructed, v_ratio = pca_compress_2d(channel, k)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        reconstructed_channels.append(reconstructed)
        variance_ratios.append(v_ratio)

    avg_variance_ratio = np.mean(variance_ratios, axis=0)
    reconstructed_img = np.stack(reconstructed_channels, axis=2)
    return reconstructed_img, avg_variance_ratio

# =====================================================
# Face Detection & Processing
# =====================================================
def detect_and_crop_face(image_bytes):
    """Detect face using Haar cascade, crop, resize, normalize."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image — cannot decode")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # Fallback: use whole image
        face_resized = cv2.resize(gray, FACE_SIZE)
        flat = face_resized.flatten() / 255.0
        return flat, None

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

    return flat, (x1, y1, x2, y2)

def process_face_simple(uploaded_file):
    """Simple face processing without detection (fallback)."""
    img = Image.open(uploaded_file).convert('L')
    img = img.resize(FACE_SIZE)
    return np.array(img).flatten() / 255.0

# =====================================================
# Dataset Functions
# =====================================================
def load_uploaded_dataset(uploaded_files):
    """Process uploaded files into dataset format."""
    images = []
    labels = []

    for file in uploaded_files:
        filename = file.name
        parts = Path(filename).parts

        if len(parts) < 2:
            continue

        if not any(filename.lower().endswith(ext) for ext in VALID_EXTENSIONS):
            continue

        person_name = parts[-2] if len(parts) > 1 else "unknown"

        try:
            img = Image.open(file).convert('L')
            img = img.resize(FACE_SIZE)
            flat = np.array(img).flatten() / 255.0
            images.append(flat)
            labels.append(person_name)
        except Exception:
            continue

    if not images:
        return None, None

    return np.array(images), labels

def build_gallery(model, X, labels):
    """Build gallery embeddings from dataset."""
    embeddings = model.transform(X)
    return embeddings, labels

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.title("🤖 PCA Vision")
    st.button(t['lang_toggle'], on_click=toggle_lang)
    st.divider()
    st.caption("Lab Matriks v1.0")
    st.caption("Standard PCA Subspace")

# =====================================================
# Main UI
# =====================================================
st.title(t['subtitle'])

tabs = st.tabs([t['nav_compression'], t['nav_recognition'], t['nav_docs']])

# --- Tab 1: Compression ---
with tabs[0]:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(t['pca_input'])
        uploaded_file = st.file_uploader(t['pca_upload'], type=['png', 'jpg', 'jpeg'])
        k_val = st.slider(t['pca_k'], 1, 200, 50)

        if uploaded_file and st.button(t['pca_btn'], use_container_width=True):
            with st.spinner("Processing..."):
                bytes_data = uploaded_file.read()
                recon_img, var_ratio = process_image_pca(bytes_data, k_val)
                st.session_state.recon_img = recon_img
                st.session_state.var_ratio = var_ratio
                st.session_state.orig_img = Image.open(io.BytesIO(bytes_data))

    with col2:
        if 'recon_img' in st.session_state:
            st.subheader(t['pca_orig'] + " vs " + t['pca_recon'])
            img_col1, img_col2 = st.columns(2)
            img_col1.image(st.session_state.orig_img, use_container_width=True)
            img_col2.image(st.session_state.recon_img, use_container_width=True)

            st.divider()
            st.subheader(t['pca_variance'])
            df_var = pd.DataFrame({
                'Component': range(1, len(st.session_state.var_ratio[:100]) + 1),
                'Variance': [v * 100 for v in st.session_state.var_ratio[:100]]
            })
            st.area_chart(df_var, x='Component', y='Variance')
            st.info(t['pca_stats_note'])
        else:
            st.info("Upload an image and click process to see results.")

# --- Tab 2: Recognition ---
with tabs[1]:
    st.subheader(t['esm_title'])

    # Initialize session state
    if 'esm_model' not in st.session_state:
        st.session_state.esm_model = None
        st.session_state.trained = False
        st.session_state.gallery_embeddings = None
        st.session_state.gallery_labels = None

    # Dataset Requirements Clarification
    st.info(f"""
    **{t['esm_dataset_req_title']}**
    - {t['esm_dataset_req_1']}
    - {t['esm_dataset_req_2']}
    - {t['esm_dataset_req_3']}
    - {t['esm_dataset_req_4']}
    """)

    # Training Section
    train_col1, train_col2 = st.columns(2)

    with train_col1:
        if st.button(t['esm_btn_train'], disabled=st.session_state.trained, use_container_width=True):
            with st.spinner("Training on Olivetti Faces..."):
                data = fetch_olivetti_faces()
                faces = data.data
                model = SklearnPCA(n_components=100, whiten=True)
                model.fit(faces)
                st.session_state.esm_model = model
                st.session_state.gallery_embeddings = model.transform(faces)
                st.session_state.gallery_labels = [f"person_{i//10}" for i in range(len(faces))]
                st.session_state.trained = True
                st.rerun()

    with train_col2:
        # Custom Dataset Upload
        st.markdown(f"**{t['esm_upload_dataset']}**")
        uploaded_dataset = st.file_uploader(
            t['esm_upload_hint'],
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="dataset_upload"
        )

        if uploaded_dataset and st.button(t['esm_btn_train_custom'], use_container_width=True):
            with st.spinner("Processing custom dataset..."):
                X, labels = load_uploaded_dataset(uploaded_dataset)
                if X is not None:
                    n_components = min(100, len(X) - 1)
                    model = SklearnPCA(n_components=n_components, whiten=True)
                    model.fit(X)
                    st.session_state.esm_model = model
                    st.session_state.gallery_embeddings, st.session_state.gallery_labels = build_gallery(model, X, labels)
                    st.session_state.trained = True
                    st.success(f"Trained on {len(X)} images, {len(set(labels))} persons")
                    st.rerun()
                else:
                    st.error("No valid images found. Check folder structure.")

    # Show dataset info if custom dataset uploaded
    if uploaded_dataset and st.session_state.trained:
        st.success(f"""
        **{t['esm_dataset_info']}**
        - {len(st.session_state.gallery_labels)} {t['esm_images']}
        - {len(set(st.session_state.gallery_labels))} {t['esm_persons']}
        """)

    if st.session_state.trained:
        st.success(t['esm_btn_ready'])

        # Threshold Reference
        with st.expander(t['esm_threshold_ref'], expanded=False):
            st.markdown(f"""
            | Range | Level | Description |
            |-------|-------|-------------|
            | ≥ 0.80 | 🟢 {t['esm_th_high']} | {t['esm_th_high_desc']} |
            | 0.60 - 0.80 | 🟡 {t['esm_th_moderate']} | {t['esm_th_moderate_desc']} |
            | 0.40 - 0.60 | 🟠 {t['esm_th_low']} | {t['esm_th_low_desc']} |
            | < 0.40 | 🔴 {t['esm_th_very_low']} | {t['esm_th_very_low_desc']} |
            """)

        # Mode Toggle
        mode = st.radio(
            "Mode",
            [t['esm_mode_compare'], t['esm_mode_recognize']],
            horizontal=True,
            label_visibility="collapsed"
        )

        if mode == t['esm_mode_compare']:
            # Compare Mode
            col_face1, col_face2 = st.columns(2)
            with col_face1:
                face1_file = st.file_uploader(t['esm_source'], type=['png', 'jpg', 'jpeg'], key="face1")
            with col_face2:
                face2_file = st.file_uploader(t['esm_target'], type=['png', 'jpg', 'jpeg'], key="face2")

            if face1_file and face2_file:
                if st.button(t['esm_btn_compare'], use_container_width=True):
                    with st.spinner("Computing similarity..."):
                        try:
                            f1_bytes = face1_file.read()
                            f2_bytes = face2_file.read()

                            f1, bbox1 = detect_and_crop_face(f1_bytes)
                            f2, bbox2 = detect_and_crop_face(f2_bytes)

                            p1 = st.session_state.esm_model.transform([f1])
                            p2 = st.session_state.esm_model.transform([f2])

                            sim = cosine_similarity(p1, p2)[0][0]
                            decision = t['esm_match'] if sim >= SIMILARITY_THRESHOLD else t['esm_no_match']

                            # Display results
                            st.divider()
                            result_col1, result_col2 = st.columns(2)

                            with result_col1:
                                st.metric(t['esm_sim'], f"{sim:.4f}")
                                color = "green" if sim >= SIMILARITY_THRESHOLD else "red"
                                st.markdown(f"**{t['esm_decision']}:** :{color}[{decision}]")
                                st.caption(f"Threshold: {SIMILARITY_THRESHOLD}")

                            with result_col2:
                                # Similarity bar
                                bar_color = "green" if sim >= 0.8 else "orange" if sim >= 0.6 else "red"
                                st.progress(max(0, min(1, (sim + 1) / 2)))

                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        else:
            # Recognize Mode
            face_file = st.file_uploader(t['esm_source'], type=['png', 'jpg', 'jpeg'], key="face_recog")

            if face_file:
                if st.button(t['esm_btn_recognize'], use_container_width=True):
                    with st.spinner("Identifying face..."):
                        try:
                            f_bytes = face_file.read()
                            f, bbox = detect_and_crop_face(f_bytes)

                            p = st.session_state.esm_model.transform([f])

                            # Search gallery
                            sims = cosine_similarity(p, st.session_state.gallery_embeddings)[0]
                            best_idx = np.argmax(sims)
                            best_sim = sims[best_idx]
                            best_label = st.session_state.gallery_labels[best_idx]

                            unknown = best_sim < SIMILARITY_THRESHOLD

                            # Display results
                            st.divider()

                            if unknown:
                                st.markdown(f"### :red[{t['esm_unknown']}]")
                            else:
                                st.markdown(f"### :green[{best_label}]")

                            st.metric(t['esm_sim'], f"{best_sim:.4f}")

                            decision = t['esm_no_match'] if unknown else t['esm_match']
                            color = "red" if unknown else "green"
                            st.markdown(f"**{t['esm_decision']}:** :{color}[{decision}]")
                            st.caption(f"Threshold: {SIMILARITY_THRESHOLD}")

                            # Show top matches
                            st.divider()
                            st.subheader("Top Matches")
                            top_indices = np.argsort(sims)[::-1][:5]
                            for idx in top_indices:
                                label = st.session_state.gallery_labels[idx]
                                score = sims[idx]
                                bar = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                                st.text(f"{bar} {label}: {score:.4f}")

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    else:
        st.warning("Initialize Subspace first.")

# --- Tab 3: Documentation ---
with tabs[2]:
    st.header(t['docs_title'])
    st.write(t['docs_intro'])
    st.markdown("""
    ### Module 01: Image Compression
    PCA finds the directions (eigenvectors) that capture the most variance. By projecting onto the top $k$ directions, we compress the data while keeping essential information.

    ### Module 02: Face Recognition (ESM)
    Eigenfaces represent the principal components of a face dataset. New faces are projected onto this subspace, and similarity is measured using cosine similarity between projections.

    ### How to Use
    1. **Compression**: Upload image → adjust k → see reconstruction
    2. **Recognition**: Train model → upload faces → compare or identify

    ### Limitations
    - PCA captures pixel patterns, not identity features
    - Sensitive to lighting, pose, expression
    - Not suitable for age-gap recognition
    - Works best with controlled conditions
    """)
