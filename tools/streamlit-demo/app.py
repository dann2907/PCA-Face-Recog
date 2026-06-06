import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA as SklearnPCA
import io
import base64

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
        'esm_btn_train': "Initialize Basis",
        'esm_btn_ready': "Subspace Ready",
        'esm_source': "Subject Matrix (Source)",
        'esm_target': "Subject Matrix (Target)",
        'esm_btn_compare': "Compute Subspace Distance",
        'esm_sim': "Similarity Coefficient",
        'esm_dist': "Euclidean Distance",
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
        'esm_btn_train': "Inisialisasi Basis",
        'esm_btn_ready': "Subspace Siap",
        'esm_source': "Matriks Subjek (Sumber)",
        'esm_target': "Matriks Subjek (Target)",
        'esm_btn_compare': "Hitung Jarak Subspace",
        'esm_sim': "Koefisien Kemiripan",
        'esm_dist': "Jarak Euclidean",
        'docs_title': "Fondasi Matematika",
        'docs_intro': "Platform ini memanfaatkan PCA untuk representasi dan pengenalan pola yang dioptimalkan.",
        'lang_toggle': "🇺🇸 English"
    }
}

t = translations[st.session_state.lang]

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
# PCA Logic
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
    
    if 'esm_model' not in st.session_state:
        st.session_state.esm_model = None
        st.session_state.trained = False

    if st.button(t['esm_btn_train'], disabled=st.session_state.trained):
        with st.spinner("Training on Olivetti Faces..."):
            data = fetch_olivetti_faces()
            faces = data.data
            model = SklearnPCA(n_components=100, whiten=True)
            model.fit(faces)
            st.session_state.esm_model = model
            st.session_state.trained = True
            st.rerun()

    if st.session_state.trained:
        st.success(t['esm_btn_ready'])
        
        col_face1, col_face2 = st.columns(2)
        with col_face1:
            face1_file = st.file_uploader(t['esm_source'], type=['png', 'jpg', 'jpeg'], key="face1")
        with col_face2:
            face2_file = st.file_uploader(t['esm_target'], type=['png', 'jpg', 'jpeg'], key="face2")
            
        if face1_file and face2_file:
            if st.button(t['esm_btn_compare'], use_container_width=True):
                def process_face(uploaded):
                    img = Image.open(uploaded).convert('L')
                    img = img.resize((64, 64))
                    return np.array(img).flatten() / 255.0

                f1 = process_face(face1_file)
                f2 = process_face(face2_file)
                
                p1 = st.session_state.esm_model.transform([f1])
                p2 = st.session_state.esm_model.transform([f2])
                
                dist = np.linalg.norm(p1 - p2)
                sim = max(0, 100 - (dist * 10))
                
                st.divider()
                st.metric(t['esm_sim'], f"{sim:.2f}%")
                st.write(f"{t['esm_dist']}: {dist:.4f}")
                st.progress(sim / 100)
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
    Eigenfaces represent the principal components of a face dataset. New faces are projected onto this subspace, and similarity is measured using Euclidean distance between projections.
    """)
