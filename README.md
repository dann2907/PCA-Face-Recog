# PCA Vision Web App

Web application for Principal Component Analysis (PCA) featuring Image Compression, Exploratory Data Analysis (EDA), and Face Recognition (Eigenface Subspace Method - ESM).

## Features

- **Image Compression:** Upload color/grayscale photos and compress them using $k$ principal components.
- **EDA (Exploratory Data Analysis):** Interactive charts showing Cumulative Explained Variance to help determine the optimal $k$ value.
- **Face Similarity (ESM):** Compare two faces by projecting them into an Eigenface subspace. Uses cosine similarity with threshold-based decision (Mirip / Tidak Mirip).
- **Face Recognition:** Identify a face against the gallery database. Returns best match or "Unknown" if below threshold.

## Tech Stack

- **Backend:** FastAPI (Python), NumPy, OpenCV, Scikit-Learn.
- **Frontend:** React (Vite, TypeScript), Tailwind CSS, Recharts, Lucide Icons.
- **Demo (archived):** Streamlit — see `tools/streamlit-demo/`.

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js & npm

### Backend Setup
1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`.

### Frontend Setup
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   The web app will be available at `http://localhost:5173`.

## Usage

1. **Compression Tab:** Upload an image, adjust the $k$ slider, and click "Process Decomposition". Check the EDA chart to see how much variance is captured.
2. **ESM Tab:** Click "Initialize Basis" to train the PCA model (Olivetti faces). Choose **Compare** mode to measure similarity between two faces, or **Recognize** mode to identify a face against the gallery. Results show cosine similarity score and threshold decision.
3. **Docs Tab:** Mathematical foundation — SVD decomposition, projection, cosine similarity.

## Project Structure

```
PCA/
├── backend/
│   ├── main.py              # FastAPI server
│   └── requirements.txt     # Python deps
├── frontend/
│   └── src/
│       └── App.tsx          # React SPA (single-file)
├── dataset/                 # User-provided face dataset
│   ├── person_1/
│   └── person_2/
├── tools/
│   └── streamlit-demo/      # Archived Streamlit demo
│       └── app.py
└── .claude/
    └── plan/                # Implementation plans
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/compress` | PCA image compression |
| POST | `/api/esm/train` | Train PCA model on Olivetti |
| POST | `/api/esm/compare` | Compare two faces (cosine similarity) |
| POST | `/api/esm/recognize` | Identify face against gallery |
| POST | `/api/face/detect` | Detect & preprocess single face |
| GET | `/api/dataset/load` | Load local face dataset metadata |
| GET | `/api/debug/similarity-stats` | Benchmark same/diff person cosine distributions |

## Author
Developed for Aljabar Linear & Matriks project.
