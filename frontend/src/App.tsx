import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import { 
  Upload, Cpu, UserCheck, BarChart3, Loader2, 
  ArrowRight, LayoutDashboard, 
  BookOpen, Image as ImageIcon, Info, ChevronRight,
  RefreshCw, CheckCircle2, AlertCircle, Languages
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

type Lang = 'en' | 'id';

const translations = {
  en: {
    subtitle: "Image Decomposition & Face Subspace",
    nav_compression: "Compression",
    nav_recognition: "Recognition",
    nav_docs: "Documentation",
    status_api: "API Connected",
    lab_version: "Matrix Lab v1.0",
    lab_desc: "Standard PCA Subspace",
    tab_pca_title: "Image Decomposition",
    tab_esm_title: "Face Subspace",
    tab_docs_title: "System Guide",
    pca_input: "Input Matrix",
    pca_upload: "Upload Source Image",
    pca_upload_hint: "PNG, JPG up to 10MB",
    pca_k: "Components",
    pca_k_hint: "Target Rank (k)",
    pca_btn: "Process Decomposition",
    pca_note_title: "Analysis Note",
    pca_note_desc: "Reducing k significantly increases compression but may lead to artifacting. Optimal k is usually where the cumulative variance stabilizes.",
    pca_comparison: "Visual Comparison",
    pca_split: "Split View",
    pca_original: "Original (Input)",
    pca_reconstructed: "Reconstructed",
    pca_variance: "Variance Statistics",
    pca_no_data: "No data to project",
    pca_stats_note: "The area above shows the cumulative energy captured by the first k eigenvectors. For efficient compression, aim for the point where the curve starts to level off (the 'elbow').",
    esm_title: "Eigenface Subspace",
    esm_desc: "Trained on Olivetti High-Dimensional Dataset",
    esm_btn_train: "Initialize Basis",
    esm_btn_ready: "Subspace Ready",
    esm_alert_title: "Subspace Required",
    esm_alert_desc: "You must initialize the subspace before projecting faces. This loads the Olivetti dataset into the principal component basis.",
    esm_source: "Subject Matrix (Source)",
    esm_target: "Subject Matrix (Target)",
    esm_import: "Select Source",
    esm_btn_compare: "Compute Subspace Distance",
    esm_sim: "Cosine Similarity",
    esm_conf: "Confidence Level",
    esm_high: "High",
    esm_mod: "Moderate",
    esm_low: "Low",
    esm_decision: "Decision",
    esm_match: "Match",
    esm_no_match: "No Match",
    esm_mode_compare: "Compare",
    esm_mode_recognize: "Recognize",
    esm_recog_title: "Face Identification",
    esm_recog_btn: "Identify Face",
    esm_recog_result: "Identification Result",
    esm_unknown: "Unknown",
    esm_viz_title: "PCA Visualization",
    esm_mean_face: "Mean Face",
    esm_eigenfaces: "Top Eigenfaces",
    docs_title: "Mathematical Foundation",
    docs_intro: "This platform leverages Principal Component Analysis to map high-dimensional visual data into an optimized feature space, enabling efficient representation and pattern recognition.",
    docs_decomp: "Decomposition",
    docs_decomp_desc: "By applying SVD to image matrices, we extract orthogonal eigenvectors. Retaining the top k components preserves the structural integrity while shedding redundant data.",
    docs_proj: "Projections",
    docs_proj_desc: "Face recognition treats images as vectors in a large space. PCA finds a lower-dimensional 'face space' where relative distances between subjects are maintained.",
    docs_step1: "Upload Data",
    docs_step1_desc: "Import image matrices for analysis into the secure lab environment.",
    docs_step2: "Configure k",
    docs_step2_desc: "Adjust the rank parameter to control fidelity vs. compression ratio.",
    docs_step3: "Validate",
    docs_step3_desc: "Review variance statistics and similarity coefficients for accuracy."
  },
  id: {
    subtitle: "Dekomposisi Gambar & Subspace Wajah",
    nav_compression: "Kompresi",
    nav_recognition: "Pengenalan",
    nav_docs: "Dokumentasi",
    status_api: "API Terhubung",
    lab_version: "Lab Matriks v1.0",
    lab_desc: "Subspace PCA Standar",
    tab_pca_title: "Dekomposisi Gambar",
    tab_esm_title: "Subspace Wajah",
    tab_docs_title: "Panduan Sistem",
    pca_input: "Matriks Input",
    pca_upload: "Unggah Gambar Sumber",
    pca_upload_hint: "PNG, JPG hingga 10MB",
    pca_k: "Komponen",
    pca_k_hint: "Rank Target (k)",
    pca_btn: "Proses Dekomposisi",
    pca_note_title: "Catatan Analisis",
    pca_note_desc: "Mengurangi k meningkatkan kompresi secara signifikan tetapi dapat menyebabkan artifak. k optimal biasanya berada di titik di mana varians kumulatif stabil.",
    pca_comparison: "Perbandingan Visual",
    pca_split: "Tampilan Terpisah",
    pca_original: "Orisinal (Input)",
    pca_reconstructed: "Rekonstruksi",
    pca_variance: "Statistik Varians",
    pca_no_data: "Tidak ada data untuk diproyeksikan",
    pca_stats_note: "Area di atas menunjukkan energi kumulatif yang ditangkap oleh k eigenvector pertama. Untuk kompresi yang efisien, targetkan titik di mana kurva mulai mendatar (titik 'siku').",
    esm_title: "Subspace Eigenface",
    esm_desc: "Dilatih pada Dataset Dimensi Tinggi Olivetti",
    esm_btn_train: "Inisialisasi Basis",
    esm_btn_ready: "Subspace Siap",
    esm_alert_title: "Subspace Diperlukan",
    esm_alert_desc: "Anda harus menginisialisasi subspace sebelum memproyeksikan wajah. Ini memuat dataset Olivetti ke dalam basis komponen utama.",
    esm_source: "Matriks Subjek (Sumber)",
    esm_target: "Matriks Subjek (Target)",
    esm_import: "Pilih Sumber",
    esm_btn_compare: "Hitung Jarak Subspace",
    esm_sim: "Kemiripan Kosinus",
    esm_conf: "Tingkat Kepercayaan",
    esm_high: "Tinggi",
    esm_mod: "Sedang",
    esm_low: "Rendah",
    esm_decision: "Keputusan",
    esm_match: "Mirip",
    esm_no_match: "Tidak Mirip",
    esm_mode_compare: "Bandingkan",
    esm_mode_recognize: "Identifikasi",
    esm_recog_title: "Identifikasi Wajah",
    esm_recog_btn: "Identifikasi Wajah",
    esm_recog_result: "Hasil Identifikasi",
    esm_unknown: "Tidak Dikenali",
    esm_viz_title: "Visualisasi PCA",
    esm_mean_face: "Wajah Rata-rata",
    esm_eigenfaces: "Eigenface Teratas",
    docs_title: "Fondasi Matematika",
    docs_intro: "Platform ini memanfaatkan Principal Component Analysis untuk memetakan data visual dimensi tinggi ke dalam ruang fitur yang dioptimalkan, memungkinkan representasi dan pengenalan pola yang efisien.",
    docs_decomp: "Dekomposisi",
    docs_decomp_desc: "Dengan menerapkan SVD ke matriks gambar, kita mengekstrak eigenvector ortogonal. Mempertahankan k komponen teratas menjaga integritas struktural sambil membuang data redundan.",
    docs_proj: "Proyeksi",
    docs_proj_desc: "Pengenalan wajah memperlakukan gambar sebagai vektor dalam ruang besar. PCA menemukan 'ruang wajah' berdimensi rendah di mana jarak relatif antar subjek dipertahankan.",
    docs_step1: "Unggah Data",
    docs_step1_desc: "Impor matriks gambar untuk analisis ke dalam lingkungan lab yang aman.",
    docs_step2: "Konfigurasi k",
    docs_step2_desc: "Sesuaikan parameter rank untuk mengontrol rasio fidelitas vs kompresi.",
    docs_step3: "Validasi",
    docs_step3_desc: "Tinjau statistik varians dan koefisien kemiripan untuk akurasi."
  }
};

// Custom Tooltip for Recharts
const CustomTooltip = ({ active, payload, label, lang }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white border border-slate-200 p-3 shadow-xl rounded-lg">
        <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">
          {lang === 'en' ? 'Component' : 'Komponen'} {label}
        </p>
        <p className="text-sm font-semibold text-slate-900">
          Variance: <span className="text-indigo-600">{payload[0].value.toFixed(2)}%</span>
        </p>
      </div>
    );
  }
  return null;
};

const App = () => {
  const [activeTab, setActiveTab] = useState<'pca' | 'esm' | 'docs'>('pca');
  const [lang, setLang] = useState<Lang>('en');
  const [isSidebarOpen] = useState(true);
  
  const t = translations[lang];

  // Load Instrument Sans font
  useEffect(() => {
    const link = document.createElement('link');
    link.href = 'https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400..700;1,400..700&display=swap';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
  }, []);

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 font-['Instrument_Sans',sans-serif] flex">
      {/* Sidebar */}
      <aside className={`fixed inset-y-0 left-0 z-50 w-64 bg-white border-r border-slate-200 transition-transform duration-300 transform ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:relative lg:translate-x-0`}>
        <div className="h-full flex flex-col">
          <div className="p-6 border-b border-slate-100 flex items-center gap-3">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <Cpu className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight">PCA <span className="text-indigo-600">Vision</span></h1>
          </div>
          
          <nav className="flex-1 p-4 space-y-2">
            <SidebarLink 
              icon={<LayoutDashboard className="w-4 h-4" />} 
              label={t.nav_compression} 
              active={activeTab === 'pca'} 
              onClick={() => setActiveTab('pca')} 
            />
            <SidebarLink 
              icon={<UserCheck className="w-4 h-4" />} 
              label={t.nav_recognition} 
              active={activeTab === 'esm'} 
              onClick={() => setActiveTab('esm')} 
            />
            <SidebarLink 
              icon={<BookOpen className="w-4 h-4" />} 
              label={t.nav_docs} 
              active={activeTab === 'docs'} 
              onClick={() => setActiveTab('docs')} 
            />
          </nav>

          <div className="px-6 py-4 border-t border-slate-100">
            <button 
              onClick={() => setLang(lang === 'en' ? 'id' : 'en')}
              className="w-full flex items-center justify-between px-4 py-2 bg-slate-50 hover:bg-slate-100 rounded-xl transition-colors text-[10px] font-bold uppercase tracking-widest text-slate-500"
            >
              <div className="flex items-center gap-2">
                <Languages className="w-3.5 h-3.5" />
                <span>{lang === 'en' ? 'English' : 'Indonesian'}</span>
              </div>
              <span className="text-indigo-600">Change</span>
            </button>
          </div>
          
          <div className="p-4 border-t border-slate-100">
            <div className="bg-slate-50 rounded-xl p-4">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Status</p>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                <span className="text-xs font-medium text-slate-600">{t.status_api}</span>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-16 bg-white border-b border-slate-200 px-8 flex items-center justify-between sticky top-0 z-40">
          <div className="flex items-center gap-4">
            <h2 className="text-sm font-semibold text-slate-500 capitalize">
              {activeTab === 'pca' ? t.nav_compression : activeTab === 'esm' ? t.nav_recognition : t.nav_docs}
            </h2>
            <ChevronRight className="w-4 h-4 text-slate-300" />
            <span className="text-sm font-bold text-slate-900">
              {activeTab === 'pca' ? t.tab_pca_title : activeTab === 'esm' ? t.tab_esm_title : t.tab_docs_title}
            </span>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="text-right hidden sm:block">
              <p className="text-xs font-bold text-slate-900 leading-none">{t.lab_version}</p>
              <p className="text-[10px] text-slate-400 font-medium">{t.lab_desc}</p>
            </div>
            <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center">
              <Info className="w-4 h-4 text-slate-400" />
            </div>
          </div>
        </header>

        <main className="flex-1 p-8 overflow-y-auto">
          <div className="max-w-6xl mx-auto">
            {activeTab === 'pca' ? <PCATab t={t} lang={lang} /> : activeTab === 'esm' ? <ESMTab t={t} /> : <DocsTab t={t} />}
          </div>
        </main>
      </div>
    </div>
  );
};

const SidebarLink = ({ icon, label, active, onClick }: any) => (
  <button 
    onClick={onClick}
    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-bold transition-all ${
      active 
        ? 'bg-indigo-50 text-indigo-600 shadow-sm' 
        : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'
    }`}
  >
    {icon}
    {label}
  </button>
);

const PCATab = ({ t, lang }: { t: any, lang: Lang }) => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [compressed, setCompressed] = useState<string | null>(null);
  const [k, setK] = useState(50);
  const [loading, setLoading] = useState(false);
  const [varianceData, setVarianceData] = useState<{name: number, value: number}[]>([]);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const f = e.target.files[0];
      setFile(f);
      setPreview(URL.createObjectURL(f));
    }
  };

  const runPCA = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('k', k.toString());
    try {
      const { data } = await axios.post(`${API_BASE}/api/compress`, formData);
      setCompressed(data.compressed_image);
      setVarianceData(data.variance_data.map((v: number, i: number) => ({ name: i + 1, value: v * 100 })));
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Configuration Card */}
      <div className="lg:col-span-4 space-y-6">
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-8 h-8 bg-indigo-50 rounded-lg flex items-center justify-center text-indigo-600">
              <ImageIcon className="w-4 h-4" />
            </div>
            <h3 className="text-lg font-bold">{t.pca_input}</h3>
          </div>

          <div 
            className="relative aspect-square rounded-xl border-2 border-dashed border-slate-200 hover:border-indigo-400 bg-slate-50 transition-all flex flex-col items-center justify-center p-4 group cursor-pointer overflow-hidden"
          >
            {preview ? (
              <img src={preview} className="w-full h-full object-cover rounded-lg group-hover:opacity-90 transition-opacity" />
            ) : (
              <>
                <div className="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center mb-4">
                  <Upload className="w-6 h-6 text-slate-400" />
                </div>
                <p className="text-xs font-bold text-slate-900 mb-1">{t.pca_upload}</p>
                <p className="text-[10px] text-slate-400 font-medium">{t.pca_upload_hint}</p>
              </>
            )}
            <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" onChange={handleUpload} />
          </div>

          <div className="mt-8 space-y-6">
            <div className="flex justify-between items-end">
              <div>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">{t.pca_k}</p>
                <p className="text-3xl font-bold text-indigo-600">{k}</p>
              </div>
              <p className="text-[10px] text-slate-400 font-medium pb-1">{t.pca_k_hint}</p>
            </div>
            <input 
              type="range" min="1" max="200" value={k} 
              onChange={(e) => setK(parseInt(e.target.value))}
              className="w-full h-1.5 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <button 
              onClick={runPCA}
              disabled={loading || !file}
              className="w-full py-4 bg-indigo-600 text-white rounded-xl text-sm font-bold shadow-lg shadow-indigo-200 hover:bg-indigo-700 active:scale-[0.98] transition-all disabled:opacity-40 disabled:hover:bg-indigo-600 flex items-center justify-center gap-3"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <>{t.pca_btn} <ArrowRight className="w-4 h-4" /></>}
            </button>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
          <div className="flex items-center gap-3 mb-4 text-slate-400">
            <div className="p-1 bg-slate-50 rounded">
              <Info className="w-3 h-3" />
            </div>
            <h4 className="text-[10px] font-bold uppercase tracking-wider">{t.pca_note_title}</h4>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            {t.pca_note_desc}
          </p>
        </div>
      </div>

      {/* Results Viewport */}
      <div className="lg:col-span-8 space-y-8">
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-slate-50 rounded-lg flex items-center justify-center text-slate-400">
                <BarChart3 className="w-4 h-4" />
              </div>
              <h3 className="text-lg font-bold">{t.pca_comparison}</h3>
            </div>
            <div className="flex gap-2">
              <span className="px-3 py-1 bg-slate-50 text-slate-500 text-[10px] font-bold rounded-full uppercase tracking-wider">{t.pca_split}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">{t.pca_original}</p>
                <div className="w-1.5 h-1.5 rounded-full bg-slate-300" />
              </div>
              <div className="aspect-[4/3] bg-slate-50 rounded-xl overflow-hidden border border-slate-100 flex items-center justify-center">
                {preview ? <img src={preview} className="w-full h-full object-cover" /> : <p className="text-slate-200 font-bold text-4xl italic">01</p>}
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <p className="text-xs font-bold text-indigo-600 uppercase tracking-widest">{t.pca_reconstructed}</p>
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
              </div>
              <div className="aspect-[4/3] bg-indigo-50/20 rounded-xl overflow-hidden border border-indigo-100 flex items-center justify-center">
                {compressed ? <img src={compressed} className="w-full h-full object-cover" /> : <p className="text-slate-200 font-bold text-4xl italic">02</p>}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-8 h-8 bg-indigo-50 rounded-lg flex items-center justify-center text-indigo-600">
              <BarChart3 className="w-4 h-4" />
            </div>
            <h3 className="text-lg font-bold">{t.pca_variance}</h3>
          </div>

          <div className="h-72 w-full">
            {varianceData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={varianceData} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4f46e5" stopOpacity={0.1}/>
                      <stop offset="95%" stopColor="#4f46e5" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="name" hide />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8' }} unit="%" />
                  <Tooltip content={<CustomTooltip lang={lang} />} />
                  <Area 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#4f46e5" 
                    fillOpacity={1} 
                    fill="url(#colorValue)" 
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex flex-col items-center justify-center h-full space-y-3 bg-slate-50 rounded-xl border border-slate-100">
                <BarChart3 className="w-8 h-8 text-slate-200" />
                <p className="text-xs font-bold text-slate-300 uppercase tracking-widest">{t.pca_no_data}</p>
              </div>
            )}
          </div>
          <div className="mt-6 flex items-start gap-3 p-4 bg-slate-50 rounded-xl">
            <Info className="w-4 h-4 text-slate-400 mt-0.5" />
            <p className="text-[11px] text-slate-500 leading-relaxed italic">
              {t.pca_stats_note}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

const ESMTab = ({ t }: { t: any }) => {
  const [mode, setMode] = useState<'compare' | 'recognize'>('compare');
  const [source, setSource] = useState<{f: File, p: string} | null>(null);
  const [target, setTarget] = useState<{f: File, p: string} | null>(null);
  const [result, setResult] = useState<{similarity: number; decision: string; threshold: number; name?: string; unknown?: boolean} | null>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [meanFace, setMeanFace] = useState<string | null>(null);
  const [eigenfaces, setEigenfaces] = useState<string[]>([]);

  useEffect(() => {
    if (!trained) return;
    Promise.all([
      axios.get(`${API_BASE}/api/esm/mean-face`),
      axios.get(`${API_BASE}/api/esm/eigenfaces`),
    ]).then(([mf, ef]) => {
      setMeanFace(mf.data.mean_face);
      setEigenfaces(ef.data.eigenfaces);
    }).catch(console.error);
  }, [trained]);

  const train = async () => {
    setTraining(true);
    try {
      await axios.post(`${API_BASE}/api/esm/train`);
      setTrained(true);
    } catch (e) { console.error(e); } finally { setTraining(false); }
  };

  const compare = async () => {
    if (!source || !target) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('source_file', source.f);
    formData.append('target_file', target.f);
    try {
      const { data } = await axios.post(`${API_BASE}/api/esm/compare`, formData);
      setResult(data);
    } catch (e) { console.error(e); } finally { setLoading(false); }
  };

  const recognize = async () => {
    if (!source) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', source.f);
    try {
      const { data } = await axios.post(`${API_BASE}/api/esm/recognize`, formData);
      setResult(data);
    } catch (e) { console.error(e); } finally { setLoading(false); }
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200 flex flex-col md:flex-row items-center justify-between gap-8">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600">
            <UserCheck className="w-8 h-8" />
          </div>
          <div>
            <h2 className="text-2xl font-bold tracking-tight">{t.esm_title}</h2>
            <p className="text-sm font-medium text-slate-400">{t.esm_desc}</p>
          </div>
        </div>
        
        <button 
          onClick={train} disabled={training || trained}
          className={`px-8 py-4 rounded-xl text-sm font-bold transition-all flex items-center gap-3 ${
            trained 
              ? 'bg-emerald-50 text-emerald-600 border border-emerald-100 cursor-default' 
              : 'bg-white border border-slate-200 text-slate-900 hover:bg-slate-50 active:scale-95 shadow-sm shadow-slate-100'
          }`}
        >
          {training ? <Loader2 className="w-4 h-4 animate-spin" /> : trained ? <><CheckCircle2 className="w-4 h-4" /> {t.esm_btn_ready}</> : t.esm_btn_train}
        </button>
      </div>

      {!trained && (
        <div className="bg-amber-50 border border-amber-100 rounded-2xl p-6 flex items-start gap-4">
          <AlertCircle className="w-5 h-5 text-amber-500 mt-0.5" />
          <div>
            <p className="text-sm font-bold text-amber-900 mb-1">{t.esm_alert_title}</p>
            <p className="text-xs text-amber-700 font-medium">{t.esm_alert_desc}</p>
          </div>
        </div>
      )}

      {/* Mode Toggle */}
      {trained && (
        <div className="flex justify-center">
          <div className="inline-flex bg-slate-100 rounded-xl p-1">
            <button
              onClick={() => { setMode('compare'); setResult(null); }}
              className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-all ${mode === 'compare' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >
              {t.esm_mode_compare}
            </button>
            <button
              onClick={() => { setMode('recognize'); setResult(null); }}
              className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-all ${mode === 'recognize' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >
              {t.esm_mode_recognize}
            </button>
          </div>
        </div>
      )}

      {/* PCA Visualization */}
      {trained && (meanFace || eigenfaces.length > 0) && (
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-8">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-indigo-50 rounded-lg flex items-center justify-center text-indigo-600">
              <BarChart3 className="w-4 h-4" />
            </div>
            <h3 className="text-lg font-bold">{t.esm_viz_title}</h3>
          </div>

          {meanFace && (
            <div className="space-y-3">
              <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">{t.esm_mean_face}</p>
              <div className="w-32 h-32 bg-slate-50 rounded-xl border border-slate-100 overflow-hidden">
                <img src={meanFace} className="w-full h-full object-cover" />
              </div>
            </div>
          )}

          {eigenfaces.length > 0 && (
            <div className="space-y-3">
              <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">{t.esm_eigenfaces}</p>
              <div className="grid grid-cols-4 sm:grid-cols-8 gap-3">
                {eigenfaces.map((ef, i) => (
                  <div key={i} className="space-y-1">
                    <div className="aspect-square bg-slate-50 rounded-lg border border-slate-100 overflow-hidden">
                      <img src={ef} className="w-full h-full object-cover" />
                    </div>
                    <p className="text-[9px] font-bold text-slate-300 text-center">EF {i + 1}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {mode === 'compare' ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <FaceBox label={t.esm_source} val={source} set={setSource} color="indigo" t={t} />
            <FaceBox label={t.esm_target} val={target} set={setTarget} color="slate" t={t} />
          </div>
          <CompareResult t={t} source={source} target={target} loading={loading} trained={trained} result={result} compare={compare} />
        </>
      ) : (
        <>
          <div className="max-w-md mx-auto">
            <FaceBox label={t.esm_recog_title} val={source} set={setSource} color="indigo" t={t} />
          </div>
          <RecognizeResult t={t} source={source} loading={loading} trained={trained} result={result} recognize={recognize} />
        </>
      )}
    </div>
  );
};

const CompareResult = ({ t, source, target, loading, trained, result, compare }: any) => (
  <div className="flex flex-col items-center pt-8">
    <button
      onClick={compare}
      disabled={!source || !target || loading || !trained}
      className="px-12 py-5 bg-indigo-600 text-white rounded-2xl text-base font-bold shadow-xl shadow-indigo-100 hover:bg-indigo-700 active:scale-95 transition-all disabled:opacity-40 flex items-center gap-4"
    >
      {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <>{t.esm_btn_compare} <RefreshCw className="w-4 h-4" /></>}
    </button>

    {result && (
      <div className="mt-16 w-full max-w-2xl bg-white rounded-3xl p-10 shadow-xl border border-slate-100 animate-in zoom-in-95 duration-500 text-center relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1.5 bg-slate-50">
          <div className="h-full bg-indigo-600 transition-all duration-1000 ease-out" style={{ width: `${((result.similarity + 1) / 2) * 100}%` }} />
        </div>

        <p className="text-xs font-bold text-slate-400 uppercase tracking-[0.4em] mb-4">{t.esm_sim}</p>
        <div className="text-[120px] font-bold tracking-tighter text-indigo-600 leading-none mb-6">
          {result.similarity.toFixed(2)}
        </div>

        <div className={`inline-flex items-center gap-3 px-8 py-4 rounded-2xl text-sm font-bold mb-8 ${
          result.decision === "Mirip"
            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
            : 'bg-rose-50 text-rose-700 border border-rose-200'
        }`}>
          <span className="text-[10px] uppercase tracking-[0.3em] text-current opacity-60">{t.esm_decision}</span>
          <span className="text-lg">{result.decision === "Mirip" ? t.esm_match : t.esm_no_match}</span>
          <span className="text-[10px] opacity-50">(t={result.threshold})</span>
        </div>

        <div className="inline-flex items-center gap-6 px-6 py-3 bg-slate-50 rounded-full border border-slate-100">
          <div className="text-left">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-tight">Range</p>
            <p className="text-sm font-bold text-slate-900">-1.00 – 1.00</p>
          </div>
          <div className="w-px h-8 bg-slate-200" />
          <div className="text-left">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-tight">{t.esm_conf}</p>
            <p className="text-sm font-bold text-slate-900">{result.similarity > 0.7 ? t.esm_high : result.similarity > 0.4 ? t.esm_mod : t.esm_low}</p>
          </div>
        </div>
      </div>
    )}
  </div>
);

const RecognizeResult = ({ t, source, loading, trained, result, recognize }: any) => (
  <div className="flex flex-col items-center pt-8">
    <button
      onClick={recognize}
      disabled={!source || loading || !trained}
      className="px-12 py-5 bg-indigo-600 text-white rounded-2xl text-base font-bold shadow-xl shadow-indigo-100 hover:bg-indigo-700 active:scale-95 transition-all disabled:opacity-40 flex items-center gap-4"
    >
      {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <>{t.esm_recog_btn} <RefreshCw className="w-4 h-4" /></>}
    </button>

    {result && (
      <div className="mt-16 w-full max-w-2xl bg-white rounded-3xl p-10 shadow-xl border border-slate-100 animate-in zoom-in-95 duration-500 text-center relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1.5 bg-slate-50">
          <div className={`h-full transition-all duration-1000 ease-out ${result.unknown ? 'bg-rose-400' : 'bg-emerald-500'}`} style={{ width: `${((result.similarity + 1) / 2) * 100}%` }} />
        </div>

        <p className="text-xs font-bold text-slate-400 uppercase tracking-[0.4em] mb-4">{t.esm_recog_result}</p>

        <div className={`text-5xl font-bold tracking-tight mb-4 ${result.unknown ? 'text-slate-400' : 'text-indigo-600'}`}>
          {result.unknown ? t.esm_unknown : result.name}
        </div>

        <div className="text-8xl font-bold tracking-tighter text-indigo-600 leading-none mb-6">
          {result.similarity.toFixed(2)}
        </div>

        <div className={`inline-flex items-center gap-3 px-8 py-4 rounded-2xl text-sm font-bold mb-8 ${
          !result.unknown
            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
            : 'bg-rose-50 text-rose-700 border border-rose-200'
        }`}>
          <span className="text-[10px] uppercase tracking-[0.3em] text-current opacity-60">{t.esm_decision}</span>
          <span className="text-lg">{result.unknown ? t.esm_no_match : t.esm_match}</span>
          <span className="text-[10px] opacity-50">(t={result.threshold})</span>
        </div>

        <div className="inline-flex items-center gap-6 px-6 py-3 bg-slate-50 rounded-full border border-slate-100">
          <div className="text-left">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-tight">{t.esm_sim}</p>
            <p className="text-sm font-bold text-slate-900">{result.similarity.toFixed(4)}</p>
          </div>
          <div className="w-px h-8 bg-slate-200" />
          <div className="text-left">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-tight">{t.esm_conf}</p>
            <p className="text-sm font-bold text-slate-900">{result.similarity > 0.7 ? t.esm_high : result.similarity > 0.4 ? t.esm_mod : t.esm_low}</p>
          </div>
        </div>
      </div>
    )}
  </div>
);

const FaceBox = ({ label, val, set, color, t }: any) => (
  <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-4">
    <div className="flex justify-between items-center">
      <h4 className="text-sm font-bold text-slate-900">{label}</h4>
      <div className={`w-2 h-2 rounded-full ${color === 'indigo' ? 'bg-indigo-500' : 'bg-slate-400'}`} />
    </div>
    <div className="aspect-[4/5] bg-slate-50 rounded-xl relative overflow-hidden group border border-slate-100">
      {val ? (
        <img src={val.p} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" />
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-300 space-y-4">
          <div className="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center">
            <Upload className="w-5 h-5" />
          </div>
          <span className="text-xs font-bold uppercase tracking-widest">{t.esm_import}</span>
        </div>
      )}
      <input 
        type="file" className="absolute inset-0 opacity-0 cursor-pointer" 
        onChange={(e) => e.target.files?.[0] && set({ f: e.target.files[0], p: URL.createObjectURL(e.target.files[0]) })} 
      />
    </div>
  </div>
);

const DocsTab = ({ t }: { t: any }) => (
  <div className="max-w-4xl mx-auto space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
    <div className="bg-white rounded-3xl p-10 shadow-sm border border-slate-200">
      <h2 className="text-3xl font-bold tracking-tight mb-6">{t.docs_title}</h2>
      <p className="text-slate-500 leading-relaxed mb-8">
        {t.docs_intro}
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="p-6 bg-slate-50 rounded-2xl border border-slate-100">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-3">
            <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center text-indigo-600 shadow-sm">
              <BarChart3 className="w-4 h-4" />
            </div>
            {t.docs_decomp}
          </h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            {t.docs_decomp_desc}
          </p>
        </div>
        
        <div className="p-6 bg-slate-50 rounded-2xl border border-slate-100">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-3">
            <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center text-emerald-600 shadow-sm">
              <UserCheck className="w-4 h-4" />
            </div>
            {t.docs_proj}
          </h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            {t.docs_proj_desc}
          </p>
        </div>
      </div>
    </div>

    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
        <div className="text-2xl font-bold text-slate-200 mb-2">01.</div>
        <h4 className="text-sm font-bold mb-2">{t.docs_step1}</h4>
        <p className="text-xs text-slate-400">{t.docs_step1_desc}</p>
      </div>
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
        <div className="text-2xl font-bold text-indigo-100 mb-2">02.</div>
        <h4 className="text-sm font-bold mb-2">{t.docs_step2}</h4>
        <p className="text-xs text-slate-400">{t.docs_step2_desc}</p>
      </div>
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
        <div className="text-2xl font-bold text-slate-200 mb-2">03.</div>
        <h4 className="text-sm font-bold mb-2">{t.docs_step3}</h4>
        <p className="text-xs text-slate-400">{t.docs_step3_desc}</p>
      </div>
    </div>
  </div>
);

export default App;
