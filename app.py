"""ABSA Ulasan Marketplace Indonesia.

Aplikasi komparasi 4 model Aspect-Based Sentiment Analysis:
TF-IDF Nearest Centroid, BiLSTM, Transformer, dan mBERT.
Aspek dideteksi secara otomatis dari teks ulasan pengguna.
"""

import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

APP = Path(__file__).resolve().parent
MODELS = APP / "models"
RESULTS_FP32 = APP.parent / "results" / "augmented"
MBERT_HF = "zmuhammadrr/absa-v2-mbert-int8"

KELAS = ["negatif", "netral", "positif"]
MAXLEN = 256
ASPEK = ["Kualitas Produk", "Pengiriman", "Pelayanan", "Harga", "Umum"]

st.set_page_config(
    page_title="ABSA Marketplace Indonesia",
    page_icon="📊",
    layout="centered",
)

# ============================================================
# Deteksi aspek otomatis berbasis leksikon domain marketplace
# ============================================================
POLA_ASPEK = {
    "Pengiriman": re.compile(
        r"\b(kirim|pengiriman|kurir|ekspedisi|paket|sampai|nyampe|datang|tiba|"
        r"antar|cepat|cepet|lambat|lama|ongkir|resi|tracking)\b", re.I),
    "Pelayanan": re.compile(
        r"\b(seller|penjual|admin|toko|chat|respon|respons|balas|ramah|kemas|"
        r"kemasan|packing|bungkus|bubble|dus|kardus|retur|refund|garansi|"
        r"komplain|klaim|salah kirim|pesanan)\b", re.I),
    "Harga": re.compile(
        r"\b(harga|harganya|mahal|murah|diskon|promo|voucher|terjangkau|"
        r"kemahalan|worth|segini|kantong|duit|uang)\b", re.I),
    "Kualitas Produk": re.compile(
        r"\b(kualitas|barang|barangnya|produk|bahan|fungsi|berfungsi|rusak|"
        r"pecah|cacat|palsu|ori|original|asli|awet|warna|ukuran|size|wangi|"
        r"bau|aroma|tekstur|jahit|jahitan|bocor|patah|tipis|tebal|mati|"
        r"sesuai|bagus|jelek|exp|expired|kadaluarsa|b aja)\b", re.I),
}


def deteksi_aspek(teks):
    ditemukan = [a for a, pat in POLA_ASPEK.items() if pat.search(teks)]
    return ditemukan if ditemukan else ["Umum"]


# ============================================================
# Pemuatan Model (Cached)
# ============================================================
@st.cache_resource(show_spinner="Memuat model TF-IDF...")
def muat_tfidf():
    import joblib
    vec = joblib.load(MODELS / "tfidf" / "vectorizer.joblib")
    z = np.load(MODELS / "tfidf" / "centroids.npz")
    centroids = {k.replace("c_", ""): z[k] for k in z.files}
    cfg = json.load(open(MODELS / "tfidf" / "config.json", encoding="utf-8"))
    return vec, centroids, cfg


@st.cache_resource(show_spinner="Memuat model BiLSTM...")
def muat_bilstm():
    cfg = json.load(open(MODELS / "bilstm" / "config.json", encoding="utf-8"))
    vocab = json.load(open(MODELS / "bilstm" / "vocab.json", encoding="utf-8"))

    class BiLSTMModel(nn.Module):
        def __init__(self, vocab_size, embed=128, hidden=128, num_labels=3,
                     dropout=0.3):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
            self.lstm = nn.LSTM(embed, hidden, batch_first=True,
                                bidirectional=True)
            self.fc = nn.Linear(hidden * 2, num_labels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(self.embed(x))
            return self.fc(self.dropout(out.mean(dim=1)))

    model = BiLSTMModel(cfg["vocab_size"], cfg.get("embed", 128),
                        cfg.get("hidden", 128))
    model.load_state_dict(torch.load(MODELS / "bilstm" / "model.pt",
                                     map_location="cpu", weights_only=False))
    model.eval()
    return model, vocab, cfg


@st.cache_resource(show_spinner="Memuat model Transformer...")
def muat_transformer():
    cfg = json.load(open(MODELS / "transformer" / "config.json",
                         encoding="utf-8"))
    vocab = json.load(open(MODELS / "transformer" / "vocab.json",
                           encoding="utf-8"))

    class TransformerModel(nn.Module):
        def __init__(self, vocab_size, embed=128, heads=4, layers=2, ff=256,
                     num_labels=3, dropout=0.1):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
            layer = nn.TransformerEncoderLayer(
                d_model=embed, nhead=heads, dim_feedforward=ff,
                batch_first=True, dropout=dropout)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.fc = nn.Linear(embed, num_labels)

        def forward(self, x):
            return self.fc(self.encoder(self.embed(x)).mean(dim=1))

    model = TransformerModel(cfg["vocab_size"], cfg.get("embed", 128),
                             cfg.get("heads", 4), cfg.get("layers", 2),
                             cfg.get("ff", 256))
    model.load_state_dict(torch.load(MODELS / "transformer" / "model.pt",
                                     map_location="cpu", weights_only=False))
    model.eval()
    return model, vocab, cfg


@st.cache_resource(show_spinner="Memuat mBERT...")
def muat_mbert():
    from transformers import AutoTokenizer

    fp32 = RESULTS_FP32 / "mbert" / "final"
    if fp32.exists():
        from transformers import AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(fp32)
        model = AutoModelForSequenceClassification.from_pretrained(fp32)
        model.eval()
        return model, tok

    from huggingface_hub import hf_hub_download
    tok = AutoTokenizer.from_pretrained(MBERT_HF)
    pt = hf_hub_download(MBERT_HF, "model_int8.pt")
    model = torch.load(pt, map_location="cpu", weights_only=False)
    model.eval()
    return model, tok


# ============================================================
# Helper Inferensi
# ============================================================
def input_text(teks, aspek):
    plain = " ".join(str(teks).split())
    return f"{plain} [ASPECT] {aspek}"


def encode(vocab, teks):
    ids = [vocab.get(w, 0) for w in str(teks).lower().split()][:MAXLEN]
    return ids + [0] * (MAXLEN - len(ids))


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def pred_tfidf(teks, aspek):
    vec, centroids, _ = muat_tfidf()
    x = vec.transform([input_text(teks, aspek)]).toarray().flatten()
    nv = max(1e-9, np.linalg.norm(x))
    skor = {k: float(np.dot(x, c)) / (nv * max(1e-9, np.linalg.norm(c)))
            for k, c in centroids.items()}
    p = softmax(np.array([skor.get(k, 0.0) for k in KELAS]) * 12)
    return p


def pred_neural(model, vocab, teks, aspek):
    x = torch.LongTensor([encode(vocab, input_text(teks, aspek))])
    with torch.no_grad():
        out = model(x)
    return softmax(out[0].numpy())


def pred_mbert(teks, aspek):
    model, tok = muat_mbert()
    enc = tok(input_text(teks, aspek), truncation=True, padding="max_length",
              max_length=MAXLEN, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    return softmax(out.logits[0].detach().numpy())


# ============================================================
# Styling (Clean Anti-Slop: No fake glow, purposeful contrast)
# ============================================================
st.markdown("""<style>
:root {
    --bg-base: #0c0e12;
    --panel-surface: #141820;
    --panel-subtle: #1a202c;
    --border-dim: #263040;
    --border-focus: #3b82f6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --pos: #34d399;
    --net: #facc15;
    --neg: #f87171;
}

.stApp {
    background-color: var(--bg-base);
}

.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 50rem;
}

#MainMenu, header, footer {
    visibility: hidden;
}

.app-header {
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 1rem;
    margin-bottom: 1.4rem;
}

.app-title {
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0;
}

.app-subtitle {
    font-size: 0.92rem;
    color: var(--text-secondary);
    margin-top: 0.35rem;
    line-height: 1.45;
}

.stTextArea textarea {
    background-color: var(--panel-surface);
    border: 1px solid var(--border-dim);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.5;
}

.stTextArea textarea:focus {
    border-color: var(--border-focus);
    box-shadow: none;
}

/* Tombol submit yang kontras dan fungsional */
button[kind="primary"] {
    background-color: #f1f5f9;
    color: #0c0e12;
    border: none;
    border-radius: 8px;
    height: 2.75rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: background-color 0.15s ease;
}

button[kind="primary"]:hover {
    background-color: #ffffff;
}

/* Card per aspek */
.aspect-section {
    background-color: var(--panel-surface);
    border: 1px solid var(--border-dim);
    border-radius: 10px;
    padding: 1.15rem 1.25rem;
    margin: 1.2rem 0;
}

.aspect-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 0.75rem;
    margin-bottom: 0.9rem;
}

.aspect-name {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
}

.consensus-tag {
    font-size: 0.82rem;
    font-weight: 600;
}

/* Box model */
.model-cell {
    background-color: var(--panel-subtle);
    border: 1px solid var(--border-dim);
    border-radius: 6px;
    padding: 0.7rem 0.8rem;
    min-height: 5.5rem;
}

.model-name {
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.model-pred {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0.2rem 0 0.4rem 0;
}

.model-probs {
    font-size: 0.72rem;
    color: var(--text-secondary);
    line-height: 1.35;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
}

.example-caption {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin: 0.8rem 0 0.35rem 0;
}
</style>""", unsafe_allow_html=True)

# Header Aplikasi
st.markdown("""
<div class="app-header">
  <div class="app-title">ABSA Marketplace Indonesia</div>
  <div class="app-subtitle">
    Deteksi aspek otomatis dan komparasi polaritas lintas 4 arsitektur:
    TF-IDF, BiLSTM, Transformer, dan mBERT.
  </div>
</div>
""", unsafe_allow_html=True)

# Contoh ulasan spesifik
CONTOH = [
    ("barangnya bagus banget, cuma lama banget sampainya", "Multi-aspek: Kualitas & Pengiriman"),
    ("pengiriman cepat banget, packing rapi aman, recommended seller", "Multi-aspek: Pengiriman & Pelayanan"),
    ("barang rusak parah, minta ganti susah banget, kecewa", "Multi-aspek: Kualitas & Pelayanan"),
    ("harganya murah dan terjangkau tapi pengiriman seminggu lebih", "Multi-aspek: Harga & Pengiriman"),
    ("diterima hari ini, sesuai deskripsi", "Single-aspek: Kualitas Produk"),
    ("admin toko ramah sekali, barang original garansi resmi", "Multi-aspek: Pelayanan & Kualitas"),
]


def pilih_contoh(t):
    st.session_state["ta"] = t


teks = st.text_area(
    "Ulasan Marketplace",
    height=110,
    key="ta",
    placeholder="Ketik ulasan marketplace atau pilih salah satu contoh di bawah...",
)

st.markdown('<div class="example-caption">Pilih Contoh Pengujian:</div>', unsafe_allow_html=True)

cols = st.columns(3)
for i, (c_teks, label_c) in enumerate(CONTOH):
    ringkas = c_teks if len(c_teks) <= 36 else c_teks[:36] + "..."
    cols[i % 3].button(
        ringkas,
        key=f"c_{i}",
        help=f"{label_c}\n\n\"{c_teks}\"",
        on_click=pilih_contoh,
        args=(c_teks,),
        use_container_width=True,
    )

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
jalankan = st.button("Analisa Aspek & Sentimen", type="primary", use_container_width=True, disabled=not teks.strip())

if jalankan:
    with st.spinner("Menganalisa aspek dan menjalankan inferensi 4 model..."):
        aspek_list = deteksi_aspek(teks)

        st.markdown(
            f"<div style='margin-top:1.2rem; font-size:0.9rem; color:var(--text-secondary);'>"
            f"Aspek terdeteksi: <b style='color:var(--text-primary);'>" + ", ".join(aspek_list) + "</b></div>",
            unsafe_allow_html=True,
        )

        warna_map = {
            "negatif": "var(--neg)",
            "netral": "var(--net)",
            "positif": "var(--pos)",
        }

        try:
            m_bi, v_bi, _ = muat_bilstm()
            m_tr, v_tr, _ = muat_transformer()
        except Exception as e:
            st.error(f"Gagal memuat model neural: {e}")
            st.stop()

        for asp in aspek_list:
            baris = []

            # TF-IDF
            try:
                p = pred_tfidf(teks, asp)
                baris.append({"model": "TF-IDF", "prediksi": KELAS[int(np.argmax(p))],
                              "p": p, "macro_f1": 0.5794})
            except Exception as e:
                baris.append({"model": "TF-IDF", "error": str(e)})

            # BiLSTM
            try:
                p = pred_neural(m_bi, v_bi, teks, asp)
                baris.append({"model": "BiLSTM", "prediksi": KELAS[int(np.argmax(p))],
                              "p": p, "macro_f1": 0.5857})
            except Exception as e:
                baris.append({"model": "BiLSTM", "error": str(e)})

            # Transformer
            try:
                p = pred_neural(m_tr, v_tr, teks, asp)
                baris.append({"model": "Transformer", "prediksi": KELAS[int(np.argmax(p))],
                              "p": p, "macro_f1": 0.6008})
            except Exception as e:
                baris.append({"model": "Transformer", "error": str(e)})

            # mBERT
            try:
                p = pred_mbert(teks, asp)
                baris.append({"model": "mBERT", "prediksi": KELAS[int(np.argmax(p))],
                              "p": p, "macro_f1": 0.7100})
            except Exception as e:
                baris.append({"model": "mBERT", "error": str(e)})

            sukses = [b for b in baris if "prediksi" in b]
            if sukses:
                suara = {}
                for b in sukses:
                    suara[b["prediksi"]] = suara.get(b["prediksi"], 0) + 1
                konsensus = max(suara, key=suara.get)

                card_header = f"""
<div class="aspect-section">
  <div class="aspect-header">
    <div class="aspect-name">Aspek: {asp}</div>
    <div class="consensus-tag">
      Konsensus: <span style="color:{warna_map[konsensus]}; font-size:0.95rem;">{konsensus.upper()}</span>
      <span style="color:var(--text-secondary); font-weight:400; font-size:0.78rem;">
        ({suara[konsensus]}/{len(sukses)} model)
      </span>
    </div>
  </div>
"""
                st.markdown(card_header, unsafe_allow_html=True)

                m_cols = st.columns(4)
                for j, b in enumerate(sukses):
                    with m_cols[j]:
                        p = b["p"]
                        detail_str = "<br>".join(f"{KELAS[k]}: {p[k]*100:.1f}%" for k in range(3))
                        box_html = f"""
<div class="model-cell">
  <div class="model-name">{b['model']}</div>
  <div class="model-pred" style="color:{warna_map[b['prediksi']]};">{b['prediksi'].upper()}</div>
  <div class="model-probs">{detail_str}</div>
</div>
"""
                        st.markdown(box_html, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
FIG_DIR = APP / "figures"

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

# Section Visualisasi Hasil Penelitian
with st.expander("Hasil Eksperimen & Visualisasi Grafis", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Performa Model", "Efek Augmentasi & Ketahanan", "Distribusi Dataset"])

    with tab1:
        st.markdown("**Perbandingan Macro-F1 Lintas Model (Kondisi Augmented)**")
        st.caption("Hasil uji pada 1.261 instance. mBERT memimpin secara signifikan (p < 0,001).")
        p3 = FIG_DIR / "grafik_3_macro_f1_model.png"
        if p3.exists():
            st.image(str(p3), use_container_width=True)

        st.markdown("**F1-Score per Kelas Polaritas**")
        st.caption("F1 kelas netral lebih rendah di seluruh model akibat proporsi data latih netral sebesar 5,2%.")
        p4 = FIG_DIR / "grafik_4_f1_per_kelas.png"
        if p4.exists():
            st.image(str(p4), use_container_width=True)

    with tab2:
        st.markdown("**Augmentation Gain (Augmented vs Original)**")
        st.caption("Augmentasi menyelamatkan model BiLSTM dari kegagalan konvergensi (+0,3530 Macro-F1).")
        p6 = FIG_DIR / "grafik_6_augmentation_gain.png"
        if p6.exists():
            st.image(str(p6), use_container_width=True)

        st.markdown("**Robustness Drop mBERT terhadap 8 Fenomena Bahasa Informal**")
        st.caption("Penurunan performa saat data uji diberi perturbasi noise sintetis. Typo dan kombinasi noise adalah titik paling rentan.")
        p5 = FIG_DIR / "grafik_5_robustness_drop.png"
        if p5.exists():
            st.image(str(p5), use_container_width=True)

    with tab3:
        st.markdown("**Distribusi Kelas Polaritas per Split Data**")
        st.caption("Proporsi polaritas dijaga seimbang dan stratifikasi konsisten antara Train, Validation, dan Test.")
        p1 = FIG_DIR / "grafik_1_distribusi_kelas.png"
        if p1.exists():
            st.image(str(p1), use_container_width=True)

        st.markdown("**Distribusi Aspek pada Data Latih**")
        st.caption("Kualitas Produk dan Pelayanan merupakan aspek yang paling mendominasi diskusi pengguna e-commerce.")
        p2 = FIG_DIR / "grafik_2_distribusi_aspek.png"
        if p2.exists():
            st.image(str(p2), use_container_width=True)

with st.expander("Metodologi dan Benchmark Data"):
    st.markdown("""
- **Eksperimen 2 Kondisi Pelatihan:**
  - **Original (Kontrol):** Dilatih hanya dengan 5.726 data latih asli tanpa penambahan data sintetis.
  - **Augmented (Aktif di Web):** Dilatih dengan 11.452 data latih (asli + 7 aturan augmentasi informal). **Model yang melayani inferensi interaktif di aplikasi ini adalah versi Augmented.**
- **Benchmark Macro-F1 (Data Uji 1.261 Instance):**
  - **mBERT:** 0,7100 (Augmented) vs 0,6945 (Original) -> *Gain: +0,0155*
  - **Transformer:** 0,6008 (Augmented) vs 0,5682 (Original) -> *Gain: +0,0326*
  - **BiLSTM:** 0,5857 (Augmented) vs 0,2327 (Original) -> *Gain: +0,3530*
  - **TF-IDF:** 0,5794 (Augmented) vs 0,6175 (Original) -> *Gain: -0,0381*
- **Format Input Model:** Setiap ulasan dipasangkan dengan aspek terdeteksi: `{ulasan} [ASPECT] {aspek}`.
- **Karakteristik Kelas Netral:** Porsi kelas netral adalah 5,2% pada data latih, sehingga confidence probabilitas netral bersifat lebih konservatif dibanding positif/negatif.
""")
