"""ABSA Ulasan Marketplace Indonesia.

Aplikasi komparasi 4 model Aspect-Based Sentiment Analysis:
TF-IDF Nearest Centroid, BiLSTM, Transformer, dan mBERT.
Dilengkapi deteksi aspek otomatis, explainability kata kunci, dan visualisasi riset.
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
# Leksikon domain marketplace (Regex dengan boundary & morfem)
# ============================================================
POLA_ASPEK = {
    "Pengiriman": re.compile(
        r"\b(kirim\w*|pengiriman|dikirim|kurir\w*|ekspedisi\w*|paket\w*|"
        r"sampai\w*|nyampe\w*|nyampai\w*|datang\w*|tiba\w*|antar\w*|"
        r"pengantaran|cepat|cepet|lambat|lama|ongkir\w*|resi|tracking|"
        r"tertunda)\b", re.I),
    "Pelayanan": re.compile(
        r"\b(seller\w*|penjual\w*|admin\w*|toko\w*|chat\w*|respon\w*|"
        r"respons\w*|balas\w*|dibalas|balasan|ramah|kemas\w*|packing\w*|"
        r"bungkus\w*|bubble|dus|kardus|retur\w*|refund\w*|garansi\w*|"
        r"komplain\w*|klaim\w*|salah kirim|salah barang|pesanan\w*|ganti\w*|"
        r"tukar\w*|pelayanan\w*|layanan\w*|service|cs|customer service|"
        r"bantu\w*|tanggung jawab)\b", re.I),
    "Harga": re.compile(
        r"\b(harga\w*|mahal\w*|murah\w*|diskon\w*|promo\w*|voucher\w*|"
        r"terjangkau|kemahalan|worth|segini|kantong|duit\w*|uang\w*|biaya\w*|"
        r"sepadan|sebanding)\b", re.I),
    "Kualitas Produk": re.compile(
        r"\b(kualitas\w*|barang\w*|produk\w*|bahan\w*|fungsi\w*|berfungsi|"
        r"rusak\w*|pecah\w*|cacat\w*|palsu|ori|original|asli|awet\w*|"
        r"warna\w*|luntur\w*|ukuran\w*|size|wangi\w*|bau\w*|aroma\w*|"
        r"tekstur\w*|jahit\w*|bocor\w*|patah\w*|tipis|tebal|mati|sesuai|"
        r"bagus\w*|jelek\w*|exp|expired|kadaluarsa|b aja|kw|tiruan)\b", re.I),
}


def deteksi_aspek(teks):
    ditemukan = [a for a, pat in POLA_ASPEK.items() if pat.search(teks)]
    return ditemukan if ditemukan else ["Umum"]


def ambil_kata_kunci(teks):
    """Mencari pasangan kata pemicu dan aspeknya untuk explainability."""
    temuan = []
    for asp, pat in POLA_ASPEK.items():
        matches = pat.findall(teks)
        for m in set(matches):
            temuan.append((m, asp))
    return temuan


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
# Styling (Clean Anti-Slop: Pure contrast & purposeful tokens)
# ============================================================
st.markdown("""<style>
:root {
    --bg-base: #0a0c10;
    --panel-surface: #12161f;
    --panel-subtle: #171d28;
    --border-dim: #232c3d;
    --border-highlight: #3b82f6;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --pos: #34d399;
    --net: #facc15;
    --neg: #f87171;
}

.stApp {
    background-color: var(--bg-base);
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 52rem;
}

#MainMenu, header, footer {
    visibility: hidden;
}

.app-header {
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 1rem;
    margin-bottom: 1.2rem;
}

.app-title {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0;
}

.app-subtitle {
    font-size: 0.9rem;
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
    border-color: var(--border-highlight);
    box-shadow: none;
}

button[kind="primary"] {
    background-color: #f1f5f9;
    color: #0c0e12;
    border: none;
    border-radius: 8px;
    height: 2.8rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: background-color 0.15s ease;
}

button[kind="primary"]:hover {
    background-color: #ffffff;
}

/* Executive Overview Bar */
.exec-bar {
    background-color: var(--panel-surface);
    border: 1px solid var(--border-dim);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin: 1.2rem 0 1rem 0;
}

.exec-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.65rem;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.45rem;
    margin-bottom: 0.35rem;
    background: var(--panel-subtle);
    border: 1px solid var(--border-dim);
}

/* Aspect Container Card */
.aspect-section {
    background-color: var(--panel-surface);
    border: 1px solid var(--border-dim);
    border-radius: 10px;
    padding: 1.2rem 1.3rem;
    margin: 1.1rem 0;
}

.aspect-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 0.75rem;
    margin-bottom: 0.95rem;
}

.aspect-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
}

.consensus-tag {
    font-size: 0.85rem;
    font-weight: 600;
}

/* Model Cell Grid */
.model-cell {
    background-color: var(--panel-subtle);
    border: 1px solid var(--border-dim);
    border-radius: 8px;
    padding: 0.75rem 0.85rem;
    min-height: 8.2rem;
}

.model-cell-top {
    border-color: #3b82f6;
    background-color: #172033;
}

.model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.model-name {
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 600;
}

.top-badge {
    font-size: 0.65rem;
    font-weight: 700;
    background: #1e3a8a;
    color: #93c5fd;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    letter-spacing: 0.03em;
}

.model-pred {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0.25rem 0 0.5rem 0;
}

/* Mini Progress Bar Probabilitas */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin: 0.2rem 0;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
}

.prob-track {
    flex: 1;
    height: 4px;
    background: #232c3d;
    border-radius: 2px;
    overflow: hidden;
}

.prob-fill-neg { background-color: var(--neg); height: 100%; }
.prob-fill-net { background-color: var(--net); height: 100%; }
.prob-fill-pos { background-color: var(--pos); height: 100%; }

.example-caption {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin: 0.8rem 0 0.35rem 0;
}

.kw-badge {
    display: inline-block;
    padding: 0.15rem 0.45rem;
    margin: 0.15rem;
    font-size: 0.75rem;
    border-radius: 4px;
    background: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1;
}

.team-card {
    background-color: var(--panel-surface);
    border: 1px solid var(--border-dim);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-top: 1.5rem;
}

.team-heading {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
}

.team-list {
    font-size: 0.85rem;
    color: var(--text-primary);
    line-height: 1.6;
}
</style>""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="app-header">
  <div class="app-title">ABSA Marketplace Indonesia</div>
  <div class="app-subtitle">
    Deteksi aspek otomatis dan komparasi polaritas lintas 4 arsitektur:
    TF-IDF, BiLSTM, Transformer, dan mBERT.<br>
    <span style="font-size:0.8rem; color:var(--text-secondary);">Proyek Penelitian Kelompok 17 · Research Methodology</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Skenario Pengujian Tematik
SKENARIO = [
    ("barangnya rusak parah, minta ganti susah banget. Kecewa", "Keluhan Kualitas & Pelayanan"),
    ("pengiriman cepat banget, packing rapi aman, recommended seller", "Pujian Pengiriman & Pelayanan"),
    ("harganya murah dan terjangkau tapi pengiriman seminggu lebih", "Kontras Harga vs Pengiriman"),
    ("diterima hari ini, sesuai deskripsi", "Ulasan Faktual (Netral)"),
    ("admin toko tidak pernah balas chat, barangnya original garansi resmi", "Kontras Layanan vs Kualitas"),
    ("worth it banget buat yang bingung duitnya mau dibuang ke mana", "Sarkasme Nilai & Harga"),
]


def pilih_skenario(teks_terpilih):
    st.session_state["ta"] = teks_terpilih


teks = st.text_area(
    "Ulasan Marketplace",
    height=100,
    key="ta",
    placeholder="Ketik ulasan marketplace atau pilih salah satu skenario pengujian di bawah...",
)

st.markdown('<div class="example-caption">Skenario Uji Cepat (Klik untuk mencoba):</div>', unsafe_allow_html=True)

cols = st.columns(3)
for i, (teks_s, label_s) in enumerate(SKENARIO):
    cols[i % 3].button(
        label_s,
        key=f"sc_{i}",
        help=f"Ulasan:\n\"{teks_s}\"",
        on_click=pilih_skenario,
        args=(teks_s,),
        width="stretch",
    )

st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
jalankan = st.button("Analisa Aspek & Sentimen", type="primary", width="stretch", disabled=not teks.strip())

if jalankan:
    with st.spinner("Menganalisa aspek dan menjalankan inferensi 4 model..."):
        aspek_list = deteksi_aspek(teks)
        kata_kunci = ambil_kata_kunci(teks)

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

        # Eksekusi inferensi semua aspek terlebih dahulu untuk ringkasan cepat
        rekap_hasil = {}
        for asp in aspek_list:
            baris = []

            # TF-IDF
            try:
                p = pred_tfidf(teks, asp)
                baris.append({"model": "TF-IDF", "prediksi": KELAS[int(np.argmax(p))],
                              "p": p, "macro_f1": 0.5794, "is_top": False})
            except Exception as e:
                baris.append({"model": "TF-IDF", "error": str(e)})

            # BiLSTM
            try:
                p = pred_neural(m_bi, v_bi, teks, asp)
                baris.append({"model": "BiLSTM", "prediksi": KELAS[int(np.argmax(p))],
                              "p": p, "macro_f1": 0.5857, "is_top": False})
            except Exception as e:
                baris.append({"model": "BiLSTM", "error": str(e)})

            # Transformer
            try:
                p = pred_neural(m_tr, v_tr, teks, asp)
                baris.append({"model": "Transformer", "prediksi": KELAS[int(np.argmax(p))],
                              "p": p, "macro_f1": 0.6008, "is_top": False})
            except Exception as e:
                baris.append({"model": "Transformer", "error": str(e)})

            # mBERT
            try:
                p = pred_mbert(teks, asp)
                baris.append({"model": "mBERT", "prediksi": KELAS[int(np.argmax(p))],
                              "p": p, "macro_f1": 0.7100, "is_top": True})
            except Exception as e:
                baris.append({"model": "mBERT", "error": str(e)})

            sukses = [b for b in baris if "prediksi" in b]
            if sukses:
                suara = {}
                for b in sukses:
                    suara[b["prediksi"]] = suara.get(b["prediksi"], 0) + 1
                konsensus = max(suara, key=suara.get)
                rekap_hasil[asp] = {"konsensus": konsensus, "suara": suara, "sukses": sukses}

        # -------------------------------------------------------------
        # 1. Executive Summary Bar (Ringkasan 1 Detik)
        # -------------------------------------------------------------
        chips_html = "".join([
            f"<span class='chip'>"
            f"<b>{asp}:</b> "
            f"<span style='color:{warna_map[info['konsensus']]}; font-weight:700;'>"
            f"{info['konsensus'].upper()}</span>"
            f"</span>"
            for asp, info in rekap_hasil.items()
        ])

        st.markdown(
            f"""
<div class="exec-bar">
  <div class="exec-title">Ringkasan Konsensus Aspek Terdeteksi:</div>
  <div>{chips_html}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # 2. Explainability: Kata Kunci Pemicu Aspek
        # -------------------------------------------------------------
        if kata_kunci:
            kw_html = "".join([
                f"<span class='kw-badge'><b>\"{kw}\"</b> -> {asp}</span>"
                for kw, asp in kata_kunci
            ])
            st.markdown(
                f"<div style='font-size:0.75rem; color:var(--text-secondary); margin-bottom:0.8rem;'>"
                f"Kata kunci pemicu: {kw_html}</div>",
                unsafe_allow_html=True,
            )

        # -------------------------------------------------------------
        # 3. Detail Per Aspek dengan Probability Bars & Top Benchmark
        # -------------------------------------------------------------
        for asp, info in rekap_hasil.items():
            konsensus = info["konsensus"]
            suara = info["suara"]
            sukses = info["sukses"]

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
                    top_badge = '<span class="top-badge">TOP 0.71</span>' if b.get("is_top") else ""
                    cell_class = "model-cell model-cell-top" if b.get("is_top") else "model-cell"

                    # Mini Progress Bars
                    neg_pct = p[0] * 100
                    net_pct = p[1] * 100
                    pos_pct = p[2] * 100

                    bars_html = f"""
<div class="prob-row">
  <span style="width:2.2rem;">Neg</span>
  <div class="prob-track"><div class="prob-fill-neg" style="width:{neg_pct:.1f}%;"></div></div>
  <span style="width:2.4rem; text-align:right;">{neg_pct:.0f}%</span>
</div>
<div class="prob-row">
  <span style="width:2.2rem;">Net</span>
  <div class="prob-track"><div class="prob-fill-net" style="width:{net_pct:.1f}%;"></div></div>
  <span style="width:2.4rem; text-align:right;">{net_pct:.0f}%</span>
</div>
<div class="prob-row">
  <span style="width:2.2rem;">Pos</span>
  <div class="prob-track"><div class="prob-fill-pos" style="width:{pos_pct:.1f}%;"></div></div>
  <span style="width:2.4rem; text-align:right;">{pos_pct:.0f}%</span>
</div>
"""
                    box_html = f"""
<div class="{cell_class}">
  <div class="model-header">
    <span class="model-name">{b['model']}</span>
    {top_badge}
  </div>
  <div class="model-pred" style="color:{warna_map[b['prediksi']]};">{b['prediksi'].upper()}</div>
  {bars_html}
</div>
"""
                    st.markdown(box_html, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

# Section Visualisasi Hasil Penelitian
FIG_DIR = APP / "figures"
with st.expander("Hasil Eksperimen & Visualisasi Grafis", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Performa Model", "Efek Augmentasi & Ketahanan", "Distribusi Dataset"])

    with tab1:
        st.markdown("**Perbandingan Macro-F1 Lintas Model (Kondisi Augmented)**")
        st.caption("Hasil uji pada 1.261 instance. mBERT memimpin secara signifikan (p < 0,001).")
        p3 = FIG_DIR / "grafik_3_macro_f1_model.png"
        if p3.exists():
            st.image(str(p3), width="stretch")

        st.markdown("**F1-Score per Kelas Polaritas**")
        st.caption("F1 kelas netral lebih rendah di seluruh model akibat proporsi data latih netral sebesar 5,2%.")
        p4 = FIG_DIR / "grafik_4_f1_per_kelas.png"
        if p4.exists():
            st.image(str(p4), width="stretch")

    with tab2:
        st.markdown("**Augmentation Gain (Augmented vs Original)**")
        st.caption("Augmentasi menyelamatkan model BiLSTM dari kegagalan konvergensi (+0,3530 Macro-F1).")
        p6 = FIG_DIR / "grafik_6_augmentation_gain.png"
        if p6.exists():
            st.image(str(p6), width="stretch")

        st.markdown("**Robustness Drop mBERT terhadap 8 Fenomena Bahasa Informal**")
        st.caption("Penurunan performa saat data uji diberi perturbasi noise sintetis. Typo dan kombinasi noise adalah titik paling rentan.")
        p5 = FIG_DIR / "grafik_5_robustness_drop.png"
        if p5.exists():
            st.image(str(p5), width="stretch")

    with tab3:
        st.markdown("**Distribusi Kelas Polaritas per Split Data**")
        st.caption("Proporsi polaritas dijaga seimbang dan stratifikasi konsisten antara Train, Validation, dan Test.")
        p1 = FIG_DIR / "grafik_1_distribusi_kelas.png"
        if p1.exists():
            st.image(str(p1), width="stretch")

        st.markdown("**Distribusi Aspek pada Data Latih**")
        st.caption("Kualitas Produk dan Pelayanan merupakan aspek yang paling mendominasi diskusi pengguna e-commerce.")
        p2 = FIG_DIR / "grafik_2_distribusi_aspek.png"
        if p2.exists():
            st.image(str(p2), width="stretch")

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

# Footer Anggota Kelompok 17
st.markdown("""
<div class="team-card">
  <div class="team-heading">Kelompok 17 — Research Methodology</div>
  <div class="team-list">
    1. <b>Keizha Tanuwijaya Sumbaymider</b> — 2802512184<br>
    2. <b>Rizky Putra Ichfiandra</b> — 2802530331<br>
    3. <b>Muhammad Raid Zakwan</b> — 2802532684
  </div>
</div>
""", unsafe_allow_html=True)
