"""ABSA Marketplace Indonesia - uji keempat model secara langsung.

Paradigma AOL v2: ASPEK adalah INPUT model, bukan keluaran.
User mengetik teks ulasan + memilih aspek, keempat model memberi prediksi.

Jalankan:
    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

MODELS = Path(__file__).resolve().parent / "models"      # repo GitHub
FP32 = Path(__file__).resolve().parent.parent / "results" / "augmented"
MBERT_HF = "zmuhammadrr/absa-v2-mbert-int8"              # int8 untuk cloud
KELAS = ["negatif", "netral", "positif"]

st.set_page_config(page_title="ABSA Marketplace Indonesia",
                   page_icon="📊", layout="centered")

# ============================================================
# muat model (cache)
# ============================================================


@st.cache_resource(show_spinner="Memuat model ...")
def muat_tfidf():
    import joblib
    vec = joblib.load(MODELS / "tfidf" / "vectorizer.joblib")
    z = np.load(MODELS / "tfidf" / "centroids.npz")
    centroids = {}
    for k in z.files:
        centroids[k.replace("c_", "")] = z[k]
    cfg = json.load(open(MODELS / "tfidf" / "config.json", encoding="utf-8"))
    return vec, centroids, cfg


@st.cache_resource(show_spinner="Memuat BiLSTM ...")
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
                                     map_location="cpu"))
    model.eval()
    return model, vocab, cfg


@st.cache_resource(show_spinner="Memuat Transformer ...")
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
                                     map_location="cpu"))
    model.eval()
    return model, vocab, cfg


@st.cache_resource(show_spinner="Memuat mBERT (±40 detik) ...")
def muat_mbert():
    """Lokal: fp32 dari results/ (unduh sekali). Cloud: int8 dari HF Hub."""
    from transformers import AutoTokenizer

    fp32 = FP32 / "mbert" / "final"
    if fp32.exists():
        from transformers import AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(fp32)
        model = AutoModelForSequenceClassification.from_pretrained(fp32)
        model.eval()
        return model, tok

    from huggingface_hub import hf_hub_download
    tok = AutoTokenizer.from_pretrained(MBERT_HF)
    pt = hf_hub_download(MBERT_HF, "model_int8.pt")
    model = torch.load(pt, map_location="cpu")
    model.eval()
    return model, tok


# ============================================================
# fungsi prediksi
# ============================================================
MAXLEN = 256
ASPEK = ["Kualitas Produk", "Pelayanan", "Pengiriman", "Harga", "Umum"]


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
    # skor kosinus belum terkalibrasi jadi probabilitas;
    # tampilkan sebagai skor kemiripan
    p = softmax(np.array([skor.get(k, 0.0) for k in KELAS]) * 12)
    return p, skor


def pred_neural(model, vocab, teks, aspek):
    x = torch.LongTensor([encode(vocab, input_text(teks, aspek))])
    with torch.no_grad():
        out = model(x)
    p = softmax(out[0].numpy())
    return p


def pred_mbert(teks, aspek):
    model, tok = muat_mbert()
    enc = tok(input_text(teks, aspek), truncation=True, padding="max_length",
              max_length=MAXLEN, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    p = softmax(out.logits[0].detach().numpy())
    return p


# ============================================================
# UI
# ============================================================
st.markdown("""<style>
:root { --bg:#0b0d10; --panel:#13161c; --line:#232833;
       --ink:#e8ecf4; --mut:#8b94a7; --acc:#34d399; }
.stApp { background:var(--bg); }
.block-container { padding-top:2rem !important; max-width:50rem; }
#MainMenu, header, footer { visibility:hidden; }
h1 { color:var(--ink); font-size:1.55rem; letter-spacing:-0.3px; }
.stTextArea textarea { background:var(--panel); border:1px solid var(--line);
    border-radius:12px; color:var(--ink); }
.stTextArea textarea:focus { border-color:var(--acc); box-shadow:none; }
div[data-baseweb="radio"] label { color:var(--ink); }
button[kind="primary"] { background:var(--ink); color:#0b0d10;
    border:none; border-radius:10px; height:2.7rem; font-weight:700; }
</style>""", unsafe_allow_html=True)

st.title("ABSA Ulasan Marketplace Indonesia")
st.caption("Klasifikasi polaritas aspek · aspek sebagai masukan model · "
           "empat model pada kondisi augmented")

# contoh ulasan siap-klik: callback dijalankan sebelum widget dirender,
# sehingga session_state widget bervalid key bisa diatur dengan aman
CONTOH = [
    ("barangnya bagus banget, cuma lama banget sampainya", "Pengiriman"),
    ("pengiriman cepat banget, packing rapi aman, recommended seller",
     "Pengiriman"),
    ("barang rusak parah, minta ganti susah banget, kecewa", "Pelayanan"),
    ("harganya oke banget buat ukuran segini", "Harga"),
    ("diterima hari ini, sesuai deskripsi", "Kualitas Produk"),
    ("admin toko tidak pernah balas chat, tapi barang original", "Pelayanan"),
]


def pilih_contoh(t, a):
    st.session_state["ta"] = t
    st.session_state["ra"] = a


col_t, col_a = st.columns([2.2, 1])

with col_t:
    teks = st.text_area("Teks ulasan", height=130, key="ta",
                        placeholder="Contoh: barangnya bagus banget, "
                                    "cuma lama banget sampainya")

with col_a:
    aspek = st.radio("Aspek yang dinilai", ASPEK, key="ra",
                     help="Aspek adalah masukan model - pilih aspek yang "
                          "ingin dinilai polaritasnya")

st.markdown(
    "<div style='margin:0.9rem 0 0.4rem; color:var(--mut); "
    "font-size:0.72rem; font-weight:600; letter-spacing:1.6px; "
    "text-transform:uppercase'>Contoh ulasan (klik satu)</div>",
    unsafe_allow_html=True)
cols = st.columns(3)
for i, (teks_contoh, aspek_contoh) in enumerate(CONTOH):
    label = teks_contoh if len(teks_contoh) <= 38 else teks_contoh[:38] + "…"
    cols[i % 3].button(label, key=f"contoh_{i}",
                       on_click=pilih_contoh, args=(teks_contoh, aspek_contoh),
                       use_container_width=True)

jalankan = st.button("Prediksi dengan 4 model", type="primary",
                      use_container_width=True, disabled=not teks.strip())

if jalankan:

    if aspek == "Umum":
        st.info("Aspek **Umum** cuma 0,8% data latih - hati-hati "
                "menginterpretasi hasilnya.")

    baris = []

    # ---- TF-IDF
    try:
        p, _ = pred_tfidf(teks, aspek)
        baris.append({"model": "TF-IDF Nearest Centroid",
                      "prediksi": KELAS[int(np.argmax(p))],
                      "kemungkinan": p,
                      "macro_f1": 0.5794})
    except Exception as e:
        st.error(f"TF-IDF gagal: {e}")

    # ---- BiLSTM
    try:
        m, v, _ = muat_bilstm()
        p = pred_neural(m, v, teks, aspek)
        baris.append({"model": "BiLSTM", "prediksi": KELAS[int(np.argmax(p))],
                      "kemungkinan": p, "macro_f1": 0.5857})
    except Exception as e:
        st.error(f"BiLSTM gagal: {e}")

    # ---- Transformer
    try:
        m, v, _ = muat_transformer()
        p = pred_neural(m, v, teks, aspek)
        baris.append({"model": "Transformer",
                      "prediksi": KELAS[int(np.argmax(p))],
                      "kemungkinan": p, "macro_f1": 0.6008})
    except Exception as e:
        st.error(f"Transformer gagal: {e}")

    # ---- mBERT
    try:
        p = pred_mbert(teks, aspek)
        baris.append({"model": "mBERT (terbaik)",
                      "prediksi": KELAS[int(np.argmax(p))],
                      "kemungkinan": p, "macro_f1": 0.7100})
    except Exception as e:
        st.error(f"mBERT gagal: {e}")

    if baris:

        # konsensus
        suara = {}
        for b in baris:
            suara[b["prediksi"]] = suara.get(b["prediksi"], 0) + 1
        konsensus = max(suara, key=suara.get)
        serempak = suara[konsensus] == len(baris)

        st.markdown("---")
        warna = {"negatif": "#ff7a76", "netral": "#ffd166",
                 "positif": "#34d399"}
        ikon = {True: "Semua model sepakat", False: "Model berbeda pendapat"}
        st.markdown(
            f"### Konsensus: <span style='color:{warna[konsensus]}'"
            f">{konsensus}</span> "
            f"<span style='color:var(--mut);font-size:0.85rem'>"
            f"({suara[konsensus]}/{len(baris)} model · "
            f"{ikon[serempak]})</span>",
            unsafe_allow_html=True)

        # kartu tiap model
        for b in baris:
            p = b["kemungkinan"]
            urut = np.argsort(p)[::-1]
            details = " · ".join(
                f"{KELAS[i]} {p[i]*100:.1f}%" for i in urut)
            st.markdown(
                f"""
<div style="background:var(--panel);border:1px solid var(--line);
     border-radius:12px;padding:0.85rem 1.1rem;margin:0.55rem 0;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;">
     <b style="color:var(--ink);font-size:0.98rem">{b["model"]}</b>
     <span style="color:var(--mut);font-size:0.72rem">Macro-F1 uji
       {b["macro_f1"]:.4f}</span>
  </div>
  <div style="margin-top:0.35rem;
       font-size:1.18rem;font-weight:700;color:{warna[b["prediksi"]]}">
       {b["prediksi"].upper()}</div>
  <div style="color:var(--mut);font-size:0.8rem;margin-top:0.25rem">
       {details}</div>
</div>""",
                unsafe_allow_html=True)

        # input yang dikirim ke model
        with st.expander("Input yang dikirim ke model"):
            st.code(input_text(teks, aspek))

st.markdown("---")
with st.expander("Tentang dan keterbatasan"):
    st.markdown("""
**Kondisi yang dipakai:** semua model dilatih pada data augmented (11.452
instance). Macro-F1 pada data uji 1.261 instance:
mBERT 0,7100 · Transformer 0,6008 · BiLSTM 0,5857 · TF-IDF 0,5794.

**Keterbatasan yang perlu diingat saat membaca hasil:**

- Kelas netral cuma 5,2% data latih, jadi prediksi netral lebih jarang
  dan lebih mudah salah (F1 mBERT untuk netral hanya 0,417)
- Skor TD-IDF adalah kemiripan kosinus yang diubah jadi distribusi
  sederhana agar bisa dibandingkan; bukan probabilitas terkalibrasi
- Aspek **Umum** hampir tidak terwakili di data latih (0,8%)
- Penurunan terbesar model ada pada teks yang banyak salah tulis
  (uji robustness: typo menurunkan Macro-F1 mBERT 5,7 poin)
- Model dievaluasi pada ulasan marketplace; keluar dari domain itu
  kinerjanya tidak dijamin
""")
