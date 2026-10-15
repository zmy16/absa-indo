# ============================================================
# ABSA Bahasa Indonesia — Perbandingan 4 Model
# TF-IDF · BiLSTM · Transformer · mBERT
# ============================================================
# Teks bebas -> BIO menemukan aspek -> 4 model memprediksi polaritas
# per aspek, ditampilkan berdampingan (studi komparatif).
import json
import re
import sys
import urllib.request
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

st.set_page_config(page_title="ABSA Bahasa Indonesia", page_icon="📊",
                   layout="centered")

# kode model (arsitektur) dari repo
sys.path.insert(0, str(Path(__file__).parent / "model_code"))
from neural import BiLSTMClassifier, TransformerClassifier  # noqa: E402
from tfidf_centroid import TfidfCentroid  # noqa: E402
import vocab as vocab_mod  # noqa: E402

# ------------------------------------------------------------
# Tampilan
# ------------------------------------------------------------
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 46rem; }
  #MainMenu, footer { visibility: hidden; }
  h1 { font-size: 1.9rem !important; margin-bottom: 0 !important; }
  .sub { color: #9aa3af; font-size: 0.95rem; margin-bottom: 1.4rem; }
  .chip {
      display: inline-block; padding: 0.25rem 0.75rem; margin: 0.15rem 0.2rem;
      background: #1d2530; border: 1px solid #2c3848; border-radius: 999px;
      font-size: 0.9rem;
  }
  .card {
      background: #161b24; border-radius: 12px; padding: 0.9rem 1.1rem;
      margin: 0.55rem 0; border: 1px solid #232c3b;
  }
  .card .row1 { display: flex; justify-content: space-between; align-items: center; }
  .card .kategori { font-weight: 700; font-size: 1.02rem; }
  .card .span-txt { color: #9aa3af; font-size: 0.85rem; margin-top: 0.25rem; }
  .badge {
      padding: 0.12rem 0.55rem; border-radius: 999px; font-size: 0.72rem;
      font-weight: 700; letter-spacing: 0.3px; margin-left: 0.3rem;
  }
  .models { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.7rem; }
  .model-box {
      flex: 1 1 40%; background: #10151d; border: 1px solid #232c3b;
      border-radius: 9px; padding: 0.5rem 0.7rem;
  }
  .model-name { color: #9aa3af; font-size: 0.72rem; letter-spacing: 0.4px; }
  .model-pol { font-weight: 700; font-size: 0.95rem; }
  .disagree {
      margin-top: 0.6rem; font-size: 0.8rem; color: #fbbf24;
      background: #fbbf2415; border: 1px solid #fbbf2444;
      border-radius: 8px; padding: 0.4rem 0.7rem;
  }
  .stTextArea textarea { font-size: 1rem; }
  div.stButton > button {
      height: 2.9rem; font-size: 1.05rem; font-weight: 700;
      border-radius: 10px;
  }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Repo HF Hub
# ------------------------------------------------------------
BIO_REPO = "zmuhammadrr/absa-bio-mbert-int8"
CLS_REPO = "zmuhammadrr/absa-klasifikasi-mbert-int8"
BIO_TOK_REPO = "zmuhammadrr/absa-bio-mbert"
CLS_TOK_REPO = "zmuhammadrr/absa-klasifikasi-mbert"
BILSTM_REPO = "zmuhammadrr/absa-bilstm-aug"
TRANSFORMER_REPO = "zmuhammadrr/absa-transformer-aug"
TFIDF_REPO = "zmuhammadrr/absa-tfidf-aug"

BIO_LABELS = ["O", "B-ASP", "I-ASP", "B-POS", "I-POS", "B-NEG", "I-NEG"]
CLS_LABELS = ["negatif", "netral", "positif"]
WARNA = {"negatif": "#f87171", "netral": "#fbbf24", "positif": "#4ade80"}
MODELS = ["TF-IDF", "BiLSTM", "Transformer", "mBERT"]


def baca_teks_hub(repo: str, fname: str) -> str:
    """Baca file kecil langsung dari Hub (tanpa cache hf_hub).
    Cache hf_hub_download di container pernah menyimpan salinan kosong
    sehingga json.loads gagal — direct HTTP menghindarinya."""
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


@st.cache_resource(show_spinner="Memuat 4 model (sekali, ~40 detik)...")
def load_models():
    # --- BIO (penemu aspek) ---
    bio = torch.load(hf_hub_download(BIO_REPO, "model_int8.pt"),
                     map_location="cpu", weights_only=False).eval()
    bio_tok = AutoTokenizer.from_pretrained(BIO_TOK_REPO)

    # --- mBERT klasifikasi (int8) ---
    cls = torch.load(hf_hub_download(CLS_REPO, "model_int8.pt"),
                     map_location="cpu", weights_only=False).eval()
    cls_tok = AutoTokenizer.from_pretrained(CLS_TOK_REPO)

    # --- BiLSTM ---
    cfg = json.loads(baca_teks_hub(BILSTM_REPO, "config.json"))
    v = vocab_mod.Vocab(json.loads(baca_teks_hub(BILSTM_REPO, "vocab_augmented.json"))["token_to_id"])
    bilstm = BiLSTMClassifier(vocab_size=len(v), num_classes=cfg["num_classes"],
                              embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
                              num_layers=cfg["num_layers"], dropout=cfg["dropout"],
                              pad_id=0, pooling=cfg["pooling"])
    bilstm.load_state_dict(torch.load(
        hf_hub_download(BILSTM_REPO, "model_augmented.pt"),
        map_location="cpu", weights_only=True))
    bilstm.eval()

    # --- Transformer ---
    cfg_t = json.loads(baca_teks_hub(TRANSFORMER_REPO, "config.json"))
    v_t = vocab_mod.Vocab(json.loads(baca_teks_hub(TRANSFORMER_REPO, "vocab_augmented.json"))["token_to_id"])
    trans = TransformerClassifier(vocab_size=len(v_t), num_classes=cfg_t["num_classes"],
                                  embed_dim=cfg_t["embed_dim"], num_heads=cfg_t["num_heads"],
                                  num_layers=cfg_t["num_layers"], ff_dim=cfg_t["ff_dim"],
                                  max_len=cfg_t["max_len"], dropout=cfg_t["dropout"],
                                  pad_id=0, pooling=cfg_t["pooling"])
    trans.load_state_dict(torch.load(
        hf_hub_download(TRANSFORMER_REPO, "model_augmented.pt"),
        map_location="cpu", weights_only=True))
    trans.eval()

    # --- TF-IDF ---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for f in ["vectorizer.joblib", "centroids.npy", "config.json"]:
            Path(td, f).write_bytes(hf_hub_download(TFIDF_REPO, f))
        tfidf = TfidfCentroid.load(Path(td))

    return bio, bio_tok, cls, cls_tok, (bilstm, v, cfg), (trans, v_t, cfg_t), tfidf


(bio_model, bio_tokenizer, cls_model, cls_tokenizer,
 (bilstm, bilstm_vocab, bilstm_cfg), (transf, trans_vocab, trans_cfg), tfidf) = load_models()

# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------
SPAN_MAP = [
    (r"kirim|pengirim|paket|kurir|antar|sampai|delivery|packaging|packing|bungkus", "Pengiriman"),
    (r"harga|biaya|tarif|bayar|mahal|murah|price", "Harga"),
    (r"bersih|kotor|bau|debu|kebersihan|clean", "Kebersihan"),
    (r"lokasi|strategis|dekat|jauh|akses|location", "Lokasi"),
    (r"pelayanan|layanan|staf|resepsionis|staff|servis|service|cs|admin|seller|penjual", "Pelayanan"),
    (r"fasilitas|kolam|parkir|lift|wifi|pool|gym", "Fasilitas"),
    (r"kamar|bed|makanan|sarapan|minuman|produk|barang|kualitas|quality|baterai|layar|kamera", "Kualitas Produk"),
]


def split_words(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


@torch.no_grad()
def detect_aspects(text):
    words = split_words(text)[:250]
    if not words:
        return []
    ids, first_idx = [bio_tokenizer.cls_token_id], []
    for w in words:
        first_idx.append(len(ids))
        sub = bio_tokenizer.encode(w, add_special_tokens=False) or [bio_tokenizer.unk_token_id]
        ids.extend(sub)
    ids.append(bio_tokenizer.sep_token_id)
    t = torch.tensor([ids])
    logits = bio_model(input_ids=t, attention_mask=torch.ones_like(t)).logits[0]
    pred = logits.argmax(-1).tolist()
    spans, cur = [], []
    for wi, w in enumerate(words):
        if first_idx[wi] >= len(pred):
            break
        tag = BIO_LABELS[pred[first_idx[wi]]]
        if tag == "B-ASP":
            if cur:
                spans.append(cur)
            cur = [w]
        elif tag == "I-ASP" and cur:
            cur.append(w)
        else:
            if cur:
                spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return [" ".join(s) for s in spans]


def span_to_category(span):
    low = span.lower()
    for pat, cat in SPAN_MAP:
        if re.search(pat, low):
            return cat
    return "Umum"


@torch.no_grad()
def predict_all(text, category):
    """Input gabungan -> polaritas dari 4 model: {model: (polaritas, conf)"""
    inp_text = f"{text} [ASPECT] {category}"
    hasil = {}

    # TF-IDF
    scores = tfidf.decision_scores([inp_text])[0]
    probs_t = F.softmax(torch.tensor(scores), -1)
    i = int(probs_t.argmax())
    hasil["TF-IDF"] = (tfidf.classes_[i], float(probs_t[i]))

    # BiLSTM & Transformer (vocab + padding sama seperti train.py)
    import numpy as np
    for name, net, voc, cfg in [("BiLSTM", bilstm, bilstm_vocab, bilstm_cfg),
                                ("Transformer", transf, trans_vocab, trans_cfg)]:
        ids = voc.encode_text(inp_text)[: cfg["max_len"]]
        arr = np.full((1, max(1, len(ids))), 0, dtype=np.int64)
        arr[0, : len(ids)] = ids
        logits = net(torch.from_numpy(arr))[0]
        probs = F.softmax(logits, -1)
        j = int(probs.argmax())
        hasil[name] = (CLS_LABELS[j], float(probs[j]))

    # mBERT (int8)
    enc = cls_tokenizer(inp_text, truncation=True, max_length=256, return_tensors="pt")
    probs_m = F.softmax(cls_model(**enc).logits, -1)[0]
    k = int(probs_m.argmax())
    hasil["mBERT"] = (CLS_LABELS[k], float(probs_m[k]))

    return hasil


def pipeline(text):
    spans = detect_aspects(text)
    if not spans:
        kategori = ["Umum"]
        sumber = {None: "Umum"}
    else:
        sumber = {}
        for s in spans:
            sumber.setdefault(span_to_category(s), s)
        kategori = list(sumber.keys())
    return [{"kategori": k, "span": sumber[k], "prediksi": predict_all(text, k)}
            for k in kategori]


# ------------------------------------------------------------
# Komponen
# ------------------------------------------------------------
def kartu_hasil(row):
    preds = row["prediksi"]
    setuju = len({p for p, _ in preds.values()}) == 1
    boxes = "".join(
        f"""<div class="model-box">
              <div class="model-name">{m.upper()}</div>
              <div class="model-pol" style="color:{WARNA[p]};">{p.upper()}
                <span style="color:#9aa3af;font-weight:400;font-size:0.72rem;">{int(c*100)}%</span>
              </div>
            </div>"""
        for m, (p, c) in preds.items())
    cat = row["kategori"]
    span = row["span"]
    span_txt = f'span: &ldquo;{span}&rdquo;' if span else "fallback (tidak ada aspek terdeteksi)"
    cat_badge = f'<span class="badge" style="background:#5b9dff22;color:#5b9dff;border:1px solid #5b9dff66;">{cat}</span>'
    disagree = ""
    if not setuju:
        disagree = ('<div class="disagree">⚠ Model tidak sepakat — '
                    'menarik untuk analisis komparatif.</div>')
    return f"""
    <div class="card">
      <div class="row1">{cat_badge}</div>
      <div class="span-txt">{span_txt}</div>
      <div class="models">{boxes}</div>
      {disagree}
    </div>"""


CONTOH = [
    "Kamarnya bersih dan luas, pelayanan staff ramah banget. Tapi AC nya bocor dan sarapan hambar.",
    "Ah sumpah jelek banget packagingnya, tapi untung barangnya bagus bangett",
    "Pengiriman lama banget 2 minggu, admin seller gak respon respon.",
    "Wifi kenceng, lokasi deket stasiun, cuma parkirnya sempit.",
]

# ------------------------------------------------------------
# Halaman
# ------------------------------------------------------------
if "hasil_terakhir" not in st.session_state:
    st.session_state["hasil_terakhir"] = None

st.markdown("# ABSA Bahasa Indonesia")
st.markdown(
    "<div class='sub'>Ketik ulasan apa saja — model <b>BIO</b> menemukan aspeknya, "
    "lalu <b>4 model</b> (TF-IDF, BiLSTM, Transformer, mBERT) memprediksi polaritasnya "
    "berdampingan. Studi komparatif pada ulasan informal bahasa Indonesia.</div>",
    unsafe_allow_html=True)


def set_example(text):
    st.session_state["teks"] = text
    st.session_state["hasil_terakhir"] = None


st.caption("💡 Coba contoh:")
cols = st.columns(2)
for i, c in enumerate(CONTOH):
    with cols[i % 2]:
        st.button(c[:52] + ("…" if len(c) > 52 else ""),
                  key=f"ex{i}", on_click=set_example, args=(c,),
                  use_container_width=True)

teks = st.text_area("Teks ulasan", key="teks", height=110,
                    placeholder="Tulis ulasan apa saja, atau klik contoh di atas…",
                    label_visibility="collapsed")

kiri, tengah, kanan = st.columns([1.2, 1, 1.2])
with tengah:
    klik = st.button("🔍  Analisa", type="primary", use_container_width=True)

if klik:
    if not teks.strip():
        st.warning("Tulis ulasan dulu, atau klik salah satu contoh di atas.")
    else:
        with st.spinner("4 model sedang membaca ulasan…"):
            st.session_state["hasil_terakhir"] = pipeline(teks)

simpan = st.session_state.get("hasil_terakhir")

if simpan is None:
    st.markdown("""
    <div class="card" style="text-align:center;color:#9aa3af;padding:2rem 1rem;">
      Hasil perbandingan 4 model akan muncul di sini.<br>
      <span style="font-size:0.85rem;">1 teks &rarr; aspek &rarr; polaritas dari tiap model.</span>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"**{len(simpan)} aspek terdeteksi**")
    for row in simpan:
        st.markdown(kartu_hasil(row), unsafe_allow_html=True)

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
st.caption("Semua model dilatih pada data augmented · mBERT Macro-F1 0,671 · "
           "BiLSTM 0,595 · Transformer 0,613 · TF-IDF 0,609 · "
           "data: IndoHotelABSA + Tech ABSA + Tokopedia")
