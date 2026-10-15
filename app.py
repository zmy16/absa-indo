# ============================================================
# ABSA Bahasa Indonesia - perbandingan 4 model
# Teks bebas -> BIO (deteksi aspek) -> TF-IDF / BiLSTM / Transformer / mBERT
# ============================================================
import json
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

st.set_page_config(page_title="ABSA Bahasa Indonesia", page_icon="📊",
                   layout="centered")

sys.path.insert(0, str(Path(__file__).parent / "model_code"))
from neural import BiLSTMClassifier, TransformerClassifier  # noqa: E402
from tfidf_centroid import TfidfCentroid  # noqa: E402
import vocab as vocab_mod  # noqa: E402

# ------------------------------------------------------------
# Tema: dark-tech restrained. Near-black, satu aksen, mono untuk angka.
# Tanpa gradient, tanpa emoji, tanpa glow.
# ------------------------------------------------------------
st.markdown("""<style>
:root { --bg:#0b0d10; --panel:#13161c; --line:#232833; --ink:#e8ecf4; --mut:#8b94a7; --acc:#34d399; }
.stApp { background: var(--bg); }
.block-container { padding-top: 2rem !important; max-width: 46rem; }
#MainMenu, header, footer { visibility: hidden; }
.hero { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 1.7rem 1.9rem; }
.hero h1 { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.3px; color: var(--ink); margin: 0; }
.hero p { color: var(--mut); font-size: 0.95rem; line-height: 1.55; margin: 0.6rem 0 0; max-width: 40rem; }
.hero .meta { margin-top: 1.05rem; font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace; font-size: 0.72rem; color: var(--mut); letter-spacing: 0.4px; }
.sec { font-size: 0.72rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--mut); margin: 1.5rem 0 0.6rem; }
.pill button { width: 100%; text-align: left; background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 10px; padding: 0.55rem 0.85rem; font-size: 0.87rem; line-height: 1.4; transition: border-color .15s; }
.pill button:hover { border-color: var(--acc); }
.pill button p { margin: 0; }
.stTextArea textarea { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; font-size: 1rem; color: var(--ink); }
.stTextArea textarea:focus { border-color: var(--acc); box-shadow: none; }
button[kind="primary"] { background: var(--ink); color: #0b0d10; border: none; border-radius: 10px; height: 2.8rem; font-weight: 700; font-size: 0.98rem; }
button[kind="primary"]:hover { background: #ffffff; }
button[kind="primary"]:active { transform: translateY(1px); }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.15rem; margin: 0.7rem 0; }
.head { display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap; }
.cat { font-weight: 700; font-size: 1rem; color: var(--ink); }
.spanq { font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace; font-size: 0.72rem; color: var(--mut); }
.cons { margin-left: auto; font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace; font-size: 0.7rem; letter-spacing: 0.4px; }
.rows { margin-top: 0.5rem; }
.mrow { display: grid; grid-template-columns: 7.6rem 1fr 8.4rem; gap: 0.8rem; align-items: center; padding: 0.5rem 0; }
.mrow + .mrow { border-top: 1px solid var(--line); }
.mname { font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace; font-size: 0.72rem; color: var(--mut); }
.mname b { color: var(--ink); font-weight: 500; margin-left: 0.45rem; }
.mbar { height: 3px; }
.mbar i { display: block; height: 3px; border-radius: 2px; }
.mval { font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace; font-size: 0.74rem; text-align: right; white-space: nowrap; }
.mval span { color: var(--mut); }
.note { margin-top: 0.7rem; font-size: 0.8rem; color: #d9b06c; border-top: 1px solid var(--line); padding-top: 0.6rem; }
.empty { border: 1px dashed var(--line); border-radius: 14px; padding: 2.4rem 1rem; text-align: center; color: var(--mut); margin-top: 1rem; font-size: 0.92rem; }
</style>""", unsafe_allow_html=True)

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
WARNA = {"negatif": "#e07a6b", "netral": "#d9b06c", "positif": "#6fc491"}


def baca_teks_hub(repo: str, fname: str) -> str:
    """Baca file kecil langsung dari Hub (tanpa cache hf_hub).
    Cache hf_hub_download di container pernah menyimpan salinan kosong
    sehingga json.loads gagal. Direct HTTP menghindarinya."""
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def baca_bytes_hub(repo: str, fname: str) -> bytes:
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


@st.cache_resource(show_spinner="Memuat 4 model (sekali, sekitar 40 detik)...")
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
    tmp1 = Path(tempfile.gettempdir(), "bilstm_aug.pt")
    tmp1.write_bytes(baca_bytes_hub(BILSTM_REPO, "model_augmented.pt"))
    bilstm.load_state_dict(torch.load(tmp1, map_location="cpu", weights_only=True))
    bilstm.eval()

    # --- Transformer ---
    cfg_t = json.loads(baca_teks_hub(TRANSFORMER_REPO, "config.json"))
    v_t = vocab_mod.Vocab(json.loads(baca_teks_hub(TRANSFORMER_REPO, "vocab_augmented.json"))["token_to_id"])
    trans = TransformerClassifier(vocab_size=len(v_t), num_classes=cfg_t["num_classes"],
                                  embed_dim=cfg_t["embed_dim"], num_heads=cfg_t["num_heads"],
                                  num_layers=cfg_t["num_layers"], ff_dim=cfg_t["ff_dim"],
                                  max_len=cfg_t["max_len"], dropout=cfg_t["dropout"],
                                  pad_id=0, pooling=cfg_t["pooling"])
    tmp2 = Path(tempfile.gettempdir(), "transformer_aug.pt")
    tmp2.write_bytes(baca_bytes_hub(TRANSFORMER_REPO, "model_augmented.pt"))
    trans.load_state_dict(torch.load(tmp2, map_location="cpu", weights_only=True))
    trans.eval()

    # --- TF-IDF (file kecil, langsung HTTP) ---
    with tempfile.TemporaryDirectory() as td:
        for f in ["vectorizer.joblib", "centroids.npy", "config.json"]:
            Path(td, f).write_bytes(baca_bytes_hub(TFIDF_REPO, f))
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
    """Input gabungan -> polaritas dari 4 model: {model: (polaritas, conf)}"""
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
        return [{"kategori": "Umum", "span": None, "prediksi": predict_all(text, "Umum")}]
    sumber = {}
    for s in spans:
        sumber.setdefault(span_to_category(s), s)
    return [{"kategori": k, "span": sumber[k], "prediksi": predict_all(text, k)}
            for k in sumber]


# ------------------------------------------------------------
# Kartu hasil
# ------------------------------------------------------------
def kartu(row):
    preds = row["prediksi"]
    urut = sorted(preds.items(), key=lambda kv: -kv[1][1])
    n_pola = len({p for p, _ in preds.values()})

    if n_pola == 1:
        cons = f'<span class="cons" style="color:{WARNA[urut[0][1][0]]};">4/4 sepakat</span>'
    else:
        cons = f'<span class="cons" style="color:#d9b06c;">terbelah {n_pola} polaritas</span>'

    baris = ""
    for rank, (nama, (pol, conf)) in enumerate(urut, start=1):
        w = WARNA[pol]
        pct = int(conf * 100)
        baris += f"""
        <div class="mrow">
          <div class="mname">{rank}<b>{nama}</b></div>
          <div class="mbar"><i style="width:{pct}%;background:{w};"></i></div>
          <div class="mval" style="color:{w};">{pct}% <span>{pol}</span></div>
        </div>"""

    span = row["span"]
    spanq = (f'<span class="spanq">&ldquo;{span}&rdquo;</span>' if span
             else '<span class="spanq">tanpa span, fallback Umum</span>')
    note = "" if n_pola == 1 else (
        '<div class="note">Model tidak sepakat pada aspek ini. '
        'Menarik untuk dibandingkan di analisis.</div>')

    return f"""
    <div class="card">
      <div class="head">
        <span class="cat">{row['kategori']}</span>
        {spanq}
        {cons}
      </div>
      <div class="rows">{baris}</div>
      {note}
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

st.markdown("""
<div class="hero">
  <h1>ABSA Bahasa Indonesia</h1>
  <p>Ketik ulasan informal apa saja. Model BIO menemukan aspeknya secara otomatis,
  lalu empat model memprediksi polaritasnya berdampingan, diurutkan per keyakinan.</p>
  <div class="meta">4 model &nbsp; 8 aspek &nbsp; 3 domain &nbsp; 19.116 instance latih</div>
</div>
""", unsafe_allow_html=True)


def set_example(text):
    st.session_state["teks"] = text
    st.session_state["hasil_terakhir"] = None


st.markdown("<div class='sec'>Contoh</div>", unsafe_allow_html=True)
cols = st.columns(2)
for i, c in enumerate(CONTOH):
    with cols[i % 2]:
        st.markdown("<div class='pill'>", unsafe_allow_html=True)
        st.button(c[:56] + ("…" if len(c) > 56 else ""),
                  key=f"ex{i}", on_click=set_example, args=(c,),
                  use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

teks = st.text_area("Teks ulasan", key="teks", height=110,
                    placeholder="Tulis ulasan apa saja, atau klik contoh di atas…",
                    label_visibility="collapsed")

kiri, tengah, kanan = st.columns([1.5, 1, 1.5])
with tengah:
    klik = st.button("Analisa", type="primary", use_container_width=True)

if klik:
    if not teks.strip():
        st.warning("Tulis ulasan dulu, atau klik salah satu contoh.")
    else:
        with st.spinner("Empat model sedang memproses…"):
            st.session_state["hasil_terakhir"] = pipeline(teks)

simpan = st.session_state.get("hasil_terakhir")

st.markdown("<div class='sec'>Hasil</div>", unsafe_allow_html=True)

if simpan is None:
    st.markdown("""
    <div class="empty">
      Hasil perbandingan empat model akan muncul di sini setelah analisa.
    </div>""", unsafe_allow_html=True)
else:
    for row in simpan:
        st.markdown(kartu(row), unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
st.caption("Semua model dilatih pada data augmented. Macro-F1 test: mBERT 0,671, "
           "Transformer 0,613, TF-IDF 0,609, BiLSTM 0,595. "
           "Data: IndoHotelABSA, Tech ABSA, Tokopedia.")
