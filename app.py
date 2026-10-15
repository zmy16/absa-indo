# ============================================================
# ABSA Bahasa Indonesia — Model Arena
# Teks bebas -> BIO (deteksi aspek) -> 4 model berlomba
# TF-IDF · BiLSTM · Transformer · mBERT
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

st.set_page_config(page_title="ABSA Bahasa Indonesia — Model Arena",
                   page_icon="📊", layout="centered")

sys.path.insert(0, str(Path(__file__).parent / "model_code"))
from neural import BiLSTMClassifier, TransformerClassifier  # noqa: E402
from tfidf_centroid import TfidfCentroid  # noqa: E402
import vocab as vocab_mod  # noqa: E402

# ------------------------------------------------------------
# Tema: dark editorial + gradient
# ------------------------------------------------------------
st.markdown("""<style>  :root { --bg:#0a0d14; --panel:#121722; --line:#202a3c; --ink:#e8ecf4; --mut:#8b94a7; }  .stApp { background: var(--bg); font-family: -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; }  .block-container { padding-top: 0 !important; max-width: 47rem; }  #MainMenu, header, footer { visibility: hidden; }  .hero { background: linear-gradient(135deg,#1b2440 0%,#3b2a6d 55%,#6d2a5c 100%); border-radius: 0 0 22px 22px; padding: 2.4rem 2.2rem 2rem; margin: 0 -5rem 1.4rem -5rem; position: relative; overflow: hidden; }  .hero::after { content:""; position:absolute; inset:0; background: radial-gradient(600px 200px at 85% -20%, #ffffff22, transparent 60%); }  .hero h1 { font-weight:800; font-size:2.1rem; color:#fff; margin:0; letter-spacing:-0.5px; }  .hero p { color:#c9c4e8; font-size:0.97rem; margin:0.7rem 0 0; max-width:38rem; }  .hero .tag { display:inline-block; font-size:0.66rem; letter-spacing:2px; color:#a5f3d0; background:#0e3b2e88; border:1px solid #2dd4a055; padding:0.2rem 0.7rem; border-radius:999px; margin-bottom:0.9rem; }  .strip { display:flex; gap:1.8rem; margin-top:1.3rem; position:relative; z-index:1; }  .strip b { font-size:1.25rem; color:#fff; display:block; line-height:1.1; }  .strip span { color:#b9b3d9; font-size:0.7rem; letter-spacing:1.5px; }  h2.sec { font-size:0.78rem; letter-spacing:2.5px; color:var(--mut); text-transform:uppercase; margin:1.5rem 0 0.7rem; font-weight:600; }  .pill button { width:100%; text-align:left; background:var(--panel); color:var(--ink); border:1px solid var(--line); border-radius:12px; padding:0.6rem 0.9rem; font-size:0.87rem; line-height:1.35; transition: border-color .15s, transform .15s; }  .pill button:hover { border-color:#7c6cf0; transform: translateY(-1px); }  .pill button p { margin:0; }  .stTextArea textarea { background:var(--panel); border:1px solid var(--line); border-radius:14px; font-size:1.02rem; color:var(--ink); }  .stTextArea textarea:focus { border-color:#7c6cf0; box-shadow:0 0 0 3px #7c6cf028; }  div.stButton > button { width:100%; height:3.1rem; font-weight:800; font-size:1.03rem; color:#fff; background:linear-gradient(90deg,#6d5df0,#a55eea); border:none; border-radius:13px; letter-spacing:0.3px; transition: filter .15s, transform .15s; }  div.stButton > button:hover { filter:brightness(1.12); transform:translateY(-1px); }  .card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1.05rem 1.15rem; margin:0.8rem 0; }  .card .head { display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap; }  .cat { font-weight:800; font-size:1.04rem; color:#fff; }  .spanq { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.72rem; color:#9fe8c8; background:#0e3b2e66; border:1px solid #2dd4a033; padding:0.12rem 0.55rem; border-radius:999px; }  .consensus { margin-left:auto; font-size:0.68rem; padding:0.18rem 0.6rem; border-radius:999px; letter-spacing:0.5px; font-weight:700; }  .arena { margin-top:0.85rem; display:flex; flex-direction:column; gap:0.55rem; }  .mrow { display:grid; grid-template-columns: 8.6rem 1fr 5rem; gap:0.7rem; align-items:center; }  .mname { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.73rem; color:var(--mut); display:flex; align-items:center; gap:0.3rem; }  .mname .medal { font-size:0.82rem; }  .track { background:#0d1220; border-radius:999px; height:14px; overflow:hidden; border:1px solid #1a2334; }  .fill { height:100%; border-radius:999px; transition: width .5s cubic-bezier(.2,.8,.2,1); }  .mval { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.75rem; font-weight:700; text-align:right; }  .mval small { color:var(--mut); font-weight:500; }  .note { margin-top:0.75rem; font-size:0.8rem; color:#fbbf24; background:#fbbf2412; border:1px solid #fbbf2433; border-radius:10px; padding:0.5rem 0.75rem; }  .empty { border:1px dashed #2a3550; border-radius:16px; padding:2.6rem 1rem; text-align:center; color:var(--mut); margin-top:1.4rem; }  .empty .big { font-size:1.9rem; margin-bottom:0.4rem; }  ::selection { background:#7c6cf066; }</style>""", unsafe_allow_html=True)

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
MEDALI = {0: "🥇", 1: "🥈", 2: "🥉", 3: " ·"}


def baca_teks_hub(repo: str, fname: str) -> str:
    """Baca file kecil langsung dari Hub (tanpa cache hf_hub).
    Cache hf_hub_download di container pernah menyimpan salinan kosong
    sehingga json.loads gagal — direct HTTP menghindarinya."""
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def baca_bytes_hub(repo: str, fname: str) -> bytes:
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


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

    # --- TF-IDF (file kecil, langsung HTTP — hindari cache rusak) ---
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
# Komponen tampilan
# ------------------------------------------------------------
def kartu_arena(row):
    preds = row["prediksi"]
    urut = sorted(preds.items(), key=lambda kv: -kv[1][1])  # conf tertinggi dulu
    setuju = len({p for p, _ in preds.values()}) == 1

    baris = ""
    for rank, (nama, (pol, conf)) in enumerate(urut):
        w = WARNA[pol]
        pct = int(conf * 100)
        baris += f"""
        <div class="mrow">
          <div class="mname"><span class="medal">{MEDALI[rank]}</span>{nama.upper()}</div>
          <div class="track"><div class="fill" style="width:{pct}%;background:{w};opacity:{1 - rank * 0.16};"></div></div>
          <div class="mval" style="color:{w};">{pct}%<small> {pol}</small></div>
        </div>"""

    span = row["span"]
    spanq = (f'<span class="spanq">&ldquo;{span}&rdquo;</span>' if span
             else '<span class="spanq" style="color:#8b94a7;background:#1a2130aa;border-color:#2a3550;">fallback</span>')
    if setuju:
        cons = (f'<span class="consensus" style="color:#4ade80;background:#4ade8015;'
                f'border:1px solid #4ade8044;">KONSENSUS 4/4</span>')
    else:
        n_bed = len({p for p, _ in preds.values()})
        cons = (f'<span class="consensus" style="color:#fbbf24;background:#fbbf2415;'
                f'border:1px solid #fbbf2444;">TERPISAH {n_bed}/4</span>')
    note = "" if setuju else ('<div class="note">⚠ Model tidak sepakat — span ini menarik '
                              'untuk analisis komparatif.</div>')

    return f"""
    <div class="card">
      <div class="head">
        <span class="cat">{row['kategori']}</span>
        {spanq}
        {cons}
      </div>
      <div class="arena">{baris}</div>
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

# ---- HERO ----
st.markdown("""
<div class="hero">
  <span class="tag">ASPECT-BASED SENTIMENT ANALYSIS · BAHASA INDONESIA</span>
  <h1>Model Arena</h1>
  <p>Ketik ulasan informal apa saja. Model <b>BIO</b> menemukan aspeknya secara otomatis —
  lalu <b>4 model</b> bersaing memprediksi polaritasnya, peringkat berdasarkan keyakinan.</p>
  <div class="strip">
    <div class="st"><b>4</b><span>MODEL</span></div>
    <div class="st"><b>8</b><span>ASPEK</span></div>
    <div class="st"><b>3</b><span>DOMAIN</span></div>
    <div class="st"><b>19.116</b><span>INSTANCE LATIH</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


def set_example(text):
    st.session_state["teks"] = text
    st.session_state["hasil_terakhir"] = None


st.markdown("<h2 class='sec'>Coba contoh</h2>", unsafe_allow_html=True)
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

kiri, tengah, kanan = st.columns([1.4, 1, 1.4])
with tengah:
    klik = st.button("⚡  Analisa Sekarang", use_container_width=True)

if klik:
    if not teks.strip():
        st.warning("Tulis ulasan dulu, atau klik salah satu contoh.")
    else:
        with st.spinner("4 model sedang bertanding…"):
            st.session_state["hasil_terakhir"] = pipeline(teks)

simpan = st.session_state.get("hasil_terakhir")

st.markdown("<h2 class='sec'>Papan peringkat</h2>", unsafe_allow_html=True)

if simpan is None:
    st.markdown("""
    <div class="empty">
      <div class="big">🏟️</div>
      Hasil pertandingan 4 model akan muncul di sini.<br>
      <span style="font-size:0.85rem;">1 teks → aspek → 4 prediksi diperingkat per keyakinan.</span>
    </div>""", unsafe_allow_html=True)
else:
    for row in simpan:
        st.markdown(kartu_arena(row), unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)
st.caption("Semua model dilatih pada data augmented · Macro-F1: mBERT 0,671 · Transformer 0,613 · "
           "TF-IDF 0,609 · BiLSTM 0,595 · data: IndoHotelABSA + Tech ABSA + Tokopedia")
