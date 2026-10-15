# ============================================================
# ABSA Bahasa Indonesia — Streamlit Web
# Teks bebas -> BIO (deteksi aspek) -> klasifikasi (polaritas)
# ============================================================
import re

import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

st.set_page_config(page_title="ABSA Bahasa Indonesia", page_icon="📊",
                   layout="centered")

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
      padding: 0.15rem 0.65rem; border-radius: 999px; font-size: 0.78rem;
      font-weight: 700; letter-spacing: 0.4px;
  }
  .conf-track { background: #232c3b; border-radius: 999px; height: 7px; margin-top: 0.6rem; }
  .conf-bar { height: 7px; border-radius: 999px; }
  .conf-label { color: #9aa3af; font-size: 0.75rem; margin-top: 0.3rem; }
  .stTextArea textarea { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

BIO_REPO = "zmuhammadrr/absa-bio-mbert-int8"
CLS_REPO = "zmuhammadrr/absa-klasifikasi-mbert-int8"
BIO_TOK_REPO = "zmuhammadrr/absa-bio-mbert"
CLS_TOK_REPO = "zmuhammadrr/absa-klasifikasi-mbert"

BIO_LABELS = ["O", "B-ASP", "I-ASP", "B-POS", "I-POS", "B-NEG", "I-NEG"]
CLS_LABELS = ["negatif", "netral", "positif"]
WARNA = {"negatif": "#f87171", "netral": "#fbbf24", "positif": "#4ade80"}


@st.cache_resource(show_spinner="Memuat model (sekali, ~20 detik)...")
def load_models():
    bio_model = torch.load(hf_hub_download(BIO_REPO, "model_int8.pt"),
                           map_location="cpu", weights_only=False).eval()
    cls_model = torch.load(hf_hub_download(CLS_REPO, "model_int8.pt"),
                           map_location="cpu", weights_only=False).eval()
    bio_tok = AutoTokenizer.from_pretrained(BIO_TOK_REPO)
    cls_tok = AutoTokenizer.from_pretrained(CLS_TOK_REPO)
    return bio_model, bio_tok, cls_model, cls_tok


bio_model, bio_tokenizer, cls_model, cls_tokenizer = load_models()

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
def classify(text, category):
    inp = cls_tokenizer(f"{text} [ASPECT] {category}", truncation=True,
                        max_length=256, return_tensors="pt")
    prob = F.softmax(cls_model(**inp).logits, -1)[0]
    i = int(prob.argmax())
    return CLS_LABELS[i], float(prob[i])


def pipeline(text):
    spans = detect_aspects(text)
    if not spans:
        pol, conf = classify(text, "Umum")
        return [{"span": None, "kategori": "Umum", "polaritas": pol,
                 "confidence": round(conf, 3)}]
    hasil = []
    for span in spans:
        cat = span_to_category(span)
        pol, conf = classify(text, cat)
        hasil.append({"span": span, "kategori": cat, "polaritas": pol,
                      "confidence": round(conf, 3)})
    best = {}
    for h in hasil:
        if h["kategori"] not in best or h["confidence"] > best[h["kategori"]]["confidence"]:
            best[h["kategori"]] = h
    return list(best.values())


# ------------------------------------------------------------
# Komponen tampilan
# ------------------------------------------------------------
def kartu_hasil(h):
    w = WARNA[h["polaritas"]]
    span = h["span"] or "—"
    pct = int(h["confidence"] * 100)
    return f"""
    <div class="card">
      <div class="row1">
        <span class="kategori">{h['kategori']}</span>
        <span class="badge" style="background:{w}22;color:{w};border:1px solid {w}66;">{h['polaritas'].upper()}</span>
      </div>
      <div class="span-txt">span terdeteksi: &ldquo;{span}&rdquo;</div>
      <div class="conf-track"><div class="conf-bar" style="width:{pct}%;background:{w};"></div></div>
      <div class="conf-label">confidence {pct}%</div>
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
st.markdown("# ABSA Bahasa Indonesia")
st.markdown(
    "<div class='sub'>Ketik ulasan apa saja — model <b>BIO</b> menemukan aspeknya otomatis, "
    "lalu klasifikator <b>mBERT</b> memprediksi polaritasnya (negatif / netral / positif). "
    "Dilatih pada gabungan 3 dataset: hotel, teknologi, marketplace.</div>",
    unsafe_allow_html=True)


def set_example(text):
    st.session_state["teks"] = text


cols = st.columns(2)
for i, c in enumerate(CONTOH):
    with cols[i % 2]:
        st.button("💡 " + (c[:52] + "…" if len(c) > 52 else c),
                  key=f"ex{i}", on_click=set_example, args=(c,),
                  use_container_width=True)

teks = st.text_area("Teks ulasan", key="teks", height=110,
                    placeholder="Tulis ulasan apa saja, atau klik contoh di atas…",
                    label_visibility="collapsed")

if teks.strip():
    with st.spinner("Menganalisis…"):
        spans = detect_aspects(teks)
        hasil = pipeline(teks)

    st.markdown("**Aspek terdeteksi**")
    if spans:
        st.markdown("".join(f"<span class='chip'>{s}</span>" for s in spans),
                    unsafe_allow_html=True)
    else:
        st.markdown("<span class='chip'>tidak ada — fallback ke Umum</span>",
                    unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    st.markdown(f"**Hasil ({len(hasil)} aspek)**")
    for h in hasil:
        st.markdown(kartu_hasil(h), unsafe_allow_html=True)

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
st.caption("mBERT BIO span F1 0,684 · klasifikasi Macro-F1 0,671 · "
           "data: IndoHotelABSA + Tech ABSA + Tokopedia (19.116 instance)")
