# ============================================================
# ABSA Bahasa Indonesia — Streamlit Web
# Teks bebas -> BIO (deteksi aspek) -> klasifikasi (polaritas)
# ============================================================
# Model int8 dimuat dari Hugging Face Hub (repo zmuhammadrr).

import re

import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

BIO_REPO = "zmuhammadrr/absa-bio-mbert-int8"
CLS_REPO = "zmuhammadrr/absa-klasifikasi-mbert-int8"
BIO_TOK_REPO = "zmuhammadrr/absa-bio-mbert"       # tokenizer ambil dari repo asli
CLS_TOK_REPO = "zmuhammadrr/absa-klasifikasi-mbert"

# Config menyimpan LABEL generik; urutan sesuai LABEL_LIST train_bio.py
BIO_LABELS = ["O", "B-ASP", "I-ASP", "B-POS", "I-POS", "B-NEG", "I-NEG"]
# Urutan sesuai POLARITIES di schema.py: 0=negatif, 1=netral, 2=positif
CLS_LABELS = ["negatif", "netral", "positif"]


@st.cache_resource(show_spinner="Memuat model (sekali, ~20 detik)...")
def load_models():
    bio_path = hf_hub_download(BIO_REPO, "model_int8.pt")
    cls_path = hf_hub_download(CLS_REPO, "model_int8.pt")
    bio_model = torch.load(bio_path, map_location="cpu", weights_only=False).eval()
    cls_model = torch.load(cls_path, map_location="cpu", weights_only=False).eval()
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
# UI
# ------------------------------------------------------------
st.title("🇮🇩 ABSA Bahasa Indonesia")
st.caption("Model BIO menemukan aspek otomatis dari teks, lalu klasifikator mBERT "
           "memprediksi polaritasnya (negatif/netral/positif). "
           "Dilatih pada gabungan 3 dataset: hotel, teknologi, marketplace.")

teks = st.text_area("Teks ulasan", height=100,
                    placeholder="Tulis ulasan apa saja...")

if st.button("Analisis", type="primary") and teks.strip():
    spans = detect_aspects(teks)
    st.subheader("Aspek terdeteksi (model BIO)")
    st.write(", ".join(spans) if spans else
             "(tidak ada aspek terdeteksi — fallback ke Umum)")

    st.subheader("Hasil")
    st.dataframe(
        [{"Span": h["span"] or "-", "Kategori": h["kategori"],
          "Polaritas": h["polaritas"], "Confidence": h["confidence"]}
         for h in pipeline(teks)],
        use_container_width=True, hide_index=True,
    )

st.divider()
st.caption("Contoh — klik untuk mencoba:")
for c in ["Kamarnya bersih dan luas, pelayanan staff ramah banget. Tapi AC nya bocor dan sarapan hambar.",
          "Ah sumpah jelek banget packagingnya, tapi untung barangnya bagus bangett",
          "Pengiriman lama banget 2 minggu, admin seller gak respon respon.",
          "Wifi kenceng, lokasi deket stasiun, cuma parkirnya sempit."]:
    if st.button(c, key=c[:20]):
        st.session_state["isi"] = c
        st.rerun()

if st.session_state.get("isi"):
    st.session_state.setdefault("_tmp", None)
