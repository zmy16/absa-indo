# Panduan Deploy Streamlit (web permanen gratis)

Total waktu: ~30 menit (mayoritas menunggu upload).

---

## Langkah 1 — Quantize model di Colab (~10 menit)

1. Buka Colab (GPU tidak wajib, CPU cukup)
2. Copy isi `colab_quantize_int8.py` (folder AOL) → paste 1 sel
3. **Ganti `hf_TOKEN_KAMU`** dengan token Write baru (bikin di huggingface.co/settings/tokens)
4. Jalankan → hasil: 2 repo baru di HF Hub:
   - `zmuhammadrr/absa-bio-mbert-int8` (~200 MB)
   - `zmuhammadrr/absa-klasifikasi-mbert-int8` (~200 MB)

## Langkah 2 — GitHub (5 menit)

1. Daftar/login: github.com
2. New repository → nama: `absa-indo` → Public → Create
3. Di halaman repo: **Add file → Upload files** → upload 2 file dari folder `streamlit_app/`:
   - `app.py`
   - `requirements.txt`
4. Commit changes

## Langkah 3 — Deploy Streamlit (5 menit)

1. Buka: share.streamlit.io → **Sign in with GitHub**
2. **Create app → Deploy a public app from GitHub**
3. Pilih repo `absa-indo`, branch `main`, file `app.py`
4. Klik **Deploy**
5. Build ~5 menit (download torch + model int8)

## Hasil

```
https://USERNAME-absa-indo.streamlit.app
```

**Permanen, gratis, 24/7.** Ubah kode = edit file di GitHub, otomatis redeploy.

---

## Kalau build gagal / app mati (OOM)

RAM gratis Streamlit 1 GB — dua model int8 + torch mepet. Kalau mati:

1. Cek log di Streamlit Cloud (Manage app → Logs). Kalau `Killed` / `MemoryError`:
2. Solusi A: di `app.py`, muat model BIO pakai `st.cache_resource` (sudah) dan turunkan
   panjang teks maksimal di `split_words(text)[:250]` → `[:150]`
3. Solusi B (paling aman): upgrade Streamlit Cloud ke resource yang lebih besar
   ($1/donation tier) atau pakai opsi Gradio Colab untuk demo

## Keamanan token

Token HF yang lewat chat/upload: **revoke setelah dipakai**
(huggingface.co → Settings → Access Tokens → Delete).
