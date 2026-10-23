<div align="center">

# ABSA Marketplace Indonesia

### Multi-Aspect Sentiment Analysis on Informal Indonesian E-Commerce Reviews

Fine-grained aspect-based opinion mining comparing statistical, recurrent, self-attention, and pretrained language models with aspect-preserving data augmentation.

[![Live Demo](https://img.shields.io/badge/Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://zmy16-absa-indo.streamlit.app)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-mBERT%20int8-yellow?style=for-the-badge)](https://huggingface.co/zmuhammadrr/absa-v2-mbert-int8)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

</div>

---

## Overview

Consumer reviews across Indonesian e-commerce platforms (Shopee, Tokopedia, Lazada) are inherently non-standard and dialectally fluid: heavily featuring slang, colloquial abbreviations, typographical character omissions, elongated vowels, and English-Indonesian code-mixing. Document-level sentiment classification is inadequate because a single review often expresses conflicting polarities across distinct transaction dimensions (e.g., praising product quality while rebuking courier delivery speed).

This repository hosts an interactive web application that operationalizes an end-to-end ABSA pipeline:
- **Automatic Multi-Aspect Detection**: Lexicon-driven domain extraction identifying up to 5 commercial aspects without manual tagging.
- **Fair Multi-Model Benchmarking**: Simultaneous polarity inference across four architectural paradigms conditioned on `{review_text} [ASPECT] {aspect_category}`.
- **Lightweight Cloud Deployment**: Full-precision mBERT (711 MB) is dynamically quantized to `qint8` (179 MB), enabling resilient inference on resource-constrained hosting (1 GB RAM quota).
- **Explainable Inference**: Interactive highlight of trigger keywords and proportional probability progress bars.

> [!NOTE]
> All classification models are evaluated using **Macro-averaged F1** as the primary metric to ensure the minority neutral class (5.2% natural occurrence) is treated with equal importance.

---

## Architecture & How It Works

```
User Input Review
       │
       ▼
[Morphology-Aware Aspect Lexicon]
       │
       ├──► Aspect 1: "Pengiriman"   ──┐
       ├──► Aspect 2: "Pelayanan"    ──┼──► Format: "{review} [ASPECT] {aspect}"
       └──► Aspect N: "..."          ──┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [Stat / From-Scratch]                       [Pretrained Benchmark]
          ├── TF-IDF Centroid (20k feats)              └── mBERT int8 (179 MB)
          ├── BiLSTM (2-layer, 128 hidden)
          └── Transformer (2-layer, 4-head)
                       │
                       ▼
           [Consensus & Confidence Bars]
```

---

## Empirical Benchmark Results

Evaluated on an independent held-out test set of 1,261 instances from a curated 8,212-instance multi-platform corpus (70/15/15 review-level split).

| Architecture | Paradigm | Macro-F1 (Original) | Macro-F1 (Augmented) | Augmentation Gain | Paired McNemar Test |
|---|---|:---:|:---:|:---:|:---:|
| **mBERT** | Pre-trained Multilingual | 0.6945 | **0.7100** | +0.0155 | Superior vs all ($p < 0.001$) |
| **Transformer** | Self-Attention (*from scratch*) | 0.5682 | **0.6008** | +0.0326 | Statistically tied vs BiLSTM |
| **BiLSTM** | Recurrent (*from scratch*) | 0.2327 | **0.5857** | **+0.3530** | Superior vs TF-IDF ($p < 0.001$) |
| **TF-IDF** | Nearest Class Centroid | 0.6175 | **0.5794** | -0.0381 | Statistical Baseline |

### Key Findings
1. **Augmentation Rescues Recurrent Failure**: BiLSTM collapses under compact original data (0.2327 F1) due to vanishing gradients on rare classes. 7-rule informal augmentation doubles training volume to 11,452 instances, driving an unprecedented **+35.3 point** leap.
2. **mBERT Holds Significant Advantage**: Pretrained multilingual subword representations establish the definitive ceiling (0.7100), verified as statistically significant over all baselines ($p < 0.001$).
3. **Robustness Vulnerability**: Stress-testing across 8 synthetic noise variants proves mBERT is near-impervious to negation and code-mixing, but highly vulnerable to **typographical errors (-5.67% drop)** and **combined noise (-7.77% drop)**.

---

## Aspect Taxonomy & Boundaries

| Aspect Category | Scope & Semantic Boundaries |
|---|---|
| **Kualitas Produk** | Physical condition, materials, functionality, authenticity (ori/kw), sizing, colors, aroma, defect, expiration date. |
| **Pengiriman** | Courier efficiency, dispatch speed, transit wait times, and package damage incurred during transit. |
| **Pelayanan** | Merchant responsiveness, chat friendliness, packaging craft, order completeness (wrong items/shortages), warranty, return, and refund handling. |
| **Harga** | Monetary cost, perceived price-to-quality value, promotional discounts, cashback, and vouchers. |
| **Umum** | Global affective impressions that contain no specific feature mentions (e.g., *"mantap recommended"*). |

> [!TIP]
> **Subject-Binding Protocol**: Packaging quality is assigned to **Pelayanan** (merchant responsibility), while courier speed/transit damage is strictly assigned to **Pengiriman** (logistics partner responsibility).

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zmy16/absa-indo.git
   cd absa-indo
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

Launch the Streamlit application:
```bash
streamlit run app.py
```
The interface will automatically open at `http://localhost:8501`.

---

## Repository Structure

```text
absa-indo/
├── .streamlit/
│   └── config.toml          # Dark theme and layout styling
├── figures/                 # Pre-generated publication-ready vector charts
│   ├── grafik_1_distribusi_kelas.png
│   ├── grafik_2_distribusi_aspek.png
│   ├── grafik_3_macro_f1_model.png
│   ├── grafik_4_f1_per_kelas.png
│   ├── grafik_5_robustness_drop.png
│   └── grafik_6_augmentation_gain.png
├── models/                  # Lightweight checkpoint artifacts
│   ├── bilstm/              # State dict, config, and vocabulary
│   ├── tfidf/               # Vectorizer and class centroids
│   └── transformer/         # State dict, config, and vocabulary
├── app.py                   # Streamlit interface and multi-model inference pipeline
├── requirements.txt         # Pinned runtime dependencies
└── README.md                # Project documentation
```

---

## Dataset & Data Sources

Rather than performing primary manual web scraping, this research leverages three pre-existing, publicly accessible Indonesian e-commerce review repositories compiled by previous researchers and open-source contributors:

| Dataset Source | Platform & Domain | Raw Size | License | Repository Link |
|---|---|:---:|:---:|---|
| **Beauty Products Review** | Shopee & Tokopedia (Beauty) | 76,256 | Apache 2.0 | [HuggingFace Dataset](https://huggingface.co/datasets/sekarmulyani/ulasan-beauty-products) |
| **Lazada Indonesian Reviews** | Lazada Indonesia (General Retail) | 203,787 | CC0: Public Domain | [Kaggle Dataset](https://www.kaggle.com/datasets/grikomsn/lazada-indonesian-reviews) |
| **Tokopedia Product Reviews** | Tokopedia (Electronics & Retail) | 65,543 | Open Data | Archived Benchmark Repository |

### Data Curation Pipeline:
1. **Deduplication & Length Filtering**: Retained reviews between 20 and 1,000 characters $\rightarrow$ 156,005 valid reviews.
2. **Aspect-Lexicon Filtering**: Screened reviews containing substantive retail aspect keywords $\rightarrow$ 112,938 eligible reviews.
3. **Stratified Sampling**: Extracted a balanced sample of **5,000 reviews** (52% beauty, 24% Lazada, 24% Tokopedia; 1,700 positive, 1,700 negative, 1,600 neutral).
4. **Instance Decomposition**: Annotated via LLM (`gpt-6-luna`, Prompt v3, 92.5% audit accuracy) into **8,212 ABSA instances** (1.64 aspects/review), partitioned 70/15/15 into 5,726 train, 1,225 validation, and 1,261 test instances.

---

## Authors & Acknowledgments

This research project was conducted by **Kelompok 17** for the **Research Methodology** curriculum at the School of Information Systems / Computer Science, **Bina Nusantara University (BINUS)**:

- **Muhammad Raid Zakwan** — NIM: 2802532684
- **Keizha Tanuwijaya Sumbaymider** — NIM: 2802512184
- **Rizky Putra Ichfiandra** — NIM: 2802530331

Special thanks to the open-source contributors of HuggingFace `ulasan-beauty-products` and Kaggle `lazada-indonesian-reviews` for curating the foundational public corpora.
