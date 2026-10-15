"""Baseline TF-IDF dengan nearest class centroid dan cosine similarity.

Sesuai `overview.md`, TF-IDF diperlakukan sebagai representasi teks, bukan
classifier mandiri. Prediksi ditentukan oleh kedekatan cosine antara vektor
instance dan centroid setiap kelas.

Centroid dihitung dari vektor TF-IDF yang sudah dinormalisasi L2, sehingga
cosine similarity tetap bermakna. Vectorizer di-fit hanya pada data latih
kondisi terkait; data uji tidak pernah memengaruhi vocabulary maupun IDF.

Pemakaian:
    model = TfidfCentroid().fit(texts, labels)
    preds = model.predict(texts_test)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as exc:  # pragma: no cover
    raise ImportError("TfidfCentroid memerlukan scikit-learn") from exc


class TfidfCentroid:
    """TF-IDF + nearest class centroid dengan cosine similarity."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        max_features: int | None = 50000,
        sublinear_tf: bool = True,
        norm: str = "l2",
        analyzer: str = "word",
        centroid_rule: str = "mean",
    ):
        if centroid_rule not in ("mean", "sum"):
            raise ValueError(f"centroid_rule tidak dikenal: {centroid_rule!r}")
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            norm=norm,
            analyzer=analyzer,
        )
        self.centroid_rule = centroid_rule
        self.classes_: list[str] = []
        self.centroids_: np.ndarray | None = None
        self._is_fit = False
        self.config = {
            "ngram_range": list(ngram_range),
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_features,
            "sublinear_tf": sublinear_tf,
            "norm": norm,
            "analyzer": analyzer,
            "centroid_rule": centroid_rule,
        }

    # ---------- Latih ----------

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TfidfCentroid":
        if len(texts) != len(labels):
            raise ValueError("jumlah teks dan label tidak sama")
        X = self.vectorizer.fit_transform(texts)
        self.classes_ = sorted(set(labels))

        rows = []
        for name in self.classes_:
            mask = np.array([lab == name for lab in labels])
            if not mask.any():
                raise ValueError(f"kelas {name!r} tidak punya contoh latih")
            block = X[mask]
            total = np.asarray(block.sum(axis=0)).ravel()
            if self.centroid_rule == "mean":
                total = total / block.shape[0]
            rows.append(total)

        C = np.vstack(rows)
        # normalisasi L2 supaya cosine similarity tidak dipengaruhi panjang centroid
        norms = np.linalg.norm(C, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.centroids_ = C / norms
        self._is_fit = True
        return self

    # ---------- Prediksi ----------

    def decision_scores(self, texts: Sequence[str]) -> np.ndarray:
        """Cosine similarity setiap instance terhadap setiap centroid."""
        if not self._is_fit or self.centroids_ is None:
            raise RuntimeError("TfidfCentroid.predict dipanggil sebelum fit()")
        X = self.vectorizer.transform(texts)
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        return X @ self.centroids_.T

    def predict(self, texts: Sequence[str]) -> list[str]:
        scores = self.decision_scores(texts)
        idx = scores.argmax(axis=1)
        return [self.classes_[i] for i in idx]

    # ---------- Simpan / muat ----------

    def save(self, directory: Path) -> None:
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "vectorizer.joblib")
        np.save(directory / "centroids.npy", self.centroids_)
        (directory / "config.json").write_text(
            json.dumps(
                {"config": self.config, "classes": self.classes_}, indent=2
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "TfidfCentroid":
        import joblib

        directory = Path(directory)
        meta = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        cfg = dict(meta["config"])
        cfg["ngram_range"] = tuple(cfg["ngram_range"])
        obj = cls(**cfg)
        obj.vectorizer = joblib.load(directory / "vectorizer.joblib")
        obj.centroids_ = np.load(directory / "centroids.npy")
        obj.classes_ = list(meta["classes"])
        obj._is_fit = True
        return obj

    # ---------- Pemeriksaan ----------

    def top_terms(self, label: str, n: int = 15) -> list[tuple[str, float]]:
        """N-gram dengan bobot centroid tertinggi untuk satu kelas."""
        if not self._is_fit or self.centroids_ is None:
            raise RuntimeError("model belum dilatih")
        names = self.vectorizer.get_feature_names_out()
        row = self.centroids_[self.classes_.index(label)]
        order = np.argsort(-row)[:n]
        return [(names[i], float(row[i])) for i in order]
