"""Tokenizer kata dan vocabulary bersama untuk BiLSTM dan Transformer.

mBERT tidak memakai modul ini; ia memakai tokenizer checkpoint-nya sendiri.

Token khusus memakai tanda kurung siku gaya BERT dan diekstrak dengan regex
tersendiri sebelum regex kata dijalankan. Tanpa langkah itu, `[ASPECT]` akan
terpecah menjadi `[`, `aspect`, `]` dan penanda aspek kehilangan maknanya.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
ASPECT_TOKEN = "[ASPECT]"

SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, ASPECT_TOKEN)
PAD_ID, UNK_ID, CLS_ID, SEP_ID, ASPECT_ID = range(len(SPECIAL_TOKENS))

SPECIAL_RE = re.compile(r"\[(PAD|UNK|CLS|SEP|ASPECT)\]", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)?|\d+|[^\sA-Za-zÀ-ÿ\d]")


def tokenize(text: str) -> list[str]:
    """Pecah teks menjadi token; token khusus tetap satu unit."""
    tokens: list[str] = []
    pos = 0
    for match in SPECIAL_RE.finditer(text):
        tokens.extend(WORD_RE.findall(text[pos:match.start()]))
        tokens.append(match.group(0).upper())
        pos = match.end()
    tokens.extend(WORD_RE.findall(text[pos:]))
    return [t.lower() if not t.startswith("[") else t for t in tokens]


def combine_text(review: str, aspect: str) -> str:
    """Bentuk input model: ulasan diikuti penanda aspek."""
    return f"{review} {ASPECT_TOKEN} {aspect}"


class Vocab:
    """Pemetaan token ke id, dengan token khusus pada id tetap."""

    def __init__(self, token_to_id: dict[str, int]):
        for name, idx in zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS))):
            if token_to_id.get(name) != idx:
                raise ValueError(f"id token khusus {name} harus {idx}")
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def aspect_id(self) -> int:
        return ASPECT_ID

    def encode(self, tokens: Sequence[str], prepend_cls: bool = False) -> list[int]:
        ids = [self.token_to_id.get(t, UNK_ID) for t in tokens]
        if prepend_cls:
            ids = [CLS_ID] + ids
        return ids

    def encode_text(self, text: str, prepend_cls: bool = False) -> list[int]:
        return self.encode(tokenize(text), prepend_cls=prepend_cls)

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token.get(i, UNK_TOKEN) for i in ids]

    # ---------- Bangun ----------

    @classmethod
    def build(
        cls, texts: Sequence[str], min_freq: int = 2, max_size: int | None = None
    ) -> "Vocab":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        # token khusus selalu ada dan tidak ikut dihitung dari korpus
        for name in SPECIAL_TOKENS:
            counts.pop(name, None)

        items = [(t, c) for t, c in counts.items() if c >= min_freq]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        if max_size is not None:
            items = items[: max(0, max_size - len(SPECIAL_TOKENS))]

        token_to_id = {name: idx for idx, name in enumerate(SPECIAL_TOKENS)}
        for token, _ in items:
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    # ---------- Simpan / muat ----------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"special_tokens": list(SPECIAL_TOKENS), "token_to_id": self.token_to_id},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["token_to_id"])

    # ---------- Diagnostik ----------

    def oov_rate(self, texts: Sequence[str]) -> dict[str, float]:
        """Porsi token di luar vocabulary, unik dan per kemunculan."""
        unique_total = unique_oov = occ_total = occ_oov = 0
        for text in texts:
            for token in tokenize(text):
                occ_total += 1
                if token not in self.token_to_id:
                    occ_oov += 1
            for token in set(tokenize(text)):
                unique_total += 1
                if token not in self.token_to_id:
                    unique_oov += 1
        return {
            "unique_oov_rate": unique_oov / unique_total if unique_total else 0.0,
            "occurrence_oov_rate": occ_oov / occ_total if occ_total else 0.0,
        }
