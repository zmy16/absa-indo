"""BiLSTM dan Transformer Encoder untuk klasifikasi polaritas per aspek.

Keduanya menerima `input_ids` berukuran (B, T) dan mengembalikan logits
berukuran (B, jumlah kelas). Label mengikuti urutan tetap `LABELS`, sehingga
indeks logits selalu sama di seluruh pipeline.

BiLSTM memakai `pack_padded_sequence` supaya token PAD tidak memengaruhi
rekurensi. Transformer memakai `src_key_padding_mask` untuk alasan yang sama.
"""
from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class BiLSTMClassifier(nn.Module):
        """Embedding -> BiLSTM -> pooling -> dropout -> linear."""

        def __init__(
            self,
            vocab_size: int,
            num_classes: int,
            embed_dim: int = 128,
            hidden_dim: int = 128,
            num_layers: int = 1,
            dropout: float = 0.3,
            pad_id: int = 0,
            pooling: str = "last",
        ):
            super().__init__()
            if pooling not in ("last", "mean", "max"):
                raise ValueError(f"pooling tidak dikenal: {pooling!r}")
            self.vocab_size = vocab_size
            self.num_classes = num_classes
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout_p = dropout
            self.pad_id = pad_id
            self.pooling = pooling

            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
            mask = input_ids.ne(self.pad_id)  # (B, T)
            lengths = mask.sum(dim=1).clamp(min=1).cpu()

            emb = self.embedding(input_ids)
            packed = pack_padded_sequence(
                emb, lengths, batch_first=True, enforce_sorted=False
            )
            out, (h_n, _) = self.lstm(packed)

            if self.pooling == "last":
                pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2H)
            else:
                padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
                    out, batch_first=True, total_length=input_ids.size(1)
                )
                valid = mask.unsqueeze(-1).float()
                if self.pooling == "mean":
                    pooled = (padded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                else:
                    padded = padded.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                    pooled = padded.max(dim=1).values

            pooled = self.dropout(pooled)
            return self.classifier(pooled)

        def config_dict(self) -> dict:
            return {
                "vocab_size": self.vocab_size,
                "num_classes": self.num_classes,
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout_p,
                "pad_id": self.pad_id,
                "pooling": self.pooling,
            }

    class PositionalEncoding(nn.Module):
        """Encoding posisi sinusoidal, bukan parameter terlatih."""

        def __init__(self, embed_dim: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
            )
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if x.size(1) > self.pe.size(1):
                raise RuntimeError(
                    f"panjang urutan {x.size(1)} melebihi max_len {self.pe.size(1)}"
                )
            return x + self.pe[:, : x.size(1), :]

    class TransformerClassifier(nn.Module):
        """Encoder-only Transformer dari awal, tanpa pretrained weights."""

        def __init__(
            self,
            vocab_size: int,
            num_classes: int,
            embed_dim: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            ff_dim: int = 256,
            max_len: int = 320,
            dropout: float = 0.1,
            pad_id: int = 0,
            pooling: str = "mean",
        ):
            super().__init__()
            if pooling not in ("mean", "cls"):
                raise ValueError(f"pooling tidak dikenal: {pooling!r}")
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim ({embed_dim}) harus habis dibagi num_heads ({num_heads})"
                )
            self.vocab_size = vocab_size
            self.num_classes = num_classes
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.ff_dim = ff_dim
            self.max_len = max_len
            self.dropout_p = dropout
            self.pad_id = pad_id
            self.pooling = pooling

            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
            self.pos_encoding = PositionalEncoding(embed_dim, max_len)
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # pre-norm lebih stabil pada data kecil
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(embed_dim, num_classes)

        def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
            mask = input_ids.eq(self.pad_id)
            assert (~mask).any(dim=1).all(), (
                "ada baris yang seluruhnya PAD; encoder akan menghasilkan NaN"
            )

            x = self.embedding(input_ids) * math.sqrt(self.embed_dim)
            x = self.pos_encoding(x)
            x = self.encoder(x, src_key_padding_mask=mask)

            if self.pooling == "cls":
                pooled = x[:, 0, :]
            else:
                valid = (~mask).float().unsqueeze(-1)
                pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

            return self.classifier(self.dropout(pooled))

        def config_dict(self) -> dict:
            return {
                "vocab_size": self.vocab_size,
                "num_classes": self.num_classes,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "ff_dim": self.ff_dim,
                "max_len": self.max_len,
                "dropout": self.dropout_p,
                "pad_id": self.pad_id,
                "pooling": self.pooling,
            }

else:  # pragma: no cover

    class BiLSTMClassifier:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("BiLSTMClassifier memerlukan PyTorch")

    class TransformerClassifier:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("TransformerClassifier memerlukan PyTorch")

    class PositionalEncoding:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("PositionalEncoding memerlukan PyTorch")
