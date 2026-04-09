"""
Model architectures for fake news detection.
"""
import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Attention pooling layer for sequence models."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, outputs, mask):
        energy = torch.tanh(self.proj(outputs))
        scores = self.score(energy).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    """Bidirectional LSTM with attention for text classification."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = AttentionPooling(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, lengths):
        emb = self.embedding(x)
        emb = self.embedding_dropout(emb)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=x.size(1)
        )

        mask = x != self.pad_idx
        context, _ = self.attention(outputs, mask)
        context = self.norm(context)

        logits = self.classifier(context).squeeze(-1)
        return logits


class TextCNNClassifier(nn.Module):
    """Text CNN model for text classification."""

    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int = 128,
                 kernel_sizes=(3, 4, 5), dropout: float = 0.35, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, 1)
        )

    def forward(self, x, lengths=None):
        emb = self.embedding(x).transpose(1, 2)

        feats = []
        for conv in self.convs:
            h = torch.relu(conv(emb))
            p = torch.max(h, dim=2).values
            feats.append(p)

        feat = torch.cat(feats, dim=1)
        feat = self.dropout(feat)
        logits = self.fc(feat).squeeze(-1)
        return logits


def build_model(arch: str, vocab_size: int, cfg):
    """Build a model based on architecture name."""
    if arch.lower() == "bilstm":
        return BiLSTMAttentionClassifier(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_idx=0
        )

    if arch.lower() == "cnn":
        return TextCNNClassifier(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            num_filters=128,
            kernel_sizes=(3, 4, 5),
            dropout=cfg.dropout,
            pad_idx=0
        )

    raise ValueError(f"Unknown architecture: {arch}")
