"""
Dataset and DataLoader for fake news detection.
"""
import torch
from torch.utils.data import Dataset, DataLoader

from .data_processing import encode_text


class FakeNewsDataset(Dataset):
    """Dataset for fake news classification."""

    def __init__(self, df, vocab, max_len: int = 220):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.raw_labels = df["label_raw"].tolist()
        self.vocab = vocab
        self.max_len = max_len
        self.encoded = [encode_text(t, vocab, max_len) for t in self.texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq, length = self.encoded[idx]
        label = self.labels[idx]
        text = self.texts[idx]
        raw_label = self.raw_labels[idx]

        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
            text,
            raw_label
        )


def collate_fn(batch):
    """Collate function for DataLoader."""
    seqs = torch.stack([x[0] for x in batch], dim=0)
    lengths = torch.stack([x[1] for x in batch], dim=0)
    labels = torch.stack([x[2] for x in batch], dim=0)
    texts = [x[3] for x in batch]
    raw_labels = [x[4] for x in batch]
    return seqs, lengths, labels, texts, raw_labels


def make_loaders(train_df, val_df, test_df, vocab, cfg, batch_size=None):
    """Create train, validation, and test DataLoaders."""
    batch_size = batch_size or cfg.batch_size

    train_ds = FakeNewsDataset(train_df, vocab, cfg.max_len)
    val_ds = FakeNewsDataset(val_df, vocab, cfg.max_len)
    test_ds = FakeNewsDataset(test_df, vocab, cfg.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
