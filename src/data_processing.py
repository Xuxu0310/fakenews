"""
Data loading and processing module.
"""
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Clean and normalize text."""
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Tokenize text into words."""
    return re.findall(r"[a-z0-9']+", text.lower())


def load_data(csv_path: str):
    """Load data from CSV file."""
    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("The CSV file must contain 'text' and 'label' columns.")

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["label_raw"] = df["label"].astype(str)

    label_values = sorted(df["label_raw"].unique().tolist())
    label2id = {lab: idx for idx, lab in enumerate(label_values)}
    id2label = {idx: lab for lab, idx in label2id.items()}

    df["label"] = df["label_raw"].map(label2id).astype(int)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    return df, label2id, id2label


def split_data(df, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    """Split data into train, validation, and test sets."""
    train_df, temp_df = train_test_split(
        df,
        test_size=(test_size + val_size),
        random_state=seed,
        stratify=df["label"]
    )

    relative_test = test_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df["label"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_vocab(texts, max_vocab_size: int = 30000, min_freq: int = 2):
    """Build vocabulary from texts."""
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_vocab_size:
            break

    return vocab


def encode_text(text, vocab, max_len: int = 220):
    """Encode text to token IDs."""
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens[:max_len]]
    length = len(ids)

    if length < max_len:
        ids += [vocab["<PAD>"]] * (max_len - length)

    return ids, min(length, max_len)


def encode_texts(texts, vocab, max_len: int = 220):
    """Encode multiple texts to token IDs."""
    encoded = [encode_text(text, vocab, max_len) for text in texts]
    seqs = [x[0] for x in encoded]
    lengths = [x[1] for x in encoded]
    return np.array(seqs, dtype=np.int64), np.array(lengths, dtype=np.int64)
