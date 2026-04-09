"""
Configuration module for the fake news detection project.
"""
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for all hyperparameters and settings."""

    # Data paths
    csv_path: str = "fakenews.csv"
    output_dir: str = "outputs"

    # Random seed
    seed: int = 42

    # Data split
    test_size: float = 0.15
    val_size: float = 0.15

    # Vocabulary and text processing
    max_vocab_size: int = 30000
    max_len: int = 220

    # Model architecture
    embed_dim: int = 200
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.35

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 6
    patience: int = 3
    batch_size: int = 32

    # DataLoader
    num_workers: int = 0

    # Device
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"


# Default configuration instance
cfg = Config()
