"""
Loss functions for binary classification.
"""
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probas = torch.sigmoid(logits)
        pt = targets * probas + (1 - targets) * (1 - probas)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = alpha_t * (1 - pt).pow(self.gamma) * bce_loss

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def get_criterion(name: str):
    """Get loss function by name."""
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "focal":
        return FocalLoss(alpha=0.75, gamma=2.0)
    raise ValueError(f"Unknown loss function: {name}")
