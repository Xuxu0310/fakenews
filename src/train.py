"""
Training functions for fake news detection models.
"""
import copy
import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from .models import build_model
from .loss import get_criterion
from .dataset import make_loaders


def batch_to_device(batch, device):
    """Move a batch to the specified device."""
    seqs, lengths, labels, texts, raw_labels = batch
    return (
        seqs.to(device),
        lengths.to(device),
        labels.to(device),
        texts,
        raw_labels
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        seqs, lengths, labels, texts, raw_labels = batch_to_device(batch, device)

        optimizer.zero_grad()
        logits = model(seqs, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        total_loss += loss.item() * seqs.size(0)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(labels.detach().cpu().numpy().astype(int).tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


def train_model(arch, loss_name, lr, batch_size, train_df, val_df, test_df,
                vocab, cfg, epochs=None, patience=None):
    """
    Train a model with early stopping.

    Returns:
        model, history_df, final_test_stats, loaders
    """
    epochs = epochs or cfg.epochs
    patience = patience or cfg.patience

    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, vocab, cfg, batch_size
    )

    model = build_model(arch, len(vocab), cfg).to(cfg.device)
    criterion = get_criterion(loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)

        val_stats = evaluate(model, val_loader, criterion, cfg.device)
        test_stats = evaluate(model, test_loader, criterion, cfg.device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "test_loss": test_stats["loss"],
            "test_acc": test_stats["acc"],
            "val_f1": val_stats["f1"],
            "test_f1": test_stats["f1"],
            "seconds": time.time() - start_time
        })

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    final_test_stats = evaluate(model, test_loader, criterion, cfg.device)

    return model, history, final_test_stats, (train_loader, val_loader, test_loader)


def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix, classification_report,
        roc_curve, precision_recall_curve, average_precision_score
    )

    model.eval()
    total_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []
    all_texts = []
    all_raw_labels = []

    with torch.no_grad():
        for batch in loader:
            seqs, lengths, labels, texts, raw_labels = batch_to_device(batch, device)
            logits = model(seqs, lengths)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            total_loss += loss.item() * seqs.size(0)
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(labels.detach().cpu().numpy().astype(int).tolist())
            all_texts.extend(texts)
            all_raw_labels.extend(raw_labels)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)

    auc = None
    pr_auc = None
    try:
        if len(set(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_probs)
            pr_auc = average_precision_score(all_targets, all_probs)
    except Exception:
        pass

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "probs": np.array(all_probs),
        "preds": np.array(all_preds),
        "targets": np.array(all_targets),
        "texts": all_texts,
        "raw_labels": all_raw_labels
    }
