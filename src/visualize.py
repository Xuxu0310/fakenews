"""
Visualization utilities for training results.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from sklearn.metrics import confusion_matrix


def show_and_save(fig, path):
    """Display and save a figure."""
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def save_line_figure(history_df, title, path):
    """Save training curves (loss, accuracy, F1)."""
    epochs = history_df["epoch"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle(title, fontsize=16)

    axes[0].plot(epochs, history_df["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history_df["val_loss"], marker="o", label="Validation Loss")
    axes[0].plot(epochs, history_df["test_loss"], marker="o", label="Test Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history_df["train_acc"], marker="o", label="Train Accuracy")
    axes[1].plot(epochs, history_df["val_acc"], marker="o", label="Validation Accuracy")
    axes[1].plot(epochs, history_df["test_acc"], marker="o", label="Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history_df["val_f1"], marker="o", label="Validation F1")
    axes[2].plot(epochs, history_df["test_f1"], marker="o", label="Test F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    show_and_save(fig, path)


def save_sweep_figure(histories, labels, title, path):
    """Save comparison figure for multiple experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(title, fontsize=16)

    for hist, label in zip(histories, labels):
        epochs = hist["epoch"].tolist()
        axes[0].plot(epochs, hist["train_loss"], marker="o", label=label)
        axes[1].plot(epochs, hist["train_acc"], marker="o", label=label)
        axes[2].plot(epochs, hist["test_acc"], marker="o", label=label)

    axes[0].set_title("Training Loss")
    axes[1].set_title("Training Accuracy")
    axes[2].set_title("Test Accuracy")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    axes[2].set_ylabel("Accuracy")

    show_and_save(fig, path)


def save_summary_bar(histories, labels, title, path):
    """Save bar chart comparing best and final accuracies."""
    best_test_acc = [max(h["test_acc"]) for h in histories]
    final_test_acc = [h["test_acc"].iloc[-1] for h in histories]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, best_test_acc, width, label="Best Test Accuracy")
    ax.bar(x + width / 2, final_test_acc, width, label="Final Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    show_and_save(fig, path)


def save_class_distribution(df, path):
    """Save class distribution bar chart."""
    counts = df["label_raw"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
    show_and_save(fig, path)


def save_length_distribution(df, path, tokenize_fn):
    """Save text length distribution histogram."""
    lengths = df["text"].apply(lambda x: len(tokenize_fn(x)))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lengths, bins=40)
    ax.set_title("Text Length Distribution")
    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    show_and_save(fig, path)


def save_confusion_matrix(y_true, y_pred, class_names, path):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax)
    show_and_save(fig, path)


def save_roc_pr_curves(y_true, y_prob, path):
    """Save ROC and Precision-Recall curves."""
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(fpr, tpr, label=f"ROC AUC = {auc_score:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(recall, precision, label=f"Average Precision = {ap_score:.4f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    show_and_save(fig, path)


def save_prediction_table(pred_df, path_png, path_csv):
    """Save prediction table as CSV and PNG."""
    pred_df.to_csv(path_csv, index=False)

    display_df = pred_df.head(100).copy()
    display_df["text"] = display_df["text"].apply(
        lambda x: textwrap.shorten(str(x).replace("\n", " "), width=60, placeholder="...")
    )
    display_df["prob_positive"] = display_df["prob_positive"].round(4)
    display_df["correct"] = display_df["correct"].map({True: "Yes", False: "No"})

    fig_height = max(12, 0.42 * len(display_df) + 2)
    fig, ax = plt.subplots(figsize=(28, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="left",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.25)

    show_and_save(fig, path_png)


def save_text_samples(df, path):
    """Save sample texts by class."""
    sample_rows = []
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label].head(3)
        for _, row in sub.iterrows():
            sample_rows.append({
                "label": row["label_raw"],
                "text": textwrap.shorten(row["text"], width=180, placeholder="...")
            })
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(path, index=False)
    return sample_df
