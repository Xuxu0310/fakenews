"""
Main entry point for the fake news detection project.
"""
import os
import json
import warnings
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from src.config import cfg
from src.utils import set_seed, print_section, print_dataframe, print_dict_pretty, ensure_dir
from src.data_processing import load_data, split_data, build_vocab, tokenize
from src.models import build_model
from src.dataset import make_loaders
from src.train import train_model, evaluate
from src.visualize import (
    save_line_figure, save_sweep_figure, save_summary_bar,
    save_class_distribution, save_length_distribution,
    save_confusion_matrix, save_roc_pr_curves,
    save_prediction_table, save_text_samples
)

warnings.filterwarnings("ignore")


def create_prediction_dataframe(test_stats, id2label):
    """Create prediction dataframe from test statistics."""
    probs = test_stats["probs"]
    preds = test_stats["preds"]
    targets = test_stats["targets"]
    texts = test_stats["texts"]
    raw_labels = test_stats["raw_labels"]

    rows = []
    for text, raw_label, y_true, y_pred, prob in zip(texts, raw_labels, targets, preds, probs):
        rows.append({
            "text": text,
            "true_label": raw_label,
            "pred_label": id2label[int(y_pred)],
            "prob_positive": float(prob),
            "correct": bool(int(y_true) == int(y_pred))
        })
    return pd.DataFrame(rows)


def print_metrics(name, stats):
    """Print evaluation metrics."""
    print_section(name)
    print(f"Loss: {stats['loss']:.4f}")
    print(f"Accuracy: {stats['acc']:.4f}")
    print(f"F1: {stats['f1']:.4f}")
    print(f"Precision: {stats['precision']:.4f}")
    print(f"Recall: {stats['recall']:.4f}")
    if stats["roc_auc"] is not None:
        print(f"ROC AUC: {stats['roc_auc']:.4f}")
    if stats["pr_auc"] is not None:
        print(f"PR AUC: {stats['pr_auc']:.4f}")


def main():
    """Main execution function."""
    # Setup
    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    print_section("Environment")
    print(f"Device: {cfg.device}")

    # Load and prepare data
    df, label2id, id2label = load_data(cfg.csv_path)
    train_df, val_df, test_df = split_data(df, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed)
    vocab = build_vocab(train_df["text"].tolist(), max_vocab_size=cfg.max_vocab_size, min_freq=2)

    print_section("Dataset Information")
    print(f"Total samples: {len(df)}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Label mapping: {label2id}")

    class_counts = df["label_raw"].value_counts().sort_index()
    length_series = df["text"].apply(lambda x: len(tokenize(x)))

    print_section("Class Distribution")
    print(class_counts.to_string())

    print_section("Text Length Statistics")
    print(length_series.describe().to_string())

    # Save initial visualizations
    save_class_distribution(df, os.path.join(cfg.output_dir, "class_distribution.png"))
    save_length_distribution(df, os.path.join(cfg.output_dir, "text_length_distribution.png"), tokenize)

    sample_df = save_text_samples(df, os.path.join(cfg.output_dir, "sample_texts_by_class.csv"))
    print_dataframe(sample_df, "Sample Texts by Class", max_rows=20)

    # ==============================
    # Baseline model
    # ==============================
    baseline_model, baseline_history, baseline_test_stats, baseline_loaders = train_model(
        arch="bilstm",
        loss_name="bce",
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        vocab=vocab,
        cfg=cfg,
        epochs=cfg.epochs
    )

    baseline_history_df = pd.DataFrame(baseline_history)
    baseline_history_path = os.path.join(cfg.output_dir, "baseline_history.csv")
    baseline_history_df.to_csv(baseline_history_path, index=False)
    print_dataframe(baseline_history_df, "Baseline Training History", max_rows=50)

    save_line_figure(
        baseline_history_df,
        "BiLSTM-Attention Baseline Training Curves",
        os.path.join(cfg.output_dir, "baseline_training_curves.png")
    )
    print_metrics("Baseline Model", baseline_test_stats)

    baseline_pred_df = create_prediction_dataframe(baseline_test_stats, id2label)
    save_prediction_table(
        baseline_pred_df,
        os.path.join(cfg.output_dir, "baseline_first_100_predictions.png"),
        os.path.join(cfg.output_dir, "baseline_predictions.csv")
    )
    print_dataframe(baseline_pred_df, "Baseline Predictions Preview", max_rows=20)

    save_confusion_matrix(
        baseline_test_stats["targets"],
        baseline_test_stats["preds"],
        [id2label[i] for i in sorted(id2label.keys())],
        os.path.join(cfg.output_dir, "baseline_confusion_matrix.png")
    )
    save_roc_pr_curves(
        baseline_test_stats["targets"],
        baseline_test_stats["probs"],
        os.path.join(cfg.output_dir, "baseline_roc_pr_curve.png")
    )

    print_section("Baseline Confusion Matrix")
    print(confusion_matrix(baseline_test_stats["targets"], baseline_test_stats["preds"]).astype(int))

    print_section("Baseline Classification Report")
    print(classification_report(
        baseline_test_stats["targets"],
        baseline_test_stats["preds"],
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        zero_division=0
    ))

    # ==============================
    # Loss function comparison
    # ==============================
    loss_histories = []
    loss_labels = []
    loss_stats_records = []

    for loss_name in ["bce", "focal"]:
        _, hist, stats, _ = train_model(
            arch="bilstm",
            loss_name=loss_name,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            vocab=vocab,
            cfg=cfg,
            epochs=cfg.epochs
        )
        hist_df = pd.DataFrame(hist)
        hist_df.to_csv(os.path.join(cfg.output_dir, f"history_loss_{loss_name}.csv"), index=False)
        loss_histories.append(hist_df)
        loss_labels.append(loss_name.upper())
        loss_stats_records.append({
            "loss_name": loss_name.upper(),
            "best_test_acc": float(hist_df["test_acc"].max()),
            "final_test_acc": float(hist_df["test_acc"].iloc[-1]),
            "final_test_f1": float(hist_df["test_f1"].iloc[-1]),
            "final_test_loss": float(hist_df["test_loss"].iloc[-1]),
        })
        print_dataframe(hist_df, f"Loss Comparison History - {loss_name.upper()}", max_rows=50)
        print_metrics(f"Loss Comparison - {loss_name.upper()}", stats)

    print_dataframe(pd.DataFrame(loss_stats_records), "Loss Comparison Summary Table", max_rows=20)

    save_sweep_figure(
        loss_histories,
        loss_labels,
        "Loss Function Comparison",
        os.path.join(cfg.output_dir, "loss_function_comparison.png")
    )
    save_summary_bar(
        loss_histories,
        loss_labels,
        "Loss Function Comparison Summary",
        os.path.join(cfg.output_dir, "loss_function_comparison_summary.png")
    )

    # ==============================
    # Learning rate comparison
    # ==============================
    lr_values = [1e-4, 1e-3, 1e-2, 1e-1]
    lr_histories = []
    lr_labels = []
    lr_stats_records = []

    for lr in lr_values:
        _, hist, stats, _ = train_model(
            arch="bilstm",
            loss_name="bce",
            lr=lr,
            batch_size=cfg.batch_size,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            vocab=vocab,
            cfg=cfg,
            epochs=cfg.epochs
        )
        hist_df = pd.DataFrame(hist)
        hist_df.to_csv(os.path.join(cfg.output_dir, f"history_lr_{lr}.csv"), index=False)
        lr_histories.append(hist_df)
        lr_labels.append(f"lr={lr:g}")
        lr_stats_records.append({
            "learning_rate": lr,
            "best_test_acc": float(hist_df["test_acc"].max()),
            "final_test_acc": float(hist_df["test_acc"].iloc[-1]),
            "final_test_f1": float(hist_df["test_f1"].iloc[-1]),
            "final_test_loss": float(hist_df["test_loss"].iloc[-1]),
        })
        print_dataframe(hist_df, f"Learning Rate History - lr={lr:g}", max_rows=50)
        print_metrics(f"Learning Rate Comparison - {lr:g}", stats)

    print_dataframe(pd.DataFrame(lr_stats_records), "Learning Rate Comparison Summary Table", max_rows=20)

    save_sweep_figure(
        lr_histories,
        lr_labels,
        "Learning Rate Comparison",
        os.path.join(cfg.output_dir, "learning_rate_comparison.png")
    )
    save_summary_bar(
        lr_histories,
        lr_labels,
        "Learning Rate Comparison Summary",
        os.path.join(cfg.output_dir, "learning_rate_comparison_summary.png")
    )

    # ==============================
    # Batch size comparison
    # ==============================
    batch_sizes = [8, 16, 32, 64, 128]
    batch_histories = []
    batch_labels = []
    batch_stats_records = []

    for bs in batch_sizes:
        _, hist, stats, _ = train_model(
            arch="bilstm",
            loss_name="bce",
            lr=cfg.lr,
            batch_size=bs,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            vocab=vocab,
            cfg=cfg,
            epochs=cfg.epochs
        )
        hist_df = pd.DataFrame(hist)
        hist_df.to_csv(os.path.join(cfg.output_dir, f"history_batch_{bs}.csv"), index=False)
        batch_histories.append(hist_df)
        batch_labels.append(f"batch={bs}")
        batch_stats_records.append({
            "batch_size": bs,
            "best_test_acc": float(hist_df["test_acc"].max()),
            "final_test_acc": float(hist_df["test_acc"].iloc[-1]),
            "final_test_f1": float(hist_df["test_f1"].iloc[-1]),
            "final_test_loss": float(hist_df["test_loss"].iloc[-1]),
        })
        print_dataframe(hist_df, f"Batch Size History - batch={bs}", max_rows=50)
        print_metrics(f"Batch Size Comparison - {bs}", stats)

    print_dataframe(pd.DataFrame(batch_stats_records), "Batch Size Comparison Summary Table", max_rows=20)

    save_sweep_figure(
        batch_histories,
        batch_labels,
        "Batch Size Comparison",
        os.path.join(cfg.output_dir, "batch_size_comparison.png")
    )
    save_summary_bar(
        batch_histories,
        batch_labels,
        "Batch Size Comparison Summary",
        os.path.join(cfg.output_dir, "batch_size_comparison_summary.png")
    )

    # ==============================
    # Architecture comparison
    # ==============================
    arch_histories = []
    arch_labels = []
    arch_stats_records = []

    for arch in ["bilstm", "cnn"]:
        _, hist, stats, _ = train_model(
            arch=arch,
            loss_name="bce",
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            vocab=vocab,
            cfg=cfg,
            epochs=cfg.epochs
        )
        hist_df = pd.DataFrame(hist)
        hist_df.to_csv(os.path.join(cfg.output_dir, f"history_arch_{arch}.csv"), index=False)
        arch_histories.append(hist_df)
        arch_labels.append(arch.upper())
        arch_stats_records.append({
            "architecture": arch.upper(),
            "best_test_acc": float(hist_df["test_acc"].max()),
            "final_test_acc": float(hist_df["test_acc"].iloc[-1]),
            "final_test_f1": float(hist_df["test_f1"].iloc[-1]),
            "final_test_loss": float(hist_df["test_loss"].iloc[-1]),
        })
        print_dataframe(hist_df, f"Architecture History - {arch.upper()}", max_rows=50)
        print_metrics(f"Architecture Comparison - {arch.upper()}", stats)

    print_dataframe(pd.DataFrame(arch_stats_records), "Architecture Comparison Summary Table", max_rows=20)

    save_sweep_figure(
        arch_histories,
        arch_labels,
        "Architecture Comparison",
        os.path.join(cfg.output_dir, "architecture_comparison.png")
    )
    save_summary_bar(
        arch_histories,
        arch_labels,
        "Architecture Comparison Summary",
        os.path.join(cfg.output_dir, "architecture_comparison_summary.png")
    )

    # ==============================
    # Save final outputs
    # ==============================
    import torch

    summary = {
        "device": cfg.device,
        "samples": int(len(df)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "vocab_size": int(len(vocab)),
        "label_mapping": label2id,
        "baseline_test_accuracy": float(baseline_test_stats["acc"]),
        "baseline_test_f1": float(baseline_test_stats["f1"]),
        "baseline_test_precision": float(baseline_test_stats["precision"]),
        "baseline_test_recall": float(baseline_test_stats["recall"]),
        "baseline_test_roc_auc": None if baseline_test_stats["roc_auc"] is None else float(baseline_test_stats["roc_auc"]),
        "baseline_test_pr_auc": None if baseline_test_stats["pr_auc"] is None else float(baseline_test_stats["pr_auc"])
    }

    summary_path = os.path.join(cfg.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_dict_pretty(summary, "Final Summary")

    torch.save(baseline_model.state_dict(), os.path.join(cfg.output_dir, "baseline_bilstm_model.pt"))

    print_section("Output Files")
    print("All files have been saved to:", cfg.output_dir)
    print("Finished successfully.")


if __name__ == "__main__":
    main()
