# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.

import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

from utils import setup_logger
import baseline_model
from baseline_model import predict_from_segments_csv
from train_model import FlagPatternClassifier  # reuse model definition

logger = setup_logger()

BASE_DATA_DIR = os.path.abspath("../data")
EXPORT_DIR = os.path.join(BASE_DATA_DIR, "export")

def evaluate():
    logger.info("Evaluating model...")

    # ------------------------------------------------------------------
    # Load metadata and checkpoint
    # ------------------------------------------------------------------
    metadata_path = os.path.join(EXPORT_DIR, "eval_metadata.pt")
    meta = torch.load(metadata_path, weights_only=False)

    checkpoint_path = meta["checkpoint_path"]
    num_classes = meta["num_classes"]
    input_dim = meta["input_dim"]
    class_weights = np.array(meta["class_weights"])
    label_to_idx = meta["label_to_idx"]
    idx_to_label = meta["idx_to_label"]
    feature_cols = meta["feature_cols"]
    label_values = np.array(list(label_to_idx.keys()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlagPatternClassifier.load_from_checkpoint(
        checkpoint_path,
        input_dim=input_dim,
        num_classes=num_classes,
        class_weights=class_weights,
        weights_only=False
    ).to(device)

    model.eval()
    model.freeze()

    logger.info(f"Loaded model from: {checkpoint_path}")
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset helper
    # ------------------------------------------------------------------
    class SegmentDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y).long()

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    def get_predictions(loader):
        preds, probs, targets = [], [], []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                p = F.softmax(logits, dim=1).cpu().numpy()
                preds.append(np.argmax(p, axis=1))
                probs.append(p)
                targets.append(yb.numpy())

        return (
            np.concatenate(preds),
            np.concatenate(targets),
            np.concatenate(probs),
        )

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    TEST_PREPROC_CSV = os.path.join(EXPORT_DIR, "segments_preproc_24_test.csv")
    df_test = pd.read_csv(TEST_PREPROC_CSV)
    df_test = df_test.sort_values(["segment_id", "seq_pos"])

    segments, labels = [], []

    for _, g in df_test.groupby("segment_id"):
        feat = g[feature_cols].to_numpy(dtype=np.float32)

        if feat.shape[0] < 24:
            pad = np.repeat(feat[-1:], 24 - feat.shape[0], axis=0)
            feat = np.concatenate([feat, pad])
        else:
            feat = feat[:24]

        segments.append(feat)
        labels.append(g["label"].iloc[0])

    X_test = np.stack(segments)
    y_test = np.vectorize(label_to_idx.get)(np.array(labels))

    test_loader = DataLoader(
        SegmentDataset(X_test, y_test),
        batch_size=12,
        shuffle=False
    )

    # ------------------------------------------------------------------
    # CNN evaluation
    # ------------------------------------------------------------------
    preds, targets, probs = get_predictions(test_loader)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    auc_ovo = roc_auc_score(targets, probs, multi_class="ovo", average="macro")
    auc_ovr = roc_auc_score(targets, probs, multi_class="ovr", average="macro")

    targets_bin = label_binarize(targets, classes=range(num_classes))
    pr_auc = np.mean([
        average_precision_score(targets_bin[:, i], probs[:, i])
        for i in range(num_classes)
    ])

    logger.info("CNN TEST RESULTS")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1 (macro): {f1:.4f}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"AUC-OvO: {auc_ovo:.4f}")
    logger.info(f"AUC-OvR: {auc_ovr:.4f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_values)
    disp.plot(ax=ax, cmap="Greens", xticks_rotation=45)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Baseline evaluation
    # ------------------------------------------------------------------
    TEST_RAW_CSV = os.path.join(EXPORT_DIR, "segments_test_raw.csv")
    BASELINE_SLOPE_THRESHOLD = 0.0002

    baseline_results = predict_from_segments_csv(
        TEST_RAW_CSV,
        slope_threshold=BASELINE_SLOPE_THRESHOLD
    )

    mask = baseline_results["gold_label"].notna()
    base_y = baseline_results.loc[mask, "gold_label"].values
    base_p = baseline_results.loc[mask, "predicted_label"].values

    base_acc = accuracy_score(base_y, base_p)
    base_f1 = f1_score(base_y, base_p, average="macro")

    logger.info("BASELINE TEST RESULTS")
    logger.info(f"Accuracy: {base_acc:.4f}")
    logger.info(f"F1 (macro): {base_f1:.4f}")
    
    # Ensure consistent class order
    all_labels = label_values
    logger.info(f"Confusion matrix label order: {list(all_labels)}")

    # CNN confusion matrix (test set)
    cm_cnn = confusion_matrix(
        np.vectorize(idx_to_label.get)(targets),
        np.vectorize(idx_to_label.get)(preds),
        labels=all_labels
    )

    # Baseline confusion matrix (test set)
    cm_baseline = confusion_matrix(
        base_y,
        base_p,
        labels=all_labels
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # CNN confusion matrix
    disp_cnn = ConfusionMatrixDisplay(
        confusion_matrix=cm_cnn,
        display_labels=all_labels
    )
    disp_cnn.plot(
        ax=axes[0],
        cmap="Greens",
        xticks_rotation=45,
        colorbar=False
    )
    axes[0].set_title(
        f"CNN Model – Test Set\n"
        f"Acc={acc:.4f} | F1={f1:.4f} | PR-AUC={pr_auc:.4f}"
    )

    # Baseline confusion matrix
    disp_base = ConfusionMatrixDisplay(
        confusion_matrix=cm_baseline,
        display_labels=all_labels
    )
    disp_base.plot(
        ax=axes[1],
        cmap="Oranges",
        xticks_rotation=45,
        colorbar=False
    )
    axes[1].set_title(
        f"Baseline Model – Test Set\n"
        f"Acc={base_acc:.4f} | F1={base_f1:.4f}"
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
