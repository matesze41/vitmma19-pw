"""
Model evaluation script for flag pattern classification.

This script:
1. Loads the best trained model
2. Evaluates on validation and test sets
3. Compares CNN model with baseline heuristic model
4. Generates confusion matrices, ROC curves, and PR curves
5. Saves all figures and metrics

Usage:
    python 03-evaluation.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve,
    auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

import config
from utils import setup_logger, ensure_dir
import sys
sys.path.append(os.path.dirname(__file__))
from baseline_model import predict_from_segments_csv


# Import model and dataset from training script
import importlib.util
spec = importlib.util.spec_from_file_location("training", os.path.join(os.path.dirname(__file__), "02-training.py"))
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)

SegmentDataset = training_module.SegmentDataset
FlagPatternClassifier = training_module.FlagPatternClassifier


def get_predictions(loader, model, device, logger):
    """
    Get model predictions for a data loader.
    
    Args:
        loader: DataLoader instance
        model: Trained model
        device: torch device
        logger: Logger instance
        
    Returns:
        Tuple of (predictions, targets, probabilities)
    """
    all_preds = []
    all_probs = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(yb.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_probs)


def plot_confusion_matrices(cm_list, labels_list, titles, filename, logger):
    """
    Plot multiple confusion matrices side by side.
    
    Args:
        cm_list: List of confusion matrices
        labels_list: List of label arrays
        titles: List of titles
        filename: Output filename
        logger: Logger instance
    """
    n = len(cm_list)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 6))
    
    if n == 1:
        axes = [axes]
    
    for i, (cm, labels, title) in enumerate(zip(cm_list, labels_list, titles)):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=axes[i], cmap="Blues" if "CNN" in title else "Oranges", 
                 xticks_rotation=45, colorbar=True)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix: {filename}")


def plot_roc_pr_curves(y_true_bin, y_probs, label_values, pr_auc_per_class, filename, logger):
    """
    Plot ROC and PR curves for each class.
    
    Args:
        y_true_bin: Binarized true labels
        y_probs: Predicted probabilities
        label_values: Label names
        pr_auc_per_class: PR-AUC values per class
        filename: Output filename
        logger: Logger instance
    """
    num_classes = len(label_values)
    fig, axes = plt.subplots(2, num_classes, figsize=(3*num_classes, 6))
    
    for i in range(num_classes):
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc_class = auc(fpr, tpr)
        
        axes[0, i].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_class:.3f}')
        axes[0, i].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.5)
        axes[0, i].set_xlim([0.0, 1.0])
        axes[0, i].set_ylim([0.0, 1.05])
        axes[0, i].set_xlabel('FPR', fontsize=9)
        axes[0, i].set_ylabel('TPR', fontsize=9)
        axes[0, i].set_title(f'ROC: {label_values[i]}', fontsize=10, fontweight='bold')
        axes[0, i].legend(loc="lower right", fontsize=8)
        axes[0, i].grid(alpha=0.3)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        pr_auc_class = pr_auc_per_class[i]
        
        axes[1, i].plot(recall, precision, color='green', lw=2, label=f'AP = {pr_auc_class:.3f}')
        axes[1, i].set_xlim([0.0, 1.0])
        axes[1, i].set_ylim([0.0, 1.05])
        axes[1, i].set_xlabel('Recall', fontsize=9)
        axes[1, i].set_ylabel('Precision', fontsize=9)
        axes[1, i].set_title(f'PR: {label_values[i]}', fontsize=10, fontweight='bold')
        axes[1, i].legend(loc="lower left", fontsize=8)
        axes[1, i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC/PR curves: {filename}")


def plot_training_history(trainer_log_dir, filename, logger):
    """
    Plot training and validation loss/metrics over epochs.
    
    Args:
        trainer_log_dir: Directory with PyTorch Lightning logs
        filename: Output filename
        logger: Logger instance
    """
    try:
        metrics_df = pd.read_csv(f"{trainer_log_dir}/metrics.csv")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss curves
        train_loss = metrics_df.dropna(subset=['train_loss'])
        val_loss = metrics_df.dropna(subset=['val_loss'])
        
        ax1.plot(train_loss['epoch'], train_loss['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(val_loss['epoch'], val_loss['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot metric curves
        metric_name = config.METRIC_CONFIG[config.PRIMARY_METRIC]['monitor']
        train_metric_name = metric_name.replace('val_', 'train_')
        train_metric = metrics_df.dropna(subset=[train_metric_name])
        val_metric = metrics_df.dropna(subset=[metric_name])
        
        ax2.plot(train_metric['epoch'], train_metric[train_metric_name], 'b-',
                label=f'Training {config.METRIC_CONFIG[config.PRIMARY_METRIC]["name"]}', linewidth=2)
        ax2.plot(val_metric['epoch'], val_metric[metric_name], 'r-',
                label=f'Validation {config.METRIC_CONFIG[config.PRIMARY_METRIC]["name"]}', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel(config.METRIC_CONFIG[config.PRIMARY_METRIC]['name'], fontsize=12)
        ax2.set_title(f'Training and Validation {config.METRIC_CONFIG[config.PRIMARY_METRIC]["name"]}',
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training history: {filename}")
    except Exception as e:
        logger.warning(f"Could not plot training history: {e}")


def main():
    """Main evaluation pipeline."""
    # Setup logger
    logger = setup_logger(__name__, config.LOG_FILE)
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL EVALUATION PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Create output directory
    ensure_dir(config.OUTPUT_DIR_03)
    figures_dir = os.path.join(config.OUTPUT_DIR_03, "figures")
    ensure_dir(figures_dir)
    
    # Load training metadata
    metadata_path = os.path.join(config.EXPORT_DIR, 'training_metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    best_model_path = metadata['best_model_path']
    num_classes = metadata['num_classes']
    label_to_idx = metadata['label_to_idx']
    idx_to_label = metadata['idx_to_label']
    feature_cols = metadata['feature_cols']
    input_dim = metadata['input_dim']
    
    logger.info(f"Loaded training metadata from: {metadata_path}")
    logger.info(f"Best model path: {best_model_path}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Load preprocessed data for validation evaluation
    logger.info("\n" + "=" * 80)
    logger.info("LOADING VALIDATION DATA")
    logger.info("=" * 80)
    
    df = pd.read_csv(config.SEGMENTS_PREPROC_CSV)
    df = df.sort_values(["segment_id", "seq_pos"], kind="mergesort").reset_index(drop=True)
    
    # Build segments
    segments = []
    labels = []
    
    for seg_id, g in df.groupby("segment_id", sort=True):
        g = g.sort_values("seq_pos", kind="mergesort")
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        
        if feat.shape[0] < config.SEQUENCE_LENGTH:
            pad = np.repeat(feat[-1:, :], config.SEQUENCE_LENGTH - feat.shape[0], axis=0)
            feat = np.concatenate([feat, pad], axis=0)
        elif feat.shape[0] > config.SEQUENCE_LENGTH:
            feat = feat[:config.SEQUENCE_LENGTH, :]
        
        segments.append(feat)
        labels.append(g["label"].iloc[0])
    
    X = np.stack(segments, axis=0)
    y = np.array(labels)
    y_idx = np.vectorize(label_to_idx.get)(y)
    
    label_values = np.array([idx_to_label[i] for i in range(num_classes)])
    
    logger.info(f"Loaded {X.shape[0]} segments for validation")
    
    # Create full dataset and loader
    full_ds = SegmentDataset(X, y_idx)
    full_loader = DataLoader(full_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Load best model
    logger.info("\n" + "=" * 80)
    logger.info("LOADING BEST MODEL")
    logger.info("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Calculate class weights (same as training)
    class_counts = np.bincount(y_idx)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    best_model = FlagPatternClassifier.load_from_checkpoint(
        best_model_path,
        input_dim=input_dim,
        num_classes=num_classes,
        class_weights=class_weights,
        map_location=device
    )
    best_model.eval()
    best_model.freeze()
    best_model = best_model.to(device)
    
    logger.info(f"Model loaded from: {best_model_path}")
    
    # Get predictions on full training data
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING ON TRAINING DATA")
    logger.info("=" * 80)
    
    cnn_preds, cnn_targets, cnn_probs = get_predictions(full_loader, best_model, device, logger)
    
    # Calculate CNN metrics
    cnn_acc = accuracy_score(cnn_targets, cnn_preds)
    cnn_f1 = f1_score(cnn_targets, cnn_preds, average='macro')
    cnn_auc_ovo = roc_auc_score(cnn_targets, cnn_probs, multi_class='ovo', average='macro')
    cnn_auc_ovr = roc_auc_score(cnn_targets, cnn_probs, multi_class='ovr', average='macro')
    
    cnn_targets_bin = label_binarize(cnn_targets, classes=range(num_classes))
    cnn_pr_auc_per_class = [average_precision_score(cnn_targets_bin[:, i], cnn_probs[:, i])
                            for i in range(num_classes)]
    cnn_pr_auc = np.mean(cnn_pr_auc_per_class)
    
    logger.info("CNN Model (Training Set):")
    logger.info(f"  Accuracy:         {cnn_acc:.4f}")
    logger.info(f"  F1 Score (macro): {cnn_f1:.4f}")
    logger.info(f"  PR-AUC (macro):   {cnn_pr_auc:.4f}")
    logger.info(f"  AUC-ROC (OvO):    {cnn_auc_ovo:.4f}")
    logger.info(f"  AUC-ROC (OvR):    {cnn_auc_ovr:.4f}")
    
    # Map to label strings
    cnn_true_labels = np.vectorize(idx_to_label.get)(cnn_targets)
    cnn_pred_labels = np.vectorize(idx_to_label.get)(cnn_preds)
    
    # Confusion matrix for CNN on training data
    cm_cnn = confusion_matrix(cnn_true_labels, cnn_pred_labels, labels=label_values)
    
    # Evaluate baseline model on training data
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING BASELINE MODEL ON TRAINING DATA")
    logger.info("=" * 80)
    
    baseline_results = predict_from_segments_csv(config.SEGMENTS_VALUES_CSV,
                                                 slope_threshold=config.BASELINE_SLOPE_THRESHOLD)
    mask = baseline_results["gold_label"].notna()
    baseline_labels = baseline_results.loc[mask, "gold_label"].values
    baseline_preds = baseline_results.loc[mask, "predicted_label"].values
    
    baseline_acc = accuracy_score(baseline_labels, baseline_preds)
    baseline_f1 = f1_score(baseline_labels, baseline_preds, average='macro')
    
    logger.info("Baseline Model (Training Set):")
    logger.info(f"  Accuracy:         {baseline_acc:.4f}")
    logger.info(f"  F1 Score (macro): {baseline_f1:.4f}")
    
    cm_baseline = confusion_matrix(baseline_labels, baseline_preds, labels=label_values)
    
    # Plot comparison on training data
    plot_confusion_matrices(
        [cm_cnn, cm_baseline],
        [label_values, label_values],
        [f"CNN Model (Training Set)\nPR-AUC: {cnn_pr_auc:.4f} | Acc: {cnn_acc:.4f}",
         f"Baseline Model (Training Set)\nAcc: {baseline_acc:.4f}"],
        os.path.join(figures_dir, "confusion_matrix_training_comparison.png"),
        logger
    )
    
    # Load and evaluate on TEST set
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 80)
    
    df_test = pd.read_csv(config.SEGMENTS_PREPROC_TEST_CSV)
    df_test = df_test.sort_values(["segment_id", "seq_pos"], kind="mergesort").reset_index(drop=True)
    
    test_segments = []
    test_labels = []
    
    for seg_id, g in df_test.groupby("segment_id", sort=True):
        g = g.sort_values("seq_pos", kind="mergesort")
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        
        if feat.shape[0] < config.SEQUENCE_LENGTH:
            pad = np.repeat(feat[-1:, :], config.SEQUENCE_LENGTH - feat.shape[0], axis=0)
            feat = np.concatenate([feat, pad], axis=0)
        elif feat.shape[0] > config.SEQUENCE_LENGTH:
            feat = feat[:config.SEQUENCE_LENGTH, :]
        
        test_segments.append(feat)
        test_labels.append(g["label"].iloc[0])
    
    X_test = np.stack(test_segments, axis=0)
    y_test_str = np.array(test_labels)
    y_test = np.vectorize(label_to_idx.get)(y_test_str)
    
    logger.info(f"Test set loaded: {X_test.shape[0]} segments")
    logger.info("Test label distribution:")
    test_label_dist = pd.Series(y_test_str).value_counts()
    for label, count in test_label_dist.items():
        logger.info(f"  {label}: {count} samples ({count/len(y_test_str)*100:.1f}%)")
    
    # Create test dataset and loader
    test_ds = SegmentDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Get CNN predictions on test set
    test_preds, test_targets, test_probs = get_predictions(test_loader, best_model, device, logger)
    
    # Calculate CNN metrics on test set
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    test_auc_ovo = roc_auc_score(test_targets, test_probs, multi_class='ovo', average='macro')
    test_auc_ovr = roc_auc_score(test_targets, test_probs, multi_class='ovr', average='macro')
    
    test_targets_bin = label_binarize(test_targets, classes=range(num_classes))
    test_pr_auc_per_class = [average_precision_score(test_targets_bin[:, i], test_probs[:, i])
                             for i in range(num_classes)]
    test_pr_auc = np.mean(test_pr_auc_per_class)
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION - Test Set Results (CNN Model)")
    logger.info("=" * 80)
    logger.info(f"Accuracy:         {test_acc:.4f}")
    logger.info(f"F1 Score (macro): {test_f1:.4f}")
    logger.info(f"PR-AUC (macro):   {test_pr_auc:.4f} {'← PRIMARY METRIC' if config.PRIMARY_METRIC == 'pr_auc' else ''}")
    logger.info(f"AUC-ROC (OvO):    {test_auc_ovo:.4f}")
    logger.info(f"AUC-ROC (OvR):    {test_auc_ovr:.4f}")
    
    # Map to label strings
    test_true_labels = np.vectorize(idx_to_label.get)(test_targets)
    test_pred_labels = np.vectorize(idx_to_label.get)(test_preds)
    
    # Log confusion matrix
    cm_test = confusion_matrix(test_true_labels, test_pred_labels, labels=label_values)
    logger.info("\nConfusion Matrix (Test Set):")
    logger.info(f"Classes: {list(label_values)}")
    for i, row in enumerate(cm_test):
        logger.info(f"  {label_values[i]}: {list(row)}")
    
    # Log classification report
    logger.info("\nDetailed Classification Report (Test Set):")
    report = classification_report(test_true_labels, test_pred_labels,
                                   target_names=[str(lbl) for lbl in label_values])
    for line in report.split('\n'):
        if line.strip():
            logger.info(f"  {line}")
    
    # Evaluate baseline on test set
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING BASELINE MODEL ON TEST SET")
    logger.info("=" * 80)
    
    baseline_test_results = predict_from_segments_csv(config.SEGMENTS_TEST_RAW_CSV,
                                                      slope_threshold=config.BASELINE_SLOPE_THRESHOLD)
    mask_test = baseline_test_results["gold_label"].notna()
    baseline_test_labels = baseline_test_results.loc[mask_test, "gold_label"].values
    baseline_test_preds = baseline_test_results.loc[mask_test, "predicted_label"].values
    
    baseline_test_acc = accuracy_score(baseline_test_labels, baseline_test_preds)
    baseline_test_f1 = f1_score(baseline_test_labels, baseline_test_preds, average='macro')
    
    logger.info("Baseline Model (Test Set):")
    logger.info(f"  Accuracy:         {baseline_test_acc:.4f}")
    logger.info(f"  F1 Score (macro): {baseline_test_f1:.4f}")
    
    cm_baseline_test = confusion_matrix(baseline_test_labels, baseline_test_preds, labels=label_values)
    
    # Final comparison
    logger.info("\n" + "=" * 80)
    logger.info("FINAL MODEL COMPARISON - Test Set Performance")
    logger.info("=" * 80)
    logger.info(f"\nCNN Model (Deep Learning):")
    logger.info(f"  Accuracy:         {test_acc:.4f}")
    logger.info(f"  F1 Score (macro): {test_f1:.4f}")
    logger.info(f"  PR-AUC (macro):   {test_pr_auc:.4f}")
    logger.info(f"  AUC-ROC (OvO):    {test_auc_ovo:.4f}")
    logger.info(f"  AUC-ROC (OvR):    {test_auc_ovr:.4f}")
    
    logger.info(f"\nBaseline Model (Heuristic):")
    logger.info(f"  Accuracy:         {baseline_test_acc:.4f}")
    logger.info(f"  F1 Score (macro): {baseline_test_f1:.4f}")
    
    acc_improvement = test_acc - baseline_test_acc
    f1_improvement = test_f1 - baseline_test_f1
    acc_pct = (test_acc / baseline_test_acc - 1) * 100 if baseline_test_acc > 0 else 0
    f1_pct = (test_f1 / baseline_test_f1 - 1) * 100 if baseline_test_f1 > 0 else 0
    
    logger.info(f"\nImprovement (CNN vs Baseline):")
    logger.info(f"  Accuracy:  {acc_improvement:+.4f} ({acc_pct:+.1f}%)")
    logger.info(f"  F1 Score:  {f1_improvement:+.4f} ({f1_pct:+.1f}%)")
    
    # Generate all visualizations
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    # Test set confusion matrices comparison
    plot_confusion_matrices(
        [cm_test, cm_baseline_test],
        [label_values, label_values],
        [f"CNN Model - Test Set\nPR-AUC: {test_pr_auc:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}",
         f"Baseline Model - Test Set\nAcc: {baseline_test_acc:.4f} | F1: {baseline_test_f1:.4f}"],
        os.path.join(figures_dir, "confusion_matrix_test_comparison.png"),
        logger
    )
    
    # Individual test confusion matrix for CNN
    plot_confusion_matrices(
        [cm_test],
        [label_values],
        [f"CNN Model - Test Set\nPR-AUC: {test_pr_auc:.4f} | AUC-OvO: {test_auc_ovo:.4f} | Accuracy: {test_acc:.4f}"],
        os.path.join(figures_dir, "confusion_matrix_cnn_test.png"),
        logger
    )
    
    # ROC and PR curves for test set
    plot_roc_pr_curves(
        test_targets_bin,
        test_probs,
        label_values,
        test_pr_auc_per_class,
        os.path.join(figures_dir, "roc_pr_curves_test.png"),
        logger
    )
    
    # Plot training history if available
    lightning_logs_dir = os.path.join("lightning_logs", "version_0")
    if os.path.exists(lightning_logs_dir):
        plot_training_history(
            lightning_logs_dir,
            os.path.join(figures_dir, "training_history.png"),
            logger
        )
    
    # Save metrics to file
    metrics_file = os.path.join(config.OUTPUT_DIR_03, "evaluation_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION METRICS SUMMARY\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"CNN Model:\n")
        f.write(f"  Accuracy:         {test_acc:.4f}\n")
        f.write(f"  F1 Score (macro): {test_f1:.4f}\n")
        f.write(f"  PR-AUC (macro):   {test_pr_auc:.4f}\n")
        f.write(f"  AUC-ROC (OvO):    {test_auc_ovo:.4f}\n")
        f.write(f"  AUC-ROC (OvR):    {test_auc_ovr:.4f}\n\n")
        
        f.write(f"Baseline Model:\n")
        f.write(f"  Accuracy:         {baseline_test_acc:.4f}\n")
        f.write(f"  F1 Score (macro): {baseline_test_f1:.4f}\n\n")
        
        f.write(f"Improvement (CNN vs Baseline):\n")
        f.write(f"  Accuracy:  {acc_improvement:+.4f} ({acc_pct:+.1f}%)\n")
        f.write(f"  F1 Score:  {f1_improvement:+.4f} ({f1_pct:+.1f}%)\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("\nCLASSIFICATION REPORT (CNN - Test Set)\n")
        f.write("-" * 80 + "\n")
        f.write(report + "\n")
    
    logger.info(f"Saved evaluation metrics: {metrics_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"All figures saved to: {figures_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

