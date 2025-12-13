import config
from utils import setup_logger
import os
import sys
import math
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import baseline_model
from baseline_model import predict_from_segments_csv, evaluate_on_segments_csv
from sklearn.preprocessing import label_binarize

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve, 
    auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath('..'))
from utils import setup_logger

logger = setup_logger(__name__)

BASE_DATA_DIR = os.path.abspath("../data")
EXPORT_DIR = os.path.join(BASE_DATA_DIR, "export")
PREPROC_CSV = os.path.join(EXPORT_DIR, "segments_preproc_24.csv")
SEED = 1
# Training hyperparameters
MAX_EPOCHS = 50
BATCH_SIZE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PRIMARY_METRIC = 'pr_auc'

# Metric display names and whether higher is better
METRIC_CONFIG = {
    'auc_ovo': {
        'name': 'AUC-ROC (OvO)',
        'short': 'ovo',
        'higher_is_better': True,
        'monitor': 'val_auc_ovo',
        'description': 'One-vs-One: evaluates all pairwise class comparisons'
    },
    'auc_ovr': {
        'name': 'AUC-ROC (OvR)',
        'short': 'ovr',
        'higher_is_better': True,
        'monitor': 'val_auc_ovr',
        'description': 'One-vs-Rest: evaluates each class vs all others'
    },
    'f1': {
        'name': 'F1 Score (macro)',
        'short': 'f1',
        'higher_is_better': True,
        'monitor': 'val_f1',
        'description': 'Harmonic mean of precision and recall'
    },
    'accuracy': {
        'name': 'Accuracy',
        'short': 'acc',
        'higher_is_better': True,
        'monitor': 'val_accuracy',
        'description': 'Proportion of correct predictions'
    },
    'pr_auc': {
        'name': 'PR-AUC (macro)',
        'short': 'pr',
        'higher_is_better': True,
        'monitor': 'val_pr_auc',
        'description': 'Precision-Recall curve area (macro): better for imbalanced classes'
    }
}

class FlagPatternClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        class_weights: np.ndarray,
        hidden_channels: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 12
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture with 3 conv layers and max pooling
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, num_classes),
        )
        
        # Loss function with class weights
        self.class_weights = torch.FloatTensor(class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Store predictions for epoch-end metrics
        self.validation_step_outputs = []
        self.training_step_outputs = []
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.pool(h).squeeze(-1)
        logits = self.fc(h)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        self.training_step_outputs.append({
            'loss': loss,
            'preds': preds.detach().cpu(),
            'probs': probs.detach().cpu(),
            'targets': y.detach().cpu()
        })
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds.detach().cpu(),
            'probs': probs.detach().cpu(),
            'targets': y.detach().cpu()
        })
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self._compute_epoch_metrics(self.training_step_outputs, 'train')
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        self._compute_epoch_metrics(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()
    
    def _compute_epoch_metrics(self, outputs, prefix):
        all_preds = torch.cat([x['preds'] for x in outputs]).numpy()
        all_probs = torch.cat([x['probs'] for x in outputs]).numpy()
        all_targets = torch.cat([x['targets'] for x in outputs]).numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        try:
            auc_ovo = roc_auc_score(all_targets, all_probs, multi_class='ovo', average='macro')
            auc_ovr = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc_ovo = 0.0
            auc_ovr = 0.0
        
        try:
            y_bin = label_binarize(all_targets, classes=range(all_probs.shape[1]))
            pr_auc_per_class = []
            for i in range(all_probs.shape[1]):
                pr_auc_per_class.append(average_precision_score(y_bin[:, i], all_probs[:, i]))
            pr_auc = np.mean(pr_auc_per_class)
        except ValueError:
            pr_auc = 0.0
        
        # Log all metrics
        self.log(f'{prefix}_accuracy', accuracy, prog_bar=True)
        self.log(f'{prefix}_f1', f1, prog_bar=True)
        self.log(f'{prefix}_auc_ovo', auc_ovo, prog_bar=True)
        self.log(f'{prefix}_auc_ovr', auc_ovr, prog_bar=True)
        self.log(f'{prefix}_pr_auc', pr_auc, prog_bar=True)
        
        # Log to our custom logger
        epoch = self.current_epoch
        logger.info(f"Epoch {epoch} - {prefix.upper()} | Loss: {np.mean([x['loss'].item() for x in outputs]):.4f} | "
                   f"Acc: {accuracy:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | "
                   f"AUC-OvO: {auc_ovo:.4f} | AUC-OvR: {auc_ovr:.4f}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

def train():
    logger.info(f"Starting training process")
    logger.info(f"Using preprocessed file: {PREPROC_CSV}")
    assert os.path.exists(PREPROC_CSV), f"Preprocessed CSV not found: {PREPROC_CSV}"
    logger.info(f"PyTorch Lightning version: {pl.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    seed_everything(SEED, workers=True)
    logger.info(f"Random seed set to: {SEED}")
    
    logger.info(f"Primary metric: {METRIC_CONFIG[PRIMARY_METRIC]['name']}")
    logger.info(f"Monitor: {METRIC_CONFIG[PRIMARY_METRIC]['monitor']}")
    logger.info(f"Description: {METRIC_CONFIG[PRIMARY_METRIC]['description']}")
    logger.info(f"Higher is better: {METRIC_CONFIG[PRIMARY_METRIC]['higher_is_better']}")
    
    df = pd.read_csv(PREPROC_CSV)
    logger.info(f"Loaded preprocessed data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Ensure correct ordering within each segment
    df = df.sort_values(["segment_id", "seq_pos"], kind="mergesort").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["segment_id", "label", "csv_file", "seq_pos"]]
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # Group into (segment, sequence of length 24, label)
    segments = []
    labels = []

    for seg_id, g in df.groupby("segment_id", sort=True):
        g = g.sort_values("seq_pos", kind="mergesort")
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        # Expect 24 steps; if shorter/longer, adjust with simple strategies
        if feat.shape[0] < 24:
            # pad by repeating last step
            pad = np.repeat(feat[-1:, :], 24 - feat.shape[0], axis=0)
            feat = np.concatenate([feat, pad], axis=0)
        elif feat.shape[0] > 24:
            # truncate extra steps
            feat = feat[:24, :]

        assert feat.shape[0] == 24, feat.shape
        segments.append(feat)
        labels.append(g["label"].iloc[0])

    X = np.stack(segments, axis=0)  # (N, 24, F)
    y = np.array(labels)

    logger.info(f"Data shape: {X.shape[0]} segments × {X.shape[1]} timesteps × {X.shape[2]} features")
    logger.info(f"Label distribution:")
    label_dist = pd.Series(y).value_counts()
    for label, count in label_dist.items():
        logger.info(f"  {label}: {count} samples ({count/len(y)*100:.1f}%)")
    logger.info("Data loading completed successfully")
    
    label_values = np.sort(pd.unique(y))
    label_to_idx = {lbl: i for i, lbl in enumerate(label_values)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    y_idx = np.vectorize(label_to_idx.get)(y)
    num_classes = len(label_values)

    logger.info(f"\nLabel encoding: {num_classes} classes")
    for lbl, idx in label_to_idx.items():
        logger.info(f"  '{lbl}' → {idx}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_idx, test_size=0.2, random_state=SEED, stratify=y_idx,
    )

    logger.info("DATA SPLIT - Train/Validation")
    logger.info(f"Training segments: {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    logger.info(f"Validation segments: {X_val.shape[0]} ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
    logger.info(f"Split ratio: 80/20 (stratified by label)")
    
    class SegmentDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X)  # (N, T, F)
            self.y = torch.from_numpy(y).long()
        def __len__(self):
            return self.X.shape[0]
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_ds = SegmentDataset(X_train, y_train)
    val_ds = SegmentDataset(X_val, y_val)
    
    class_counts = np.bincount(y_idx)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    logger.info(f"\nClass distribution (all data):")
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        logger.info(f"  Class {i} ({idx_to_label[i]}): {count} samples, weight: {weight:.4f}")
    
    batch_size = BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FlagPatternClassifier(
        input_dim=X.shape[2],
        num_classes=num_classes,
        class_weights=class_weights,
        hidden_channels=64,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=batch_size
    )
    
    logger.info("MODEL ARCHITECTURE")
    logger.info(str(model))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nTotal parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    checkpoint_dir = os.path.join(EXPORT_DIR, "checkpoints_v2")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'best_model_{METRIC_CONFIG[PRIMARY_METRIC]["short"]}' + '_{epoch:02d}_{' + METRIC_CONFIG[PRIMARY_METRIC]["monitor"] + ':.4f}',
        monitor=METRIC_CONFIG[PRIMARY_METRIC]["monitor"],
        mode='max' if METRIC_CONFIG[PRIMARY_METRIC]['higher_is_better'] else 'min',
        save_top_k=1,
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    logger.info("CONFIGURATION - Training Hyperparameters")
    logger.info(f"Maximum epochs: {MAX_EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Weight decay: {WEIGHT_DECAY}")
    logger.info(f"Optimizer: AdamW")
    logger.info(f"LR Scheduler: CosineAnnealingLR (T_max=20)")
    logger.info(f"Loss function: CrossEntropyLoss (with class weights)")
    logger.info(f"Accelerator: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Model selection metric: {METRIC_CONFIG[PRIMARY_METRIC]['name']}")
    logger.info(f"Monitor: {METRIC_CONFIG[PRIMARY_METRIC]['monitor']} (mode: {'max' if METRIC_CONFIG[PRIMARY_METRIC]['higher_is_better'] else 'min'})")
    
    logger.info("TRAINING PROGRESS")
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.fit(model, train_loader, val_loader)

    logger.info(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Best checkpoint saved: {checkpoint_callback.best_model_path}")
    logger.info(f"Best {METRIC_CONFIG[PRIMARY_METRIC]['name']}: {checkpoint_callback.best_model_score:.4f}")
    
    # Plot training history from PyTorch Lightning logs
    metrics_df = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

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
    metric_name = METRIC_CONFIG[PRIMARY_METRIC]['monitor']
    train_metric_name = metric_name.replace('val_', 'train_')
    train_metric = metrics_df.dropna(subset=[train_metric_name])
    val_metric = metrics_df.dropna(subset=[metric_name])

    ax2.plot(train_metric['epoch'], train_metric[train_metric_name], 'b-', 
            label=f'Training {METRIC_CONFIG[PRIMARY_METRIC]["name"]}', linewidth=2)
    ax2.plot(val_metric['epoch'], val_metric[metric_name], 'r-', 
            label=f'Validation {METRIC_CONFIG[PRIMARY_METRIC]["name"]}', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel(METRIC_CONFIG[PRIMARY_METRIC]['name'], fontsize=12)
    ax2.set_title(f'Training and Validation {METRIC_CONFIG[PRIMARY_METRIC]["name"]}', 
                fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    # Save metadata needed for evaluation
    eval_metadata = {
        "checkpoint_path": checkpoint_callback.best_model_path,
        "num_classes": num_classes,
        "input_dim": X.shape[2],
        "class_weights": class_weights.tolist(),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "feature_cols": feature_cols,
    }

    metadata_path = os.path.join(EXPORT_DIR, "eval_metadata.pt")
    torch.save(eval_metadata, metadata_path)

    logger.info(f"Saved evaluation metadata to: {metadata_path}")
if __name__ == "__main__":
    train()