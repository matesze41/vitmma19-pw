"""
Model training script for flag pattern classification.

This script:
1. Loads preprocessed training data
2. Defines the CNN model architecture
3. Sets up PyTorch Lightning trainer with callbacks
4. Trains the model with validation monitoring
5. Saves the best model checkpoint

Usage:
    python 02-training.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score
)
from sklearn.preprocessing import label_binarize

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

import config
from utils import setup_logger, ensure_dir


class SegmentDataset(Dataset):
    """PyTorch dataset for time series segments."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Numpy array of shape (N, T, F) - segments x timesteps x features
            y: Numpy array of shape (N,) - labels
        """
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FlagPatternClassifier(pl.LightningModule):
    """
    CNN-based classifier for flag pattern recognition.
    
    Architecture:
    - 4 Conv1D layers with BatchNorm, ReLU, and Dropout
    - Adaptive average pooling
    - 3 fully connected layers
    - CrossEntropyLoss with class weights
    """
    
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
        
        # Model architecture
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
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
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.criterion = nn.CrossEntropyLoss()
        
        # Store predictions for epoch-end metrics
        self.validation_step_outputs = []
        self.training_step_outputs = []
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
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
    
    def on_load_checkpoint(self, checkpoint):
        """Handle checkpoint loading, ignore criterion weights if not needed."""
        # Remove criterion.weight if class_weights is None (inference mode)
        if self.class_weights is None and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove any criterion-related keys
            keys_to_remove = [k for k in state_dict.keys() if k.startswith('criterion.')]
            for key in keys_to_remove:
                del state_dict[key]
    
    def forward(self, x):
        """Forward pass."""
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T)
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        logits = self.fc(h)
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
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
        """Validation step."""
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
        """Compute metrics at end of training epoch."""
        self._compute_epoch_metrics(self.training_step_outputs, 'train')
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Compute metrics at end of validation epoch."""
        self._compute_epoch_metrics(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()
    
    def _compute_epoch_metrics(self, outputs, prefix):
        """Compute and log metrics."""
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
        
        # Log all metrics to PyTorch Lightning
        self.log(f'{prefix}_accuracy', accuracy, prog_bar=True)
        self.log(f'{prefix}_f1', f1, prog_bar=True)
        self.log(f'{prefix}_auc_ovo', auc_ovo, prog_bar=True)
        self.log(f'{prefix}_auc_ovr', auc_ovr, prog_bar=True)
        self.log(f'{prefix}_pr_auc', pr_auc, prog_bar=True)
        
        # Log to our custom logger
        epoch = self.current_epoch
        from utils import setup_logger
        logger = setup_logger(__name__, config.LOG_FILE)
        logger.info(f"Epoch {epoch} - {prefix.upper()} | Loss: {np.mean([x['loss'].item() for x in outputs]):.4f} | "
                   f"Acc: {accuracy:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | "
                   f"AUC-OvO: {auc_ovo:.4f} | AUC-OvR: {auc_ovr:.4f}")
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.LR_SCHEDULER_T_MAX
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def load_and_prepare_data(logger):
    """Load preprocessed data and prepare for training."""
    logger.info("[DATA LOADING AND PREPARATION]")
    
    # Load preprocessed dataset
    df = pd.read_csv(config.SEGMENTS_PREPROC_CSV)
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
        
        # Expect 24 steps; pad or truncate if needed
        if feat.shape[0] < config.SEQUENCE_LENGTH:
            pad = np.repeat(feat[-1:, :], config.SEQUENCE_LENGTH - feat.shape[0], axis=0)
            feat = np.concatenate([feat, pad], axis=0)
        elif feat.shape[0] > config.SEQUENCE_LENGTH:
            feat = feat[:config.SEQUENCE_LENGTH, :]
        
        assert feat.shape[0] == config.SEQUENCE_LENGTH, feat.shape
        segments.append(feat)
        labels.append(g["label"].iloc[0])
    
    X = np.stack(segments, axis=0)  # (N, 24, F)
    y = np.array(labels)
    
    logger.info(f"Data shape: {X.shape[0]} segments × {X.shape[1]} timesteps × {X.shape[2]} features")
    logger.info(f"Label distribution:")
    label_dist = pd.Series(y).value_counts()
    for label, count in label_dist.items():
        logger.info(f"  {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Encode labels as integers
    label_values = np.sort(pd.unique(y))
    label_to_idx = {lbl: i for i, lbl in enumerate(label_values)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    
    y_idx = np.vectorize(label_to_idx.get)(y)
    num_classes = len(label_values)
    
    logger.info(f"\nLabel encoding: {num_classes} classes")
    for lbl, idx in label_to_idx.items():
        logger.info(f"  '{lbl}' → {idx}")
    
    return X, y_idx, num_classes, label_to_idx, idx_to_label, feature_cols


def main():
    """Main training pipeline."""
    # Setup global pipeline logger
    logger = setup_logger("pipeline", config.LOG_FILE)
    
    logger.info("[FLAG PATTERN CLASSIFICATION - MODEL TRAINING]")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set random seeds for reproducibility
    seed_everything(config.SEED, workers=True)
    logger.info("[CONFIGURATION - Random Seed]")
    logger.info(f"Random seed set to: {config.SEED}")
    logger.info("This ensures reproducible results across runs")
    
    # Log primary metric
    logger.info("[CONFIGURATION - Primary Evaluation Metric]")
    logger.info(f"Primary metric: {config.METRIC_CONFIG[config.PRIMARY_METRIC]['name']}")
    logger.info(f"Monitor: {config.METRIC_CONFIG[config.PRIMARY_METRIC]['monitor']}")
    logger.info(f"Description: {config.METRIC_CONFIG[config.PRIMARY_METRIC]['description']}")
    logger.info(f"Higher is better: {config.METRIC_CONFIG[config.PRIMARY_METRIC]['higher_is_better']}")
    
    # Load and prepare data
    X, y_idx, num_classes, label_to_idx, idx_to_label, feature_cols = load_and_prepare_data(logger)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_idx, test_size=config.VAL_SIZE, random_state=config.SEED, stratify=y_idx
    )
    
    logger.info("[DATA SPLIT - Train/Validation]")
    logger.info(f"Training segments: {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    logger.info(f"Validation segments: {X_val.shape[0]} ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
    logger.info(f"Split ratio: {int((1-config.VAL_SIZE)*100)}/{int(config.VAL_SIZE*100)} (stratified by label)")
    
    # Create datasets
    train_ds = SegmentDataset(X_train, y_train)
    val_ds = SegmentDataset(X_val, y_val)
    
    # Calculate class weights for imbalanced dataset
    class_counts = np.bincount(y_idx)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    logger.info(f"\nClass distribution (all data):")
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        logger.info(f"  Class {i} ({idx_to_label[i]}): {count} samples, weight: {weight:.4f}")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    model = FlagPatternClassifier(
        input_dim=X.shape[2],
        num_classes=num_classes,
        class_weights=class_weights,
        hidden_channels=config.HIDDEN_CHANNELS,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        batch_size=config.BATCH_SIZE
    )
    
    logger.info("[MODEL ARCHITECTURE]")
    logger.info(str(model))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Setup callbacks and trainer
    ensure_dir(config.CHECKPOINT_DIR)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename=f'best_model_{config.METRIC_CONFIG[config.PRIMARY_METRIC]["short"]}' + 
                 '_{epoch:02d}_{' + config.METRIC_CONFIG[config.PRIMARY_METRIC]["monitor"] + ':.4f}',
        monitor=config.METRIC_CONFIG[config.PRIMARY_METRIC]["monitor"],
        mode='max' if config.METRIC_CONFIG[config.PRIMARY_METRIC]['higher_is_better'] else 'min',
        save_top_k=1,
        verbose=True
    )
    
    logger.info("[CONFIGURATION - Training Hyperparameters]")
    logger.info(f"Maximum epochs: {config.MAX_EPOCHS}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")
    logger.info(f"Weight decay: {config.WEIGHT_DECAY}")
    logger.info(f"Optimizer: AdamW")
    logger.info(f"LR Scheduler: CosineAnnealingLR (T_max={config.LR_SCHEDULER_T_MAX})")
    logger.info(f"Loss function: CrossEntropyLoss (with class weights)")
    logger.info(f"Accelerator: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Checkpoint directory: {config.CHECKPOINT_DIR}")
    logger.info(f"Model selection metric: {config.METRIC_CONFIG[config.PRIMARY_METRIC]['name']}")
    logger.info(f"Monitor: {config.METRIC_CONFIG[config.PRIMARY_METRIC]['monitor']} (mode: {'max' if config.METRIC_CONFIG[config.PRIMARY_METRIC]['higher_is_better'] else 'min'})")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train the model
    logger.info("[TRAINING PROGRESS]")
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer.fit(model, train_loader, val_loader)
    
    logger.info(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Best checkpoint saved: {checkpoint_callback.best_model_path}")
    logger.info(f"Best {config.METRIC_CONFIG[config.PRIMARY_METRIC]['name']}: {checkpoint_callback.best_model_score:.4f}")
    
    # Save metadata for evaluation
    metadata = {
        'best_model_path': checkpoint_callback.best_model_path,
        'best_metric_value': float(checkpoint_callback.best_model_score),
        'num_classes': num_classes,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'feature_cols': feature_cols,
        'input_dim': X.shape[2]
    }
    
    import pickle
    metadata_path = os.path.join(config.EXPORT_DIR, 'training_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"\nSaved training metadata: {metadata_path}")
    logger.info("[TRAINING COMPLETED SUCCESSFULLY]")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

