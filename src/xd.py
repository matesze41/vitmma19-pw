# %%
# Imports and configuration
import os
import sys
import math
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime

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

# Setup logger
logger = setup_logger(__name__)

BASE_DATA_DIR = os.path.abspath("../data")
EXPORT_DIR = os.path.join(BASE_DATA_DIR, "export")
PREPROC_CSV = os.path.join(EXPORT_DIR, "segments_preproc_24.csv")

# Log experiment start
logger.info("="*80)
logger.info("FLAG PATTERN CLASSIFICATION EXPERIMENT")
logger.info(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)

logger.info(f"Using preprocessed file: {PREPROC_CSV}")
assert os.path.exists(PREPROC_CSV), f"Preprocessed CSV not found: {PREPROC_CSV}"
logger.info(f"PyTorch Lightning version: {pl.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"PyTorch version: {torch.__version__}")

# %%
# Set random seeds for reproducibility
SEED = 1
seed_everything(SEED, workers=True)

logger.info("\n" + "="*80)
logger.info("CONFIGURATION - Random Seed")
logger.info("="*80)
logger.info(f"Random seed set to: {SEED}")
logger.info("This ensures reproducible results across runs")

# %%
# Model evaluation configuration
# Change this single variable to switch the primary metric for model selection
# Options: 'auc_ovo', 'auc_ovr', 'f1', 'accuracy', 'pr_auc'
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

logger.info("\n" + "="*80)
logger.info("CONFIGURATION - Primary Evaluation Metric")
logger.info("="*80)
logger.info(f"Primary metric: {METRIC_CONFIG[PRIMARY_METRIC]['name']}")
logger.info(f"Monitor: {METRIC_CONFIG[PRIMARY_METRIC]['monitor']}")
logger.info(f"Description: {METRIC_CONFIG[PRIMARY_METRIC]['description']}")
logger.info(f"Higher is better: {METRIC_CONFIG[PRIMARY_METRIC]['higher_is_better']}")

# %%
# Load preprocessed dataset and build segment-level sequences
logger.info("\n" + "="*80)
logger.info("DATA PROCESSING - Loading and Preparing Data")
logger.info("="*80)

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
logger.info("Data loading and preprocessing completed successfully")

# %%
# Encode labels as integers
label_values = np.sort(pd.unique(y))
label_to_idx = {lbl: i for i, lbl in enumerate(label_values)}
idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

y_idx = np.vectorize(label_to_idx.get)(y)
num_classes = len(label_values)

logger.info(f"\nLabel encoding: {num_classes} classes")
for lbl, idx in label_to_idx.items():
    logger.info(f"  '{lbl}' → {idx}")

# %%
# Train/validation split at segment level
X_train, X_val, y_train, y_val = train_test_split(
    X, y_idx, test_size=0.2, random_state=SEED, stratify=y_idx,
 )

logger.info("\n" + "="*80)
logger.info("DATA SPLIT - Train/Validation")
logger.info("="*80)
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

# Calculate class weights for imbalanced dataset
class_counts = np.bincount(y_idx)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_weights)

logger.info(f"\nClass distribution (all data):")
for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
    logger.info(f"  Class {i} ({idx_to_label[i]}): {count} samples, weight: {weight:.4f}")

# %%
# PyTorch Lightning Module
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

batch_size = 12

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

logger.info("\n" + "="*80)
logger.info("MODEL ARCHITECTURE")
logger.info("="*80)
logger.info(str(model))
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"\nTotal parameters: {total_params:,}")
logger.info(f"Trainable parameters: {trainable_params:,}")
logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")

# %%
# Setup callbacks and trainer
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

# Training hyperparameters
MAX_EPOCHS = 50
BATCH_SIZE = batch_size
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Trainer
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

logger.info("\n" + "="*80)
logger.info("CONFIGURATION - Training Hyperparameters")
logger.info("="*80)
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

# %%
# Train the model
logger.info("\n" + "="*80)
logger.info("TRAINING PROGRESS")
logger.info("="*80)
logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

trainer.fit(model, train_loader, val_loader)

logger.info(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Best checkpoint saved: {checkpoint_callback.best_model_path}")
logger.info(f"Best {METRIC_CONFIG[PRIMARY_METRIC]['name']}: {checkpoint_callback.best_model_score:.4f}")

# %%
import matplotlib.pyplot as plt

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

# %%
# Load best model and evaluate
logger.info("\n" + "="*80)
logger.info("MODEL EVALUATION - Loading Best Model")
logger.info("="*80)

best_model = FlagPatternClassifier.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    input_dim=X.shape[2],
    num_classes=num_classes,
    class_weights=class_weights,
    weights_only=False  # Allow loading numpy arrays from checkpoint
)
best_model.eval()
best_model.freeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = best_model.to(device)

logger.info(f"Best model loaded from: {checkpoint_callback.best_model_path}")
logger.info(f"Device: {device}")

def get_predictions(loader, model):
    all_preds = []
    all_probs = []
    all_targets = []
    
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

train_preds, train_targets, train_probs = get_predictions(train_loader, best_model)
val_preds, val_targets, val_probs = get_predictions(val_loader, best_model)

# Calculate metrics
train_acc = accuracy_score(train_targets, train_preds)
train_f1 = f1_score(train_targets, train_preds, average='macro')
train_auc_ovo = roc_auc_score(train_targets, train_probs, multi_class='ovo', average='macro')
train_auc_ovr = roc_auc_score(train_targets, train_probs, multi_class='ovr', average='macro')

val_acc = accuracy_score(val_targets, val_preds)
val_f1 = f1_score(val_targets, val_preds, average='macro')
val_auc_ovo = roc_auc_score(val_targets, val_probs, multi_class='ovo', average='macro')
val_auc_ovr = roc_auc_score(val_targets, val_probs, multi_class='ovr', average='macro')

train_targets_bin = label_binarize(train_targets, classes=range(num_classes))
val_targets_bin = label_binarize(val_targets, classes=range(num_classes))

train_pr_auc_per_class = [average_precision_score(train_targets_bin[:, i], train_probs[:, i]) for i in range(num_classes)]
val_pr_auc_per_class = [average_precision_score(val_targets_bin[:, i], val_probs[:, i]) for i in range(num_classes)]

train_pr_auc = np.mean(train_pr_auc_per_class)
val_pr_auc = np.mean(val_pr_auc_per_class)

logger.info("\n" + "="*80)
logger.info("VALIDATION RESULTS (Best Model)")
logger.info("="*80)
logger.info(f"Training Set:")
logger.info(f"  Accuracy:         {train_acc:.4f}")
logger.info(f"  F1 Score (macro): {train_f1:.4f}")
logger.info(f"  PR-AUC (macro):   {train_pr_auc:.4f}")
logger.info(f"  AUC-ROC (OvO):    {train_auc_ovo:.4f}")
logger.info(f"  AUC-ROC (OvR):    {train_auc_ovr:.4f}")
logger.info(f"\nValidation Set:")
logger.info(f"  Accuracy:         {val_acc:.4f}")
logger.info(f"  F1 Score (macro): {val_f1:.4f}")
logger.info(f"  PR-AUC (macro):   {val_pr_auc:.4f} {'← PRIMARY METRIC' if PRIMARY_METRIC == 'pr_auc' else ''}")
logger.info(f"  AUC-ROC (OvO):    {val_auc_ovo:.4f} {'← PRIMARY METRIC' if PRIMARY_METRIC == 'auc_ovo' else ''}")
logger.info(f"  AUC-ROC (OvR):    {val_auc_ovr:.4f} {'← PRIMARY METRIC' if PRIMARY_METRIC == 'auc_ovr' else ''}")

# %%
# Visualization - Confusion matrices and curves
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_train = confusion_matrix(train_targets, train_preds)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=label_values)
disp_train.plot(ax=axes[0], cmap='Blues', xticks_rotation=45)
axes[0].set_title(f"Training Confusion Matrix\nPR-AUC={train_pr_auc:.4f} | AUC-OvO={train_auc_ovo:.4f}")

cm_val = confusion_matrix(val_targets, val_preds)
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=label_values)
disp_val.plot(ax=axes[1], cmap='Blues', xticks_rotation=45)
axes[1].set_title(f"Validation Confusion Matrix\nPR-AUC={val_pr_auc:.4f} | AUC-OvO={val_auc_ovo:.4f}")

plt.tight_layout()
plt.show()

print("\nClassification report (validation):")
print(classification_report(val_targets, val_preds, target_names=[str(lbl) for lbl in label_values]))

# Plot ROC and PR curves
fig, axes = plt.subplots(2, 6, figsize=(18, 6))

for i in range(num_classes):
    # ROC curve
    fpr, tpr, _ = roc_curve(val_targets_bin[:, i], val_probs[:, i])
    roc_auc_class = auc(fpr, tpr)
    
    axes[0, i].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_class:.3f}')
    axes[0, i].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.5)
    axes[0, i].set_xlim([0.0, 1.0])
    axes[0, i].set_ylim([0.0, 1.05])
    axes[0, i].set_xlabel('FPR', fontsize=8)
    axes[0, i].set_ylabel('TPR', fontsize=8)
    axes[0, i].set_title(f'ROC: {label_values[i]}', fontsize=9)
    axes[0, i].legend(loc="lower right", fontsize=7)
    axes[0, i].grid(alpha=0.3)
    axes[0, i].tick_params(labelsize=7)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(val_targets_bin[:, i], val_probs[:, i])
    pr_auc_class = val_pr_auc_per_class[i]
    
    axes[1, i].plot(recall, precision, color='green', lw=2, label=f'AP = {pr_auc_class:.3f}')
    axes[1, i].set_xlim([0.0, 1.0])
    axes[1, i].set_ylim([0.0, 1.05])
    axes[1, i].set_xlabel('Recall', fontsize=8)
    axes[1, i].set_ylabel('Precision', fontsize=8)
    axes[1, i].set_title(f'PR: {label_values[i]}', fontsize=9)
    axes[1, i].legend(loc="lower left", fontsize=7)
    axes[1, i].grid(alpha=0.3)
    axes[1, i].tick_params(labelsize=7)

plt.tight_layout()
plt.show()

# %%
# Reload the baseline_model module to pick up any changes
import importlib
import sys

# Remove cached module if it exists
if 'baseline_model' in sys.modules:
    del sys.modules['baseline_model']

# Re-import
import baseline_model
print("Baseline model reloaded with current slope_threshold:", baseline_model.BaselineModel().slope_threshold)

# %%
# Evaluate baseline model on the combined segments CSV and plot confusion matrix
import os
from baseline_model import predict_from_segments_csv, evaluate_on_segments_csv

# Path to the combined segment CSV produced by 01-data-exploration.ipynb
segments_csv_path = os.path.abspath("../data/export/segments_values.csv")
print("Using segments CSV for baseline:", segments_csv_path)

# Custom slope threshold for baseline model
BASELINE_SLOPE_THRESHOLD = 0.0002

# Run baseline model on all segments; this returns a DataFrame
baseline_results = predict_from_segments_csv(segments_csv_path, slope_threshold=BASELINE_SLOPE_THRESHOLD)
print("Baseline results shape:", baseline_results.shape)
print(f"Using slope_threshold: {BASELINE_SLOPE_THRESHOLD}")

# Keep only rows with ground-truth labels
mask = baseline_results["gold_label"].notna()
baseline_labels = baseline_results.loc[mask, "gold_label"].values
baseline_preds = baseline_results.loc[mask, "predicted_label"].values

print("Number of evaluated segments:", baseline_labels.shape[0])

# Print accuracy, F1 score, and classification report
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
baseline_acc = accuracy_score(baseline_labels, baseline_preds)
baseline_f1 = f1_score(baseline_labels, baseline_preds, average='macro')
print(f"Baseline accuracy: {baseline_acc:.4f}")
print(f"Baseline F1 score (macro): {baseline_f1:.4f}")

# Use label set from the baseline outputs
import numpy as np
baseline_label_values = np.sort(pd.unique(baseline_labels))
print("Baseline classes:", baseline_label_values)

print("\nClassification report (baseline):")
print(classification_report(baseline_labels, baseline_preds,
                            labels=baseline_label_values,
                            target_names=[str(lbl) for lbl in baseline_label_values]))

# Plot confusion matrix for baseline model
import matplotlib.pyplot as plt
cm_baseline = confusion_matrix(baseline_labels, baseline_preds, labels=baseline_label_values)
disp_baseline = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=baseline_label_values)
fig, ax = plt.subplots(figsize=(6, 5))
disp_baseline.plot(ax=ax, cmap="Oranges", xticks_rotation=45)
ax.set_title(f"Baseline Model Confusion Matrix (threshold={BASELINE_SLOPE_THRESHOLD})")
plt.tight_layout()
plt.show()

# %%
# Compare confusion matrices of CNN model vs baseline on all segments
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score

# 1) CNN model predictions on all segments used for training
full_ds = SegmentDataset(X, y_idx)
full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False)

cnn_preds_all, cnn_targets_all, cnn_probs_all = get_predictions(full_loader, best_model)

# Map integer indices back to label strings
cnn_true_labels = np.vectorize(idx_to_label.get)(cnn_targets_all)
cnn_pred_labels = np.vectorize(idx_to_label.get)(cnn_preds_all)

# Calculate all CNN metrics
cnn_acc = accuracy_score(cnn_true_labels, cnn_pred_labels)
cnn_f1 = f1_score(cnn_true_labels, cnn_pred_labels, average='macro')
cnn_auc_ovo = roc_auc_score(cnn_targets_all, cnn_probs_all, multi_class='ovo', average='macro')
cnn_auc_ovr = roc_auc_score(cnn_targets_all, cnn_probs_all, multi_class='ovr', average='macro')

# Calculate PR-AUC for CNN
cnn_targets_bin = label_binarize(cnn_targets_all, classes=range(num_classes))
cnn_pr_auc_per_class = [average_precision_score(cnn_targets_bin[:, i], cnn_probs_all[:, i]) for i in range(num_classes)]
cnn_pr_auc = np.mean(cnn_pr_auc_per_class)

# 2) Baseline predictions (already computed from segments_values.csv)
#    Using baseline_labels and baseline_preds from the previous cell

# Use the global label set from the CNN pipeline for consistent ordering
all_labels = label_values
print("Confusion matrix classes (ordered):", all_labels)

cm_cnn_all = confusion_matrix(cnn_true_labels, cnn_pred_labels, labels=all_labels)
cm_baseline_all = confusion_matrix(baseline_labels, baseline_preds, labels=all_labels)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# CNN confusion matrix
cnn_disp = ConfusionMatrixDisplay(confusion_matrix=cm_cnn_all, display_labels=all_labels)
cnn_disp.plot(ax=axes[0], cmap="Blues", xticks_rotation=45, colorbar=False)
axes[0].set_title(f"CNN Model Confusion Matrix (All Segments)\n"
                  f"Primary: {METRIC_CONFIG[PRIMARY_METRIC]['name']} | "
                  f"PR-AUC: {cnn_pr_auc:.4f} | AUC-OvO: {cnn_auc_ovo:.4f}")

# Baseline confusion matrix
base_disp = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_all, display_labels=all_labels)
base_disp.plot(ax=axes[1], cmap="Oranges", xticks_rotation=45, colorbar=False)
axes[1].set_title(f"Baseline Model Confusion Matrix (All Segments)\nAccuracy: {baseline_acc:.4f}")

plt.tight_layout()
plt.show()

# Print comparison summary
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
print(f"CNN Model (selected by {METRIC_CONFIG[PRIMARY_METRIC]['name']}):")
print(f"  Accuracy:         {cnn_acc:.4f}")
print(f"  F1 Score (macro): {cnn_f1:.4f}")
print(f"  PR-AUC (macro):   {cnn_pr_auc:.4f}  ← precision-recall curve")
print(f"  AUC-ROC (OvO):    {cnn_auc_ovo:.4f}  ← pairwise discrimination (15 comparisons)")
print(f"  AUC-ROC (OvR):    {cnn_auc_ovr:.4f}  ← each-vs-rest (6 comparisons)")
print(f"\nBaseline Model (heuristic):")
print(f"  Accuracy:         {baseline_acc:.4f}")
print(f"  F1 Score (macro): {baseline_f1:.4f}")
print("="*70)
print(f"\nPR-AUC is particularly useful for imbalanced datasets")
print(f"It focuses on positive class performance and precision-recall trade-offs")

# %%
# Evaluate on held-out test set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    ConfusionMatrixDisplay, f1_score, roc_auc_score, 
    average_precision_score
)
from sklearn.preprocessing import label_binarize

logger.info("\n" + "="*80)
logger.info("TEST SET EVALUATION - CNN Model")
logger.info("="*80)

# Load test data
TEST_PREPROC_CSV = os.path.join(EXPORT_DIR, "segments_preproc_24_test.csv")
TEST_RAW_CSV = os.path.join(EXPORT_DIR, "segments_test_raw.csv")

logger.info(f"Loading test data from:")
logger.info(f"  Preprocessed: {TEST_PREPROC_CSV}")
logger.info(f"  Raw (for baseline): {TEST_RAW_CSV}")

df_test = pd.read_csv(TEST_PREPROC_CSV)
df_test = df_test.sort_values(["segment_id", "seq_pos"], kind="mergesort").reset_index(drop=True)

# Build test sequences (same as training pipeline)
test_segments = []
test_labels = []

for seg_id, g in df_test.groupby("segment_id", sort=True):
    g = g.sort_values("seq_pos", kind="mergesort")
    feat = g[feature_cols].to_numpy(dtype=np.float32)
    
    # Pad or truncate to 24 steps
    if feat.shape[0] < 24:
        pad = np.repeat(feat[-1:, :], 24 - feat.shape[0], axis=0)
        feat = np.concatenate([feat, pad], axis=0)
    elif feat.shape[0] > 24:
        feat = feat[:24, :]
    
    assert feat.shape[0] == 24, feat.shape
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
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Get predictions on test set
test_preds, test_targets, test_probs = get_predictions(test_loader, best_model)

# Calculate all metrics for test set
test_acc = accuracy_score(test_targets, test_preds)
test_f1 = f1_score(test_targets, test_preds, average='macro')
test_auc_ovo = roc_auc_score(test_targets, test_probs, multi_class='ovo', average='macro')
test_auc_ovr = roc_auc_score(test_targets, test_probs, multi_class='ovr', average='macro')

# Calculate PR-AUC
test_targets_bin = label_binarize(test_targets, classes=range(num_classes))
test_pr_auc_per_class = [average_precision_score(test_targets_bin[:, i], test_probs[:, i]) for i in range(num_classes)]
test_pr_auc = np.mean(test_pr_auc_per_class)

# Map predictions back to labels
test_true_labels = np.vectorize(idx_to_label.get)(test_targets)
test_pred_labels = np.vectorize(idx_to_label.get)(test_preds)

logger.info("\n" + "="*80)
logger.info("FINAL EVALUATION - Test Set Results (CNN Model)")
logger.info("="*80)
logger.info(f"Accuracy:         {test_acc:.4f}")
logger.info(f"F1 Score (macro): {test_f1:.4f}")
logger.info(f"PR-AUC (macro):   {test_pr_auc:.4f} {'← PRIMARY METRIC' if PRIMARY_METRIC == 'pr_auc' else ''}")
logger.info(f"AUC-ROC (OvO):    {test_auc_ovo:.4f}")
logger.info(f"AUC-ROC (OvR):    {test_auc_ovr:.4f}")

# Log confusion matrix
cm_test = confusion_matrix(test_true_labels, test_pred_labels, labels=label_values)
logger.info("\nConfusion Matrix (Test Set):")
logger.info(f"Classes: {list(label_values)}")
for i, row in enumerate(cm_test):
    logger.info(f"  {label_values[i]}: {list(row)}")

# Log classification report
logger.info("\nDetailed Classification Report (Test Set):")
report = classification_report(test_true_labels, test_pred_labels, target_names=[str(lbl) for lbl in label_values])
for line in report.split('\n'):
    if line.strip():
        logger.info(f"  {line}")

# Plot test confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=label_values)
disp_test.plot(ax=ax, cmap="Greens", xticks_rotation=45, colorbar=True)
ax.set_title(f"CNN Model - Test Set Confusion Matrix\n"
             f"PR-AUC: {test_pr_auc:.4f} | AUC-OvO: {test_auc_ovo:.4f} | Accuracy: {test_acc:.4f}")
plt.tight_layout()
plt.show()

# %%
# Evaluate baseline model on test set
from baseline_model import predict_from_segments_csv

print("\n" + "="*70)
print("TEST SET EVALUATION - BASELINE MODEL")
print("="*70)

# Run baseline on test raw data
baseline_test_results = predict_from_segments_csv(TEST_RAW_CSV, slope_threshold=BASELINE_SLOPE_THRESHOLD)
print(f"Baseline test results shape: {baseline_test_results.shape}")

# Keep only rows with ground-truth labels
mask_test = baseline_test_results["gold_label"].notna()
baseline_test_labels = baseline_test_results.loc[mask_test, "gold_label"].values
baseline_test_preds = baseline_test_results.loc[mask_test, "predicted_label"].values

print(f"Number of evaluated test segments: {baseline_test_labels.shape[0]}")

# Calculate metrics
baseline_test_acc = accuracy_score(baseline_test_labels, baseline_test_preds)
baseline_test_f1 = f1_score(baseline_test_labels, baseline_test_preds, average='macro')

print(f"Accuracy:         {baseline_test_acc:.4f}")
print(f"F1 Score (macro): {baseline_test_f1:.4f}")
print("="*70)

# Plot baseline test confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm_baseline_test = confusion_matrix(baseline_test_labels, baseline_test_preds, labels=label_values)
disp_baseline_test = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_test, display_labels=label_values)
disp_baseline_test.plot(ax=ax, cmap="Oranges", xticks_rotation=45, colorbar=True)
ax.set_title(f"Baseline Model - Test Set Confusion Matrix\n"
             f"Accuracy: {baseline_test_acc:.4f} | F1: {baseline_test_f1:.4f}")
plt.tight_layout()
plt.show()

print("\nClassification Report (Baseline - Test Set):")
print(classification_report(baseline_test_labels, baseline_test_preds, 
                            labels=label_values,
                            target_names=[str(lbl) for lbl in label_values]))

# %%
# Side-by-side comparison: CNN vs Baseline on Test Set
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# CNN test confusion matrix
cnn_disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=label_values)
cnn_disp_test.plot(ax=axes[0], cmap="Greens", xticks_rotation=45, colorbar=False)
axes[0].set_title(f"CNN Model - Test Set\n"
                  f"PR-AUC: {test_pr_auc:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

# Baseline test confusion matrix
base_disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_test, display_labels=label_values)
base_disp_test.plot(ax=axes[1], cmap="Oranges", xticks_rotation=45, colorbar=False)
axes[1].set_title(f"Baseline Model - Test Set\n"
                  f"Acc: {baseline_test_acc:.4f} | F1: {baseline_test_f1:.4f}")

plt.tight_layout()
plt.show()

# Print comprehensive comparison
logger.info("\n" + "="*80)
logger.info("FINAL MODEL COMPARISON - Test Set Performance")
logger.info("="*80)
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

logger.info("\n" + "="*80)
logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)


