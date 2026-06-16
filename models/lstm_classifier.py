"""
lstm_classifier.py — Layer 3: LSTM Meta-Classifier for Transition Confirmation & Prediction.

Takes outputs from Layer 1 (HMM posteriors) and Layer 2 (BOCPD signals),
combined with PCA features, to make the final regime + transition decision.

Key capabilities:
  - CPU-optimized LSTM architecture (~200K parameters)
  - Focal Loss for class-imbalanced transition detection
  - Sequence-based input (lookback window of 50 bars)
  - 6-class output (5 regimes + "transition-in-progress")
  - Calibrated probability output via softmax
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import lstm_cfg, MODEL_DIR, NUM_REGIMES
from utils import logger


# ══════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Addresses class imbalance by down-weighting easy examples and
    focusing on hard/misclassified ones. Essential because regime
    transitions are rare events.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (batch_size, num_classes).
            targets: Class labels (batch_size,).
        """
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.class_weights, reduction="none"
        )
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        loss = focal_weight * ce_loss
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════

class RegimeSequenceDataset(Dataset):
    """
    Creates sliding-window sequences for LSTM training.

    Each sample is a sequence of (seq_len) consecutive feature vectors,
    with the label being the regime at the LAST timestep.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 50,
    ):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.sequence_length = sequence_length

        # Validate
        assert len(features) == len(labels), "Features and labels must have same length"
        assert len(features) >= sequence_length, (
            f"Need at least {sequence_length} samples, got {len(features)}"
        )

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length - 1]
        return torch.from_numpy(x), torch.tensor(y)


# ══════════════════════════════════════════════════════════════════════
# LSTM MODEL
# ══════════════════════════════════════════════════════════════════════

class RegimeLSTMNetwork(nn.Module):
    """
    CPU-optimized 2-layer LSTM for regime classification.

    Architecture (~200K params):
        Input → LSTM(96) → Dropout → LSTM(48) → Dropout → Dense(32) → Output(6)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int = 96,
        hidden_size_2: int = 48,
        dense_size: int = 32,
        num_classes: int = 6,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size_2, dense_size)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_size, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            logits: (batch_size, num_classes)
        """
        # LSTM Layer 1
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        # LSTM Layer 2
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        # Take the last timestep output
        out = out[:, -1, :]

        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        logits = self.fc2(out)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# WRAPPER CLASS
# ══════════════════════════════════════════════════════════════════════

class RegimeLSTM:
    """
    Layer 3: LSTM Meta-Classifier wrapper with training, prediction, and saving.

    Combines HMM posteriors + BOCPD signals + PCA features into
    the final regime classification with transition detection.
    """

    def __init__(self, config=None):
        self.config = config or lstm_cfg
        self.model = None
        self.is_fitted = False
        self._input_size = None
        self._class_weights = None
        self._train_losses = []
        self._val_losses = []

        # Set CPU threads
        torch.set_num_threads(self.config.num_threads)

    def build_model(self, input_size: int):
        """Initialize the LSTM network."""
        self._input_size = input_size
        self.model = RegimeLSTMNetwork(
            input_size=input_size,
            hidden_size_1=self.config.hidden_size_1,
            hidden_size_2=self.config.hidden_size_2,
            dense_size=self.config.dense_size,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        )
        n_params = self.model.count_parameters()
        logger.info(f"Built LSTM model: {n_params:,} parameters, input_size={input_size}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        class_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model.

        Args:
            X_train: Feature array (n_samples, n_features).
            y_train: Label array (n_samples,).
            X_val: Optional validation features.
            y_val: Optional validation labels.
            epochs: Override max epochs.
            class_weights: Optional array of per-class weights.

        Returns:
            Dict with train_loss and val_loss history.
        """
        epochs = epochs or self.config.max_epochs

        if self.model is None:
            self.build_model(X_train.shape[1])

        # Create datasets
        train_dataset = RegimeSequenceDataset(X_train, y_train, self.config.sequence_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=False,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = RegimeSequenceDataset(X_val, y_val, self.config.sequence_length)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

        # Loss function
        if class_weights is not None:
            cw = torch.FloatTensor(class_weights)
        else:
            cw = None
        criterion = FocalLoss(
            gamma=self.config.focal_loss_gamma,
            alpha=self.config.focal_loss_alpha,
            class_weights=cw,
        )

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )

        # Training loop
        self.model.train()
        best_val_loss = float("inf")
        patience_counter = 0
        self._train_losses = []
        self._val_losses = []
        best_state = None

        logger.info(f"Training LSTM: {epochs} epochs, {len(train_dataset)} samples, "
                    f"batch_size={self.config.batch_size}")

        for epoch in range(epochs):
            # ── Train ──
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / max(n_batches, 1)
            self._train_losses.append(avg_train_loss)

            # ── Validate ──
            avg_val_loss = None
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        logits = self.model(batch_x)
                        loss = criterion(logits, batch_y)
                        val_loss += loss.item()
                        n_val += 1

                avg_val_loss = val_loss / max(n_val, 1)
                self._val_losses.append(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f", val_loss={avg_val_loss:.4f}" if avg_val_loss else ""
                logger.info(f"  Epoch {epoch+1}/{epochs}: "
                            f"train_loss={avg_train_loss:.4f}{val_str}")

        # Restore best model
        if best_state is not None and self.config.save_best_only:
            self.model.load_state_dict(best_state)
            logger.info(f"  Restored best model (val_loss={best_val_loss:.4f})")

        self.is_fitted = True

        return {
            "train_loss": self._train_losses,
            "val_loss": self._val_losses,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for sequences.

        Args:
            X: Feature array (n_samples, n_features).

        Returns:
            Array of predicted labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for sequences.

        Args:
            X: Feature array (n_samples, n_features).

        Returns:
            Array of shape (n_valid_samples, num_classes).
        """
        self._check_fitted()
        self.model.eval()

        dataset = RegimeSequenceDataset(
            X, np.zeros(len(X), dtype=np.int64),  # Dummy labels
            self.config.sequence_length,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        all_probs = []
        with torch.no_grad():
            for batch_x, _ in loader:
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.numpy())

        return np.concatenate(all_probs, axis=0)

    def _check_fitted(self):
        if not self.is_fitted or self.model is None:
            raise RuntimeError("RegimeLSTM is not fitted. Call fit() first.")

    def save(self, filepath: Optional[str] = None):
        """Save model to disk."""
        filepath = filepath or str(MODEL_DIR / "regime_lstm.pt")
        data = {
            "model_state": self.model.state_dict(),
            "config": self.config,
            "input_size": self._input_size,
            "is_fitted": self.is_fitted,
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
        }
        torch.save(data, filepath)
        logger.info(f"Saved RegimeLSTM to {filepath}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "RegimeLSTM":
        """Load model from disk."""
        filepath = filepath or str(MODEL_DIR / "regime_lstm.pt")
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        obj = cls(config=data["config"])
        obj.build_model(data["input_size"])
        obj.model.load_state_dict(data["model_state"])
        obj.is_fitted = data["is_fitted"]
        obj._train_losses = data.get("train_losses", [])
        obj._val_losses = data.get("val_losses", [])
        logger.info(f"Loaded RegimeLSTM from {filepath}")
        return obj
