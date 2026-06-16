"""
train.py — Walk-Forward Training Pipeline for the 3-Layer Regime Detection System.

Orchestrates the full training flow:
  1. Load & clean data
  2. Extract features
  3. Walk-forward cross-validation:
     a. Fit PCA + HMM on training fold → generate regime labels
     b. Run BOCPD on training fold → generate changepoint signals
     c. Combine HMM posteriors + BOCPD features + PCA features → LSTM input
     d. Train LSTM on combined input
     e. Evaluate on out-of-sample test fold
  4. Save models and aggregate metrics
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config import (
    training_cfg, hmm_cfg, bocpd_cfg, lstm_cfg,
    RAW_DATA_FILE, MODEL_DIR, LOG_DIR, PLOT_DIR,
    NUM_REGIMES, REGIME_NAMES,
)
from utils import load_raw_data, clean_data, logger
from feature_engine import extract_all_features
from models.hmm_regime import RegimeHMM
from models.bocpd import MultiFeatureBOCPD
from models.lstm_classifier import RegimeLSTM

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def generate_transition_labels(
    regime_labels: np.ndarray,
    transition_window: int = 5,
) -> np.ndarray:
    """
    Generate labels that include a "transition-in-progress" class.

    For each regime change point, the `transition_window` bars BEFORE
    the change are labeled as "transition" (class NUM_REGIMES = 5).

    Args:
        regime_labels: Array of regime labels (0-4).
        transition_window: Number of bars before transition to label.

    Returns:
        Array of labels (0-5), where 5 = transition-in-progress.
    """
    labels = regime_labels.copy()
    transition_class = NUM_REGIMES  # Class 5

    # Find transition points
    transitions = np.where(np.diff(regime_labels) != 0)[0] + 1

    for t in transitions:
        start = max(0, t - transition_window)
        labels[start:t] = transition_class

    n_transitions = len(transitions)
    n_transition_bars = (labels == transition_class).sum()
    logger.info(f"  Generated transition labels: {n_transitions} transitions, "
                f"{n_transition_bars} transition bars "
                f"({n_transition_bars/len(labels)*100:.1f}%)")

    return labels


def prepare_lstm_input(
    features: pd.DataFrame,
    hmm_posteriors: np.ndarray,
    bocpd_output: pd.DataFrame,
    pca_features: np.ndarray,
) -> np.ndarray:
    """
    Combine all inputs for the LSTM meta-classifier.

    Input vector per timestep:
      - HMM posterior state probabilities (5 values)
      - BOCPD changepoint probability (1 value)
      - BOCPD run length stats (3 values: mean, mode, entropy)
      - Top PCA features (N values)
      - Regime duration counter (1 value)

    Returns:
        Array of shape (n_samples, n_combined_features).
    """
    # Align lengths
    n = min(len(hmm_posteriors), len(bocpd_output), len(pca_features))
    offset_hmm = len(hmm_posteriors) - n
    offset_bocpd = len(bocpd_output) - n
    offset_pca = len(pca_features) - n

    hmm_part = hmm_posteriors[offset_hmm:offset_hmm + n]
    pca_part = pca_features[offset_pca:offset_pca + n]

    bocpd_cols = ["bocpd_cp_prob", "bocpd_rl_mean", "bocpd_rl_mode", "bocpd_rl_entropy"]
    available_bocpd_cols = [c for c in bocpd_cols if c in bocpd_output.columns]
    bocpd_part = bocpd_output[available_bocpd_cols].values[offset_bocpd:offset_bocpd + n]

    # Regime duration counter (bars since last HMM state change)
    hmm_labels = np.argmax(hmm_part, axis=1)
    duration = np.zeros(n)
    count = 0
    for i in range(n):
        if i == 0 or hmm_labels[i] != hmm_labels[i - 1]:
            count = 0
        count += 1
        duration[i] = count
    # Normalize duration
    duration = duration / 200.0  # Scale to ~[0, 2] range

    # Combine
    combined = np.column_stack([
        hmm_part,           # 5 features
        bocpd_part,         # 4 features
        pca_part,           # N features
        duration,           # 1 feature
    ])

    return combined.astype(np.float32)


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency class weights for imbalanced classes."""
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weights = np.ones(NUM_REGIMES + 1)
    for cls, cnt in zip(unique, counts):
        if cls < len(weights):
            weights[int(cls)] = total / (len(unique) * cnt)
    return weights


def walk_forward_split(
    n_samples: int,
    n_folds: int,
    test_size: int,
    purge_gap: int,
    min_train_size: int,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Generate expanding-window walk-forward splits.

    Returns list of ((train_start, train_end), (test_start, test_end)).
    """
    splits = []

    # Work backwards from the end
    for fold in range(n_folds - 1, -1, -1):
        test_end = n_samples - fold * test_size
        test_start = test_end - test_size

        if test_start < 0:
            continue

        train_end = test_start - purge_gap
        train_start = 0  # Expanding window — always starts from beginning

        if train_end - train_start < min_train_size:
            logger.warning(f"Fold {n_folds - fold}: insufficient training data, skipping")
            continue

        splits.append(((train_start, train_end), (test_start, test_end)))

    return splits


def train_pipeline(
    data_path: Optional[Path] = None,
    save_models: bool = True,
) -> Dict:
    """
    Execute the full walk-forward training pipeline.

    Returns:
        Dict with training results, metrics, and model paths.
    """
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING WALK-FORWARD TRAINING PIPELINE")
    logger.info("=" * 70)

    # ── Step 1: Load & Clean Data ──
    logger.info("\n[Step 1/6] Loading and cleaning data...")
    df = load_raw_data(data_path or RAW_DATA_FILE)
    df = clean_data(df)

    # ── Step 2: Extract Features ──
    logger.info("\n[Step 2/6] Extracting features...")
    features, timeframe_data = extract_all_features(df, normalize=True, drop_na=True)

    # Align price series with features
    price_series = df["close"].reindex(features.index)

    logger.info(f"  Feature matrix: {features.shape}")
    logger.info(f"  Date range: {features.index[0]} to {features.index[-1]}")

    # ── Step 3: Walk-Forward Splits ──
    logger.info("\n[Step 3/6] Creating walk-forward splits...")
    n_samples = len(features)
    splits = walk_forward_split(
        n_samples=n_samples,
        n_folds=training_cfg.n_folds,
        test_size=training_cfg.test_size_bars,
        purge_gap=training_cfg.purge_gap_bars,
        min_train_size=training_cfg.min_train_size_bars,
    )
    logger.info(f"  Generated {len(splits)} walk-forward folds")

    # ── Step 4-6: Train each fold ──
    all_fold_results = []
    best_hmm = None
    best_bocpd = None
    best_lstm = None

    for fold_idx, ((train_start, train_end), (test_start, test_end)) in enumerate(splits):
        logger.info(f"\n{'─' * 70}")
        logger.info(f"FOLD {fold_idx + 1}/{len(splits)}")
        logger.info(f"  Train: idx [{train_start:,} – {train_end:,}] "
                    f"({train_end - train_start:,} samples)")
        logger.info(f"  Test:  idx [{test_start:,} – {test_end:,}] "
                    f"({test_end - test_start:,} samples)")
        logger.info(f"  Train dates: {features.index[train_start]} → "
                    f"{features.index[train_end - 1]}")
        logger.info(f"  Test dates:  {features.index[test_start]} → "
                    f"{features.index[min(test_end - 1, n_samples - 1)]}")

        # Split data
        train_feat = features.iloc[train_start:train_end]
        test_feat = features.iloc[test_start:test_end]
        train_prices = price_series.iloc[train_start:train_end]
        test_prices = price_series.iloc[test_start:test_end]

        fold_result = train_single_fold(
            train_features=train_feat,
            test_features=test_feat,
            train_prices=train_prices,
            test_prices=test_prices,
            fold_idx=fold_idx,
        )
        all_fold_results.append(fold_result)

        # Keep the last fold's models as the "production" models
        best_hmm = fold_result["hmm"]
        best_bocpd = fold_result["bocpd"]
        best_lstm = fold_result["lstm"]

    # ── Save production models ──
    if save_models and best_hmm:
        logger.info("\n[Saving] Saving production models...")
        best_hmm.save()
        best_bocpd.save()
        best_lstm.save()

    # ── Aggregate results ──
    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"TRAINING COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"{'=' * 70}")

    return {
        "fold_results": all_fold_results,
        "n_folds": len(splits),
        "elapsed_seconds": elapsed,
        "hmm": best_hmm,
        "bocpd": best_bocpd,
        "lstm": best_lstm,
    }


def train_single_fold(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_prices: pd.Series,
    test_prices: pd.Series,
    fold_idx: int = 0,
) -> Dict:
    """
    Train all 3 layers on a single walk-forward fold.

    Returns dict with models and per-fold metrics.
    """
    # ── Layer 1: HMM ──
    logger.info(f"\n  [Layer 1] Fitting Bayesian HMM...")
    hmm = RegimeHMM()
    hmm.fit(train_features, train_prices)

    # Generate regime labels (retrospective — with full hindsight on training data)
    train_regime_labels = hmm.predict(train_features)
    train_posteriors = hmm.predict_proba(train_features)

    test_regime_labels = hmm.predict(test_features)
    test_posteriors = hmm.predict_proba(test_features)

    logger.info(hmm.summary())

    # ── Layer 2: BOCPD ──
    logger.info(f"\n  [Layer 2] Running BOCPD on leading features...")
    bocpd = MultiFeatureBOCPD()

    # Run BOCPD on training data
    train_bocpd_output = bocpd.process_batch(train_features, reset_first=True)

    # Continue BOCPD on test data (no reset — maintains state)
    test_bocpd_output = bocpd.process_batch(test_features, reset_first=False)

    # ── Prepare LSTM input ──
    logger.info(f"\n  [Preparing] Combining inputs for LSTM...")

    # Get PCA features
    train_pca = hmm._transform(train_features)
    test_pca = hmm._transform(test_features)

    # Combine for LSTM
    train_lstm_input = prepare_lstm_input(
        train_features, train_posteriors, train_bocpd_output, train_pca
    )
    test_lstm_input = prepare_lstm_input(
        test_features, test_posteriors, test_bocpd_output, test_pca
    )

    # Generate transition-aware labels
    train_labels = generate_transition_labels(train_regime_labels, transition_window=5)
    test_labels = generate_transition_labels(test_regime_labels, transition_window=5)

    # Align label lengths with LSTM input
    n_train = len(train_lstm_input)
    n_test = len(test_lstm_input)
    train_labels_aligned = train_labels[-n_train:]
    test_labels_aligned = test_labels[-n_test:]

    logger.info(f"  LSTM input shape: train={train_lstm_input.shape}, test={test_lstm_input.shape}")

    # ── Layer 3: LSTM ──
    logger.info(f"\n  [Layer 3] Training LSTM meta-classifier...")

    # Compute class weights for imbalanced classes
    class_weights = compute_class_weights(train_labels_aligned)
    logger.info(f"  Class weights: {dict(enumerate(class_weights.round(2)))}")

    lstm = RegimeLSTM()
    lstm.build_model(train_lstm_input.shape[1])

    history = lstm.fit(
        X_train=train_lstm_input,
        y_train=train_labels_aligned,
        X_val=test_lstm_input,
        y_val=test_labels_aligned,
        class_weights=class_weights,
    )

    # ── Evaluate ──
    logger.info(f"\n  [Evaluate] Computing fold metrics...")

    # Predictions on test set
    test_preds = lstm.predict(test_lstm_input)
    test_probs = lstm.predict_proba(test_lstm_input)

    # Account for sequence offset
    seq_len = lstm_cfg.sequence_length
    test_labels_eval = test_labels_aligned[seq_len - 1:]

    # Basic accuracy
    if len(test_preds) == len(test_labels_eval):
        accuracy = (test_preds == test_labels_eval).mean()
    else:
        min_len = min(len(test_preds), len(test_labels_eval))
        accuracy = (test_preds[:min_len] == test_labels_eval[:min_len]).mean()

    logger.info(f"  Fold {fold_idx+1} accuracy: {accuracy:.4f}")

    return {
        "hmm": hmm,
        "bocpd": bocpd,
        "lstm": lstm,
        "accuracy": accuracy,
        "train_losses": history["train_loss"],
        "val_losses": history["val_loss"],
        "test_predictions": test_preds,
        "test_labels": test_labels_eval,
        "test_probabilities": test_probs,
        "test_dates": test_features.index[-(len(test_preds)):] if len(test_preds) <= len(test_features) else None,
    }


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = train_pipeline()

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Folds completed: {results['n_folds']}")
    print(f"Total time: {results['elapsed_seconds']:.0f}s")

    for i, fold in enumerate(results["fold_results"]):
        print(f"  Fold {i+1}: accuracy={fold['accuracy']:.4f}")

    avg_acc = np.mean([f["accuracy"] for f in results["fold_results"]])
    print(f"\nAverage accuracy: {avg_acc:.4f}")
