"""
evaluate.py — Evaluation & Metrics for the Regime Detection System.

Computes classification metrics, financial metrics, and transition
detection quality. Generates visualizations for regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score,
)

from config import (
    NUM_REGIMES, REGIME_NAMES, REGIME_COLORS,
    PLOT_DIR, LOG_DIR,
)
from utils import logger, plot_regime_overlay, plot_transition_probabilities

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_transition_class: bool = True,
) -> Dict:
    """
    Compute per-regime and overall classification metrics.

    Returns dict with accuracy, per-class precision/recall/f1, confusion matrix.
    """
    # Define label names
    n_classes = NUM_REGIMES + 1 if include_transition_class else NUM_REGIMES
    label_names = list(REGIME_NAMES.values())
    if include_transition_class:
        label_names.append("Transition")

    labels = list(range(n_classes))

    # Filter to valid labels
    valid_mask = (y_true < n_classes) & (y_pred < n_classes)
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]

    if len(y_true_v) == 0:
        logger.warning("No valid samples for evaluation!")
        return {}

    # Overall metrics
    accuracy = accuracy_score(y_true_v, y_pred_v)
    f1_weighted = f1_score(y_true_v, y_pred_v, labels=labels, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true_v, y_pred_v, labels=labels, average="macro", zero_division=0)

    # Per-class metrics
    report = classification_report(
        y_true_v, y_pred_v,
        labels=labels,
        target_names=label_names[:n_classes],
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true_v, y_pred_v, labels=labels)

    result = {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "per_class": report,
        "confusion_matrix": cm,
        "label_names": label_names[:n_classes],
        "n_samples": len(y_true_v),
    }

    return result


def compute_transition_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_labels_true: np.ndarray,
    regime_labels_pred: np.ndarray,
) -> Dict:
    """
    Compute metrics specific to transition detection quality.

    Measures:
      - Transition detection precision (of predicted transitions, how many are real)
      - Transition detection recall (of real transitions, how many did we catch)
      - Average detection latency (bars between true and predicted transition)
      - False transition rate
    """
    transition_class = NUM_REGIMES  # Class 5

    # True and predicted transition masks
    true_transitions = y_true == transition_class
    pred_transitions = y_pred == transition_class

    # Precision: of predicted transitions, how many overlap with true transitions
    if pred_transitions.sum() > 0:
        transition_precision = (true_transitions & pred_transitions).sum() / pred_transitions.sum()
    else:
        transition_precision = 0.0

    # Recall: of true transitions, how many did we predict
    if true_transitions.sum() > 0:
        transition_recall = (true_transitions & pred_transitions).sum() / true_transitions.sum()
    else:
        transition_recall = 0.0

    # F1
    if transition_precision + transition_recall > 0:
        transition_f1 = 2 * transition_precision * transition_recall / (transition_precision + transition_recall)
    else:
        transition_f1 = 0.0

    # Detection latency: for each true regime change, how many bars until
    # the predicted labels also change?
    true_changes = np.where(np.diff(regime_labels_true) != 0)[0] + 1
    pred_changes = np.where(np.diff(regime_labels_pred) != 0)[0] + 1

    latencies = []
    for tc in true_changes:
        # Find the nearest predicted change after (or at) this true change
        future_pred = pred_changes[pred_changes >= tc]
        if len(future_pred) > 0:
            latency = future_pred[0] - tc
            latencies.append(latency)
        # Also check if there's a predicted change slightly before (early detection!)
        past_pred = pred_changes[(pred_changes >= tc - 10) & (pred_changes < tc)]
        if len(past_pred) > 0:
            # Negative latency = early detection!
            early_latency = past_pred[-1] - tc
            latencies.append(early_latency)

    avg_latency = np.mean(latencies) if latencies else float("nan")
    median_latency = np.median(latencies) if latencies else float("nan")

    # False transition rate
    if len(pred_changes) > 0 and len(true_changes) > 0:
        # A predicted change is "false" if no true change occurs within ±10 bars
        false_count = 0
        for pc in pred_changes:
            if not any(abs(pc - tc) <= 10 for tc in true_changes):
                false_count += 1
        false_transition_rate = false_count / len(pred_changes)
    else:
        false_transition_rate = 0.0

    return {
        "transition_precision": transition_precision,
        "transition_recall": transition_recall,
        "transition_f1": transition_f1,
        "avg_detection_latency_bars": avg_latency,
        "median_detection_latency_bars": median_latency,
        "false_transition_rate": false_transition_rate,
        "n_true_transitions": len(true_changes),
        "n_pred_transitions": len(pred_changes),
        "n_true_transition_bars": true_transitions.sum(),
        "n_pred_transition_bars": pred_transitions.sum(),
    }


# ══════════════════════════════════════════════════════════════════════
# FINANCIAL METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_financial_metrics(
    prices: pd.Series,
    regime_labels: np.ndarray,
) -> Dict:
    """
    Compute per-regime financial characteristics to validate
    that detected regimes have economically meaningful properties.
    """
    returns = prices.pct_change().fillna(0)
    aligned_returns = returns.values[-len(regime_labels):]

    result = {}
    for regime_id in range(NUM_REGIMES):
        mask = regime_labels == regime_id
        if mask.sum() < 10:
            continue

        r = aligned_returns[mask]
        name = REGIME_NAMES[regime_id]

        # Annualized metrics (M5 bars: 12/hr * ~22hr/day * ~252 days)
        bars_per_year = 12 * 22 * 252
        ann_return = np.mean(r) * bars_per_year
        ann_vol = np.std(r) * np.sqrt(bars_per_year)
        sharpe = ann_return / (ann_vol + 1e-10)

        # Drawdown
        cumulative = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        result[name] = {
            "count_bars": int(mask.sum()),
            "pct_time": mask.mean(),
            "ann_return": ann_return,
            "ann_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "mean_bar_return": np.mean(r),
            "skewness": float(pd.Series(r).skew()),
        }

    return result


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """Plot confusion matrix as a heatmap."""
    if not HAS_PLOTTING:
        return

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    # Normalize
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax, linewidths=0.5, linecolor="#1a1a2e",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=15)
    ax.set_xlabel("Predicted", fontsize=12, color="white")
    ax.set_ylabel("True", fontsize=12, color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
):
    """Plot training and validation loss curves."""
    if not HAS_PLOTTING:
        return

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.plot(train_losses, color="#2ecc71", label="Train Loss", linewidth=1.5)
    if val_losses:
        ax.plot(val_losses, color="#e74c3c", label="Val Loss", linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Epoch", fontsize=12, color="white")
    ax.set_ylabel("Focal Loss", fontsize=12, color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax.grid(True, alpha=0.15, color="white")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# FULL EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════

def generate_evaluation_report(
    fold_results: List[Dict],
    prices: Optional[pd.Series] = None,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Generate comprehensive evaluation report across all folds.

    Args:
        fold_results: List of per-fold result dicts from train_pipeline.
        prices: Full price series for financial metrics.
        save_dir: Directory to save plots and reports.

    Returns:
        Comprehensive metrics dict.
    """
    save_dir = save_dir or str(PLOT_DIR)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION REPORT")
    logger.info("=" * 70)

    all_metrics = []

    for i, fold in enumerate(fold_results):
        logger.info(f"\n--- Fold {i+1} ---")

        y_true = fold.get("test_labels")
        y_pred = fold.get("test_predictions")

        if y_true is None or y_pred is None:
            continue

        # Classification metrics
        cls_metrics = compute_classification_metrics(y_true, y_pred)

        if cls_metrics:
            logger.info(f"  Accuracy: {cls_metrics['accuracy']:.4f}")
            logger.info(f"  F1 (weighted): {cls_metrics['f1_weighted']:.4f}")
            logger.info(f"  F1 (macro): {cls_metrics['f1_macro']:.4f}")

            # Per-class
            for name in cls_metrics.get("label_names", []):
                if name in cls_metrics["per_class"]:
                    pc = cls_metrics["per_class"][name]
                    logger.info(f"    {name:25s} — P={pc['precision']:.3f} "
                                f"R={pc['recall']:.3f} F1={pc['f1-score']:.3f} "
                                f"n={pc['support']:.0f}")

            # Confusion matrix plot
            plot_confusion_matrix(
                cls_metrics["confusion_matrix"],
                cls_metrics["label_names"],
                title=f"Fold {i+1} Confusion Matrix",
                save_path=str(Path(save_dir) / f"confusion_matrix_fold{i+1}.png"),
            )

        # Transition detection metrics
        regime_true = y_true.copy()
        regime_true[regime_true == NUM_REGIMES] = 0  # Map transition back for regime comparison
        regime_pred = y_pred.copy()
        regime_pred[regime_pred == NUM_REGIMES] = 0

        trans_metrics = compute_transition_detection_metrics(
            y_true, y_pred, regime_true, regime_pred
        )
        logger.info(f"\n  Transition Detection:")
        logger.info(f"    Precision: {trans_metrics['transition_precision']:.3f}")
        logger.info(f"    Recall:    {trans_metrics['transition_recall']:.3f}")
        logger.info(f"    F1:        {trans_metrics['transition_f1']:.3f}")
        logger.info(f"    Avg latency: {trans_metrics['avg_detection_latency_bars']:.1f} bars")
        logger.info(f"    False rate:  {trans_metrics['false_transition_rate']:.3f}")

        # Training curves
        plot_training_curves(
            fold.get("train_losses", []),
            fold.get("val_losses", []),
            title=f"Fold {i+1} Training Curves",
            save_path=str(Path(save_dir) / f"training_curves_fold{i+1}.png"),
        )

        all_metrics.append({
            "fold": i + 1,
            "classification": cls_metrics,
            "transition": trans_metrics,
        })

    # Summary across folds
    if all_metrics:
        logger.info(f"\n{'=' * 70}")
        logger.info("AGGREGATE RESULTS")
        logger.info(f"{'=' * 70}")

        accs = [m["classification"]["accuracy"] for m in all_metrics if "classification" in m and m["classification"]]
        f1s = [m["classification"]["f1_weighted"] for m in all_metrics if "classification" in m and m["classification"]]
        t_precs = [m["transition"]["transition_precision"] for m in all_metrics if "transition" in m]
        t_recs = [m["transition"]["transition_recall"] for m in all_metrics if "transition" in m]

        if accs:
            logger.info(f"  Accuracy:      {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        if f1s:
            logger.info(f"  F1 (weighted): {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        if t_precs:
            logger.info(f"  Trans. Prec:   {np.mean(t_precs):.4f} ± {np.std(t_precs):.4f}")
        if t_recs:
            logger.info(f"  Trans. Recall: {np.mean(t_recs):.4f} ± {np.std(t_recs):.4f}")

    return {"fold_metrics": all_metrics}
