"""
predict.py — Inference module for the 3-Layer Regime Detection System.

Loads trained models and runs the full pipeline on new data.
Used by both the MT5 connector (live) and standalone analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass

from config import (
    MODEL_DIR, NUM_REGIMES, REGIME_NAMES, REGIME_COLORS,
    lstm_cfg, bocpd_cfg,
)
from utils import load_raw_data, clean_data, logger
from feature_engine import extract_all_features
from models.hmm_regime import RegimeHMM
from models.bocpd import MultiFeatureBOCPD
from models.lstm_classifier import RegimeLSTM


@dataclass
class RegimePrediction:
    """Container for a single regime prediction."""
    regime_id: int
    regime_name: str
    confidence: float
    transition_probability: float
    changepoint_probability: float
    all_probabilities: Dict[str, float]
    is_transition_alert: bool

    def __repr__(self):
        alert = " ⚠️ TRANSITION ALERT" if self.is_transition_alert else ""
        return (
            f"Regime: {self.regime_name} (#{self.regime_id}) | "
            f"Confidence: {self.confidence:.1%} | "
            f"Transition Prob: {self.transition_probability:.1%} | "
            f"Changepoint: {self.changepoint_probability:.1%}{alert}"
        )


class RegimePredictor:
    """
    Orchestrates the full 3-layer pipeline for inference.

    Usage:
        predictor = RegimePredictor.load()
        prediction = predictor.predict_latest(df_m5)
    """

    def __init__(
        self,
        hmm: Optional[RegimeHMM] = None,
        bocpd: Optional[MultiFeatureBOCPD] = None,
        lstm: Optional[RegimeLSTM] = None,
    ):
        self.hmm = hmm
        self.bocpd = bocpd
        self.lstm = lstm
        self._warmup_done = False

    @classmethod
    def load(cls, model_dir: Optional[str] = None) -> "RegimePredictor":
        """Load all trained models from disk."""
        model_dir = Path(model_dir or MODEL_DIR)

        hmm = RegimeHMM.load(str(model_dir / "regime_hmm.pkl"))
        bocpd = MultiFeatureBOCPD.load(str(model_dir / "multi_bocpd.pkl"))
        lstm = RegimeLSTM.load(str(model_dir / "regime_lstm.pt"))

        logger.info("Loaded all 3 layers for regime prediction")
        return cls(hmm=hmm, bocpd=bocpd, lstm=lstm)

    def predict_batch(
        self,
        df_m5: pd.DataFrame,
        return_details: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run full pipeline on a batch of M5 data.

        Args:
            df_m5: Clean M5 OHLCV DataFrame.
            return_details: If True, also return intermediate outputs.

        Returns:
            (regime_labels, regime_probabilities)
        """
        # Step 1: Extract features
        features, _ = extract_all_features(df_m5, normalize=True, drop_na=True)

        if len(features) < lstm_cfg.sequence_length + 50:
            raise ValueError(
                f"Need at least {lstm_cfg.sequence_length + 50} valid samples, "
                f"got {len(features)}"
            )

        # Step 2: HMM posteriors
        hmm_posteriors = self.hmm.predict_proba(features)

        # Step 3: BOCPD
        bocpd_output = self.bocpd.process_batch(features, reset_first=True)

        # Step 4: PCA features
        pca_features = self.hmm._transform(features)

        # Step 5: Combine for LSTM
        from train import prepare_lstm_input
        lstm_input = prepare_lstm_input(features, hmm_posteriors, bocpd_output, pca_features)

        # Step 6: LSTM prediction
        regime_probs = self.lstm.predict_proba(lstm_input)
        regime_labels = np.argmax(regime_probs, axis=1)

        # Map transition class back to regime for display
        # (keep raw labels for analysis, but regime_labels shows the "best guess" regime)
        display_labels = regime_labels.copy()
        for i in range(len(display_labels)):
            if display_labels[i] == NUM_REGIMES:
                # For transition bars, use the HMM's best guess as fallback
                offset = len(hmm_posteriors) - len(regime_labels)
                if offset + i >= 0 and offset + i < len(hmm_posteriors):
                    display_labels[i] = np.argmax(hmm_posteriors[offset + i])

        return display_labels, regime_probs

    def predict_latest(
        self,
        df_m5: pd.DataFrame,
        lookback_bars: int = 1500,
    ) -> RegimePrediction:
        """
        Get the current regime prediction for the latest bar.

        Uses the last `lookback_bars` of data for context.

        Args:
            df_m5: Clean M5 OHLCV DataFrame (at least lookback_bars rows).
            lookback_bars: Number of bars to use for context.

        Returns:
            RegimePrediction for the latest timestep.
        """
        # Use last N bars
        df_recent = df_m5.iloc[-lookback_bars:]

        labels, probs = self.predict_batch(df_recent)

        # Get latest prediction
        latest_probs = probs[-1]

        # Regime probabilities (first 5 classes)
        regime_probs = latest_probs[:NUM_REGIMES]
        transition_prob = latest_probs[NUM_REGIMES] if len(latest_probs) > NUM_REGIMES else 0.0

        # Best regime (excluding transition class)
        best_regime = int(np.argmax(regime_probs))
        confidence = float(regime_probs[best_regime])

        # Get BOCPD changepoint probability
        # Run BOCPD on features to get latest changepoint prob
        features, _ = extract_all_features(df_recent, normalize=True, drop_na=True)
        bocpd_output = self.bocpd.process_batch(features, reset_first=True)
        cp_prob = float(bocpd_output["bocpd_cp_prob"].iloc[-1]) if len(bocpd_output) > 0 else 0.0

        # Alert if transition prob or changepoint prob is high
        is_alert = bool((transition_prob > 0.3) or (cp_prob > bocpd_cfg.alert_threshold))

        all_probs = {REGIME_NAMES[i]: float(regime_probs[i]) for i in range(NUM_REGIMES)}

        return RegimePrediction(
            regime_id=best_regime,
            regime_name=REGIME_NAMES[best_regime],
            confidence=confidence,
            transition_probability=float(transition_prob),
            changepoint_probability=cp_prob,
            all_probabilities=all_probs,
            is_transition_alert=is_alert,
        )


# ══════════════════════════════════════════════════════════════════════
# STANDALONE USAGE
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from config import RAW_DATA_FILE

    print("Loading trained models...")
    predictor = RegimePredictor.load()

    print("Loading data...")
    df = load_raw_data(RAW_DATA_FILE)
    df = clean_data(df)

    print("\nRunning prediction on latest data...")
    prediction = predictor.predict_latest(df)
    print(f"\n{prediction}")
    print(f"\nAll probabilities:")
    for name, prob in prediction.all_probabilities.items():
        bar = "█" * int(prob * 40)
        print(f"  {name:25s} {prob:6.1%} {bar}")
