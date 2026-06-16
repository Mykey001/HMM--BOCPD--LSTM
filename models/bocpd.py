"""
bocpd.py — Layer 2: Online Bayesian Changepoint Detection (BOCPD).

Implements the Adams & MacKay (2007) algorithm for detecting structural
breaks in streaming data — enabling EARLY detection of regime transitions
before they fully manifest.

REWRITE vs. original
---------------------
1. Priors are AUTO-CALIBRATED from a warm-up window, not hardcoded.
2. Changepoint signal uses the canonical P(r_t = 0 | x_{1:t}) — the
   probability that the current observation starts a new run — instead
   of summing short run-length masses (which was semantically wrong).
3. All internal arithmetic is in LOG-SPACE to prevent underflow.
4. A confidence score ramps from 0 → 1 over the warm-up period. The
   raw changepoint_probability is always computed honestly; only the
   alert boolean is gated during warm-up so the LSTM sees real signal.
5. MultiFeatureBOCPD uses consensus-based alerts (multiple features
   must agree) and exposes a weighted changepoint probability.

Key capabilities:
  - Real-time changepoint probability computation
  - Run-length distribution tracking
  - Auto-calibrated observation model
  - Multi-feature monitoring (watches "leading" features)
  - Alert system with warm-up gating & consensus
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from pathlib import Path
import joblib
from scipy.special import gammaln

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import bocpd_cfg, MODEL_DIR
from utils import logger


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BOCPDResult:
    """Container for BOCPD output at a single timestep."""
    changepoint_probability: float   # P(r_t = 0 | x_{1:t}) — canonical Adams & MacKay
    run_length_mean: float           # Expected run length
    run_length_mode: int             # Most likely run length
    run_length_entropy: float        # Entropy of run-length distribution
    alert: bool                      # Gated: (cp_prob > threshold) AND past warm-up
    growth_probability: float        # 1 - changepoint_probability
    confidence: float                # 0 → 1 ramp reflecting statistical maturity


# ──────────────────────────────────────────────────────────────────────
# Observation model
# ──────────────────────────────────────────────────────────────────────

class NormalInverseGamma:
    """
    Normal-Inverse-Gamma conjugate prior for Gaussian observations
    with unknown mean and variance.

    Can be CALIBRATED from data so the prior is informed rather than
    arbitrary — this is the key fix for the zero-alert problem.
    """

    def __init__(self, mu_0: float = 0.0, kappa_0: float = 1.0,
                 alpha_0: float = 1.0, beta_0: float = 1.0):
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

    def calibrate_from_data(self, data: np.ndarray) -> None:
        """
        Empirically set NIG prior hyperparameters from a data window.

        The prior must be WEAKLY informed — it should know the right
        SCALE (so predictions aren't wildly off) but be flexible about
        the MEAN (so a new regime with a different mean is immediately
        recognisable as a changepoint).

        Sets:
          mu_0    = sample mean          (centred on the data)
          kappa_0 = 1.0                  (one pseudo-obs → weak mean)
          alpha_0 = 1.5                  (df=3 Student-t → heavy tails)
          beta_0  = sample_var * alpha_0 (predictive scale ≈ data scale)

        With these settings the predictive distribution for a CONTINUING
        run has accumulated many observations and is tight, while the
        predictive for a NEW run (r=0) is broad.  A mean-shift creates
        a large likelihood ratio favouring r=0, producing a strong
        changepoint signal.

        Args:
            data: 1D array of observations (at least 3 values).
        """
        data = np.asarray(data, dtype=np.float64)
        data = data[np.isfinite(data)]
        if len(data) < 3:
            logger.warning("calibrate_from_data got < 3 finite values; "
                           "using fallback N(0,1) prior")
            self.mu_0 = 0.0
            self.kappa_0 = 1.0
            self.alpha_0 = 1.5
            self.beta_0 = 1.5   # beta/alpha = 1.0 → unit variance
            return

        self.mu_0 = float(np.mean(data))
        self.kappa_0 = 1.0                              # Weak: 1 pseudo-observation
        self.alpha_0 = 1.5                              # Minimal (df=3 → finite var)
        sample_var = max(1e-6, float(np.var(data)))     # Floor to prevent degeneracy
        self.beta_0 = sample_var * self.alpha_0         # So predictive scale ≈ sample_var

    def compute_log_predictive(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """
        Compute log predictive probability p(x | params) under a
        Student-t predictive distribution.

        The predictive distribution is:
            t_{2α}(x | μ, β(κ+1)/(ακ))

        Args:
            x: New observation.
            mu, kappa, alpha, beta: Arrays of sufficient statistics
                (one entry per run length).

        Returns:
            Array of log predictive probabilities.
        """
        df = 2.0 * alpha                         # Degrees of freedom
        scale = beta * (kappa + 1.0) / (alpha * kappa)  # Variance scale

        # Protect against degenerate values
        scale = np.maximum(scale, 1e-12)
        df = np.maximum(df, 1e-12)

        # Standardised deviation
        z = (x - mu) ** 2 / scale

        # Log PDF of Student-t
        log_prob = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(np.pi * df * scale)
            - ((df + 1.0) / 2.0) * np.log1p(z / df)
        )

        return log_prob

    def update_stats(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Bayesian update of sufficient statistics after observing x.

        Returns updated (mu, kappa, alpha, beta).
        """
        new_kappa = kappa + 1.0
        new_mu = (kappa * mu + x) / new_kappa
        new_alpha = alpha + 0.5
        new_beta = beta + 0.5 * kappa * (x - mu) ** 2 / new_kappa

        return new_mu, new_kappa, new_alpha, new_beta


# ──────────────────────────────────────────────────────────────────────
# Single-feature BOCPD detector
# ──────────────────────────────────────────────────────────────────────

class BOCPD:
    """
    Layer 2: Online Bayesian Changepoint Detection (single feature).

    Key differences from the original implementation:
    - Auto-calibrates NIG priors from a warm-up window.
    - Uses canonical P(r_t = 0) as the changepoint signal.
    - Full log-space recursion for numerical stability.
    - Confidence-gated alerts (suppressed during warm-up).
    - Raw changepoint_probability is NEVER zeroed — LSTM always sees
      real signal.
    """

    def __init__(self, config=None):
        self.config = config or bocpd_cfg
        self.observation_model = NormalInverseGamma()  # Will be calibrated

        # State variables
        self._log_run_length_dist = None   # Log run-length posterior
        self._mu = None
        self._kappa = None
        self._alpha = None
        self._beta = None
        self._t = 0                        # Time index
        self._max_run_length = 500         # Truncation limit
        self._warm_up_length = self.config.warm_up_length
        self._confidence_ramp = self.config.confidence_ramp_length

        self.is_initialized = False
        self.is_calibrated = False
        self._history: List[BOCPDResult] = []

    def calibrate(self, data: np.ndarray) -> None:
        """
        Calibrate the NIG prior from a data array.

        Should be called BEFORE update() for best results. Called
        automatically by process_batch() using the first warm_up_length
        observations.

        Args:
            data: 1D array of observations for calibration.
        """
        self.observation_model.calibrate_from_data(data)
        self.is_calibrated = True
        logger.debug(
            f"BOCPD calibrated: mu_0={self.observation_model.mu_0:.4f}, "
            f"kappa_0={self.observation_model.kappa_0:.2f}, "
            f"alpha_0={self.observation_model.alpha_0:.2f}, "
            f"beta_0={self.observation_model.beta_0:.4f}"
        )

    def reset(self, calibration_data: Optional[np.ndarray] = None) -> None:
        """
        Reset the detector to initial state.

        Args:
            calibration_data: If provided, calibrate priors from this
                array before resetting.
        """
        if calibration_data is not None:
            self.calibrate(calibration_data)

        om = self.observation_model
        self._log_run_length_dist = np.array([0.0])  # log(1.0) = 0
        self._mu = np.array([om.mu_0])
        self._kappa = np.array([om.kappa_0])
        self._alpha = np.array([om.alpha_0])
        self._beta = np.array([om.beta_0])
        self._t = 0
        self._history = []
        self.is_initialized = True

    def update(self, x: float) -> BOCPDResult:
        """
        Process a single observation and update the changepoint detector.

        All internal computations use LOG-SPACE to prevent underflow.

        The returned changepoint_probability is P(r_t = 0 | x_{1:t}),
        the canonical Adams & MacKay signal. The alert boolean is
        additionally gated by the warm-up length.

        Args:
            x: New observation value.

        Returns:
            BOCPDResult with changepoint probability, run-length stats,
            alert, growth probability, and confidence.
        """
        if not self.is_initialized:
            self.reset()

        self._t += 1

        # Handle NaN / Inf observations gracefully
        if not np.isfinite(x):
            result = BOCPDResult(
                changepoint_probability=0.0,
                run_length_mean=float(self._t),
                run_length_mode=self._t,
                run_length_entropy=0.0,
                alert=False,
                growth_probability=1.0,
                confidence=min(1.0, self._t / self._confidence_ramp),
            )
            self._history.append(result)
            return result

        # ── Step 1: Log predictive probabilities for each run length ──
        log_pred = self.observation_model.compute_log_predictive(
            x, self._mu, self._kappa, self._alpha, self._beta
        )

        # ── Step 2: Hazard ──
        H = self.config.hazard_rate
        log_H = np.log(H)
        log_1mH = np.log(1.0 - H)

        # ── Step 3: Growth probabilities (log-space) ──
        log_growth = self._log_run_length_dist + log_pred + log_1mH

        # ── Step 4: Changepoint mass (log-space) ──
        # log_cp = log( sum_r  P(r) * pi(x|prior) * H )
        # Since the prior predictive probability (log_pred[0]) and hazard (log_H)
        # do not depend on r, and sum_r P(r) = 1, this simplifies to log_pred[0] + log_H
        log_cp = log_pred[0] + log_H

        # ── Step 5: Concatenate [changepoint, growth] ──
        new_log_dist = np.concatenate([[log_cp], log_growth])

        # ── Step 6: Normalise (log-space) ──
        log_evidence = _logsumexp(new_log_dist)
        new_log_dist -= log_evidence

        # ── Step 7: Update sufficient statistics ──
        new_mu, new_kappa, new_alpha, new_beta = \
            self.observation_model.update_stats(
                x, self._mu, self._kappa, self._alpha, self._beta
            )

        om = self.observation_model
        self._mu = np.concatenate([[om.mu_0], new_mu])
        self._kappa = np.concatenate([[om.kappa_0], new_kappa])
        self._alpha = np.concatenate([[om.alpha_0], new_alpha])
        self._beta = np.concatenate([[om.beta_0], new_beta])

        # ── Step 8: Truncate to bound memory ──
        if len(new_log_dist) > self._max_run_length:
            # Merge tail mass into last kept position
            tail = _logsumexp(new_log_dist[self._max_run_length - 1:])
            new_log_dist = new_log_dist[:self._max_run_length]
            new_log_dist[-1] = tail
            self._mu = self._mu[:self._max_run_length]
            self._kappa = self._kappa[:self._max_run_length]
            self._alpha = self._alpha[:self._max_run_length]
            self._beta = self._beta[:self._max_run_length]

        self._log_run_length_dist = new_log_dist

        # ── Step 9: Compute output statistics ──
        # Convert to probability-space for output (only here, not in
        # the recursion, to preserve numerical stability)
        dist = np.exp(new_log_dist)

        # Canonical changepoint probability: P(r_t = 0)
        cp_prob = float(dist[0])

        run_lengths = np.arange(len(dist))
        rl_mean = float(np.sum(run_lengths * dist))
        rl_mode = int(np.argmax(dist))

        # Run-length entropy
        nonzero = dist > 1e-15
        rl_entropy = float(-np.sum(dist[nonzero] * np.log(dist[nonzero])))

        # Confidence ramp: 0 → 1 over confidence_ramp_length steps
        confidence = min(1.0, self._t / self._confidence_ramp)

        # Alert gating: signal must exceed threshold AND detector must
        # be past the warm-up period
        alert = (cp_prob > self.config.alert_threshold) and \
                (self._t > self._warm_up_length)

        result = BOCPDResult(
            changepoint_probability=cp_prob,
            run_length_mean=rl_mean,
            run_length_mode=rl_mode,
            run_length_entropy=rl_entropy,
            alert=alert,
            growth_probability=1.0 - cp_prob,
            confidence=confidence,
        )

        self._history.append(result)
        return result

    def process_batch(
        self,
        data: np.ndarray,
        reset_first: bool = True,
        warm_up_length: Optional[int] = None,
    ) -> List[BOCPDResult]:
        """
        Process a batch of observations sequentially.

        The first `warm_up_length` observations are used to calibrate
        the NIG prior, then ALL observations (including the warm-up
        window) are processed through the detector. This ensures the
        output length matches the input length exactly.

        Args:
            data: 1D array of observations.
            reset_first: Whether to reset detector state before processing.
            warm_up_length: Override config.warm_up_length for this batch.

        Returns:
            List of BOCPDResult for each timestep.
        """
        wul = warm_up_length if warm_up_length is not None \
            else self._warm_up_length

        if reset_first:
            # Calibrate from the first warm_up_length observations
            cal_end = min(wul, len(data))
            cal_data = np.asarray(data[:cal_end], dtype=np.float64)
            self.reset(calibration_data=cal_data)

        results = []
        for x in data:
            results.append(self.update(float(x)))

        return results

    # ── Convenience accessors ──

    def get_changepoint_probs(self) -> np.ndarray:
        """Get array of changepoint probabilities from history."""
        return np.array([r.changepoint_probability for r in self._history])

    def get_run_length_means(self) -> np.ndarray:
        """Get array of mean run lengths from history."""
        return np.array([r.run_length_mean for r in self._history])

    def get_alerts(self) -> np.ndarray:
        """Get boolean array of alerts from history."""
        return np.array([r.alert for r in self._history])

    def get_summary_features(self) -> np.ndarray:
        """
        Get summary features for each timestep (for LSTM input).

        Returns array of shape (n_timesteps, 4):
            [changepoint_prob, run_length_mean, run_length_mode,
             run_length_entropy]
        """
        if not self._history:
            return np.array([]).reshape(0, 4)

        return np.array([
            [r.changepoint_probability, r.run_length_mean,
             r.run_length_mode, r.run_length_entropy]
            for r in self._history
        ])

    # ── Persistence ──

    def save(self, filepath: Optional[str] = None):
        """Save BOCPD state to disk."""
        filepath = filepath or str(MODEL_DIR / "bocpd.pkl")
        data = {
            "config": self.config,
            "observation_model_params": {
                "mu_0": self.observation_model.mu_0,
                "kappa_0": self.observation_model.kappa_0,
                "alpha_0": self.observation_model.alpha_0,
                "beta_0": self.observation_model.beta_0,
            },
            "log_run_length_dist": self._log_run_length_dist,
            "mu": self._mu,
            "kappa": self._kappa,
            "alpha": self._alpha,
            "beta": self._beta,
            "t": self._t,
            "is_initialized": self.is_initialized,
            "is_calibrated": self.is_calibrated,
        }
        joblib.dump(data, filepath)
        logger.info(f"Saved BOCPD to {filepath}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "BOCPD":
        """Load BOCPD state from disk."""
        filepath = filepath or str(MODEL_DIR / "bocpd.pkl")
        data = joblib.load(filepath)
        obj = cls(config=data["config"])

        # Restore calibrated observation model
        om_params = data["observation_model_params"]
        obj.observation_model = NormalInverseGamma(
            mu_0=om_params["mu_0"],
            kappa_0=om_params["kappa_0"],
            alpha_0=om_params["alpha_0"],
            beta_0=om_params["beta_0"],
        )
        obj.is_calibrated = data.get("is_calibrated", True)

        obj._log_run_length_dist = data["log_run_length_dist"]
        obj._mu = data["mu"]
        obj._kappa = data["kappa"]
        obj._alpha = data["alpha"]
        obj._beta = data["beta"]
        obj._t = data["t"]
        obj.is_initialized = data["is_initialized"]
        logger.info(f"Loaded BOCPD from {filepath}")
        return obj


# ──────────────────────────────────────────────────────────────────────
# Multi-feature BOCPD orchestrator
# ──────────────────────────────────────────────────────────────────────

class MultiFeatureBOCPD:
    """
    Monitors multiple leading features with independent BOCPD detectors,
    then aggregates their changepoint signals.

    Improvements over original:
    - Each feature auto-calibrates its own NIG prior.
    - Consensus-based alerts (multiple features must agree).
    - Weighted changepoint probability (weighted by confidence).
    - Exposes confidence and consensus_ratio for LSTM.
    """

    def __init__(self, feature_names: Optional[List[str]] = None,
                 config=None):
        self.config = config or bocpd_cfg
        self.feature_names = feature_names or self.config.leading_features
        self.detectors = {
            name: BOCPD(self.config) for name in self.feature_names
        }

    def reset(self):
        """Reset all detectors."""
        for det in self.detectors.values():
            det.reset()

    def update(self, feature_values: Dict[str, float]) -> Dict:
        """
        Update all detectors with new feature values.

        Args:
            feature_values: Dict mapping feature name → value.

        Returns:
            Dict with per-feature results and aggregated metrics.
        """
        results = {}
        cp_probs = []
        confidences = []

        for name, detector in self.detectors.items():
            if name in feature_values:
                result = detector.update(feature_values[name])
                results[name] = result
                cp_probs.append(result.changepoint_probability)
                confidences.append(result.confidence)

        if cp_probs:
            cp_arr = np.array(cp_probs)
            conf_arr = np.array(confidences)

            max_cp = float(np.max(cp_arr))
            mean_cp = float(np.mean(cp_arr))

            # Weighted by confidence
            conf_sum = conf_arr.sum()
            if conf_sum > 0:
                weighted_cp = float(np.sum(cp_arr * conf_arr) / conf_sum)
            else:
                weighted_cp = mean_cp

            n_alerting = int(np.sum(cp_arr > self.config.alert_threshold))
            consensus_ratio = n_alerting / len(cp_arr)
            avg_confidence = float(np.mean(conf_arr))
        else:
            max_cp = mean_cp = weighted_cp = 0.0
            n_alerting = 0
            consensus_ratio = 0.0
            avg_confidence = 0.0

        # Consensus alert: enough features must agree AND detector must
        # be past warm-up
        consensus_alert = (
            n_alerting >= self.config.min_consensus_features and
            avg_confidence >= 0.5  # At least halfway through warm-up
        )

        return {
            "per_feature": results,
            "max_changepoint_prob": max_cp,
            "mean_changepoint_prob": mean_cp,
            "weighted_changepoint_prob": weighted_cp,
            "any_alert": consensus_alert,
            "n_alerts": n_alerting,
            "consensus_ratio": consensus_ratio,
            "avg_confidence": avg_confidence,
        }

    def process_batch(
        self,
        features_df: pd.DataFrame,
        reset_first: bool = True,
    ) -> pd.DataFrame:
        """
        Process a DataFrame of features through all detectors.

        Each detector is calibrated independently from its own feature
        column's warm-up window. Output length matches input length.

        Args:
            features_df: DataFrame with columns matching self.feature_names.
            reset_first: Whether to reset before processing.

        Returns:
            DataFrame with BOCPD output columns aligned with input index.
            Columns:
              bocpd_cp_prob     — weighted changepoint probability
              bocpd_rl_mean     — mean run length across features
              bocpd_rl_mode     — median mode across features
              bocpd_rl_entropy  — mean entropy across features
              bocpd_alert       — consensus-based alert
              bocpd_confidence  — average confidence across features
        """
        available_features = [
            f for f in self.feature_names if f in features_df.columns
        ]
        if not available_features:
            logger.warning(
                "No BOCPD leading features found in input DataFrame!"
            )
            return pd.DataFrame(index=features_df.index)

        # ── Calibrate each detector from its own warm-up window ──
        if reset_first:
            wul = self.config.warm_up_length
            for fname in available_features:
                col = features_df[fname].values
                cal_end = min(wul, len(col))
                cal_data = np.asarray(col[:cal_end], dtype=np.float64)
                self.detectors[fname].reset(calibration_data=cal_data)

        # ── Process row-by-row ──
        cp_probs = []
        rl_means = []
        rl_modes = []
        rl_entropies = []
        alerts = []
        confidences = []

        for idx in range(len(features_df)):
            row = features_df.iloc[idx]
            feature_vals = {
                f: float(row[f])
                for f in available_features
                if pd.notna(row[f])
            }
            result = self.update(feature_vals)

            cp_probs.append(result["weighted_changepoint_prob"])
            alerts.append(result["any_alert"])
            confidences.append(result["avg_confidence"])

            # Aggregate run-length stats across detectors
            per_feat = result["per_feature"]
            if per_feat:
                rl_means.append(np.mean(
                    [r.run_length_mean for r in per_feat.values()]
                ))
                rl_modes.append(np.median(
                    [r.run_length_mode for r in per_feat.values()]
                ))
                rl_entropies.append(np.mean(
                    [r.run_length_entropy for r in per_feat.values()]
                ))
            else:
                rl_means.append(0.0)
                rl_modes.append(0.0)
                rl_entropies.append(0.0)

        output = pd.DataFrame({
            "bocpd_cp_prob": cp_probs,
            "bocpd_rl_mean": rl_means,
            "bocpd_rl_mode": rl_modes,
            "bocpd_rl_entropy": rl_entropies,
            "bocpd_alert": alerts,
            "bocpd_confidence": confidences,
        }, index=features_df.index)

        n_alerts = sum(alerts)
        logger.info(
            f"BOCPD processed {len(features_df):,} samples, "
            f"{n_alerts} alerts fired "
            f"({n_alerts / len(features_df) * 100:.1f}%)"
        )

        return output

    # ── Persistence ──

    def save(self, filepath: Optional[str] = None):
        """Save all detectors."""
        filepath = filepath or str(MODEL_DIR / "multi_bocpd.pkl")
        data = {
            "config": self.config,
            "feature_names": self.feature_names,
            "detectors": {name: {
                "observation_model_params": {
                    "mu_0": det.observation_model.mu_0,
                    "kappa_0": det.observation_model.kappa_0,
                    "alpha_0": det.observation_model.alpha_0,
                    "beta_0": det.observation_model.beta_0,
                },
                "log_run_length_dist": det._log_run_length_dist,
                "mu": det._mu, "kappa": det._kappa,
                "alpha": det._alpha, "beta": det._beta,
                "t": det._t, "is_initialized": det.is_initialized,
                "is_calibrated": det.is_calibrated,
            } for name, det in self.detectors.items()},
        }
        joblib.dump(data, filepath)
        logger.info(f"Saved MultiFeatureBOCPD to {filepath}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "MultiFeatureBOCPD":
        """Load all detectors."""
        filepath = filepath or str(MODEL_DIR / "multi_bocpd.pkl")
        data = joblib.load(filepath)
        obj = cls(feature_names=data["feature_names"], config=data["config"])
        for name, state in data["detectors"].items():
            det = obj.detectors[name]
            om = state["observation_model_params"]
            det.observation_model = NormalInverseGamma(
                mu_0=om["mu_0"], kappa_0=om["kappa_0"],
                alpha_0=om["alpha_0"], beta_0=om["beta_0"],
            )
            det.is_calibrated = state.get("is_calibrated", True)
            det._log_run_length_dist = state["log_run_length_dist"]
            det._mu = state["mu"]
            det._kappa = state["kappa"]
            det._alpha = state["alpha"]
            det._beta = state["beta"]
            det._t = state["t"]
            det.is_initialized = state["is_initialized"]
        logger.info(f"Loaded MultiFeatureBOCPD from {filepath}")
        return obj


# ──────────────────────────────────────────────────────────────────────
# Utility: numerically stable logsumexp
# ──────────────────────────────────────────────────────────────────────

def _logsumexp(log_x: np.ndarray) -> float:
    """
    Compute log(sum(exp(log_x))) in a numerically stable way.

    Equivalent to scipy.special.logsumexp but inlined to avoid the
    import overhead on every call.
    """
    if len(log_x) == 0:
        return -np.inf
    c = np.max(log_x)
    if not np.isfinite(c):
        return -np.inf
    return float(c + np.log(np.sum(np.exp(log_x - c))))
