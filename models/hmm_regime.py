"""
hmm_regime.py — Layer 1: Bayesian Gaussian Hidden Markov Model for Regime Detection.

This module fits a Gaussian HMM to PCA-reduced feature vectors and identifies
the current market regime state (one of 5 states).

Key capabilities:
  - Fit HMM with multiple random restarts (picks best log-likelihood)
  - Decode most likely state sequence (Viterbi algorithm)
  - Compute posterior state probabilities per timestep
  - Post-hoc regime labeling (map HMM states to semantic regime labels)
  - Serialization (save/load)
"""

import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple, Dict
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import hmm_cfg, feature_cfg, REGIME_NAMES, NUM_REGIMES, MODEL_DIR
from utils import logger


class RegimeHMM:
    """
    Layer 1: Bayesian Gaussian HMM for market regime detection.

    Pipeline:
        Raw Features → StandardScaler → PCA → GaussianHMM → Regime Labels + Posteriors
    """

    def __init__(self, config=None):
        self.config = config or hmm_cfg
        self.pca = PCA(n_components=feature_cfg.pca_n_components)
        self.scaler = StandardScaler()
        self.hmm = None
        self.is_fitted = False
        self.state_mapping = None  # Maps HMM internal states → semantic regime labels
        self._feature_names = None

    def fit(
        self,
        features: pd.DataFrame,
        price_series: Optional[pd.Series] = None,
    ) -> "RegimeHMM":
        """
        Fit the HMM pipeline on training features.

        Args:
            features: DataFrame of normalized features (output of feature_engine).
            price_series: Close prices aligned with features (used for state labeling).

        Returns:
            self
        """
        logger.info(f"Fitting RegimeHMM on {len(features):,} samples, "
                    f"{features.shape[1]} features...")
        self._feature_names = features.columns.tolist()

        # Step 1: Scale features
        X_scaled = self.scaler.fit_transform(features.values)

        # Step 2: PCA dimensionality reduction
        n_components = min(feature_cfg.pca_n_components, X_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)

        explained_var = self.pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA: {n_components} components explain "
                    f"{explained_var:.1%} of variance")

        # Step 3: Fit HMM with multiple restarts
        best_hmm = None
        best_score = -np.inf

        for i in range(self.config.n_init):
            try:
                hmm = GaussianHMM(
                    n_components=self.config.n_states,
                    covariance_type=self.config.covariance_type,
                    n_iter=self.config.n_iter,
                    tol=self.config.tol,
                    random_state=self.config.random_state + i,
                    min_covar=self.config.min_covar,
                    verbose=False,
                )
                hmm.fit(X_pca)
                score = hmm.score(X_pca)

                if score > best_score:
                    best_score = score
                    best_hmm = hmm

            except Exception as e:
                logger.warning(f"  HMM restart {i+1}/{self.config.n_init} failed: {e}")
                continue

        if best_hmm is None:
            raise RuntimeError("All HMM fitting attempts failed!")

        self.hmm = best_hmm
        self.is_fitted = True
        logger.info(f"  Best HMM log-likelihood: {best_score:.2f} "
                    f"(from {self.config.n_init} restarts)")

        # Step 4: Auto-label states using return/volatility characteristics
        if price_series is not None:
            self._auto_label_states(features, price_series)

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict regime labels for new data using Viterbi decoding.

        Returns:
            Array of regime labels (0-4).
        """
        self._check_fitted()
        X_pca = self._transform(features)
        raw_states = self.hmm.predict(X_pca)

        # Apply state mapping if available
        if self.state_mapping is not None:
            return np.array([self.state_mapping[s] for s in raw_states])
        return raw_states

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Compute posterior state probabilities for each timestep.

        Returns:
            Array of shape (n_samples, n_states) with posterior probabilities.
        """
        self._check_fitted()
        X_pca = self._transform(features)
        raw_posteriors = self.hmm.predict_proba(X_pca)

        # Rearrange columns if state mapping exists
        if self.state_mapping is not None:
            mapped_posteriors = np.zeros_like(raw_posteriors)
            for raw_state, mapped_state in self.state_mapping.items():
                mapped_posteriors[:, mapped_state] = raw_posteriors[:, raw_state]
            return mapped_posteriors

        return raw_posteriors

    def score(self, features: pd.DataFrame) -> float:
        """Compute log-likelihood of the data under the model."""
        self._check_fitted()
        X_pca = self._transform(features)
        return self.hmm.score(X_pca)

    def get_transition_matrix(self) -> np.ndarray:
        """Get the state transition probability matrix."""
        self._check_fitted()
        transmat = self.hmm.transmat_.copy()

        # Rearrange if state mapping exists
        if self.state_mapping is not None:
            n = self.config.n_states
            mapped = np.zeros((n, n))
            for i_raw, i_mapped in self.state_mapping.items():
                for j_raw, j_mapped in self.state_mapping.items():
                    mapped[i_mapped, j_mapped] = transmat[i_raw, j_raw]
            return mapped

        return transmat

    def _transform(self, features: pd.DataFrame) -> np.ndarray:
        """Apply scaler + PCA transform."""
        X_scaled = self.scaler.transform(features.values)
        return self.pca.transform(X_scaled)

    def _auto_label_states(
        self,
        features: pd.DataFrame,
        price_series: pd.Series,
    ):
        """
        Automatically map HMM internal states to semantic regime labels
        based on return and volatility characteristics.

        Mapping logic:
          - Compute mean return and mean volatility per HMM state
          - Sort and assign:
            * Highest vol + negative return → High-Vol Bearish/Crisis (4)
            * Highest vol + positive return → High-Vol Bullish (1)
            * Lowest vol + positive return → Low-Vol Bullish (0)
            * Lowest vol + negative return → Low-Vol Bearish (3)
            * Middle vol + near-zero return → Ranging (2)
        """
        X_pca = self._transform(features)
        raw_states = self.hmm.predict(X_pca)

        # Align price series with features
        aligned_prices = price_series.reindex(features.index)
        returns = aligned_prices.pct_change().fillna(0)

        # Compute per-state statistics
        state_stats = {}
        for state in range(self.config.n_states):
            mask = raw_states == state
            if mask.sum() > 0:
                state_returns = returns.values[mask]
                state_stats[state] = {
                    "mean_return": np.mean(state_returns),
                    "volatility": np.std(state_returns),
                    "count": mask.sum(),
                }
            else:
                state_stats[state] = {
                    "mean_return": 0,
                    "volatility": 0,
                    "count": 0,
                }

        # Sort states by volatility
        sorted_by_vol = sorted(state_stats.keys(),
                               key=lambda s: state_stats[s]["volatility"])

        # Assign mapping
        mapping = {}
        assigned = set()

        # Highest volatility states
        high_vol_states = sorted_by_vol[-2:] if len(sorted_by_vol) >= 2 else sorted_by_vol[-1:]
        low_vol_states = sorted_by_vol[:2] if len(sorted_by_vol) >= 2 else sorted_by_vol[:1]
        mid_vol_states = [s for s in sorted_by_vol if s not in high_vol_states and s not in low_vol_states]

        # High-vol: bearish (4) vs bullish (1)
        for s in high_vol_states:
            if state_stats[s]["mean_return"] < 0 and 4 not in assigned:
                mapping[s] = 4  # High-Vol Bearish/Crisis
                assigned.add(4)
            elif 1 not in assigned:
                mapping[s] = 1  # High-Vol Bullish
                assigned.add(1)
            else:
                # Fallback
                for label in [4, 1]:
                    if label not in assigned:
                        mapping[s] = label
                        assigned.add(label)
                        break

        # Low-vol: bullish (0) vs bearish (3)
        for s in low_vol_states:
            if s in mapping:
                continue
            if state_stats[s]["mean_return"] >= 0 and 0 not in assigned:
                mapping[s] = 0  # Low-Vol Bullish
                assigned.add(0)
            elif 3 not in assigned:
                mapping[s] = 3  # Low-Vol Bearish
                assigned.add(3)
            else:
                for label in [0, 3]:
                    if label not in assigned:
                        mapping[s] = label
                        assigned.add(label)
                        break

        # Mid-vol → Ranging (2)
        for s in mid_vol_states:
            if s not in mapping:
                if 2 not in assigned:
                    mapping[s] = 2
                    assigned.add(2)
                else:
                    # Assign any remaining label
                    for label in range(NUM_REGIMES):
                        if label not in assigned:
                            mapping[s] = label
                            assigned.add(label)
                            break

        # Fill any unassigned states
        for s in range(self.config.n_states):
            if s not in mapping:
                for label in range(NUM_REGIMES):
                    if label not in assigned:
                        mapping[s] = label
                        assigned.add(label)
                        break

        self.state_mapping = mapping

        logger.info("  State mapping (HMM internal → semantic):")
        for raw, mapped in sorted(mapping.items()):
            stats = state_stats[raw]
            logger.info(
                f"    HMM State {raw} → {REGIME_NAMES[mapped]} "
                f"(mean_ret={stats['mean_return']:.6f}, "
                f"vol={stats['volatility']:.6f}, "
                f"n={stats['count']:,})"
            )

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("RegimeHMM is not fitted. Call fit() first.")

    def save(self, filepath: Optional[str] = None):
        """Save model to disk."""
        filepath = filepath or str(MODEL_DIR / "regime_hmm.pkl")
        data = {
            "hmm": self.hmm,
            "pca": self.pca,
            "scaler": self.scaler,
            "config": self.config,
            "state_mapping": self.state_mapping,
            "feature_names": self._feature_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(data, filepath)
        logger.info(f"Saved RegimeHMM to {filepath}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "RegimeHMM":
        """Load model from disk."""
        filepath = filepath or str(MODEL_DIR / "regime_hmm.pkl")
        data = joblib.load(filepath)
        obj = cls(config=data["config"])
        obj.hmm = data["hmm"]
        obj.pca = data["pca"]
        obj.scaler = data["scaler"]
        obj.state_mapping = data["state_mapping"]
        obj._feature_names = data["feature_names"]
        obj.is_fitted = data["is_fitted"]
        logger.info(f"Loaded RegimeHMM from {filepath}")
        return obj

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model."""
        self._check_fitted()
        lines = [
            "=" * 60,
            "RegimeHMM Summary",
            "=" * 60,
            f"States: {self.config.n_states}",
            f"PCA components: {self.pca.n_components_}",
            f"Explained variance: {self.pca.explained_variance_ratio_.sum():.1%}",
            f"Covariance type: {self.config.covariance_type}",
            "",
            "Transition Matrix:",
        ]

        transmat = self.get_transition_matrix()
        header = "       " + "  ".join(f"  S{i}" for i in range(self.config.n_states))
        lines.append(header)
        for i in range(self.config.n_states):
            row = f"  S{i}:  " + "  ".join(f"{transmat[i,j]:.2f}" for j in range(self.config.n_states))
            name = REGIME_NAMES.get(i, "")
            lines.append(f"{row}  ({name})")

        lines.append("")
        lines.append("Stationary Distribution:")
        try:
            # Compute stationary distribution from transition matrix
            eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, idx])
            stationary = stationary / stationary.sum()
            for i in range(self.config.n_states):
                lines.append(f"  {REGIME_NAMES.get(i, f'State {i}')}: {stationary[i]:.1%}")
        except Exception:
            lines.append("  (Could not compute stationary distribution)")

        return "\n".join(lines)
