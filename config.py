"""
config.py — Central configuration for the High-Precision Market Regime Detection System.

All hyperparameters, file paths, and model settings live here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"

# Create directories
for d in [DATA_DIR, MODEL_DIR, PLOT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Raw data file
RAW_DATA_FILE = DATA_DIR / "XAUUSD_M5_201809211615_202604301140.csv"


# ──────────────────────────────────────────────────────────────────────
# DATA CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # CSV format (MetaTrader 5 export)
    separator: str = "\t"
    date_col: str = "<DATE>"
    time_col: str = "<TIME>"
    open_col: str = "<OPEN>"
    high_col: str = "<HIGH>"
    low_col: str = "<LOW>"
    close_col: str = "<CLOSE>"
    tickvol_col: str = "<TICKVOL>"
    vol_col: str = "<VOL>"
    spread_col: str = "<SPREAD>"

    # Resampling timeframes (in minutes)
    base_timeframe: int = 5       # M5
    timeframes: Dict[str, int] = field(default_factory=lambda: {
        "M5": 5,
        "M15": 15,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
    })

    # Data cleaning
    min_tickvol: int = 5          # Filter bars with very low tick volume
    max_spread: int = 100         # Filter bars with abnormally wide spread


# ──────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """Configuration for multi-timeframe feature extraction."""

    # Return lookback periods
    return_periods: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 50, 100])

    # Rolling windows for statistics
    stat_windows: List[int] = field(default_factory=lambda: [20, 50, 100])

    # Volatility
    atr_periods: List[int] = field(default_factory=lambda: [14, 50])
    bb_period: int = 20
    bb_std: float = 2.0

    # Trend / Momentum
    ema_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    rsi_periods: List[int] = field(default_factory=lambda: [14, 50])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14

    # Volume
    volume_ma_periods: List[int] = field(default_factory=lambda: [20, 50])

    # Hurst exponent
    hurst_window: int = 100

    # Normalization
    zscore_window: int = 252 * 12 * 5  # ~1 year of M5 bars (expanding window used in practice)

    # PCA
    pca_n_components: int = 12     # Number of PCA components for HMM input
    pca_variance_threshold: float = 0.95  # Alternative: keep components explaining this much variance


# ──────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATIONS
# ──────────────────────────────────────────────────────────────────────

# ── Regime definitions ──
REGIME_NAMES = {
    0: "Low-Vol Bullish",
    1: "High-Vol Bullish",
    2: "Ranging",
    3: "Low-Vol Bearish",
    4: "High-Vol Bearish/Crisis",
}

NUM_REGIMES = len(REGIME_NAMES)

REGIME_COLORS = {
    0: "#2ecc71",   # Green — Low-Vol Bullish
    1: "#f39c12",   # Orange — High-Vol Bullish
    2: "#3498db",   # Blue — Ranging
    3: "#e74c3c",   # Red — Low-Vol Bearish
    4: "#8e44ad",   # Purple — High-Vol Bearish/Crisis
}


@dataclass
class HMMConfig:
    """Layer 1: Bayesian Gaussian HMM configuration."""
    n_states: int = NUM_REGIMES
    covariance_type: str = "full"      # "full", "diag", "spherical", "tied"
    n_iter: int = 200                   # Max EM iterations
    tol: float = 1e-4                   # Convergence tolerance
    n_init: int = 10                    # Number of random restarts (pick best)
    random_state: int = 42
    min_covar: float = 1e-3            # Floor for covariance to prevent singularity


@dataclass
class BOCPDConfig:
    """Layer 2: Online Bayesian Changepoint Detection configuration."""
    hazard_rate: float = 1 / 50        # Expected run length ~50 bars (~4 hours on M5)
    alert_threshold: float = 0.5       # Conservative default — canonical P(r_t=0) is peaky
    observation_model: str = "gaussian"  # "gaussian" or "student_t"

    # Warm-up & calibration (priors are auto-calibrated from data, not hardcoded)
    warm_up_length: int = 100          # Bars for prior calibration (~8 hours of M5 data)
    confidence_ramp_length: int = 200  # Bars over which confidence ramps 0 → 1
    min_consensus_features: int = 2    # Min features agreeing for a consensus alert

    # Leading features to monitor
    leading_features: List[str] = field(default_factory=lambda: [
        "vol_term_structure_slope",
        "return_autocorr_change",
        "drawdown_velocity",
        "bb_width_roc",
        "hurst_exponent",
    ])


@dataclass
class LSTMConfig:
    """Layer 3: LSTM Meta-Classifier configuration (CPU-optimized)."""
    # Architecture
    input_size: int = 0                # Set dynamically based on features
    hidden_size_1: int = 96
    hidden_size_2: int = 48
    dense_size: int = 32
    num_classes: int = NUM_REGIMES + 1  # 5 regimes + 1 "transition-in-progress"
    dropout: float = 0.25
    sequence_length: int = 50          # Lookback window in bars

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4         # AdamW regularization
    max_epochs: int = 100
    patience: int = 15                 # Early stopping patience
    focal_loss_gamma: float = 2.0      # Focal loss focusing parameter
    focal_loss_alpha: float = 0.25     # Focal loss class balancing

    # CPU optimization
    num_workers: int = 0               # DataLoader workers (0 for Windows)
    num_threads: int = 4               # torch threads (match CPU cores)

    # Checkpointing
    save_best_only: bool = True


# ──────────────────────────────────────────────────────────────────────
# TRAINING / VALIDATION CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Walk-forward training pipeline configuration."""
    # Walk-forward splits
    n_folds: int = 5
    test_size_bars: int = 12 * 24 * 90  # ~90 days of M5 bars (12 bars/hr * 24hr * 90 days)
    purge_gap_bars: int = 12 * 24 * 5   # 5-day purge gap between train/test
    min_train_size_bars: int = 12 * 24 * 365  # Minimum 1 year of training data

    # Retraining
    hmm_retrain_every_n_bars: int = 12 * 24 * 30  # Monthly HMM retraining
    lstm_finetune_epochs: int = 20     # Fine-tune (not full retrain) on subsequent folds

    # Random seed
    random_state: int = 42


# ──────────────────────────────────────────────────────────────────────
# MT5 CONNECTOR CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MT5Config:
    """MetaTrader 5 live connector configuration."""
    symbol: str = "XAUUSD"
    timeframe_mt5: str = "M5"          # MT5 timeframe constant name
    lookback_bars: int = 500           # Bars to fetch for feature computation
    signal_file: str = str(PROJECT_ROOT / "outputs" / "mt5_signal.json")
    log_file: str = str(LOG_DIR / "mt5_live.log")
    poll_interval_seconds: int = 10    # How often to check for new bar


# ──────────────────────────────────────────────────────────────────────
# INSTANTIATE DEFAULT CONFIGS
# ──────────────────────────────────────────────────────────────────────

data_cfg = DataConfig()
feature_cfg = FeatureConfig()
hmm_cfg = HMMConfig()
bocpd_cfg = BOCPDConfig()
lstm_cfg = LSTMConfig()
training_cfg = TrainingConfig()
mt5_cfg = MT5Config()
