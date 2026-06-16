"""
utils.py — Shared utilities for the High-Precision Market Regime Detection System.

Includes: data loading, logging, visualization helpers, and common transforms.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from config import (
    RAW_DATA_FILE, data_cfg, LOG_DIR, PLOT_DIR,
    REGIME_NAMES, REGIME_COLORS, NUM_REGIMES,
)


# ──────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Create a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


logger = setup_logger("regime", str(LOG_DIR / "regime_system.log"))


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────

def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw M5 OHLCV data from MetaTrader CSV export.

    Returns a DataFrame indexed by datetime with columns:
        open, high, low, close, tickvol, vol, spread
    """
    filepath = filepath or RAW_DATA_FILE

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Loading data from {filepath.name}...")

    df = pd.read_csv(
        filepath,
        sep=data_cfg.separator,
        dtype={
            data_cfg.open_col: np.float64,
            data_cfg.high_col: np.float64,
            data_cfg.low_col: np.float64,
            data_cfg.close_col: np.float64,
            data_cfg.tickvol_col: np.int64,
            data_cfg.vol_col: np.int64,
            data_cfg.spread_col: np.int64,
        },
    )

    # Parse datetime
    df["datetime"] = pd.to_datetime(
        df[data_cfg.date_col] + " " + df[data_cfg.time_col],
        format="%Y.%m.%d %H:%M:%S",
    )
    df.set_index("datetime", inplace=True)

    # Rename columns to clean names
    rename_map = {
        data_cfg.open_col: "open",
        data_cfg.high_col: "high",
        data_cfg.low_col: "low",
        data_cfg.close_col: "close",
        data_cfg.tickvol_col: "tickvol",
        data_cfg.vol_col: "vol",
        data_cfg.spread_col: "spread",
    }
    df.rename(columns=rename_map, inplace=True)
    df = df[["open", "high", "low", "close", "tickvol", "vol", "spread"]]

    # Sort by datetime (should already be sorted, but ensure)
    df.sort_index(inplace=True)

    # Remove exact duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    logger.info(
        f"Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]} "
        f"({(df.index[-1] - df.index[0]).days} days)"
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data: remove invalid bars, handle weekends/gaps.

    Returns cleaned DataFrame.
    """
    initial_len = len(df)

    # Remove bars with zero or negative prices
    price_cols = ["open", "high", "low", "close"]
    mask_positive = (df[price_cols] > 0).all(axis=1)
    df = df[mask_positive]

    # Remove bars where high < low (data error)
    mask_hl = df["high"] >= df["low"]
    df = df[mask_hl]

    # Remove bars with very low tick volume (likely illiquid periods)
    mask_vol = df["tickvol"] >= data_cfg.min_tickvol
    df = df[mask_vol]

    # Remove bars with abnormally wide spread
    mask_spread = df["spread"] <= data_cfg.max_spread
    df = df[mask_spread]

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Cleaned data: removed {removed:,} invalid bars ({removed/initial_len*100:.2f}%)")

    return df


def resample_ohlcv(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """
    Resample M5 data to a higher timeframe (e.g., H1, H4, D1).

    Uses proper OHLCV aggregation rules.
    """
    rule = f"{timeframe_minutes}min"

    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tickvol": "sum",
        "vol": "sum",
        "spread": "mean",
    }).dropna(subset=["open"])

    return resampled


# ──────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────

def plot_regime_overlay(
    price_series: pd.Series,
    regime_labels: np.ndarray,
    title: str = "Market Regime Detection — XAUUSD",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 8),
    show: bool = True,
) -> plt.Figure:
    """
    Plot price with colored background for each regime state.

    Args:
        price_series: Close prices indexed by datetime.
        regime_labels: Array of regime labels (0-4) aligned with price_series.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.
        show: Whether to display the plot.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    # Plot price line
    ax.plot(price_series.index, price_series.values,
            color="white", linewidth=0.5, alpha=0.9)

    # Color background by regime
    if len(regime_labels) == len(price_series):
        for regime_id, color in REGIME_COLORS.items():
            mask = regime_labels == regime_id
            if mask.any():
                ax.fill_between(
                    price_series.index, price_series.min() * 0.99,
                    price_series.max() * 1.01,
                    where=mask, alpha=0.25, color=color, linewidth=0,
                )

    # Legend
    legend_patches = [
        Patch(facecolor=color, alpha=0.4, label=f"State {k}: {name}")
        for k, (name, color) in enumerate(zip(REGIME_NAMES.values(), REGIME_COLORS.values()))
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              fontsize=9, facecolor="#1a1a2e", edgecolor="white",
              labelcolor="white")

    # Formatting
    ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=15)
    ax.set_xlabel("Date", fontsize=12, color="white")
    ax.set_ylabel("Price (USD)", fontsize=12, color="white")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.15, color="white")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Saved regime overlay plot to {save_path}")

    if show:
        plt.show()

    return fig


def plot_transition_probabilities(
    dates: pd.DatetimeIndex,
    transition_probs: np.ndarray,
    changepoint_probs: Optional[np.ndarray] = None,
    title: str = "Regime Transition Probabilities",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 6),
    show: bool = True,
) -> plt.Figure:
    """
    Plot regime posterior probabilities and optional BOCPD changepoint probability.
    """
    fig, axes = plt.subplots(2 if changepoint_probs is not None else 1, 1,
                             figsize=figsize, facecolor="#1a1a2e", sharex=True)

    if changepoint_probs is None:
        axes = [axes]

    # Regime posterior probabilities (stacked area)
    ax1 = axes[0]
    ax1.set_facecolor("#16213e")

    if transition_probs.ndim == 2:
        bottom = np.zeros(len(dates))
        for i in range(transition_probs.shape[1]):
            color = REGIME_COLORS.get(i, "#ffffff")
            label = REGIME_NAMES.get(i, f"State {i}")
            ax1.fill_between(dates, bottom, bottom + transition_probs[:, i],
                             alpha=0.7, color=color, label=label)
            bottom += transition_probs[:, i]

    ax1.set_ylabel("Posterior Probability", fontsize=11, color="white")
    ax1.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e",
               edgecolor="white", labelcolor="white")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.15, color="white")

    # BOCPD changepoint probability
    if changepoint_probs is not None and len(axes) > 1:
        ax2 = axes[1]
        ax2.set_facecolor("#16213e")
        ax2.plot(dates, changepoint_probs, color="#e74c3c", linewidth=0.8, alpha=0.9)
        ax2.axhline(y=0.5, color="#f39c12", linestyle="--", alpha=0.6, label="Alert Threshold")
        ax2.set_ylabel("Changepoint Prob", fontsize=11, color="white")
        ax2.set_xlabel("Date", fontsize=11, color="white")
        ax2.tick_params(colors="white")
        ax2.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e",
                   edgecolor="white", labelcolor="white")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.15, color="white")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Saved transition probabilities plot to {save_path}")

    if show:
        plt.show()

    return fig


# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def expanding_zscore(series: pd.Series, min_periods: int = 100) -> pd.Series:
    """
    Compute expanding-window Z-score (no look-ahead bias).

    Uses all data up to each point to compute mean and std.
    """
    expanding_mean = series.expanding(min_periods=min_periods).mean()
    expanding_std = series.expanding(min_periods=min_periods).std()
    return (series - expanding_mean) / (expanding_std + 1e-10)


def rolling_zscore(series: pd.Series, window: int = 100) -> pd.Series:
    """Compute rolling-window Z-score."""
    rolling_mean = series.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = series.rolling(window=window, min_periods=window // 2).std()
    return (series - rolling_mean) / (rolling_std + 1e-10)


def compute_drawdown(prices: pd.Series) -> pd.DataFrame:
    """
    Compute drawdown series from price.

    Returns DataFrame with columns: drawdown, drawdown_pct, drawdown_duration
    """
    running_max = prices.expanding().max()
    dd = prices - running_max
    dd_pct = dd / running_max

    # Drawdown duration (bars since last new high)
    new_high = prices >= running_max
    duration = (~new_high).groupby(new_high.cumsum()).cumsum()

    return pd.DataFrame({
        "drawdown": dd,
        "drawdown_pct": dd_pct,
        "drawdown_duration": duration,
    }, index=prices.index)
