"""
feature_engine.py — Multi-Timeframe Feature Extraction Pipeline.

Computes 60+ features across M5, H1, and H4/D1 timeframes for regime detection.
All features are designed to be look-ahead-bias-free.

Feature Groups:
  1. Return-Based (12 features)
  2. Volatility-Based (15 features)
  3. Trend/Momentum (12 features)
  4. Volume/Microstructure (8 features)
  5. Cross-Timeframe (8 features)
  6. Regime-Predictive Leading (5+ features)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from config import feature_cfg, data_cfg
from utils import (
    resample_ohlcv, expanding_zscore, rolling_zscore,
    compute_drawdown, logger,
)


# ══════════════════════════════════════════════════════════════════════
# CORE INDICATOR FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(H-L, |H-Cprev|, |L-Cprev|)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range."""
    tr = _true_range(high, low, close)
    return tr.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Average Directional Index, returns ADX, +DI, -DI."""
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr = _true_range(high, low, close)
    atr_val = tr.ewm(span=period, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr_val + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr_val + 1e-10)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_val = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame({
        "adx": adx_val,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }, index=close.index)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD, Signal Line, Histogram."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    }, index=close.index)


def _bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: upper, middle, lower, width, %B."""
    sma = _sma(close, period)
    rolling_std = close.rolling(window=period, min_periods=period).std()

    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    width = (upper - lower) / (sma + 1e-10)
    pct_b = (close - lower) / (upper - lower + 1e-10)

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": sma,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct_b": pct_b,
    }, index=close.index)


def _hurst_exponent(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Rolling Hurst exponent via rescaled range (R/S) method.

    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    def _compute_hurst(data):
        if len(data) < 20 or np.std(data) < 1e-10:
            return 0.5
        n = len(data)
        max_k = min(n // 2, 50)
        if max_k < 4:
            return 0.5

        rs_values = []
        sizes = []
        for k in [max_k // 4, max_k // 2, max_k]:
            if k < 4:
                continue
            num_chunks = n // k
            rs_chunk = []
            for i in range(num_chunks):
                chunk = data[i * k:(i + 1) * k]
                mean_chunk = np.mean(chunk)
                deviate = np.cumsum(chunk - mean_chunk)
                r = np.max(deviate) - np.min(deviate)
                s = np.std(chunk, ddof=1)
                if s > 1e-10:
                    rs_chunk.append(r / s)
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
                sizes.append(k)

        if len(rs_values) < 2:
            return 0.5

        log_rs = np.log(rs_values)
        log_n = np.log(sizes)
        try:
            slope, _ = np.polyfit(log_n, log_rs, 1)
            return np.clip(slope, 0.0, 1.0)
        except (np.linalg.LinAlgError, ValueError):
            return 0.5

    return series.rolling(window=window, min_periods=window // 2).apply(
        _compute_hurst, raw=True
    )


def _parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson volatility estimator (uses high/low — more efficient than close-to-close)."""
    log_hl_ratio = np.log(high / (low + 1e-10))
    return np.sqrt(
        (1 / (4 * np.log(2))) * (log_hl_ratio ** 2).rolling(window=window, min_periods=window // 2).mean()
    )


def _garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    window: int = 20
) -> pd.Series:
    """Garman-Klass volatility estimator (uses OHLC — most efficient)."""
    log_hl = np.log(high / (low + 1e-10))
    log_co = np.log(close / (open_ + 1e-10))
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return np.sqrt(gk.rolling(window=window, min_periods=window // 2).mean().clip(lower=0))


# ══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — BY GROUP
# ══════════════════════════════════════════════════════════════════════

def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 1: Return-Based Features (12 features).
    """
    close = df["close"]
    features = pd.DataFrame(index=df.index)

    # Log returns at multiple lookbacks
    for period in feature_cfg.return_periods:
        features[f"log_return_{period}"] = np.log(close / close.shift(period).replace(0, np.nan))

    # Rolling skewness and kurtosis of returns
    log_ret_1 = features["log_return_1"]
    for window in [20, 50]:
        features[f"return_skew_{window}"] = log_ret_1.rolling(window=window, min_periods=window // 2).skew()
        features[f"return_kurtosis_{window}"] = log_ret_1.rolling(window=window, min_periods=window // 2).kurt()

    # Hurst exponent (rolling)
    features["hurst_exponent"] = _hurst_exponent(log_ret_1, window=feature_cfg.hurst_window)

    return features


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 2: Volatility-Based Features (15 features).
    """
    features = pd.DataFrame(index=df.index)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Realized variance (sum of squared returns)
    log_ret = np.log(c / c.shift(1).replace(0, np.nan))
    for window in [20, 50]:
        features[f"realized_var_{window}"] = (log_ret ** 2).rolling(
            window=window, min_periods=window // 2
        ).sum()

    # Parkinson volatility
    features["parkinson_vol_20"] = _parkinson_volatility(h, l, window=20)
    features["parkinson_vol_50"] = _parkinson_volatility(h, l, window=50)

    # Garman-Klass volatility
    features["garman_klass_vol_20"] = _garman_klass_volatility(o, h, l, c, window=20)

    # ATR at multiple periods
    for period in feature_cfg.atr_periods:
        features[f"atr_{period}"] = _atr(h, l, c, period)

    # ATR ratio (short/long) — trend strength proxy
    if len(feature_cfg.atr_periods) >= 2:
        short_atr = features[f"atr_{feature_cfg.atr_periods[0]}"]
        long_atr = features[f"atr_{feature_cfg.atr_periods[1]}"]
        features["atr_ratio"] = short_atr / (long_atr + 1e-10)

    # Bollinger Bands
    bb = _bollinger_bands(c, feature_cfg.bb_period, feature_cfg.bb_std)
    features["bb_width"] = bb["bb_width"]
    features["bb_pct_b"] = bb["bb_pct_b"]

    # Volatility of volatility
    features["vol_of_vol"] = features["atr_14"].rolling(window=50, min_periods=25).std()

    # Intra-bar volatility ratio (bar range / ATR)
    bar_range = (h - l) / (c + 1e-10)
    features["intrabar_vol_ratio"] = bar_range / (features["atr_14"] / c + 1e-10)

    return features


def compute_trend_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 3: Trend/Momentum Features (12 features).
    """
    features = pd.DataFrame(index=df.index)
    c = df["close"]
    h, l = df["high"], df["low"]

    # EMA slopes (normalized by ATR for scale-independence)
    atr_val = _atr(h, l, c, 14)
    for period in feature_cfg.ema_periods:
        ema_val = _ema(c, period)
        slope = ema_val.diff(5) / 5  # 5-bar slope
        features[f"ema_slope_{period}"] = slope / (atr_val + 1e-10)

    # ADX
    adx_df = _adx(h, l, c, feature_cfg.adx_period)
    features["adx"] = adx_df["adx"]
    features["di_differential"] = adx_df["plus_di"] - adx_df["minus_di"]

    # RSI
    for period in feature_cfg.rsi_periods:
        features[f"rsi_{period}"] = _rsi(c, period)

    # RSI rate of change
    features["rsi_roc"] = features[f"rsi_{feature_cfg.rsi_periods[0]}"].diff(5)

    # MACD
    macd_df = _macd(c, feature_cfg.macd_fast, feature_cfg.macd_slow, feature_cfg.macd_signal)
    features["macd_hist"] = macd_df["macd_hist"] / (atr_val + 1e-10)  # Normalized
    features["macd_signal_dist"] = (macd_df["macd"] - macd_df["macd_signal"]) / (atr_val + 1e-10)

    return features


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 4: Volume/Microstructure Features (8 features).

    Uses tickvol since MT5 real volume is typically 0 for forex/CFDs.
    """
    features = pd.DataFrame(index=df.index)
    vol = df["tickvol"].astype(float)
    close = df["close"]

    # Volume Z-score
    for period in feature_cfg.volume_ma_periods:
        vol_ma = vol.rolling(window=period, min_periods=period // 2).mean()
        vol_std = vol.rolling(window=period, min_periods=period // 2).std()
        features[f"vol_zscore_{period}"] = (vol - vol_ma) / (vol_std + 1e-10)

    # OBV slope
    obv_direction = np.sign(close.diff())
    obv = (vol * obv_direction).cumsum()
    features["obv_slope_20"] = obv.rolling(window=20, min_periods=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
    ) / (vol.rolling(20).mean() + 1e-10)

    # Volume-weighted return (captures conviction)
    log_ret = np.log(close / close.shift(1).replace(0, np.nan))
    vw_return = log_ret * vol
    features["vw_return_20"] = vw_return.rolling(window=20, min_periods=10).mean()

    # Spread as liquidity proxy
    if "spread" in df.columns:
        features["spread_zscore"] = rolling_zscore(df["spread"].astype(float), window=100)

    # Tick activity ratio (normalized volume rate)
    features["tick_activity_ratio"] = vol / vol.rolling(window=100, min_periods=50).mean()

    return features


def compute_cross_timeframe_features(
    df_m5: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
) -> pd.DataFrame:
    """
    Group 5: Cross-Timeframe Features (8 features).

    Propagates higher-timeframe context down to M5 level.
    All higher-TF features are lagged to prevent look-ahead bias.
    """
    features = pd.DataFrame(index=df_m5.index)

    # H1 features propagated to M5
    h1_ema20 = _ema(df_h1["close"], 20)
    h1_ema50 = _ema(df_h1["close"], 50)
    h1_trend = (h1_ema20 - h1_ema50) / (h1_ema50 + 1e-10)
    # Shift by 1 to prevent look-ahead (use previous completed H1 bar)
    h1_trend_shifted = h1_trend.shift(1)
    # Reindex to M5 using forward fill (propagate last known value)
    features["h1_trend_alignment"] = h1_trend_shifted.reindex(df_m5.index, method="ffill")

    # H4 features propagated to M5
    h4_atr = _atr(df_h4["high"], df_h4["low"], df_h4["close"], 14)
    h4_atr_shifted = h4_atr.shift(1)
    features["h4_atr"] = h4_atr_shifted.reindex(df_m5.index, method="ffill")

    # Multi-scale volatility ratio (M5 vol / H4 vol)
    m5_atr = _atr(df_m5["high"], df_m5["low"], df_m5["close"], 14)
    # Annualize: M5 ATR * sqrt(48) ≈ H4 scale, but we use ratio directly
    h4_atr_m5 = features["h4_atr"]
    features["vol_ratio_m5_h4"] = m5_atr / (h4_atr_m5 / np.sqrt(48) + 1e-10)

    # H1 RSI
    h1_rsi = _rsi(df_h1["close"], 14).shift(1)
    features["h1_rsi"] = h1_rsi.reindex(df_m5.index, method="ffill")

    # H4 ADX
    h4_adx = _adx(df_h4["high"], df_h4["low"], df_h4["close"], 14)["adx"].shift(1)
    features["h4_adx"] = h4_adx.reindex(df_m5.index, method="ffill")

    # Price position relative to H1 moving average
    h1_sma50_shifted = _sma(df_h1["close"], 50).shift(1)
    h1_sma50_m5 = h1_sma50_shifted.reindex(df_m5.index, method="ffill")
    features["price_vs_h1_ma50"] = (df_m5["close"] - h1_sma50_m5) / (h1_sma50_m5 + 1e-10)

    # H1 Bollinger Band Width (regime context)
    h1_bb = _bollinger_bands(df_h1["close"], 20, 2.0)
    h1_bb_width_shifted = h1_bb["bb_width"].shift(1)
    features["h1_bb_width"] = h1_bb_width_shifted.reindex(df_m5.index, method="ffill")

    # Higher-TF trend consensus (sign agreement between EMAs)
    h1_ema_sign = np.sign(h1_trend_shifted)
    h4_trend = ((_ema(df_h4["close"], 20) - _ema(df_h4["close"], 50)) / (_ema(df_h4["close"], 50) + 1e-10)).shift(1)
    h4_ema_sign = np.sign(h4_trend).reindex(df_m5.index, method="ffill")
    h1_ema_sign_m5 = h1_ema_sign.reindex(df_m5.index, method="ffill")
    features["trend_consensus"] = (h1_ema_sign_m5 + h4_ema_sign) / 2

    return features


def compute_leading_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 6: Regime-Predictive 'Leading' Features (5+ features).

    These features are designed to change BEFORE a full regime transition.
    """
    features = pd.DataFrame(index=df.index)
    close = df["close"]

    # Correlation breakdown detector
    log_ret = np.log(close / close.shift(1).replace(0, np.nan))
    lag_ret = log_ret.shift(1)
    rolling_corr = log_ret.rolling(window=50, min_periods=25).corr(lag_ret)
    features["return_autocorr"] = rolling_corr

    # Rate of change of autocorrelation
    features["return_autocorr_change"] = rolling_corr.diff(10)

    # Drawdown depth and velocity
    dd = compute_drawdown(close)
    features["drawdown_depth"] = dd["drawdown_pct"]
    features["drawdown_velocity"] = dd["drawdown_pct"].diff(5)

    # Consecutive directional bar counter
    direction = np.sign(close.diff())
    # Count consecutive same-direction bars
    change = direction.ne(direction.shift(1))
    features["consecutive_direction"] = direction * change.groupby(change.cumsum()).cumcount().add(1)

    # Volatility term-structure slope (short-term RV vs long-term RV)
    rv_short = (log_ret ** 2).rolling(window=10, min_periods=5).sum()
    rv_long = (log_ret ** 2).rolling(window=50, min_periods=25).sum()
    features["vol_term_structure_slope"] = (rv_short / 10) / (rv_long / 50 + 1e-10) - 1

    # BB width rate of change (expansion/contraction speed)
    bb = _bollinger_bands(close, 20, 2.0)
    features["bb_width_roc"] = bb["bb_width"].pct_change(10)

    return features


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def extract_all_features(
    df_m5: pd.DataFrame,
    normalize: bool = True,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Full feature extraction pipeline.

    Args:
        df_m5: Clean M5 OHLCV DataFrame.
        normalize: Whether to apply expanding Z-score normalization.
        drop_na: Whether to drop rows with NaN values.

    Returns:
        features: DataFrame with all features at M5 resolution.
        timeframe_data: Dict of resampled DataFrames for reference.
    """
    logger.info("Starting feature extraction pipeline...")

    # Resample to higher timeframes
    logger.info("  Resampling to H1, H4...")
    df_h1 = resample_ohlcv(df_m5, data_cfg.timeframes["H1"])
    df_h4 = resample_ohlcv(df_m5, data_cfg.timeframes["H4"])
    timeframe_data = {"M5": df_m5, "H1": df_h1, "H4": df_h4}

    # Compute feature groups
    logger.info("  Computing return features...")
    feat_returns = compute_return_features(df_m5)

    logger.info("  Computing volatility features...")
    feat_vol = compute_volatility_features(df_m5)

    logger.info("  Computing trend/momentum features...")
    feat_trend = compute_trend_momentum_features(df_m5)

    logger.info("  Computing volume features...")
    feat_volume = compute_volume_features(df_m5)

    logger.info("  Computing cross-timeframe features...")
    feat_xtf = compute_cross_timeframe_features(df_m5, df_h1, df_h4)

    logger.info("  Computing leading features...")
    feat_leading = compute_leading_features(df_m5)

    # Combine all features
    features = pd.concat(
        [feat_returns, feat_vol, feat_trend, feat_volume, feat_xtf, feat_leading],
        axis=1,
    )

    # Replace infinities
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN
    if drop_na:
        initial_len = len(features)
        features.dropna(inplace=True)
        dropped = initial_len - len(features)
        logger.info(f"  Dropped {dropped:,} rows with NaN ({dropped/initial_len*100:.1f}%)")

    # Normalize using expanding Z-score (no look-ahead bias)
    if normalize:
        logger.info("  Normalizing features with expanding Z-score...")
        feature_names = features.columns.tolist()
        for col in feature_names:
            features[col] = expanding_zscore(features[col], min_periods=200)
        # Drop the warm-up period NaNs from normalization
        features.dropna(inplace=True)

    logger.info(
        f"  Feature extraction complete: {features.shape[1]} features, "
        f"{len(features):,} samples"
    )

    return features, timeframe_data


def get_feature_names() -> Dict[str, list]:
    """Return feature names organized by group."""
    # These are approximate — exact names depend on config
    return {
        "return": [f"log_return_{p}" for p in feature_cfg.return_periods]
                 + ["return_skew_20", "return_skew_50", "return_kurtosis_20",
                    "return_kurtosis_50", "hurst_exponent"],
        "volatility": ["realized_var_20", "realized_var_50", "parkinson_vol_20",
                        "parkinson_vol_50", "garman_klass_vol_20"]
                      + [f"atr_{p}" for p in feature_cfg.atr_periods]
                      + ["atr_ratio", "bb_width", "bb_pct_b", "vol_of_vol",
                         "intrabar_vol_ratio"],
        "trend_momentum": [f"ema_slope_{p}" for p in feature_cfg.ema_periods]
                         + ["adx", "di_differential"]
                         + [f"rsi_{p}" for p in feature_cfg.rsi_periods]
                         + ["rsi_roc", "macd_hist", "macd_signal_dist"],
        "volume": ["vol_zscore_20", "vol_zscore_50", "obv_slope_20",
                    "vw_return_20", "spread_zscore", "tick_activity_ratio"],
        "cross_timeframe": ["h1_trend_alignment", "h4_atr", "vol_ratio_m5_h4",
                            "h1_rsi", "h4_adx", "price_vs_h1_ma50",
                            "h1_bb_width", "trend_consensus"],
        "leading": ["return_autocorr", "return_autocorr_change",
                     "drawdown_depth", "drawdown_velocity",
                     "consecutive_direction", "vol_term_structure_slope",
                     "bb_width_roc"],
    }
