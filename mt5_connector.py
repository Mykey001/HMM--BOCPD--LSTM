"""
mt5_connector.py — MetaTrader 5 Live Signal Service.

Connects the trained 3-layer regime detection system to a live MT5 terminal.
Fetches real-time M5 OHLCV bars, runs the full pipeline, and writes
regime signals that MT5 Expert Advisors can read.

Usage:
    python mt5_connector.py

Signal output format (JSON):
    {
        "timestamp": "2025-01-15 14:30:00",
        "regime_id": 0,
        "regime_name": "Low-Vol Bullish",
        "confidence": 0.87,
        "transition_probability": 0.05,
        "changepoint_probability": 0.12,
        "is_transition_alert": false,
        "all_probabilities": {...}
    }
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import mt5_cfg, REGIME_NAMES, NUM_REGIMES, LOG_DIR
from utils import clean_data, logger, setup_logger
from predict import RegimePredictor, RegimePrediction

# Setup dedicated logger for live trading
live_logger = setup_logger("mt5_live", mt5_cfg.log_file)


# ══════════════════════════════════════════════════════════════════════
# MT5 DATA FETCHER
# ══════════════════════════════════════════════════════════════════════

def init_mt5() -> bool:
    """Initialize MetaTrader 5 connection."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        live_logger.error(
            "MetaTrader5 package not installed. Install with: pip install MetaTrader5"
        )
        return False

    if not mt5.initialize():
        live_logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False

    terminal_info = mt5.terminal_info()
    if terminal_info:
        live_logger.info(f"MT5 connected: {terminal_info.name} (build {terminal_info.build})")

    # Check symbol availability
    symbol_info = mt5.symbol_info(mt5_cfg.symbol)
    if symbol_info is None:
        live_logger.error(f"Symbol {mt5_cfg.symbol} not found!")
        mt5.shutdown()
        return False

    if not symbol_info.visible:
        if not mt5.symbol_select(mt5_cfg.symbol, True):
            live_logger.error(f"Failed to select {mt5_cfg.symbol}")
            mt5.shutdown()
            return False

    live_logger.info(f"Symbol {mt5_cfg.symbol}: spread={symbol_info.spread}, "
                     f"bid={symbol_info.bid}, ask={symbol_info.ask}")

    return True


def fetch_mt5_bars(n_bars: int = 500) -> Optional[pd.DataFrame]:
    """
    Fetch the latest M5 bars from MT5.

    Returns DataFrame in the same format as our CSV data.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None

    # Map timeframe string to MT5 constant
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    timeframe = tf_map.get(mt5_cfg.timeframe_mt5, mt5.TIMEFRAME_M5)

    rates = mt5.copy_rates_from_pos(mt5_cfg.symbol, timeframe, 0, n_bars)

    if rates is None or len(rates) == 0:
        live_logger.error(f"Failed to fetch bars: {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("datetime", inplace=True)

    # Rename to our standard column names
    df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "tick_volume": "tickvol",
        "real_volume": "vol",
        "spread": "spread",
    }, inplace=True)

    # Ensure required columns exist
    for col in ["open", "high", "low", "close", "tickvol"]:
        if col not in df.columns:
            live_logger.error(f"Missing column: {col}")
            return None

    if "vol" not in df.columns:
        df["vol"] = 0
    if "spread" not in df.columns:
        df["spread"] = 0

    df = df[["open", "high", "low", "close", "tickvol", "vol", "spread"]]

    return df


# ══════════════════════════════════════════════════════════════════════
# SIGNAL WRITER
# ══════════════════════════════════════════════════════════════════════

def write_signal(prediction: RegimePrediction, timestamp: str):
    """Write the prediction to a JSON file that MT5 EAs can read."""
    signal = {
        "timestamp": timestamp,
        "regime_id": prediction.regime_id,
        "regime_name": prediction.regime_name,
        "confidence": round(prediction.confidence, 4),
        "transition_probability": round(prediction.transition_probability, 4),
        "changepoint_probability": round(prediction.changepoint_probability, 4),
        "is_transition_alert": prediction.is_transition_alert,
        "all_probabilities": {
            k: round(v, 4) for k, v in prediction.all_probabilities.items()
        },
    }

    signal_path = Path(mt5_cfg.signal_file)
    signal_path.parent.mkdir(parents=True, exist_ok=True)

    with open(signal_path, "w") as f:
        json.dump(signal, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
# LIVE LOOP
# ══════════════════════════════════════════════════════════════════════

def run_live_service():
    """
    Main live signal service loop.

    Every M5 bar close:
      1. Fetch latest bars from MT5
      2. Run full 3-layer pipeline
      3. Write signal to shared file
      4. Log prediction
    """
    live_logger.info("=" * 60)
    live_logger.info("STARTING MT5 LIVE REGIME DETECTION SERVICE")
    live_logger.info("=" * 60)

    # Initialize MT5
    if not init_mt5():
        live_logger.error("Failed to initialize MT5. Exiting.")
        return

    # Load trained models
    live_logger.info("Loading trained models...")
    try:
        predictor = RegimePredictor.load()
    except Exception as e:
        live_logger.error(f"Failed to load models: {e}")
        live_logger.error("Run train.py first to train the models.")
        return

    live_logger.info(f"Symbol: {mt5_cfg.symbol}")
    live_logger.info(f"Timeframe: {mt5_cfg.timeframe_mt5}")
    live_logger.info(f"Lookback bars: {mt5_cfg.lookback_bars}")
    live_logger.info(f"Signal file: {mt5_cfg.signal_file}")
    live_logger.info(f"Poll interval: {mt5_cfg.poll_interval_seconds}s")
    live_logger.info("")

    last_bar_time = None
    prediction_count = 0

    try:
        while True:
            # Fetch latest bars
            df = fetch_mt5_bars(mt5_cfg.lookback_bars)
            if df is None or len(df) < 200:
                live_logger.warning("Insufficient data, waiting...")
                time.sleep(mt5_cfg.poll_interval_seconds)
                continue

            current_bar_time = str(df.index[-1])

            # Only process on new bar
            if current_bar_time == last_bar_time:
                time.sleep(mt5_cfg.poll_interval_seconds)
                continue

            last_bar_time = current_bar_time
            prediction_count += 1

            # Clean data
            df_clean = clean_data(df)

            if len(df_clean) < 200:
                live_logger.warning("Too few clean bars, waiting...")
                time.sleep(mt5_cfg.poll_interval_seconds)
                continue

            # Run prediction
            try:
                prediction = predictor.predict_latest(df_clean, lookback_bars=mt5_cfg.lookback_bars)

                # Write signal
                write_signal(prediction, current_bar_time)

                # Log
                alert_str = " ⚠️ ALERT" if prediction.is_transition_alert else ""
                live_logger.info(
                    f"[{current_bar_time}] {prediction.regime_name} "
                    f"(conf={prediction.confidence:.1%}, "
                    f"trans={prediction.transition_probability:.1%}, "
                    f"cp={prediction.changepoint_probability:.1%})"
                    f"{alert_str}"
                )

                if prediction_count % 12 == 0:  # Every hour
                    live_logger.info(f"  --- {prediction_count} predictions made ---")

            except Exception as e:
                live_logger.error(f"Prediction error: {e}")

            time.sleep(mt5_cfg.poll_interval_seconds)

    except KeyboardInterrupt:
        live_logger.info("\nService stopped by user.")
    finally:
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except Exception:
            pass
        live_logger.info(f"Total predictions: {prediction_count}")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_live_service()
