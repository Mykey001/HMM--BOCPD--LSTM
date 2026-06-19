#!/usr/bin/env python3
"""
ML Market Regime Detection - PyQt5 Dashboard
Real-Time Market Regime Detection and Trade Filtering for HMM+BOCPD+LSTM System
"""

import sys
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QComboBox, 
                             QGroupBox, QCheckBox, QGridLayout,
                             QTabWidget, QTableWidget, QTableWidgetItem,
                             QSplitter, QSpinBox, QTextEdit, QLineEdit)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import project modules
from config import mt5_cfg, REGIME_NAMES, REGIME_COLORS, NUM_REGIMES
from utils import clean_data, setup_logger
from predict import RegimePredictor, RegimePrediction

# Attempt to import MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# Colors for transition state
TRANSITION_COLOR = "#95a5a6"

# Setup logger for GUI worker
gui_logger = setup_logger("gui_worker", None)


# ============================================================
# BACKGROUND WORKER
# ============================================================

class DashboardWorker(QThread):
    """Background thread that handles MT5 connection, feature extraction, and prediction."""
    prediction_ready = pyqtSignal(object)  # Emits RegimePrediction
    features_ready = pyqtSignal(dict)      # Emits latest features dict
    layer_status_ready = pyqtSignal(dict)  # Emits status info from all 3 layers
    log_message = pyqtSignal(str)          # Emits log messages
    connection_status = pyqtSignal(bool, str) # Emits connected state and info string

    def __init__(self, symbol, timeframe_str, lookback):
        super().__init__()
        self.symbol = symbol
        self.lookback = lookback
        self.is_running = False
        self.predictor = None
        
        # Map timeframe string to MT5 constant
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        self.timeframe = tf_map.get(timeframe_str, mt5.TIMEFRAME_M5)

    def log(self, msg):
        self.log_message.emit(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def run(self):
        self.is_running = True
        self.log("Starting prediction worker thread...")

        if not MT5_AVAILABLE:
            self.log("ERROR: MetaTrader5 package not installed. Windows required.")
            self.connection_status.emit(False, "MT5 Missing")
            return

        if not mt5.initialize():
            self.log(f"ERROR: MT5 initialization failed: {mt5.last_error()}")
            self.connection_status.emit(False, "MT5 Init Failed")
            return

        term_info = mt5.terminal_info()
        self.log(f"Connected to MT5: {term_info.company if term_info else 'Unknown'}")
        
        # Check symbol
        sym_info = mt5.symbol_info(self.symbol)
        if sym_info is None:
            self.log(f"ERROR: Symbol {self.symbol} not found.")
            self.connection_status.emit(False, "Symbol Error")
            return
            
        if not sym_info.visible:
            mt5.symbol_select(self.symbol, True)
            
        self.connection_status.emit(True, f"Connected ({self.symbol})")

        # Load Predictor
        try:
            self.log("Loading models...")
            self.predictor = RegimePredictor.load()
            self.log("Models loaded successfully.")
        except Exception as e:
            self.log(f"ERROR loading models: {e}")
            self.connection_status.emit(False, "Model Error")
            return

        last_time = None

        while self.is_running:
            try:
                # Fetch bars
                rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.lookback)
                if rates is None or len(rates) == 0:
                    self.log("Failed to fetch rates from MT5.")
                    time.sleep(5)
                    continue

                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Rename to standard column names expected by feature_engine
                df.rename(columns={
                    "tick_volume": "tickvol",
                    "real_volume": "vol",
                    "spread": "spread",
                }, inplace=True)
                
                # Ensure 'vol' and 'spread' exist if they weren't in the rates array
                if "vol" not in df.columns:
                    df["vol"] = 0
                if "spread" not in df.columns:
                    df["spread"] = 0
                
                current_time = df.index[-1]
                
                # Check if we have a new bar (or it's the first run)
                if last_time is None or current_time > last_time:
                    self.log(f"Processing new bar: {current_time} (Lookback: {len(df)})")
                    
                    # Ensure minimum lookback for feature engine (e.g., expanding z-score needs 1000+)
                    if len(df) < 1000:
                        self.log(f"WARNING: Need at least 1000 bars for feature stability, got {len(df)}")
                        
                    prediction = self.predictor.predict_latest(df)
                    
                    if prediction:
                        # Write signal file
                        self._write_signal(prediction)
                        
                        # Emit results to GUI
                        self.prediction_ready.emit(prediction)
                        
                        # Extract features to display (predictor caches the last computed features)
                        # We use a protected attribute here for visualization
                        if hasattr(self.predictor, '_last_features') and self.predictor._last_features is not None:
                            latest_feats = self.predictor._last_features.iloc[-1].to_dict()
                            self.features_ready.emit(latest_feats)
                        
                        # Extract layer status
                        if hasattr(self.predictor, '_last_hmm_probs') and hasattr(self.predictor, '_last_bocpd_output'):
                            hmm_state = int(np.argmax(self.predictor._last_hmm_probs))
                            hmm_prob = float(self.predictor._last_hmm_probs[hmm_state])
                            
                            bocpd = self.predictor._last_bocpd_output
                            if len(bocpd) > 0:
                                cp_prob = float(bocpd['bocpd_cp_prob'].iloc[-1])
                                rl = float(bocpd['bocpd_rl_mean'].iloc[-1])
                                bocpd_alert = bool(bocpd['bocpd_alert'].iloc[-1])
                            else:
                                cp_prob, rl, bocpd_alert = 0.0, 0.0, False
                                
                            layer_status = {
                                "hmm": f"State {hmm_state} - {REGIME_NAMES.get(hmm_state, '')} (Post: {hmm_prob:.2f})",
                                "bocpd": f"CP Prob: {cp_prob:.4f} | Run Length: {rl:.1f} | Alert: {bocpd_alert}",
                                "lstm": f"Regime {prediction.regime_id} - Conf: {prediction.confidence:.1%} | Trans: {prediction.transition_probability:.1%}"
                            }
                            self.layer_status_ready.emit(layer_status)

                    last_time = current_time

            except Exception as e:
                self.log(f"Prediction loop error: {str(e)}")
            
            # Poll interval (10s)
            for _ in range(10):
                if not self.is_running:
                    break
                time.sleep(1)

        self.log("Worker thread stopped.")
        mt5.shutdown()
        self.connection_status.emit(False, "Disconnected")

    def stop(self):
        self.is_running = False

    def _write_signal(self, prediction: RegimePrediction):
        """Write JSON signal file for EA."""
        signal_file = Path(mt5_cfg.signal_file)
        try:
            signal_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "regime_id": int(prediction.regime_id),
                "regime_name": str(prediction.regime_name),
                "confidence": float(prediction.confidence),
                "transition_probability": float(prediction.transition_probability),
                "changepoint_probability": float(prediction.changepoint_probability),
                "is_transition_alert": bool(prediction.is_transition_alert),
                "all_probabilities": {str(k): float(v) for k, v in prediction.all_probabilities.items()}
            }
            
            # Add filter configuration (read from GUI settings via a shared config later if needed)
            
            # Write atomic
            temp_file = signal_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(signal_data, f, indent=2)
            temp_file.replace(signal_file)
            
        except Exception as e:
            self.log(f"Error writing signal file: {e}")


# ============================================================
# MATPLOTLIB CANVASES
# ============================================================

class ProbabilityCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        
        # Dark theme settings
        fig.patch.set_facecolor('#1c2128')
        self.axes.set_facecolor('#161b22')
        self.axes.tick_params(colors='#8b949e')
        self.axes.spines['bottom'].set_color('#30363d')
        self.axes.spines['top'].set_color('#30363d')
        self.axes.spines['left'].set_color('#30363d')
        self.axes.spines['right'].set_color('#30363d')
        self.axes.yaxis.label.set_color('#8b949e')
        self.axes.xaxis.label.set_color('#8b949e')
        
        super(ProbabilityCanvas, self).__init__(fig)
        self.setParent(parent)
        self.figure.tight_layout()

    def update_chart(self, probs: dict, transition_prob: float):
        self.axes.clear()
        
        labels = list(probs.keys()) + ["Transition"]
        values = list(probs.values()) + [transition_prob]
        
        # Colors map
        colors = []
        for l in probs.keys():
            # Find regime ID for this name
            rid = next((k for k, v in REGIME_NAMES.items() if v == l), 0)
            colors.append(REGIME_COLORS.get(rid, "#ffffff"))
        colors.append(TRANSITION_COLOR)
        
        bars = self.axes.bar(labels, values, color=colors)
        
        # Add labels
        for bar in bars:
            yval = bar.get_height()
            self.axes.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.1%}", 
                           ha='center', va='bottom', color='#e6edf3', fontsize=9)
            
        self.axes.set_ylim(0, 1.1)
        self.axes.set_ylabel("Probability")
        self.axes.set_title("Current Model Probabilities", color='#e6edf3')
        
        # Rotate x labels
        self.axes.set_xticks(range(len(labels)))
        self.axes.set_xticklabels(labels, rotation=45, ha='right')
        
        self.figure.tight_layout()
        self.draw()


class HistoryCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        
        fig.patch.set_facecolor('#1c2128')
        self.axes.set_facecolor('#161b22')
        self.axes.tick_params(colors='#8b949e')
        for spine in self.axes.spines.values():
            spine.set_color('#30363d')
            
        super(HistoryCanvas, self).__init__(fig)
        self.setParent(parent)
        
        self.history_len = 50
        self.probs_history = deque(maxlen=self.history_len)
        self.trans_history = deque(maxlen=self.history_len)
        
        # Initialize with zeros
        for _ in range(self.history_len):
            self.probs_history.append({k: 0.0 for k in REGIME_NAMES.values()})
            self.trans_history.append(0.0)

    def update_history(self, probs: dict, trans: float):
        self.probs_history.append(probs)
        self.trans_history.append(trans)
        
        self.axes.clear()
        
        x = np.arange(len(self.probs_history))
        y_data = []
        labels = []
        colors = []
        
        for k in REGIME_NAMES.values():
            y_data.append([p.get(k, 0.0) for p in self.probs_history])
            labels.append(k)
            rid = next((id for id, name in REGIME_NAMES.items() if name == k), 0)
            colors.append(REGIME_COLORS.get(rid, "#ffffff"))
            
        # Add transition line
        trans_data = list(self.trans_history)
            
        self.axes.stackplot(x, *y_data, labels=labels, colors=colors, alpha=0.8)
        self.axes.plot(x, trans_data, color=TRANSITION_COLOR, linewidth=2, label="Transition Prob", linestyle='--')
        
        self.axes.set_ylim(0, 1.0)
        self.axes.set_xlim(0, self.history_len - 1)
        self.axes.set_xticks([])
        self.axes.set_title("Probability History (Last 50 Updates)", color='#e6edf3')
        
        self.figure.tight_layout()
        self.draw()


# ============================================================
# MAIN WINDOW
# ============================================================

class RegimeDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Regime Detection - 3 Layer System")
        self.setMinimumSize(1000, 800)
        
        self.worker = None
        self.prediction_count = 0
        self.start_time = None
        
        self.init_ui()
        self.apply_dark_theme()

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0d1117; }
            QWidget { color: #e6edf3; font-family: 'Segoe UI', Arial, sans-serif; }
            
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 1ex;
                background-color: #161b22;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #8b949e;
            }
            
            QPushButton {
                background-color: #21262d;
                border: 1px solid #363b42;
                border-radius: 4px;
                padding: 5px 15px;
                color: #c9d1d9;
            }
            QPushButton:hover { background-color: #30363d; border-color: #8b949e; }
            QPushButton:pressed { background-color: #282e33; }
            QPushButton#startBtn { background-color: #238636; color: white; border: none; }
            QPushButton#startBtn:hover { background-color: #2ea043; }
            QPushButton#stopBtn { background-color: #da3633; color: white; border: none; }
            QPushButton#stopBtn:hover { background-color: #f85149; }
            
            QLineEdit, QComboBox, QSpinBox {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px;
                color: #e6edf3;
            }
            
            QTabWidget::pane { border: 1px solid #30363d; background-color: #1c2128; border-radius: 4px; }
            QTabBar::tab {
                background-color: #161b22;
                border: 1px solid #30363d;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background-color: #1c2128; border-bottom-color: #1c2128; color: #58a6ff; }
            
            QTableWidget {
                background-color: #1c2128;
                alternate-background-color: #161b22;
                gridline-color: #30363d;
                border: none;
            }
            QHeaderView::section {
                background-color: #0d1117;
                color: #8b949e;
                padding: 4px;
                border: 1px solid #30363d;
            }
            
            QTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                font-family: Consolas, monospace;
            }
        """)

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # 1. Connection Toolbar
        toolbar = QHBoxLayout()
        
        self.conn_led = QLabel("●")
        self.conn_led.setStyleSheet("color: #8b949e; font-size: 18px;")
        toolbar.addWidget(self.conn_led)
        
        self.conn_status = QLabel("Disconnected")
        self.conn_status.setStyleSheet("color: #8b949e;")
        toolbar.addWidget(self.conn_status)
        
        toolbar.addSpacing(20)
        toolbar.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit("XAUUSD")
        self.symbol_input.setFixedWidth(80)
        toolbar.addWidget(self.symbol_input)
        
        toolbar.addWidget(QLabel("TF:"))
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
        self.tf_combo.setCurrentText("M5")
        toolbar.addWidget(self.tf_combo)
        
        toolbar.addWidget(QLabel("Bars:"))
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(1000, 5000)
        self.bars_spin.setValue(1500)
        self.bars_spin.setSingleStep(500)
        toolbar.addWidget(self.bars_spin)
        
        toolbar.addStretch()
        
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.start_worker)
        toolbar.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_worker)
        self.stop_btn.setEnabled(False)
        toolbar.addWidget(self.stop_btn)
        
        main_layout.addLayout(toolbar)
        
        # 2. Hero Panel (Current Regime)
        hero_group = QGroupBox("Live Market Regime")
        hero_layout = QVBoxLayout()
        
        self.regime_banner = QLabel("WAITING FOR DATA...")
        self.regime_banner.setAlignment(Qt.AlignCenter)
        self.regime_banner.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            color: #8b949e;
            background-color: #0d1117;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #30363d;
        """)
        hero_layout.addWidget(self.regime_banner)
        
        metrics_layout = QHBoxLayout()
        
        self.conf_label = QLabel("Confidence: --%")
        self.conf_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        metrics_layout.addWidget(self.conf_label)
        
        self.trans_label = QLabel("Transition Prob: --%")
        self.trans_label.setStyleSheet("font-size: 16px;")
        metrics_layout.addWidget(self.trans_label)
        
        self.cp_label = QLabel("Changepoint Prob: --%")
        self.cp_label.setStyleSheet("font-size: 16px;")
        metrics_layout.addWidget(self.cp_label)
        
        self.alert_label = QLabel("STATUS: NORMAL")
        self.alert_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #8b949e;")
        metrics_layout.addWidget(self.alert_label)
        
        hero_layout.addLayout(metrics_layout)
        hero_group.setLayout(hero_layout)
        main_layout.addWidget(hero_group)
        
        # 3. Pipeline Status
        pipeline_group = QGroupBox("3-Layer Pipeline Status")
        pipeline_layout = QVBoxLayout()
        
        self.l1_label = QLabel("Layer 1 (HMM): Waiting...")
        self.l2_label = QLabel("Layer 2 (BOCPD): Waiting...")
        self.l3_label = QLabel("Layer 3 (LSTM): Waiting...")
        
        for lbl in [self.l1_label, self.l2_label, self.l3_label]:
            lbl.setStyleSheet("font-family: Consolas; font-size: 13px;")
            pipeline_layout.addWidget(lbl)
            
        pipeline_group.setLayout(pipeline_layout)
        main_layout.addWidget(pipeline_group)
        
        # 4. Tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Charts
        chart_tab = QWidget()
        chart_layout = QVBoxLayout()
        self.prob_canvas = ProbabilityCanvas(self, width=8, height=4)
        self.hist_canvas = HistoryCanvas(self, width=8, height=2)
        chart_layout.addWidget(self.prob_canvas, 2)
        chart_layout.addWidget(self.hist_canvas, 1)
        chart_tab.setLayout(chart_layout)
        self.tabs.addTab(chart_tab, "Probabilities")
        
        # Tab 2: Statistics
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["Feature", "Current Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        stats_layout.addWidget(self.stats_table)
        stats_tab.setLayout(stats_layout)
        self.tabs.addTab(stats_tab, "Feature Statistics")
        
        # Tab 3: Log
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_tab.setLayout(log_layout)
        self.tabs.addTab(log_tab, "System Log")
        
        main_layout.addWidget(self.tabs, 1) # Give tabs stretch factor 1
        
        # 5. Status Bar
        self.footer = QLabel("Status: Ready | Predictions: 0 | Uptime: 0s")
        self.footer.setStyleSheet("color: #8b949e; padding: 5px;")
        main_layout.addWidget(self.footer)
        
        self.setCentralWidget(central_widget)

    def log(self, msg):
        self.log_text.append(msg)
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_connection_status(self, is_connected, text):
        if is_connected:
            self.conn_led.setStyleSheet("color: #3fb950; font-size: 18px;") # Green
            self.conn_status.setStyleSheet("color: #3fb950;")
        else:
            self.conn_led.setStyleSheet("color: #f85149; font-size: 18px;") # Red
            self.conn_status.setStyleSheet("color: #f85149;")
        self.conn_status.setText(text)

    def start_worker(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.symbol_input.setEnabled(False)
        self.tf_combo.setEnabled(False)
        self.bars_spin.setEnabled(False)
        
        self.start_time = time.time()
        self.prediction_count = 0
        
        self.worker = DashboardWorker(
            self.symbol_input.text(), 
            self.tf_combo.currentText(),
            self.bars_spin.value()
        )
        self.worker.log_message.connect(self.log)
        self.worker.connection_status.connect(self.update_connection_status)
        self.worker.prediction_ready.connect(self.on_prediction)
        self.worker.features_ready.connect(self.on_features)
        self.worker.layer_status_ready.connect(self.on_layer_status)
        
        self.worker.start()

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.symbol_input.setEnabled(True)
        self.tf_combo.setEnabled(True)
        self.bars_spin.setEnabled(True)
        
        self.update_connection_status(False, "Stopped")

    def on_prediction(self, p: RegimePrediction):
        self.prediction_count += 1
        uptime = int(time.time() - self.start_time)
        m, s = divmod(uptime, 60)
        h, m = divmod(m, 60)
        self.footer.setText(f"Status: Running | Predictions: {self.prediction_count} | Uptime: {h}h {m}m {s}s")
        
        # Update Hero Panel
        color = REGIME_COLORS.get(p.regime_id, "#8b949e")
        if p.is_transition_alert:
            # Pulse the color or show transition style
            color = TRANSITION_COLOR
            self.alert_label.setText("⚠️ TRANSITION ALERT")
            self.alert_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #f39c12;")
        else:
            self.alert_label.setText("STATUS: NORMAL")
            self.alert_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #8b949e;")
            
        self.regime_banner.setText(p.regime_name.upper())
        self.regime_banner.setStyleSheet(f"""
            font-size: 28px; 
            font-weight: bold; 
            color: #ffffff;
            background-color: {color};
            padding: 20px;
            border-radius: 8px;
            border: 2px solid {color};
        """)
        
        self.conf_label.setText(f"Confidence: {p.confidence:.1%}")
        self.trans_label.setText(f"Transition Prob: {p.transition_probability:.1%}")
        self.cp_label.setText(f"Changepoint: {p.changepoint_probability:.1%}")
        
        # Update Charts
        self.prob_canvas.update_chart(p.all_probabilities, p.transition_probability)
        self.hist_canvas.update_history(p.all_probabilities, p.transition_probability)

    def on_layer_status(self, status: dict):
        self.l1_label.setText(f"Layer 1 (HMM):   {status.get('hmm', '')}")
        self.l2_label.setText(f"Layer 2 (BOCPD): {status.get('bocpd', '')}")
        self.l3_label.setText(f"Layer 3 (LSTM):  {status.get('lstm', '')}")

    def on_features(self, features: dict):
        self.stats_table.setRowCount(len(features))
        row = 0
        for k, v in sorted(features.items()):
            item_k = QTableWidgetItem(str(k))
            item_v = QTableWidgetItem(f"{v:.4f}")
            item_k.setForeground(QColor('#e6edf3'))
            item_v.setForeground(QColor('#58a6ff'))
            self.stats_table.setItem(row, 0, item_k)
            self.stats_table.setItem(row, 1, item_v)
            row += 1

    def closeEvent(self, event):
        self.stop_worker()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RegimeDashboard()
    window.show()
    sys.exit(app.exec_())
