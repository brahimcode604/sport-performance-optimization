"""
=============================================================================
  SPORT PERFORMANCE ANALYTICS — Premium Visualization Dashboard
  PyQt6 · Matplotlib · joblib · pandas
=============================================================================
  Tabs:
    1. 🎯 Prédiction   — Formulaire + résultat avec jauge animée
    2. 📊 Visualisations — Radar biométrique + jauge + donut risque + KPIs
    3. 🔬 Importance    — Graphique des features importantes
    4. 📋 Historique    — Tableau + graphique de toutes les prédictions
=============================================================================
"""

import sys, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import joblib

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton,
    QTextEdit, QFrame, QScrollArea, QMessageBox, QProgressBar,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

# ===========================================================================
#  THEME CONSTANTS
# ===========================================================================
DARK_BG  = "#0f0f12"
CARD_BG  = "#16161e"
ACCENT   = "#3b82f6"
SUCCESS  = "#22c55e"
WARNING  = "#f59e0b"
DANGER   = "#ef4444"
TEXT_PRI = "#e2e8f0"
TEXT_SEC = "#64748b"
BORDER   = "#2a2a36"

RISK_COLORS = {
    "Faible":   (SUCCESS,  "#0d2b1a"),
    "Modéré":   (WARNING,  "#2b1d08"),
    "Élevé":    ("#f97316","#2b1208"),
    "Critique": (DANGER,   "#2b0a0a"),
}
PERF_COLORS = {
    "Insuffisante": DANGER,
    "Bonne":        WARNING,
    "Excellente":   SUCCESS,
}

# Matplotlib global style
plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT_PRI,
    "xtick.color": TEXT_SEC, "ytick.color": TEXT_SEC,
    "text.color": TEXT_PRI, "grid.color": BORDER,
    "grid.linewidth": 0.6, "font.family": "DejaVu Sans", "font.size": 10,
})

# ===========================================================================
#  APP CONSTANTS
# ===========================================================================
ARTIFACT_DIR = Path(__file__).parent

NUMERIC_FIELDS = [
    ("heart_rate_bpm",    "Heart Rate (BPM)",      "0.63"),
    ("performance_score", "Performance Score",     "93.9"),
    ("injury_risk_score", "Injury Risk Score",     "0.61"),
    ("step_frequency_hz", "Step Frequency (Hz)",   "-0.88"),
    ("stride_length_m",   "Stride Length (m)",     "1.66"),
    ("acceleration_mps2", "Acceleration (m/s²)",   "-0.30"),
    ("signal_energy",     "Signal Energy",         "1.05"),
    ("dominant_freq_hz",  "Dominant Freq (Hz)",    "1.05"),
]
CATEGORICAL_FIELDS = [
    ("risk_level",        "Risk Level",
     ["Faible","Modéré","Élevé","Critique"],                     "Élevé"),
    ("hr_zone",           "HR Zone",
     ["Zone_Basse","Zone_Aerobie","Zone_Anaerobie","Zone_Max"],   "Zone_Anaerobie"),
    ("performance_level", "Performance Level",
     ["Insuffisante","Bonne","Excellente"],                       "Excellente"),
    ("event_type",        "Event Type",
     ["sprint","high_jump","long_jump","unknown"],                "high_jump"),
]
RADAR_KEYS   = ["heart_rate_bpm","performance_score","step_frequency_hz",
                "stride_length_m","signal_energy","acceleration_mps2"]
RADAR_LABELS = ["Heart Rate","Perf Score","Step Freq","Stride Len","Signal E","Accel"]

# ===========================================================================
#  MATPLOTLIB CANVAS BASE
# ===========================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, w=5, h=4, dpi=100):
        self.fig = Figure(figsize=(w, h), dpi=dpi, facecolor=DARK_BG)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: transparent;")

# ===========================================================================
#  GAUGE CHART
# ===========================================================================
class GaugeCanvas(MplCanvas):
    def __init__(self):
        super().__init__(4, 2.6, 100)
        self._draw(None, "")

    def _draw(self, score, perf_level):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_aspect("equal"); ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.35,1.3); ax.axis("off")

        # Background arc
        t_bg = np.linspace(np.pi, 0, 300)
        ax.plot(np.cos(t_bg), np.sin(t_bg), color="#1e1e28", linewidth=18, solid_capstyle="round")

        # Zone color bands
        for lo, hi, col in [(0,40,"#ef4444"),(40,70,"#f59e0b"),(70,100,"#22c55e")]:
            t = np.linspace(np.pi-(lo/100)*np.pi, np.pi-(hi/100)*np.pi, 100)
            ax.plot(np.cos(t), np.sin(t), color=col, linewidth=18, solid_capstyle="butt", alpha=0.15)

        if score is None:
            ax.text(0, 0.35, "—", ha="center", va="center",
                    fontsize=32, fontweight="bold", color=TEXT_SEC)
        else:
            s = max(0.0, min(100.0, float(score)))
            fc = "#22c55e" if s>=70 else "#f59e0b" if s>=40 else "#ef4444"
            t_fill = np.linspace(np.pi, np.pi-(s/100)*np.pi, 300)
            ax.plot(np.cos(t_fill), np.sin(t_fill), color=fc, linewidth=18,
                    solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=22, foreground=fc, alpha=0.25), pe.Normal()])
            # Needle
            ang = np.pi - (s/100)*np.pi
            ax.annotate("", xy=(0.75*np.cos(ang), 0.75*np.sin(ang)), xytext=(0,0),
                        arrowprops=dict(arrowstyle="-|>", color="#fff", lw=2, mutation_scale=14))
            ax.plot(0, 0, "o", color="#fff", markersize=8, zorder=10)
            ax.text(0, 0.35, f"{s:.1f}", ha="center", va="center",
                    fontsize=30, fontweight="bold", color="#fff")
            ax.text(0, 0.12, "/ 100", ha="center", va="center", fontsize=10, color=TEXT_SEC)
            if perf_level:
                bc = PERF_COLORS.get(perf_level, ACCENT)
                ax.text(0, -0.20, perf_level.upper(), ha="center", va="center",
                        fontsize=9, fontweight="bold", color=bc,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_BG, edgecolor=bc, lw=1.5))
            ax.text(-1.2, -0.10, "0",   ha="center", va="center", fontsize=8, color=TEXT_SEC)
            ax.text( 1.2, -0.10, "100", ha="center", va="center", fontsize=8, color=TEXT_SEC)

        ax.text(0, 0.80, "PERFORMANCE SCORE", ha="center", va="center",
                fontsize=8, fontweight="600", color=TEXT_SEC)
        self.fig.tight_layout(pad=0.2)
        self.draw()

    def update_score(self, score, perf_level=""):
        self._draw(score, perf_level)

# ===========================================================================
#  RADAR / SPIDER CHART
# ===========================================================================
class RadarCanvas(MplCanvas):
    def __init__(self):
        super().__init__(4, 4, 100)
        self._draw({})

    def _draw(self, features: dict):
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="polar", facecolor=CARD_BG)
        N = len(RADAR_KEYS)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

        raw = np.array([float(features.get(k, 0)) for k in RADAR_KEYS])
        norm = np.clip((raw + 3) / 6, 0, 1)
        if "performance_score" in RADAR_KEYS:
            idx = RADAR_KEYS.index("performance_score")
            norm[idx] = np.clip(float(features.get("performance_score", 50)) / 100, 0, 1)

        values = norm.tolist() + [norm[0]]
        ax.fill(angles, values, color=ACCENT, alpha=0.20)
        ax.plot(angles, values, color=ACCENT, linewidth=2)
        ax.scatter(angles[:-1], values[:-1], s=45, color=ACCENT, zorder=10)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_LABELS, fontsize=8, color=TEXT_PRI)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["","","",""])
        ax.set_ylim(0, 1); ax.grid(color=BORDER, linewidth=0.8, alpha=0.7)
        ax.spines["polar"].set_color(BORDER)
        ax.set_title("PROFIL BIOMÉTRIQUE", color=TEXT_SEC, fontsize=9, fontweight="600", pad=14)
        self.fig.tight_layout(pad=1.0)
        self.draw()

    def update_radar(self, features: dict):
        self._draw(features)

# ===========================================================================
#  RISK DONUT CHART
# ===========================================================================
class RiskDonutCanvas(MplCanvas):
    def __init__(self):
        super().__init__(3, 3, 100)
        self._draw(None)

    def _draw(self, risk):
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor=DARK_BG)
        ax.set_aspect("equal"); ax.axis("off")

        levels = ["Faible","Modéré","Élevé","Critique"]
        clrs   = ["#22c55e","#f59e0b","#f97316","#ef4444"]
        idx    = levels.index(risk) if risk in levels else -1
        sizes  = [2 if i==idx else 1 for i in range(4)]
        alphas = [1.0 if i==idx else 0.18 for i in range(4)]
        expl   = [0.06 if i==idx else 0 for i in range(4)]

        wedges, _ = ax.pie(sizes, colors=clrs, startangle=90, explode=expl,
                           wedgeprops=dict(width=0.45, edgecolor=DARK_BG, linewidth=2),
                           counterclock=False)
        for w, a in zip(wedges, alphas): w.set_alpha(a)

        if risk:
            ac = RISK_COLORS.get(risk, (ACCENT, DARK_BG))[0]
            ax.text(0, 0, risk.upper(), ha="center", va="center",
                    fontsize=9, fontweight="bold", color=ac)

        ax.set_title("NIVEAU DE RISQUE", color=TEXT_SEC, fontsize=9, fontweight="600", pad=8)
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def update_risk(self, risk: str):
        self._draw(risk)

# ===========================================================================
#  FEATURE IMPORTANCE CHART
# ===========================================================================
class FeatureImportanceCanvas(MplCanvas):
    def __init__(self):
        super().__init__(7, 5, 100)
        self._draw()

    def _draw(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor=CARD_BG)
        try:
            fp = ARTIFACT_DIR / "feature_importance.csv"
            if not fp.exists():
                raise FileNotFoundError("feature_importance.csv introuvable")
            df = pd.read_csv(fp).dropna()
            df.columns = ["Feature","Score"]
            df = df.sort_values("Score", ascending=True).tail(14)

            def _col(f):
                if "performance_level" in f: return "#6366f1"
                if "hr_zone"           in f: return "#3b82f6"
                if "risk_level"        in f: return "#f97316"
                if "event_type"        in f: return "#22c55e"
                if "heart_rate"        in f: return "#ec4899"
                if "injury"            in f: return "#ef4444"
                if "performance_score" in f: return "#a78bfa"
                return ACCENT

            colors = [_col(f) for f in df["Feature"]]
            bars = ax.barh(df["Feature"], df["Score"], color=colors,
                           height=0.62, edgecolor=DARK_BG, linewidth=0.5)
            for bar, val in zip(bars, df["Score"]):
                ax.text(val+0.008, bar.get_y()+bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=8, color=TEXT_SEC)

            ax.set_xlabel("Coefficient moyen absolu", fontsize=9, color=TEXT_SEC)
            ax.tick_params(colors=TEXT_SEC, labelsize=8)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            for sp in ["left","bottom"]: ax.spines[sp].set_color(BORDER)
            ax.set_xlim(0, df["Score"].max()*1.18)
            ax.grid(axis="x", alpha=0.2)

            legend_items = [
                mpatches.Patch(color="#6366f1", label="Perf. Level"),
                mpatches.Patch(color="#3b82f6", label="Zone HR"),
                mpatches.Patch(color="#f97316", label="Risk Level"),
                mpatches.Patch(color="#22c55e", label="Event Type"),
                mpatches.Patch(color="#ec4899", label="Heart Rate"),
                mpatches.Patch(color="#ef4444", label="Injury Risk"),
                mpatches.Patch(color="#a78bfa", label="Perf. Score"),
            ]
            ax.legend(handles=legend_items, fontsize=8, framealpha=0.15,
                      labelcolor=TEXT_PRI, loc="lower right")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    color=DANGER, transform=ax.transAxes, fontsize=10)
            ax.axis("off")

        ax.set_title("IMPORTANCE DES FEATURES", color=TEXT_PRI,
                     fontsize=12, fontweight="700", pad=14)
        self.fig.tight_layout(pad=1.4)
        self.draw()

# ===========================================================================
#  HISTORY SCORE CHART
# ===========================================================================
class HistoryScoreCanvas(MplCanvas):
    def __init__(self):
        super().__init__(7, 3, 100)
        self.scores = []; self.labels = []
        self._draw()

    def add_score(self, score: float, event: str):
        self.scores.append(score)
        self.labels.append(f"#{len(self.scores)} {event[:5]}")
        self._draw()

    def clear_data(self):
        self.scores = []; self.labels = []
        self._draw()

    def _draw(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor=CARD_BG)

        if not self.scores:
            ax.text(0.5, 0.5, "Aucune prédiction dans cette session",
                    ha="center", va="center", color=TEXT_SEC,
                    transform=ax.transAxes, fontsize=11)
            ax.axis("off")
        else:
            x = np.arange(len(self.scores))
            colors = ["#22c55e" if s>=70 else "#f59e0b" if s>=40 else "#ef4444"
                      for s in self.scores]
            ax.bar(x, self.scores, color=colors, alpha=0.75, width=0.55,
                   edgecolor=DARK_BG, linewidth=0.5)
            ax.plot(x, self.scores, color=ACCENT, linewidth=2,
                    marker="o", markersize=7, zorder=5)
            for xi, s in zip(x, self.scores):
                ax.text(xi, s+1.5, f"{s:.0f}", ha="center", va="bottom",
                        fontsize=8, color=TEXT_PRI, fontweight="600")
            ax.axhline(70, color="#22c55e", lw=1.2, ls="--", alpha=0.5, label="Excellent (70)")
            ax.axhline(40, color="#ef4444", lw=1.2, ls="--", alpha=0.5, label="Insuffisant (40)")
            ax.set_xticks(x); ax.set_xticklabels(self.labels, fontsize=9, color=TEXT_SEC)
            ax.set_ylabel("Score (/100)", fontsize=9, color=TEXT_SEC)
            ax.set_ylim(0, 112)
            ax.tick_params(colors=TEXT_SEC)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            for sp in ["left","bottom"]: ax.spines[sp].set_color(BORDER)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8, framealpha=0.1, labelcolor=TEXT_PRI)

        ax.set_title("HISTORIQUE DES SCORES DE PERFORMANCE", color=TEXT_PRI,
                     fontsize=11, fontweight="700", pad=12)
        self.fig.tight_layout(pad=1.0)
        self.draw()

# ===========================================================================
#  PREDICTION WORKER (background thread)
# ===========================================================================
class PredictionWorker(QThread):
    result_ready = pyqtSignal(dict)
    error        = pyqtSignal(str)

    def __init__(self, pipeline, encoder, lookup, features: dict):
        super().__init__()
        self.pipeline = pipeline; self.encoder = encoder
        self.lookup = lookup; self.features = features

    def run(self):
        try:
            for k, v in {"hour":0,"dayofweek":0,"gyroscope_x":0.0,"gyroscope_y":0.0,
                         "gyroscope_z":0.0,"accelerometer_x":0.0,"accelerometer_y":0.0,
                         "accelerometer_z":0.0}.items():
                self.features.setdefault(k, v)
            df      = pd.DataFrame([self.features])
            encoded = self.pipeline.predict(df)[0]
            label   = self.encoder.inverse_transform([encoded])[0]
            rec     = (self.lookup.loc[label,"recommendation_entrainement"]
                       if label in self.lookup.index else f"(aucune reco pour: {label})")
            parts   = label.split("__")
            self.result_ready.emit({
                "label": label, "recommendation": rec,
                "risk":  parts[0] if len(parts)>0 else "Modéré",
                "zone":  parts[1] if len(parts)>1 else "",
                "perf":  parts[2] if len(parts)>2 else "",
                "event": parts[3] if len(parts)>3 else "",
                "features": self.features,
            })
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")

# ===========================================================================
#  STYLESHEET
# ===========================================================================
STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG}; color: {TEXT_PRI};
    font-family: 'Segoe UI', sans-serif; font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER}; border-radius: 12px; background-color: {DARK_BG};
}}
QTabBar::tab {{
    background-color: #12121a; color: {TEXT_SEC};
    padding: 11px 22px; border: none;
    border-bottom: 2px solid transparent;
    font-size: 12px; font-weight: 600; min-width: 130px;
}}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; background-color: {DARK_BG}; }}
QTabBar::tab:hover {{ color: {TEXT_PRI}; background-color: #1a1a22; }}
QFrame#card {{
    background-color: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: #1a1a22; width: 6px; border-radius: 3px; }}
QScrollBar::handle:vertical {{ background: #3a3a4a; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QLabel#section-title {{
    color: {TEXT_SEC}; font-size: 10px; font-weight: 600; letter-spacing: 2px;
}}
QLabel#field-label {{ color: #cbd5e1; font-size: 12px; font-weight: 500; }}
QLineEdit, QComboBox {{
    background-color: #1e1e28; border: 1px solid #2e2e3e;
    border-radius: 8px; padding: 8px 12px; color: {TEXT_PRI}; font-size: 13px;
}}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SEC}; width: 0; height: 0; margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: #1e1e28; border: 1px solid #2e2e3e;
    border-radius: 8px; color: {TEXT_PRI}; selection-background-color: {ACCENT};
}}
QPushButton#btn-predict {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #3b82f6,stop:1 #6366f1);
    color: #fff; font-size: 13px; font-weight: 600;
    border: none; border-radius: 10px; padding: 12px 28px;
}}
QPushButton#btn-predict:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #2563eb,stop:1 #4f46e5);
}}
QPushButton#btn-predict:disabled {{ background: #2a2a36; color: #4a4a5a; }}
QPushButton#btn-secondary {{
    background-color: transparent; color: {TEXT_SEC};
    font-size: 12px; border: 1px solid {BORDER};
    border-radius: 8px; padding: 8px 16px;
}}
QPushButton#btn-secondary:hover {{ background-color: #1e1e28; color: {TEXT_PRI}; }}
QTextEdit#result-box {{
    background-color: #12121a; border: 1px solid {BORDER};
    border-radius: 10px; color: {TEXT_PRI}; font-size: 13px; padding: 12px;
}}
QProgressBar {{
    background-color: #1e1e28; border: none; border-radius: 3px; height: 4px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {ACCENT},stop:1 #6366f1);
    border-radius: 3px;
}}
QTableWidget {{
    background-color: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 10px; gridline-color: {BORDER};
    color: {TEXT_PRI}; font-size: 12px; selection-background-color: #1e3a5f;
    alternate-background-color: #12121a;
}}
QTableWidget::item {{ padding: 8px 12px; border-bottom: 1px solid {BORDER}; }}
QHeaderView::section {{
    background-color: #12121a; color: {TEXT_SEC};
    font-size: 10px; font-weight: 600; letter-spacing: 1px;
    padding: 10px 12px; border: none; border-bottom: 1px solid {BORDER};
}}
QToolTip {{
    background-color: #1e1e28; color: #cbd5e1;
    border: 1px solid #3a3a4a; border-radius: 6px; padding: 6px 10px;
}}
"""

# ===========================================================================
#  MAIN APPLICATION WINDOW
# ===========================================================================
class SportsDashboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pipeline = None; self.encoder = None
        self.lookup   = None; self.worker  = None
        self.prediction_history = []

        self.setWindowTitle("Sport Performance Analytics — Dashboard")
        self.setMinimumSize(1200, 760)
        self.resize(1380, 860)

        self._load_artifacts()
        self._build_ui()
        self.setStyleSheet(STYLESHEET)

    # ──────────────────────────────────────────────────────────────────
    # LOAD ML ARTIFACTS
    # ──────────────────────────────────────────────────────────────────
    def _load_artifacts(self):
        self._artifact_errors = []
        for name, attr in [("best_model.pkl","pipeline"),("label_encoder.pkl","encoder")]:
            try:    setattr(self, attr, joblib.load(ARTIFACT_DIR / name))
            except Exception as e: self._artifact_errors.append(f"{name}: {e}")
        try:
            self.lookup = (pd.read_csv(ARTIFACT_DIR/"class_recommendation_lookup.csv")
                           .set_index("target_label"))
        except Exception as e:
            self._artifact_errors.append(f"class_recommendation_lookup.csv: {e}")
        self._artifacts_loaded = len(self._artifact_errors) == 0

    # ──────────────────────────────────────────────────────────────────
    # BUILD UI
    # ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        root.addWidget(self._build_topbar())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_predict_tab(),       "  🎯  Prédiction")
        self.tabs.addTab(self._build_viz_tab(),           "  📊  Visualisations")
        self.tabs.addTab(self._build_analytics_tab(),     "  🔬  Importance")
        self.tabs.addTab(self._build_history_tab(),       "  📋  Historique")
        root.addWidget(self.tabs)

    # ──────────────────────────────────────────────────────────────────
    # TOP BAR
    # ──────────────────────────────────────────────────────────────────
    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(f"background-color:#0c0c10; border-bottom:1px solid {BORDER};")
        bar.setFixedHeight(56)
        lay = QHBoxLayout(bar); lay.setContentsMargins(24,0,24,0)

        lbl = QLabel("⚡ Sport Performance Analytics")
        lbl.setStyleSheet(f"font-size:16px; font-weight:700; color:{TEXT_PRI};")
        lay.addWidget(lbl)
        sub = QLabel("— ML-powered training recommendation engine")
        sub.setStyleSheet(f"color:{TEXT_SEC}; font-size:12px; margin-left:6px;")
        lay.addWidget(sub)
        lay.addStretch()

        if self._artifacts_loaded:
            ct, cs = "  ● Modèle chargé", f"background:#0d2b1a;color:#22c55e;border:1px solid #16a34a;"
        else:
            ct, cs = "  ● Modèle introuvable", f"background:#2b0a0a;color:#ef4444;border:1px solid #dc2626;"
        chip = QLabel(ct)
        chip.setStyleSheet(f"{cs} border-radius:20px; padding:5px 14px; font-size:11px; font-weight:600;")
        lay.addWidget(chip)
        return bar

    # ──────────────────────────────────────────────────────────────────
    # TAB 1 — PRÉDICTION
    # ──────────────────────────────────────────────────────────────────
    def _build_predict_tab(self) -> QWidget:
        tab = QWidget()
        lay = QHBoxLayout(tab)
        lay.setContentsMargins(20,20,20,20); lay.setSpacing(16)
        lay.addWidget(self._build_form_card(), 1)
        lay.addWidget(self._build_result_card(), 1)
        return tab

    def _build_form_card(self) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        lay = QVBoxLayout(host); lay.setContentsMargins(20,16,20,20); lay.setSpacing(14)

        if not self._artifacts_loaded:
            w = QLabel(f"⚠  Artefacts ML introuvables — prédiction désactivée\n"
                       + "\n".join(f"  • {e}" for e in self._artifact_errors))
            w.setStyleSheet("color:#fb923c;background:#2b1208;border:1px solid #c2410c;"
                            "border-radius:8px;padding:12px;font-size:12px;")
            w.setWordWrap(True); lay.addWidget(w)

        self.input_widgets = {}
        # Numeric grid
        h = QLabel("MÉTRIQUES CAPTEURS & PERFORMANCE"); h.setObjectName("section-title")
        lay.addWidget(h)
        g = QGridLayout(); g.setHorizontalSpacing(14); g.setVerticalSpacing(10)
        for i,(key,label,default) in enumerate(NUMERIC_FIELDS):
            r,c = divmod(i,2)
            lb = QLabel(label); lb.setObjectName("field-label")
            g.addWidget(lb, r*2, c*2)
            inp = QLineEdit(default); inp.setMinimumWidth(180)
            g.addWidget(inp, r*2+1, c*2)
            self.input_widgets[key] = inp
        lay.addLayout(g)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background:{BORDER}; max-height:1px;"); lay.addWidget(sep)

        # Categorical grid
        h2 = QLabel("PARAMÈTRES DE CLASSIFICATION"); h2.setObjectName("section-title")
        lay.addWidget(h2)
        g2 = QGridLayout(); g2.setHorizontalSpacing(14); g2.setVerticalSpacing(10)
        for i,(key,label,opts,default) in enumerate(CATEGORICAL_FIELDS):
            r,c = divmod(i,2)
            lb = QLabel(label); lb.setObjectName("field-label")
            g2.addWidget(lb, r*2, c*2)
            cmb = QComboBox(); cmb.addItems(opts); cmb.setCurrentText(default)
            cmb.setMinimumWidth(180)
            g2.addWidget(cmb, r*2+1, c*2)
            self.input_widgets[key] = cmb
        lay.addLayout(g2)
        lay.addStretch()

        # Buttons
        br = QHBoxLayout(); br.addStretch()
        self.btn_reset = QPushButton("↺  Réinitialiser"); self.btn_reset.setObjectName("btn-secondary")
        self.btn_reset.setFixedHeight(44); self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self._reset_fields); br.addWidget(self.btn_reset)
        self.btn_predict = QPushButton("  Prédire  →"); self.btn_predict.setObjectName("btn-predict")
        self.btn_predict.setFixedHeight(44); self.btn_predict.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_predict.setEnabled(self._artifacts_loaded)
        self.btn_predict.clicked.connect(self._run_prediction); br.addWidget(self.btn_predict)
        lay.addLayout(br)

        self.progress = QProgressBar(); self.progress.setRange(0,0)
        self.progress.setVisible(False); self.progress.setFixedHeight(4)
        lay.addWidget(self.progress)

        scroll.setWidget(host)
        out = QVBoxLayout(card); out.setContentsMargins(0,0,0,0); out.addWidget(scroll)
        return card

    def _build_result_card(self) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        lay = QVBoxLayout(card); lay.setContentsMargins(20,16,20,20); lay.setSpacing(12)

        h = QLabel("RÉSULTAT DE LA PRÉDICTION"); h.setObjectName("section-title"); lay.addWidget(h)

        self.gauge_pred = GaugeCanvas(); lay.addWidget(self.gauge_pred, 2)

        # Meta badges
        mr = QHBoxLayout(); mr.setSpacing(8)
        self.meta_labels = []
        for _ in range(3):
            ml = QLabel()
            ml.setStyleSheet(f"background:#1e1e28;color:{TEXT_SEC};border:1px solid {BORDER};"
                             "border-radius:6px;padding:4px 10px;font-size:11px;")
            ml.setVisible(False); mr.addWidget(ml); self.meta_labels.append(ml)
        mr.addStretch(); lay.addLayout(mr)

        rh = QLabel("RECOMMANDATION D'ENTRAÎNEMENT"); rh.setObjectName("section-title"); lay.addWidget(rh)
        self.result_box = QTextEdit(); self.result_box.setObjectName("result-box")
        self.result_box.setReadOnly(True); self.result_box.setMinimumHeight(150)
        self.result_box.setPlaceholderText("Remplissez le formulaire et cliquez sur  Prédire  →")
        lay.addWidget(self.result_box, 3)
        return card

    # ──────────────────────────────────────────────────────────────────
    # TAB 2 — VISUALISATIONS
    # ──────────────────────────────────────────────────────────────────
    def _build_viz_tab(self) -> QWidget:
        tab = QWidget()
        lay = QGridLayout(tab); lay.setContentsMargins(20,20,20,20); lay.setSpacing(16)

        self.gauge_viz = GaugeCanvas()
        self.radar_viz = RadarCanvas()
        self.donut_viz = RiskDonutCanvas()

        lay.addWidget(self._card(self.gauge_viz), 0, 0)
        lay.addWidget(self._card(self.radar_viz), 0, 1)
        lay.addWidget(self._card(self.donut_viz), 1, 0)
        lay.addWidget(self._build_kpi_panel(),    1, 1)

        lay.setRowStretch(0, 1); lay.setRowStretch(1, 1)
        lay.setColumnStretch(0, 1); lay.setColumnStretch(1, 1)
        return tab

    def _card(self, widget) -> QFrame:
        f = QFrame(); f.setObjectName("card")
        l = QVBoxLayout(f); l.setContentsMargins(10,10,10,10); l.addWidget(widget)
        return f

    def _build_kpi_panel(self) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        lay = QGridLayout(card); lay.setContentsMargins(20,16,20,20); lay.setSpacing(14)
        hdr = QLabel("INDICATEURS CLÉS"); hdr.setObjectName("section-title")
        lay.addWidget(hdr, 0, 0, 1, 2)

        kpi_defs = [
            ("💓","Fréq. Cardiaque","heart_rate_bpm",   ACCENT),
            ("🏃","Freq. de Pas",   "step_frequency_hz","#22c55e"),
            ("📏","Longueur Foulée","stride_length_m",  "#f59e0b"),
            ("⚡","Énergie Signal", "signal_energy",    "#6366f1"),
        ]
        self.kpi_widgets = {}
        for i,(icon,label,key,color) in enumerate(kpi_defs):
            r,c = i//2+1, i%2
            f = QFrame()
            f.setStyleSheet(f"background:#12121a;border:1px solid {BORDER};border-radius:8px;")
            fl = QVBoxLayout(f); fl.setContentsMargins(14,12,14,12); fl.setSpacing(4)
            il = QLabel(f"{icon}  {label}")
            il.setStyleSheet(f"color:{color};font-size:10px;font-weight:600;")
            vl = QLabel("—")
            vl.setStyleSheet(f"color:{TEXT_PRI};font-size:24px;font-weight:700;")
            fl.addWidget(il); fl.addWidget(vl)
            lay.addWidget(f, r, c)
            self.kpi_widgets[key] = vl
        lay.setRowStretch(3,1)
        return card

    # ──────────────────────────────────────────────────────────────────
    # TAB 3 — ANALYTICS
    # ──────────────────────────────────────────────────────────────────
    def _build_analytics_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab); lay.setContentsMargins(20,20,20,20); lay.setSpacing(0)
        card = QFrame(); card.setObjectName("card")
        cl = QVBoxLayout(card); cl.setContentsMargins(14,14,14,14)
        cl.addWidget(FeatureImportanceCanvas())
        lay.addWidget(card)
        return tab

    # ──────────────────────────────────────────────────────────────────
    # TAB 4 — HISTORIQUE
    # ──────────────────────────────────────────────────────────────────
    def _build_history_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab); lay.setContentsMargins(20,20,20,20); lay.setSpacing(16)

        self.history_chart = HistoryScoreCanvas()
        chart_card = self._card(self.history_chart)
        lay.addWidget(chart_card, 1)

        # Table card
        tc = QFrame(); tc.setObjectName("card")
        tl = QVBoxLayout(tc); tl.setContentsMargins(16,12,16,16); tl.setSpacing(10)

        hr_ = QHBoxLayout()
        hh = QLabel("HISTORIQUE DES PRÉDICTIONS"); hh.setObjectName("section-title"); hr_.addWidget(hh)
        hr_.addStretch()
        for text, slot in [("⬇  Exporter CSV", self._export_history),
                           ("🗑  Effacer",      self._clear_history)]:
            b = QPushButton(f"  {text}"); b.setObjectName("btn-secondary")
            b.setFixedHeight(34); b.clicked.connect(slot); hr_.addWidget(b)
        tl.addLayout(hr_)

        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(
            ["#","Épreuve","Risque","Zone HR","Niveau","Score","Recommandation"])
        self.history_table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.Stretch)
        for i in range(6):
            self.history_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setMinimumHeight(180)
        tl.addWidget(self.history_table)
        lay.addWidget(tc, 1)
        return tab

    # ──────────────────────────────────────────────────────────────────
    # PREDICTION LOGIC
    # ──────────────────────────────────────────────────────────────────
    def _collect_inputs(self) -> dict | None:
        features, errors = {}, []
        for key, label, _ in NUMERIC_FIELDS:
            raw = self.input_widgets[key].text().strip()
            try:
                features[key] = float(raw)
                self.input_widgets[key].setStyleSheet("")
            except ValueError:
                errors.append(f"'{label}' doit être un nombre (reçu: {raw!r})")
                self.input_widgets[key].setStyleSheet("border:1px solid #ef4444;")
        for key, label, _, _ in CATEGORICAL_FIELDS:
            features[key] = self.input_widgets[key].currentText()
        if errors:
            QMessageBox.warning(self, "Erreur de saisie",
                                "Veuillez corriger :\n\n" + "\n".join(f"• {e}" for e in errors))
            return None
        return features

    def _run_prediction(self):
        features = self._collect_inputs()
        if features is None: return
        self.btn_predict.setEnabled(False); self.btn_predict.setText("  Prédiction…")
        self.progress.setVisible(True); self.result_box.clear()
        for m in self.meta_labels: m.setVisible(False)

        self.worker = PredictionWorker(self.pipeline, self.encoder, self.lookup, features)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_result(self, res: dict):
        self._reset_loading()
        risk  = res.get("risk","Modéré")
        perf  = res.get("perf","")
        score = res["features"].get("performance_score", 0)
        event = res.get("event","?")

        # Update Prediction tab
        self.gauge_pred.update_score(score, perf)
        meta = [(f"💓 {res.get('zone','')}",""), (f"📈 {perf}",""), (f"🏅 {event}","")]
        for i,(txt,_) in enumerate(meta):
            self.meta_labels[i].setText(txt); self.meta_labels[i].setVisible(True)
        self.result_box.setHtml(self._format_rec(res.get("recommendation","")))

        # Highlight card border by risk
        ac = RISK_COLORS.get(risk,(ACCENT,DARK_BG))[0]
        self.result_box.parent().parent().setStyleSheet(
            f"QFrame#card{{background:{CARD_BG};border:1px solid {ac};border-radius:12px;}}")

        # Update Visualisation tab
        self.gauge_viz.update_score(score, perf)
        self.radar_viz.update_radar(res["features"])
        self.donut_viz.update_risk(risk)
        for key, w in self.kpi_widgets.items():
            val = res["features"].get(key, 0)
            w.setText(f"{val:.2f}")

        # Update History tab
        self.history_chart.add_score(float(score), event)
        row = len(self.prediction_history)
        self.prediction_history.append(res)
        self.history_table.insertRow(row)
        cells = [str(row+1), event, risk, res.get("zone",""), perf,
                 f"{score:.1f}", res.get("recommendation","")[:80]+"…"]
        risk_colors_map = {"Faible":"#22c55e","Modéré":"#f59e0b",
                           "Élevé":"#f97316","Critique":"#ef4444"}
        for col,txt in enumerate(cells):
            item = QTableWidgetItem(txt)
            if col==2:   # Risk
                item.setForeground(QColor(risk_colors_map.get(risk,"#fff")))
            elif col==5: # Score
                sc = float(score)
                item.setForeground(QColor("#22c55e" if sc>=70 else "#f59e0b" if sc>=40 else "#ef4444"))
            self.history_table.setItem(row, col, item)
        self.history_table.scrollToBottom()

        # Switch to visualisations tab
        self.tabs.setCurrentIndex(1)

    def _on_error(self, msg: str):
        self._reset_loading()
        self.result_box.setPlainText(f"Erreur de prédiction :\n\n{msg}")

    def _reset_loading(self):
        self.btn_predict.setEnabled(True); self.btn_predict.setText("  Prédire  →")
        self.progress.setVisible(False)

    def _format_rec(self, text: str) -> str:
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if not parts:
            return f"<p style='color:{TEXT_SEC};'>Aucune recommandation disponible.</p>"
        html = ("<style>ul{margin:0;padding-left:18px;}"
                f"li{{margin-bottom:10px;line-height:1.6;color:#cbd5e1;font-size:13px;}}"
                "li::marker{color:#3b82f6;}"
                ".alert{color:#fca5a5;font-weight:600;}</style><ul>")
        for p in parts:
            cls = "alert" if "ALERTE" in p.upper() or "arrêt" in p.lower() else ""
            html += f"<li class='{cls}'>{p}</li>"
        return html + "</ul>"

    def _reset_fields(self):
        for key, _, default in NUMERIC_FIELDS:
            self.input_widgets[key].setText(default)
            self.input_widgets[key].setStyleSheet("")
        for key, _, _, default in CATEGORICAL_FIELDS:
            self.input_widgets[key].setCurrentText(default)
        self.result_box.clear()
        for m in self.meta_labels: m.setVisible(False)
        self.gauge_pred._draw(None,"")

    def _export_history(self):
        if not self.prediction_history:
            QMessageBox.information(self,"Export","Aucune prédiction à exporter."); return
        path, _ = QFileDialog.getSaveFileName(self,"Exporter l'historique","history.csv",
                                              "CSV Files (*.csv)")
        if path:
            rows = [{
                "#": i+1,
                "event": r.get("event",""), "risk": r.get("risk",""),
                "hr_zone": r.get("zone",""), "perf_level": r.get("perf",""),
                "performance_score": r["features"].get("performance_score",""),
                "recommendation": r.get("recommendation",""),
            } for i,r in enumerate(self.prediction_history)]
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            QMessageBox.information(self,"Export",f"Historique exporté :\n{path}")

    def _clear_history(self):
        self.prediction_history.clear()
        self.history_table.setRowCount(0)
        self.history_chart.clear_data()


# ===========================================================================
#  ENTRY POINT
# ===========================================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Sport Performance Analytics")
    window = SportsDashboardApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()