"""
=============================================================================
  SPORTS INJURY ANALYTICS — Dashboard  (app_gui.py)
  PyQt6 · Matplotlib · joblib · pandas
=============================================================================
  Tabs:
    1. 🩺 Analyse       — Formulaire biométrique + résultat avec jauge de risque
    2. 📊 Visualisations — Radar biométrique + jauge risque + donut + KPIs
    3. 🔬 Importance     — Top features du modèle blessure
    4. 📋 Historique     — Tableau + graphique de toutes les analyses
=============================================================================
"""

import sys
import traceback
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
from PyQt6.QtGui import QColor, QFont

# ===========================================================================
#  THEME
# ===========================================================================
DARK_BG  = "#0d0d11"
CARD_BG  = "#14141c"
ACCENT   = "#6366f1"       # Indigo — injury theme
SUCCESS  = "#22c55e"       # Green  — Faible
WARNING  = "#f59e0b"       # Amber  — Modéré
ELEVATED = "#f97316"       # Orange — Élevé
DANGER   = "#ef4444"       # Red    — Critique
TEXT_PRI = "#e2e8f0"
TEXT_SEC = "#64748b"
BORDER   = "#252530"

RISK_COLOR_MAP = {
    "Faible":   SUCCESS,
    "Modéré":   WARNING,
    "Élevé":    ELEVATED,
    "Critique": DANGER,
}
RISK_BG_MAP = {
    "Faible":   "#0d2b1a",
    "Modéré":   "#2b1d08",
    "Élevé":    "#2b1208",
    "Critique": "#2b0a0a",
}

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
    ("heart_rate_bpm",    "Fréquence Cardiaque (BPM)", "145.0"),
    ("injury_risk_score", "Score de Risque (0–1)",      "0.72"),
    ("step_frequency_hz", "Fréquence de Pas (Hz)",      "-0.88"),
    ("stride_length_m",   "Longueur Foulée (m)",        "1.66"),
    ("acceleration_mps2", "Accélération (m/s²)",        "-0.30"),
    ("signal_energy",     "Énergie Signal",             "1.05"),
    ("dominant_freq_hz",  "Fréquence Dom. (Hz)",        "1.05"),
    ("performance_score", "Score Performance",          "55.0"),
]

CATEGORICAL_FIELDS = [
    ("risk_level",  "Niveau de Risque",
     ["Faible", "Modéré", "Élevé", "Critique"],                  "Élevé"),
    ("hr_zone",     "Zone Cardiaque",
     ["Zone_Basse","Zone_Aerobie","Zone_Anaerobie","Zone_Max"],   "Zone_Anaerobie"),
    ("event_type",  "Type d'Épreuve",
     ["sprint","high_jump","long_jump","unknown"],                "sprint"),
    ("motion_class","Phase de Mouvement",
     ["acceleration_phase","flight_phase","landing","start_phase"],"landing"),
]

RADAR_KEYS   = [
    "heart_rate_bpm","injury_risk_score","step_frequency_hz",
    "stride_length_m","signal_energy","acceleration_mps2",
]
RADAR_LABELS = ["Fréq. Card.","Risque","Fréq. Pas","Foulée","Signal E","Accél."]

# Map integer risk score (0–1) to percentage for gauge
RISK_SCORE_MAX = 1.0

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
#  RISK GAUGE  (score 0–1 mapped to 0–100 on gauge)
# ===========================================================================
class RiskGaugeCanvas(MplCanvas):
    def __init__(self):
        super().__init__(4, 2.8, 100)
        self._draw(None, "")

    def _draw(self, risk_score, risk_level):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.35, 1.3)
        ax.axis("off")

        # Background arc
        t_bg = np.linspace(np.pi, 0, 300)
        ax.plot(np.cos(t_bg), np.sin(t_bg),
                color="#1e1e2c", linewidth=18, solid_capstyle="round")

        # Zone bands (Faible / Modéré / Élevé / Critique)
        for lo, hi, col in [
            (0, 30,  SUCCESS),
            (30, 60, WARNING),
            (60, 80, ELEVATED),
            (80, 100, DANGER),
        ]:
            t = np.linspace(np.pi - (lo/100)*np.pi, np.pi - (hi/100)*np.pi, 100)
            ax.plot(np.cos(t), np.sin(t), color=col, linewidth=18,
                    solid_capstyle="butt", alpha=0.15)

        if risk_score is None:
            ax.text(0, 0.35, "—", ha="center", va="center",
                    fontsize=32, fontweight="bold", color=TEXT_SEC)
        else:
            s     = max(0.0, min(1.0, float(risk_score))) * 100  # convert to %
            fc    = RISK_COLOR_MAP.get(risk_level, DANGER) if risk_level else DANGER
            t_fill = np.linspace(np.pi, np.pi - (s/100)*np.pi, 300)
            ax.plot(np.cos(t_fill), np.sin(t_fill), color=fc, linewidth=18,
                    solid_capstyle="round",
                    path_effects=[
                        pe.Stroke(linewidth=22, foreground=fc, alpha=0.25),
                        pe.Normal(),
                    ])
            ang = np.pi - (s/100)*np.pi
            ax.annotate(
                "", xy=(0.75*np.cos(ang), 0.75*np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#fff", lw=2, mutation_scale=14)
            )
            ax.plot(0, 0, "o", color="#fff", markersize=8, zorder=10)
            ax.text(0, 0.35, f"{s:.0f}%", ha="center", va="center",
                    fontsize=30, fontweight="bold", color="#fff")
            ax.text(0, 0.12, "risque", ha="center", va="center",
                    fontsize=9, color=TEXT_SEC)
            if risk_level:
                bc = RISK_COLOR_MAP.get(risk_level, ACCENT)
                ax.text(0, -0.20, risk_level.upper(), ha="center", va="center",
                        fontsize=9, fontweight="bold", color=bc,
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor=CARD_BG, edgecolor=bc, lw=1.5))
            ax.text(-1.2, -0.10, "0",    ha="center", va="center",
                    fontsize=8, color=TEXT_SEC)
            ax.text( 1.2, -0.10, "100%", ha="center", va="center",
                    fontsize=8, color=TEXT_SEC)

        ax.text(0, 0.80, "SCORE DE RISQUE BLESSURE", ha="center", va="center",
                fontsize=8, fontweight="600", color=TEXT_SEC)
        self.fig.tight_layout(pad=0.2)
        self.draw()

    def update_risk(self, risk_score, risk_level=""):
        self._draw(risk_score, risk_level)


# ===========================================================================
#  RADAR CHART
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

        raw  = np.array([float(features.get(k, 0)) for k in RADAR_KEYS])

        # injury_risk_score is already 0–1
        norm = np.zeros(N)
        for i, key in enumerate(RADAR_KEYS):
            val = float(features.get(key, 0))
            if key == "injury_risk_score":
                norm[i] = np.clip(val, 0, 1)
            else:
                norm[i] = np.clip((val + 3) / 6, 0, 1)

        values  = norm.tolist() + [norm[0]]
        fill_c  = DANGER if (features.get("injury_risk_score", 0) or 0) > 0.6 else ACCENT

        ax.fill(angles, values, color=fill_c, alpha=0.20)
        ax.plot(angles, values, color=fill_c, linewidth=2)
        ax.scatter(angles[:-1], values[:-1], s=45, color=fill_c, zorder=10)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_LABELS, fontsize=8, color=TEXT_PRI)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""])
        ax.set_ylim(0, 1)
        ax.grid(color=BORDER, linewidth=0.8, alpha=0.7)
        ax.spines["polar"].set_color(BORDER)
        ax.set_title("PROFIL BIOMÉTRIQUE BLESSURE", color=TEXT_SEC,
                     fontsize=9, fontweight="600", pad=14)
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

        levels = ["Faible", "Modéré", "Élevé", "Critique"]
        clrs   = [SUCCESS, WARNING, ELEVATED, DANGER]
        idx    = levels.index(risk) if risk in levels else -1
        sizes  = [2 if i == idx else 1 for i in range(4)]
        alphas = [1.0 if i == idx else 0.18 for i in range(4)]
        expl   = [0.06 if i == idx else 0 for i in range(4)]

        wedges, _ = ax.pie(
            sizes, colors=clrs, startangle=90, explode=expl,
            wedgeprops=dict(width=0.45, edgecolor=DARK_BG, linewidth=2),
            counterclock=False,
        )
        for w, a in zip(wedges, alphas):
            w.set_alpha(a)

        if risk:
            ac = RISK_COLOR_MAP.get(risk, ACCENT)
            ax.text(0, 0, risk.upper(), ha="center", va="center",
                    fontsize=9, fontweight="bold", color=ac)

        ax.set_title("NIVEAU DE RISQUE", color=TEXT_SEC,
                     fontsize=9, fontweight="600", pad=8)
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def update_risk(self, risk: str):
        self._draw(risk)


# ===========================================================================
#  FEATURE IMPORTANCE CHART
# ===========================================================================
class InjuryFeatImportanceCanvas(MplCanvas):
    def __init__(self):
        super().__init__(7, 5, 100)
        self._draw()

    def _draw(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor=CARD_BG)
        try:
            fp = ARTIFACT_DIR / "injury_feature_importance.csv"
            if not fp.exists():
                raise FileNotFoundError(
                    "injury_feature_importance.csv introuvable.\n"
                    "Lancez d'abord train.py pour entraîner le modèle."
                )
            df = pd.read_csv(fp).dropna()
            df.columns = ["Feature", "Score"]
            df = df.sort_values("Score", ascending=True).tail(14)

            def _col(f):
                if "injury_risk"    in f: return DANGER
                if "risk_level"    in f: return ELEVATED
                if "hr_zone"       in f: return "#3b82f6"
                if "heart_rate"    in f: return "#ec4899"
                if "event_type"    in f: return SUCCESS
                if "motion_class"  in f: return "#a78bfa"
                if "performance"   in f: return WARNING
                return ACCENT

            colors = [_col(f) for f in df["Feature"]]
            bars   = ax.barh(df["Feature"], df["Score"], color=colors,
                             height=0.62, edgecolor=DARK_BG, linewidth=0.5)
            for bar, val in zip(bars, df["Score"]):
                ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=8, color=TEXT_SEC)

            ax.set_xlabel("Importance du Feature", fontsize=9, color=TEXT_SEC)
            ax.tick_params(colors=TEXT_SEC, labelsize=8)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            for sp in ["left", "bottom"]:
                ax.spines[sp].set_color(BORDER)
            ax.set_xlim(0, df["Score"].max() * 1.18)
            ax.grid(axis="x", alpha=0.2)

            legend_items = [
                mpatches.Patch(color=DANGER,   label="Injury Risk Score"),
                mpatches.Patch(color=ELEVATED, label="Risk Level"),
                mpatches.Patch(color="#3b82f6",label="HR Zone"),
                mpatches.Patch(color="#ec4899",label="Heart Rate"),
                mpatches.Patch(color=SUCCESS,  label="Event Type"),
                mpatches.Patch(color="#a78bfa",label="Motion Class"),
                mpatches.Patch(color=WARNING,  label="Performance"),
            ]
            ax.legend(handles=legend_items, fontsize=8, framealpha=0.15,
                      labelcolor=TEXT_PRI, loc="lower right")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    color=DANGER, transform=ax.transAxes, fontsize=10,
                    wrap=True)
            ax.axis("off")

        ax.set_title("IMPORTANCE DES FEATURES — MODÈLE BLESSURE",
                     color=TEXT_PRI, fontsize=12, fontweight="700", pad=14)
        self.fig.tight_layout(pad=1.4)
        self.draw()


# ===========================================================================
#  HISTORY CHART
# ===========================================================================
class HistoryRiskCanvas(MplCanvas):
    def __init__(self):
        super().__init__(7, 3, 100)
        self.scores = []
        self.labels = []
        self._draw()

    def add_score(self, score: float, event: str):
        self.scores.append(score * 100)   # store as %
        self.labels.append(f"#{len(self.scores)} {event[:5]}")
        self._draw()

    def clear_data(self):
        self.scores = []
        self.labels = []
        self._draw()

    def _draw(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor=CARD_BG)

        if not self.scores:
            ax.text(0.5, 0.5, "Aucune analyse dans cette session",
                    ha="center", va="center", color=TEXT_SEC,
                    transform=ax.transAxes, fontsize=11)
            ax.axis("off")
        else:
            x      = np.arange(len(self.scores))
            colors = [
                DANGER   if s >= 80 else
                ELEVATED if s >= 60 else
                WARNING  if s >= 30 else
                SUCCESS
                for s in self.scores
            ]
            ax.bar(x, self.scores, color=colors, alpha=0.75,
                   width=0.55, edgecolor=DARK_BG, linewidth=0.5)
            ax.plot(x, self.scores, color=ACCENT, linewidth=2,
                    marker="o", markersize=7, zorder=5)
            for xi, s in zip(x, self.scores):
                ax.text(xi, s + 1.5, f"{s:.0f}%",
                        ha="center", va="bottom", fontsize=8,
                        color=TEXT_PRI, fontweight="600")
            ax.axhline(80, color=DANGER,   lw=1.2, ls="--", alpha=0.5, label="Critique (80%)")
            ax.axhline(60, color=ELEVATED, lw=1.2, ls="--", alpha=0.5, label="Élevé (60%)")
            ax.axhline(30, color=WARNING,  lw=1.2, ls="--", alpha=0.5, label="Modéré (30%)")
            ax.set_xticks(x)
            ax.set_xticklabels(self.labels, fontsize=9, color=TEXT_SEC)
            ax.set_ylabel("Risque (%)", fontsize=9, color=TEXT_SEC)
            ax.set_ylim(0, 112)
            ax.tick_params(colors=TEXT_SEC)
            for sp in ["top", "right"]:  ax.spines[sp].set_visible(False)
            for sp in ["left", "bottom"]: ax.spines[sp].set_color(BORDER)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8, framealpha=0.1, labelcolor=TEXT_PRI)

        ax.set_title("HISTORIQUE DES SCORES DE RISQUE BLESSURE",
                     color=TEXT_PRI, fontsize=11, fontweight="700", pad=12)
        self.fig.tight_layout(pad=1.0)
        self.draw()


# ===========================================================================
#  PREDICTION WORKER (background thread)
# ===========================================================================
class PredictionWorker(QThread):
    result_ready = pyqtSignal(dict)
    error        = pyqtSignal(str)

    RISK_ENCODING = {"Faible": 0, "Modéré": 1, "Élevé": 2, "Critique": 3}
    HR_ENCODING   = {
        "Zone_Basse": 0, "Zone_Aerobie": 1,
        "Zone_Anaerobie": 2, "Zone_Max": 3,
    }
    MOTION_MAP = {
        "acceleration_phase": "motion_class_acceleration_phase",
        "flight_phase":       "motion_class_flight_phase",
        "landing":            "motion_class_landing",
        "start_phase":        "motion_class_start_phase",
    }

    def __init__(self, pipeline, encoder, lookup, features: dict):
        super().__init__()
        self.pipeline = pipeline
        self.encoder  = encoder
        self.lookup   = lookup
        self.features = dict(features)

    def run(self):
        try:
            f = self.features

            # Derived ordinal encodings
            f.setdefault("risk_level_encoded",
                         self.RISK_ENCODING.get(f.get("risk_level","Modéré"), 1))
            f.setdefault("hr_zone_encoded",
                         self.HR_ENCODING.get(f.get("hr_zone","Zone_Aerobie"), 1))

            # Binary event type flags
            event = f.get("event_type", "unknown")
            f["event_type_high_jump"] = int(event == "high_jump")
            f["event_type_long_jump"] = int(event == "long_jump")
            f["event_type_sprint"]    = int(event == "sprint")

            # Binary motion class flags
            motion = f.pop("motion_class", "")
            for key, col in self.MOTION_MAP.items():
                f[col] = int(motion == key)

            # Default missing numeric sensors
            for col in ["gyroscope_x","gyroscope_y","gyroscope_z",
                        "accelerometer_x","accelerometer_y","accelerometer_z"]:
                f.setdefault(col, 0.0)

            df      = pd.DataFrame([f])
            encoded = self.pipeline.predict(df)[0]
            label   = self.encoder.inverse_transform([encoded])[0]
            rec     = (self.lookup.loc[label, "recommendation_blessure"]
                       if label in self.lookup.index
                       else f"(aucune recommendation pour: {label})")

            parts = label.split("__")
            self.result_ready.emit({
                "label":      label,
                "recommendation": rec,
                "risk":       parts[0] if len(parts) > 0 else "Modéré",
                "zone":       parts[1] if len(parts) > 1 else "",
                "event":      parts[2] if len(parts) > 2 else "",
                "features":   f,
            })
        except Exception as exc:
            self.error.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )


# ===========================================================================
#  STYLESHEET
# ===========================================================================
STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG}; color: {TEXT_PRI};
    font-family: 'Segoe UI', sans-serif; font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER}; border-radius: 12px;
    background-color: {DARK_BG};
}}
QTabBar::tab {{
    background-color: #10101a; color: {TEXT_SEC};
    padding: 11px 22px; border: none;
    border-bottom: 2px solid transparent;
    font-size: 12px; font-weight: 600; min-width: 130px;
}}
QTabBar::tab:selected {{
    color: {ACCENT}; border-bottom: 2px solid {ACCENT};
    background-color: {DARK_BG};
}}
QTabBar::tab:hover {{ color: {TEXT_PRI}; background-color: #181824; }}
QFrame#card {{
    background-color: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 12px;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: #181824; width: 6px; border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: #38384a; border-radius: 3px; min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QLabel#section-title {{
    color: {TEXT_SEC}; font-size: 10px; font-weight: 600;
    letter-spacing: 2px;
}}
QLabel#field-label {{
    color: #cbd5e1; font-size: 12px; font-weight: 500;
}}
QLineEdit, QComboBox {{
    background-color: #1a1a26; border: 1px solid #2c2c3e;
    border-radius: 8px; padding: 8px 12px;
    color: {TEXT_PRI}; font-size: 13px;
}}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SEC};
    width: 0; height: 0; margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: #1a1a26; border: 1px solid #2c2c3e;
    border-radius: 8px; color: {TEXT_PRI};
    selection-background-color: {ACCENT};
}}
QPushButton#btn-analyse {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ELEVATED},stop:1 {DANGER});
    color: #fff; font-size: 13px; font-weight: 600;
    border: none; border-radius: 10px; padding: 12px 28px;
}}
QPushButton#btn-analyse:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #e06010,stop:1 #dc2626);
}}
QPushButton#btn-analyse:disabled {{ background: #252530; color: #44445a; }}
QPushButton#btn-secondary {{
    background-color: transparent; color: {TEXT_SEC};
    font-size: 12px; border: 1px solid {BORDER};
    border-radius: 8px; padding: 8px 16px;
}}
QPushButton#btn-secondary:hover {{
    background-color: #1a1a26; color: {TEXT_PRI};
}}
QTextEdit#result-box {{
    background-color: #10101a; border: 1px solid {BORDER};
    border-radius: 10px; color: {TEXT_PRI};
    font-size: 13px; padding: 12px;
}}
QProgressBar {{
    background-color: #1a1a26; border: none;
    border-radius: 3px; height: 4px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT},stop:1 {DANGER});
    border-radius: 3px;
}}
QTableWidget {{
    background-color: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 10px; gridline-color: {BORDER};
    color: {TEXT_PRI}; font-size: 12px;
    selection-background-color: #1e1e3a;
    alternate-background-color: #10101a;
}}
QTableWidget::item {{ padding: 8px 12px; border-bottom: 1px solid {BORDER}; }}
QHeaderView::section {{
    background-color: #10101a; color: {TEXT_SEC};
    font-size: 10px; font-weight: 600; letter-spacing: 1px;
    padding: 10px 12px; border: none;
    border-bottom: 1px solid {BORDER};
}}
QToolTip {{
    background-color: #1a1a26; color: #cbd5e1;
    border: 1px solid #3a3a50; border-radius: 6px; padding: 6px 10px;
}}
"""


# ===========================================================================
#  MAIN APPLICATION WINDOW
# ===========================================================================
class InjuryDashboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pipeline = None; self.encoder = None
        self.lookup   = None; self.worker  = None
        self.prediction_history = []

        self.setWindowTitle("🩺 Sport Injury Analytics — Dashboard")
        self.setMinimumSize(1200, 760)
        self.resize(1400, 880)

        self._load_artifacts()
        self._build_ui()
        self.setStyleSheet(STYLESHEET)

    # ── LOAD ML ARTIFACTS ──────────────────────────────────────────────
    def _load_artifacts(self):
        self._artifact_errors = []
        for fname, attr in [
            ("injury_model.pkl",   "pipeline"),
            ("injury_encoder.pkl", "encoder"),
        ]:
            try:
                setattr(self, attr, joblib.load(ARTIFACT_DIR / fname))
            except Exception as e:
                self._artifact_errors.append(f"{fname}: {e}")
        try:
            self.lookup = (
                pd.read_csv(ARTIFACT_DIR / "injury_lookup.csv")
                .set_index("target_label")
            )
        except Exception as e:
            self._artifact_errors.append(f"injury_lookup.csv: {e}")
        self._artifacts_loaded = len(self._artifact_errors) == 0

    # ── BUILD UI ───────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(self._build_topbar())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_analyse_tab(),  "  🩺  Analyse")
        self.tabs.addTab(self._build_viz_tab(),      "  📊  Visualisations")
        self.tabs.addTab(self._build_analytics_tab(),"  🔬  Importance")
        self.tabs.addTab(self._build_history_tab(),  "  📋  Historique")
        root.addWidget(self.tabs)

    # ── TOP BAR ────────────────────────────────────────────────────────
    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(f"background-color:#09090f; border-bottom:1px solid {BORDER};")
        bar.setFixedHeight(56)
        lay = QHBoxLayout(bar); lay.setContentsMargins(24, 0, 24, 0)

        title = QLabel("🩺 Sport Injury Analytics")
        title.setStyleSheet(f"font-size:16px; font-weight:700; color:{TEXT_PRI};")
        lay.addWidget(title)

        sub = QLabel("— ML-powered injury recommendation engine")
        sub.setStyleSheet(f"color:{TEXT_SEC}; font-size:12px; margin-left:6px;")
        lay.addWidget(sub)
        lay.addStretch()

        if self._artifacts_loaded:
            ct = "  ● Modèle blessure chargé"
            cs = f"background:#0d2b1a;color:{SUCCESS};border:1px solid #16a34a;"
        else:
            ct = "  ● Modèle introuvable — lancez train.py"
            cs = f"background:#2b0a0a;color:{DANGER};border:1px solid #dc2626;"

        chip = QLabel(ct)
        chip.setStyleSheet(
            f"{cs} border-radius:20px; padding:5px 14px; font-size:11px; font-weight:600;"
        )
        lay.addWidget(chip)
        return bar

    # ── TAB 1 — ANALYSE ────────────────────────────────────────────────
    def _build_analyse_tab(self) -> QWidget:
        tab = QWidget()
        lay = QHBoxLayout(tab)
        lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(16)
        lay.addWidget(self._build_form_card(), 1)
        lay.addWidget(self._build_result_card(), 1)
        return tab

    def _build_form_card(self) -> QFrame:
        card   = QFrame(); card.setObjectName("card")
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        lay  = QVBoxLayout(host); lay.setContentsMargins(20, 16, 20, 20)
        lay.setSpacing(14)

        # Warning if artifacts not loaded
        if not self._artifacts_loaded:
            w = QLabel(
                "⚠  Modèle ML introuvable — analyse désactivée\n\n"
                "Lancez d'abord :\n  python analyser__blessure/train.py\n\n"
                + "\n".join(f"  • {e}" for e in self._artifact_errors)
            )
            w.setStyleSheet(
                f"color:{ELEVATED};background:#2b1208;border:1px solid #c2410c;"
                "border-radius:8px;padding:12px;font-size:12px;"
            )
            w.setWordWrap(True); lay.addWidget(w)

        self.input_widgets = {}

        # Numeric fields
        h = QLabel("MÉTRIQUES BIOMÉTRIQUES"); h.setObjectName("section-title")
        lay.addWidget(h)
        g = QGridLayout(); g.setHorizontalSpacing(14); g.setVerticalSpacing(10)
        for i, (key, label, default) in enumerate(NUMERIC_FIELDS):
            r, c = divmod(i, 2)
            lb = QLabel(label); lb.setObjectName("field-label")
            g.addWidget(lb, r*2, c*2)
            inp = QLineEdit(default); inp.setMinimumWidth(180)
            inp.setToolTip(f"Valeur pour : {label}")
            g.addWidget(inp, r*2+1, c*2)
            self.input_widgets[key] = inp
        lay.addLayout(g)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background:{BORDER}; max-height:1px;"); lay.addWidget(sep)

        # Categorical fields
        h2 = QLabel("PARAMÈTRES DE CLASSIFICATION"); h2.setObjectName("section-title")
        lay.addWidget(h2)
        g2 = QGridLayout(); g2.setHorizontalSpacing(14); g2.setVerticalSpacing(10)
        for i, (key, label, opts, default) in enumerate(CATEGORICAL_FIELDS):
            r, c = divmod(i, 2)
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
        self.btn_reset = QPushButton("↺  Réinitialiser")
        self.btn_reset.setObjectName("btn-secondary")
        self.btn_reset.setFixedHeight(44)
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self._reset_fields)
        br.addWidget(self.btn_reset)

        self.btn_analyse = QPushButton("  🩺  Analyser  →")
        self.btn_analyse.setObjectName("btn-analyse")
        self.btn_analyse.setFixedHeight(44)
        self.btn_analyse.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_analyse.setEnabled(self._artifacts_loaded)
        self.btn_analyse.clicked.connect(self._run_prediction)
        br.addWidget(self.btn_analyse)
        lay.addLayout(br)

        self.progress = QProgressBar(); self.progress.setRange(0, 0)
        self.progress.setVisible(False); self.progress.setFixedHeight(4)
        lay.addWidget(self.progress)

        scroll.setWidget(host)
        out = QVBoxLayout(card); out.setContentsMargins(0, 0, 0, 0)
        out.addWidget(scroll)
        return card

    def _build_result_card(self) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        lay  = QVBoxLayout(card); lay.setContentsMargins(20, 16, 20, 20)
        lay.setSpacing(12)

        h = QLabel("RÉSULTAT DE L'ANALYSE BLESSURE")
        h.setObjectName("section-title"); lay.addWidget(h)

        self.gauge_pred = RiskGaugeCanvas(); lay.addWidget(self.gauge_pred, 2)

        # Meta badges
        mr = QHBoxLayout(); mr.setSpacing(8)
        self.meta_labels = []
        for _ in range(3):
            ml = QLabel()
            ml.setStyleSheet(
                f"background:#1a1a26;color:{TEXT_SEC};border:1px solid {BORDER};"
                "border-radius:6px;padding:4px 10px;font-size:11px;"
            )
            ml.setVisible(False); mr.addWidget(ml); self.meta_labels.append(ml)
        mr.addStretch(); lay.addLayout(mr)

        rh = QLabel("RECOMMANDATION DE PRÉVENTION")
        rh.setObjectName("section-title"); lay.addWidget(rh)

        self.result_box = QTextEdit(); self.result_box.setObjectName("result-box")
        self.result_box.setReadOnly(True); self.result_box.setMinimumHeight(160)
        self.result_box.setPlaceholderText(
            "Remplissez le formulaire et cliquez sur  🩺 Analyser  →"
        )
        lay.addWidget(self.result_box, 3)
        return card

    # ── TAB 2 — VISUALISATIONS ─────────────────────────────────────────
    def _build_viz_tab(self) -> QWidget:
        tab = QWidget()
        lay = QGridLayout(tab); lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(16)

        self.gauge_viz = RiskGaugeCanvas()
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
        l = QVBoxLayout(f); l.setContentsMargins(10, 10, 10, 10)
        l.addWidget(widget)
        return f

    def _build_kpi_panel(self) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        lay  = QGridLayout(card); lay.setContentsMargins(20, 16, 20, 20)
        lay.setSpacing(14)
        hdr = QLabel("INDICATEURS CLÉS DE RISQUE"); hdr.setObjectName("section-title")
        lay.addWidget(hdr, 0, 0, 1, 2)

        kpi_defs = [
            ("🚨", "Score de Risque",    "injury_risk_score", DANGER),
            ("💓", "Fréq. Cardiaque",   "heart_rate_bpm",    ACCENT),
            ("🏃", "Freq. de Pas",       "step_frequency_hz", SUCCESS),
            ("⚡", "Énergie Signal",     "signal_energy",     "#a78bfa"),
        ]
        self.kpi_widgets = {}
        for i, (icon, label, key, color) in enumerate(kpi_defs):
            r, c = i//2 + 1, i % 2
            f = QFrame()
            f.setStyleSheet(
                f"background:#10101a; border:1px solid {BORDER}; border-radius:8px;"
            )
            fl = QVBoxLayout(f); fl.setContentsMargins(14, 12, 14, 12); fl.setSpacing(4)
            il = QLabel(f"{icon}  {label}")
            il.setStyleSheet(f"color:{color};font-size:10px;font-weight:600;")
            vl = QLabel("—")
            vl.setStyleSheet(f"color:{TEXT_PRI};font-size:24px;font-weight:700;")
            fl.addWidget(il); fl.addWidget(vl)
            lay.addWidget(f, r, c)
            self.kpi_widgets[key] = vl
        lay.setRowStretch(3, 1)
        return card

    # ── TAB 3 — ANALYTICS ──────────────────────────────────────────────
    def _build_analytics_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab); lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(0)
        card = QFrame(); card.setObjectName("card")
        cl   = QVBoxLayout(card); cl.setContentsMargins(14, 14, 14, 14)
        cl.addWidget(InjuryFeatImportanceCanvas())
        lay.addWidget(card)
        return tab

    # ── TAB 4 — HISTORIQUE ─────────────────────────────────────────────
    def _build_history_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab); lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(16)

        self.history_chart = HistoryRiskCanvas()
        lay.addWidget(self._card(self.history_chart), 1)

        tc  = QFrame(); tc.setObjectName("card")
        tl  = QVBoxLayout(tc); tl.setContentsMargins(16, 12, 16, 16); tl.setSpacing(10)

        hr_ = QHBoxLayout()
        hh  = QLabel("HISTORIQUE DES ANALYSES BLESSURE"); hh.setObjectName("section-title")
        hr_.addWidget(hh); hr_.addStretch()
        for text, slot in [
            ("⬇  Exporter CSV", self._export_history),
            ("🗑  Effacer",      self._clear_history),
        ]:
            b = QPushButton(f"  {text}"); b.setObjectName("btn-secondary")
            b.setFixedHeight(34); b.clicked.connect(slot); hr_.addWidget(b)
        tl.addLayout(hr_)

        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(
            ["#", "Épreuve", "Risque", "Zone HR", "Score %", "Recommandation"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        for i in range(5):
            self.history_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.ResizeMode.ResizeToContents
            )
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setMinimumHeight(180)
        tl.addWidget(self.history_table)
        lay.addWidget(tc, 1)
        return tab

    # ── PREDICTION LOGIC ───────────────────────────────────────────────
    def _collect_inputs(self) -> dict | None:
        features, errors = {}, []
        for key, label, _ in NUMERIC_FIELDS:
            raw = self.input_widgets[key].text().strip()
            try:
                features[key] = float(raw)
                self.input_widgets[key].setStyleSheet("")
            except ValueError:
                errors.append(f"'{label}' doit être un nombre (reçu: {raw!r})")
                self.input_widgets[key].setStyleSheet(f"border:1px solid {DANGER};")
        for key, _, _, _ in CATEGORICAL_FIELDS:
            features[key] = self.input_widgets[key].currentText()
        if errors:
            QMessageBox.warning(
                self, "Erreur de saisie",
                "Veuillez corriger :\n\n" + "\n".join(f"• {e}" for e in errors)
            )
            return None
        return features

    def _run_prediction(self):
        features = self._collect_inputs()
        if features is None: return

        self.btn_analyse.setEnabled(False)
        self.btn_analyse.setText("  ⏳  Analyse en cours…")
        self.progress.setVisible(True)
        self.result_box.clear()
        for m in self.meta_labels: m.setVisible(False)

        self.worker = PredictionWorker(
            self.pipeline, self.encoder, self.lookup, features
        )
        self.worker.result_ready.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_result(self, res: dict):
        self._reset_loading()
        risk  = res.get("risk", "Modéré")
        zone  = res.get("zone", "")
        event = res.get("event", "?")
        risk_score = res["features"].get("injury_risk_score", 0)

        # Tab 1
        self.gauge_pred.update_risk(risk_score, risk)
        meta = [
            (f"🚨 {risk}", ""),
            (f"💓 {zone}", ""),
            (f"🏅 {event}", ""),
        ]
        for i, (txt, _) in enumerate(meta):
            self.meta_labels[i].setText(txt); self.meta_labels[i].setVisible(True)
        self.result_box.setHtml(self._format_rec(res.get("recommendation", "")))

        # Highlight border by risk level
        bc = RISK_COLOR_MAP.get(risk, ACCENT)
        self.result_box.parent().parent().setStyleSheet(
            f"QFrame#card{{background:{CARD_BG};border:1px solid {bc};border-radius:12px;}}"
        )

        # Tab 2
        self.gauge_viz.update_risk(risk_score, risk)
        self.radar_viz.update_radar(res["features"])
        self.donut_viz.update_risk(risk)
        for key, w in self.kpi_widgets.items():
            val = res["features"].get(key, 0)
            w.setText(f"{val:.2f}" if key != "injury_risk_score"
                      else f"{float(val)*100:.1f}%")

        # Tab 4
        self.history_chart.add_score(float(risk_score), event)
        row = len(self.prediction_history)
        self.prediction_history.append(res)
        self.history_table.insertRow(row)
        cells = [
            str(row + 1), event, risk, zone,
            f"{float(risk_score)*100:.1f}%",
            res.get("recommendation", "")[:90] + "…",
        ]
        for col, txt in enumerate(cells):
            item = QTableWidgetItem(txt)
            if col == 2:
                item.setForeground(QColor(RISK_COLOR_MAP.get(risk, "#fff")))
            elif col == 4:
                s = float(risk_score) * 100
                item.setForeground(QColor(
                    DANGER   if s >= 80 else
                    ELEVATED if s >= 60 else
                    WARNING  if s >= 30 else
                    SUCCESS
                ))
            self.history_table.setItem(row, col, item)
        self.history_table.scrollToBottom()

        # Switch to Visualisations tab
        self.tabs.setCurrentIndex(1)

    def _on_error(self, msg: str):
        self._reset_loading()
        self.result_box.setPlainText(f"Erreur :\n\n{msg}")
        QMessageBox.critical(self, "Erreur de prédiction", msg[:400])

    def _reset_loading(self):
        self.btn_analyse.setEnabled(True)
        self.btn_analyse.setText("  🩺  Analyser  →")
        self.progress.setVisible(False)

    def _format_rec(self, text: str) -> str:
        """Format pipe-separated recommendation text as HTML list."""
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if not parts:
            return f"<p style='color:{TEXT_SEC};'>Aucune recommandation disponible.</p>"
        html = (
            f"<style>"
            f"ul{{margin:0;padding-left:18px;}}"
            f"li{{margin-bottom:10px;line-height:1.7;color:#cbd5e1;font-size:13px;}}"
            f"li::marker{{color:{ACCENT};}}"
            f".alerte{{color:#fca5a5;font-weight:700;}}"
            f"</style><ul>"
        )
        for p in parts:
            cls = "alerte" if any(
                kw in p.upper() for kw in ("ALERTE", "CRITIQUE", "ARRÊT", "ARRET")
            ) else ""
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
        self.gauge_pred._draw(None, "")

    def _export_history(self):
        if not self.prediction_history:
            QMessageBox.information(self, "Export", "Aucune analyse à exporter.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter l'historique", "injury_history.csv", "CSV Files (*.csv)"
        )
        if path:
            rows = [{
                "#":                i + 1,
                "event":            r.get("event", ""),
                "risk_level":       r.get("risk", ""),
                "hr_zone":          r.get("zone", ""),
                "injury_risk_score": r["features"].get("injury_risk_score", ""),
                "recommendation":   r.get("recommendation", ""),
            } for i, r in enumerate(self.prediction_history)]
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "Export", f"Historique exporté :\n{path}")

    def _clear_history(self):
        self.prediction_history.clear()
        self.history_table.setRowCount(0)
        self.history_chart.clear_data()


# ===========================================================================
#  ENTRY POINT
# ===========================================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Sport Injury Analytics")
    window = InjuryDashboardApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
