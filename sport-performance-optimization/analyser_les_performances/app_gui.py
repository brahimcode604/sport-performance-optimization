"""
=============================================================================
  SPORT PERFORMANCE ANALYTICS — Desktop Prediction App
  PyQt6 · joblib · pandas
=============================================================================
  Run:  python app_gui.py
  Deps: pip install PyQt6 joblib pandas scikit-learn xgboost imbalanced-learn
=============================================================================
"""

import sys
import traceback
from pathlib import Path

import pandas as pd
import joblib

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton,
    QTextEdit, QFrame, QScrollArea, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


# ===========================================================================
#  CONSTANTS & THEME
# ===========================================================================

ARTIFACT_DIR = Path(__file__).parent  # same folder as app_gui.py

RISK_COLORS = {
    "Faible":   {"bg": "#0d2b1a", "accent": "#22c55e", "badge": "#16a34a"},
    "Modéré":   {"bg": "#2b1d08", "accent": "#f59e0b", "badge": "#d97706"},
    "Élevé":    {"bg": "#2b1208", "accent": "#f97316", "badge": "#ea580c"},
    "Critique": {"bg": "#2b0a0a", "accent": "#ef4444", "badge": "#dc2626"},
}

STYLESHEET = """
/* ── Base ── */
QMainWindow, QWidget { background-color: #0f0f12; color: #e2e8f0; font-family: 'Segoe UI', 'SF Pro Display', sans-serif; font-size: 13px; }
/* ── Scroll area ── */
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #1a1a22; width: 6px; border-radius: 3px; }
QScrollBar::handle:vertical { background: #3a3a4a; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
/* ── Card panels ── */
QFrame#card { background-color: #16161e; border: 1px solid #2a2a36; border-radius: 12px; padding: 4px; }
/* ── Section labels ── */
QLabel#section-title { color: #94a3b8; font-size: 10px; font-weight: 600; letter-spacing: 2px; }
QLabel#field-label { color: #cbd5e1; font-size: 12px; font-weight: 500; }
QLabel#app-title { color: #f1f5f9; font-size: 22px; font-weight: 700; }
QLabel#app-sub { color: #64748b; font-size: 12px; }
/* ── Inputs ── */
QLineEdit, QComboBox { background-color: #1e1e28; border: 1px solid #2e2e3e; border-radius: 8px; padding: 9px 12px; color: #e2e8f0; font-size: 13px; selection-background-color: #3b82f6; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #3b82f6; background-color: #1a1a28; }
QLineEdit:hover, QComboBox:hover { border: 1px solid #3a3a50; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid #64748b; width: 0; height: 0; margin-right: 8px; }
QComboBox QAbstractItemView { background-color: #1e1e28; border: 1px solid #2e2e3e; border-radius: 8px; color: #e2e8f0; selection-background-color: #3b82f6; padding: 4px; }
/* ── Buttons ── */
QPushButton#btn-predict { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #6366f1); color: #ffffff; font-size: 14px; font-weight: 600; border: none; border-radius: 10px; padding: 13px 32px; letter-spacing: 0.5px; }
QPushButton#btn-predict:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #4f46e5); }
QPushButton#btn-predict:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1d4ed8, stop:1 #4338ca); }
QPushButton#btn-predict:disabled { background: #2a2a36; color: #4a4a5a; }
QPushButton#btn-reset { background-color: transparent; color: #64748b; font-size: 13px; font-weight: 500; border: 1px solid #2a2a36; border-radius: 10px; padding: 13px 24px; }
QPushButton#btn-reset:hover { background-color: #1e1e28; color: #94a3b8; border: 1px solid #3a3a4a; }
/* ── Result box ── */
QTextEdit#result-box { background-color: #12121a; border: 1px solid #2a2a36; border-radius: 10px; color: #e2e8f0; font-size: 13px; line-height: 1.7; padding: 14px; }
/* ── Progress bar ── */
QProgressBar { background-color: #1e1e28; border: none; border-radius: 3px; height: 4px; }
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #6366f1); border-radius: 3px; }
/* ── Status badge ── */
QLabel#status-badge { border-radius: 6px; padding: 4px 10px; font-size: 11px; font-weight: 600; }
/* ── Separator ── */
QFrame#separator { background-color: #2a2a36; max-height: 1px; }
/* ── ToolTip ── */
QToolTip { background-color: #1e1e28; color: #cbd5e1; border: 1px solid #3a3a4a; border-radius: 6px; padding: 6px 10px; font-size: 12px; }
"""

# Field definitions: (key, label, tooltip, default_value)
NUMERIC_FIELDS = [
    ("heart_rate_bpm",      "Heart Rate (BPM)",       "Normalised heart rate value from sensor (float)", "0.63"),
    ("performance_score",   "Performance Score",      "Score between 0–100 computed from session metrics", "93.9"),
    ("injury_risk_score",   "Injury Risk Score",      "Injury risk probability between 0 and 1", "0.61"),
    ("step_frequency_hz",   "Step Frequency (Hz)",    "Steps per second captured by accelerometer", "-0.88"),
    ("stride_length_m",     "Stride Length (m)",      "Normalised stride length in metres", "1.66"),
    ("acceleration_mps2",   "Acceleration (m/s²)",    "Net body acceleration from IMU sensor", "-0.30"),
    ("signal_energy",       "Signal Energy",          "Energy of the motion signal window", "1.05"),
    ("dominant_freq_hz",    "Dominant Frequency (Hz)","Peak frequency from FFT of motion signal", "1.05"),
]

CATEGORICAL_FIELDS = [
    ("risk_level",        "Risk Level",        ["Faible", "Modéré", "Élevé", "Critique"],
     "Overall injury risk classification for this session",          "Élevé"),
    ("hr_zone",           "Heart Rate Zone",   ["Zone_Basse", "Zone_Aerobie", "Zone_Anaerobie", "Zone_Max"],
     "Cardiac training zone based on % of max heart rate",           "Zone_Anaerobie"),
    ("performance_level", "Performance Level", ["Insuffisante", "Bonne", "Excellente"],
     "Qualitative assessment of session performance",                "Excellente"),
    ("event_type",        "Event Type",        ["sprint", "high_jump", "long_jump", "unknown"],
     "Athletic event type performed during this session",            "high_jump"),
]


# ===========================================================================
#  PREDICTION WORKER  (runs in background thread — keeps UI responsive)
# ===========================================================================

class PredictionWorker(QThread):
    result_ready = pyqtSignal(dict)
    error        = pyqtSignal(str)

    def __init__(self, pipeline, encoder, lookup, features: dict):
        super().__init__()
        self.pipeline = pipeline
        self.encoder  = encoder
        self.lookup   = lookup
        self.features = features

    def run(self):
        try:
            # Faster to assign dictionary keys before building the DataFrame
            defaults = {
                "hour": 0, "dayofweek": 0,
                "gyroscope_x": 0.0, "gyroscope_y": 0.0, "gyroscope_z": 0.0,
                "accelerometer_x": 0.0, "accelerometer_y": 0.0, "accelerometer_z": 0.0
            }
            for col, val in defaults.items():
                self.features.setdefault(col, val)

            df = pd.DataFrame([self.features])
            encoded   = self.pipeline.predict(df)[0]
            label     = self.encoder.inverse_transform([encoded])[0]

            recommendation = self.lookup.loc[label, "recommendation_entrainement"] if label in self.lookup.index else f"No recommendation text found for label: {label}"

            # Parse components from composite label
            parts = label.split("__")
            
            self.result_ready.emit({
                "label":          label,
                "recommendation": recommendation,
                "risk":           parts[0] if len(parts) > 0 else "Modéré",
                "zone":           parts[1] if len(parts) > 1 else "",
                "perf":           parts[2] if len(parts) > 2 else "",
                "event":          parts[3] if len(parts) > 3 else "",
            })
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")


# ===========================================================================
#  MAIN APPLICATION WINDOW
# ===========================================================================

class SportsPredictionApp(QMainWindow):
    """Main window — sports training recommendation predictor."""

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.encoder  = None
        self.lookup   = None
        self.worker   = None

        self.input_widgets: dict = {}

        self.setWindowTitle("Sport Performance Analytics")
        self.setMinimumSize(960, 700)
        self.resize(1120, 780)

        self._load_artifacts()
        self._build_ui()
        self._apply_styles()

    def _load_artifacts(self):
        """Load ML pipeline, label encoder, and lookup table."""
        self._artifact_errors = []
        
        try:
            self.pipeline = joblib.load(ARTIFACT_DIR / "best_model.pkl")
        except Exception as e:
            self._artifact_errors.append(f"best_model.pkl : {e}")

        try:
            self.encoder = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
        except Exception as e:
            self._artifact_errors.append(f"label_encoder.pkl : {e}")

        try:
            self.lookup = pd.read_csv(ARTIFACT_DIR / "class_recommendation_lookup.csv").set_index("target_label")
        except Exception as e:
            self._artifact_errors.append(f"class_recommendation_lookup.csv : {e}")

        self._artifacts_loaded = len(self._artifact_errors) == 0

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_sidebar())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content_host = QWidget()
        content_layout = QVBoxLayout(content_host)
        content_layout.setContentsMargins(32, 28, 32, 32)
        content_layout.setSpacing(20)

        content_layout.addWidget(self._build_header())

        if not self._artifacts_loaded:
            content_layout.addWidget(self._build_warning_banner())

        content_layout.addWidget(self._build_input_card())
        content_layout.addWidget(self._build_action_row())

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(4)
        content_layout.addWidget(self.progress)

        content_layout.addWidget(self._build_result_panel())
        content_layout.addStretch()
        
        scroll.setWidget(content_host)
        root_layout.addWidget(scroll, 1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(64)
        sidebar.setStyleSheet("QFrame#sidebar { background-color: #0c0c10; border-right: 1px solid #1e1e28; }")
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        logo = QLabel("⚡")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("font-size: 22px; color: #3b82f6; padding: 10px;")
        layout.addWidget(logo)
        layout.addStretch()

        for icon, tooltip in [("📊", "Analytics"), ("🏃", "Athletes"), ("⚙️", "Settings")]:
            btn = QLabel(icon)
            btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn.setToolTip(tooltip)
            btn.setStyleSheet("font-size: 16px; padding: 10px; border-radius: 8px; color: #4a4a5a;")
            layout.addWidget(btn)

        return sidebar

    def _build_header(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        title = QLabel("Sport Performance Analytics")
        title.setObjectName("app-title")
        sub = QLabel("ML-powered training recommendation engine · v1.0")
        sub.setObjectName("app-sub")
        left.addWidget(title)
        left.addWidget(sub)
        layout.addLayout(left)
        layout.addStretch()

        status_text, style = ("  ● Model loaded", "#0d2b1a", "#22c55e", "#16a34a") if self._artifacts_loaded else ("  ● Model not found", "#2b0a0a", "#ef4444", "#dc2626")
        self.status_chip = QLabel(status_text)
        self.status_chip.setStyleSheet(f"background-color: {style[1]}; color: {style[2]}; border: 1px solid {style[3]}; border-radius: 20px; padding: 5px 14px; font-size: 11px; font-weight: 600;")
        layout.addWidget(self.status_chip)
        
        return w

    def _build_warning_banner(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #2b1208; border: 1px solid #c2410c; border-radius: 10px; padding: 4px; }")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 12, 16, 12)

        lbl = QLabel("⚠  Artifact files not found — prediction is disabled")
        lbl.setStyleSheet("color: #fb923c; font-weight: 600; font-size: 13px;")
        layout.addWidget(lbl)

        for err in self._artifact_errors:
            detail = QLabel(f"   • {err}")
            detail.setStyleSheet("color: #f97316; font-size: 12px;")
            layout.addWidget(detail)

        hint = QLabel("Place best_model.pkl, label_encoder.pkl, and class_recommendation_lookup.csv in the same folder as app_gui.py, then restart.")
        hint.setStyleSheet("color: #94a3b8; font-size: 11px; margin-top: 4px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return frame

    def _build_input_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(18)

        # ── Numeric fields ────────────────────────────────────────────
        num_header = QLabel("SENSOR & PERFORMANCE METRICS")
        num_header.setObjectName("section-title")
        layout.addWidget(num_header)

        grid_num = QGridLayout()
        grid_num.setHorizontalSpacing(20)
        grid_num.setVerticalSpacing(14)

        for i, (key, label, tooltip, default) in enumerate(NUMERIC_FIELDS):
            row, col = divmod(i, 2)
            col_offset = col * 2

            lbl = QLabel(label)
            lbl.setObjectName("field-label")
            lbl.setToolTip(tooltip)
            grid_num.addWidget(lbl, row * 2, col_offset)

            inp = QLineEdit()
            inp.setPlaceholderText(f"e.g. {default}")
            inp.setText(default)
            inp.setToolTip(tooltip)
            inp.setMinimumWidth(200)
            grid_num.addWidget(inp, row * 2 + 1, col_offset)
            self.input_widgets[key] = inp

        grid_num.setColumnMinimumWidth(1, 20)
        grid_num.setColumnMinimumWidth(3, 20)
        layout.addLayout(grid_num)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        # ── Categorical fields ────────────────────────────────────────
        cat_header = QLabel("CLASSIFICATION PARAMETERS")
        cat_header.setObjectName("section-title")
        layout.addWidget(cat_header)

        grid_cat = QGridLayout()
        grid_cat.setHorizontalSpacing(20)
        grid_cat.setVerticalSpacing(14)

        for i, (key, label, options, tooltip, default) in enumerate(CATEGORICAL_FIELDS):
            row, col = divmod(i, 2)
            col_offset = col * 2

            lbl = QLabel(label)
            lbl.setObjectName("field-label")
            lbl.setToolTip(tooltip)
            grid_cat.addWidget(lbl, row * 2, col_offset)

            cmb = QComboBox()
            cmb.addItems(options)
            cmb.setCurrentText(default)
            cmb.setToolTip(tooltip)
            cmb.setMinimumWidth(200)
            grid_cat.addWidget(cmb, row * 2 + 1, col_offset)
            self.input_widgets[key] = cmb

        grid_cat.setColumnMinimumWidth(1, 20)
        grid_cat.setColumnMinimumWidth(3, 20)
        layout.addLayout(grid_cat)

        return card

    def _build_action_row(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addStretch()

        self.btn_reset = QPushButton("↺  Reset fields")
        self.btn_reset.setObjectName("btn-reset")
        self.btn_reset.setFixedHeight(46)
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self._reset_fields)
        layout.addWidget(self.btn_reset)

        self.btn_predict = QPushButton("  Predict Recommendation  →")
        self.btn_predict.setObjectName("btn-predict")
        self.btn_predict.setFixedHeight(46)
        self.btn_predict.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_predict.setEnabled(self._artifacts_loaded)
        self.btn_predict.clicked.connect(self._run_prediction)
        layout.addWidget(self.btn_predict)

        return w

    def _build_result_panel(self) -> QFrame:
        self.result_card = QFrame()
        self.result_card.setObjectName("card")

        layout = QVBoxLayout(self.result_card)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(14)

        header_row = QHBoxLayout()
        res_title = QLabel("PREDICTION RESULT")
        res_title.setObjectName("section-title")
        header_row.addWidget(res_title)
        header_row.addStretch()

        self.risk_badge = QLabel("")
        self.risk_badge.setObjectName("status-badge")
        self.risk_badge.setVisible(False)
        header_row.addWidget(self.risk_badge)

        self.class_label_lbl = QLabel("")
        self.class_label_lbl.setStyleSheet("color: #475569; font-size: 11px;")
        header_row.addWidget(self.class_label_lbl)
        layout.addLayout(header_row)

        self.meta_row = QHBoxLayout()
        self.meta_widgets = []
        for _ in range(3):
            ml = QLabel()
            ml.setStyleSheet("color: #64748b; font-size: 11px; margin-right: 16px;")
            ml.setVisible(False)
            self.meta_row.addWidget(ml)
            self.meta_widgets.append(ml)
            
        self.meta_row.addStretch()
        layout.addLayout(self.meta_row)

        self.result_box = QTextEdit()
        self.result_box.setObjectName("result-box")
        self.result_box.setReadOnly(True)
        self.result_box.setMinimumHeight(160)
        self.result_box.setPlaceholderText("Fill in the form above and click  Predict Recommendation  to see the result.")
        layout.addWidget(self.result_box)

        return self.result_card

    def _collect_inputs(self) -> dict | None:
        """Collect, validate, and return feature dict. Returns None on error."""
        features, errors = {}, []

        for key, label, tooltip, _ in NUMERIC_FIELDS:
            widget = self.input_widgets[key]
            raw = widget.text().strip()
            if not raw:
                errors.append(f"'{label}' is required.")
                widget.setStyleSheet("border: 1px solid #ef4444;")
                continue
            try:
                features[key] = float(raw)
                widget.setStyleSheet("")
            except ValueError:
                errors.append(f"'{label}' must be a number (got: {raw!r}).")
                widget.setStyleSheet("border: 1px solid #ef4444;")

        for key, label, _, tooltip, _ in CATEGORICAL_FIELDS:
            features[key] = self.input_widgets[key].currentText()

        if errors:
            QMessageBox.warning(self, "Validation Error", "Please fix the following:\n\n" + "\n".join(f"  • {e}" for e in errors))
            return None

        return features

    def _run_prediction(self):
        features = self._collect_inputs()
        if features is None: return

        self.btn_predict.setEnabled(False)
        self.btn_predict.setText("  Predicting…")
        self.progress.setVisible(True)
        self.result_box.setPlainText("")
        self.risk_badge.setVisible(False)
        self.class_label_lbl.setText("")
        
        for w in self.meta_widgets: w.setVisible(False)

        self.worker = PredictionWorker(self.pipeline, self.encoder, self.lookup, features)
        self.worker.result_ready.connect(self._on_prediction_done)
        self.worker.error.connect(self._on_prediction_error)
        self.worker.start()

    def _on_prediction_done(self, result: dict):
        self._reset_loading_state()

        risk    = result.get("risk", "Modéré")
        colors  = RISK_COLORS.get(risk, RISK_COLORS["Modéré"])

        self.risk_badge.setText(f"  {risk}  ")
        self.risk_badge.setStyleSheet(f"background-color: {colors['bg']}; color: {colors['accent']}; border: 1px solid {colors['badge']}; border-radius: 6px; padding: 4px 10px; font-size: 11px; font-weight: 600;")
        self.risk_badge.setVisible(True)

        self.class_label_lbl.setText(f"Class: {result.get('label', '')}")

        meta = [
            (f"Zone : {result.get('zone','')}",    "💓"),
            (f"Level : {result.get('perf','')}",   "📈"),
            (f"Event : {result.get('event','')}",  "🏅"),
        ]
        for i, (text, icon) in enumerate(meta):
            self.meta_widgets[i].setText(f"{icon}  {text}")
            self.meta_widgets[i].setVisible(True)

        self.result_box.setHtml(self._format_recommendation(result.get("recommendation", "")))
        self.result_card.setStyleSheet(f"QFrame#card {{ background-color: #16161e; border: 1px solid {colors['badge']}; border-radius: 12px; }}")

    def _on_prediction_error(self, message: str):
        self._reset_loading_state()
        self.result_box.setPlainText(f"Prediction failed:\n\n{message}")
        self.result_card.setStyleSheet("QFrame#card { background-color: #16161e; border: 1px solid #dc2626; border-radius: 12px; }")

    def _reset_loading_state(self):
        self.btn_predict.setEnabled(True)
        self.btn_predict.setText("  Predict Recommendation  →")
        self.progress.setVisible(False)

    def _format_recommendation(self, text: str) -> str:
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if not parts:
            return "<p style='color:#94a3b8;'>No recommendation available.</p>"

        html = "<style>body { font-family: 'Segoe UI', sans-serif; color: #e2e8f0; } ul { margin: 0; padding-left: 18px; } li { margin-bottom: 10px; line-height: 1.6; color: #cbd5e1; font-size: 13px; } li::marker { color: #3b82f6; } .alert { color: #fca5a5; font-weight: 600; }</style><ul>"
        for part in parts:
            cls = 'alert' if "ALERTE" in part.upper() or "arrêt" in part.lower() else ""
            html += f"<li class='{cls}'>{part}</li>"
        html += "</ul>"
        
        return html

    def _reset_fields(self):
        for key, _, _, default in NUMERIC_FIELDS:
            self.input_widgets[key].setText(default)
            self.input_widgets[key].setStyleSheet("")

        for key, _, _, _, default in CATEGORICAL_FIELDS:
            self.input_widgets[key].setCurrentText(default)

        self.result_box.clear()
        self.risk_badge.setVisible(False)
        self.class_label_lbl.setText("")
        for w in self.meta_widgets: w.setVisible(False)
        self.result_card.setStyleSheet("QFrame#card { background-color: #16161e; border: 1px solid #2a2a36; border-radius: 12px; }")

    def _apply_styles(self):
        self.setStyleSheet(STYLESHEET)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Sport Performance Analytics")
    
    window = SportsPredictionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()