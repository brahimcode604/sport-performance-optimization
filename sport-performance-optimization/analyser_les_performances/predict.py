"""
=============================================================================
  SPORTS PERFORMANCE — RECOMMENDATION PREDICTOR
  Production-ready inference module
=============================================================================

Usage (CLI):
    python prediction.py --input data_new.csv --output predictions.csv

Usage (API / import):
    from prediction import SportsRecommendationPredictor
    predictor = SportsRecommendationPredictor()
    result = predictor.predict_single({...})
    df_out = predictor.predict_batch("new_data.csv")
"""

import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
#  CONSTANTS — must mirror train.py exactly
# ===========================================================================

NUMERIC_FEATURES = [
    "heart_rate_bpm",
    "step_frequency_hz",
    "stride_length_m",
    "acceleration_mps2",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "signal_energy",
    "dominant_freq_hz",
    "performance_score",
    "injury_risk_score",
    "hour",
    "dayofweek",
]

CATEGORICAL_FEATURES = ["risk_level", "hr_zone", "performance_level", "event_type"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EVENT_COLUMNS = {
    "event_type_high_jump": "high_jump",
    "event_type_long_jump": "long_jump",
    "event_type_sprint":    "sprint",
}


# ===========================================================================
#  PREDICTOR CLASS
# ===========================================================================

class SportsRecommendationPredictor:
    """
    Loads the three artifacts produced by train.py and exposes two
    prediction methods:
        - predict_single(sample: dict)  → single recommendation string
        - predict_batch(input_path)     → DataFrame with predictions
    """

    def __init__(
        self,
        model_path:   str = "best_model.pkl",
        encoder_path: str = "label_encoder.pkl",
        lookup_path:  str = "class_recommendation_lookup.csv",
    ):
        self.model_path   = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.lookup_path  = Path(lookup_path)

        self._pipeline = None
        self._encoder  = None
        self._lookup   = None

        self._load_artifacts()

    # ------------------------------------------------------------------
    # LOAD ARTIFACTS
    # ------------------------------------------------------------------

    def _load_artifacts(self):
        logger.info("Loading model artifacts …")

        for path in (self.model_path, self.encoder_path, self.lookup_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {path}\n"
                    "Run train.py first to generate all artifacts."
                )

        self._pipeline = joblib.load(self.model_path)
        self._encoder  = joblib.load(self.encoder_path)
        self._lookup   = (
            pd.read_csv(self.lookup_path)
            .set_index("target_label")
        )

        logger.info("  ✓ Model          : %s", self.model_path)
        logger.info("  ✓ LabelEncoder   : %s", self.encoder_path)
        logger.info(
            "  ✓ Lookup table   : %s  (%d classes)",
            self.lookup_path,
            len(self._lookup),
        )

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING (mirrors train.py logic)
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Temporal features from timestamp ───────────────────────────
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour"]      = df["timestamp"].dt.hour.fillna(0).astype(int)
            df["dayofweek"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
        else:
            df.setdefault("hour",      pd.Series(0, index=df.index))
            df.setdefault("dayofweek", pd.Series(0, index=df.index))

        # ── event_type: derive from binary flags if column absent ───────
        if "event_type" not in df.columns:
            series = pd.Series("unknown", index=df.index)
            for col, label in EVENT_COLUMNS.items():
                if col in df.columns:
                    series = series.where(df[col] != 1, label)
            df["event_type"] = series

        # ── Numeric imputation (median per column) ──────────────────────
        num_base = [c for c in NUMERIC_FEATURES if c not in ("hour", "dayofweek")]
        for col in num_base:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = 0.0

        # ── Categorical imputation ──────────────────────────────────────
        cat_defaults = {
            "risk_level":        "Modéré",
            "hr_zone":           "Zone_Aerobie",
            "performance_level": "Bonne",
            "event_type":        "unknown",
        }
        for col, default in cat_defaults.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)

        return df

    # ------------------------------------------------------------------
    # DECODE PREDICTION → LABEL + RECOMMENDATION TEXT
    # ------------------------------------------------------------------

    def _decode(self, encoded_predictions: np.ndarray) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            predicted_label, recommendation_entrainement, confidence_*
        """
        decoded_labels = self._encoder.inverse_transform(encoded_predictions)

        recommendations = []
        for label in decoded_labels:
            if label in self._lookup.index:
                rec = self._lookup.loc[label, "recommendation_entrainement"]
            else:
                rec = f"(no recommendation text for label: {label})"
            recommendations.append(rec)

        # ── Parse label components for readability ──────────────────────
        components = pd.Series(decoded_labels).str.split("__", expand=True)
        components.columns = (
            ["pred_risk_level", "pred_hr_zone", "pred_performance_level",
             "pred_event_type", "pred_hr_critical_flag"]
            [: len(components.columns)]
        )

        result = components.copy()
        result["predicted_label"]              = decoded_labels
        result["recommendation_entrainement"]  = recommendations
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # PREDICT — SINGLE SAMPLE
    # ------------------------------------------------------------------

    def predict_single(self, sample: dict) -> dict:
        """
        Predicts the training recommendation for a single athlete session.

        Parameters
        ----------
        sample : dict
            Feature values for one session.  Missing features are imputed.

        Returns
        -------
        dict with keys:
            predicted_label, recommendation_entrainement,
            pred_risk_level, pred_hr_zone, pred_performance_level,
            pred_event_type, pred_hr_critical_flag
        """
        df    = self._engineer_features(pd.DataFrame([sample]))
        X     = df[[c for c in ALL_FEATURES if c in df.columns]]
        preds = self._pipeline.predict(X)
        decoded = self._decode(preds)
        return decoded.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # PREDICT — BATCH (CSV)
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        input_path:  str,
        output_path: str | None = None,
        add_proba:   bool       = False,
    ) -> pd.DataFrame:
        """
        Predicts recommendations for every row in a CSV file.

        Parameters
        ----------
        input_path  : path to input CSV (must contain the feature columns)
        output_path : if provided, saves results to this CSV path
        add_proba   : if True and the model supports it, adds per-class
                      probability columns (top-3)

        Returns
        -------
        Original DataFrame with prediction columns appended.
        """
        logger.info("Loading input data from %s …", input_path)
        df_raw = pd.read_csv(input_path)
        logger.info("  → %d rows × %d columns", *df_raw.shape)

        df_eng = self._engineer_features(df_raw.copy())
        X      = df_eng[[c for c in ALL_FEATURES if c in df_eng.columns]]

        logger.info("Running inference …")
        preds   = self._pipeline.predict(X)
        decoded = self._decode(preds)

        # ── Optional: predicted probabilities (top-3) ───────────────────
        if add_proba and hasattr(self._pipeline, "predict_proba"):
            try:
                probas     = self._pipeline.predict_proba(X)
                top3_idx   = np.argsort(probas, axis=1)[:, -3:][:, ::-1]
                top3_proba = np.sort(probas, axis=1)[:, -3:][:, ::-1]

                for rank in range(3):
                    labels = self._encoder.inverse_transform(top3_idx[:, rank])
                    decoded[f"top{rank+1}_label"] = labels
                    decoded[f"top{rank+1}_proba"]  = np.round(
                        top3_proba[:, rank], 4
                    )
                logger.info("  ✓ Top-3 class probabilities added")
            except Exception as exc:
                logger.warning("Could not add probabilities: %s", exc)

        df_out = pd.concat(
            [df_raw.reset_index(drop=True), decoded], axis=1
        )

        if output_path:
            df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info("Predictions saved → %s", output_path)

        logger.info(
            "Inference complete — %d predictions generated.", len(df_out)
        )
        return df_out

    # ------------------------------------------------------------------
    # EVALUATION ON LABELLED DATA
    # ------------------------------------------------------------------

    def evaluate(self, labelled_path: str) -> pd.DataFrame:
        """
        If the input CSV already contains 'recommendation_entrainement',
        computes accuracy and a per-class report by comparing predictions
        against ground-truth labels.

        Parameters
        ----------
        labelled_path : path to CSV with ground-truth column

        Returns
        -------
        DataFrame with per-class precision / recall / F1
        """
        from sklearn.metrics import classification_report, accuracy_score

        logger.info("Evaluating on labelled data: %s …", labelled_path)
        df = pd.read_csv(labelled_path)

        if "recommendation_entrainement" not in df.columns:
            raise ValueError(
                "Column 'recommendation_entrainement' not found in the file."
            )

        df_out = self.predict_batch(labelled_path)

        # We compare at the structured-label level (not raw text) because
        # minor whitespace differences in free text would inflate errors.
        # Re-build ground-truth label from the 5 component columns.
        def make_label(row):
            hr_flag = "critical" if row.get("hr_zone") == "Zone_Max" else "normal"
            event   = row.get("event_type", "unknown")
            if event == "unknown":
                for col, label in EVENT_COLUMNS.items():
                    if row.get(col, 0) == 1:
                        event = label
                        break
            return "__".join([
                str(row.get("risk_level",        "?")),
                str(row.get("hr_zone",           "?")),
                str(row.get("performance_level", "?")),
                str(event),
                str(hr_flag),
            ])

        df_out["true_label"] = df.apply(make_label, axis=1)
        y_true = self._encoder.transform(
            df_out["true_label"].where(
                df_out["true_label"].isin(self._encoder.classes_),
                self._encoder.classes_[0],          # fallback for unseen labels
            )
        )
        y_pred = self._encoder.transform(df_out["predicted_label"])

        acc = accuracy_score(y_true, y_pred)
        logger.info("Accuracy on labelled data: %.4f", acc)

        report = classification_report(
            y_true,
            y_pred,
            target_names=self._encoder.classes_,
            zero_division=0,
            output_dict=True,
        )
        report_df = pd.DataFrame(report).T
        print(f"\nAccuracy: {acc:.4f}\n")
        print(report_df.round(4).to_string())
        return report_df


# ===========================================================================
#  COMMAND-LINE INTERFACE
# ===========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Sports Recommendation Predictor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file (feature columns required).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save the output CSV with predictions.\n"
             "Default: <input_filename>_predictions.csv",
    )
    parser.add_argument(
        "--model",
        default="best_model.pkl",
        help="Path to the trained model .pkl file. (default: best_model.pkl)",
    )
    parser.add_argument(
        "--encoder",
        default="label_encoder.pkl",
        help="Path to the LabelEncoder .pkl file. (default: label_encoder.pkl)",
    )
    parser.add_argument(
        "--lookup",
        default="class_recommendation_lookup.csv",
        help="Path to the class→recommendation lookup CSV.",
    )
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Add top-3 predicted class probabilities to output.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="If the input CSV has ground-truth labels, compute evaluation metrics.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    output_path = args.output or (
        Path(args.input).stem + "_predictions.csv"
    )

    predictor = SportsRecommendationPredictor(
        model_path   = args.model,
        encoder_path = args.encoder,
        lookup_path  = args.lookup,
    )

    if args.evaluate:
        predictor.evaluate(args.input)
    else:
        predictor.predict_batch(
            input_path  = args.input,
            output_path = output_path,
            add_proba   = args.proba,
        )


if __name__ == "__main__":
    main()