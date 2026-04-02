"""
=============================================================================
  SPORTS INJURY — RECOMMENDATION PREDICTOR  (predict.py)
  Production-ready inference module
=============================================================================

Usage (CLI):
    python predict.py --input new_data.csv
    python predict.py --input new_data.csv --output results.csv --proba
    python predict.py --input labelled.csv --evaluate

Usage (API / import):
    from predict import InjuryRecommendationPredictor
    predictor = InjuryRecommendationPredictor()
    result    = predictor.predict_single({...})
    df_out    = predictor.predict_batch("athletes.csv")
=============================================================================
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
#  CONSTANTS  (must mirror train.py exactly)
# ===========================================================================

ARTIFACT_DIR = Path(__file__).parent

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
    "injury_risk_score",
    "performance_score",
    "risk_level_encoded",
    "hr_zone_encoded",
    "event_type_high_jump",
    "event_type_long_jump",
    "event_type_sprint",
    "motion_class_acceleration_phase",
    "motion_class_flight_phase",
    "motion_class_landing",
    "motion_class_start_phase",
]

CATEGORICAL_FEATURES = ["risk_level", "hr_zone", "event_type"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EVENT_MAP = {
    "event_type_high_jump": "high_jump",
    "event_type_long_jump": "long_jump",
    "event_type_sprint":    "sprint",
}

RISK_DEFAULTS = {
    "risk_level":        "Modéré",
    "hr_zone":           "Zone_Aerobie",
    "performance_level": "Bonne",
    "event_type":        "unknown",
}

RISK_ENCODING = {"Faible": 0, "Modéré": 1, "Élevé": 2, "Critique": 3}
HR_ENCODING   = {
    "Zone_Basse": 0, "Zone_Aerobie": 1,
    "Zone_Anaerobie": 2, "Zone_Max": 3,
}


# ===========================================================================
#  PREDICTOR CLASS
# ===========================================================================

class InjuryRecommendationPredictor:
    """
    Loads the three artifacts produced by train.py and exposes:
        predict_single(sample: dict)  → recommendation string + meta info
        predict_batch(path)           → DataFrame with prediction columns
        evaluate(labelled_path)       → accuracy + classification report
    """

    def __init__(
        self,
        model_path:   str | None = None,
        encoder_path: str | None = None,
        lookup_path:  str | None = None,
    ):
        self.model_path   = Path(model_path)   if model_path   else ARTIFACT_DIR / "injury_model.pkl"
        self.encoder_path = Path(encoder_path) if encoder_path else ARTIFACT_DIR / "injury_encoder.pkl"
        self.lookup_path  = Path(lookup_path)  if lookup_path  else ARTIFACT_DIR / "injury_lookup.csv"

        self._pipeline = None
        self._encoder  = None
        self._lookup   = None
        self._load_artifacts()

    # ------------------------------------------------------------------
    # LOAD ARTIFACTS
    # ------------------------------------------------------------------
    def _load_artifacts(self):
        logger.info("Loading injury model artifacts …")
        for path in (self.model_path, self.encoder_path, self.lookup_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Artifact not found: {path}\n"
                    "Run train.py first to generate all artifacts."
                )
        self._pipeline = joblib.load(self.model_path)
        self._encoder  = joblib.load(self.encoder_path)
        self._lookup   = (
            pd.read_csv(self.lookup_path)
            .set_index("target_label")
        )
        logger.info("  ✓ Model        : %s", self.model_path)
        logger.info("  ✓ Encoder      : %s", self.encoder_path)
        logger.info(
            "  ✓ Lookup table : %s  (%d classes)",
            self.lookup_path, len(self._lookup),
        )

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Derive event_type from binary flags if not present
        if "event_type" not in df.columns:
            series = pd.Series("unknown", index=df.index)
            for col, label in EVENT_MAP.items():
                if col in df.columns:
                    series = series.where(df[col] != 1, label)
            df["event_type"] = series

        # Encode ordinal categoricals if raw columns exist but encoded absent
        if "risk_level" in df.columns and "risk_level_encoded" not in df.columns:
            df["risk_level_encoded"] = df["risk_level"].map(RISK_ENCODING).fillna(1)
        if "hr_zone" in df.columns and "hr_zone_encoded" not in df.columns:
            df["hr_zone_encoded"] = df["hr_zone"].map(HR_ENCODING).fillna(1)

        # Impute numeric features
        num_base = [
            c for c in NUMERIC_FEATURES
            if c not in (
                "event_type_high_jump", "event_type_long_jump", "event_type_sprint",
                "motion_class_acceleration_phase", "motion_class_flight_phase",
                "motion_class_landing", "motion_class_start_phase",
            )
        ]
        for col in num_base:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
                    df[col].median() if not df[col].isna().all() else 0.0
                )
            else:
                df[col] = 0.0

        # Binary flag columns — fill missing with 0
        binary_cols = [
            "event_type_high_jump", "event_type_long_jump", "event_type_sprint",
            "motion_class_acceleration_phase", "motion_class_flight_phase",
            "motion_class_landing", "motion_class_start_phase",
        ]
        for col in binary_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0).astype(int)

        # Categorical defaults
        for col, default in RISK_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)

        return df

    # ------------------------------------------------------------------
    # DECODE PREDICTIONS
    # ------------------------------------------------------------------
    def _decode(self, encoded_preds: np.ndarray) -> pd.DataFrame:
        decoded = self._encoder.inverse_transform(encoded_preds)

        recommendations = []
        for label in decoded:
            if label in self._lookup.index:
                rec = self._lookup.loc[label, "recommendation_blessure"]
            else:
                rec = f"(aucune recommendation pour: {label})"
            recommendations.append(rec)

        # Parse label components
        components = pd.Series(decoded).str.split("__", expand=True)
        col_names  = ["pred_risk_level", "pred_hr_zone", "pred_event_type"]
        components.columns = col_names[: len(components.columns)]

        result = components.copy()
        result["predicted_label"]      = decoded
        result["recommendation_blessure"] = recommendations
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # PREDICT — SINGLE
    # ------------------------------------------------------------------
    def predict_single(self, sample: dict) -> dict:
        """
        Predict injury recommendation for one athlete session.

        Parameters
        ----------
        sample : dict of feature values (missing fields are imputed)

        Returns
        -------
        dict with keys:
            predicted_label, recommendation_blessure,
            pred_risk_level, pred_hr_zone, pred_event_type
        """
        df      = self._engineer_features(pd.DataFrame([sample]))
        X       = df[[c for c in ALL_FEATURES if c in df.columns]]
        preds   = self._pipeline.predict(X)
        decoded = self._decode(preds)
        return decoded.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # PREDICT — BATCH
    # ------------------------------------------------------------------
    def predict_batch(
        self,
        input_path:  str,
        output_path: str | None = None,
        add_proba:   bool       = False,
    ) -> pd.DataFrame:
        """
        Predict injury recommendations for every row in a CSV file.

        Parameters
        ----------
        input_path  : path to input CSV
        output_path : if given, saves results to CSV
        add_proba   : if True, adds top-3 predicted class probabilities

        Returns
        -------
        Original DataFrame with prediction columns appended.
        """
        logger.info("Loading input data: %s …", input_path)
        df_raw = pd.read_csv(input_path)
        logger.info("  → %d rows × %d columns", *df_raw.shape)

        df_eng = self._engineer_features(df_raw.copy())
        X      = df_eng[[c for c in ALL_FEATURES if c in df_eng.columns]]

        logger.info("Running inference …")
        preds   = self._pipeline.predict(X)
        decoded = self._decode(preds)

        # Optional top-3 probabilities
        if add_proba and hasattr(self._pipeline, "predict_proba"):
            try:
                probas      = self._pipeline.predict_proba(X)
                top3_idx    = np.argsort(probas, axis=1)[:, -3:][:, ::-1]
                top3_proba  = np.sort(probas, axis=1)[:, -3:][:, ::-1]
                for rank in range(3):
                    labels = self._encoder.inverse_transform(top3_idx[:, rank])
                    decoded[f"top{rank+1}_label"] = labels
                    decoded[f"top{rank+1}_proba"] = np.round(top3_proba[:, rank], 4)
                logger.info("  ✓ Top-3 probabilities added")
            except Exception as exc:
                logger.warning("Could not add probabilities: %s", exc)

        df_out = pd.concat([df_raw.reset_index(drop=True), decoded], axis=1)

        if output_path:
            df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info("Predictions saved → %s", output_path)

        logger.info("Inference complete — %d predictions generated.", len(df_out))
        return df_out

    # ------------------------------------------------------------------
    # EVALUATE ON LABELLED DATA
    # ------------------------------------------------------------------
    def evaluate(self, labelled_path: str) -> pd.DataFrame:
        """Compare predictions against ground-truth labels and print a report."""
        from sklearn.metrics import classification_report, accuracy_score

        logger.info("Evaluating on: %s …", labelled_path)
        df = pd.read_csv(labelled_path)

        if "recommendation_blessure" not in df.columns:
            raise ValueError("Column 'recommendation_blessure' not found.")

        df_out   = self.predict_batch(labelled_path)
        df_eng   = self._engineer_features(df.copy())

        def make_label(row):
            return "__".join([
                str(row.get("risk_level", "?")),
                str(row.get("hr_zone",    "?")),
                str(row.get("event_type", "unknown")),
            ])

        df_out["true_label"] = df_eng.apply(make_label, axis=1)
        y_true = self._encoder.transform(
            df_out["true_label"].where(
                df_out["true_label"].isin(self._encoder.classes_),
                self._encoder.classes_[0],
            )
        )
        y_pred = self._encoder.transform(df_out["predicted_label"])

        acc = accuracy_score(y_true, y_pred)
        logger.info("Accuracy: %.4f", acc)

        report    = classification_report(
            y_true, y_pred,
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
        description="Injury Recommendation Predictor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input",   "-i", required=True,
                        help="Path to input CSV file.")
    parser.add_argument("--output",  "-o", default=None,
                        help="Path to output CSV (default: <input>_injury_predictions.csv).")
    parser.add_argument("--model",   default=None,
                        help="Path to injury_model.pkl (default: auto).")
    parser.add_argument("--encoder", default=None,
                        help="Path to injury_encoder.pkl (default: auto).")
    parser.add_argument("--lookup",  default=None,
                        help="Path to injury_lookup.csv (default: auto).")
    parser.add_argument("--proba",   action="store_true",
                        help="Add top-3 class probabilities to output.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate against ground-truth labels.")
    return parser.parse_args()


def main():
    args = _parse_args()
    output_path = args.output or (
        Path(args.input).stem + "_injury_predictions.csv"
    )
    predictor = InjuryRecommendationPredictor(
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
