"""
=============================================================================
  SPORTS PERFORMANCE — TRAINING RECOMMENDATION CLASSIFIER
  Production-ready ML pipeline | Senior Data Science level
=============================================================================

ROOT CAUSE OF ORIGINAL ERROR
------------------------------
`recommendation_entrainement` is a free-text column: each row contains a
long, dynamically composed string with hundreds of near-unique values.
Encoding that text directly as a classification target produces 700+ classes,
most with a single sample → stratified split becomes impossible and SMOTE
is meaningless.

SOLUTION
--------
We build a **structured composite target** from the 5 columns that
deterministically drive every recommendation:
    risk_level + hr_zone + performance_level + event_type + hr_critical_flag

This yields ~30–60 balanced, semantically meaningful classes that a
classifier can actually learn.  The model then predicts *which combination of
conditions* applies to a given session; a downstream lookup table
(auto-generated here) maps each class back to the full recommendation text.
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ===========================================================================
#  CONSTANTS
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

# Columns used to assemble the structured target label
LABEL_COMPONENTS = [
    "risk_level",
    "hr_zone",
    "performance_level",
    "event_type",
    "hr_critical_flag",
]


# ===========================================================================
#  HELPER — EVENT TYPE EXTRACTION
# ===========================================================================

def extract_event_type(df: pd.DataFrame) -> pd.Series:
    """
    Derive a clean 'event_type' column from the binary event flag columns
    present in the dataset (event_type_high_jump, event_type_long_jump,
    event_type_sprint).  Falls back to 'unknown' when none is set.
    """
    event_map = {
        "event_type_high_jump": "high_jump",
        "event_type_long_jump": "long_jump",
        "event_type_sprint":    "sprint",
    }
    series = pd.Series("unknown", index=df.index)
    for col, label in event_map.items():
        if col in df.columns:
            series = series.where(df[col] != 1, label)
    return series


# ===========================================================================
#  MAIN CLASS
# ===========================================================================

class SportsRecommendationTrainer:
    """
    End-to-end pipeline: load → clean → engineer → split → train → evaluate
    → save best model + label encoder + class→recommendation lookup table.
    """

    def __init__(
        self,
        data_path: str,
        model_save_path: str = "best_model.pkl",
        test_size: float = 0.10,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.data_path        = Path(data_path)
        self.model_save_path  = Path(model_save_path)
        self.test_size        = test_size
        self.cv_folds         = cv_folds
        self.random_state     = random_state
        self.label_encoder    = LabelEncoder()

    # ------------------------------------------------------------------
    # 1. LOAD & CLEAN
    # ------------------------------------------------------------------

    def load_and_preprocess(self):
        logger.info("=" * 65)
        logger.info("STEP 1 — Loading dataset from %s", self.data_path)
        logger.info("=" * 65)

        df = pd.read_csv(self.data_path)
        logger.info("Raw shape: %s", df.shape)

        # ── Drop rows without a recommendation (target) ────────────────
        initial_rows = len(df)
        df = df.dropna(subset=["recommendation_entrainement"])
        logger.info(
            "Dropped %d rows with missing target → %d rows remain",
            initial_rows - len(df),
            len(df),
        )

        # ── Timestamp → temporal features ──────────────────────────────
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour"]      = df["timestamp"].dt.hour.fillna(0).astype(int)
            df["dayofweek"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
        else:
            df["hour"]      = 0
            df["dayofweek"] = 0

        # ── Impute numeric features ─────────────────────────────────────
        numeric_base = [
            c for c in NUMERIC_FEATURES if c not in ("hour", "dayofweek")
        ]
        for col in numeric_base:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # ── Impute categorical features ─────────────────────────────────
        for col in ["risk_level", "hr_zone", "performance_level"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        # ── Derive event_type feature ───────────────────────────────────
        df["event_type"] = extract_event_type(df)

        # ── HR critical flag (binary) ───────────────────────────────────
        df["hr_critical_flag"] = df["hr_zone"].apply(
            lambda z: "critical" if z == "Zone_Max" else "normal"
        )

        logger.info(
            "Categorical value counts:\n%s",
            df[LABEL_COMPONENTS].apply(lambda c: c.nunique()).to_string(),
        )
        return df

    # ------------------------------------------------------------------
    # 2. BUILD STRUCTURED TARGET
    # ------------------------------------------------------------------

    def build_target(self, df: pd.DataFrame):
        """
        Constructs a composite label such as:
            'Critique__Zone_Max__Insuffisante__sprint__critical'

        Also persists a lookup table  class_label → recommendation_text
        so predictions can be decoded back to human-readable text.
        """
        logger.info("=" * 65)
        logger.info("STEP 2 — Engineering structured target label")
        logger.info("=" * 65)

        # Composite label
        df["target_label"] = (
            df["risk_level"].astype(str)
            + "__"
            + df["hr_zone"].astype(str)
            + "__"
            + df["performance_level"].astype(str)
            + "__"
            + df["event_type"].astype(str)
            + "__"
            + df["hr_critical_flag"].astype(str)
        )

        n_classes = df["target_label"].nunique()
        class_counts = df["target_label"].value_counts()
        logger.info("Unique structured target classes: %d", n_classes)
        logger.info("Class size — min: %d | median: %.0f | max: %d",
                    class_counts.min(), class_counts.median(), class_counts.max())

        # ── Drop singletons (cannot be stratified) ─────────────────────
        valid_classes = class_counts[class_counts >= 2].index
        dropped = (~df["target_label"].isin(valid_classes)).sum()
        if dropped > 0:
            logger.warning(
                "Removed %d singleton rows (classes with < 2 samples)", dropped
            )
            df = df[df["target_label"].isin(valid_classes)].copy()

        # ── Lookup table: label → first seen recommendation text ────────
        lookup = (
            df.groupby("target_label")["recommendation_entrainement"]
            .first()
            .reset_index()
        )
        lookup.columns = ["target_label", "recommendation_entrainement"]
        lookup.to_csv("class_recommendation_lookup.csv", index=False)
        logger.info(
            "Saved class→recommendation lookup → class_recommendation_lookup.csv"
        )

        # ── Encode target ───────────────────────────────────────────────
        y = self.label_encoder.fit_transform(df["target_label"])
        joblib.dump(self.label_encoder, "label_encoder.pkl")
        logger.info("Saved LabelEncoder → label_encoder.pkl")

        return df, y

    # ------------------------------------------------------------------
    # 3. TRAIN / TEST SPLIT
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame, y: np.ndarray):
        logger.info("=" * 65)
        logger.info(
            "STEP 3 — Train/test split (%.0f%% / %.0f%%, stratified)",
            (1 - self.test_size) * 100,
            self.test_size * 100,
        )
        logger.info("=" * 65)

        # Keep only features present in the DataFrame
        available_features = [f for f in ALL_FEATURES if f in df.columns]
        X = df[available_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info(
            "Train: %d rows | Test: %d rows | Features: %d",
            len(X_train), len(X_test), X.shape[1],
        )
        return X_train, X_test, y_train, y_test, available_features

    # ------------------------------------------------------------------
    # 4. PREPROCESSOR
    # ------------------------------------------------------------------

    def build_preprocessor(self, available_features: list):
        num_cols = [f for f in NUMERIC_FEATURES   if f in available_features]
        cat_cols = [f for f in CATEGORICAL_FEATURES if f in available_features]

        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(),                         num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"),   cat_cols),
            ],
            remainder="drop",
        )

    # ------------------------------------------------------------------
    # 5. TRAIN & EVALUATE
    # ------------------------------------------------------------------

    def train_and_evaluate(
        self,
        X_train, X_test, y_train, y_test,
        available_features: list,
    ):
        logger.info("=" * 65)
        logger.info("STEP 4 — Model training with SMOTE + RandomizedSearchCV")
        logger.info("=" * 65)

        preprocessor = self.build_preprocessor(available_features)

        # ── SMOTE k_neighbors: safe even for small minority classes ─────
        min_class_count = np.bincount(y_train).min()
        k_neighbors = min(5, min_class_count - 1)
        logger.info("SMOTE k_neighbors set to %d (min class size = %d)",
                    k_neighbors, min_class_count)

        models = {
            "Logistic Regression": {
                "estimator": LogisticRegression(
                    max_iter=1000, random_state=self.random_state, solver="saga"
                ),
                "params": {
                    "classifier__C": [0.01, 0.1, 1.0, 10.0],
                    "classifier__penalty": ["l1", "l2"],
                },
            },
            "Random Forest": {
                "estimator": RandomForestClassifier(
                    random_state=self.random_state, class_weight="balanced"
                ),
                "params": {
                    "classifier__n_estimators": [100, 200, 300],
                    "classifier__max_depth": [None, 10, 20, 30],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__max_features": ["sqrt", "log2"],
                },
            },
            "XGBoost": {
                "estimator": XGBClassifier(
                    eval_metric="mlogloss",
                    random_state=self.random_state,
                    use_label_encoder=False,
                    verbosity=0,
                ),
                "params": {
                    "classifier__n_estimators": [100, 200, 300],
                    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "classifier__max_depth": [3, 5, 7, 9],
                    "classifier__subsample": [0.7, 0.8, 1.0],
                    "classifier__colsample_bytree": [0.7, 0.8, 1.0],
                },
            },
        }

        best_pipeline   = None
        best_f1         = -1.0
        best_name       = ""
        results_summary = []

        for name, cfg in models.items():
            logger.info("── Training: %-25s ──", name)

            pipeline = ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("smote",        SMOTE(
                                         random_state=self.random_state,
                                         k_neighbors=k_neighbors,
                                     )),
                    ("classifier",   cfg["estimator"]),
                ]
            )

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=cfg["params"],
                n_iter=10,
                cv=self.cv_folds,
                scoring="f1_weighted",
                n_jobs=-1,
                random_state=self.random_state,
                error_score=0.0,
            )
            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )

            logger.info(
                "   Best params : %s", search.best_params_
            )
            logger.info(
                "   Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f",
                acc, prec, rec, f1,
            )

            results_summary.append(
                {"Model": name, "Accuracy": acc, "Precision": prec,
                 "Recall": rec, "F1_weighted": f1}
            )

            if f1 > best_f1:
                best_f1      = f1
                best_pipeline = search.best_estimator_
                best_name    = name

        # ── Summary table ───────────────────────────────────────────────
        summary_df = (
            pd.DataFrame(results_summary)
            .sort_values("F1_weighted", ascending=False)
            .reset_index(drop=True)
        )
        logger.info("\n%s", "=" * 65)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("\n%s", summary_df.to_string(index=False))
        logger.info("=" * 65)
        logger.info("BEST MODEL: %s  (F1=%.4f)", best_name, best_f1)
        logger.info("=" * 65)

        # ── Detailed report for best model ──────────────────────────────
        logger.info("STEP 5 — Detailed evaluation of best model")
        y_pred_best = best_pipeline.predict(X_test)

        print("\n" + "=" * 65)
        print(f"Classification Report — {best_name}")
        print("=" * 65)
        print(
            classification_report(
                y_test,
                y_pred_best,
                target_names=self.label_encoder.classes_,
                zero_division=0,
            )
        )
        print("Confusion Matrix (rows=actual, cols=predicted):")
        cm = confusion_matrix(y_test, y_pred_best)
        print(cm)
        print()

        return best_pipeline, best_name

    # ------------------------------------------------------------------
    # 6. FEATURE IMPORTANCE
    # ------------------------------------------------------------------

    def extract_feature_importance(
        self,
        pipeline,
        model_name: str,
        available_features: list,
        top_n: int = 15,
    ):
        logger.info("=" * 65)
        logger.info("STEP 6 — Feature importance extraction")
        logger.info("=" * 65)

        try:
            classifier   = pipeline.named_steps["classifier"]
            preprocessor = pipeline.named_steps["preprocessor"]

            # Recover feature names after encoding
            num_cols = [
                f for f in NUMERIC_FEATURES if f in available_features
            ]
            cat_cols = [
                f for f in CATEGORICAL_FEATURES if f in available_features
            ]
            cat_encoder       = preprocessor.named_transformers_["cat"]
            cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
            all_names         = num_cols + cat_feature_names

            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
                metric_label = "Importance"
            elif hasattr(classifier, "coef_"):
                coef = classifier.coef_
                importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef[0])
                metric_label = "Abs Coef (avg)"
            else:
                logger.warning("Feature importance not supported for %s", model_name)
                return

            if len(importances) != len(all_names):
                logger.warning(
                    "Feature count mismatch (%d importances vs %d names) — skipping",
                    len(importances), len(all_names),
                )
                return

            fi_df = (
                pd.DataFrame({"Feature": all_names, metric_label: importances})
                .sort_values(metric_label, ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
            fi_df.index += 1  # 1-based rank

            print(f"\nTop {top_n} most important features ({model_name}):")
            print(fi_df.to_string())
            print()

            fi_df.to_csv("feature_importance.csv", index=False)
            logger.info("Saved feature importance → feature_importance.csv")

        except Exception as exc:
            logger.warning("Could not extract feature importance: %s", exc)

    # ------------------------------------------------------------------
    # 7. SAVE
    # ------------------------------------------------------------------

    def save(self, pipeline, model_name: str):
        logger.info("=" * 65)
        logger.info("STEP 7 — Saving artifacts")
        logger.info("=" * 65)

        joblib.dump(pipeline, self.model_save_path)
        logger.info(
            "Best model (%s) saved → %s", model_name, self.model_save_path
        )

    # ------------------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------------------

    def run(self):
        logger.info("=" * 65)
        logger.info("SPORTS PERFORMANCE — RECOMMENDATION CLASSIFIER")
        logger.info("=" * 65)

        # 1. Load & clean
        df = self.load_and_preprocess()

        # 2. Build structured target
        df, y = self.build_target(df)

        # 3. Split
        X_train, X_test, y_train, y_test, available_features = self.split(df, y)

        # 4 + 5. Train, optimise, evaluate
        best_pipeline, best_name = self.train_and_evaluate(
            X_train, X_test, y_train, y_test, available_features
        )

        # 6. Feature importance
        self.extract_feature_importance(best_pipeline, best_name, available_features)

        # 7. Persist
        self.save(best_pipeline, best_name)

        logger.info("Pipeline complete. Ready for production.")
        logger.info(
            "Artifacts: %s | label_encoder.pkl | "
            "class_recommendation_lookup.csv | feature_importance.csv",
            self.model_save_path,
        )


# ===========================================================================
#  INFERENCE HELPER (for use after training)
# ===========================================================================

def predict_recommendation(
    sample: dict,
    model_path: str   = "best_model.pkl",
    encoder_path: str = "label_encoder.pkl",
    lookup_path: str  = "class_recommendation_lookup.csv",
) -> str:
    """
    Given a dict of feature values, returns the predicted recommendation text.

    Example
    -------
    >>> rec = predict_recommendation({
    ...     "heart_rate_bpm": 1.2, "performance_score": 85.0,
    ...     "injury_risk_score": 0.3, "risk_level": "Modéré",
    ...     "hr_zone": "Zone_Aerobie", "performance_level": "Excellente",
    ...     "event_type": "sprint", ...
    ... })
    >>> print(rec)
    """
    pipeline = joblib.load(model_path)
    encoder  = joblib.load(encoder_path)
    lookup   = pd.read_csv(lookup_path).set_index("target_label")

    X = pd.DataFrame([sample])
    # Ensure temporal columns exist
    for col in ("hour", "dayofweek"):
        if col not in X.columns:
            X[col] = 0

    encoded_pred  = pipeline.predict(X)[0]
    decoded_label = encoder.inverse_transform([encoded_pred])[0]

    if decoded_label in lookup.index:
        return lookup.loc[decoded_label, "recommendation_entrainement"]
    return f"(label={decoded_label} — no lookup entry found)"


# ===========================================================================
#  MAIN
# ===========================================================================

if __name__ == "__main__":
    DATA_PATH        = "../data_nettoyer.csv"   # ← adjust path as needed
    MODEL_SAVE_PATH  = "best_model.pkl"

    trainer = SportsRecommendationTrainer(
        data_path       = DATA_PATH,
        model_save_path = MODEL_SAVE_PATH,
        test_size       = 0.10,
        cv_folds        = 5,
        random_state    = 42,
    )
    trainer.run()