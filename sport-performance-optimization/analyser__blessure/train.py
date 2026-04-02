"""
=============================================================================
  SPORTS INJURY — RECOMMENDATION CLASSIFIER  (train.py)
  Expert ML pipeline | Random Forest · XGBoost · Logistic Regression
=============================================================================

TARGET STRATEGY
---------------
`recommendation_blessure` is free text with hundreds of near-unique values.
We build a **structured composite label** from the 3 columns that
deterministically drive every injury recommendation:
    risk_level  +  hr_zone  +  event_type

→ ~12–48 balanced, semantically meaningful classes.
A downstream lookup table maps each class back to the full text.

USAGE
-----
    # from sport-performance-optimization/
    python analyser__blessure/train.py

OUTPUTS (saved in analyser__blessure/)
-------
    injury_model.pkl
    injury_encoder.pkl
    injury_lookup.csv
    injury_feature_importance.csv
=============================================================================
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

# Directory where artifacts will be saved (same folder as this script)
ARTIFACT_DIR = Path(__file__).parent

# Numeric features fed to the model
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
    "injury_risk_score",      # ← KEY feature for injury prediction
    "performance_score",
    "risk_level_encoded",
    "hr_zone_encoded",
    # Binary event flags
    "event_type_high_jump",
    "event_type_long_jump",
    "event_type_sprint",
    # Binary motion flags
    "motion_class_acceleration_phase",
    "motion_class_flight_phase",
    "motion_class_landing",
    "motion_class_start_phase",
]

# Categorical features (passed through OneHotEncoder inside pipeline)
CATEGORICAL_FEATURES = ["risk_level", "hr_zone", "event_type"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Components used to build the structured target label
LABEL_COMPONENTS = ["risk_level", "hr_zone", "event_type"]


# ===========================================================================
#  HELPERS
# ===========================================================================

def derive_event_type(df: pd.DataFrame) -> pd.Series:
    """Recover a clean event_type column from binary one-hot columns."""
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
#  TRAINER CLASS
# ===========================================================================

class InjuryRecommendationTrainer:
    """
    End-to-end ML pipeline for injury recommendation classification:
        load → clean → engineer → split → train (3 models) → evaluate → save
    """

    def __init__(
        self,
        data_path: str,
        model_save_path: str | None = None,
        test_size: float = 0.10,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.data_path       = Path(data_path)
        self.model_save_path = Path(model_save_path) if model_save_path \
                               else ARTIFACT_DIR / "injury_model.pkl"
        self.test_size       = test_size
        self.cv_folds        = cv_folds
        self.random_state    = random_state
        self.label_encoder   = LabelEncoder()

    # ------------------------------------------------------------------
    # STEP 1 — LOAD & CLEAN
    # ------------------------------------------------------------------
    def load_and_preprocess(self) -> pd.DataFrame:
        logger.info("=" * 65)
        logger.info("STEP 1 — Loading dataset: %s", self.data_path)
        logger.info("=" * 65)

        df = pd.read_csv(self.data_path)
        logger.info("Raw shape: %s", df.shape)

        # Drop rows without target
        initial = len(df)
        df = df.dropna(subset=["recommendation_blessure"])
        logger.info(
            "Dropped %d rows with missing target → %d rows remain",
            initial - len(df), len(df),
        )

        # Derive clean event_type column
        df["event_type"] = derive_event_type(df)

        # Impute numeric features
        num_base = [c for c in NUMERIC_FEATURES
                    if c in df.columns and c not in
                    ("event_type_high_jump","event_type_long_jump","event_type_sprint",
                     "motion_class_acceleration_phase","motion_class_flight_phase",
                     "motion_class_landing","motion_class_start_phase")]
        for col in num_base:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

        # Fill binary flags with 0
        binary_cols = [
            "event_type_high_jump","event_type_long_jump","event_type_sprint",
            "motion_class_acceleration_phase","motion_class_flight_phase",
            "motion_class_landing","motion_class_start_phase",
        ]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Impute categorical
        for col in ["risk_level", "hr_zone"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        logger.info(
            "Feature value counts:\n%s",
            df[LABEL_COMPONENTS].apply(lambda c: c.nunique()).to_string(),
        )
        return df

    # ------------------------------------------------------------------
    # STEP 2 — BUILD STRUCTURED TARGET
    # ------------------------------------------------------------------
    def build_target(self, df: pd.DataFrame):
        """
        Build composite label: e.g. 'Critique__Zone_Max__sprint'
        Save lookup table: label → recommendation_blessure text
        """
        logger.info("=" * 65)
        logger.info("STEP 2 — Engineering structured target label")
        logger.info("=" * 65)

        df["target_label"] = (
            df["risk_level"].astype(str) + "__"
            + df["hr_zone"].astype(str)  + "__"
            + df["event_type"].astype(str)
        )

        n_classes    = df["target_label"].nunique()
        class_counts = df["target_label"].value_counts()
        logger.info("Unique structured target classes: %d", n_classes)
        logger.info(
            "Class size — min: %d | median: %.0f | max: %d",
            class_counts.min(), class_counts.median(), class_counts.max(),
        )

        # Drop singletons (cannot be stratified)
        valid = class_counts[class_counts >= 2].index
        dropped = (~df["target_label"].isin(valid)).sum()
        if dropped:
            logger.warning("Removed %d singleton rows", dropped)
            df = df[df["target_label"].isin(valid)].copy()

        # Build lookup: label → first recommendation text found
        lookup = (
            df.groupby("target_label")["recommendation_blessure"]
            .first()
            .reset_index()
        )
        lookup.columns = ["target_label", "recommendation_blessure"]
        lookup_path = ARTIFACT_DIR / "injury_lookup.csv"
        lookup.to_csv(lookup_path, index=False, encoding="utf-8-sig")
        logger.info("Saved lookup table → %s (%d classes)", lookup_path, len(lookup))

        # Encode target
        y = self.label_encoder.fit_transform(df["target_label"])
        encoder_path = ARTIFACT_DIR / "injury_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        logger.info("Saved LabelEncoder → %s", encoder_path)

        return df, y

    # ------------------------------------------------------------------
    # STEP 3 — TRAIN / TEST SPLIT
    # ------------------------------------------------------------------
    def split(self, df: pd.DataFrame, y: np.ndarray):
        logger.info("=" * 65)
        logger.info(
            "STEP 3 — Train/test split (%.0f%% / %.0f%%, stratified)",
            (1 - self.test_size) * 100, self.test_size * 100,
        )
        logger.info("=" * 65)

        available = [f for f in ALL_FEATURES if f in df.columns]
        X = df[available]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info(
            "Train: %d rows | Test: %d rows | Features: %d",
            len(X_train), len(X_test), X.shape[1],
        )
        return X_train, X_test, y_train, y_test, available

    # ------------------------------------------------------------------
    # STEP 4 — PREPROCESSOR
    # ------------------------------------------------------------------
    def build_preprocessor(self, available_features: list) -> ColumnTransformer:
        num_cols = [f for f in NUMERIC_FEATURES   if f in available_features]
        cat_cols = [f for f in CATEGORICAL_FEATURES if f in available_features]
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(),                        num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"),  cat_cols),
            ],
            remainder="drop",
        )

    # ------------------------------------------------------------------
    # STEP 5 — TRAIN & EVALUATE (3 models)
    # ------------------------------------------------------------------
    def train_and_evaluate(
        self,
        X_train, X_test, y_train, y_test,
        available_features: list,
    ):
        logger.info("=" * 65)
        logger.info("STEP 4 — Model training with SMOTE + RandomizedSearchCV")
        logger.info("=" * 65)

        preprocessor   = self.build_preprocessor(available_features)
        min_class_size = np.bincount(y_train).min()
        k_neighbors    = min(5, min_class_size - 1)
        logger.info(
            "SMOTE k_neighbors=%d  (min class size=%d)",
            k_neighbors, min_class_size,
        )

        models = {
            "Logistic Regression": {
                "estimator": LogisticRegression(
                    max_iter=1000, random_state=self.random_state, solver="saga"
                ),
                "params": {
                    "classifier__C":       [0.01, 0.1, 1.0, 10.0],
                    "classifier__penalty": ["l1", "l2"],
                },
            },
            "Random Forest": {
                "estimator": RandomForestClassifier(
                    random_state=self.random_state, class_weight="balanced"
                ),
                "params": {
                    "classifier__n_estimators":    [100, 200, 300],
                    "classifier__max_depth":       [None, 10, 20, 30],
                    "classifier__min_samples_split":[2, 5, 10],
                    "classifier__max_features":    ["sqrt", "log2"],
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
                    "classifier__n_estimators":    [100, 200, 300],
                    "classifier__learning_rate":   [0.01, 0.05, 0.1, 0.2],
                    "classifier__max_depth":       [3, 5, 7, 9],
                    "classifier__subsample":       [0.7, 0.8, 1.0],
                    "classifier__colsample_bytree":[0.7, 0.8, 1.0],
                },
            },
        }

        best_pipeline = None
        best_f1       = -1.0
        best_name     = ""
        results_log   = []

        for name, cfg in models.items():
            logger.info("── Training: %-25s ──", name)

            pipeline = ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                ("smote",        SMOTE(
                                     random_state=self.random_state,
                                     k_neighbors=k_neighbors,
                                 )),
                ("classifier",   cfg["estimator"]),
            ])

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

            logger.info("   Best params : %s", search.best_params_)
            logger.info(
                "   Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f",
                acc, prec, rec, f1,
            )
            results_log.append(
                {"Model": name, "Accuracy": acc,
                 "Precision": prec, "Recall": rec, "F1_weighted": f1}
            )

            if f1 > best_f1:
                best_f1      = f1
                best_pipeline = search.best_estimator_
                best_name    = name

        # Summary table
        summary = (
            pd.DataFrame(results_log)
            .sort_values("F1_weighted", ascending=False)
            .reset_index(drop=True)
        )
        logger.info("\n%s\nMODEL COMPARISON\n%s\n%s",
                    "=" * 65, summary.to_string(index=False), "=" * 65)
        logger.info("BEST MODEL: %s  (F1=%.4f)", best_name, best_f1)

        # Detailed report for best model
        logger.info("STEP 5 — Detailed evaluation of best model")
        y_pred_best = best_pipeline.predict(X_test)
        print("\n" + "=" * 65)
        print(f"Classification Report — {best_name}")
        print("=" * 65)
        print(classification_report(
            y_test, y_pred_best,
            target_names=self.label_encoder.classes_,
            zero_division=0,
        ))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_best))

        return best_pipeline, best_name

    # ------------------------------------------------------------------
    # STEP 6 — FEATURE IMPORTANCE
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

            num_cols = [f for f in NUMERIC_FEATURES   if f in available_features]
            cat_cols = [f for f in CATEGORICAL_FEATURES if f in available_features]
            cat_enc  = preprocessor.named_transformers_["cat"]
            cat_names = list(cat_enc.get_feature_names_out(cat_cols))
            all_names = num_cols + cat_names

            if hasattr(classifier, "feature_importances_"):
                importances  = classifier.feature_importances_
                metric_label = "Importance"
            elif hasattr(classifier, "coef_"):
                coef = classifier.coef_
                importances  = np.abs(coef).mean(axis=0) if coef.ndim > 1 \
                               else np.abs(coef[0])
                metric_label = "Abs Coef (avg)"
            else:
                logger.warning("Feature importance not available for %s", model_name)
                return

            if len(importances) != len(all_names):
                logger.warning(
                    "Feature count mismatch (%d vs %d) — skipping",
                    len(importances), len(all_names),
                )
                return

            fi_df = (
                pd.DataFrame({"Feature": all_names, metric_label: importances})
                .sort_values(metric_label, ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
            fi_df.index += 1
            print(f"\nTop {top_n} features ({model_name}):\n{fi_df.to_string()}\n")

            save_path = ARTIFACT_DIR / "injury_feature_importance.csv"
            fi_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            logger.info("Saved feature importance → %s", save_path)

        except Exception as exc:
            logger.warning("Could not extract feature importance: %s", exc)

    # ------------------------------------------------------------------
    # STEP 7 — SAVE
    # ------------------------------------------------------------------
    def save(self, pipeline, model_name: str):
        logger.info("=" * 65)
        logger.info("STEP 7 — Saving best model: %s", model_name)
        logger.info("=" * 65)
        joblib.dump(pipeline, self.model_save_path)
        logger.info("Saved → %s", self.model_save_path)

    # ------------------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------------------
    def run(self):
        logger.info("=" * 65)
        logger.info("INJURY RECOMMENDATION CLASSIFIER — START")
        logger.info("=" * 65)

        df                                         = self.load_and_preprocess()
        df, y                                      = self.build_target(df)
        X_train, X_test, y_train, y_test, features = self.split(df, y)
        best_pipeline, best_name                   = self.train_and_evaluate(
            X_train, X_test, y_train, y_test, features
        )
        self.extract_feature_importance(best_pipeline, best_name, features)
        self.save(best_pipeline, best_name)

        logger.info("Pipeline complete.")
        logger.info(
            "Artifacts saved in: %s", ARTIFACT_DIR.resolve()
        )


# ===========================================================================
#  MAIN
# ===========================================================================
if __name__ == "__main__":
    # Path to data_nettoyer.csv  (one level up from this script)
    DATA_PATH = Path(__file__).parent.parent / "data_nettoyer.csv"

    trainer = InjuryRecommendationTrainer(
        data_path=str(DATA_PATH),
        test_size=0.10,
        cv_folds=5,
        random_state=42,
    )
    trainer.run()
