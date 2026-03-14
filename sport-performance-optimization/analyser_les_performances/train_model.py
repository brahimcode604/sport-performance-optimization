import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

# ─────────────────────────────────────────────
#  Features communes aux deux modèles
# ─────────────────────────────────────────────
FEATURES = [
    'heart_rate_bpm', 'step_frequency_hz', 'stride_length_m',
    'acceleration_mps2', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
    'accelerometer_x', 'accelerometer_y', 'accelerometer_z',
    'signal_energy', 'dominant_freq_hz',
    'event_type_high_jump', 'event_type_long_jump', 'event_type_sprint',
    'motion_class_acceleration_phase', 'motion_class_flight_phase',
    'motion_class_landing', 'motion_class_start_phase',
    'risk_level_encoded', 'hr_zone_encoded'
]

# ─────────────────────────────────────────────
#  Modèle 1 : Prédiction du score de performance
# ─────────────────────────────────────────────
def train_performance_model(df, model_save_path):
    print("\n🎯 [Modèle 1] Entraînement : Score de Performance")

    X = df[FEATURES]
    y = df['performance_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2  = r2_score(y_test, predictions)

    print(f"   MAE  : {mae:.4f} (erreur moyenne absolue des scores)")
    print(f"   R²   : {r2:.4f}  (plus proche de 1 = meilleur)")

    joblib.dump(model, model_save_path)
    print(f"   ✅  Modèle sauvegardé → {model_save_path}")
    return model


# ─────────────────────────────────────────────
#  Modèle 2 : Classification du niveau de performance
# ─────────────────────────────────────────────
def train_recommendation_model(df, model_save_path):
    print("\n🏋️  [Modèle 2] Entraînement : Niveau de Performance (classification)")

    X = df[FEATURES]
    y = df['performance_level_encoded']   # 0=Insuffisante, 1=Bonne, 2=Excellente

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(classification_report(
        y_test, predictions,
        target_names=['Insuffisante', 'Bonne', 'Excellente'],
        zero_division=0
    ))

    joblib.dump(model, model_save_path)
    print(f"   ✅  Modèle sauvegardé → {model_save_path}")
    return model


# ─────────────────────────────────────────────
#  Point d'entrée principal
# ─────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH  = "../data_nettoyer.csv"
    PERF_MODEL = "./perf_model.pkl"
    RECO_MODEL = "./reco_model.pkl"

    print("=" * 55)
    print(" OPTIMISATION DES PERFORMANCES SPORTIVES - ENTRAÎNEMENT")
    print("=" * 55)

    # ── Chargement
    print(f"\n📂 Chargement : {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Erreur : fichier introuvable → {DATA_PATH}")
        exit(1)

    print(f"   {len(df):,} lignes · {df.shape[1]} colonnes chargées.")

    # ── Vérification des colonnes requises
    missing = [c for c in FEATURES + ['performance_score', 'performance_level_encoded']
               if c not in df.columns]
    if missing:
        print(f"❌ Colonnes manquantes : {missing}")
        exit(1)

    # ── Suppression des NaN éventuels
    df.dropna(subset=FEATURES + ['performance_score', 'performance_level_encoded'], inplace=True)
    print(f"   {len(df):,} lignes après suppression des NaN.")

    # ── Entraînements
    os.makedirs(os.path.dirname(PERF_MODEL) if os.path.dirname(PERF_MODEL) else ".", exist_ok=True)
    train_performance_model(df, PERF_MODEL)
    train_recommendation_model(df, RECO_MODEL)

    print("\n" + "=" * 55)
    print(" 🏁 Entraînement terminé avec succès !")
    print("=" * 55)