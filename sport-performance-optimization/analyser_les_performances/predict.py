import joblib
import pandas as pd

# ─────────────────────────────────────────────
#  Features (doit correspondre exactement à train_model.py)
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

LABELS_PERF = {0: "Insuffisante ❌", 1: "Bonne ✔️", 2: "Excellente 🏆"}


def predict_performance(perf_model_path, reco_model_path, athlete_data: dict):
    """
    Charge les deux modèles et retourne :
      - score_estime  : float  (performance_score prédit)
      - niveau_estime : str    (0=Insuffisante / 1=Bonne / 2=Excellente)
    """
    # ── Chargement des modèles
    try:
        perf_model = joblib.load(perf_model_path)
        reco_model = joblib.load(reco_model_path)
    except FileNotFoundError as e:
        print(f"❌ Modèle introuvable : {e}")
        return None, None

    # ── Mise en forme des données
    df_input = pd.DataFrame([athlete_data])

    # ── Prédictions
    score   = perf_model.predict(df_input)[0]
    niveau  = reco_model.predict(df_input)[0]

    return score, LABELS_PERF.get(niveau, str(niveau))


# ─────────────────────────────────────────────
#  Test rapide (exécution directe)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    PERF_MODEL = "./perf_model.pkl"
    RECO_MODEL = "./reco_model.pkl"

    # Exemple d'athlète (valeurs standardisées comme dans data_nettoyer.csv)
    exemple_seance = {
        'heart_rate_bpm'               :  0.63,
        'step_frequency_hz'            : -0.88,
        'stride_length_m'              :  1.66,
        'acceleration_mps2'            : -0.30,
        'gyroscope_x'                  : -0.55,
        'gyroscope_y'                  :  0.25,
        'gyroscope_z'                  : -0.07,
        'accelerometer_x'              : -1.63,
        'accelerometer_y'              : -1.30,
        'accelerometer_z'              :  0.23,
        'signal_energy'                :  1.06,
        'dominant_freq_hz'             :  1.05,
        # Flags one-hot : sprint avec phase de départ
        'event_type_high_jump'         : 1,
        'event_type_long_jump'         : 0,
        'event_type_sprint'            : 0,
        'motion_class_acceleration_phase': 0,
        'motion_class_flight_phase'    : 1,
        'motion_class_landing'         : 0,
        'motion_class_start_phase'     : 0,
        # Zone cardiaque et niveau de risque encodés
        'risk_level_encoded'           : 2,
        'hr_zone_encoded'              : 2,
    }

    score_estime, niveau_estime = predict_performance(PERF_MODEL, RECO_MODEL, exemple_seance)

    if score_estime is not None:
        print("\n" + "=" * 52)
        print("  🎯 RÉSULTAT DE L'INTELLIGENCE ARTIFICIELLE")
        print("=" * 52)
        print(f"  Score de performance estimé : {score_estime:.2f} / 100")
        print(f"  Niveau de performance       : {niveau_estime}")
        print("=" * 52)