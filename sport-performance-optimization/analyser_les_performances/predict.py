import joblib
import pandas as pd

def predict_performance(model_path, athlete_data):
    print(" Démarrage du module de prédiction...")
    
    # 1. Chargement du modèle préalablement entraîné
    try:
        model = joblib.load(model_path)
        print("-> Modèle chargé avec succès.")
    except FileNotFoundError:
        print(f" Erreur : Le modèle {model_path} est introuvable.")
        return None

    # 2. Transformation des données d'entrée en tableau (DataFrame)
    # Le modèle a besoin que les données soient présentées exactement 
    # dans le même ordre et format que lors de l'entraînement.
    df_new_data = pd.DataFrame([athlete_data])

    # 3. Génération de la prédiction
    print("-> Analyse des biométriques en cours...\n")
    prediction = model.predict(df_new_data)

    # On retourne la première (et unique) prédiction du tableau
    return prediction[0]

if __name__ == "__main__":
    # Le chemin vers le fichier que vous venez de générer
    CHEMIN_MODELE = "./perf_model_v1.pkl"

    # Simulation : Imaginons un athlète (ex: 28 ans, 72kg) qui prévoit une 
    # séance de 10km, après avoir dormi 8h et avec un bon niveau de VFC (65).
    # IMPORTANT : Il faut fournir les 13 variables exactes utilisées à l'entraînement !
    nouvelle_seance = {
        'age': 28,
        'weight': 72.5,
        'resting_hr': 55.0,
        'avg_hr': 155.0,
        'max_hr': 175.0,
        'hrv': 65.0,
        'vo2max': 50.0,
        'speed_avg': 12.5,
        'distance_km': 10.0,
        'duration_min': 48.0,
        'training_load': 75.0,
        'sleep_hours': 8.0,
        'fatigue_score': 0.5
    }

    # Lancement de la fonction
    score_estime = predict_performance(CHEMIN_MODELE, nouvelle_seance)

    # Affichage du résultat
    if score_estime is not None:
        print("🎯 --- RÉSULTAT DE L'INTELLIGENCE ARTIFICIELLE ---")
        print(f"Performance estimée pour cette séance : {score_estime:.1f} / 100")
        print("--------------------------------------------------")