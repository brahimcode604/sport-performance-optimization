import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_performance_model(data_path, model_save_path):
    print(" Démarrage de l'entraînement du modèle de Performance...")
    
    # 1. Chargement des données
    print("-> 1. Chargement des données...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {data_path} est introuvable.")
        return

    # 2. Définition des Features (X) et de la Target (y)
    print("-> 2. Préparation des variables...")
    # On sélectionne les colonnes qui influencent la performance
    features = [
        'age', 'weight', 'resting_hr', 'avg_hr', 'max_hr', 'hrv', 
        'vo2max', 'speed_avg', 'distance_km', 'duration_min', 
        'training_load', 'sleep_hours', 'fatigue_score'
    ]
    
    X = df[features]
    y = df['performance_score'] # Ce qu'on veut prédire

    # 3. Séparation en données d'entraînement (80%) et de test (20%)
    print("-> 3. Séparation des données (Train/Test Split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialisation et entraînement du modèle
    print("-> 4. Entraînement de l'algorithme (Random Forest)...")
    # n_estimators=100 signifie que l'on crée "100 arbres de décision" pour avoir un résultat robuste
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Évaluation des performances du modèle
    print("-> 5. Évaluation de la précision...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n📊 --- RÉSULTATS DU MODÈLE ---")
    print(f"Erreur Absolue Moyenne (MAE) : {mae:.2f} points")
    print(f"Score R2 (Explication de la variance) : {r2:.2f} (plus c'est proche de 1.0, mieux c'est)")
    print("------------------------------\n")

    # 6. Sauvegarde du modèle pour une utilisation future
    print("-> 6. Sauvegarde du modèle...")
    # Création du dossier 'models' s'il n'existe pas déjà
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # joblib permet de sauvegarder l'objet Python (le modèle) dans un fichier physique
    joblib.dump(model, model_save_path)
    print(f" Modèle sauvegardé avec succès sous : {model_save_path}")

if __name__ == "__main__":
    # Définition des chemins relatifs (basés sur l'arborescence de notre projet)
    # Assurez-vous d'avoir votre fichier CSV à cet emplacement !
    CHEMIN_DONNEES = "../sport_performance_dataset_2000.csv" 
    CHEMIN_MODELE = "./perf_model_v1.pkl"
    
    # Lancement de la fonction principale
    train_performance_model(CHEMIN_DONNEES, CHEMIN_MODELE)