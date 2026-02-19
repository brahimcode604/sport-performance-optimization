import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

def predire_performance():
    """Fonction déclenchée lors du clic sur le bouton Prédire"""
    try:
        # 1. Récupération des données saisies dans l'interface
        donnees_athlete = {
            'age': float(entrees['age'].get()),
            'weight': float(entrees['weight'].get()),
            'resting_hr': float(entrees['resting_hr'].get()),
            'avg_hr': float(entrees['avg_hr'].get()),
            'max_hr': float(entrees['max_hr'].get()),
            'hrv': float(entrees['hrv'].get()),
            'vo2max': float(entrees['vo2max'].get()),
            'speed_avg': float(entrees['speed_avg'].get()),
            'distance_km': float(entrees['distance_km'].get()),
            'duration_min': float(entrees['duration_min'].get()),
            'training_load': float(entrees['training_load'].get()),
            'sleep_hours': float(entrees['sleep_hours'].get()),
            'fatigue_score': float(entrees['fatigue_score'].get())
        }

        # 2. Chargement du modèle
        chemin_modele = "./perf_model_v1.pkl"
        modele = joblib.load(chemin_modele)

        # 3. Prédiction
        df_nouvelle_seance = pd.DataFrame([donnees_athlete])
        prediction = modele.predict(df_nouvelle_seance)[0]

        # 4. Affichage du résultat dans l'interface
        label_resultat.config(text=f"Score de Performance Estimé :\n⭐ {prediction:.1f} / 100 ⭐", fg="green")

    except FileNotFoundError:
        messagebox.showerror("Erreur", "Le fichier 'perf_model_v1.pkl' est introuvable. Placez-le dans le même dossier.")
    except ValueError:
        messagebox.showerror("Erreur de saisie", "Veuillez vérifier que toutes les cases contiennent des nombres valides.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur inattendue est survenue : {e}")


# --- CRÉATION DE LA FENÊTRE PRINCIPALE ---
fenetre = tk.Tk()
fenetre.title("IA Sports Tech - Optimisation des Performances")
fenetre.geometry("400x650")
fenetre.configure(padx=20, pady=20)

titre = tk.Label(fenetre, text="Paramètres de la Séance", font=("Helvetica", 14, "bold"))
titre.grid(row=0, column=0, columnspan=2, pady=(0, 15))

# Les valeurs par défaut que vous avez fournies
valeurs_par_defaut = {
    'age': 28, 'weight': 72.5, 'resting_hr': 55.0, 'avg_hr': 155.0,
    'max_hr': 175.0, 'hrv': 65.0, 'vo2max': 50.0, 'speed_avg': 12.5,
    'distance_km': 10.0, 'duration_min': 48.0, 'training_load': 75.0,
    'sleep_hours': 8.0, 'fatigue_score': 0.5
}

entrees = {}
ligne = 1

# Création automatique des étiquettes et des champs de saisie
for variable, valeur in valeurs_par_defaut.items():
    # Création du texte (Label)
    label = tk.Label(fenetre, text=f"{variable} :", font=("Helvetica", 10))
    label.grid(row=ligne, column=0, sticky="e", pady=5, padx=5)
    
    # Création de la zone de saisie (Entry)
    champ = tk.Entry(fenetre, width=15, font=("Helvetica", 10))
    champ.insert(0, str(valeur)) # On pré-remplit avec votre valeur
    champ.grid(row=ligne, column=1, sticky="w", pady=5)
    
    # On stocke le champ pour pouvoir récupérer sa valeur plus tard
    entrees[variable] = champ
    ligne += 1

# --- BOUTON DE PRÉDICTION ---
bouton_predire = tk.Button(fenetre, text="Lancer l'IA (Prédire)", font=("Helvetica", 12, "bold"), bg="#0078D7", fg="white", command=predire_performance)
bouton_predire.grid(row=ligne, column=0, columnspan=2, pady=20, ipadx=10, ipady=5)

# --- ZONE DE RÉSULTAT ---
label_resultat = tk.Label(fenetre, text="En attente des données...", font=("Helvetica", 14, "bold"), fg="gray")
label_resultat.grid(row=ligne+1, column=0, columnspan=2, pady=10)

# Lancement de l'application
fenetre.mainloop()