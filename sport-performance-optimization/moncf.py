import numpy as np
import pandas as pd

#Supprimer les doublons
def Supprimer_doublons(nom_fich):
    
    #chargement du fichier CSV
    df = pd.read_csv(nom_fich , sep = ',' , encoding = 'UTF-8')
    
    #Supprimer les doublons selon
    df = df.drop_duplicates(keep = 'first')
    #if df.duplicated().sum() == 0:
        #return df
    #return Supprimer_doublons(data)
    return df
    

#Convertir les types de données
def Convertir_types_donnees(data):
    
    #Chargement du fichier CSV
    #df = pd.read_csv(data , sep = ',' , encoding = 'UTF-8')
    df = data.copy()
    
    #Sélection des colonnes numériques
    colonnes_numeriques = df.select_dtypes(include = [np.number]).columns
    
    # Exclure éventuellement 'athlete_id' , 'performance_score' et 'injury_risk' si ce sont des identifiants ou binaires
    colonnes_a_exclure = ['athlete_id','session_id','performance_score','injury_risk_score','timestamp']
    colonnes_a_traiter = [col for col in colonnes_numeriques if col not in colonnes_a_exclure]
    
    #Conversion des types
    for col in colonnes_a_traiter:
        df[col] = pd.to_numeric(df[col] , errors='coerce')  #errors='coerce' signifie que toute valeur non convertible en nombre est automatiquement remplacée par NaN.
    return df


def Detection_traitement_valeurs_aberrantes(data):
    #Chargement du fichier CSV
    #df = pd.read_csv(data , sep = ',',encoding = 'UTF-8')
    #print(df)
    df = data.copy()

    # Sélectionner uniquement les colonnes numériques
    colonnes_numeriques = df.select_dtypes(include=[np.number]).columns

    # Exclure éventuellement 'athlete_id' , 'performance_score' et 'injury_risk' si ce sont des identifiants ou binaires
    colonnes_a_exclure = ['athlete_id','session_id','performance_score','injury_risk_score','timestamp']
    colonnes_a_traiter = [col for col in colonnes_numeriques if col not in colonnes_a_exclure]

    #Initialisation du dictionnaire des outliers
    outliers_summary = {}

    #Copie du DataFrame original (sécurité)
    df_corrected = df.copy()

    # Calculer Q1, Q3 et IQR pour chaque colonne
    for col in colonnes_a_traiter:
    
        # Calcul des quartiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
    
        # Calcul de l'IQR
        IQR = Q3 - Q1
    
        # Définition des bornes
        borne_inf = Q1 - 1.5 * IQR
        borne_sup = Q3 + 1.5 * IQR
        # Correction des valeurs aberrantes
        df_corrected[col] = df_corrected[col].clip(borne_inf, borne_sup)
    return df_corrected

#Detection_raitement_valeurs_aberrantes().to_csv("sport_performance_dataset_corrected.csv", index=False)

def Gestion_valeurs_manquantes(data):
    
    #Chargement du fichier CSV
    #df = pd.read_csv(data,sep = ',',encoding = 'UTF-8')
    df = data.copy()
    
    #Calcul du taux de valeurs manquantes par ligne
    taux_manquants = df.isna().mean(axis = 1)
    
    #Suppression des observations avec > 30 % de valeurs manquantes
    seuil = 0.30
    df_clean = df[taux_manquants <= seuil].copy()
    
    #Sélection des colonnes numériques
    colonnes_numeriques = df_clean.select_dtypes(include=[np.number]).columns
    
    # Exclure éventuellement 'athlete_id' , 'performance_score' et 'injury_risk' si ce sont des identifiants ou binaires
    colonnes_a_exclure = ['athlete_id','session_id','performance_score','injury_risk_score','timestamp']
    colonnes_a_traiter = [col for col in colonnes_numeriques if col not in colonnes_a_exclure]
    
    #Imputation par la MÉDIANE
    for col in colonnes_a_traiter:
        mediane = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(mediane)  #inplace=True signifie que la modification est appliquée directement sur l’objet existant, sans créer une nouvelle copie.
    return df_clean

#Normalisation (Min–Max : valeurs entre 0 et 1)
def normalisation(data):
    
    #Chargement du fichier CSV
    #df = pd.read_csv(data , sep = ',' , encoding = 'UTF-8')
    df = data.copy()
    
    #Sélection des colonnes numériques
    colonnes_numeriques = df.select_dtypes(include=[np.number]).columns
    
    # Exclure éventuellement 'athlete_id' , 'performance_score' et 'injury_risk' si ce sont des identifiants ou binaires
    colonnes_a_exclure = ['athlete_id','session_id','performance_score','injury_risk_score','timestamp']
    colonnes_a_traiter = [col for col in colonnes_numeriques if col not in colonnes_a_exclure]
    
    # Normalisation Min-Max
    for col in colonnes_a_traiter:
        vmin = df[col].min()
        vmax = df[col].max()
        if vmin != vmax:
            df[col] = (df[col] - vmin) / (vmax-vmin)
    return df

#Standardisation
def Standardisation(data):
    
    #Chargement du fichier CSV
    #df = pd.read_csv(data , sep = ',' , encoding = 'UTF-8')
    df = data.copy()
    
    #Sélection des colonnes numériques
    colonnes_numeriques = df.select_dtypes(include=[np.number]).columns
    
    #Exclure éventuellement 'athlete_id' , 'performance_score' et 'injury_risk' si ce sont des identifiants ou binaires
    colonnes_a_exclure = ['athlete_id','session_id','performance_score','injury_risk_score','timestamp']
    colonnes_a_traiter = [col for col in colonnes_numeriques if col not in colonnes_a_exclure]
    
    for col in colonnes_a_traiter:
        moyen = df[col].mean()
        ecart_type = df[col].std()
        if ecart_type != 0 :
            df[col] = (df[col] - moyen) / ecart_type
    return df

def Encodage_des_variables_categorielles(data):
    df = data.copy()
    
    cat = ['event_type', 'motion_class']
    df = pd.get_dummies(df, columns=cat, prefix=cat, dtype=int)
    
    ordre_risk = {'Faible': 0, 'Modéré': 1, 'Élevé': 2, 'Critique': 3}
    df['risk_level_encoded'] = df['risk_level'].map(ordre_risk)
    
    ordre_hr = {'Zone_Basse': 0, 'Zone_Aerobie': 1, 'Zone_Anaerobie': 2, 'Zone_Max': 3}
    df['hr_zone_encoded'] = df['hr_zone'].map(ordre_hr)
    
    ordre_perf = {'Insuffisante': 0, 'Bonne': 1, 'Excellente': 2}
    df['performance_level_encoded'] = df['performance_level'].map(ordre_perf)
    
    return df


data_sans_doublons = Supprimer_doublons('athletes_full_recommendations.csv')
data_converti = Convertir_types_donnees(data_sans_doublons)
data_sans_valeures_aberrantes = Detection_traitement_valeurs_aberrantes(data_converti)
data_sans_valeures_manquantes = Gestion_valeurs_manquantes(data_sans_valeures_aberrantes)
#data_normaliser = normalisation(data_sans_valeures_manquantes)
data_standariser = Standardisation(data_sans_valeures_manquantes)
data_coder = Encodage_des_variables_categorielles(data_standariser)

data_coder.to_csv('data_nettoyer.csv',index=False)