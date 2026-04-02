# 📉 Optimisation des Performances Sportives

> Analyse de données biométriques par Machine Learning pour optimiser les performances des athlètes et fournir des recommandations personnalisées.

---

## ![Description](https://img.shields.io/badge/Description-Projet-blue?style=flat-square&logo=googledocs&logoColor=white) Description du projet

Ce projet analyse les données biométriques collectées lors de séances sportives (fréquence cardiaque, accélération, fréquence de pas, gyroscope, etc.) afin de :

- **Comprendre** les facteurs influençant la performance sportive
- **Prédire** le score de performance d'un athlète
- **Classifier** son niveau (Insuffisant / Bon / Excellent)
- **Recommander** des ajustements personnalisés d'entraînement

---

## ![Structure](https://img.shields.io/badge/Structure-Dossiers-orange?style=flat-square&logo=gitbook&logoColor=white) Structure du projet

```
project_python/
│
└── sport-performance-optimization/
    │
    ├── athletes_full_recommendations.csv   # Dataset brut (biométrie + recommandations)
    ├── data_nettoyer.csv                   # Dataset nettoyé et standardisé
    ├── moncf.py                            # Script de nettoyage des données
    ├── main.py                             # Fichier d'entrée du projet
    ├── requirements.txt                    # Dépendances Python
    ├── README.md                           # Ce fichier
    │
    ├── analyser_les_performances/          # 🏆 Module de Prédiction des Performances
    │   ├── train_model.py                  # Entraînement des modèles ML (Performance)
    │   ├── predict.py                      # Test de prédiction en ligne de commande
    │   ├── app_gui.py                      # Interface graphique (Tkinter)
    │   ├── perf_model.pkl                  # Modèle de régression Score
    │   ├── reco_model.pkl                  # Modèle de classification Niveau
    │   └── performance_lookup.csv          # Classes vers texte de recommandation
    │
    └── analyser__blessure/                 # 🩺 Module de Prévention des Blessures
        ├── train_model.py ou train.py      # Entraînement du modèle ML (Blessure)
        ├── predict.py                      # Inférence (unitaire et batch)
        ├── app_gui.py                      # Dashboard graphique (PyQt6)
        ├── injury_model.pkl                # Modèle de régression logistique / XGBoost
        ├── injury_encoder.pkl              # Encodeur de la target (LabelEncoder)
        └── injury_lookup.csv               # Classes vers texte de recommandation
```

---

## ![Architecture](https://img.shields.io/badge/Architecture-Système-6C63FF?style=flat-square&logo=diagrams.net&logoColor=white) Architecture du Système

```mermaid
flowchart TD
    subgraph Data_Layer [Couche Données]
        RawData["athletes_full_recommendations.csv<br/>(Données Brutes)"]
        CleanData["data_nettoyer.csv<br/>(Données Standardisées)"]
    end

    subgraph Processing_Layer [Couche Traitement]
        Cleaning["moncf.py<br/>(Nettoyage, IQR, Z-Score)"]
    end

    subgraph Model_Layer [Couche IA / Modèles]
        subgraph Perf_Module [Module Performance]
            TrainPerf["train_model.py"]
            PerfModel["perf_model.pkl<br/>reco_model.pkl"]
        end
        subgraph Injury_Module [Module Blessure]
            TrainInj["train.py"]
            InjModel["injury_model.pkl<br/>injury_lookup.csv"]
        end
    end

    subgraph Application_Layer [Couche Application]
        GUI_Perf["app_gui.py (Tkinter)<br/>Dashboard Performance"]
        GUI_Inj["app_gui.py (PyQt6)<br/>Dashboard Blessures"]
    end

    RawData --> Cleaning
    Cleaning --> CleanData
    CleanData --> TrainPerf
    CleanData --> TrainInj
    
    TrainPerf --> PerfModel
    TrainInj --> InjModel
    
    PerfModel --> GUI_Perf
    InjModel --> GUI_Inj

    style RawData fill:#f9f,stroke:#333,stroke-width:2px
    style CleanData fill:#f9f,stroke:#333,stroke-width:2px
    style PerfModel fill:#6C63FF,stroke:#fff,stroke-width:2px,color:#fff
    style InjModel fill:#E74C3C,stroke:#fff,stroke-width:2px,color:#fff
    style GUI_Perf fill:#0078D7,stroke:#fff,stroke-width:2px,color:#fff
    style GUI_Inj fill:#0078D7,stroke:#fff,stroke-width:2px,color:#fff
```

---

## ![Pipeline](https://img.shields.io/badge/Pipeline-Traitement-success?style=flat-square&logo=githubactions&logoColor=white) Pipeline de traitement

### Tâche 1 — Collecte et nettoyage des données (`moncf.py`)

Le script `moncf.py` applique les étapes suivantes sur `athletes_full_recommendations.csv` :

1. **Suppression des doublons**
2. **Conversion des types numériques** (`pd.to_numeric`)
3. **Détection des valeurs aberrantes** (méthode IQR + clipping)
4. **Gestion des valeurs manquantes** (imputation médiane, suppression si > 30% manquants)
5. **Standardisation Z-score** (moyenne = 0, écart-type = 1)
6. **Encodage des variables catégorielles** :
   - One-hot : `event_type`, `motion_class`
   - Ordinal : `risk_level`, `hr_zone`, `performance_level`

**Résultat :** `data_nettoyer.csv` — 10 000+ lignes, 32 colonnes

---

### Tâche 2 — Prédiction des Performances (`analyser_les_performances/`)

Entraînement de modèles de performance via **Random Forest** et **XGBoost** sur 21 variables.
Deux modèles générés :
- `perf_model.pkl` : Régression pour le score de performance (0-100).
- `reco_model.pkl` : Classification pour le niveau global.
L'interface `app_gui.py` (Tkinter) offre une vue sur les recommandations liées aux performances brutes.

---

### Tâche 3 — Prévention des Blessures (`analyser__blessure/`)

Génération de modèles de Machine Learning prédictifs avec stratégie **SMOTE** (pour traiter le déséquilibre des données) et paramétrage automatique (GridSearchCV).
- Utilise la régression logistique / XGBoost pour classifier précisément le texte descriptif des recommandations médicales/préventives selon la zone de rythme cardiaque, le risque et le type de saut/course.
- `app_gui.py` (PyQt6) lance un **Dashboard analytique poussé**, incluant jauges, graphiques Radar, importance des variables, et table d'historique.

---

### Tâche 4 — Interface Unifiée & Test CLI

Chaque module dispose de son app_gui mais également de `predict.py` pour lancer des requêtes depuis le terminal (inférence simple ou par batch un fichier csv entier).

---

## ![Utilisation](https://img.shields.io/badge/Installation-Utilisation-success?style=flat-square&logo=rocket&logoColor=white) Installation et utilisation

### 1. Installer les dépendances

```bash
pip install -r sport-performance-optimization/requirements.txt
```

### 2. Nettoyer les données

```bash
cd sport-performance-optimization
python moncf.py
```

### 3. Entraîner les modèles

```bash
python analyser_les_performances/train_model.py
```

### 4. (Optionnel) Tester les prédictions en console

```bash
python analyser_les_performances/predict.py
```

### 5. Lancer l'interface graphique

```bash
python analyser_les_performances/app_gui.py
```

---

## ![Technologies](https://img.shields.io/badge/Technologies-Outils-informational?style=flat-square&logo=stackshare&logoColor=white) Technologies utilisées

| Outil | Rôle |
|-------|------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Langage principal |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) / ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Manipulation et nettoyage des données |
| ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | Modèles ML (RandomForest, XGBoost, LogReg, SMOTE imblearn) |
| ![Joblib](https://img.shields.io/badge/Joblib-grey?style=flat-square) | Sauvegarde / chargement des modèles |
| ![PyQt6 & Tkinter](https://img.shields.io/badge/Desktop_GUI-green?style=flat-square) | PyQt6 + Matplotlib pour le Dashboard Blessure ; Tkinter pour la performance |

---

## ![Dataset](https://img.shields.io/badge/Dataset-Donn%C3%A9es-lightgrey?style=flat-square&logo=database&logoColor=white) À propos du dataset

`athletes_full_recommendations.csv` contient des enregistrements biométriques pour trois épreuves :

| Épreuve | Colonne |
|---------|---------|
| Sprint | `event_type_sprint` |
| Saut en hauteur | `event_type_high_jump` |
| Saut en longueur | `event_type_long_jump` |

Chaque enregistrement inclut un score de performance, un niveau de risque de blessure et des recommandations textuelles personnalisées.

---

## ![Équipe](https://img.shields.io/badge/%C3%89quipe-Membres-important?style=flat-square&logo=microsoftteams&logoColor=white) Équipe

- **Brahim EL BAHLOUL**
- **Yassine Mokrame**
- **Mounssif Saih**
- **Jaouad Nainiaa**
- **Bargaa Issa**

---

Projet réalisé dans le cadre d'un cours sur l'optimisation des performances sportives par intelligence artificielle.
