"""
app_gui.py — Interface graphique : Optimisation des Performances Sportives
Utilise les modèles perf_model.pkl (régression) et reco_model.pkl (classification).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERF_MODEL_PATH = os.path.join(SCRIPT_DIR, "perf_model.pkl")
RECO_MODEL_PATH = os.path.join(SCRIPT_DIR, "reco_model.pkl")

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

LABELS_PERF = {0: "Insuffisante", 1: "Bonne", 2: "Excellente"}
LABELS_PERF_COLOR = {0: "#E74C3C", 1: "#F39C12", 2: "#2ECC71"}
LABELS_PERF_EMOJI = {0: "❌", 1: "✔️", 2: "🏆"}

# Valeurs par défaut (normalisées, basées sur data_nettoyer.csv)
DEFAULTS = {
    'heart_rate_bpm'                  : '0.63',
    'step_frequency_hz'               : '-0.88',
    'stride_length_m'                 : '1.66',
    'acceleration_mps2'               : '-0.30',
    'gyroscope_x'                     : '-0.55',
    'gyroscope_y'                     : '0.25',
    'gyroscope_z'                     : '-0.07',
    'accelerometer_x'                 : '-1.63',
    'accelerometer_y'                 : '-1.30',
    'accelerometer_z'                 : '0.23',
    'signal_energy'                   : '1.06',
    'dominant_freq_hz'                : '1.05',
    'event_type_high_jump'            : '1',
    'event_type_long_jump'            : '0',
    'event_type_sprint'               : '0',
    'motion_class_acceleration_phase' : '0',
    'motion_class_flight_phase'       : '1',
    'motion_class_landing'            : '0',
    'motion_class_start_phase'        : '0',
    'risk_level_encoded'              : '2',
    'hr_zone_encoded'                 : '2',
}

LABELS_FR = {
    'heart_rate_bpm'                  : 'Fréquence cardiaque (std)',
    'step_frequency_hz'               : 'Fréquence de pas (std)',
    'stride_length_m'                 : 'Longueur de foulée (std)',
    'acceleration_mps2'               : 'Accélération (std)',
    'gyroscope_x'                     : 'Gyroscope X (std)',
    'gyroscope_y'                     : 'Gyroscope Y (std)',
    'gyroscope_z'                     : 'Gyroscope Z (std)',
    'accelerometer_x'                 : 'Accéléromètre X (std)',
    'accelerometer_y'                 : 'Accéléromètre Y (std)',
    'accelerometer_z'                 : 'Accéléromètre Z (std)',
    'signal_energy'                   : 'Énergie du signal (std)',
    'dominant_freq_hz'                : 'Fréquence dominante (std)',
    'event_type_high_jump'            : 'Saut en hauteur (0/1)',
    'event_type_long_jump'            : 'Saut en longueur (0/1)',
    'event_type_sprint'               : 'Sprint (0/1)',
    'motion_class_acceleration_phase' : 'Phase accélération (0/1)',
    'motion_class_flight_phase'       : 'Phase vol (0/1)',
    'motion_class_landing'            : 'Phase atterrissage (0/1)',
    'motion_class_start_phase'        : 'Phase départ (0/1)',
    'risk_level_encoded'              : 'Niveau risque (0-3)',
    'hr_zone_encoded'                 : 'Zone cardiaque (0-3)',
}

# ─────────────────────────────────────────────
#  Palette de couleurs
# ─────────────────────────────────────────────
BG_DARK    = "#1A1E2E"
BG_CARD    = "#252A3D"
BG_FIELD   = "#0F1220"
ACCENT     = "#6C63FF"
ACCENT_L   = "#8B83FF"
TEXT_PRI   = "#FFFFFF"
TEXT_SEC   = "#A0A8C0"
BORDER     = "#343A52"
BTN_HOV    = "#5A52E0"

# ─────────────────────────────────────────────
#  Logique de prédiction
# ─────────────────────────────────────────────
def predire():
    """Lecture des champs → prédiction → affichage."""
    try:
        data = {feat: float(entrees[feat].get()) for feat in FEATURES}
    except ValueError:
        messagebox.showerror(
            "Erreur de saisie",
            "Tous les champs doivent contenir des valeurs numériques valides."
        )
        return

    # Chargement des modèles
    try:
        perf_model = joblib.load(PERF_MODEL_PATH)
        reco_model = joblib.load(RECO_MODEL_PATH)
    except FileNotFoundError as e:
        messagebox.showerror(
            "Modèle manquant",
            f"Fichier introuvable : {e}\n\n"
            "Veuillez d'abord exécuter train_model.py pour générer les modèles."
        )
        return

    df_in = pd.DataFrame([data])
    score  = perf_model.predict(df_in)[0]
    niveau = int(reco_model.predict(df_in)[0])

    couleur = LABELS_PERF_COLOR.get(niveau, "#FFFFFF")
    emojie  = LABELS_PERF_EMOJI.get(niveau, "")
    label   = LABELS_PERF.get(niveau, "Inconnu")

    # ── Mise à jour de la zone de résultat ──
    lbl_score.config(
        text=f"Score de Performance\n{score:.1f} / 100",
        fg=couleur
    )
    lbl_niveau.config(
        text=f"{emojie}  Niveau : {label}",
        fg=couleur
    )

    # Barre de progression
    progress_var.set(min(score, 100))

    # Recommandation textuelle
    if niveau == 0:
        reco = (
            "💡 Recommandations :\n"
            "• Réduisez le volume d'entraînement de 20%.\n"
            "• Concentrez-vous sur la technique et la récupération.\n"
            "• Sommeil ≥ 8h, nutrition optimisée."
        )
    elif niveau == 1:
        reco = (
            "💡 Recommandations :\n"
            "• Augmentez progressivement la charge (+5–10%/semaine).\n"
            "• Cycle 3+1 : 3 semaines de charge, 1 semaine de récupération.\n"
            "• Intégrez des exercices spécifiques à votre discipline."
        )
    else:
        reco = (
            "💡 Recommandations :\n"
            "• Introduisez du cross-training et de la plyométrie.\n"
            "• Surveillez les signaux de surentraînement (HRV).\n"
            "• Variez les intensités pour éviter la stagnation."
        )
    lbl_reco.config(text=reco)


def reset_fields():
    """Réinitialise tous les champs aux valeurs par défaut."""
    for feat, widget in entrees.items():
        widget.delete(0, tk.END)
        widget.insert(0, DEFAULTS[feat])
    lbl_score.config(text="Score de Performance\n— / 100", fg=TEXT_SEC)
    lbl_niveau.config(text="En attente de l'analyse…", fg=TEXT_SEC)
    lbl_reco.config(text="")
    progress_var.set(0)


# ─────────────────────────────────────────────
#  Construction de l'interface
# ─────────────────────────────────────────────
root = tk.Tk()
root.title("IA Sports Tech — Optimisation des Performances")
root.configure(bg=BG_DARK)
root.resizable(False, False)

# ── Style ttk ──
style = ttk.Style()
style.theme_use("clam")
style.configure(
    "bar.Horizontal.TProgressbar",
    troughcolor=BG_FIELD,
    background=ACCENT,
    bordercolor=BG_FIELD,
    lightcolor=ACCENT_L,
    darkcolor=ACCENT
)

# ═══════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════
header = tk.Frame(root, bg=BG_DARK, pady=18)
header.pack(fill="x", padx=20)

tk.Label(
    header,
    text="⚡  IA Sports Tech",
    font=("Helvetica", 22, "bold"),
    bg=BG_DARK, fg=TEXT_PRI
).pack(anchor="w")

tk.Label(
    header,
    text="Optimisation des performances sportives par intelligence artificielle",
    font=("Helvetica", 10),
    bg=BG_DARK, fg=TEXT_SEC
).pack(anchor="w")

tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=20)

# ═══════════════════════════════════════════
#  BODY : formulaire + résultats côte à côte
# ═══════════════════════════════════════════
body = tk.Frame(root, bg=BG_DARK)
body.pack(fill="both", expand=True, padx=20, pady=12)

# ── Panneau gauche : saisie des paramètres ──
left = tk.Frame(body, bg=BG_CARD, bd=0, relief="flat",
                highlightbackground=BORDER, highlightthickness=1)
left.pack(side="left", fill="both", expand=True, padx=(0, 8))

tk.Label(
    left,
    text="Paramètres Biométriques",
    font=("Helvetica", 11, "bold"),
    bg=BG_CARD, fg=ACCENT_L,
    padx=14, pady=10
).grid(row=0, column=0, columnspan=4, sticky="w")

entrees = {}
for idx, feat in enumerate(FEATURES):
    col_base = (idx % 2) * 2       # 0 ou 2
    row      = (idx // 2) + 1

    tk.Label(
        left,
        text=LABELS_FR[feat],
        font=("Helvetica", 9),
        bg=BG_CARD, fg=TEXT_SEC,
        anchor="e", width=22
    ).grid(row=row, column=col_base, sticky="e", padx=(10, 4), pady=3)

    e = tk.Entry(
        left,
        font=("Helvetica", 9),
        bg=BG_FIELD, fg=TEXT_PRI,
        insertbackground=TEXT_PRI,
        relief="flat", bd=4,
        width=10
    )
    e.insert(0, DEFAULTS[feat])
    e.grid(row=row, column=col_base + 1, sticky="w", pady=3, padx=(0, 12))
    entrees[feat] = e

# ── Panneau droit : résultats ──
right = tk.Frame(body, bg=BG_CARD, bd=0, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1,
                 width=260)
right.pack(side="right", fill="y", padx=(8, 0))
right.pack_propagate(False)

tk.Label(
    right,
    text="Résultats de l'IA",
    font=("Helvetica", 11, "bold"),
    bg=BG_CARD, fg=ACCENT_L,
    padx=14, pady=10
).pack(anchor="w")

tk.Frame(right, bg=BORDER, height=1).pack(fill="x", padx=14)

lbl_score = tk.Label(
    right,
    text="Score de Performance\n— / 100",
    font=("Helvetica", 18, "bold"),
    bg=BG_CARD, fg=TEXT_SEC,
    pady=20
)
lbl_score.pack()

# Barre de progression
progress_var = tk.DoubleVar(value=0)
ttk.Progressbar(
    right,
    variable=progress_var,
    maximum=100,
    style="bar.Horizontal.TProgressbar",
    length=220
).pack(padx=18, pady=(0, 14))

lbl_niveau = tk.Label(
    right,
    text="En attente de l'analyse…",
    font=("Helvetica", 12, "bold"),
    bg=BG_CARD, fg=TEXT_SEC,
    pady=6
)
lbl_niveau.pack()

tk.Frame(right, bg=BORDER, height=1).pack(fill="x", padx=14, pady=8)

lbl_reco = tk.Label(
    right,
    text="",
    font=("Helvetica", 9),
    bg=BG_CARD, fg=TEXT_SEC,
    justify="left",
    wraplength=220,
    anchor="nw",
    padx=14, pady=6
)
lbl_reco.pack(fill="both", expand=True)

# ═══════════════════════════════════════════
#  FOOTER : boutons
# ═══════════════════════════════════════════
tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=20)

footer = tk.Frame(root, bg=BG_DARK, pady=14)
footer.pack()


def make_button(parent, text, cmd, color=ACCENT, hov=BTN_HOV):
    btn = tk.Button(
        parent,
        text=text,
        command=cmd,
        font=("Helvetica", 11, "bold"),
        bg=color, fg=TEXT_PRI,
        activebackground=hov, activeforeground=TEXT_PRI,
        relief="flat", bd=0,
        padx=22, pady=8,
        cursor="hand2"
    )
    btn.pack(side="left", padx=8)
    return btn


make_button(footer, "🚀  Analyser", predire)
make_button(footer, "↺  Réinitialiser", reset_fields, color="#343A52", hov="#454C6A")

root.mainloop()