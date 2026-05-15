# src/feature_engineering.py

import numpy as np
import pandas as pd

# Colonnes inutiles pour le ML
DROP_COLS = ["id", "titre", "lien", "scraped_at", "loaded_at", "surface"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── CORRECTION : nettoyer la colonne etage ────────────────────────────
    df["etage"] = pd.to_numeric(df["etage"], errors="coerce")
    df["etage"] = df["etage"].fillna(0).astype(int)

    # ── CORRECTION : nettoyer nb_chambres et nb_salles_bain ──────────────
    df["nb_chambres"]    = pd.to_numeric(df["nb_chambres"],    errors="coerce").fillna(0).astype(int)
    df["nb_salles_bain"] = pd.to_numeric(df["nb_salles_bain"], errors="coerce").fillna(0).astype(int)
    df["surface_m2"]     = pd.to_numeric(df["surface_m2"],     errors="coerce").fillna(0)
    df["prix"]           = pd.to_numeric(df["prix"],           errors="coerce").fillna(0)

    # ── 1. Log du prix ────────────────────────────────────────────────────
    df["log_prix"] = np.log1p(df["prix"])

    # ── 2. Surface par chambre ────────────────────────────────────────────
    df["surface_par_chambre"] = df["surface_m2"] / (df["nb_chambres"] + 1)

    # ── 3. Score de confort ───────────────────────────────────────────────
    df["confort_score"] = df["nb_chambres"] * df["nb_salles_bain"]

    # ── 4. Étage élevé ? (maintenant etage est bien un entier) ───────────
    df["etage_eleve"] = (df["etage"] > 3).astype(int)

    # ── 5. Variables temporelles ──────────────────────────────────────────
    df["scraped_at"]     = pd.to_datetime(df["scraped_at"], errors="coerce")
    df["mois_scraping"]  = df["scraped_at"].dt.month.fillna(0).astype(int)
    df["heure_scraping"] = df["scraped_at"].dt.hour.fillna(0).astype(int)

    # ── 6. Supprimer colonnes inutiles ────────────────────────────────────
    df = df.drop(columns=DROP_COLS, errors="ignore")

    print(f"✅ Feature Engineering — {df.shape[1]} colonnes")
    return df