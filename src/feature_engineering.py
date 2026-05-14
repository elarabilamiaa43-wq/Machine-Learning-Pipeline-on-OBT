# src/feature_engineering.py

import numpy as np
import pandas as pd


# Colonnes inutiles pour le ML (texte libre, IDs, dates brutes, URLs)
DROP_COLS = ["id", "titre", "lien", "scraped_at", "loaded_at", "surface"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de nouvelles features à partir des colonnes Avito réelles :
    prix, ville, quartier, surface_m2, nb_chambres,
    nb_salles_bain, etage, region_label, is_grande_ville,
    prix_par_m2, categorie_prix
    """
    df = df.copy()

    # ── 1. Log du prix (réduit les valeurs extrêmes) ──────────────────────
    df["log_prix"] = np.log1p(df["prix"])

    # ── 2. Surface par chambre ─────────────────────────────────────────────
    df["surface_par_chambre"] = df["surface_m2"] / (df["nb_chambres"] + 1)

    # ── 3. Score de confort (chambres × salles de bain) ───────────────────
    df["confort_score"] = df["nb_chambres"] * df["nb_salles_bain"]

    # ── 4. Étage élevé ? (> 3) ────────────────────────────────────────────
    df["etage_eleve"] = (df["etage"] > 3).astype(int)

    # ── 5. Variables temporelles depuis scraped_at ────────────────────────
    df["scraped_at"]    = pd.to_datetime(df["scraped_at"])
    df["mois_scraping"] = df["scraped_at"].dt.month
    df["heure_scraping"] = df["scraped_at"].dt.hour

    # ── 6. Supprimer les colonnes inutiles ────────────────────────────────
    df = df.drop(columns=DROP_COLS, errors="ignore")

    print(f"✅ Feature Engineering — {df.shape[1]} colonnes au total")
    print(f"   Colonnes : {df.columns.tolist()}")
    return df