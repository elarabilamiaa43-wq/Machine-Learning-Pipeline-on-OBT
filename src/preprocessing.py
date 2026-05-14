# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def prepare_data(df: pd.DataFrame, target: str, task: str = "regression"):
    """
    Prépare les données pour le ML :
      - Sépare X et y
      - Split 80 % train / 20 % test
      - Gère les valeurs manquantes
      - Encode les colonnes texte (ville, quartier, region_label, is_grande_ville)
      - Normalise les colonnes numériques
      - Encode la cible si classification

    Paramètres
    ----------
    df     : DataFrame après feature_engineering
    target : "prix"          → régression
             "categorie_prix" → classification
    task   : "regression" | "classification"
    """
    df = df.copy()

    # ── Séparer features / cible ──────────────────────────────────────────
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    # ── Split 80 / 20 ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Valeurs manquantes ───────────────────────────────────────────────
    num_imp = SimpleImputer(strategy="median")
    cat_imp = SimpleImputer(strategy="most_frequent")

    X_train[num_cols] = num_imp.fit_transform(X_train[num_cols])
    X_test[num_cols]  = num_imp.transform(X_test[num_cols])

    if cat_cols:
        X_train[cat_cols] = cat_imp.fit_transform(X_train[cat_cols])
        X_test[cat_cols]  = cat_imp.transform(X_test[cat_cols])

    # ── Encodage texte → nombre ──────────────────────────────────────────
    # Colonnes texte présentes : ville, quartier, region_label, is_grande_ville
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    # ── Normalisation numérique ──────────────────────────────────────────
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # ── Encodage cible (classification seulement) ────────────────────────
    le_target = None
    if task == "classification":
        le_target = LabelEncoder()           # Élevé=0  Luxe=1  Moyen=2
        y_train = le_target.fit_transform(y_train)
        y_test  = le_target.transform(y_test)
        print(f"   Classes : {le_target.classes_}")

    print(f"✅ Preprocessing — Train : {X_train.shape} | Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, le_target