# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def safe_label_encode(X_train_col, X_test_col):
    """
    LabelEncoder sécurisé :
    Les valeurs inconnues dans X_test (absentes du train)
    sont remplacées par la valeur la plus fréquente du train.
    """
    le = LabelEncoder()
    le.fit(X_train_col.astype(str))

    known = set(le.classes_)
    most_freq = X_train_col.mode()[0]

    # Remplacer les valeurs inconnues dans X_test par most_freq
    X_test_safe = X_test_col.astype(str).apply(
        lambda x: x if x in known else most_freq
    )

    return le.transform(X_train_col.astype(str)), le.transform(X_test_safe), le


def prepare_data(df: pd.DataFrame, target: str, task: str = "regression"):
    df = df.copy()

    # ── Séparer features / cible ──────────────────────────────────────────
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    # ── Split 80 / 20 ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Valeurs manquantes numériques → médiane du train ──────────────────
    for col in num_cols:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col]  = X_test[col].fillna(median)

    # ── Valeurs manquantes texte → mode du train ──────────────────────────
    for col in cat_cols:
        most_freq = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(most_freq)
        X_test[col]  = X_test[col].fillna(most_freq)

    # ── Encodage texte → nombre (avec gestion des inconnues) ─────────────
    for col in cat_cols:
        X_train[col], X_test[col], _ = safe_label_encode(X_train[col], X_test[col])

    # ── Normalisation numérique ───────────────────────────────────────────
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # ── Encodage cible classification ─────────────────────────────────────
    le_target = None
    if task == "classification":
        le_target = LabelEncoder()
        y_train = le_target.fit_transform(y_train)
        y_test  = le_target.transform(y_test)
        print(f"   Classes : {le_target.classes_}")

    print(f"✅ Preprocessing — Train : {X_train.shape} | Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, le_target