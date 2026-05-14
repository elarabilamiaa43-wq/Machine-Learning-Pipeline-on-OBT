# src/regression_model.py

# ─────────────────────────────────────────────────────────────
# Régression - Entraînement du modèle
# ─────────────────────────────────────────────────────────────

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib
import os


def train_regression(X_train, X_test, y_train, y_test):

    # Création du modèle
    model = LinearRegression()

    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("\n📊 Résultats Régression")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"R²  : {r2:.2f}")

    # Création du dossier models
    os.makedirs("models", exist_ok=True)

    # Sauvegarde du modèle
    joblib.dump(model, "models/regression_model.pkl")

    print("\n✅ Modèle sauvegardé : models/regression_model.pkl")

    return model