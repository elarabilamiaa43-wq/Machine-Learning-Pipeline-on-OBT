# src/regression_model.py

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_regression(X_train, y_train, X_test, y_test):
    """
    Entraîne un Random Forest pour prédire le PRIX (MAD)
    Métriques : MAE, RMSE, R²
    Sauvegarde : models/regression.pkl
    """
    print("\n" + "─" * 45)
    print("📈  RÉGRESSION — Prédiction du Prix (MAD)")
    print("─" * 45)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"  MAE  : {mae:>12,.0f} MAD   ← erreur moyenne")
    print(f"  RMSE : {rmse:>12,.0f} MAD")
    print(f"  R²   : {r2:>12.4f}        ← 1.0 = parfait")

    # Top 5 features importantes
    feat_imp = sorted(
        zip(X_train.columns, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:5]
    print("\n  🔑 Top 5 features importantes :")
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 40)
        print(f"    {feat:<25} {bar}  {imp:.3f}")

    # Sauvegarde
    import os; os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/regression.pkl")
    print("\n  💾 Sauvegardé → models/regression.pkl")
    return model