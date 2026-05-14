# src/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error


def plot_regression(y_test, model, X_test):
    """Graphique : prix réels vs prix prédits"""
    y_pred = model.predict(X_test)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color="#3B82F6", edgecolors="white", s=60)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--", lw=2, label="Prédiction parfaite")
    plt.xlabel("Prix réel (MAD)")
    plt.ylabel("Prix prédit (MAD)")
    plt.title(f"Régression — R² = {r2_score(y_test, y_pred):.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/regression_plot.png", dpi=150)
    plt.show()
    print("✅ Graphique sauvegardé → models/regression_plot.png")


def plot_feature_importance(model, X_train, top_n=10):
    """Graphique : importance des features"""
    feat_imp = sorted(
        zip(X_train.columns, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:top_n]

    features, importances = zip(*feat_imp)

    plt.figure(figsize=(8, 5))
    bars = plt.barh(features[::-1], importances[::-1], color="#6366F1")
    plt.xlabel("Importance")
    plt.title("Top Features — Random Forest")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=150)
    plt.show()
    print("✅ Graphique sauvegardé → models/feature_importance.png")# src/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error


def plot_regression(y_test, model, X_test):
    """Graphique : prix réels vs prix prédits"""
    y_pred = model.predict(X_test)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color="#3B82F6", edgecolors="white", s=60)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--", lw=2, label="Prédiction parfaite")
    plt.xlabel("Prix réel (MAD)")
    plt.ylabel("Prix prédit (MAD)")
    plt.title(f"Régression — R² = {r2_score(y_test, y_pred):.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/regression_plot.png", dpi=150)
    plt.show()
    print("✅ Graphique sauvegardé → models/regression_plot.png")


def plot_feature_importance(model, X_train, top_n=10):
    """Graphique : importance des features"""
    feat_imp = sorted(
        zip(X_train.columns, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:top_n]

    features, importances = zip(*feat_imp)

    plt.figure(figsize=(8, 5))
    bars = plt.barh(features[::-1], importances[::-1], color="#6366F1")
    plt.xlabel("Importance")
    plt.title("Top Features — Random Forest")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=150)
    plt.show()
    print("✅ Graphique sauvegardé → models/feature_importance.png")