# src/classification_model.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)


def train_classification(X_train, y_train, X_test, y_test):
    """
    Entraîne un Random Forest pour prédire CATEGORIE_PRIX
    Classes : Élevé / Luxe / Moyen
    Métriques : Accuracy, Precision, Recall, F1
    Sauvegarde : models/classification.pkl
    """
    print("\n" + "─" * 45)
    print("🧠  CLASSIFICATION — Catégorie de Prix")
    print("    (Élevé / Luxe / Moyen)")
    print("─" * 45)

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",   # gère le déséquilibre des classes
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc:.4f}\n")

    print("  Rapport complet :")
    print(classification_report(
        y_test, y_pred,
        target_names=["Élevé", "Luxe", "Moyen"]
    ))

    print("  Matrice de confusion :")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {cm}")

    # Sauvegarde
    joblib.dump(clf, "models/classification.pkl")
    print("\n  💾 Sauvegardé → models/classification.pkl")
    return clf