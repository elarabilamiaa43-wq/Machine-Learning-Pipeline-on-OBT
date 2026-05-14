# pipeline.py
# ─────────────────────────────────────────────────────────
#  Commande pour lancer :  python pipeline.py
# ─────────────────────────────────────────────────────────

from src.extract            import extract_obt, extract_csv
from src.feature_engineering import engineer_features
from src.preprocessing       import prepare_data
from src.regression_model    import train_regression
from src.classification_model import train_classification
from src.evaluate            import plot_regression, plot_feature_importance


# ── Changer ici : True = CSV local | False = PostgreSQL ──
USE_CSV = True


def run():
    print("=" * 50)
    print("🚀  PIPELINE ML AVITO — DÉMARRAGE")
    print("=" * 50)

    # ── ÉTAPE 1 : Extraction ─────────────────────────────
    print("\n[1/5] Extraction des données...")
    df = extract_csv() if USE_CSV else extract_obt()

    # ── ÉTAPE 2 : Feature Engineering ────────────────────
    print("\n[2/5] Feature Engineering...")
    df = engineer_features(df)

    # ── ÉTAPE 3 : Régression — prédire le PRIX ───────────
    print("\n[3/5] Préparation pour la Régression...")
    X_tr, X_te, y_tr, y_te, scaler, _ = prepare_data(
        df, target="prix", task="regression"
    )

    print("\n[4/5] Entraînement du modèle de Régression...")
    reg_model = train_regression(X_tr, y_tr, X_te, y_te)

    # ── ÉTAPE 4 : Classification — prédire CATÉGORIE ─────
    print("\n[5/5] Préparation pour la Classification...")
    X_tr2, X_te2, y_tr2, y_te2, _, le = prepare_data(
        df, target="categorie_prix", task="classification"
    )
    train_classification(X_tr2, y_tr2, X_te2, y_te2)

    # ── ÉTAPE 5 : Graphiques (optionnel) ─────────────────
    print("\n📊 Génération des graphiques...")
    plot_regression(y_te, reg_model, X_te)
    plot_feature_importance(reg_model, X_tr)

    print("\n" + "=" * 50)
    print("✅  PIPELINE TERMINÉ")
    print("   models/regression.pkl       → prédire le prix")
    print("   models/classification.pkl   → prédire la catégorie")
    print("   models/regression_plot.png  → graphique résultats")
    print("=" * 50)


if __name__ == "__main__":
    run()