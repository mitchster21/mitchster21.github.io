# compare_models.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

# ========== Config ==========
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
OUT_DIR = "results"
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)
# ============================

def simulate_games(n_games=2000, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    team_strength = rng.normal(0, 1, size=n_games)
    opp_strength = rng.normal(0, 1, size=n_games)
    strength_diff = team_strength - opp_strength

    home = rng.binomial(1, 0.5, size=n_games)
    rest_diff = rng.normal(0, 1, size=n_games)
    injury = rng.binomial(1, 0.05, size=n_games)
    weather_bad = rng.binomial(1, 0.10, size=n_games)


    lin = (
        0.0 * home          # remove linear home effect so LR loses advantage
        + 0.0 * strength_diff
        + 1.2 * home * strength_diff   # strong interaction: RF can capture
        - 0.7 * rest_diff**2            # bigger nonlinear effect
        + 0.5 * strength_diff**3        # cubic term
        + 0.4 * home * rest_diff        # another interaction
    )

    p_home_win = 1 / (1 + np.exp(-lin))
    y = rng.binomial(1, p_home_win)

    return pd.DataFrame({
        "home": home,
        "strength_diff": strength_diff,
        "rest_diff": rest_diff,
        "injury": injury,
        "weather_bad": weather_bad,
        "home_win": y
    })

def train_and_compare(df):
    X = df.drop(columns="home_win")
    y = df["home_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)

    # Logistic Regression
    scaler = StandardScaler()
    numeric_cols = ["strength_diff", "rest_diff"]
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    lr = LogisticRegression(solver="liblinear", random_state=RANDOM_SEED)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    def get_metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba)
        }

    metrics_lr = get_metrics(y_test, y_pred_lr, y_proba_lr)
    metrics_rf = get_metrics(y_test, y_pred_rf, y_proba_rf)

    results = pd.DataFrame([metrics_lr, metrics_rf],
                           index=["LogisticRegression", "RandomForest"]).round(3)

    # Save CSV + JSON
    os.makedirs(OUT_DIR, exist_ok=True)
    results.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "Logistic Regression": metrics_lr["accuracy"],
            "Random Forest": metrics_rf["accuracy"]
        }, f, indent=2)

    # ROC curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={metrics_lr['roc_auc']:.3f})")
    plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={metrics_rf['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(IMG_DIR, "roc.png")
    plt.savefig(roc_path)
    plt.close()

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Random Forest: Feature Importances")
    plt.tight_layout()
    fi_path = os.path.join(IMG_DIR, "rf_feature_importances.png")
    plt.savefig(fi_path)
    plt.close()

    # Logistic Regression confusion matrix
    ConfusionMatrixDisplay.from_estimator(lr, X_test_scaled, y_test)
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig(os.path.join(IMG_DIR, "confusion_matrix_logreg.png"))
    plt.close()

    # Random Forest confusion matrix
    ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
    plt.title("Random Forest Confusion Matrix")
    plt.savefig(os.path.join(IMG_DIR, "confusion_matrix_rf.png"))
    plt.close()

    # Accuracy comparison bar plot
    plt.figure(figsize=(6,4))
    plt.bar(["LogReg","RF"], [metrics_lr["accuracy"], metrics_rf["accuracy"]])
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.savefig(os.path.join(IMG_DIR, "accuracy_comparison.png"))
    plt.close()

    print("\n=== Model comparison ===")
    print(results)
    print(f"\nSaved results CSV → {os.path.join(OUT_DIR,'model_comparison.csv')}")
    print(f"Saved ROC plot → {roc_path}")
    print(f"Saved RF feature importances → {fi_path}")

    return results, roc_path, fi_path

def main():
    df = simulate_games(n_games=2000)
    df.to_csv(os.path.join(OUT_DIR, "simulated_games.csv"), index=False)
    train_and_compare(df)

if __name__ == "__main__":
    main()
