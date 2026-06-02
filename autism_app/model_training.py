"""
model_training.py
-----------------
Generates a realistic synthetic ASD screening dataset, trains three ML models
(Logistic Regression, Random Forest, Decision Tree), evaluates them, saves the
best model as autism_model.pkl and produces a comparison chart.

Run:  python model_training.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

RANDOM_STATE = 42
DATASET_PATH = "dataset.csv"
MODEL_PATH = "autism_model.pkl"
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Build a realistic synthetic ASD-screening dataset
# ---------------------------------------------------------------------------
def build_dataset(n: int = 5000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(2, 60, size=n)
    gender = rng.integers(0, 2, size=n)            # 0=female, 1=male
    family_history = rng.integers(0, 2, size=n)
    speech_delay = rng.integers(0, 2, size=n)
    social_difficulty = rng.integers(0, 2, size=n)
    repetitive_behavior = rng.integers(0, 2, size=n)
    eye_contact = rng.integers(0, 2, size=n)        # 1=poor, 0=normal
    learning_difficulty = rng.integers(0, 2, size=n)
    anxiety_level = rng.integers(0, 11, size=n)     # 0-10
    communication_score = rng.integers(0, 11, size=n)  # 0-10 (lower = worse)

    # Enhanced weighted risk score → probability of ASD label
    # Increased weights for stronger features to create better separation
    risk = (
        3.5 * speech_delay
        + 3.2 * social_difficulty
        + 3.0 * repetitive_behavior
        + 2.8 * eye_contact
        + 2.4 * learning_difficulty
        + 1.8 * family_history
        + 0.25 * anxiety_level
        + 0.35 * (10 - communication_score)
        + 0.5 * (gender == 1)        # ASD slightly more diagnosed in males
        - 0.01 * age
    )
    prob = 1 / (1 + np.exp(-(risk - 4.5)))
    label = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "family_history": family_history,
        "speech_delay": speech_delay,
        "social_difficulty": social_difficulty,
        "repetitive_behavior": repetitive_behavior,
        "eye_contact": eye_contact,
        "learning_difficulty": learning_difficulty,
        "anxiety_level": anxiety_level,
        "communication_score": communication_score,
        "label": label,
    })
    return df


# ---------------------------------------------------------------------------
# 2. Train + evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate():
    print("📊 Building dataset...")
    df = build_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"   Saved {DATASET_PATH}  ({len(df)} rows, "
          f"ASD positive rate = {df['label'].mean():.2%})")

    X = df.drop(columns=["label"])
    y = df["label"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            min_samples_split=5, min_samples_leaf=2,
            random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_split=5, min_samples_leaf=2,
            random_state=RANDOM_STATE),
    }

    results = {}
    best_name, best_model, best_acc = None, None, -1.0

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        
        # Cross-validation score for more robust evaluation
        cv_score = cross_val_score(model, X_train_s, y_train, cv=5).mean()

        results[name] = {
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "cm": cm,
            "cv_score": cv_score,
        }

        print(f"\n🤖 {name}")
        print(f"   Accuracy : {acc:.4f}")
        print(f"   CV Score : {cv_score:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall   : {rec:.4f}")
        print(f"   F1 Score : {f1:.4f}")
        print(classification_report(y_test, preds, zero_division=0))

        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, model

    # -----------------------------------------------------------------------
    # 3. Visualisations
    # -----------------------------------------------------------------------
    plot_model_comparison(results)
    plot_confusion_matrix(results[best_name]["cm"], best_name)

    # -----------------------------------------------------------------------
    # 4. Persist the best model bundle
    # -----------------------------------------------------------------------
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "features": feature_names,
        "model_name": best_name,
        "metrics": {k: {kk: float(vv) if not isinstance(vv, np.ndarray) else vv.tolist()
                        for kk, vv in v.items()} for k, v in results.items()},
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n✅ Best model: {best_name} (acc = {best_acc:.4f})")
    print(f"   Saved to {MODEL_PATH}")


def plot_model_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1"]
    names = list(results.keys())
    data = {m: [results[n][m] for n in names] for m in metrics}

    x = np.arange(len(names))
    width = 0.2

    plt.figure(figsize=(9, 5.5))
    for i, m in enumerate(metrics):
        plt.bar(x + i * width, data[m], width, label=m.capitalize())

    plt.xticks(x + 1.5 * width, names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(STATIC_DIR, "model_comparison.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"   📈 Comparison chart -> {out}")


def plot_confusion_matrix(cm, name):
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-ASD", "ASD"],
                yticklabels=["Non-ASD", "ASD"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out = os.path.join(STATIC_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"   📈 Confusion matrix -> {out}")


if __name__ == "__main__":
    train_and_evaluate()
