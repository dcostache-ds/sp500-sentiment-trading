import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

os.makedirs("reports", exist_ok=True)

plt.style.use("dark_background")


def load_data():
    required = [
        "data/processed/predictions.csv",
        "data/processed/walk_forward_results.csv",
        "data/processed/feature_importance.csv",
        "data/processed/features.csv",
        "models/xgb_model.pkl",
        "models/feature_cols.pkl",
    ]
    for path in required:
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            exit()

    predictions = pd.read_csv("data/processed/predictions.csv")
    folds       = pd.read_csv("data/processed/walk_forward_results.csv")
    importance  = pd.read_csv("data/processed/feature_importance.csv")
    features    = pd.read_csv("data/processed/features.csv")
    features["Date"] = pd.to_datetime(features["Date"])

    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    return predictions, folds, importance, features, model, feature_cols


def plot_confusion_matrix(predictions):
    y_true = predictions["y_true"].values
    y_pred = predictions["y_pred"].values
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=["Pred: Down", "Pred: Up"],
                yticklabels=["Real: Down", "Real: Up"],
                ax=ax, annot_kws={"size": 16, "weight": "bold"})

    ax.set_title("Confusion Matrix", color="white", fontsize=14, pad=15)
    ax.tick_params(colors="white")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fig.text(0.5, 0.01, f"Accuracy: {round(acc, 3)}   |   F1: {round(f1, 3)}", ha="center", color="gold", fontsize=12)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print("Saved: reports/confusion_matrix.png")


def plot_feature_importance(importance):
    top = importance.sort_values("importance", ascending=True).tail(12)

    colors = []
    technical_indicators = ["rsi_14", "macd", "atr_14"]
    for feature_name in top["feature"]:
        if "sentiment" in feature_name:
            colors.append("teal")
        elif feature_name in technical_indicators:
            colors.append("gold")
        else:
            colors.append("steelblue")

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.barh(top["feature"], top["importance"], color=colors, height=0.6)

    for bar, val in zip(bars, top["importance"]):
        x_pos = bar.get_width() + 0.001
        y_pos = bar.get_y() + bar.get_height() / 2
        ax.text(x_pos, y_pos, str(round(val, 4)), va="center", ha="left", color="white", fontsize=9)

    ax.set_title("Feature Importance", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="white", alpha=0.1)

    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print("Saved: reports/feature_importance.png")



if __name__ == "__main__":
    predictions, folds, importance, features, model, feature_cols = load_data()
    plot_confusion_matrix(predictions)
    plot_feature_importance(importance)