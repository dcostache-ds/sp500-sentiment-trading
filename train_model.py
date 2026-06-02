import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

os.makedirs("models", exist_ok=True)

SEED = 42
INITIAL_TRAIN = 252
STEP = 63
TEST_SIZE = 63


def load_features():
    if not os.path.exists("data/processed/features.csv"):
        print("Error: features.csv not found.")
        exit()

    df = pd.read_csv("data/processed/features.csv")
    df = df.sort_values("Date")
    df = df.reset_index(drop=True)

    feature_cols = []
    for col in df.columns:
        if col not in ["Date", "Close", "target"]:
            feature_cols.append(col)

    print(f"Loaded {len(df)} rows, {len(feature_cols)} features.")
    return df, feature_cols, "target"


def find_best_params(df, features, target_col):
    cutoff = int(INITIAL_TRAIN * 0.8)
    X_train = df[features].values[:cutoff]
    y_train = df[target_col].values[:cutoff]
    X_val = df[features].values[cutoff:INITIAL_TRAIN]
    y_val = df[target_col].values[cutoff:INITIAL_TRAIN]

    n_down = (y_train == 0).sum()
    n_up = (y_train == 1).sum()
    scale_pw = n_down / max(n_up, 1)

    param_grid = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,  "subsample": 0.8},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1,  "subsample": 1.0},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "subsample": 1.0},
    ]

    best_f1 = -1
    best_params = {}

    for params in param_grid:
        model = XGBClassifier(**params, scale_pos_weight=scale_pw, random_state=SEED, eval_metric="logloss", verbosity=0)
        model.fit(X_train, y_train)
        preds_val = model.predict(X_val)
        f1 = f1_score(y_val, preds_val, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    print(f"Best params: {best_params} (F1={round(best_f1, 3)})")
    return best_params


def walk_forward(df, features, target_col, params):
    print("Running walk-forward validation...")
    n = len(df)
    y_all = df[target_col].values
    n_down = (y_all == 0).sum()
    n_up = (y_all == 1).sum()
    scale_pw = n_down / max(n_up, 1)

    fold_results = []
    all_true = []
    all_pred = []
    train_end = INITIAL_TRAIN
    fold = 1

    while train_end + TEST_SIZE <= n:
        X_train = df[features].values[:train_end]
        y_train = df[target_col].values[:train_end]
        X_test = df[features].values[train_end:train_end + TEST_SIZE]
        y_test = df[target_col].values[train_end:train_end + TEST_SIZE]

        model = XGBClassifier(**params, scale_pos_weight=scale_pw, random_state=SEED, eval_metric="logloss", verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        d_start = df["Date"].iloc[train_end]
        last_test_idx = min(train_end + TEST_SIZE - 1, n - 1)
        d_end = df["Date"].iloc[last_test_idx]

        print(f"  Fold {fold}: train={train_end}d | {d_start} to {d_end} | acc={round(acc, 3)} f1={round(f1, 3)}")

        fold_results.append({
            "fold":       fold,
            "train_days": train_end,
            "date_start": d_start,
            "date_end":   d_end,
            "accuracy":   round(acc, 4),
            "f1":         round(f1, 4)
        })

        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())
        train_end += STEP
        fold += 1

    return fold_results, all_true, all_pred


def train_final_model(df, features, target_col, params):
    print("Training final model on all data...")
    X = df[features].values
    y = df[target_col].values
    n_down = (y == 0).sum()
    n_up = (y == 1).sum()
    scale_pw = n_down / max(n_up, 1)

    model = XGBClassifier(**params, scale_pos_weight=scale_pw, random_state=SEED, eval_metric="logloss", verbosity=0)
    model.fit(X, y)

    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(features, f)

    print("Model saved to models/xgb_model.pkl")
    return model


def print_report(fold_results, y_true, y_pred, model, features):
    folds_df = pd.DataFrame(fold_results)

    mean_acc = round(folds_df["accuracy"].mean(), 3)
    std_acc = round(folds_df["accuracy"].std(), 3)
    mean_f1 = round(folds_df["f1"].mean(), 3)
    std_f1 = round(folds_df["f1"].std(), 3)

    print(f"\nFolds completed: {len(folds_df)}")
    print(f"Mean accuracy:  {mean_acc} (±{std_acc})")
    print(f"Mean F1:        {mean_f1} (±{std_f1})")
    print(classification_report(y_true, y_pred, target_names=["Down (0)", "Up (1)"]))

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature":    features,
        "importance": importances
    })
    importance_df = importance_df.sort_values("importance", ascending=False)

    print("Top 10 features:")
    top10 = importance_df.head(10)
    for _, row in top10.iterrows():
        name = row["feature"]
        score = round(row["importance"], 4)
        print(f"  {name:<35} {score}")

    folds_df.to_csv("data/processed/walk_forward_results.csv", index=False)
    importance_df.to_csv("data/processed/feature_importance.csv", index=False)

    predictions_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })
    predictions_df.to_csv("data/processed/predictions.csv", index=False)


if __name__ == "__main__":
    df, features, target_col = load_features()
    params = find_best_params(df, features, target_col)
    fold_results, y_true, y_pred = walk_forward(df, features, target_col, params)
    model = train_final_model(df, features, target_col, params)
    print_report(fold_results, y_true, y_pred, model, features)
