import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

SHORT = 5
LONG = 20


def load_data():
    if not os.path.exists("data/raw/prices.csv"):
        print("Error: data/raw/prices.csv not found.")
        exit()
    if not os.path.exists("data/processed/sentiment_daily.csv"):
        print("Error: data/processed/sentiment_daily.csv not found.")
        exit()

    prices = pd.read_csv("data/raw/prices.csv")
    sentiment = pd.read_csv("data/processed/sentiment_daily.csv")

    df = pd.merge(prices, sentiment, on="Date", how="left")
    df = df.sort_values("Date")
    df = df.reset_index(drop=True)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    days_with_sentiment = df["sentiment_score"].notna().sum()
    print(f"Loaded {len(df)} trading days, {days_with_sentiment} with sentiment data.")
    return df


def rsi(close, periods=14):
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(periods).mean()
    avg_loss = losses.rolling(periods).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def add_price_features(df):
    close = df["Close"]

    df["daily_return"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)

    df["rsi_14"] = rsi(close)

    df["volatility_5d"] = df["daily_return"].rolling(SHORT).std()
    df["volatility_20d"] = df["daily_return"].rolling(LONG).std()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26

    ma_20 = close.rolling(LONG).mean()
    df["price_vs_ma20"] = close / ma_20

    return df


def add_sentiment_features(df):
    df["sentiment_score"]  = df["sentiment_score"].fillna(0)
    df["article_count"]    = df["article_count"].fillna(0)
    df["sentiment_lag1"]   = df["sentiment_score"].shift(1)
    df["sentiment_lag2"]   = df["sentiment_score"].shift(2)
    df["sentiment_lag3"]   = df["sentiment_score"].shift(3)
    df["sentiment_ma5"]    = df["sentiment_score"].rolling(SHORT).mean()
    df["sentiment_ma20"]   = df["sentiment_score"].rolling(LONG).mean()
    df["sentiment_change"] = df["sentiment_score"].diff()
    return df


def add_target(df):
    df["next_return"] = df["Close"].pct_change().shift(-1)
    df["target"] = (df["next_return"] > 0).astype(int)
    up = df["target"].sum()
    down = len(df) - up - df["target"].isna().sum()
    print(f"Target: {up} up days ({round(up / len(df) * 100, 1)}%), {down} down days.")
    return df


def clean_and_save(df):
    cols_to_drop = ["next_return", "pct_positive", "pct_negative"]

    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.dropna(subset=["target"])

    feature_cols = []
    for col in df.columns:
        if col not in ["Date", "Close", "target"]:
            feature_cols.append(col)

    missing_ratio = df[feature_cols].isna().mean(axis=1)
    df = df[missing_ratio < 0.3].copy()

    df[feature_cols] = df[feature_cols].ffill()
    df = df.reset_index(drop=True)

    df.to_csv("data/processed/features.csv", index=False)
    print(f"Saved {len(df)} rows x {len(feature_cols)} features.")
    return df


if __name__ == "__main__":
    df = load_data()
    df = add_price_features(df)
    df = add_sentiment_features(df)
    df = add_target(df)
    df = clean_and_save(df)
