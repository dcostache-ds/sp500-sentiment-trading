import os
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

SOURCE_FILE = "sp500_headlines_2008_2024.csv"
START_DATE = "2010-01-01"


def load_csv():
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found.")
        exit()

    df = pd.read_csv(SOURCE_FILE)
    df = df.rename(columns={"Title": "title", "Date": "date", "CP": "close_price"})

    df = df[df["date"] >= START_DATE]
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} rows from {SOURCE_FILE} (from {START_DATE} onwards).")
    return df


def save_news(df):
    news = df[["date", "title"]].copy()
    news = news.drop_duplicates(subset=["title"])
    news = news.sort_values("date")
    news = news.reset_index(drop=True)
    news.to_csv("data/raw/news.csv", index=False)
    print(f"Saved {len(news)} headlines to data/raw/news.csv.")
    return news


def save_prices(df):
    prices = df[["date", "close_price"]].copy()
    prices = prices.drop_duplicates(subset=["date"])
    prices = prices.rename(columns={"date": "Date", "close_price": "Close"})
    prices = prices.sort_values("Date")
    prices = prices.reset_index(drop=True)
    prices.to_csv("data/raw/prices.csv", index=False)
    print(f"Saved {len(prices)} days of price data to data/raw/prices.csv.")
    return prices


if __name__ == "__main__":
    print("Step 1: Loading data from CSV")
    df = load_csv()
    news = save_news(df)
    prices = save_prices(df)
