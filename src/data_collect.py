import os
import time
import feedparser
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


today =datetime.today()
start_date= today - timedelta(days=730)

TODAY = today.strftime("%Y-%m-%d")
START_DATE =start_date.strftime("%Y-%m-%d")

os.makedirs("data/raw", exist_ok=True)

NEWS_SOURCES = {
    "Reuters":"https://feeds.reuters.com/reuters/businessNews",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "MarketWatch":"https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "Yahoo Finance":"https://finance.yahoo.com/rss/headline?s=^GSPC",
}


def download_prices():
    raw = yf.download(
        tickers="^GSPC",
        start=START_DATE,
        end=TODAY,
        progress=False,
        auto_adjust=True
    )

    raw= raw.reset_index()
    raw=raw[["Date", "Open", "High", "Low", "Close", "Volume"]]
    raw["Date"]= raw["Date"].dt.strftime("%Y-%m-%d")

    raw.to_csv("data/raw/prices.csv", index=False)

    return raw


def read_one_feed(source_name, url):
    articles = []

    try:
        feed = feedparser.parse(url)

        for entry in feed.entries:
            if not hasattr(entry, "published_parsed") or entry.published_parsed is None:
                continue

            t= entry.published_parsed
            pub_date_str = datetime(t[0], t[1], t[2], t[3], t[4], t[5]).strftime("%Y-%m-%d")

            if pub_date_str< START_DATE or pub_date_str > TODAY:
                continue

            articles.append({
                "date":   pub_date_str,
                "title":  entry.title.strip(),
                "source": source_name
            })

    except Exception as e:
        print(f"Error reading {source_name}: {e}")

    return articles


def download_news():
    all_articles = []

    for name,url in NEWS_SOURCES.items():
        batch = read_one_feed(name, url)
        all_articles.extend(batch)
        time.sleep(0.5)

    news_df= pd.DataFrame(all_articles)

    if news_df.empty:
        print("No articles found!")
        return news_df

    news_df= news_df.drop_duplicates(subset=["title"])
    news_df= news_df.sort_values("date")
    news_df= news_df.reset_index(drop=True)

    news_df.to_csv("data/raw/news.csv", index=False)

    return news_df


if __name__ == "__main__":

    prices = download_prices()
    news   = download_news()
