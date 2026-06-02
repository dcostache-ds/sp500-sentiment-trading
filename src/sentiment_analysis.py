import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.makedirs("data/processed", exist_ok=True)

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 16


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return tokenizer, model, device


def analyze_batch(titles, tokenizer, model, device):
    inputs = tokenizer(titles, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).cpu().tolist()
    labels = ["positive", "negative", "neutral"]

    results = []
    for p in probs:
        idx = p.index(max(p))
        results.append({
            "sentiment":  labels[idx],
            "confidence": p[idx],
            "prob_pos":   p[0],
            "prob_neg":   p[1],
            "prob_neu":   p[2],
        })
    return results


def analyze_news(news_df, tokenizer, model, device):
    print(f"Analyzing {len(news_df)} headlines...")
    titles = news_df["title"].tolist()
    all_results = []

    for i in range(0, len(titles), BATCH_SIZE):
        all_results.extend(analyze_batch(titles[i:i + BATCH_SIZE], tokenizer, model, device))

    result_df = pd.concat([news_df.reset_index(drop=True), pd.DataFrame(all_results)], axis=1)
    result_df.to_csv("data/processed/news_sentiment.csv", index=False)
    return result_df


def daily_scores(df):
    print("Aggregating daily scores...")
    df["net_score"] = df["prob_pos"] - df["prob_neg"]
    rows = []

    for date in sorted(df["date"].unique()):
        day = df[df["date"] == date]
        total_conf = day["confidence"].sum()
        score = (day["net_score"] * day["confidence"]).sum() / total_conf if total_conf > 0 else day["net_score"].mean()

        rows.append({
            "Date":            date,
            "sentiment_score": round(score, 4),
            "article_count":   len(day),
            "pct_positive":    round((day["sentiment"] == "positive").sum() / len(day), 3),
            "pct_negative":    round((day["sentiment"] == "negative").sum() / len(day), 3),
        })

    daily_df = pd.DataFrame(rows)
    daily_df.to_csv("data/processed/sentiment_daily.csv", index=False)
    print(f"Saved scores for {len(daily_df)} days.")
    return daily_df


if __name__ == "__main__":

    if not os.path.exists("data/raw/news.csv"):
        print("Error: data/raw/news.csv not found. Run 1_collect_data.py first.")
        exit()

    news_df = pd.read_csv("data/raw/news.csv")
    tokenizer, model, device = load_model()
    sentiment_df = analyze_news(news_df, tokenizer, model, device)
    daily_df = daily_scores(sentiment_df)
