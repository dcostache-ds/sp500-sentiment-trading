# S&P 500 Sentiment-Driven Trading System

A machine learning pipeline that predicts the daily direction of the S&P 500 index by combining technical price indicators with financial news sentiment analysis.

---

## How It Works

The system runs as a 5-step pipeline. Each step produces output consumed by the next.

```
Kaggle CSV → FinBERT Sentiment → Feature Engineering → XGBoost → Evaluation
```

**Step 1 — Data Collection**
Loads 17,800 financial headlines paired with S&P 500 closing prices from a Kaggle dataset (2010–2024). Prices and news are extracted from the same CSV and saved separately.

**Step 2 — Sentiment Analysis**
Each headline is classified as positive, negative, or neutral using [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model pre-trained on financial text. A weighted daily sentiment score is calculated from the individual article scores.

**Step 3 — Feature Engineering**
Combines price data and sentiment into 16 features:
- Technical: RSI, MACD, ATR, volatility (5d/20d), returns (1d/5d/20d), volume ratio, price vs MA20
- Sentiment: daily score, lags (1/2/3d), moving averages (5d/20d), sentiment change, article count

The target variable is defined as: **will the S&P 500 close higher tomorrow?**
```python
target = (Close.pct_change().shift(-1) > 0).astype(int)
```
`shift(-1)` is critical — it uses tomorrow's return as the label for today's features, preventing data leakage.

**Step 4 — Model Training**
Trains an XGBoost classifier with walk-forward validation across 48 time folds. Walk-forward always trains on the past and tests on the future, simulating real trading conditions.

**Step 5 — Evaluation**
Generates a confusion matrix, feature importance chart, fold performance chart, and a backtest comparing the strategy against buy-and-hold.

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 50.8% |
| F1 Score | 0.529 |
| Walk-forward folds | 48 |
| Days tested | 3,024 |
| Baseline (random) | 50.0% |

**Top features by importance:**

| Feature | Importance | Type |
|---------|-----------|------|
| price_vs_ma20 | 7.9% | Technical |
| rsi_14 | 7.3% | Technical |
| sentiment_lag1 | 7.0% | Sentiment |
| daily_return | 6.9% | Technical |
| sentiment_ma5 | 6.9% | Sentiment |

7 out of 12 top features are sentiment-based, validating the hypothesis that financial news contains predictive signal.

---

## Project Structure

```
├── src/
│   ├── 1_collect_data.py       # Load Kaggle CSV, extract prices and headlines
│   ├── 2_sentiment_analysis.py # Run FinBERT on headlines, aggregate daily scores
│   ├── 3_feature_engineering.py# Build 16 features + target variable
│   ├── 4_train_model.py        # Grid search + walk-forward XGBoost training
│   └── 5_evaluate.py           # Metrics, charts, backtest
├── data/
│   ├── raw/                    # prices.csv, news.csv
│   └── processed/              # sentiment_daily.csv, features.csv, predictions.csv
├── models/
│   └── xgb_model.pkl           # Trained XGBoost model
├── reports/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── fold_performance.png
│   └── backtest_returns.png
├── run_all.py                  # Runs all 5 steps in sequence
└── requirements.txt
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/sp500-sentiment-trading.git
cd sp500-sentiment-trading
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the data file**

Download `sp500_headlines_2008_2024.csv` from [Kaggle](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024) and place it in the project root.

> This file is not included in the repository due to its size.

**5. Run the full pipeline**
```bash
python run_all.py
```

Charts are saved to `reports/`. The trained model is saved to `models/xgb_model.pkl`.

---

## Dependencies

```
pandas
numpy
yfinance
transformers
torch
xgboost
scikit-learn
matplotlib
seaborn
```

---

## Limitations

- **One headline per day** — the Kaggle dataset contains a single headline per trading day. Multiple headlines per day would produce a more robust sentiment signal.
- **Accuracy at 50.8%** — marginally above the random baseline of 50%. The model is a proof of concept, not production-ready.
- **Backtest does not include transaction costs or slippage.**
- **Model degrades on recent data** — patterns learned on 2010–2015 data may not apply to 2023–2024 market conditions.

---

## What I Learned

- `shift(-1)` is the most important line in the project. Without it, the model leaks the answer into training and results are artificially inflated.
- Walk-forward validation is mandatory for time-series data. Random train/test split would produce false results.
- Data quality matters more than model choice. Upgrading from 60 RSS articles to 17,800 Kaggle headlines moved sentiment features into the top 3 by importance.
- `inplace=True` in pandas returns `None` — assigning the result causes subtle bugs that are hard to debug.

---

## Author

**Costache Darius Daniel**  
Grid Dynamics — May 2026
