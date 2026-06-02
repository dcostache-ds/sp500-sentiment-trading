import subprocess
import sys
import time

STEPS = [
    ("Data collection",     "src/collect_data.py"),
    ("Sentiment analysis",  "src/sentiment_analysis.py"),
    ("Feature engineering", "src/feature_engineering.py"),
    ("Model training",      "src/train_model.py"),
    ("Evaluation",          "src/evaluate.py"),
]


for name, script in STEPS:
    print(f"\n {name}")
    start = time.time()
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\nFailed at {script}.")
        sys.exit(1)

