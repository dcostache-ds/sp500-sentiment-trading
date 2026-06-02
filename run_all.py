import subprocess
import sys
import time

STEPS = [
    ("Data collection",     "collect_data.py"),
    ("Sentiment analysis",  "sentiment_analysis.py"),
    ("Feature engineering", "feature_engineering.py"),
    ("Model training",      "train_model.py"),
    ("Evaluation",          "evaluate.py"),
]


for name, script in STEPS:
    print(f"\n {name}")
    start = time.time()
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\nFailed at {script}.")
        sys.exit(1)

