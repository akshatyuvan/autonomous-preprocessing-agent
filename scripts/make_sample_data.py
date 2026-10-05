"""
scripts/make_sample_data.py — generate data/raw/sample.csv with KNOWN planted issues.

Environment: LOCAL (Mac). Run: python -m scripts.make_sample_data
data/raw/ is gitignored, so this script (not the CSV) is the committed source of truth.
Knowing exactly what was planted is what makes the eyeball check meaningful.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample.csv"


def main() -> None:
    rng = np.random.default_rng(7)  # fixed seed: same file every run
    n = 300
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "age": rng.integers(18, 70, n).astype(float),
        "annual_income": rng.lognormal(10.8, 0.6, n).round(0),
        "gender": rng.choice(["Male", "Female"], n),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "monthly_spend": [f"${v:,}" for v in rng.integers(50, 3000, n)],
    })
    df.loc[rng.choice(n, 45, replace=False), "annual_income"] = np.nan                 # 15% missing
    df.loc[[3, 17], "age"] = [-3.0, 212.0]                                              # impossible ages
    df.loc[rng.choice(n, 12, replace=False), "gender"] = rng.choice(["male", "MALE ", " female", "FEMALE"], 12)
    df.loc[[40, 41, 42], "monthly_spend"] = ["N/A", "?", "unknown"]                     # placeholders + junk
    df = pd.concat([df, df.iloc[[5, 6, 7, 8, 9]]], ignore_index=True)                   # 5 exact duplicates

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols to {OUT}")


if __name__ == "__main__":
    main()