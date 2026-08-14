"""Load Titanic dataset once from seaborn and cache to CSV."""

from pathlib import Path

import pandas as pd
import seaborn as sns

OUTPUT = Path(__file__).resolve().parent / "titanic.csv"


def main() -> None:
    df = sns.load_dataset("titanic")
    df.to_csv(OUTPUT, index=False)
    print(f"Cached {len(df)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
