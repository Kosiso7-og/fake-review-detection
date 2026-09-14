from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/deceptive-reviews.csv")


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Place your dataset CSV inside data/raw/")
        return

    df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)


if __name__ == "__main__":
    main()