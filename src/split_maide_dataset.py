from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("data/processed/maide_english.csv")

TRAIN_PATH = Path("data/processed/maide_english_train.csv")
TEST_PATH = Path("data/processed/maide_english_test.csv")


def main():
    print("=" * 60)
    print("CREATING MAiDE-UP TRAIN/TEST SPLIT")
    print("=" * 60)

    # Load processed dataset
    df = pd.read_csv(INPUT_PATH)

    print(f"\nTotal reviews: {len(df)}")

    # Split into training and testing sets
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=42
    )

    # Create output directory
    TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save datasets
    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print("\nSplit completed.")

    print(f"Training reviews: {len(train_df)}")
    print(f"Testing reviews: {len(test_df)}")

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts())

    print("\nTesting label distribution:")
    print(test_df["label"].value_counts())

    print("\nTraining label percentages:")
    print(
        train_df["label"].value_counts(
            normalize=True
        ) * 100
    )

    print("\nTesting label percentages:")
    print(
        test_df["label"].value_counts(
            normalize=True
        ) * 100
    )

    print("\nFiles created:")
    print(TRAIN_PATH)
    print(TEST_PATH)


if __name__ == "__main__":
    main()