from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/raw/all_data.csv")
OUTPUT_PATH = Path("data/processed/maide_english.csv")


def main():
    # Load the original MAiDE-up dataset
    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("PREPARING MAiDE-UP ENGLISH DATASET")
    print("=" * 60)

    # Keep English reviews only
    english_df = df[df["Review_Language"] == "English"].copy()

    print(f"\nEnglish reviews found: {len(english_df)}")

    # Combine the available review sections
    english_df["review_text"] = (
        english_df["Upside_Review"].fillna("")
        + " "
        + english_df["Downside_Review"].fillna("")
    ).str.strip()

    # Keep only the columns needed for modeling
    processed_df = english_df[
        ["review_text", "source"]
    ].copy()

    # Rename source to make the target variable clearer
    processed_df = processed_df.rename(
        columns={"source": "label"}
    )

    # Remove rows with no review text
    processed_df = processed_df[
        processed_df["review_text"].str.len() > 0
    ].copy()

    # Create the output directory if it doesn't exist
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the processed dataset
    processed_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nProcessed dataset saved to: {OUTPUT_PATH}")

    print("\nFinal dataset shape:")
    print(processed_df.shape)

    print("\nLabel distribution:")
    print(processed_df["label"].value_counts())

    print("\nLabel percentages:")
    print(
        processed_df["label"].value_counts(
            normalize=True
        ) * 100
    )

    print("\nMissing values:")
    print(processed_df.isnull().sum())

    print("\nFirst 5 processed reviews:")
    print(processed_df.head().to_string())


if __name__ == "__main__":
    main()