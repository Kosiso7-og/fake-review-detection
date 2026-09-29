from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/all_data.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nSource distribution:")
    print(df["source"].value_counts())

    print("\nSource percentages:")
    print(df["source"].value_counts(normalize=True) * 100)

    print("\nReview language distribution:")
    print(df["Review_Language"].value_counts())

    print("\nReview language percentages:")
    print(df["Review_Language"].value_counts(normalize=True) * 100)

    english_df = df[df["Review_Language"] == "English"]

    print("\nEnglish-only dataset:")
    print(f"Rows: {len(english_df)}")

    print("\nEnglish source distribution:")
    print(english_df["source"].value_counts())

    print("\nMissing values in review fields:")
    print(
        df[["Upside_Review", "Downside_Review"]].isnull().sum()
    )

    print("\nExample review:")
    print("Upside:")
    print(df.iloc[0]["Upside_Review"])

    print("\nDownside:")
    print(df.iloc[0]["Downside_Review"])

    print("\nSentiment distribution:")
    print(df["Sentiment"].value_counts())

    print("\nSentiment by source:")
    print(
        pd.crosstab(
            df["source"],
            df["Sentiment"],
            normalize="index"
        ) * 100
    )

    print("\nLanguage by source:")
    print(
        pd.crosstab(
            df["Review_Language"],
            df["source"]
        )
    )

    print("\nDuplicate review combinations:")
    review_columns = ["Upside_Review", "Downside_Review"]

    print(
        df.duplicated(subset=review_columns).sum()
    )

    print("\nDetailed duplicate review analysis:")

    duplicate_mask = df.duplicated(
        subset=["Upside_Review", "Downside_Review"],
        keep=False
    )

    duplicates = df[duplicate_mask].sort_values(
        by=["Upside_Review", "Downside_Review"]
    )

    print(
        f"Total rows involved in duplicate combinations: {len(duplicates)}"
    )

    print("\nDuplicate source distribution:")
    print(duplicates["source"].value_counts())

    print("\nDuplicate language distribution:")
    print(duplicates["Review_Language"].value_counts())

    print("\nDuplicate review examples:")
    print(
        duplicates[
            [
                "Review_Language",
                "source",
                "Upside_Review",
                "Downside_Review"
            ]
        ].head(20).to_string()
    )

    print("\nEnglish dataset text analysis:")

    english_df = df[df["Review_Language"] == "English"].copy()

    english_df["combined_review"] = (
        english_df["Upside_Review"].fillna("")
        + " "
        + english_df["Downside_Review"].fillna("")
    ).str.strip()

    print(f"English rows: {len(english_df)}")

    print(
        "English rows with completely missing review text:",
        (english_df["combined_review"] == "").sum()
    )

    print(
        "Duplicate English review texts:",
        english_df["combined_review"].duplicated().sum()
    )

    print("\nEnglish review length statistics:")

    english_df["review_length"] = (
        english_df["combined_review"].str.len()
    )

    print(english_df["review_length"].describe())


if __name__ == "__main__":
    main()