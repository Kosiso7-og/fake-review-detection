import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_PATH = "data/processed/maide_english_train.csv"

TRANSFORMER_TRAIN_PATH = "data/processed/maide_english_transformer_train.csv"
VALIDATION_PATH = "data/processed/maide_english_validation.csv"

RANDOM_STATE = 42


def main():
    df = pd.read_csv(TRAIN_PATH)

    print("Original training dataset:")
    print(df.shape)
    print(df["label"].value_counts())

    train_df, validation_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    train_df.to_csv(TRANSFORMER_TRAIN_PATH, index=False)
    validation_df.to_csv(VALIDATION_PATH, index=False)

    print("\nTransformer training set:")
    print(train_df.shape)
    print(train_df["label"].value_counts())

    print("\nValidation set:")
    print(validation_df.shape)
    print(validation_df["label"].value_counts())

    print("\nFiles saved successfully.")


if __name__ == "__main__":
    main()