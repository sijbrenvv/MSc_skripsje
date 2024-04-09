import argparse
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    set_seed,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aphasic_data",
        "-ad",
        required=True,
        help="Path to the aphasic data file.",
        type=str,
    )
    parser.add_argument(
        "--healthy_data",
        "-hd",
        required=True,
        help="Path to the healthy controls data file.",
        type=str,
    )
    parser.add_argument(
        "--output_file_path",
        "-out",
        required=True,
        help="Path where to save the data splits.",
        type=str,
    )

    args = parser.parse_args()

    # Set seed for replication
    random_seed = 0
    set_seed(random_seed)

    #df = pd.read_csv(args.csv_path, sep=',', names=["Source", "Target"], header=0)
    #df.to_csv(args.output_file_path, sep='\t', encoding='utf-8', index=False)

    # Read the aphasic data and add the label = 1
    aphasic_df = pd.read_json(args.aphasic_data, lines=True)
    aphasic_df["label"] = 1
    # If the data is synthetic, rename the column name to 'text'
    if "synthetic" in aphasic_df.columns:
        aphasic_df.rename(columns={"synthetic": "text"}, inplace=True)

    # If the data is authentic, rename the column name to 'text'
    elif "preprocessed_text" in aphasic_df.columns:
        aphasic_df.rename(columns={"preprocessed_text": "text"}, inplace=True)

    # We only need the text and label columns for the classifier
    aphasic_df = aphasic_df[["text", "label"]]
    print(f"Aphasic dataframe shape: {aphasic_df.shape}", file=sys.stderr)

    # Read the healthy data and add the label = 0
    healthy_df = pd.read_json(args.healthy_data, lines=True)
    healthy_df["label"] = 0
    # The healthy data is always authentic
    healthy_df.rename(columns={"preprocessed_text": "text"}, inplace=True)
    # We only need the text and label columns for the classifier
    healthy_df = healthy_df[["text", "label"]]
    print(f"Healthy dataframe shape: {healthy_df.shape}", file=sys.stderr)

    # Concatenate and balance the data \
    # The healthy dataframe is much larger than the aphasic one
    # Get the smallest df size and use that potion of the healthy df
    min_length = min(len(aphasic_df), len(healthy_df))
    data_df = pd.concat([aphasic_df, healthy_df.head(min_length)])
    #data_df = pd.concat([aphasic_df, healthy_df])

    # No shuffle needed as train_test_split will shuffle by default
    train_df, val_df = train_test_split(
            data_df,
            test_size=0.2,
            random_state=random_seed,
        )

    # Output splits to the predefined folder
    os.makedirs(args.output_file_path, exist_ok=True)
    train_df.to_json(os.path.join(args.output_file_path, "train.json"), orient="records", lines=True)
    val_df.to_json(os.path.join(args.output_file_path, "dev.json"), orient="records", lines=True)