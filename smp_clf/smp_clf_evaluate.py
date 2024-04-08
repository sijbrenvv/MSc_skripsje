import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    set_seed,
)


def get_data(file_path):
    """Function to read dataframe with columns"""
    #"""Function to read the tsv"""
    #train_df = pd.read_csv(train_path, sep='\t', names=['Source', 'Target'], header=0)

    #train_df, val_df = train_test_split(
    #    train_df,
    #    test_size=0.2,
    #    random_state=random_seed,
    #)

    #return train_df, val_df

    file_df = pd.read_json(file_path, lines=True)
    return file_df["text"], file_path["label"]
    #test_df = pd.read_json(test_path, lines=True)

    #train_df, val_df = train_test_split(
    #    train_df,
    #    test_size=0.2,
    #    #stratify=train_df["label"],
    #    random_state=random_seed,
    #)

    #return train_df, val_df, test_df


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test_data",
        type=str,
        help="File containing test data (gold standards)",
    )
    parser.add_argument(
        "-p",
        "--prediction_data",
        type=str,
        help="File containing model predictions",
    )
    #parser.add_argument(
    #    "-out",
    #    "--output_file",
    #    type=str,
    #    help="Name of output file",
    #)
    args = parser.parse_args()
    random_seed = 0
    set_seed(random_seed)

    # Get the dev data
    _X_dev, y_dev = get_data(args.test_data)

    # Get trained model's predictions
    with open(args.prediction_data) as file:
        y_pred = [line.strip() for line in file]

    # Evaluate predictions
    print(classification_report(y_dev, y_pred, digits=3))
    print(confusion_matrix(y_dev, y_pred))


    # Save predictions to predefined output file
    #with open(args.output_file, "w") as file:
    #    file.write("\n".join(y_pred))


main()
