import argparse
import json
import pandas as pd
import numpy as np
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
        "-m",
        "--model",
        type=str,
        help="Name of file containing a trained model",
    )
    parser.add_argument(
        "-in",
        "--input_file",
        type=str,
        help="Input file containing data",
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Name of output file",
    )
    args = parser.parse_args()
    random_seed = 0
    set_seed(random_seed)

    # Get the dev data
    X_dev, _y_dev = get_data(args.input_file)

    # Get the trained model
    model = args.model_file

    # Use trained model to predict on dev data
    y_pred = model.predict(X_dev)

    # Save predictions to predefined output file
    with open(args.output_file, "w") as file:
        file.write("\n".join(y_pred))


main()
