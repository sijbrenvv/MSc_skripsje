import argparse
import json
import pandas as pd
import numpy as np
from transformers import set_seed
import joblib
import logging
import os

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data(file_path):
    """
    Function to read dataframe with columns.
    Args:
        file_path (str): Path to the file containing the data.
    Returns:
        lists: A list containing the text and a list containing the label columns.
    """
    file_df = pd.read_json(file_path, lines=True)
    return file_df["text"], file_df["label"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Name of file containing a trained model",
    )
    parser.add_argument(
        "-inp",
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
    logger.info("Loading input data...")
    X_dev, _y_dev = get_data(args.input_file)

    # Load the trained model
    logger.info("Loading model...")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file '{args.model}' not found.")
    model = joblib.load(args.model)

    # Use the loaded model to predict on dev data
    logger.info("Making predictions...")
    y_pred = model.predict(X_dev)

    # Save predictions to predefined output file
    logger.info("Saving predictions to output file...")
    with open(args.output_file, "w") as file:
        file.write("\n".join(map(str, y_pred)))


if __name__ == "__main__":
    main()

