import argparse
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import set_seed
import logging


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

    # Define class labels
    id2label = {
        0: "Healthy",
        1: "Aphasic",
    }

    # Load test data and predictions
    try:
        X_dev, y_dev = get_data(args.test_data)
        with open(args.prediction_data) as file:
            y_pred = [line.strip() for line in file]
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Evaluate predictions
    print(
        classification_report(
            y_dev,
            y_pred,
            digits=3,
            target_names=list(id2label.values()),
        )
    )
    print(
        confusion_matrix(
            y_dev,
            y_pred,
            labels=list(id2label.values()),
        )
    )

    # Save predictions to predefined output file
    #with open(args.output_file, "w") as file:
    #    file.write("\n".join(y_pred))

if __name__ == "__main__":
    main()
