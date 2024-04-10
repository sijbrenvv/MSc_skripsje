import argparse
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from transformers import set_seed
import logging
import os
import matplotlib.pyplot as plt

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data(file_path):
    """
    Function to read dataframe with columns.
    Args:
        file_path (str): Path to the file containing the data.
    Returns:
        pd.Series, pd.Series: A series containing the text and a series containing the label columns.
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
    parser.add_argument(
        "--random_seed",
        "-seed",
        #required=True,
        help="The random seed to use.",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    # Set seed for replication
    set_seed(args.random_seed)

    # Define class labels
    logger.info("Defining labels...")
    id2label = {
        0: "Healthy",
        1: "Aphasic",
    }

    # Load test data and predictions
    logger.info("Loading test data and predictions...")
    try:
        X_dev, y_dev = get_data(args.test_data)
        with open(args.prediction_data) as file:
            y_pred = [int(line.strip()) for line in file]
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Evaluate predictions
    logger.info("Evaluating predictions...")
    print(f"Accuracy: {accuracy_score(y_dev, y_pred)}")
    print(
        classification_report(
            y_dev,
            y_pred,
            digits=3,
            target_names=list(id2label.values()),
        )
    )
    print(pd.crosstab(y_dev, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #print(
    #    confusion_matrix(
    #        y_dev,
    #        y_pred,
    #        #labels=list(id2label.values()),
    #        labels=list(id2label.keys()),
    #    )
    #)

    # Save predictions to predefined output file
    #with open(args.output_file, "w") as file:
    #    file.write("\n".join(y_pred))

if __name__ == "__main__":
    main()
