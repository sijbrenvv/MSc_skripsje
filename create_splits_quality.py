import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import set_seed
import logging
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authentic_data",
        "-ad",
        required=True,
        help="Path to the authentic data file.",
        type=str,
    )
    parser.add_argument(
        "--synthetic_data",
        "-sd",
        required=True,
        help="Path to the synthetic data file.",
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
        "--output_file",
        "-out",
        required=True,
        help="Path where to save the data splits.",
        type=str,
    )
    parser.add_argument(
        "--random_seed",
        "-seed",
        #required=True,
        help="The random seed to use.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--train_split",
        "-ts",
        #required=True,
        help="Which data to use as the train split: authentic or synthetic. Default: 'auth'",
        default="auth",
        type=str,
        choices=["auth", "authentic", "syn", "synthetic"],
    )

    args = parser.parse_args()

    # Set seed for replication
    set_seed(args.random_seed)

    #df = pd.read_csv(args.csv_path, sep=',', names=["Source", "Target"], header=0)
    #df.to_csv(args.output_file, sep='\t', encoding='utf-8', index=False)

    # Read the aphasic data and add the label = 1
    if not os.path.exists(args.authentic_data):
        raise FileNotFoundError(f"Authentic data file '{args.authentic_data}' not found.")
    if not os.path.exists(args.synthetic_data):
        raise FileNotFoundError(f"Synthetic data file '{args.synthetic_data}' not found.")

    logger.info("Loading and processing authentic data...")
    authentic_df = pd.read_json(args.authentic_data, lines=True)
    authentic_df["label"] = 1
    # Rename 'preprocessed_text' column to 'text'
    authentic_df.rename(columns={"preprocessed_text": "text"}, inplace=True)
    # We only need the text and label columns for the classifier
    authentic_df = authentic_df[["text", "label"]]
    logger.info(f"Authentic dataframe shape: {authentic_df.shape}")

    synthetic_df = pd.read_json(args.synthetic_data, lines=True)
    synthetic_df["label"] = 1
    # Rename 'synthetic' column name to 'text'
    synthetic_df.rename(columns={"synthetic": "text"}, inplace=True)
    # We only need the text and label columns for the classifier
    synthetic_df = synthetic_df[["text", "label"]]
    logger.info(f"Synthetic dataframe shape: {synthetic_df.shape}")

    # Read the healthy data and add the label = 0
    if not os.path.exists(args.healthy_data):
        raise FileNotFoundError(f"Control data file '{args.healthy_data}' not found.")
    logger.info("Loading and processing control data...")
    healthy_df = pd.read_json(args.healthy_data, lines=True)
    healthy_df["label"] = 0

    # The healthy data is always authentic
    healthy_df.rename(columns={"preprocessed_text": "text"}, inplace=True)

    # We only need the text and label columns for the classifier
    healthy_df = healthy_df[["text", "label"]]
    logger.info(f"Healthy dataframe shape: {healthy_df.shape}")

    # Although this approach is tricky and fragile, the control data is so big it will not intersect

    if args.train_split == "auth" or args.train_split == "authentic":
        len_auth = len(authentic_df.index)
        # Train split of 80%
        len_syn = round(abs(len_auth * 1.25 - len_auth))
        if len(healthy_df) - len_auth - len_syn < 0:
            raise BufferError(f"The control data '{args.healthy_data}' is not big enough. Test data contain examples seen in the train data!!!")
        train_df = pd.concat([authentic_df.head(len_auth), healthy_df.head(len_auth)])
        val_df = pd.concat([synthetic_df.tail(len_syn), healthy_df.tail(len_syn)])

    elif args.train_split == "syn" or args.train_split == "synthetic":
        len_syn = len(synthetic_df.index)
        # Train split of 80%
        len_auth = round(abs(len_syn * 1.25 - len_syn))
        if len(healthy_df) - len_auth - len_syn < 0:
            raise BufferError(f"The control data '{args.healthy_data}' is not big enough. Test data contain examples seen in the train data!!!")
        train_df = pd.concat([synthetic_df.head(len_syn), healthy_df.head(len_syn)])
        val_df = pd.concat([authentic_df.tail(len_auth), healthy_df.tail(len_auth)])
    else:
        raise KeyError(f"Specification of the train split '{args.train_split}' not found.")


    # Output splits to the predefined folder
    logger.info("Outputting splits to predefined folders...")
    os.makedirs(args.output_file, exist_ok=True)
    train_df.to_json(os.path.join(args.output_file, "train.json"), orient="records", lines=True)
    logger.info(f"Train split saved to: {os.path.join(args.output_file, 'train.json')}")
    val_df.to_json(os.path.join(args.output_file, "dev.json"), orient="records", lines=True)
    logger.info(f"Validation split saved to: {os.path.join(args.output_file, 'dev.json')}")
