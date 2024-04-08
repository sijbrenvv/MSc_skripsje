import argparse
import json
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
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
        help="Model to use. See choices for the choices. Default: 'NB'",
        choices=["SVC", "SVM", "DT", "RF", "NB"],
        default="NB",
    )
    parser.add_argument(
        "-tr",
        "--train_file",
        type=str,
        help="File containing training data",
    )
    #parser.add_argument(
    #    "-dev",
    #    "--dev_file",
    #    type=str,
    #    help="File containing development data",
    #)
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Location where trained model is saved",
    )
    args = parser.parse_args()
    random_seed = 0
    set_seed(random_seed)

    # Get the train data
    X_train, y_train = get_data(args.train_file)
    #X_dev, y_dev = get_data(args.dev_file)

    classifiers = {
        "SVC": LinearSVC(),
        "SVM": lambda: SVC(kernel="rbf"),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(n_jobs=-1, random_state=random_seed),
        "NB": MultinomialNB(),
    }

    # Get the classic classifier to use
    model = classifiers[args.model]

    # Initialise the vectoriser using Tf-Idf and unigrams
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 1),
        max_df=0.999,
        min_df=0.001,
    )

    # Transform the text to binary code
    vectorizer.fit(X_train)
    vectorizer.transform(X_train)

    # Train and save the model
    model.fit(X_train, y_train)
    model.save(args.output_file)


main()
