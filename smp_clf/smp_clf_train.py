import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from transformers import set_seed


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
    """
    Main function to train and save the classifier model.
    """
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

    classifiers = {
        "SVC": LinearSVC(),
        "SVM": SVC(kernel="rbf"),
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
    X_train = vectorizer.fit_transform(X_train)

    # Train and save the model
    model.fit(X_train, y_train)
    joblib.dump(model, args.output_file)


if __name__ == "__main__":
    main()

