import argparse
import shutil
import sys
import numpy as np
import spacy
import spacy_stanza
import spacy_udpipe
import stanza
import random
import string
from datasets import Dataset, concatenate_datasets, interleave_datasets, load_dataset
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.cli.download import download
from stanza import Pipeline
from collections import Counter
from transformers import set_seed
import logging
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess(dataset):
    """ """


def download_spacy_language_model(model_name, enable=None):
    """
     Download and load a Spacy language model.

    Args:
        model_name (str): Name of the Spacy language model to download.
        enable (list): List of pipeline components to enable.

    Returns:
        spacy.Language: Loaded Spacy language model.
    """
    # The spacy language model is not needed at the moment (for the sentences to keep)
    # Add the enable parameter: ref: https://github.com/Darwinkel/shared-task-semeval2024/blob/main/get_features.py
    if enable:
        try:
            return spacy.load(model_name, enable=enable)
        except:
            try:
                download(model_name)
                return spacy.load(model_name, enable=enable)
            except:
                return None
    try:
        return spacy.load(model_name)
    except:
        try:
            download(model_name)
            return spacy.load(model_name)
        except:
            return None


def download_spacy_stanza_pipeline():
    """
     Download and load a Spacy-Stanza pipeline for English.

    Returns:
        spacy_stanza.StanzaLanguage: Loaded Spacy-Stanza pipeline.
    """
    try:
        spacy_udpipe.download('en')
        model = spacy_udpipe.load('en')
        return model
    except Exception as e:
        stanza.download('en')
        try:
            model = spacy_stanza.load_pipeline('en')
            return model
        except Exception as e:
            model = spacy_stanza.load_pipeline("xx", lang='en')
            return model


def symbol_check(text):
    """
    Check if the provided text contains symbols.

    Args:
        text (str): Input text to check.

    Returns:
        boolean: True if text contains symbols, False otherwise.
    """
    # string.punctuation: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return any(symbol in text for symbol in string.punctuation.replace(',', '').replace('.', ''))
    #for symbol in string.punctuation.replace(',', '').replace('.', ''):
    #    if symbol in text:
    #        return True
    #return False

    #return [True if symbol in text else False for symbol in string.punctuation]


def nb_np_vp(sen):
    """ This function return the number of noun and verb phrases
    Count the number of noun and verb phrases in a SpaCy Doc.

    Args:
        doc (spacy.Doc): Input SpaCy document.

    Returns:
        integer: Number of noun phrases and verb phrases.
    """
    nb_np = len([chunk for chunk in sen.noun_chunks])

    # Create a matcher to get the verb phrases
    # 'As of spaCy v3.0, Matcher.add takes a list of patterns as the second argument instead of a variable number of arguments.'
    patterns = [{'POS': 'VERB', 'OP': '?'},
               {'POS': 'ADV', 'OP': '*'},
               {'POS': 'AUX', 'OP': '*'},
               {'POS': 'VERB', 'OP': '+'}]
    # Instantiate the Matcher instance
    matcher = Matcher(nlp.vocab)
    # Add pattern to matcher
    matcher.add("verb-phrases", [patterns])
    # call the matcher to find matches
    matches = matcher(sen)
    spans = [sen[start:end] for _, start, end in matches]
    #logger.info(f"{filter_spans(spans)}")
    nb_vp = len(filter_spans(spans))

    return nb_np, nb_vp


def keep_sentences(example, **fn_kwargs):
    """
    Extract sentences from the example text and filter based on length, symbols, and noun-verb phrase ratio.

    Args:
        example (DatasetDict): Huggingface Dataset object containing the example.
        fn_kwargs (dict): Additional keyword arguments, including 'udpipe' for the language model.

    Returns:
        string: String containing the sentence to keep.
    """

    # For each sentence:
        # Discard the sentence if it is longer than 15 words or if contains "symbols"
        # Extract the noun and verb phrases
        # If the np/vp ratio is greater than 2, discard the sentence with a probability of 80%
        # Else keep it

    # Parse the text
    doc = fn_kwargs["udpipe"](example["text"])  # If this line crashes, modify docstring (fn_kwargs arg)

    for sen in doc.sents:
        sen_text = sen.text
        #logger.info(f"{sen_text = }")
        symbol = symbol_check(sen_text)
        #logger.info(f"{symbol = }")
        sen_len = len(sen_text.split())
        # Continue when the sentence is shorter than or equal to 15 words and does not contain symbols
        if sen_len <= 15 and symbol is False:
            # Extract the number of noun and verb phrases
            nb_np, nb_vp = nb_np_vp(sen)
            #logger.info(f"{nb_np = }")
            #logger.info(f"{nb_vp = }")
            if nb_vp == 0:
                continue
            if nb_np / nb_vp > 2:
                if random.random() < .80:
                    # Go to the next iteration, in other words, discard sentence
                    continue
            else:
                # Add sentence
                fn_kwargs["ks_dict"]["text"] = sen_text
                # To make sure all sentences are stored together in a list for each example
                # use: fn_kwargs["ks_dict"]["text"].append(sen_text)

                #fn_kwargs["ks_dict"].get("text", []).append(sen_text)
                # Dataset object cannot handle Span object
                #fn_kwargs["ks_dict"["text_doc"] = sen

    #logger.info(f"{len(ks_dict)}")
    #logger.info('Kept sentences dataset:', fn_kwargs["ks_dict"])

    return fn_kwargs["ks_dict"]


def make_synthetic(example, **fn_kwargs):
    """
     Generate a synthetic sentence based on the input example.

    Args:
        example (DatasetDict): Huggingface Dataset object containing the example.
        fn_kwargs (dict): Additional keyword arguments, including 'udpipe' for the language model.

    Returns:
        dict: Dictionary containing the synthetic and original sentences.
    """

    # Parse the example/sentence
    # Get the postags
    # Loop over the tokens in parsed example/sentence:
        # Get the postag for the token
        # If postag is determiner, preposition or copula:
            # Discard the token with a probabilty of 90%
        # If postag is adjective or adverb:
            # Discard the token with a probabilty of 50%
        # If postag is a verb:
            # Lemmatise the token
        # Remain all other postags

    # Parse the text
    doc = fn_kwargs["udpipe"](example["text"]) # If this line crashes, modify docstring (fn_kwargs arg)
    # Initialise an empty string to store the synthetic sentence
    temp_syn_sent = ""

    for token in doc:
        pos = token.pos_
        # Perhaps add the Penn Treebank POS tags
        # Perhaps add the dependencies

        # Removal of subject (noun)? if token.dep_ in {"nsubj"}:
        # How to target copulas? Copulas are often parsed as the root
        ## Look for children in the tree: see if it contains two outgoing links, a subject and an object?
        # Remove particles as well (perhaps a lower drop percentage), as they are function words

        # Apply different probabilities for token removal based on part-of-speech tags
        if pos in {"DET", "ADP", "PART"}:
            if random.random() < .90:
                continue
            else:
                temp_syn_sent += " " + token.text
        elif pos in {"ADJ", "ADV"}:
            if random.random() > .50:
                continue
            else:
                temp_syn_sent += " " + token.text
        elif pos in {"AUX", "VERB"}:
            temp_syn_sent += " " + token.lemma_
        elif pos in {"PUNCT"}:
            temp_syn_sent += token.text
        else:
            #logger.info(f" Token and POS tag: {token.text, pos}")
            temp_syn_sent += " " + token.text

    #logger.info(f"{temp_syn_sent = }")
    #logger.info(f"{temp_syn_sent[1:]}")
    #logger.info(f"Original sentence: {example['text']}")

    # Store the synthetic and original sentences in the output dictionary
    fn_kwargs["syn_dict"]["synthetic"] = temp_syn_sent[1:]  # Ignore the leading space
    fn_kwargs["syn_dict"]["original"] = example["text"]

    return fn_kwargs["syn_dict"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--huggingface_dataset",
        "-hd",
        help="Name of the model on HuggingFace. Default: 'datablations/c4-filter-small'",
        #default="datablations/c4-filter-small",
        default="NeelNanda/c4-10k",
        type=str,
    )
    parser.add_argument(
       "--output_file_path",
       "-out",
       required=True,
       help="Path where to save the output file (synthetic data set).",
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

    args = parser.parse_args()

    # Set seed for replication
    set_seed(args.random_seed)

    logger.info("Loading Hugging Face data set...")
    identifier = args.huggingface_dataset  # For example, 'allenai/c4'
    #dataset = load_dataset(identifier, "en", streaming=True)
    dataset = load_dataset(identifier)
    #logger.info(dataset, file=sys.stderr)

    # We are only interested in the text column
    column_names = dataset["train"].column_names

    # Remove all columns except the first one: the 'text' column
    dataset = dataset["train"].remove_columns(column_names[1:])
    #logger.info(dataset, file=sys.stderr)


    # Create dictionary for the sentences to keep
    ks_dict = {"text": []}
    # Get the udpipe model: nlp = download_spacy_stanza_pipeline()
    logger.info("Loading the udpipe model...")
    nlp = download_spacy_stanza_pipeline()

    # Get all the sentences we want to keep and process further
    logger.info("Retrieving the sentences we want to keep and process further...")
    updated_dataset = dataset.select(range(1000)).map(
        keep_sentences,
        #batched=True,
        fn_kwargs={"ks_dict": ks_dict, "udpipe": nlp},
    )
    #logger.info(updated_dataset, file=sys.stderr)

    # Create dictionary for the synthetic aphasic sentence and the original one
    syn_dict = {"synthetic": [], "original": []}

    # Create a synthetic aphasic sentence for each kept sentence
    logger.info("Generating synthetic aphasic sentences...")
    syn_dataset = updated_dataset.map(
        make_synthetic,
        fn_kwargs={"syn_dict": syn_dict, "udpipe": nlp},
        remove_columns=["text"],
    )
    #logger.info(syn_dataset, file=sys.stderr)
    #logger.info(syn_dataset[:10])

    # Export dataset
    logger.info("Exporting the synthetic data set...")
    os.makedirs(args.output_file_path, exist_ok=True)
    syn_dataset.to_csv(os.path.join(args.output_file_path, "syn_data.csv"), index=False, sep=',')
    syn_dataset.to_json(os.path.join(args.output_file_path, "syn_data.json"), orient="records", lines=True)
    logger.info(f"Data set saved to: {os.path.join(args.output_file_path, 'syn_data.json')}")

