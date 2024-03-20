import argparse
import shutil
import sys
from pathlib import Path
from typing import Any
import nltk
import numpy as np
import spacy
import spacy_stanza
import spacy_udpipe
import stanza
import textstat
import random
import string
from datasets import Dataset, concatenate_datasets, interleave_datasets, load_dataset
from spacy.matcher import Matcher
from spacy.util import filter_spans
from lexicalrichness import LexicalRichness
from nltk import Text
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from spacy.cli.download import download
from stanza import Pipeline
from collections import Counter
from transformers import (
    set_seed
)


def preprocess(dataset):
    """ """


def download_spacy_language_model(model_name, enable=None):
    """ """
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
    """ """
    try:
        spacy_udpipe.download('en')
        model = spacy_udpipe.load('en')
        return model
    except:
        stanza.download('en')
        try:
            model = spacy_stanza.load_pipeline('en')
            return model
        except:
            model = spacy_stanza.load_pipeline("xx", lang='en')
            return model


def symbol_check(text):
    """ This function checks if the provided text contains symbols or not"""
    # string.punctuation: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for symbol in string.punctuation.replace(',', '').replace('.', ''):
        if symbol in text:
            return True
    return False

    #return [True if symbol in text else False for symbol in string.punctuation]


def nb_np_vp(sen):
    """ This function return the number of noun and verb phrases"""
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
    #print(f"{filter_spans(spans)}")
    nb_vp = len(filter_spans(spans))

    return nb_np, nb_vp


def keep_sentences(example, **fn_kwargs):
    """ """

    # Parse the whole text in the example: doc = nlp(<text in the example>)
    # Retrieve all the sentences using .sents
    # For each sentence:
        # Discard the sentence if it is longer than 15 words or if contains "symbols"
        # Extract the noun and verb phrases
        # If the np/vp ratio is greater than 2, discard the sentence with a probability of 80%
        # Else keep it
    # Return the dictionary with the sentences to keep

    #ks_dict = fn_kwargs["ks_dataset"].to_dict()

    doc = fn_kwargs["udpipe"](example["text"])
    #sentences = []
    for sen in doc.sents:
        sen_text = sen.text
        #print(f"{sen_text = }")
        symbol = symbol_check(sen_text)
        #print(f"{symbol = }")
        sen_len = len(sen_text.split())
        # Continue when the sentence is shorter than or equal 15 words and does not contain "symbols"
        if sen_len <= 15 and symbol is False:
            # Extract the number of noun and verb phrases (new function)
            nb_np, nb_vp = nb_np_vp(sen)
            #print(f"{nb_np = }")
            #print(f"{nb_vp = }")
            if nb_vp == 0:
                continue
            if nb_np / nb_vp > 2:
                if random.random() < .80:
                    # Go to the next iteration, in other words, discard sentence
                    continue
            else:
                # Add sentence
                #sentences.append(sen_text)
                #fn_kwargs["ks_dict"].get("text", []).append(sen_text)
                fn_kwargs["ks_dict"]["text"] = sen_text
                # To make sure all sentences are stored together in a list for each example
                # use: fn_kwargs["ks_dict"]["text"].append(sen_text)

                # Datset object cannot handle Span object
                #fn_kwargs["ks_dict"["text_doc"] = sen

    # Update the ks_dict dictionary with sentences for the current example
    #fn_kwargs["ks_dict"]["text"].extend(sentences)
    #fn_kwargs["ks_dataset"].reset_format()
    #print('Dataset format:', fn_kwargs["ks_dataset"].format)

    #print(f"{len(ks_dict)}")
    #print('Kept sentences dataset:', fn_kwargs["ks_dict"])

    return fn_kwargs["ks_dict"]


def make_synthetic(example, **fn_kwargs):
    """ """

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

    doc = fn_kwargs["udpipe"](example["text"])
    temp_syn_sent = " "
    for token in doc:
        pos = token.pos_
        if pos in {}:
            if random.random() < .90:
                continue
            else:
                temp_syn_sent.join(token)
        if pos in {}:
            if random.random() > .50:
                continue
            else:
                temp_syn_sent.join(token)
        if pos in {}:
            temp_syn_sent.join(token.lemma_)
        else:
            temp_syn_sent.join(token)
    print(f"{temp_syn_sent = }")
    exit()
    fn_kwargs["syn_dict"]["synthetic"] = temp_syn_sent
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
    # parser.add_argument(
    #    "--output_file_path",
    #    "-out",
    #    required=True,
    #    help="Path where to save the output file (synthetic data set).",
    #    type=str,
    # )

    args = parser.parse_args()
    random_seed = 0
    set_seed(random_seed)

    identifier = args.huggingface_dataset  # For example, 'allenai/c4'
    #dataset = load_dataset(identifier, "en", streaming=True)
    dataset = load_dataset(identifier)
    print(dataset)

    # We are only interested in the text column
    column_names = dataset["train"].column_names

    # Remove all columns except the first one: the 'text' column
    dataset = dataset["train"].remove_columns(column_names[1:])
    print(dataset)


    # Create empty Dataset object for the sentences to keep
    #ks_dataset = Dataset.from_dict({"text": [], "text_doc": []})
    #ks_dataset = Dataset.from_dict({"text": []})
    ks_dict = {"text": []}
    # Get the udpipe model: nlp = download_spacy_stanza_pipeline()
    nlp = download_spacy_stanza_pipeline()

    # Get all the sentences we want to keep and process further
    updated_dataset = dataset.select(range(10)).map(
        keep_sentences,
        #batched=True,
        fn_kwargs={"ks_dict": ks_dict, "udpipe": nlp},
    )
    print(updated_dataset)

    # Create dictionary for the synthetic aphasic sentence and the original one
    syn_dict = {"synthetic": [], "original": []}
    exit()
    # Create a synthethic aphasic sentence for each kept sentence
    syn_dataset = updated_dataset.map(
        make_synthetic,
        fn_kwargs={"syn_dict": syn_dict, "udpipe": nlp},
    )
    print(syn_dataset)
