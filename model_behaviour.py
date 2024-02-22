import argparse
import logging
import os
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch
import sys


def tokenize_function(examples, **fn_kwargs):
    return fn_kwargs["tokenizer"](examples['Source'], padding=True, return_tensors="pt")  # truncation=True

def get_data(train_path, random_seed):
    """Function to read the tsv"""
    train_df = pd.read_csv(train_path, sep='\t', names=['Source', 'Target'])

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=random_seed,
    )

    return train_df, val_df


def preprocess(example):
    return {
        "Source": [f"Complete this utterance: {example['Source']}"],
        "Target": [example['Target']]
    }


def zero_shot(test_df, model_path, random_seed):
    """ Simple test function to experiment with HF models in a zero shot setting"""

    # Pandas dataframe to huggingface Dataset
    test_dataset = Dataset.from_pandas(test_df)

    # Get tokeniser and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)
    model.to(device)

    # Tokenise data for the test set
    #tokenized_test_dataset = test_dataset.map(
    #    tokenize_function,
    #    batched=True,
    #    fn_kwargs={"tokenizer": tokenizer},
    #)

    #output = model.generate(**{'input_ids': torch.as_tensor(tokenized_test_dataset["input_ids"]), 'attention_mask': torch.as_tensor(tokenized_test_dataset["attention_mask"])})

    # Tokenise using an extensive prefix
    tokens = tokenizer(['Complete this utterance into a grammatical sentence. Remove stop words and filler words, change verb and noun inflections and the sentence structure appropriately. Add the subject or verbs if necessary: ' + s for s in test_dataset['Source']], padding=True, return_tensors="pt")

    # Tokenise without prefix c.q. any instructions
    #tokens = tokenizer(test_dataset['Source'], padding=True, return_tensors="pt")
    output = model.generate(**tokens)

    print(f"Completed sentences: {tokenizer.batch_decode(output, skip_special_tokens=True)}")
    return tokenizer.batch_decode(output, skip_special_tokens=True), test_dataset["Target"]


def temp_test(model_path):
    """Temporary function to explore the use of HF Pipeline"""
    # Create a complementer using HuggingFace pipeline
    #completer = pipeline("text2text-generation", model=model_path)
    #print(completer('He was have a good time with this.'))
    #print(completer('Complete this utterance: He was have a good time with this.'))
    #print(completer('Complete this utterance into a grammatical sentence: He was have a good time with this.'))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    sequences = ['He was have a good time with this.', 'Complete this utterance: He was have a good time with this.', 'Complete this utterance into a grammatical sentence: He was have a good time with this.']
    tokens = tokenizer(sequences, padding=True, return_tensors="pt")
    print(f"{tokens = }")
    output = model.generate(**tokens)
    print(f" Completed sentences: {tokenizer.batch_decode(output, skip_special_tokens=True)}")
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def evaluate_comp(gen_comp, tar_comp):
    """ Evaluate the predicted completions against the target completions"""
    bleu = evaluate.load("bleu")

    ### Create one for loop that computes the BLEU scores, gets the embeddings (separate function) \
    ### and computes the cosine similarity.

    # Loop over the generated completions and compute the BLEU score against the according reference
    bleu_scores = [bleu.compute(predictions=[v], references=[tar_comp[c]]) for c,v in enumerate(gen_comp)]
    print(f"Average BLEU score = {np.average([result['bleu'] for result in bleu_scores])}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file_path",
        "-tr",
        required=True,
        help="Path to the train file.",
        type=str,
    )
    #parser.add_argument(
    #    "--prediction_file_path",
    #    "-p",
    #    required=True,
    #    help="Path where to save the prediction file.",
    #    type=str,
    #)
    parser.add_argument(
        "--huggingface_model",
        "-hf",
        type=str,
        help="Name of the model on HuggingFace",
        default="google/flan-t5-small"
    )

    args = parser.parse_args()
    random_seed = 0
    set_seed(random_seed)
    train_path = args.train_file_path  # For example, 'test_samples.tsv'
    #prediction_path = args.prediction_file_path  # For example, 'test_predictions.jsonl'
    model = args.huggingface_model  # For example, 'google/flan-t5-small'

    if not os.path.exists(train_path):
        logging.error(f"File doesnt exists: {train_path}")
        raise ValueError(f"File doesnt exists: {train_path}")

    # Get the data for the train and test sets
    train_df, test_df = get_data(train_path, random_seed)
    gen_comp_zero, tar_comp = zero_shot(test_df, model_path=model, random_seed=random_seed)
    #gen_comp_temp = temp_test(model)

    evaluate_comp(gen_comp=gen_comp_zero, tar_comp=tar_comp)
