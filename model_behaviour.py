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
from sentence_transformers import SentenceTransformer
#import sister  # Ref: https://github.com/tofunlp/sister
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_function(examples, **fn_kwargs):
    return fn_kwargs["tokenizer"](
        examples[fn_kwargs["col"]],
        padding='longest',
        max_length=fn_kwargs["max_length"],
        truncation=True,
        return_tensors="pt")

def get_data(train_path, random_seed):
    """Function to read the tsv"""
    train_df = pd.read_csv(train_path, sep='\t', names=['Source', 'Target'])

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=random_seed,
    )

    return train_df, val_df


def con_sent_emb(gen_sen, tar_sen):
    """ Function to retrieve the contextualised sentence embeddings
    Input: a list of list of strings
    Output: a list of list of contextualised sentence embeddings
    """
    # Perhaps also explore SIF: https://github.com/PrincetonML/SIF

    # ref: https://huggingface.co/sentence-transformers/sentence-t5-large
    model = SentenceTransformer("sentence-transformers/sentence-t5-large")  # "sentence-transformers/sentence-t5-base"
    #model = SentenceTransformer("all-MiniLM-L6-v2")  # SentenceBERT

    # Generate the contextualised embeddings for the generated and the target completions
    gen_emb = model.encode(gen_sen)
    tar_emb = model.encode(tar_sen)

    return gen_emb, tar_emb


def fastText_sent_emb(gen_sen, tar_sen):
    """ Function to retrieve the static sentence embeddings from FastText"""
    # ref: https://towardsdatascience.com/super-easy-way-to-get-sentence-embedding-using-fasttext-in-python-a70f34ac5b7c
    
    # Initialise the embedder
    embedder = sister.MeanEmbedding(lang="en")

    # Generate the contextualised embeddings for the generated and the target completions
    gen_emb = embedder(gen_sen)
    tar_emb = embedder(tar_sen)

    return gen_emb, tar_emb


def compute_metrics(eval_pred):
    """ """
    # ref: https://colab.research.google.com/drive/1RFBIkTZEqbRt0jxpTHgRudYJBZTD3Szn?usp=sharing#scrollTo=qYq_4DWjdXYa
    metric = evaluate.load('bleu')
    predictions, labels = eval_pred

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode generated completions (predictions) and target completions (labels)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute score
    results = {}
    results.update(
        metric.compute(predictions=decoded_preds, references=decoded_labels)['blue']
    )

    return results

def zero_shot(val_df, model_path, random_seed):
    """ Simple test function to experiment with HF models in a zero shot setting"""

    # Pandas dataframe to huggingface Dataset
    test_dataset = Dataset.from_pandas(val_df)

    # Get tokeniser and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, load_in_8bit=True)
    else:
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
    #tokens = tokenizer(['Complete this utterance into a grammatical sentence. Remove stop words and filler words, change verb and noun inflections and the sentence structure appropriately. Add the subject or verbs if necessary: ' + s for s in test_dataset['Source']], padding=True, return_tensors="pt")

    # Tokenise without prefix c.q. any instructions and generate completions
    tokens = tokenizer(test_dataset['Source'], padding=True, return_tensors="pt")
    output = model.generate(**tokens, max_new_tokens=50)

    #print(f"Completed sentences: {tokenizer.batch_decode(output, skip_special_tokens=True)}")
    # Return the generated completions
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def k_shot(train_df, val_df, model_path, random_seed, k):
    """ Test function to experiment with HF models in a zero shot setting"""
    # ref: https://huggingface.co/docs/transformers/model_doc/t5

    # Pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(val_df)

    # Get tokeniser and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, load_in_8bit=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)
    model.to(device)

    # the following 2 hyperparameters are task-specific. Set to max values
    max_source_length = 512
    max_target_length = 512

    train_examples = train_dataset.shuffle(seed=random_seed).select(range(k))

    # Encode train inputs
    input_seq = train_examples['Source']
    encoding = tokenizer(
        input_seq,
        padding='longest',
        max_length=max_source_length,
        truncation=True,
        return_tensors='pt'
    )
    # We also pass attention_mask to make sure that padding tokens of the inputs are ignored
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # Encode train outputs
    output_seq = train_examples['Target']
    tar_encoding = tokenizer(
        output_seq,
        padding='longest',
        max_length=max_target_length,
        truncation=True,
        return_tensors='pt'
    )

    labels = tar_encoding.input_ids

    # Replace padding token id's of the labels by -100 so it is ignored by the loss
    labels[labels == tokenizer.pad_token_id] = -100

    # Forward pass
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    print(f"Loss: {loss.item()}")

    # Forward pass
    #new_model = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #print(f"Loss: {new_model.loss.item()}")

    # Tokenise test input and generate completions
    tokens = tokenizer(test_dataset['Source'], padding=True, return_tensors="pt", batched=True)
    output = model.generate(**tokens, max_new_tokens=50)

    # Return the generated completions along with the target completions
    return tokenizer.batch_decode(output, skip_special_tokens=True), test_dataset["Target"]


def fine_tune(train_df, checkpoints_path, model_path):  # Add 'valid_df' as argument when there is a test set
    """"""
    # Pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df.drop(train_df.tail(8).index, inplace=True))
    valid_dataset = Dataset.from_pandas(train_df.tail(8))

    # Get tokeniser and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, load_in_8bit=True)
        batch_size = 16
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        batch_size = 8

    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)
    model.to(device)

    # the following  hyperparameter is task-specific. Set to max value
    max_length = 512
    #max_source_length = 512
    #max_target_length = 512

    # Create an empty tokenized_train_dataset and tokenized_valid_dataset
    # in order to use: tokenized_train_dataset['Source'] = train_dataset.map()

    # Encode train data set
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Source'},
    )
    tokenized_train_dataset = tokenized_train_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Target'},
    )

    # Encode valid data set
    tokenized_valid_dataset = valid_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Source'},
    )
    tokenized_valid_dataset = tokenized_valid_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Target'},
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    # Create Trainer object
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=1e-4,  # 1e-4 or 3e-4 typically work best according to the T5 documentation
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        #predict_with_generate=True,  # We perform a generation task and sue BLEU as eval metric
        #metric_for_best_model="bleu",  # Use bleu score to improve the model, might use another metric
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train (fine-tune) the model
    trainer.train()

    # save best model
    best_model_path = checkpoints_path + "/best/"

    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    trainer.save_model(best_model_path)


def test(test_df, best_model_path):
    """ """
    # Pandas dataframe to huggingface Dataset
    test_dataset = Dataset.from_pandas(test_df)

    # Get tokeniser from saved model and load best model
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path, load_in_8bit=True)
        batch_size = 16
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)
        batch_size = 8

    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)
    model.to(device)

    # the following  hyperparameter is task-specific. Set to max value
    max_length = 512

    # Encode test data set
    #tokenized_test_dataset = test_dataset.map(
    #    tokenize_function,
    #    batched=True,
    #    fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Source'},
    #)
    #tokenized_test_dataset = tokenized_test_dataset.map(
    #    tokenize_function,
    #    batched=True,
    #    fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Target'},
    #)

    tokens = tokenizer(test_dataset['Source'], padding=True, return_tensors="pt", batched=True)
    output = model.generate(**tokens, max_new_tokens=50)

    # Return the generated completions
    return tokenizer.batch_decode(output, skip_special_tokens=True)

    """
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    # Create Trainer object
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Get the generated completions (predictions) and evaluate
    predictions = trainer.predict(tokenized_test_dataset)
"""

def temp_test(model_path):
    """Temporary function to explore the use of HF Pipeline"""
    # Create a complementer using HuggingFace pipeline
    #completer = pipeline("text2text-generation", model=model_path)
    #print(completer('He was have a good time with this.'))
    #print(completer('Complete this utterance: He was have a good time with this.'))
    #print(completer('Complete this utterance into a grammatical sentence: He was have a good time with this.'))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, load_in_8bit=True)
    sequences = ['He was have a good time with this.', 'Complete this utterance: He was have a good time with this.', 'Complete this utterance into a grammatical sentence: He was have a good time with this.']
    tokens = tokenizer(sequences, padding=True, return_tensors="pt")
    #print(f"{tokens = }")
    output = model.generate(**tokens, max_new_tokens=50)
    #print(f" Completed sentences: {tokenizer.batch_decode(output, skip_special_tokens=True)}")
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def evaluate_comp(gen_comp, tar_comp):
    """ Evaluate the predicted completions against the target completions"""
    bleu = evaluate.load("bleu")

    ### Revise extracting of embeddings: extract them all prior to the loop. \
    ### The current way is naive: calls the functions in each iteration
    bleu_scores = []
    con_sent_emb_cs = []
    for c,v in enumerate(gen_comp):
        bleu_scores.append(bleu.compute(predictions=[v], references=[tar_comp[c]])['bleu'])
        con_sent_emb_cs.append(cosine_similarity(con_sent_emb(v,tar_comp[c]))[0][1])

    # Loop over the generated completions and compute the BLEU score against the according reference
    #bleu_scores = [bleu.compute(predictions=[v], references=[tar_comp[c]]) for c,v in enumerate(gen_comp)]
    #print(f"Average BLEU score = {np.average([result['bleu'] for result in bleu_scores])}")

    # Mean scores
    #print(f"Average BLEU score = {np.average(bleu_scores)}")
    #print(f"Average cosine similarity (sT5) = {np.average(con_sent_emb_cs)}")

    return bleu_scores, con_sent_emb_cs


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
    train_df, val_df = get_data(train_path, random_seed)
    #gen_comp_zero = zero_shot(val_df, model_path=model, random_seed=random_seed)
    #gen_comp_k, tar_comp = k_shot(train_df,val_df,model_path=model,random_seed=random_seed,k=1)
    #gen_comp_temp = temp_test(model)

    # Train completion model
    fine_tune( # Add 'valid_df' as argument when there is a test set
        train_df=train_df,
        checkpoints_path=f"{model}/{random_seed}",
        model_path=model
    )

    # Test completion model
    gen_comp_ft = test(test_df=val_df, best_model_path=f"{model}/{random_seed}/best/")

    bleu_sc, cs_t5 = evaluate_comp(gen_comp=gen_comp_ft, tar_comp=val_df['Target'])

    output_df = pd.DataFrame({
        "Source": val_df['Source'].to_list(),
        "Target": tar_comp,
        "Gen_comp": gen_comp_k,
        "Bleu": bleu_sc,
        "Cos_sim_t5": cs_t5
    })

    # Export dataframe
    #output_df.to_csv('exp/one.tsv', index=False, sep='\t')
