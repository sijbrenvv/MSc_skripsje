import argparse
import logging
import os
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
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
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_function(examples, **fn_kwargs: dict[str:any]):
    """
    Function to swiftly tokenise the text using the provided tokeniser in fn_kwargs.
    Args:
        examples: All the texts that will be tokenised.
        fn_kwargs (dict): A dictionary of arguments containing the tokeniser, max length value and which column to tokenise.
    Returns:
         Input IDs (list of integers): The tokenised examples.
    """
    return fn_kwargs["tokenizer"](
        examples[fn_kwargs["col"]],
        padding='longest',
        max_length=fn_kwargs["max_length"],
        truncation=True,
        return_tensors="pt")

def get_data(train_path: str, test_path: str, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to read dataframe with columns.
    Args:
        train_path (str): Path to the file containing the train data.
        test_path (str): Path to the file containing the test data
    Returns:
        pd.Series, pd.Series: A series containing the source text and a series containing the target text.
    """
    #train_df = pd.read_csv(train_path, sep='\t', names=['Source', 'Target'], header=0)
    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    # Rename the columns of the data (train and test) from 'synthetic' and 'preprocessed_text' to 'Source' and 'Target' respectively
    train_df.rename(columns={"synthetic": "Source", "preprocessed_text": "Target"}, inplace=True)
    test_df.rename(columns={"synthetic": "Source", "preprocessed_text": "Target"}, inplace=True)

    # We only need the 'Source' and 'Target' columns
    #train_df = train_df[["Source", "Target"]]
    #test_df = test_df[["Source", "Target"]]

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=random_seed,
    )

    return train_df, val_df, test_df


def con_sent_emb(gen_sen: list[str], tar_sen: list[str]) -> tuple:
    """
    Function to retrieve the contextualised sentence embeddings
    Args:
         gen_sen: A list of lists of strings
         tar_sen: A list of lists of strings
    Returns:
        A list of lists of contextual sentence embeddings
    """
    # Perhaps also explore SIF: https://github.com/PrincetonML/SIF

    # ref: https://huggingface.co/sentence-transformers/sentence-t5-large
    senTran = SentenceTransformer("sentence-transformers/sentence-t5-large")  # "sentence-transformers/sentence-t5-base"
    #model = SentenceTransformer("all-MiniLM-L6-v2")  # SentenceBERT

    # Generate the contextualised embeddings for the generated and the target completions
    gen_emb = senTran.encode(gen_sen)
    tar_emb = senTran.encode(tar_sen)

    return gen_emb, tar_emb


def fastText_sent_emb(gen_sen: list[str], tar_sen: list[str]) -> tuple:
    """
    Function to retrieve the static sentence embeddings from FastText
    Args:
        gen_sen: All the generated sentences
        tar_sen: All the target sentences
    Returns:
        gen_emb, tar_emb: The FastText static embeddings for the generated and target completions
    """
    # ref: https://towardsdatascience.com/super-easy-way-to-get-sentence-embedding-using-fasttext-in-python-a70f34ac5b7c
    
    # Initialise the embedder
    embedder = sister.MeanEmbedding(lang="en")

    # Generate the static embeddings for the generated and the target completions
    gen_emb = embedder(gen_sen)
    tar_emb = embedder(tar_sen)

    return gen_emb, tar_emb


def compute_metrics(eval_pred, tokenizer, eval_metric: str) -> dict[str:float]:
    """
    Function to calculate the evaluation metric corresponding to a completion.
    Args:
        eval_pred: The fragmented sentences and their completions.
        tokenizer: The model's tokeniser.
        eval_metric: The evaluation metric to use during fine-tuning.
    Returns:
        results (dict): Dictionary containing the metric scores for each completion.
    """
    # ref: https://colab.research.google.com/drive/1RFBIkTZEqbRt0jxpTHgRudYJBZTD3Szn?usp=sharing#scrollTo=qYq_4DWjdXYa
    ## Different eval metrics and their indexes:
    metric_dic = {
        "bleu": "bleu",
        "chrf": "score",
        "google_bleu": "google_bleu",
        "meteor": "meteor"
    }

    metric = evaluate.load(eval_metric)
    predictions, labels = eval_pred

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode generated completions (predictions) and target completions (labels)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute score
    results = {'metric': 0.0}
    results.update(
        {'metric': metric.compute(predictions=decoded_preds, references=decoded_labels)[metric_dic[eval_metric]]}
    )

    return results


def zero_shot(val_df, model_path, random_seed):
    """ Simple test function to experiment with HF models in a zero shot setting"""

    # Pandas dataframe to huggingface Dataset
    test_dataset = Dataset.from_pandas(val_df)

    # Get tokeniser and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)  #, load_in_8bit=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # Tokenise using an extensive prefix
    #tokens = tokenizer(['Complete this utterance into a grammatical sentence. Remove stop words and filler words, change verb and noun inflections and the sentence structure appropriately. Add the subject or verbs if necessary: ' + s for s in test_dataset['Source']], padding=True, return_tensors="pt")

    # Tokenise using a simple prefix (the same as Misra and colleagues)
    # tokens = tokenizer(['Complete this sentence: ' + s for s in test_dataset['Source']], padding=True, return_tensors="pt")

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
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)  #, load_in_8bit=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
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
    logger.info(f"Loss: {loss.item()}")

    # Tokenise test input and generate completions
    tokens = tokenizer(test_dataset['Source'], padding=True, return_tensors="pt", batched=True)
    output = model.generate(**tokens, max_new_tokens=50)

    # Return the generated completions along with the target completions
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def fine_tune(train_data: pd.DataFrame, valid_data: pd.DataFrame,  checkpoints_path: str, model_path: str, eval_metric: str, random_seed):
    """"""

    # Pandas dataframe to huggingface Dataset
    #train_dataset = Dataset.from_pandas(train_df.iloc[:])
    #valid_dataset = Dataset.from_pandas(train_df.iloc[:])
    train_dataset = Dataset.from_pandas(train_data)
    valid_dataset = Dataset.from_pandas(valid_data)

    # Get tokeniser and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)  #, load_in_8bit=True)
        batch_size = 16
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        batch_size = 8

    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # the following  hyperparameter is task-specific. Set to max value
    max_length = 512
    #max_source_length = 512
    #max_target_length = 512

    # Encode train data set
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Source'},
    )

    train_labels = tokenized_train_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Target'},
    )['input_ids']

    # Add the labels to the train dataset
    tokenized_train_dataset = tokenized_train_dataset.to_dict()
    tokenized_train_dataset["labels"] = train_labels
    tokenized_train_dataset = Dataset.from_dict(tokenized_train_dataset)

    # Encode valid data set
    tokenized_valid_dataset = valid_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Source'},
    )
    valid_labels = tokenized_valid_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Target'},
    )['input_ids']

    # Add the labels to the validation dataset
    tokenized_valid_dataset = tokenized_valid_dataset.to_dict()
    tokenized_valid_dataset["labels"] = valid_labels
    tokenized_valid_dataset = Dataset.from_dict(tokenized_valid_dataset)

    # Create DataCollator object (creates train batches --> speeds up the training process)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    # Create Trainer object
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=1e-4,  # 1e-4 or 3e-4 typically works best according to the T5 documentation
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        predict_with_generate=True,  # We perform a generation task and use BLEU as eval metric
        #metric_for_best_model="meteor",  # Use bleu score to improve the model, might use another metric
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
        compute_metrics=lambda p: compute_metrics(p, tokenizer, eval_metric)  # Pass tokenizer and eval_metric as arguments
    )

    # Train (fine-tune) the model
    trainer.train()

    # Save best model
    best_model_path = checkpoints_path + "/best/"

    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    trainer.save_model(best_model_path)


def test(test_data: pd.DataFrame, best_model_path: str) -> list[str]:
    """ """
    # Pandas dataframe to huggingface Dataset
    test_dataset = Dataset.from_pandas(test_data)

    # Get tokeniser from saved model and load best model
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)  #, load_in_8bit=True)
        batch_size = 16
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)
        batch_size = 8

    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # the following  hyperparameter is task-specific. Set to max value
    max_length = 512

    # Encode test data set
    #tokenized_test_dataset = test_dataset.map(
    #    tokenize_function,
    #    batched=True,
    #    fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "col": 'Source'},
    #)

    tokens = tokenizer(test_dataset['Source'], padding=True, return_tensors="pt")
    output = model.generate(**tokens, max_new_tokens=50)

    # Return the generated completions
    return tokenizer.batch_decode(output, skip_special_tokens=True)

    """
    # Create DataCollator object (creates test batches --> speeds up the testing process)
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


def evaluate_comp(gen_comp: list, tar_comp: list) -> dict[str:list]:
    """
    Evaluate the predicted completions against the target completions
    Args:
        gen_comp (list): All the generated completions.
        tar_comp (list): All the target completions.
    Returns:
        Dictionary: The BLEU, ChrF and Cosine similarity (T5 emb) scores for each sentence pair.
    """
    bleu = evaluate.load("bleu")
    #meteor = evaluate.load("meteor")
    chrf = evaluate.load("chrf")

    ### Revise extracting of embeddings: extract them all prior to the loop. \
    ### The current way is naive: calls the functions in each iteration
    bleu_scores = []
    #meteor_scores = []
    chrf_scores = []
    con_sent_emb_cs = []
    for c, v in enumerate(gen_comp):
        bleu_scores.append(bleu.compute(predictions=[v], references=[tar_comp[c]])['bleu'])
        #meteor_scores.append(meteor.compute(predictions=[v], references=[tar_comp[c]])['meteor'])
        chrf_scores.append(chrf.compute(predictions=[v], references=[tar_comp[c]])['score'])
        con_sent_emb_cs.append(cosine_similarity(con_sent_emb(v,tar_comp[c]))[0][1])

    # Loop over the generated completions and compute the BLEU score against the according reference
    #bleu_scores = [bleu.compute(predictions=[v], references=[tar_comp[c]]) for c,v in enumerate(gen_comp)]
    #print(f"Average BLEU score = {np.average([result['bleu'] for result in bleu_scores])}")

    # Mean scores
    #print(f"Average BLEU score = {np.average(bleu_scores)}")
    #print(f"Average cosine similarity (sT5) = {np.average(con_sent_emb_cs)}")

    return {
        "bleu_sc": bleu_scores,
        "cs_t5": con_sent_emb_cs,
        #"meteor_sc": meteor_scores,
        "chrf_sc": chrf_scores
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file_path",
        "-tr",
        required=True,
        help="Path to the train file.",
        type=str,
    )
    parser.add_argument(
        "--test_file_path",
        "-te",
        required=True,
        help="Path to the test file.",
        type=str,
    )
    parser.add_argument(
        "--output_file_path",
        "-out",
        required=True,
        help="Path where to save the output file.",
        type=str,
    )
    parser.add_argument(
        "--huggingface_model",
        "-hf",
        type=str,
        help="Name of the model on HuggingFace. Default: 'google/flan-t5-small'",
        default="google/flan-t5-small"
    )
    parser.add_argument(
        "--eval_metric",
        "-em",
        type=str,
        help="Name of the evaluation metric to use during training. Default: 'bleu'.",
        choices=["bleu", "chrf", "google_bleu", "meteor"],
        default="bleu"
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

    if not os.path.exists(args.train_file_path):
        raise FileNotFoundError(f"Train file '{args.train_file_path}' not found.")
    if not os.path.exists(args.test_file_path):
        raise FileNotFoundError(f"Test file '{args.test_file_path}' not found.")

    train_path = args.train_file_path  # For example, 'train.json'
    test_path = args.test_file_path  # For example, 'test.json'
    model = args.huggingface_model  # For example, 'google/flan-t5-small'
    eval_metric = args.eval_metric  # For example, 'bleu'

    # Get the data for the train and dev sets
    logger.info(f"Loading the data...")
    train_df, val_df, test_df = get_data(train_path, test_path, args.random_seed)
    #gen_comp_zero = zero_shot(val_df, model_path=model, random_seed=args.random_seed)
    #gen_comp_k = k_shot(train_df,val_df,model_path=model,random_seed=args.random_seed,k=1)

    # Train completion model
    logger.info(f"Training completion model...")
    fine_tune(
        train_data=train_df,
        valid_data=val_df,
        checkpoints_path=f"models/{model}/{args.random_seed}",
        model_path=model,
        eval_metric=eval_metric,
        random_seed=args.random_seed
    )

    # Test completion model
    logger.info(f"Testing completion model...")
    gen_comp_ft = test(test_data=test_df, best_model_path=f"models/{model}/{args.random_seed}/best/")
    eval_sc = evaluate_comp(gen_comp=gen_comp_ft, tar_comp=test_df['Target'].to_list())

    output_df = pd.DataFrame({
        "Source": test_df['Source'].to_list(),
        "Target": test_df['Target'].to_list(),
        "Gen_comp": gen_comp_ft,
        #"Meteor": eval_sc['meteor_sc'],
        "Bleu": eval_sc['bleu_sc'],
        "ChrF": eval_sc['chrf_sc'],
        "Cos_sim_t5": eval_sc['cs_t5']
    })

    # Export dataframe
    os.makedirs(args.output_file_path, exist_ok=True)
    logger.info(f"Exporting dataframe to '{args.output_file_path}.[json|csv]'...")
    output_df.to_csv(args.output_file_path + ".csv", index=False, sep=',')
    output_df.to_json(args.output_file_path + ".json", orient="records", lines=True)
