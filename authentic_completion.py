import argparse
import logging
import os
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
import torch
from sentence_transformers import SentenceTransformer
#import sister  # Ref: https://github.com/tofunlp/sister
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training # prepare_model_for_int8_training

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Set the logging level for the sentence_transformers library to WARNING
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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

def get_data(test_path: str, random_seed: int) -> pd.DataFrame:
    """
    Function to read dataframe with columns.
    Args:
        test_path (str): Path to the file containing the test data.
    Returns:
        pd.Series, pd.Series: A series containing the source text and a series containing the target text.
    """
    test_df = pd.read_json(test_path, lines=True)

    # Rename the columns of the data (train and test) from 'synthetic' and 'preprocessed_text' to 'Source' and 'Target' respectively
    test_df.rename(columns={"synthetic": "Source", "preprocessed_text": "Target"}, inplace=True)

    # We only need the 'Source' and 'Target' columns
    test_df = test_df[["Source", "Target"]]

    return test_df


def test(test_data: pd.DataFrame, best_model_path: str, prefix: str) -> list[str]:
    """ """
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Pandas dataframe to huggingface Dataset
    test_dataset = Dataset.from_pandas(test_data)

    # Get tokeniser from saved model and load best model
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)

    # Define BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        quantization_dtype=torch.float16,
    )

    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path, device_map="auto", quantization_config=quantization_config)  # torch_dtype=torch.float16
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)

    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    #model.to(device)

    # Use gradient checkpointing during inference
    model.gradient_checkpointing_enable()

    if prefix != "":
        # Tokenise using a simple prefix (the same as Misra and colleagues)
        logger.info(f"Adding prefix to test set...")
        test_dataset = test_dataset.map(
            lambda example: {
                "Source": prefix + example["Source"]
            }
        )
    #tokens = tokenizer(['Complete this sentence: ' + s for s in test_dataset['Source']], padding=True, return_tensors="pt")
    tokens = tokenizer(test_dataset['Source'], padding=True, return_tensors="pt")  #.to(device)

    # Clear memory
    del test_dataset, test_data

    # Clear CUDA cache
    torch.cuda.empty_cache()

    #logger.info(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Disable gradient calculation
    #with torch.no_grad():
    output = model.generate(**tokens, max_new_tokens=25)

    # Clear CUDA cache
    #torch.cuda.empty_cache()

    # Print memory summary
    #logger.info(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Clear memory
    del tokens

    # Return the generated completions
    return tokenizer.batch_decode(output, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--random_seed",
        "-seed",
        #required=True,
        help="The random seed to use.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--prefix",
        "-px",
        type=str,
        help="The prefix to use, include colon followed by a space ': '!. Default: ' '",
        default=""
    )

    args = parser.parse_args()

    # Set seed for replication
    set_seed(args.random_seed)

    if not os.path.exists(args.train_file_path):
        raise FileNotFoundError(f"Train file '{args.train_file_path}' not found.")
    if not os.path.exists(args.test_file_path):
        raise FileNotFoundError(f"Test file '{args.test_file_path}' not found.")

    test_path = args.test_file_path  # For example, 'test.json'
    model = args.huggingface_model  # For example, 'google/flan-t5-small'
    pref = args.prefix  # For example, 'Complete this sentence: '

    # Get the data for the train and dev sets
    logger.info(f"Loading the data...")
    test_df = get_data(test_path, args.random_seed)

    # Test completion model
    logger.info(f"Testing completion model...")
    gen_comp_ft = test(test_data=test_df, best_model_path=f"models/{model}/{args.random_seed}/best/", prefix=pref)

    output_df = pd.DataFrame({
        "Source": test_df['Source'].to_list(),
        "Target": test_df['Target'].to_list(),
        "Gen_comp": gen_comp_ft,
        #"Meteor": eval_sc['meteor_sc'],
        #"Bleu": eval_sc['bleu_sc'],
        "ChrF": eval_sc['chrf_sc'],
        "Cos_sim_t5": eval_sc['cs_t5']
    })

    # Export dataframe
    output_dir = args.output_file_path.split("_")[0]
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Exporting dataframe to '{output_dir + '/' + args.output_file_path.split('/')[-1]}.[json|csv]'...")
    output_df.to_csv(output_dir + "/" + args.output_file_path.split("/")[-1] + ".csv", index=False, sep=',')
    output_df.to_json(output_dir + "/" + args.output_file_path.split("/")[-1] + ".json", orient="records", lines=True)
