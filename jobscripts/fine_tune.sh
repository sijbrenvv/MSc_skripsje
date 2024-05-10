#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem=32000

module purge
module load Python/3.11.3-GCCcore-12.3.0

# Create and load virtual environment
python3 -m venv $HOME/venvs/master_thesis
source $HOME/venvs/master_thesis/bin/activate

# Install dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Move cached datasets to the /scratch directory
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"

# Move downloaded models and tokenizers to the /scratch directory
export HF_HOME="/scratch/$USER/.cache/huggingface/hub"

# The synthetic data source (Control or SBCSAE) is a command line argument
data_source=$1

# Bleu
python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/fine-tune_bleu -hf google/flan-t5-xxl -em bleu

# Meteor
#python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/fine-tune_meteor -hf google/flan-t5-xxl -em meteor

# ChrF
python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/fine-tune_chrf -hf google/flan-t5-xxl -em chrf

# Google Bleu
python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/fine-tune_google_bleu -hf google/flan-t5-xxl -em google_bleu

# Backup scripts
cp fine_tune_t5.py exp/${data_source}/