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

# move cached datasets to the /scratch directory
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"

# move downloaded models and tokenizers to the /scratch directory
export HF_HOME="/scratch/$USER/.cache/huggingface/hub"

# Bleu
python3 model_behaviour.py -tr data/synthetic_clan.tsv -out exp/fine-tune_bleu.tsv -hf google/flan-t5-xxl -em bleu

# Meteor
#python3 model_behaviour.py -tr data/synthetic_clan.tsv -out exp/fine-tune_meteor.tsv -hf google/flan-t5-xxl -em meteor

# ChrF
python3 model_behaviour.py -tr data/synthetic_clan.tsv -out exp/fine-tune_chrf.tsv -hf google/flan-t5-xxl -em chrf

# Google Bleu
python3 model_behaviour.py -tr data/synthetic_clan.tsv -out exp/fine-tune_google_bleu.tsv -hf google/flan-t5-xxl -em google_bleu