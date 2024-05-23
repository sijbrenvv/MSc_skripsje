#!/bin/bash

# SBCSAE
## BLEU
python3 analyse_completion.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_fine-tune_bleu.json"
## ChrF
python3 analyse_completion.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_fine-tune_chrf.json"
## Google BLEU
python3 analyse_completion.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_fine-tune_google_bleu.json"