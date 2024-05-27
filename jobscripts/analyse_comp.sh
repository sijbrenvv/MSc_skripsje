#!/bin/bash

# SBCSAE
## Without prefix
python3 analyse_completion.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_fine-tune_chrf.json" > exp/completion/SBCSAE/flan-t5-xl/analyse_no_pref.txt
## "Complete this sentence: "
python3 analyse_completion.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_pref_fine-tune_chrf.json" > exp/completion/SBCSAE/flan-t5-xl/analyse_cts_txt
