#!/bin/bash

# SBCSAE
python3 create_finetuning_splits.py -inp exp/rule_base/sbcsae/exp_3word/out/syn_data.json -out data/SBCSAE/

# Control
python3 create_finetuning_splits.py -inp exp/rule_base/control/exp_3word/out/syn_data.json -out data/Control/
