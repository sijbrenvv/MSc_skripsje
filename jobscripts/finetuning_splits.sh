#!/bin/bash

# SBCSAE
python3 create_finetuning_splits.py -inp exp/rule_base/sbcsae/exp_test_2405/out/syn_data.json -out data/SBCSAE/

# Control
python3 create_finetuning_splits.py -inp exp/rule_base/control/exp_test_2405/out/syn_data.json -out data/Control/
