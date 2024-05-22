#!/bin/bash

# SBCSAE
python3 create_finetuning_splits.py -inp exp/rule_base/sbcsae/exp_upd_rules/out/syn_data.json -out data/SBCSAE/

# Control
python3 create_finetuning_splits.py -inp exp/rule_base/control/exp_upd_rules/out/syn_data.json -out data/Control/
