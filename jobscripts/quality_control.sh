#!/bin/bash

# Experiment folder is a command line argument
exp_fol="exp/data_quality/$1"

# Check if folder already exists, if so, exit, if not, continue
if [ -d "$exp_fol" ]; then
    echo "Experiment folder already exists. Exiting..."
    exit 1
fi

# Set variables
log=${exp_fol}/log/
out=${exp_fol}/out/
mod=${exp_fol}/mod/
bkp=${exp_fol}/bkp/

# Create folders in exp folder
mkdir -p $exp_fol $log $out $mod $bkp

# Call rule_base.py to create synthetic aphasic data
python3 rule_base.py -inp "data/Control/control_broca_processed.json" -out "${exp_fol}/out/" #> ${log}rb

# Define synthetic data
syn_data="${exp_fol}/out/syn_data.json"
# Define authentic data
auth_data="data/Aphasia/aphasia_broca_processed.json"
# Define control data
control_data="data/Control/control_broca_processed.json"

# Call create_splits_quality.py to create the splits composed of authentic and control data in the train set \
# and synthetic and control data in the dev set.
python3 create_splits_quality.py -ts "syn" -ad $auth_data -sd $syn_data -hd $control_data -out "${exp_fol}/out/" #> ${log}splits
train="${exp_fol}/out/train.json"
dev="${exp_fol}/out/dev.json"

# Train, predict and evaluate
python3 smp_clf/smp_clf_train.py -tr $train -m "NB" -out "${exp_fol}/mod/" #> ${log}train
python3 smp_clf/smp_clf_predict.py -inp $dev -m "${mod}model.pkl" -out "${exp_fol}/out/pred.txt" #> ${log}pred
python3 smp_clf/smp_clf_evaluate.py -t $dev -p "${exp_fol}/out/pred.txt" > ${log}eval.txt

# Backup scripts
cp smp_clf/smp_clf_train.py smp_clf/smp_clf_predict.py smp_clf/smp_clf_evaluate.py rule_base.py create_splits_quality.py $bkp