#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --job-name=neurosat_s3
#SBATCH --output=neurosat_s3.out
#SBATCH --error=neurosat_s3.error

python3 ../src/generate-val.py \
  --val_num_data 10000 \
  --min_n 60 \
  --max_n 60 \
  --val_folder "../data/val_sr60_10000" \
  --val_instance "sat_instance" \
  --val_label "label.pkl" \
  --val_n_vars "n_vars.pkl"
