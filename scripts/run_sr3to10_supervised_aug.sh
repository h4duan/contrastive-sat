#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=legacy
#SBATCH --qos=legacy
#SBATCH --partition=t4v1,p100
#SBATCH --cpus-per-task=8
#SBATCH --job-name=neurosat_s3_supervised_aug
#SBATCH --output=neurosat_s3_supervised_aug.out
#SBATCH --error=neurosat_s3_supervised_aug.error
export PATH=/pkgs/anaconda3/bin:$PATH

python3 ../src/supervised-train-aug.py\
  --task-name 'sr3to10-supervised-easy-aug'\
  --dim 128 \
  --n_rounds 26 \
  --epochs 1000 \
  --log_dir '../log/supervised_aug'\
  --model_dir '../model'\
  --min_n 3 \
  --max_n 10 \
  --num_workers 8\
  --batch_size 100 \
  --learning_rate 0.00002 \
  --label_proportion 0.5 \
  --val_folder "../data/val_sr3to10_100000" \
  --val_instance "sat_instance" \
  --val_label "label.pkl" \
  --val_n_vars "n_vars.pkl" \
  --variable_elimination 0.6\
  --clause_resolution 0.8\
  --add_trivial 0.8 \
  --blocked_clause 0.6 \
  --print-screen \
