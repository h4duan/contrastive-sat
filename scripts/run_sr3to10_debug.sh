#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --job-name=neurosat_s3
#SBATCH --output=neurosat_s3.out
#SBATCH --error=neurosat_s3.error
export PATH=/pkgs/anaconda3/bin:$PATH

python3 ../src/ssl-train.py \
  --task-name 'neurosat-3-10-ssl' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 50 \
  --n_pairs 1000 \
  --log_dir '../log'\
  --data_dir '../data'\
  --max_nodes_per_batch 12000 \
  --model_dir '../model'\
  --min_n 3 \
  --max_n 10 \
  --ssl_tau 0.1 \
  --ssl_regularizer 10000 \
  --learning_rate 0.00002\
  --val_file 'sr3to10_val.pkl' \
  --print-screen\
