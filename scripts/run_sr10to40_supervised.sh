#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --job-name=neurosat_s10
#SBATCH --output=neurosat_s10.out
#SBATCH --error=neurosat_s10.error
export PATH=/pkgs/anaconda3/bin:$PATH

python3 ../src/supervised-train.py \
  --task-name 'neurosat-10-40-supervised' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 100 \
  --n_pairs 100000 \
  --log-dir '../../neurosat_log_file'\
  --data-dir '../../neurosat_data'\
  --max_nodes_per_batch 12000 \
  --model-dir '../../neurosat_model'\
  --min_n 10 \
  --max_n 40 \
  --learning_rate 0.00002\
