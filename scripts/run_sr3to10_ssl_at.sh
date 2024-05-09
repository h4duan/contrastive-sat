#!/bin/bash
#SBATCH --account=legacy
#SBATCH --qos=legacy
#SBATCH --partition=t4v1,p100
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=neurosat_s3_ssl_heavy_diffaug
#SBATCH --output=neurosat_s3_ssl_heavy_diffaug.out
#SBATCH --error=neurosat_s3_ssl_heavy_diffaug.error
export PATH=/pkgs/anaconda3/bin:$PATH

python3 -m wandb.cli login 49fcdbf0a6500043ff9402a8555c64a7351364d0


python3 ../src/ssl-train.py \
  --task-name 'neurosat-3-10-ssl-heavy-diffaug' \
  --dim 128 \
  --n_rounds 20 \
  --epochs 2000 \
  --num_batch 1 \
  --log_dir '../log/ssl'\
  --model_dir '../model'\
  --min_n 3 \
  --max_n 10 \
  --ssl_tau 0.1 \
  --learning_rate 0.0002 \
  --num_workers 8 \
  --label_proportion 0.01 \
  --batch_size 100 \
  --val_folder "../data/val_sr3to10_100000" \
  --val_instance "sat_instance" \
  --val_label "label.pkl" \
  --val_n_vars "n_vars.pkl" \
  --variable_elimination 0\
  --clause_resolution 0\
  --add_trivial 1 \
  --at_added_literal $1 \
  --at_added_clause $2 \
  --cr_added_resolv 1 \
  --weight_decay 1e-6 \
  --blocked_clause 0 \
