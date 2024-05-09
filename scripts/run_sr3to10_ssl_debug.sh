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
  --learning_rate 0.0002 \
  --num_workers 8 \
  --label_proportion 0.1 \
  --batch_size 100 \
  --val_folder "../data/val_sr3to10_100000" \
  --val_instance "sat_instance" \
  --val_label "label.pkl" \
  --val_n_vars "n_vars.pkl" \
  --variable_elimination 1\
  --clause_resolution 0\
  --add_trivial 0 \
  --weight_decay 1e-6 \
  --blocked_clause 1 \
  --at_added_literal 0.1 \
  --at_added_clause 0.2 \
  --cr_added_resolv 0.1 \
  --ve_eliminate_var 0.1 \
  --ve_max_resolvent 2 \
  --vicreg \
  --simclr_tau 0.1\
  --vicreg_lambda 25 \
  --vicreg_mu 25 \
  --vicreg_nu 1 \
  --print-screen \
