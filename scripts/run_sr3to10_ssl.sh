#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=neurosat_s3_ssl_heavy_diffaug
#SBATCH --output=neurosat_s3_ssl_heavy_diffaug.out
#SBATCH --error=neurosat_s3_ssl_heavy_diffaug.error


python3 ../src/ssl-train.py \
  --task-name 'neurosat-3-10-ssl-heavy-diffaug' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 200000 \
  --num_batch 1 \
  --log_dir '../log/ssl'\
  --model_dir '../model'\
  --min_n 10 \
  --max_n 40 \
  --data_source "sr" \
  --nvar_step 5\
  --learning_rate 0.0006 \
  --num_workers 8 \
  --test_epoch 100 \
  --label_proportion 0.5 \
  --batch_size 32 \
  --val_folder "../data/val_sr40_10000" \
  --val_instance "sat_instance" \
  --val_label "label.pkl" \
  --val_n_vars "n_vars.pkl" \
  --variable_elimination 1\
  --clause_resolution 0\
  --add_trivial 0 \
  --weight_decay 1e-6 \
  --blocked_clause 0 \
  --subsume_clause 1 \
  --at_added_literal 0.2 \
  --at_added_clause 0.1 \
  --cr_added_resolv 0.1 \
  --ve_eliminate_var 0.1 \
  --ve_max_resolvent 0.1 \
  --neurosat \
  --num_ve 0.1 \
  --debug \
  --simclr \
  --ssl \
  --reverse_aug \
  --simclr_tau 0.5\
  --vicreg_lambda 15 \
  --vicreg_mu 1 \
  --vicreg_nu 1 \
  #  --debug \
