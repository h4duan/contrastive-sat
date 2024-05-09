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



python3 ../src/test.py \
  --task-name 'neurosat-3-10-ssl-heavy-diffaug' \
  --dim 128 \
  --n_rounds 26 \
  --neurosat \
  --log_dir '../log/ssl'\
  --label_proportion 0.0002 \
  --model_file '../model/power_ssl/net_5592074.pth'\
  --val_folder "../data/power_10_test" \
  --val_instance "test_" \
  --val_label "label.pkl" \
  --val_n_vars "nvars.pkl" 
