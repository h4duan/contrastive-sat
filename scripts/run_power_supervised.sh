#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=legacy
#SBATCH --qos=legacy
#SBATCH --partition=t4v1,p100
#SBATCH --cpus-per-task=8
#SBATCH --job-name=neurosat_s3_supervised
#SBATCH --output=neurosat_s3_supervised.out
#SBATCH --error=neurosat_s3_supervised.error
export PATH=/pkgs/anaconda3/bin:$PATH

python3 ../src/supervised-train.py\
  --task-name 'sr3to10-supervised-no-aug'\
  --dim 128 \
  --n_rounds 26  \
  --epochs 20000 \
  --log_dir '../log/supervised'\
  --model_dir '../model/sr10to40_supervised'\
  --model_file ''\
  --batch_size 100 \
  --learning_rate 0.00002 \
  --num_workers 8 \
  --data_source "sr" \
  --label_proportion 0.001 \
  --test_epoch 10 \
  --compute_train_accuracy \
  --neurosat \
  --debug \
  --weight_decay 1e-5 \
  --val_folder "../data/sudoku" \
  --val_instance "test_" \
  --val_label "label.pkl" \
  --val_n_vars "nvars.pkl" \

