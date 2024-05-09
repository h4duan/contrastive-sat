#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --account=deadline
#SBATCH --qos=dedline
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
  --num_workers 1 \
  --data_source "sr" \
  --label_proportion 0.1 \
  --supervised_augment \
  --variable_elimination 1\
  --clause_resolution 0\
  --add_trivial 0 \
  --subsume_clause 1 \
  --at_added_literal 0.2 \
  --at_added_clause 0.1 \
  --cr_added_resolv 0.1 \
  --ve_eliminate_var 0.1 \
  --ve_max_resolvent 0.1 \
  --neurosat \
  --num_ve 0.1 \
  --test_epoch 20 \
  --debug \
  --weight_decay 1e-5 \
  --val_folder "../data/double_power_10000" \
  --val_instance "test_" \
  --val_label "label.pkl" \
  --val_n_vars "nvars.pkl" \

