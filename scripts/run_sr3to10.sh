#!/bin/sh

#PBS -N neurosat_sr10t40
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /home/galen/Haonan/NeuroSAT

python3 ../src/train.py \
  --task-name 'neurosat-3-10' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 50 \
  --n_pairs 100000 \
  --log-dir '../../neurosat_log_file'\
  --data-dir 'data'\
  --max_nodes_per_batch 12000 \
  --model-dir 'model'\
  --min_n 3 \
  --max_n 10 \
  --learning_rate 0.0002\
  --print-screen \
