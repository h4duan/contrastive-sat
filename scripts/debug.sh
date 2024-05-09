#!/bin/sh

#PBS -N neurosat_sr10t40
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /home/galen/Haonan/NeuroSAT

python3 src/train.py \
  --task-name 'neurosat-3-10-debug' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 1 \
  --n_pairs 10 \
  --log-dir 'log'\
  --data-dir 'data'\
  --max_nodes_per_batch 12000 \
  --model-dir 'model'\
  --min_n 3 \
  --max_n 10 \
