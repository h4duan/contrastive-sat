#!/bin/sh

#PBS -N neurosat_sr10t40
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /home/galen/Haonan/NeuroSAT

python3 src/train.py \
  --task-name 'neurosat_4th_rnd' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 1 \
  --n_pairs 100000 \
  --log-dir 'log'\
  --data-dir 'data'\
  --max_nodes_per_batch 12000 \
  --model-dir 'model'\
  --min_n 10 \
  --max_n 40 \
