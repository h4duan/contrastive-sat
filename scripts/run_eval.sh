#!/bin/sh

#PBS -N neurosat_sr10t40
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /home/galen/Haonan/NeuroSAT

python3 src/data_maker.py \
 
  --n_pairs 100000 \
  --log-dir 'log'\
  --data-dir 'data'\
  --max_nodes_per_batch 12000 \
  --gen_log 'data_maker_sr10t40.log' \
  --min_n 10 \
  --max_n 40 \
