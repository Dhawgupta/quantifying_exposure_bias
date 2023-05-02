#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gypsum-2080ti # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 2-23:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

python ~/sleep.py
