#!/bin/bash
#SBATCH -c 16  # Number of Cores per Task
#SBATCH --mem=64G  # Requested Memory
#SBATCH -p gypsum-rtx8000 # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 0-23:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

python ~/sleep.py
