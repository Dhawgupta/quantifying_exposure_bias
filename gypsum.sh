#!/bin/bash
#SBATCH -c 32  # Number of Cores per Task
#SBATCH --mem=96G  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH -G 4  # Number of GPUs
#SBATCH -t 6-23:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

python ~/sleepy.py
