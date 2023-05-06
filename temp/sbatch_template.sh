#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=40GB  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -t 12:00:00  # Job time limit
#SBATCH -G 1  # Number of GPUs per Task
#SBATCH -o run_%j.out  # %j = job ID

module load miniconda
conda activate 696ds
python main.py