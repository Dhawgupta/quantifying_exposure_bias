#!/bin/bash
#SBATCH -c 16  # Number of Cores per Task
#SBATCH --mem=64G  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH -G 6  # Number of GPUs
#SBATCH -t 6-23:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
module load miniconda
conda activate cs685
cd /work/pi_bsilva_umass_edu/dgupta_umass_edu/quantifying_exposure_bias/
python run_clm.py     \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
     --dataset_config_name wikitext-103-raw-v1 \
     --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --do_train \
    --do_eval \
    --block_size 256 \
    --output_dir ./oracle256/ \
    --overwrite_output_dir 