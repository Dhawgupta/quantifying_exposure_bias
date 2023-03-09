 python run_clm.py     \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
     --dataset_config_name wikitext-103-raw-v1 \
     --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --block_size 256 \
    --output_dir ./oracle256/ 

