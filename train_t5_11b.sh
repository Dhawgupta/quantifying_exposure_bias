 python run_t5.py     \
    --model_name_or_path t5-11b \
    --dataset_name wikitext \
     --dataset_config_name wikitext-103-raw-v1 \
     --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --block_size 300 \
    --output_dir ./t5_models/11b/ \

