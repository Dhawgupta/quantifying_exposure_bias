 python run_clm.py     \
    --model_name_or_path gpt2-xl \
    --dataset_name wikitext \
     --dataset_config_name wikitext-103-raw-v1 \
     --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --fp16 \
    --save_steps 2000 \
    --num_train_epochs 2 \
    --do_train \
    --do_eval \
    --block_size 256 \
    --output_dir ./gpt2_models/xl/ \
    --overwrite_output_dir \

