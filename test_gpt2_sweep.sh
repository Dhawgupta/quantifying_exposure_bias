python decoding_experiments_sweep.py \
   --oracle-model ./gpt2_models/xl/ \
   --eval-model ./gpt2_models/small/ \
   --context-dataset wikitext-2 \
   --context-len 50 \
   --top-ks 5 10 50 100 \
   --top-ps 0.9 0.8 0.6 0.5 \
   --sampling-temperatures 0.5 1 1.2 1.5 --beams 2 5 10 \
   --num-samples 20 --cuda-device 0 \
   --generation-size 256 
