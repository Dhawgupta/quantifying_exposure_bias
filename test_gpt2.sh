python decoding_experiments.py \
   --oracle-model ./gpt2_models/small/ \
   --eval-model ./gpt2_models/small/ \
   --context-dataset wikitext-2 \
   --context-len 50 \
   --top-ks 10 \
   --top-ps 0.9 \
   --sampling-temperatures 0.5 --beams 2 \
   --num-samples 10 --cuda-device 0 \
   --generation-size 256 
