python decoding_experiments.py \
	   --oracle-model ./gpt2_med_wiki/ \
	      --eval-model ./gpt2_small_wiki/ \
	         --context-dataset wikitext-103 \
		    --context-len 50 \
		       --top-ks 10,50,100,5,500 \
		          --top-ps 0.9,0.8,0.6,0.5 \
			     --sampling-temperatures 0.5 1,1.2,1.5,2 --beams 2,5,10,30
