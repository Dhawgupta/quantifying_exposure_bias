from model import get_gpt2, run_model_batch, fine_tune, load_model
from get_data import get_dataset
from logger import logger

dataset = get_dataset()
print(dataset)
model, tokenizer = load_model('test_trainer/checkpoint-500')

text = dataset['train']['text']
batch_size = 20
batches = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]

import pandas as pd

df = pd.DataFrame([], columns=['input','output'])

for i,batch in enumerate(batches):
    logger.info(f"Running batch {i+1}/{len(batches)}")
    res = run_model_batch(model, tokenizer, batch)
    res_trimmed = [res.split('\n')[0] for res in res]
    res_pairs = [[batch[i], res_trimmed[i]] for i in range(len(batch))]
    batch_df = pd.DataFrame(res_pairs, columns=['input','output'])
    logger.info(f"Results from batch {i+1}/{len(batches)}:")
    logger.info(batch_df)
    df = pd.concat([df, batch_df], ignore_index=True)
    print(df)
    break

print(df)

df.to_csv('temp.csv')

# results = run_model_batch(model, tokenizer, dataset)
