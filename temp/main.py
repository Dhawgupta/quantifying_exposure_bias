from model import get_gpt2, run_model_batch
from get_data import get_wikitext_2

dataset = get_wikitext_2()
model, tokenizer = get_gpt2()

text = dataset['train']['text']
batch_size = 10
batches = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]

import pandas as pd

df = pd.DataFrame([], columns=['input','output'])

for batch in batches:
    res = run_model_batch(model, tokenizer, batch)
    res_trimmed = [res.split('\n')[0] for res in res]
    res_pairs = [[batch[i], res_trimmed[i]] for i in range(len(batch))]
    batch_df = pd.DataFrame(res_pairs, columns=['input','output'])
    df = pd.concat([df, batch_df], ignore_index=True)
    print(df)

print(df)

df.to_csv('temp.csv')

# results = run_model_batch(model, tokenizer, dataset)
