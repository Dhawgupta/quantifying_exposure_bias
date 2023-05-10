from datasets import Dataset
import pandas as pd

def get_dataset(min_tok_len=10, prompt_len=30):
    df = pd.read_csv('../gpt2_regular/output.csv')
    # df = df.iloc[:10]
    df = df[['input']].rename(columns={'input': 'text'})
    dataset = Dataset.from_pandas(df)
    dataset = dataset.filter(lambda x: len(x["text"].split()) > min_tok_len)
    dataset = dataset.map(lambda x: {"text": " ".join(x["text"].split()[:prompt_len])})
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset

if __name__ == "__main__":
    x = get_dataset()
    print(x)
