from datasets import load_dataset


def get_dataset(name, config):
    dataset = load_dataset(name, config)
    return dataset

def get_wikitext_2(min_tok_len=10, prompt_len=30):
    dataset = get_dataset("wikitext", "wikitext-2-raw-v1")
    dataset = dataset.filter(lambda x: len(x["text"].split()) > min_tok_len)
    dataset = dataset.map(lambda x: {"text": " ".join(x["text"].split()[:prompt_len])})
    return dataset