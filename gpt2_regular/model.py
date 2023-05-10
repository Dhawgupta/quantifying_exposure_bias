from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel
from logger import logger

_model, _tokenizer, _name = (None,)*3


def get_model_tokenizer(model_name, model_class):
    """Gets a tokenizer and a model for the given model name (and class)

    Args:
        model_name (str): Name of the model
        model_class (Object): Class from transformers library with a from_pretrained method to get the model

    Returns:
        tuple: The model and tokenizer as a tuple: (model, tokenizer)
    """
    global _model, _tokenizer, _name
    if _name == model_name:
        logger.info(f"Model and tokenizer already loaded: {model_name}")
        return _model, _tokenizer
    model = model_class.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info(tokenizer)
    logger.info(f"Model and tokenizer loaded: {model_name}")
    _name, _model, _tokenizer = model_name, model, tokenizer
    return model, tokenizer


def run_model(model, tokenizer, dataset, batch_size=10):
    """Runs the model on the given prompts. Runs with batch_size prompts at a time

    Args:
        model (Object): The model to run
        tokenizer (Object): The tokenizer to use for the model
        dataset (list[str]): Dataset (with train set) to run the model on
        batch_size (int, optional): The number of prompts to feed at a time to the model. Defaults to 10.

    Returns:
        list[str]: Results from the model in the order of the prompts
    """
    num_rows = dataset["train"].num_rows
    logger.info(f"Running model on {num_rows} prompts")
    batches = [dataset["train"]['text'][i:i + batch_size]
               for i in range(0, num_rows, batch_size)]
    print(batches[0])
    res = []
    for i, batch in enumerate(batches):
        logger.info(
            f"Running model on batch {i+1}/{len(batches)} with {len(batch)} prompts")
        gen = run_model_batch(model, tokenizer, batch)
        res += gen
    return res


def run_model_batch(model, tokenizer, batch):
    """Runs a single batch with the model

    Args:
        model (Object): The model to run
        tokenizer (Object): The tokenizer to encode the inputs
        batch (list[str]): The strings to feed into the model

    Returns:
        list[str]: The output of the model in the order of prompts in the batch
    """
    encoded = tokenizer.batch_encode_plus(
        batch, return_tensors="pt", padding=True, truncation=True, max_length=50)

    generated = model.generate(**encoded, max_length=256, do_sample=False,
                               num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)

    if 'gpt2' in _name:
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        # decoded = [a[:a.find('\n')].strip() for a in decoded]
    else:
        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=True)[0]

    return decoded


def get_flan(size='base'):
    """Get the Flan model and tokenizer

    Args:
        size (str, optional): The size of the flan model to get. Defaults to 'base'.

    Returns:
        tuple: The model and tokenizer as a tuple: (model, tokenizer)
    """
    logger.info(f"Loading Flan model: {size}")
    return get_model_tokenizer(f"google/flan-t5-{size}", AutoModelForSeq2SeqLM)


def get_gpt2():
    """Get the GPT2 model and tokenizer

    Returns:
        tuple: The model and tokenizer as a tuple: (model, tokenizer)
    """
    logger.info(f"Loading GPT2 model")
    return get_model_tokenizer("gpt2", GPT2LMHeadModel)


if __name__ == '__main__':
    model, tokenizer = get_gpt2()
    texts = ['I want to go to the kitchen',
             'I want to go to the bedroom', 'I want to go to the bathroom']
    res = run_model(model, tokenizer, texts)
    for r in res:
        print(r)
        input()

    # run_model(model, tokenizer, texts)
