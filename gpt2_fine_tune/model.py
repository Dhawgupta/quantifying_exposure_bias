from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from logger import logger
import evaluate
import numpy as np

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

def load_gpt2_tok():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info(tokenizer)
    return tokenizer

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

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return decoded


def get_gpt2():
    """Get the GPT2 model and tokenizer

    Returns:
        tuple: The model and tokenizer as a tuple: (model, tokenizer)
    """
    logger.info(f"Loading GPT2 model")
    return get_model_tokenizer("gpt2", GPT2LMHeadModel)

def load_model(path):
    logger.info(f"Loading model from {path}")
    tokenizer = load_gpt2_tok()
    return GPT2LMHeadModel.from_pretrained(path), tokenizer

def fine_tune(model,tokenizer,dataset):
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    
    # bleu = evaluate.load("bleu")
    # def compute_metrics(pred):
    #     logits, labels = pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return bleu.compute(predictions=predictions, references=labels)
    
    def tok_func(batch):
        res = tokenizer.batch_encode_plus(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=50)
        res['labels'] = res['input_ids']
        return res
    
    dataset = dataset.map(tok_func, batched=True, delete_columns=['text'])
    
    
    
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )
    
    trainer.train()