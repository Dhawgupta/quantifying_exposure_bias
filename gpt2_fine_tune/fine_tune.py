from model import get_gpt2, run_model_batch, fine_tune, load_model
from get_data import get_dataset
from logger import logger

# dataset = get_dataset()
model, tokenizer = load_model('test_trainer/checkpoint-500')
print(model)

# fine_tune(model, tokenizer,dataset)

