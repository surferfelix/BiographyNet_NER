"""This model is a finetuned version of the BERTje model from wietsedv for named entity recognition on biographical texts"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoConfig, 
                          AutoModelForSequenceClassification, 
                          AutoTokenizer, AdamW, 
                          get_linear_schedule_with_warmup,
                          set_seed,
                          )

def Fine_Tune_Model():
    tokenizer = AutoTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner")
    set_seed = 421
    # Number of epochs
    epochs=4
    batch_size=32
    max_length=128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = 'wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner'