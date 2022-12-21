"""This model is a finetuned version of the BERTje model from wietsedv for named entity recognition on biographical texts
It should be able to detect PER and LOC ner labels"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoConfig, 
                          AutoModelForSequenceClassification, 
                          AutoTokenizer, AdamW, 
                          get_linear_schedule_with_warmup,
                          set_seed,
                          )

def Fine_Tune_Model():
    # Getting tokenizer
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
    # Loading model 
    model = AutoModelForSequenceClassification('GroNLP/bert-base-dutch-cased')
    set_seed = 421
    # Number of epochs
    epochs=4
    batch_size=32
    max_length=128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = 'wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner'


class FineTuned_Biography_Model():
    def __init__(self, examples):
        self.n_examples = len(examples)
    
    def prepare():
        """"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = "GroNLP/bert-base-dutch-cased"
        print('Loading configuration...')
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name) #TODO n_labels
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Loading model...")
        # Load model to defined device.
        model.to(device)
        print('Model loaded to `%s`'%device)
        return model

    