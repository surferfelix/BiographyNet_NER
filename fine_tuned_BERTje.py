"""This model is a finetuned version of the BERTje model from wietsedv for named entity recognition on biographical texts
It should be able to detect PER and LOC ner labels"""
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoConfig, 
                          AutoModelForSequenceClassification, 
                          AutoTokenizer, AdamW, 
                          get_linear_schedule_with_warmup,
                          set_seed, BertTokenizer
                          )
import pandas as pd

class Read_Data_as_df():
    '''Will read the path to the data and process it as a pd dataframe'''
    def __init__(self, data):
        self.data = data
    
    def process():
        return pd.read_csv(self.data)

class BiographyModel(Dataset):
    def __init__(self):
        pass

class FineTuned_Biography_Model():
    """For now doing it all in here"""
    def __init__(self, examples):
        self.n_examples = len(examples)

    def read_in_pandas(path):# TODO: For now we read with pandas, but we should implement this into our main read function
        data = pd.read_csv(path, delimiter = '\t')
        return data

    def prepare_sentences():
        pass

    def Load_Models(dataframe):
        """"""
        #Tokenization and input
        tokenizer = BertTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
        # Tokenize dataset
        max_len = 0

        # For every sentence... #TODO Map sentences to variable
        for sent in sentences:

            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = tokenizer.encode(sent, add_special_tokens=True)

            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))

        print('Max sentence length: ', max_len)
    #     #Check to run from GPU or CPU
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model_name = "GroNLP/bert-base-dutch-cased"
    #     print('Loading configuration...')
    #     model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name) #TODO n_labels
    #     print("Loading tokenizer...")
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     print("Loading model...")
    #     model = AutoModelForSequenceClassification(pretrained_model_name_or_path = model_name, config = model_config)
    #     # Load model to defined device
    #     model.to(device)
    #     print('Model loaded to `%s`'%device)
    #     return model

    # def Load_Dataset():
    #     print('Dealing with Train...')
    #     # Create pytorch dataset.
    #     train_dataset = BiographyModel(path=training_path, 
    #                                 use_tokenizer=tokenizer, 
    #                                 labels_ids=labels_ids,
    #                                 max_sequence_len=max_length)

    #     print('Created `train_dataset` with %d examples!'%len(train_dataset))

    #     # Move pytorch dataset into dataloader.
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

    #     print()

    #     print('Dealing with ...')
    #     # Create pytorch dataset.
    #     valid_dataset =  BiographyModel(path=valid_path, 
    #                                 use_tokenizer=tokenizer, 
    #                                 labels_ids=labels_ids,
    #                                 max_sequence_len=max_length)

    #     print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

    #     # Move pytorch dataset into dataloader.
    #     valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    #     print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))
    