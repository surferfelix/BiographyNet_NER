"""This module will contain the basic BERTje system.
Here we use the BERTje embeddings and then put them into a CRF
model that will make predictions. This will be the baseline BERT system,

Credit for the BERTje model goes to wiestedv"""

from transformers import AutoModelForTokenClassification, BertTokenizer, AutoTokenizer, pipeline
from typing import List, Dict, Tuple
import logging, re
import torch
from collections import Counter

def load_model():
    # Load a trained model and vocabulary that you have fine-tuned
    model = AutoModelForTokenClassification.from_pretrained("wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner")
    tokenizer = AutoTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner")
    print("Loaded model!")
    print(model.config.vocab_size)
    print(tokenizer.vocab_size)
    return model, tokenizer

def map_tokens_to_entities(text: str, entities):
     # Convert the text to a list of tokens
    tokens = text.split()
    
    # Initialize a list of labels with "O" for each token
    labels = ["O" for i in tokens]
    
    # Iterate over the entities
    for entity in entities:
        # Get the start and end indices of the entity
        start = entity['start']
        end = entity['end']
        
        # Get the label for the entity
        label = entity['entity']
        
        # Find the tokens corresponding to the entity
        entity_tokens = text[start:end].split()
        print(entity_tokens)
        
        # Set the labels for the tokens corresponding to the entity
        for index, token in enumerate(tokens): #TODO Create labels list from here instead
            if token in entity_tokens:
                labels.insert(index, label)
    return tokens, labels

def run_baseline_BERTje(s, model, tokenizer):
    ''':s: The sentence to run on
    :type: s: A list of tokens'''
    nlp = pipeline("ner", model=model, tokenizer = tokenizer)
    ner_results = nlp(s)
    tokens, labels = map_tokens_to_entities(s, ner_results)
    return tokens, labels

if __name__ == '__main__':
    # Testing
    sentence = 'Ik heet Dagobert Jansen en ik kom uit Pakistan.'
    res = run_baseline_BERTje(sentence)
    print(res)


# def run_baseline_BERT_aligned(s, label_list = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC', 'O']):
#     toks = []
#     labs = []
    
#     s=s.split()
#     model, tokenizer = load_model()

#     # Tokenize for transformers
#     grouped_inputs = [torch.LongTensor([tokenizer.cls_token_id])]
#     subtokens_per_token = []

#     for token in s:
#         tokens = tokenizer.encode(
#             token,
#             return_tensors="pt",
#             add_special_tokens=False,
#         ).squeeze(axis=0)
#         grouped_inputs.append(tokens)
#         subtokens_per_token.append(len(tokens))
   
#     grouped_inputs.append(torch.LongTensor([tokenizer.sep_token_id]))

#     flattened_inputs = torch.cat(grouped_inputs)
#     flattened_inputs = torch.unsqueeze(flattened_inputs, 0)

#     # Predict
#     predictions_tensor = model(flattened_inputs)[0]
#     predictions_tensor = torch.argmax(predictions_tensor, dim=2)[0]
#     predictions = [label_list[prediction] for prediction in predictions_tensor]

#     # Align tokens
#     # Remove special tokens [CLS] and [SEP]
#     predictions = predictions[1:-1]

#     aligned_predictions = []

#     assert len(predictions) == sum(subtokens_per_token)

#     ptr = 0
#     for size in subtokens_per_token:
#         group = predictions[ptr:ptr + size]
#         assert len(group) == size

#         aligned_predictions.append(group)
#         ptr += size

#     assert len(s) == len(aligned_predictions)

#     for token, prediction_group in zip(s, aligned_predictions):
#         toks.append(token) 
#         lab = Counter(prediction_group).most_common(1)[0][0]
#         labs.append(lab) # We check for most common label in subpieces
#         print(token, lab)
#     return toks, labs