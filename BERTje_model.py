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
    tokenizer = BertTokenizer.from_pretrained("../saved_models/bert-base-dutch-cased")
    print("Loaded model!")
    print(model.config.vocab_size)
    print(tokenizer.vocab_size)
    return model, tokenizer

def run_baseline_BERT_aligned(s, label_list = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC', 'O']):
    toks = []
    labs = []
    
    s=s.split()
    model, tokenizer = load_model()

    # Tokenize for transformers
    grouped_inputs = [torch.LongTensor([tokenizer.cls_token_id])]
    subtokens_per_token = []

    for token in s:
        tokens = tokenizer.encode(
            token,
            return_tensors="pt",
            add_special_tokens=False,
        ).squeeze(axis=0)
        grouped_inputs.append(tokens)
        subtokens_per_token.append(len(tokens))
   
    grouped_inputs.append(torch.LongTensor([tokenizer.sep_token_id]))

    flattened_inputs = torch.cat(grouped_inputs)
    flattened_inputs = torch.unsqueeze(flattened_inputs, 0)

    # Predict
    predictions_tensor = model(flattened_inputs)[0]
    predictions_tensor = torch.argmax(predictions_tensor, dim=2)[0]
    predictions = [label_list[prediction] for prediction in predictions_tensor]

    # Align tokens
    # Remove special tokens [CLS] and [SEP]
    predictions = predictions[1:-1]

    aligned_predictions = []

    assert len(predictions) == sum(subtokens_per_token)

    ptr = 0
    for size in subtokens_per_token:
        group = predictions[ptr:ptr + size]
        assert len(group) == size

        aligned_predictions.append(group)
        ptr += size

    assert len(s) == len(aligned_predictions)

    for token, prediction_group in zip(s, aligned_predictions):
        toks.append(token) 
        lab = Counter(prediction_group).most_common(1)[0][0]
        labs.append(lab) # We check for most common label in subpieces
        print(token, lab)
    return toks, labs

def run_baseline_BERTje(s):
    ''':s: The sentence to run on
    :type: s: A list of tokens'''
    remove = ['[CLS]', '[SEP]', '[UNK]', '[PAD]']
    tokens = []
    labs = []
    text = s
    model, tokenizer = load_model()
    encoding = tokenizer(text, return_tensors = 'pt', truncation = True, max_length = 512)
    outputs = model(**encoding)
    logits = outputs.logits
    # print(logits.shape)
    predicted_label_classes = logits.argmax(-1)
    # print(predicted_label_classes)
    predicted_labels = [model.config.id2label[id] for id in predicted_label_classes.squeeze().tolist()]
    # print(predicted_labels)
    for id, label in zip(encoding.input_ids.squeeze().tolist(), predicted_labels):
        token = tokenizer.decode([id])
        if token in remove:
            continue
        else:
            tokens.append(token)
            labs.append(label)
    return tokens, labs

def wordpieces_to_tokens(wordpieces: List, labelpieces: List = None) -> Tuple[List, List]:
    textpieces = " ".join(wordpieces)
    full_words = re.sub(r'\s##', '', textpieces).split()
    full_labels = []
    if labelpieces:
        for ix, wp in enumerate(wordpieces):
            if not wp.startswith('##'):
                full_labels.append(labelpieces[ix])
        assert len(full_words) == len(full_labels)
        print(full_words, full_labels)
    return full_words, full_labels

# if __name__ == '__main__':
#     example = "Ik ben Wolfgang en ik woon in Berlijn"
#     # pret = example.split()
#     text, entities = run_BERTje(example)
#     res = map_tokens_to_entities(text, entities)
#     print(res)

if __name__ == '__main__':
    # Testing
    sentence = 'Ik heet Dagobert Jansen en ik kom uit Pakistan.'
    res = run_baseline_BERT_aligned(sentence)
    print(res)


# TODO Would be nice if I can fix this to see if it performs better? Though now it yields indexerrors

# def load_model():
#     # Load a trained model and vocabulary that you have fine-tuned
#     model = AutoModelForTokenClassification.from_pretrained("wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner")
#     tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
#     return model, tokenizer

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
#     print(torch.argmax(predictions_tensor, dim=2)[0])
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

# def run_baseline_BERTje(s):
#     ''':s: The sentence to run on
#     :type: s: A list of tokens'''
#     remove = ['[CLS]', '[SEP]', '[UNK]', '[PAD]']
#     tokens = []
#     labs = []
#     text = s
#     model, tokenizer = load_model()
#     encoding = tokenizer(text, return_tensors = 'pt', truncation = True, max_length = 512)
#     outputs = model(**encoding)
#     logits = outputs.logits
#     # print(logits.shape)
#     predicted_label_classes = logits.argmax(-1)
#     # print(predicted_label_classes)
#     predicted_labels = [model.config.id2label[id] for id in predicted_label_classes.squeeze().tolist()]
#     # print(predicted_labels)
#     for id, label in zip(encoding.input_ids.squeeze().tolist(), predicted_labels):
#         token = tokenizer.decode([id])
#         if token in remove:
#             continue
#         else:
#             tokens.append(token)
#             labs.append(label)
#     return tokens, labs

# def wordpieces_to_tokens(wordpieces: List, labelpieces: List = None) -> Tuple[List, List]:
#     textpieces = " ".join(wordpieces)
#     full_words = re.sub(r'\s##', '', textpieces).split()
#     full_labels = []
#     if labelpieces:
#         for ix, wp in enumerate(wordpieces):
#             if not wp.startswith('##'):
#                 full_labels.append(labelpieces[ix])
#         assert len(full_words) == len(full_labels)
#         print(full_words, full_labels)
#     return full_words, full_labels