"""This module will contain the basic BERTje system.
Here we use the BERTje embeddings and then put them into a CRF
model that will make predictions. This will be the baseline BERT system,

Credit for the BERTje model goes to wiestedv"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, PreTrainedTokenizerFast
from transformers import pipeline

def run_BERTje(s):
    ''':s: The sentence to run on
    :type: s: A list of tokens'''
    tokenizer = AutoTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner")
    model = AutoModelForTokenClassification.from_pretrained("wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner")
    nlp = pipeline("ner", model=model, tokenizer = tokenizer)
    example = s
    ner_results = nlp(example)
    return example, ner_results


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
        
        # Set the labels for the tokens corresponding to the entity
        for index, token in enumerate(tokens): #TODO Create labels list from here instead
            if token in entity_tokens:
                labels[index] = label
    print(tokens)
    print(labels)
    return tokens, labels


if __name__ == '__main__':
    example = "Ik ben Wolfgang en ik woon in Berlijn"
    # pret = example.split()
    text, entities = run_BERTje(example)
    res = map_tokens_to_entities(text, entities)
    print(res)