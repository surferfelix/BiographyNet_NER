"""This module will contain the basic BERTje system.
Here we use the BERTje embeddings and then put them into a CRF
model that will make predictions. This will be the baseline BERT system,

Credit for the BERTje model goes to wiestedv"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def run_BERTje(s):
    ''':s: The sentence to run on
    :type: s: A list of tokens in the sentence'''
    tokenizer = AutoTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner")
    model = AutoModelForTokenClassification.from_pretrained("wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    example = "Ik ben wolfgang en ik woon in Berlijn"
    ner_results = nlp(example)
    print(ner_results)

if __name__ == '__main__':
    run_BERTje('')