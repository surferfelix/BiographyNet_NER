# NER Classification on old Biographical Texts

This library includes several NLU models that are used for applying Named Entity Recognition in the Dutch language on older Biographical texts from the 18th-20th centuries. Additionally we explore the effects of finetuning a BERT model on older data to explore whether it adapts to the domain. 

The model evaluates the following NLU systems for NER on our data:
- Stanza
- Flair
- a BERTje NER model
- our finetuned BERTje on in-domain data

## Carry out experiment

You can carry out the entire experiment with the following steps after cloning this repository.
Note: We did not yet add the 'data' directory used for this experiment, but will consider doing so in the future. 

1. Install the dependencies in a python 3.8 environment with:

 `pip install -r requirements.txt`
 
2. Navigate to the clone of this repository

3. You can then carry out the experiment with the following:

 `python ner_systems.py`
 
## Finetune the BERTje model

Finetuning this BERTje model on other data can be easily done with the following

`python fine_tuned_BERTje.py {TRAIN_PATH}, {EVAL_PATH}, -e {EPOCHS}, -b {BATCH_SIZE}`

where epochs and batch size are optional arguments, with default epochs being 8, and default batch size being 4

# Load our model from Transformers

For easy implementation, this model has been uploaded to huggingface. You can use it for your own purposes with the following code-block. 

```
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained('surferfelix/ner-bertje-tagdetekst')
model = AutoModelForTokenClassification.from_pretrained('surferfelix/ner-bertje-tagdetekst')
```
