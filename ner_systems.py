"""This file will contain several NER systems that we can test on biographical data,
we only use PER and LOC labels because all models classify these, which allows us to make proper
comparisons"""
import stanza
from visualize_stuff import Read  # We import the read module here so we don't need to rebuild it.
# from flair.data import Sentence
# from flair.models import SequenceTagger
from sklearn.metrics import classification_report
import csv
import BERTje_model as model
import finetuned_BERTje as ft_model 
from itertools import chain
# Because sentencepiece in flair does not support python=3.10 yet, we had to change to python version 3.9
# Flair 0.11 does not output to_tagged_string properly, issue noted and project reverted to flair 0.10 for now

class Run_Models():
    '''Class for running the Stanza, Flair and BERTje models'''
    def __init__(self, bio_obj):
        self.bio_obj = bio_obj
        self.tokens = []
        self.preds = []
        self.gold = [word['label'] for dct in self.bio_obj for word in dct['text_entities'] if not word['text'].startswith("<")]

    def run_flair(self):
        tagger = SequenceTagger.load('flair/ner-dutch-large')
        for dct in self.bio_obj:
            for s in dct["text_sents"]:
                # sentence = ' '.join(s) 
                flair_piece = Sentence(s, use_tokenizer=False)
                tagger.predict(flair_piece)
                tagged_lst = flair_piece.to_tagged_string().split()
                
                for index, token in enumerate(tagged_lst):
                    if not index == len(tagged_lst)-1:
                        check = ["<B-", "<E-", "<I-", "<S-"]
                        if any(tagged_lst[index+1].startswith(i) for i in check):
                            label = tagged_lst[index+1] #[3:6]
                            self.tokens.append(token) # The label always occurs after the token
                            self.preds.append(label)
                            continue
                        elif any((token).startswith(i) for i in check):
                            continue
                        else:
                            label = "O"
                            self.tokens.append(token)
                            self.preds.append(label)
                    if index == len(tagged_lst)-1:
                        if not token.startswith('<'):
                            self.tokens.append(token)
                            self.preds.append('O') 
        og_tokens = [word['text'] for dct in self.bio_obj for word in dct['text_entities'] if not word['text'].startswith("<")] # TODO See if < really is a good way to go about this
        return self.tokens, self.preds, self.gold

    def run_stanza(self):
        ''':return: token, pred, gold'''
        nlp = stanza.Pipeline(lang = "nl", processors= 'tokenize, ner', tokenize_pretokenized=True)
        gold = [word['label'] for dct in self.bio_obj for word in dct['text_entities']]
        tokens = []
        labels = []
        for dct in self.bio_obj:
            doc= nlp(dct['text_sents'])
        for sent in doc.sentences:
            for token in sent.tokens:
                tokens.append(token.text)
                labels.append(token.ner)
        return tokens, labels, gold

    def run_baseline_bertje(self):
        for dct in self.bio_obj:
            for s in dct["text_sents"]:
                sentence = ' '.join(s)
                example, ner_results = model.run_BERTje(sentence)
                tokens, labels = model.map_tokens_to_entities(example, ner_results)
                for t,g in zip(tokens, labels):
                    self.tokens.append(t)
                    self.preds.append(g)
        return self.tokens, self.preds, self.gold


    def discover_alignment_issues(self):
        print('AssertionError, finding alignment problem')
        t = list(chain(self.bio_obj['text_sents']))
        g = self.preds
        for i, (a, b) in enumerate(zip(t,g)):
            print(i, a, b)
            if a != b:
                print(f'ISSUE AT {i}, {a} {b}')
                break

    def run_finetuned_bertje(self):
        for dct in self.bio_obj:
            for s in dct["text_sents"]:
                sentence = ' '.join(s)
                tokens, labels = ft_model.run_finetuned_BERT_aligned(sentence)
                for t,g in zip(tokens, labels):
                    self.tokens.append(t)
                    self.preds.append(g)
        try:
            assert len(self.gold) == len(self.tokens), f'Will not be able to write file with correct alignment {len(self.gold)}, {len(self.preds)}'
        except AssertionError:
            self.discover_alignment_issues()
            for a,b in zip(self.tokens, self.gold):
                print(a,b)
        return self.tokens, self.preds, self.gold

    def to_file(self, path = '', name = ''):
        print(set(self.preds))
        print(set(self.gold))

class Write_Output_to_File():
    def __init__(self, tok, pred, gold):
        self.tok = tok
        self.pred = pred
        self.gold = gold
    
    def to_tsv(self, path):
        with open(path, 'w') as f:
            writer = csv.writer(f, delimiter='\t', quotechar = "|")
            for a, b, c in zip(self.tok, self.pred, self.gold):
                writer.writerow([a,b,c])

class Clean_Model_Output():
    """Will clean out all tags other than PER and LOC"""
    def __init__(self, pred, gold):
        self.pred = pred
        self.gold = gold

    def clean_flair(self):
            # We first grab only LOC and PER labels, and then clean them out
        clean_pred = []
        clean_gold = []
        # Cleaning the predictions {'<I-ORG>', '<E-PER>', '<S-MISC>', 'O', '<S-ORG>', '<B-PER>', '<I-MISC>', '<I-LOC>', '<S-LOC>', '<S-PER>', '<E-MISC>', '<B-MISC>', '<E-LOC>', '<B-ORG>', '<I-PER>', '<B-LOC>', '<E-ORG>'}
        for label in self.pred:
            if label.endswith('LOC>'):
                clean_pred.append('LOC')
            elif label.endswith('PER>'):
                clean_pred.append('PER')
            elif label == 'O':
                clean_pred.append(label)
            else:
                clean_pred.append('O')
        # Cleaning the GOLD {'B-TIME', 'B-PER', 'O', 'I-LOC', 'B-LOC', 'I-TIME', 'I-PER'}
        for label in self.gold:
            if label.endswith('LOC') or label.endswith('PER'):
                # print('Adding slice', label[2:5])
                clean_gold.append(label[2:5])
            else:
                clean_gold.append('O')
        # assert len(clean_gold) == len(clean_pred), 'Gold size is different than pred size'
        print(f'Starting length before cleaning, pred: {len(self.pred)}, gold: {len(self.gold)}')
        print(f"Cleaned sizes, pred {len(clean_pred)}, gold: {len(clean_gold)}")
        return clean_pred, clean_gold

    def clean_stanza(self):
        """_summary_

        Returns:
            _type_: tuple
        """
        print(set(self.pred))
        print(set(self.gold))
        clean_pred = []
        clean_gold = []
        for i, e in zip(self.pred, self.gold):
            if i.endswith('LOC') or i.endswith('PER'):
                clean_pred.append(i[-3:])
            else:
                clean_pred.append('O')
            if e.endswith('LOC') or e.endswith('PER'):
                clean_gold.append(e[-3:])
            else:
                clean_gold.append('O')
        return clean_pred, clean_gold

    def clean_bertje(self):
        clean_pred = []
        clean_gold = []
        for i, e in zip(self.pred, self.gold):
            if i.upper().endswith('LOC') or i.upper().endswith('PER'):
                clean_pred.append(i[-3:].upper())
            else:
                clean_pred.append('O')
            if e.upper().endswith('LOC') or e.upper().endswith('PER'):
                clean_gold.append(e[-3:])
            else:
                clean_gold.append('O')
        return clean_pred, clean_gold

def Evaluate_Model(pred, gold):
    """_summary_

    Args:
        tok (_type_): _description_
        pred (_type_): _description_
        gold (_type_): _description_
    """
    report = classification_report(y_true = gold, y_pred = pred, output_dict = True)
    for k, v in report.items():
        print(f"{k}: {v}")
    return report        

def run_flair(path):
    r = Read(path)
    bio_obj = r.from_tsv()
    # [word['label'] for dct in self.bio_obj for word in dct['text_entities']]
    a = Run_Models(bio_obj)
    tok, pred, gold = a.run_flair()
    outputter = Write_Output_to_File(tok, pred, gold)
    writepath = "model_results/flair_"+path.split('/')[-1].rstrip('.txt')+".tsv"
    outputter.to_tsv(writepath)
    cleaner = Clean_Model_Output(pred, gold)
    pred, gold = cleaner.clean_flair()
    Evaluate_Model(pred, gold)

def run_stanza(path):
    r = Read(path)
    bio_obj = r.from_tsv()
    # [word['label'] for dct in self.bio_obj for word in dct['text_entities']]
    a = Run_Models(bio_obj)
    tok, pred, gold = a.run_stanza()
    outputter = Write_Output_to_File(tok, pred, gold)
    writepath = "model_results/stanza_"+path.split('/')[-1].rstrip('.txt')+".tsv"
    outputter.to_tsv(writepath)
    cleaner = Clean_Model_Output(pred, gold)
    pred, gold = cleaner.clean_stanza()
    Evaluate_Model(pred, gold)

def run_baseline_BERTje(path):
    r = Read(path)
    bio_obj = r.from_tsv()
    print('Running baseline bertje model')
    a = Run_Models(bio_obj)
    tok, pred, gold = a.run_baseline_bertje()
    print('Writing to file')
    outputter = Write_Output_to_File(tok, pred, gold)
    writepath = "model_results/baseline_bertje_"+path.split('/')[-1].rstrip('.txt')+".tsv"
    outputter.to_tsv(writepath)
    cleaner = Clean_Model_Output(pred, gold)
    pred, gold = cleaner.clean_bertje()
    Evaluate_Model(pred, gold)

def run_finetuned_BERTje(path):
    print('Reading path')
    r = Read(path)
    bio_obj = r.from_tsv()
    print('Running finetuned bertje model')
    a = Run_Models(bio_obj)
    tok, pred, gold = a.run_finetuned_bertje()
    print('Writing to file')
    outputter = Write_Output_to_File(tok, pred, gold)
    writepath = "model_results/finetuned_bertje_"+path.split('/')[-1].rstrip('.txt')+".tsv"
    outputter.to_tsv(writepath)
    cleaner = Clean_Model_Output(pred, gold)
    pred, gold = cleaner.clean_bertje()
    Evaluate_Model(pred, gold)

def main(path):
    '''Performs experiment'''
    print("Running Flair")
    run_flair(path)
    print("Running Stanza")
    run_stanza(path)
    print('Running Baseline BERTje')
    run_baseline_BERTje(path)
    print('Running finetuned BERTje')
    run_finetuned_BERTje(path)
    print('Success! Experiment complete')
    
if __name__ == '__main__':
    test_on_partitions = ["../data/train/AITrainingset1.0/Clean_Data/test_NHA_cleaned.txt", "../data/train/AITrainingset1.0/Clean_Data/test_RHC_cleaned.txt",
                        "../data/train/AITrainingset1.0/Clean_Data/test_SA_cleaned.txt"]
    for path in test_on_partitions:
        main(path)