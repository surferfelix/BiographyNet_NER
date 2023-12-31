"""This file will contain several NER systems that we can test on biographical data,
we only use PER and LOC labels because all models classify these, which allows us to make proper
comparisons"""
import stanza
from visualize_stuff import Read  # We import the read module here so we don't need to rebuild it.
from flair.data import Sentence
from flair.models import SequenceTagger
from sklearn.metrics import classification_report
import csv
import BERTje_model as model
import finetuned_BERTje as ft_model 
import gysbert_model as gysbert
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
        bert, tokenizer = model.load_model()
        for dct in self.bio_obj:
            for s in dct["text_sents"]:
                sentence = ' '.join(s)
                tokens, labels = model.run_baseline_BERTje(sentence, bert, tokenizer)
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
                print(sentence)
                tokens, labels = ft_model.run_finetuned_BERT_aligned(sentence)
                for t,g in zip(tokens, labels):
                    self.tokens.append(t)
                    self.preds.append(g)
        try:
            assert len(self.gold) == len(self.tokens), f'Will not be able to write file with correct alignment {len(self.gold)}, {len(self.preds)}'
        except AssertionError:
            self.discover_alignment_issues()
            # for a,b in zip(self.tokens, self.gold):
                # print(a,b)
        return self.tokens, self.preds, self.gold
    
    def run_finetuned_gysbert(self):
        for dct in self.bio_obj:
            for s in dct["text_sents"]:
                sentence = ' '.join(s)
                tokens, labels = gysbert.run_finetuned_BERT_aligned(sentence)
                for t,g in zip(tokens, labels):
                    self.tokens.append(t)
                    self.preds.append(g)
        try:
            assert len(self.gold) == len(self.tokens), f'Will not be able to write file with correct alignment {len(self.gold)}, {len(self.preds)}'
        except AssertionError:
            self.discover_alignment_issues()
            # for a,b in zip(self.tokens, self.gold):
                # print(a,b)
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
    """Will clean out all tags other than PER and LOC
    Returns new pred and gold"""
    def __init__(self, pred, gold):
        self.pred = pred
        self.gold = gold

    def clean_flair(self):
            # We first grab only LOC and PER labels, and then clean them out
        clean_pred = []
        clean_gold = []
        # Cleaning the predictions {'<I-ORG>', '<E-PER>', '<S-MISC>', 'O', '<S-ORG>', '<B-PER>', '<I-MISC>', '<I-LOC>', '<S-LOC>', '<S-PER>', '<E-MISC>', '<B-MISC>', '<E-LOC>', '<B-ORG>', '<I-PER>', '<B-LOC>', '<E-ORG>'}
        i_locs = ['I-LOC', 'E-LOC']
        b_locs = ['B-LOC', 'S-LOC']
        i_pers = ['I-PER', 'E-PER']
        b_pers = ['B-PER', 'S-PER']
        for label in self.pred:
            clabel = label.lstrip('<').rstrip('>')
            if clabel in i_locs:
                pred = 'I-LOC'
            elif clabel in b_locs:
                pred = 'B-LOC'
            elif clabel in i_pers:
                pred = 'I-PER'
            elif clabel in b_pers:
                pred = 'B-PER'
            else:
                pred = 'O'
            clean_pred.append(pred)
            # if any([label.endswith(i) for i in i_locs]):
            #     clean_pred.append('I-LOC')
            # elif any([label.endswith(b) for b in b_locs]):
            #     clean_pred.append('B-LOC')
            # elif label.endswith('PER>'):
            #     clean_pred.append('PER')
            # elif label == 'O':
            #     clean_pred.append(label)
            # else:
            #     clean_pred.append('O')
        # Cleaning the GOLD {'B-TIME', 'B-PER', 'O', 'I-LOC', 'B-LOC', 'I-TIME', 'I-PER'}
        for label in self.gold:
            if label.endswith('LOC') or label.endswith('PER'):
                # print('Adding slice', label[2:5])
                clean_gold.append(label)
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
            if i.startswith('E'):
                i = f"I-{i[-3:]}"
            elif i.startswith('S'):
                i = f"B-{i[-3:]}"
            if i.endswith('LOC') or i.endswith('PER'):
                clean_pred.append(i)
            else:
                clean_pred.append('O')
            if e.endswith('LOC') or e.endswith('PER'):
                clean_gold.append(e)
            else:
                clean_gold.append('O')
        return clean_pred, clean_gold

    def clean_bertje(self):
        clean_pred = []
        clean_gold = []
        assert len(self.pred) == len(self.gold), 'Pred and gold mismatch'
        for i, e in zip(self.pred, self.gold):
            if i.upper().endswith('LOC') or i.upper().endswith('PER'):
                clean_pred.append(i.upper())
            else:
                clean_pred.append('O')
            if e.upper().endswith('LOC') or e.upper().endswith('PER'):
                clean_gold.append(e.upper())
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
    labels = [i for i in set(pred) if not i == 'O']
    report = classification_report(y_true = gold, y_pred = pred, output_dict = True, labels = labels)
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

def run_gysbert(path):
    print('Reading path')
    r = Read(path)
    bio_obj = r.from_tsv()
    print('Running finetuned gysbert model')
    a = Run_Models(bio_obj)
    tok, pred, gold = a.run_finetuned_gysbert()
    print('Writing to file')
    outputter = Write_Output_to_File(tok, pred, gold)
    writepath = "model_results/finetuned_gysbert_"+path.split('/')[-1].rstrip('.txt')+".tsv"
    outputter.to_tsv(writepath)
    cleaner = Clean_Model_Output(pred, gold)
    pred, gold = cleaner.clean_bertje()
    Evaluate_Model(pred, gold)

def evaluate_only(path):
    '''This will only run the evaluation, but will not train any model'''
    r=Read(path)
    clean_pred = []
    clean_gold = []
    preds, golds = r.as_eval_file()
    assert len(preds) == len(golds), 'Something went wrong with reading, length of preds not the same as golds.'
    clean_pred, clean_gold = Clean_Model_Output(preds, golds).clean_bertje()
    Evaluate_Model(clean_pred, clean_gold)

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
    print('Running GijsBERT')
    run_gysbert(path)
    print('Done')
    print('Evaluating...')
    # evaluate_only(path)
    
if __name__ == '__main__':
    # run_on_partitions = ["../data/train/AITrainingset1.0/Clean_Data/test_SA_cleaned.txt"]
    # run_on_partitions = ["../data/train/AITrainingset1.0/Clean_Data/test_NHA_cleaned.txt", 
    #                      "../data/test/cleaned/biographynet_test_A_gold_cleaned.tsv", 
    #                      "qualitative_eval/biography_selection_middle_dutch.conll", 
    #                      "qualitative_eval/biography_selection_modern_dutch.conll"]
    run_on_partitions = ["foo_data/test_RHC_cleaned.txt"]
    for path in run_on_partitions:
        main(path)
    print('Success! Experiment complete')

