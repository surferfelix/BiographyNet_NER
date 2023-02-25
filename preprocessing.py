
import spacy
from visualize_stuff import Read
from spacy.pipeline.senter import DEFAULT_SENTER_MODEL
from spacy.tokens import Doc
import os
import csv
import sys
class Preprocess():
    """This class is for preprocessing the sentences of the text so that 
    the models can properly deal with them"""
    def __init__(self, path_in):
        self.path_in = path_in
        self.nlp = spacy.load('nl_core_news_sm')

    def sentence_tokenize(self, path_out):
        csv.field_size_limit(sys.maxsize)
        all_sents = []
        all_labels = []
        text = []
        ### Reading the file
        with open(self.path_in, encoding = "utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            print('Processing full text, this might take a while:')
            for row in reader:
                if row:
                    text.append(row[0])
                    all_labels.append(row[1])
            self.nlp.add_pipe('sentencizer')
            doc = Doc(self.nlp.vocab, text)
            for sentence in self.nlp(doc).sents:
                slist = []
                for i, tokens in enumerate(sentence):
                    slist.append(tokens)
                    if i == len(sentence)-1:
                        all_sents.append(slist)
            
        ### Writing the file
        with open(path_out, 'w') as f:
            writer = csv.writer(f, delimiter = '\t', quotechar = "|")
            label_fetch = 0
            for lst in all_sents:
                writer.writerow([]) # Empty line for each new sentence
                for t in lst: # Each token in the sentence
                    token = t.text
                    label = all_labels[label_fetch]
                    writer.writerow([token, label])
                    label_fetch+=1

    def find_amount_of_sentences(self):
            '''This is just another analysis tool to help find the amount of sentences'''
            scount = 0
            with open(self.path_in) as f:
                infile = f.readlines()
                for line in infile:
                    line = line.rstrip('\n')
                    if not line:
                        scount+=1
            print(f'we found {scount} sentences!')
            return scount

    def check_double_empties(self):
        bad_indexes = []
        csv.field_size_limit(sys.maxsize)
        with open(self.path_in, encoding = "utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar = "|")
            print('Processing full text, this might take a while:')
            all = [row for row in reader]
            for index, row in enumerate(reader):
                if not row: # When we find an empty
                    try:
                        if not all[index+1]:
                            bad_indexes.append(index)
                    except IndexError: # We are at end of file:
                        continue
            return bad_indexes

    def remove_single_token_sentences(self):
        csv.field_size_limit(sys.maxsize)
        all_sents = []
        all_labels = []
        with open(self.path_in, encoding = "utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar = "|")
            print('Processing full text, this might take a while:')
            l = []
            p = []
            for index, row in enumerate(reader):
                if row:
                    # print(token, label)
                    token = row[0]
                    label = row[1]
                    l.append(token)
                    p.append(label)
                if not row and len(l)>1: # WE AVOID single token sentences this way
                    # start of new sentence
                    all_sents.append(l)
                    all_labels.append(p)
                    l = []
                    p = []
        
        ### Writing the file
            print('Writing file')
            # print(all_sents)
            # print(all_labels)
        with open(self.path_in, 'w') as f:
            writer = csv.writer(f, delimiter = '\t', quotechar = "|")
            for s1, s2 in zip(all_sents, all_labels):
                if s1 and s2:
                    writer.writerow([]) # Empty line for each new sentence
                    # print("\n")
                for tok, lab in zip(s1,s2): # Each token in the sentence
                    # print(tok, lab)
                    writer.writerow([tok, lab])
                    
    def for_biographynet_test_partition(self, path_out):
        """Here we apply a heuristic to sentencize the partition"""
        # Each document starts with a commented line ## TextID = ID
        # and a blank line at end of document to seperate them
        # For each token that is a period, check if next token starts with uppercase
        all_labels = []
        full_text = []
        all_sents = []
        label_sents = []
        with open(self.path_in) as f:
            infile = f.readlines()
            print('Storing data in memory')
            for i, line in enumerate(infile):
                if line.startswith('##'):
                    continue
                row = line.rstrip('\n').split('\t')
                if len(row) > 1 and i+1 <= len(infile):
                    token = row[0]
                    label = row[1]
                    all_labels.append(label)
                    full_text.append(token)
            print('Finding sentences...')
            self.nlp.add_pipe('sentencizer')
            doc = Doc(self.nlp.vocab, full_text)
            for sentence in self.nlp(doc).sents:
                slist = []
                for i, tokens in enumerate(sentence):
                    slist.append(tokens)
                    if i == len(sentence)-1:
                        all_sents.append(slist)
            
        ### Writing the file
        with open(path_out, 'w') as f:
            writer = csv.writer(f, delimiter = '\t', quotechar = "|")
            label_fetch = 0
            for lst in all_sents:
                writer.writerow([]) # Empty line for each new sentence
                for t in lst: # Each token in the sentence
                    token = t.text
                    label = all_labels[label_fetch]
                    writer.writerow([token, label])
                    label_fetch+=1

def organize_sample_for_annotation(path):
    """Selects a random sample for annotation purposes"""
    from random import sample
    selected_texts = []
    for dct in Read(path).from_file(): # Reading from generator object
        if not 'bioport' in dct['source']:
            original_dct = dct['death_tm']
            if isinstance(original_dct, str):
                if len(original_dct) > 0:
                    cleaned = dct['death_tm'][0:4].strip('.').strip('~').strip("'").strip('"').strip()
                    try:
                        if int(cleaned) < 1800:
                            text = dct['text_clean']
                            if len(text) > 300:
                                selected_texts.append(text)
                    except ValueError:
                        continue
    selection = sample(selected_texts, k = 5)
    print(f'Selected 5 from total of {len(selected_texts)}')
    with open('samples/old_bio_portal_selection.txt', 'w') as f:
        f.write('\n'.join(selection))
    return "\n".join(selection)

def write_as_conll(selection):
    import csv
    writepath = "selection_old_biographies_mod.conll"
    tokens = selection.split(' ')
    with open(writepath, 'w') as write:
        writer = csv.writer(write, delimiter = '\t')    
        for token in tokens:
            writer.writerow([token, "O"])

if __name__ == '__main__':
    selection = organize_sample_for_annotation("../data/full/AllBios.jsonl")
    write_as_conll(selection)
    # train_dir = "../data/train/AITrainingset1.0/Data"
    # for path in os.listdir(train_dir):
    #     if not path.startswith('.') and path.endswith('txt'):
    #         path_in = f"{train_dir}/{path}"
    #         path_out = f"../data/train/AITrainingset1.0/Clean_Data/{path_in.split('/')[-1].rstrip('.txt')}_cleaned.txt"
    #         nlp = Preprocess(path_in)
    #         nlp.sentence_tokenize(path_out)

    # p_in = '../data/test/biographynet_test_A_gold.tsv'
    # p_out = '../data/test/cleaned/biographynet_test_A_gold_cleaned.tsv'

    # paths = ["../data/test/cleaned/biographynet_test_A_gold_cleaned.tsv"]
    # for path in paths:
    #     nlp = Preprocess(path)
    #     print('Looking at, :', path)
    #     a = nlp.find_amount_of_sentences()