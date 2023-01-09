
import spacy
from spacy.pipeline.senter import DEFAULT_SENTER_MODEL
from spacy.tokens import Doc
import os
import sys
class Preprocess():
    """This class is for preprocessing the sentences of the text so that 
    the models can properly deal with them"""
    def __init__(self, path_in, path_out):
        self.path_in = path_in
        self.path_out = path_out
        self.nlp = spacy.load('nl_core_news_sm')

    def sentence_tokenize(self):
        import csv
        csv.field_size_limit(sys.maxsize)
        all_sents = []
        all_labels = []
        text = []
        ### Reading the file
        with open(self.path_in, encoding = "windows-1252") as f:
            reader = csv.reader(f, delimiter="\t", quotechar = "|")
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
        with open(self.path_out, 'w') as f:
            writer = csv.writer(f, delimiter = '\t', quotechar = "|")
            label_fetch = 0
            for lst in all_sents:
                writer.writerow([]) # Empty line for each new sentence
                for t in lst: # Each token in the sentence
                    token = t.text
                    label = all_labels[label_fetch]
                    writer.writerow([token, label])
                    label_fetch+=1


if __name__ == '__main__':
    train_dir = "../data/train/AITrainingset1.0/Data"
    for path in os.listdir(train_dir):
        if not path.startswith('.') and path.endswith('txt'):
            path_in = f"{train_dir}/{path}"
            path_out = f"../data/train/AITrainingset1.0/Clean_Data/{path_in.split('/')[-1].rstrip('.txt')}_cleaned.txt"
            nlp = Preprocess(path_in, path_out)
            nlp.sentence_tokenize()