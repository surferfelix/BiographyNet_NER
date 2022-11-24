'''This script can be used to visualize dictionary data'''

import json
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from spacy.tokens import Doc

# Parser
import argparse

class Read:
    # TODO Currently only works on json objects
    '''Can read files from different sources (dir, single_file)'''
    def __init__(self, path: str):
        self.path = path
    
    def from_directory(self) -> dict: # For our development set
        bio_obj = []
        files = os.listdir(self.path)
        for f in files:
            if not f.startswith('.'):
                with open(f"{self.path}/{f}", 'r') as json_file:
                    a = json.load(json_file)
                    bio_obj.append(a)
        return bio_obj

    def from_file(self) -> list: # For the full evaluation set
        bio_obj = []
        with open(self.path, 'r') as json_file:
            for line in json_file:
                bio_obj.append(json.loads(line)) #Appending dict to list
        return bio_obj

    def from_tsv(self) -> list: # For the training data
        # word\tlabel --> [{text_entities: [{text: word, label: label}]}]
        import csv
        import sys
        ret = [{'text_entities': [], 'text_tokens': []}]
        with open(self.path, encoding='windows-1252') as file:
            csv.field_size_limit(sys.maxsize)
            infile = csv.reader(file, delimiter='\t', quotechar='|')
            # TODO Issue here is that we want to chain B I I tags together
            for row in infile:
                if row: # We skip sentence endings here, so we need to remember to see if we want it back later
                    word = row[0]
                    label = row[1]
                    info = {'text': word, 'label': label}
                    for dct in ret:
                        dct['text_entities'].append(info)
                        dct['text_tokens'].append(word)
        return ret

class Counter:
    """Will count instances of key in obj and return a new dictionary object"""
    # Example Counter(bio_obj, 'named_entities') -> {PER: 113}
    # TODO: Make this counter object
    
    def __init__(self, obj, key):
        self.obj = obj
        self.key = key
    
    def from_bio_obj(self):
        '''Takes bio_obj and returns a counter dict'''
        # This currently just works for the 'text_entities' key
        counter_obj = dict()
        # assert type(self.obj[self.key] == list), 'We need to iterate through a list of strings :('
        for i, lst in enumerate(self.obj):
            for dct in lst[self.key]:
                if dct['label'] in counter_obj:
                    counter_obj[dct['label']] += 1
                else:
                    counter_obj[dct['label']] = 1
        return counter_obj

class Interpret:
    '''For other interpretations, such as which people have tag PER for example'
    Requires the bio_obj'''
    from operator import itemgetter
    def __init__(self, obj):
        self.obj = obj
    
    def count_word_rank(self) -> dict:
        '''Counts words and removes stopwords
        :return: dict of with count for each word'''
        stopwords = set()
        with open("stopwords/stopwords.txt") as f:
            infile = f.readlines()
            for line in infile:
                stopwords.add(line.rstrip('\n'))
        print(f"Removing stopwords\n{stopwords}")
        counter_obj = dict()
        for dct in self.obj:
            for token in dct['text_tokens']:
                token = token.lower()
                if not token in stopwords:
                    if token in counter_obj:
                        counter_obj[token] += 1
                    else:
                        counter_obj[token] = 1
        return counter_obj

    def popular_persons(self, n: int):
        "Will print n most popular persons in bio_object"
        counter_obj = dict()
        for dct in self.obj:
            for lst in dct['text_entities']:
                if "PER" in lst['label']:
                    if lst['text'] in counter_obj:
                        counter_obj[lst['text']] += 1
                    else:
                        counter_obj[lst['text']] = 1
        res = dict(sorted(counter_obj.items(), key = self.itemgetter(1), reverse = True)[:n])
        return res
    
    def popular_locations(self, n: int):
        "Will print n most popular persons in bio_object"
        counter_obj = dict()
        for dct in self.obj:
            for lst in dct['text_entities']:
                if "LOC" in lst['label']:
                    if lst['text'] in counter_obj:
                        counter_obj[lst['text']] += 1
                    else:
                        counter_obj[lst['text']] = 1
        res = dict(sorted(counter_obj.items(), key = self.itemgetter(1), reverse = True)[:n])
        return res

    def popular_time(self, n: int):
        "Will print n most popular persons in bio_object"
        counter_obj = dict()
        for dct in self.obj:
            for lst in dct['text_entities']:
                if "TIME" in lst['label']:
                    if lst['text'] in counter_obj:
                        counter_obj[lst['text']] += 1
                    else:
                        counter_obj[lst['text']] = 1
        res = dict(sorted(counter_obj.items(), key = self.itemgetter(1), reverse = True)[:n])
        return res

    def popular_organizations(self, n: int):
        "Will print n most popular persons in bio_object"
        counter_obj = dict()
        for dct in self.obj:
            for lst in dct['text_entities']:
                if "ORG" in lst['label']:
                    if lst['text'] in counter_obj:
                        counter_obj[lst['text']] += 1
                    else:
                        counter_obj[lst['text']] = 1
        res = dict(sorted(counter_obj.items(), key = self.itemgetter(1), reverse = True)[:n])
        return res
    
    def popular_misc(self, n: int):
        "Will print n most popular persons in bio_object"
        counter_obj = dict()
        for dct in self.obj:
            for lst in dct['text_entities']:
                if "MISC" in lst['label']:
                    if lst['text'] in counter_obj:
                        counter_obj[lst['text']] += 1
                    else:
                        counter_obj[lst['text']] = 1
        res = dict(sorted(counter_obj.items(), key = self.itemgetter(1), reverse = True)[:n])
        return res

    def count_words(self):
        count = 0
        for dct in self.obj:
            for word in dct['text_tokens']:
                count+=1
        return count

class Compare:
    '''Will take two bio_obj and compare them''' 
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def overlap(self, lemmatize = True, verbose = True):
        '''A function for calculating the overlapping words in x and y
        calculates score -> overlap/mintotaltok(obj1/2)*100
        :param lemmatize: should we lemmatize before calculating overlap?'''
        # Prepare part #TODO: Think about splitting this bit?
        obj1 = self.overlap_help(self.x)
        obj2 = self.overlap_help(self.y)
        if lemmatize:
            obj1 = self.lemmatize_help(obj1)
            obj2 = self.lemmatize_help(obj2, verbose = False)
        if not lemmatize and verbose:
            print('Lemmatizer OFF')
        intersection = len(list(obj1.intersection(obj2)))
        obj1_size = len(list(obj1))
        obj2_size = len(list(obj2))
        ret = f"The intersection between x of \nlength {obj1_size} and y of \nlength {obj2_size} is \n{intersection}"
        score = intersection / obj2_size*100
        print(f"The score we will give is {score}")
        return ret
        
    def overlap_help(self, obj) -> set:
        ret = set()
        for dct in obj:
            for token in dct['text_tokens']:
                ret.add(token)
        return ret
    
    def lemmatize_help(self, obj, verbose = True) -> set:
        if verbose:
            print('Lemmatizer ON')
            print('Warning: Using the lemmatizer might take longer to run')
        ret = set()
        nlp = self.load_spacy()
        doc = Doc(nlp.vocab, list(obj))
        for token in nlp(doc):
            ret.add(token.lemma_)
        return ret

    def load_spacy(self):
        nlp = spacy.load('nl_core_news_lg')
        return nlp

class Visualize:
    '''This class can take a dictionary with {str: int}
        and visualize it to your liking
        :type obj: dict
        :type path: string
        :path: path to write to'''
    def __init__(self, obj: dict, path: str):
        self.path = path
        self.obj = obj
    
    def as_donut(self):
        '''Creates and saves a donut plot'''
        values = []
        sources = []

        # Populating the lists
        for source, value in self.obj.items():
            values.append(value)
            sources.append(source)
        
        # Visualizing the data
        plt.pie(values, labels = sources, frame = False, autopct='%1.1f%%')
        my_circle = plt.Circle( (0,0), 0.7, color='white')
        # p = plt.gcf()
        p.gca().add_artist(my_circle)
        plt.savefig(self.path)
    
    def as_wordcloud(self):
        '''Will take bio_obj, and count text tokens to generate a wordcloud
        :return: a wordcloud'''
        wc = WordCloud(background_color="white",width=1000,height=1000, 
                        max_words=200,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(self.obj)
        wc.to_file(self.path)

    def as_circular_barplot(self):
        pass

    #TODO circular barplot / wordcloud / treemap


def train_comparisons(bio):
    '''Takes bio object and will compare to all items in trainset'''
    loc = '../data/train/AITrainingset1.0/Data'
    for path in os.listdir(loc):
        if path.endswith('.txt') and not path.startswith('.') and not path.startswith('vocab'):
            a = Read(f"{loc}/{path}")
            bio_obj_1 = a.from_tsv()
            
            b = Interpret(bio_obj_1)
            word_count = b.count_words()

            c = Compare(bio, bio_obj_1)
            ret = c.overlap(lemmatize = True)
        
            print(f"In the set {path}")
            print(f'The word count is {word_count}')
            print(ret)


if __name__ == '__main__':
    orig = Read("../data/development/json")
    bio = orig.from_directory()
    d = Interpret(bio)
    d = d.count_word_rank()
    a = Visualize(d, '../data/plots/cloudtest.png')
    b = a.as_wordcloud()
    print(b)
    
    

    d = Visualize(c, '../data/plots/top_10_misc.png')
    d.as_donut()