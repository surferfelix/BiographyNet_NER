'''This script can be used to visualize dictionary data'''

# Some imports all the classes need
import json
import os
import matplotlib.pyplot as plt

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
        for i in bio_obj:
            print(i['text_entities'])
        return bio_obj

    def from_file(self) -> list: # For the full evaluation set
        bio_obj = []
        with open(self.path, 'r') as json_file:
            for line in json_file:
                bio_obj.append(json.loads(line)) #Appending dict to list
        return bio_obj

    def from_tsv(self) -> list: # word\tlabel --> [{text_entities: [{text: word, label: label}]}] # For the training data
        import csv
        import sys
        ret = [{'text_entities': []}]
        with open(self.path, encoding='windows-1252') as file:
            csv.field_size_limit(sys.maxsize)
            infile = csv.reader(file, delimiter='\t', quotechar='|')
            # TODO Issue here is that we want to chain B I I tags together
            for row in infile:
                if row: # We skip sentence endings here, so we need to remember to see if we want it back later
                    info = {'text': row[0], 'label': row[1]}
                    for dct in ret:
                        dct['text_entities'].append(info)
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
    def __init__(self, obj):
        self.obj = obj
    
    def popular_persons(self, n: int):
        from operator import itemgetter
        "Will print n most popular persons in bio_object"
        counter_obj = dict()
        for dct in self.obj:
            for lst in dct['text_entities']:
                if "PER" in lst['label']:
                    if lst['text'] in counter_obj:
                        counter_obj[lst['text']] += 1
                    else:
                        counter_obj[lst['text']] = 1
        res = dict(sorted(counter_obj.items(), key = itemgetter(1), reverse = True)[:n])
        return res
    
    def count_words(self):
        pass
                    
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
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        plt.savefig(self.path)

if __name__ == '__main__':
    path = '../data/train/AITrainingset1.0/Data/train.tsv'
    # path = '../data/development/json'
    a = Read(path)
    bio_obj = a.from_tsv()
    b = Interpret(bio_obj)
    res = b.popular_persons(10)
    print(res)
    # print(res)

