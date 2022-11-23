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
    
    def from_directory(self) -> dict:
        bio_obj = []
        files = os.listdir(self.path)
        for f in files:
            if not f.startswith('.'):
                with open(f"{self.path}/{f}", 'r') as json_file:
                    a = json.load(json_file)
                    bio_obj.append(a)
        return bio_obj

    def from_file(self) -> list:
        bio_obj = []
        with open(self.path, 'r') as json_file:
            for line in json_file:
                bio_obj.append(json.loads(line)) #Appending dict to list
        return bio_obj

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
    path = '../data/development/json'
    a = Read(path)
    bio_obj = a.from_directory()
    print('This is from a directory')
    b = Counter(bio_obj, 'text_entities')
    c = b.from_bio_obj()
    print(c)
    
    # path = "/Volumes/Samsung_T5/Text_Mining/MA_Thesis_2/data/development/Allbios_mini.jsonl"
    # a = Read(path)
    # b = a.from_file()
    # Counter(b, 'text_entities')
    # print('This is from a file')
    # print(b)


