import json
import csv
import sys
import os

def read_training_data(path, mode = 'train') -> list:
    ''''''
    if mode == 'train':
        with open(path, encoding = 'windows-1252') as file: # The training file is actually tsv
                csv.field_size_limit(sys.maxsize)
                infile = csv.reader(file, delimiter = '\t', quotechar = '|')
                return [row for row in infile]
    elif mode == 'dev':
        token_total = []
        files = os.listdir(path)
        for f in files:
            if not f.startswith('.'):
                print(f'Reading {f}')
                with open(f"{path}/{f}", 'r') as json_file:
                    a = json.load(json_file)
                    print(a['text_tokens'])
                    token_total += a['text_tokens'] # We append like this so it remains flattened
        for i in token_total:
            print(i)
        return token_total

def check_data_distribution(train, dev): # for now i use this to compare type distributions
    t = set(i[0] for i in train if i)
    d = set(dev)
    length_of_train = len(list(t))
    length_of_dev = len(list(d))
    intersection = len(list(t.intersection(d)))
    print(f'The total types in the trainingset is {length_of_train}\n and the total types in development is {length_of_dev} \n, the types they have in common contains {intersection} elements')
    

if __name__ == '__main__':
    train_path = '../data/train/AITrainingset1.0/Data/train.tsv'
    dev_path = '../data/development/json'
    train = read_training_data(train_path)
    dev = read_training_data(dev_path, mode = 'dev')
    check_data_distribution(train, dev)
    

