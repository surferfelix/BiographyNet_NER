import json
import csv
import sys
import os

def read_training_data(path, mode = 'train') -> list:
    ''''''
    if mode == 'train':
        key_counter = dict()
        with open(path, encoding = 'windows-1252') as file: # The training file is actually tsv
                csv.field_size_limit(sys.maxsize)
                infile = csv.reader(file, delimiter = '\t', quotechar = '|')
                labels = [row[1] for row in infile if row]
                for label in labels:
                    if not label in key_counter:
                        key_counter[label] = 1
                    else:
                        key_counter[label] +=1 
                print(key_counter)
                return [row for row in infile]
    elif mode == 'dev':
        token_total = []
        keys_total = []
        key_counter = dict()
        files = os.listdir(path)
        for f in files:
            if not f.startswith('.'):
                print(f'Reading {f}')
                with open(f"{path}/{f}", 'r') as json_file:
                    a = json.load(json_file)
                    token_total += a['text_tokens'] # We append like this so it remains flattened
                    keys_total += [d['label'] for d in a['text_entities']]
        for key in keys_total:
            if not key in key_counter:
                key_counter[key] = 1
            else:
                key_counter[key] += 1
        print(key_counter)
        return token_total, keys_total

def check_data_distribution(train, dev, dis): # for now i use this to compare type distributions
    # First we check total types and intersection
    t = set(i[0].lower() for i in train if i)
    d = set(i.lower() for i in dev)
    length_of_train = len(list(t))
    length_of_dev = len(list(d))
    intersection = len(list(t.intersection(d)))
    print(f'The total types in the trainingset is {length_of_train}\n and the total types in development is {length_of_dev} \n, the types they have in common contains {intersection} elements')
    # Now we check the distribution of labels

if __name__ == '__main__':
    train_path = '../data/train/AITrainingset1.0/Data/train.tsv'
    dev_path = '../data/development/json'
    train = read_training_data(train_path)
    # dev_tok, *dev_keys = read_training_data(dev_path, mode = 'dev')
    # check_data_distribution(train, dev, dis)
    

