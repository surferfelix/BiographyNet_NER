"""This file serves the purpose of collecting statistics information for the biographies"""

import json
import os
from os import stat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

def read_json(path: str, mode: str) -> list:
    '''Reads a json file for location specified in path
    mode: refers to whether it should be read from a directory or single file'''
    bio_obj = []
    if mode == 'file':
        with open(path, 'r') as json_file:
            for line in json_file:
                bio_obj.append(json.loads(line)) #Appending dict to list
    elif mode == 'dir':
        files = os.listdir(path)
        for f in files:
            if not f.startswith('.'):
                print(f'Reading {f}')
                with open(f"{path}/{f}", 'r') as json_file:
                    a = json.loads(json_file)
                    print(a)
                    # for line in json_file:
                    #     bio_obj.append(json.loads(line))
    return bio_obj

def collect_statistics(data, split: str):
    '''Will interpret the read data and print statistics
    with split you can determine what portion of the data you want to
    generate for.'''
    stat_dict = {'source': {}, 'partition': {}} # Pretty much a counter dict
    #TODO Clean this up a little bit to reuse code
    with open('text_output.txt', 'w') as f:
        if split == 'all':
            for d in data:
                for key, value in d.items():
                    if 'source' in key: # Counting the amount of entries per source
                        if not value in stat_dict['source']:
                            stat_dict['source'][value] = 1
                        elif value in stat_dict['source']:
                            stat_dict['source'][value] += 1
                    if 'partition' in key:
                        if not value in stat_dict['partition']:
                            stat_dict['partition'][value] = 1
                        elif value in stat_dict['partition']:
                            stat_dict['partition'][value] += 1
        elif split == 'train':
            for d in data:
                if d['partition'] == 'train':
                    for key, value in d.items():
                        if 'source' in key: # Counting the amount of entries per source
                            if not value in stat_dict['source']:
                                stat_dict['source'][value] = 1
                            elif value in stat_dict['source']:
                                stat_dict['source'][value] += 1
                        if 'partition' in key:
                            if not value in stat_dict['partition']:
                                stat_dict['partition'][value] = 1
                            elif value in stat_dict['partition']:
                                stat_dict['partition'][value] += 1
        elif split == 'train and time_period':
            pass #TODO Find the metric to measure this
        elif split == 'amount of labels per partition':
            for d in data:
                if d['text_entities']:
                    print('yay')

    return stat_dict

def visualize_stat_dict(stat_dict):
    '''Will try to visually reproduce some of the statistics
    you can select what aspect of the data you want to visualize
    "total" will select all the data for all partitions, 
    whilst "train" "dev" "test" will search for that partition only.'''
    print('\nTotal amount of sources:', len(stat_dict['source'].keys()))
    stat_dict_sources = {key: value for key, value in stat_dict['source'].items() if value > 1}
    stat_dict_partitions = {key: value for key, value in stat_dict['partition'].items()}
    # We start with source
    sources, values = list(stat_dict_sources.keys()), list(stat_dict_sources.values())
    lists = sorted(zip(*[values, sources]))
    fig = plt.figure(figsize = (30, 5))
    values, sources = list(zip(*lists))
    
    # creating the bar plot
    plt.pie(values, labels = sources, frame = False, autopct='%1.1f%%')
    my_circle = plt.Circle( (0,0), 0.7, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    # plt.title("Distribution of biographies per source")
    plt.savefig('data/plots/train_overview.png')


if __name__ == "__main__":
    print('Reading the data...')
    all_data = read_json("../data/json", mode = 'dir')
    print('Starting statistics collection')
    stat_dict = collect_statistics(all_data, split = 'train')
    visualize_stat_dict(stat_dict)