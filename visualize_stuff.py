'''This script can be used to visualize dictionary data'''

import json
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import spacy
from spacy.tokens import Doc
from ner_systems import Clean_Model_Output
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

        with open(self.path, 'r') as json_file:
            for line in json_file:
                yield json.loads(line) # Change this to yield because of amendability downline
        

    def from_tsv(self) -> list: # For the training data
        # word\tlabel --> [{text_entities: [{text: word, label: label}]}]
        import csv
        import sys
        ret = [{'text_entities': [], 'text_tokens': [], 'text_sents': []}] # We need to add text sents
        try:
            with open(self.path, encoding='windows-1252') as file:
                # csv.field_size_limit(sys.maxsize)
                # infile = csv.reader(file, delimiter='\t', quotechar='|')
                infile = file.readlines()
                s = [] # Container to hold sentence objects
                for line in infile:
                    row = line.strip().split('\t')
                    if row == ['']:
                        row = []
                    if len(row) > 1: # We skip sentence endings here, so we need to remember to see if we want it back later
                        word = row[0]
                        label = row[1]
                        info = {'text': word, 'label': label}
                        for dct in ret:
                            dct['text_entities'].append(info)
                            dct['text_tokens'].append(word)
                            s.append(word) 
                    elif not row and s: # If not row we clear s
                        dct['text_sents'].append(s)
                        s = []
                dct['text_sents'].append(s)
        except UnicodeDecodeError:
               with open(self.path, encoding='utf-8') as file:
                csv.field_size_limit(sys.maxsize)
                infile = csv.reader(file, delimiter='\t', quotechar='|')
                s = [] # Container to hold sentence objects
                for row in infile:
                    if row: # We skip sentence endings here, so we need to remember to see if we want it back later
                        word = row[0].strip('\\r')
                        label = row[1].strip('\\r')
                        info = {'text': word, 'label': label}
                        for dct in ret:
                            dct['text_entities'].append(info)
                            dct['text_tokens'].append(word)
                            s.append(word) 
                    elif not row and s: # If not row we clear s
                        dct['text_sents'].append(s)
                        s = []
                dct['text_sents'].append(s)
        return ret
    
    def as_eval_file(self):
        """_summary_
        When reading a file for evaluation purposes only. 
        Returns:
            tuple: Will instead of a bio object, return a tuple with preds and golds.
        """
        import csv
        import sys
        preds, golds = [], []
        with open(self.path, encoding = 'utf-8') as file:
            csv.field_size_limit(sys.maxsize)
            infile = csv.reader(file, delimiter='\t', quotechar='|')
            for row in infile:
                assert len(row) == 3, f'Can not evaluate on row {row}'
                if row:
                    pred = row[1]
                    gold = row[2]
                    preds.append(pred)
                    golds.append(gold)
        return preds, golds

    def log_for_graphing_losses(self, key = 'losses'):
        cleaned_training_losses = {key: []}
        with open(self.path) as f:
            inf = f.readlines()
            start_grabbing = False
            epoch_number = 0
            epoch_holder = [] # Container to store losses with respect to the epoch
            for index, line in enumerate(inf):
                loss = line.split()[-1]
                if line.startswith('INFO:root:======== Epoch'):
                    if not line.startswith("INFO:root:======== Epoch 1 / 8 ========"):
                        epoch_number+=1
                        dct = {f"epoch_{epoch_number}": epoch_holder}
                        cleaned_training_losses[key].append(dct)
                        epoch_holder = []
                if loss == 'INFO:root:Training...':
                    start_grabbing = True
                if loss == 'INFO:root:':
                    start_grabbing = False
                if start_grabbing:
                    if not 'nan' in loss:
                        try:
                            # Convert from string and scientific notation
                            epoch_holder.append(float(loss.rstrip('.')))
                        except ValueError: # This happens when we find info line
                            continue
                    else:
                        continue
            epoch_number+=1
            dct = {f"epoch_{epoch_number}": epoch_holder}
            cleaned_training_losses[key].append(dct)
        print('\nInspection of cleaned training losses')
        for lst in cleaned_training_losses.values():
            for epoch in lst:
                print(epoch.keys())
        return cleaned_training_losses

    def from_json(self):
        with open(self.path) as f:
            a = json.load(f)
        return a



class Counter:
    """Will count instances of key in obj and return a new dictionary object
    Example Counter(bio_obj, 'named_entities') -> {PER: 113}"""
    
    def __init__(self, obj, key):
        self.obj = obj
        self.key = key
    
    def from_bio_obj(self, exclude = False):
        '''Takes bio_obj and returns a counter dict
        Exlude is a variable for which you can exlude certain values'''
        # This currently just works for the 'text_entities' key
        counter_obj = dict()
        # assert type(self.obj[self.key] == list), 'We need to iterate through a list of strings :('
        for i, lst in enumerate(self.obj):
            for dct in lst[self.key]:
                if dct['label'] in counter_obj:
                    if not dct['label'] == exclude:
                        counter_obj[dct['label']] += 1
                else:
                    if not dct['label'] == exclude:
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
        "Will print n most popular locations in bio_object"
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
        "Will print n most popular times in bio_object"
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
        "Will print n most popular miscs in bio_object"
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

    def count_bionet_sources(self):
        pass
            
    
    def concatenate_bios(self):
        '''Will interpret the bio-obj and concatenate sequences of BI tags to form 'whole' tag representations
        :return: a new bio object with only whole tagged items, 
        in which case word represents the FULL entity, and label the tag for that entity'''
        obj = self.obj # I want to test if new variable helps with runtime
        storage_obj = {'text_entities': [{}], "text_tokens": []} #{'text_entities': [word: tag, label: label]}
        window = 10 # We assume no entity will have more than 10 tokens
        for dct in obj:
            for i, entity in enumerate(dct['text_entities']):
                tl = [] # We join the objects in here
                labs = []
                if entity['label'].startswith('B'): # We know it is the onset of an entity
                    tl.append(entity['text'])
                    for l in range(window):
                        try:
                            if dct['text_entities'][i+l]['label'].startswith('I'):
                                txt = dct['text_entities'][i+l]['text']
                                lab = entity['label']
                                if txt:
                                    tl.append(txt)
                                    labs.append(lab)
                        except IndexError:
                            continue # We are at the end of the file
                    entity = ' '.join(tl)
                    if labs:
                        l = labs[0][2:]
                        ret = {'word': entity, 'label': l}
                        storage_obj['text_entities'].append(ret)
                        storage_obj['text_tokens'].append(entity)
        return [storage_obj]
            

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
        score = intersection / obj1_size*100
        print(f"The score we will give is {score}")
        return score
        
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
    def __init__(self, obj: dict, out_path: str):
        self.path = out_path
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
        '''Will take dict to generate a wordcloud
        :return: a wordcloud'''
        wc = WordCloud(background_color="white",width=1000,height=1000, 
                        max_words=200,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(self.obj)
        wc.to_file(self.path)

    def as_circular_barplot(self):
        '''Will take dict to create a circular barplot'''
        ax = plt.subplot(111, polar=True)
        plt.axis('off')

        # Draw bars
        bars = ax.bar(
            x=angles, 
            height=heights, 
            width=width, 
            bottom=lowerLimit,
            linewidth=2, 
            edgecolor="white",
            color="#61a4b2",
        )

        # little space between the bar and the label
        labelPadding = 4

        # Add labels
        for bar, angle, height, label in zip(bars,angles, heights, df["Name"]):

            # Labels are rotated. Rotation must be specified in degrees :(
            rotation = np.rad2deg(angle)

            # Flip some labels upside down
            alignment = ""
            if angle >= np.pi/2 and angle < 3*np.pi/2:
                alignment = "right"
                rotation = rotation + 180
            else: 
                alignment = "left"

            # Finally add the labels
            ax.text(
                x=angle, 
                y=lowerLimit + bar.get_height() + labelPadding, 
                s=label, 
                ha=alignment, 
                va='center', 
                rotation=rotation, 
                rotation_mode="anchor")

    def as_barplot(self):
        '''Will make a barplot from the counter object'''
        data = self.obj
        # Sorting the dictionary object by value
        sorted_obj = sorted(self.obj.items(), key = lambda x:x[1]) # List of tuples [(John, 22), (Alex, 23)]
        values = []
        sources = []
        for tup in sorted_obj:
            sources.append(tup[0])
            values.append(tup[1])

        total = sum(values)
        percentage = [(val/total)*100 for val in values]

        x_pos = np.arange(len(sources))

        # Create bars
        # Modify the default style of matplotlib
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['lightcoral'])
        plt.bar(x_pos, values, width = 0.5)
        # Create names on the x-axis
        for i, v in enumerate(percentage):
            plt.text(i+.5, values[i]+.9, f"pct = {v:.2f}%", fontsize=7, ha='right', color='black', va = 'bottom')
        plt.xticks(x_pos, sources, color='black', rotation = 90)
        plt.yticks(color='black')
        plt.subplots_adjust(bottom=.2, top = .9)
        plt.title('Named Entities in Train Partition')
        plt.savefig(self.path)


def train_comparisons(bio):
    '''Takes bio object of Bionet file and will compare to all items in trainset'''
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

def create_wordclouds_for_all_train_files(dir):
    """Takes a path to a directory with trainfiles and will create wordclouds for all of them"""
    loc = '../data/train/AITrainingset1.0/Data'
    for path in os.listdir(loc):
        if path.endswith('.txt') and not path.startswith('.') and not path.startswith('vocab'):
            a = Read(f"{loc}/{path}")
            bio_obj_1 = a.from_tsv()
            
            b = Interpret(bio_obj_1)
            ranked = b.count_word_rank()

            vis = Visualize(ranked, f"wordclouds/WordCloud_{path.rstrip('.txt')}.png")
            vis.as_wordcloud()


def create_popular_entity_wordclouds_for_all_bios(path, ent = 'PER'):
    loc = path
    if path.endswith("jsonl"):
        a = Read(loc)
        bio_obj_1 = a.from_file()
        b = Interpret(bio_obj_1)
        print(b)
        bio_obj_2 = b.concatenate_bios()
        ranked = Interpret(bio_obj_2).popular_persons(n = 10)
        print(ranked)
        # ranked = c.count_word_rank()

        vis = Visualize(ranked, f"../wordclouds/WordCloud_Entities_{path.rstrip('.jsonl')}.png")
        vis.as_wordcloud()

def create_popular_entity_wordclouds_for_all_train_files(path, ent = 'PER'):
    loc = path
    for path in os.listdir(loc):
        if path.endswith('.txt') and not path.startswith('.') and not path.startswith('vocab'):
            a = Read(f"{loc}/{path}")
            bio_obj_1 = a.from_tsv()
            b = Interpret(bio_obj_1)
            bio_obj_2 = b.concatenate_bios()
            print(bio_obj_2)
            
            c = Interpret(bio_obj_2)
            ranked = c.count_word_rank()

            vis = Visualize(ranked, f"wordclouds/WordCloud_{path.rstrip('.txt')}.png")
            vis.as_wordcloud()

def generate_barplot_from_scores(bio):
    '''Takes a loaded bio obj and compares to items in train set to generate scores
    , will draw a barplot to a file'''
    gendict = dict() # This is the dict we use to draw the barplot from
    loc = '../data/train/AITrainingset1.0/Data'
    for path in os.listdir(loc):
        if path.endswith('.txt') and not path.startswith('.') and not path.startswith('vocab'):
            a = Read(f"{loc}/{path}")
            bio_obj_1 = a.from_tsv()
            
            b = Interpret(bio_obj_1)
            word_count = b.count_words()

            c = Compare(bio, bio_obj_1)
            ret = c.overlap(lemmatize = False)

            if not path in gendict:
                gendict[path] = ret

            print(f"In the set {path}")
            print(f'The word count is {word_count}')
            print(ret)
    writepath = 'wordclouds/barplot_with_overlap_scores.png'
    vis = Visualize(gendict, writepath)
    vis.as_barplot()

def visualize_loss_logs(loss_dicts: list, writepath = 'line_graphs/test_plot.png'):
    assert type(loss_dicts) == list, 'Please put it in a list first :)'
    n = 0
    keys = []
    for dct in loss_dicts:
        print(str(dct.keys()))
        label = str(dct.keys())
        x_values = []
        y_values = []
        n+=1
        x=0
        for key, value in dct.items():
            print('\n', key, '\n')
            for epoch in value:
                x += 1
                all_values = epoch[f'epoch_{x}']
                try:
                    y = sum(all_values, 0)/len(all_values)
                    x_values.append(x)
                    y_values.append(y)
                except ZeroDivisionError:
                    pass
        
        # Plotting
        palette = plt.get_cmap('Set1')
        # plt.subplot(3,3, n)
        
        # Not ticks everywhere
        if n in range(9):
            plt.tick_params(labelbottom='off')
        if n not in [1,2,4,6,8] :
            plt.tick_params(labelleft='off')
        print('Created x and y values: plotting')
        print(x_values)
        print(y_values)
        plt.plot(x_values, y_values, color = palette(n), label = label[11:-2])   
        plt.xlim(1,8)
        plt.ylim(0,.35)
    plt.title('Training losses for all models')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.savefig(writepath)

def graph_validation_losses_from_json(loss_list):
    '''Loss list should contain the jsons with losses'''
    # Containers
    x_values = []
    y_values = []
    n = 0
    writepath = 'line_graphs/validation_loss_plot.png'
    # Opening all the files:
    for path in loss_list:
        n+=1
        label = path.split('/')[-2]
        x_values = []
        y_values = []
        with open(path) as f:
            dct = json.load(f)
            for x, y in enumerate(dct['losses'], start = 1):
                x_values.append(x)
                y_values.append(y)
                print(x, y)
        palette = plt.get_cmap('Set1')
        plt.plot(x_values, y_values, color = palette(n), label = label[13:])   
    plt.xlim(1,8)
    plt.ylim(0,.43)
    plt.title('Validation losses for all models')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.savefig(writepath)

def visualize_losses_from_model_directory(trained_models_path = 'saved_models/all_models_felix'):
    models = []
    results = []
    trained_models_path = trained_models_path
    for d in os.listdir(trained_models_path):
        if not d.startswith('.'):
            model_name = '_'.join(d.split('_')[-2:])
            models.append(model_name)
            print(f'Going to {model_name}')
            print(d)
            ndir = os.listdir(f"{trained_models_path}/{d}")
            print(ndir)
            for file in ndir:
                if file.endswith('8.log') and not file.startswith('.'):
                    loss_file = f"{trained_models_path}/{d}/{file}"
            print('Reading file:', loss_file)
            res = Read(loss_file).log_for_graphing_losses(model_name)
            results.append(res)
    visualize_loss_logs(results)

def visualize_confusion_matrix_from_results(path):
    predictions, gold = Read(path).as_eval_file()
    predictions, gold = Clean_Model_Output(predictions, gold).clean_bertje()
    labels = sorted(list(set(predictions)))
    matrix = confusion_matrix(predictions, gold, labels = labels)
    print(matrix)
    fig = plt.figure()  # Create a new figure
    ax = sns.heatmap(matrix, annot=True, xticklabels=labels, yticklabels=labels,
                     fmt='g', cmap='Blues', vmin=0, vmax=900)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    newpath = f"matrices/{path.split('/')[-1].rstrip('.tsv')}_cf_matrix.png"
    fig.savefig(newpath, dpi=400)

if __name__ == '__main__':
    paths = ["model_results/stanza_test_SA_cleaned.tsv", "model_results/stanza_test_NHA_cleaned.tsv", "model_results/stanza_biographynet_test_A_gold_cleaned.tsv.tsv"]
    for path in paths:
        visualize_confusion_matrix_from_results(path)

    # run_on_partitions = ["qualitative_eval/biography_selection_middle_dutch.conll", "qualitative_eval/biography_selection_modern_dutch.conll"]
    # for path in run_on_partitions:
    #     bio_obj = Read(path).from_tsv()
    #     count = Counter(bio_obj, 'text_entities').from_bio_obj()
    #     print(path)
    #     print(count)
#This is for graphing losses

    # paths = []
    # trained_models_path = 'saved_models/all_models_felix'
    # for d in os.listdir(trained_models_path):
    #     if not d.startswith('.'):
    #         ndir = os.listdir(f"{trained_models_path}/{d}")
    #         for file in ndir:
    #             print(file)
    #             if file.startswith('Losses_Dev') and not file.startswith('.'):
    #                 loss_file = f"{trained_models_path}/{d}/{file}"
    #                 paths.append(loss_file)
    # graph_validation_losses_from_json(paths)

    # p = 'saved_models/all_models_felix/saved_models_bertje_1234500/BERT_TokenClassifier_train_8.log'




    # paths = ["../data/test/cleaned/biographynet_test_A_gold_cleaned.tsv"]
    # for path in paths:
    #     out_path = f"wordclouds/{path.split('/')[-1].rstrip('.txt')}_wordcloud.png"
    #     bio_obj = Read(path).from_tsv()
    #     new_bio = Interpret(bio_obj).concatenate_bios()
    #     answer = Interpret(new_bio).count_word_rank()
    #     done = Visualize(answer, out_path).as_wordcloud()
    #     print('Files made')


# Other useful scripts 

    # Count for each label
    # paths = ["../data/test/cleaned/biographynet_test_A_gold_cleaned.tsv"]
    # for path in paths:
    #     bio_obj = Read(path).from_tsv()
    #     ans = Counter(bio_obj, 'text_entities').from_bio_obj()
    #     cleaned = {a:b for a,b in ans.items() if 'LOC' in a or 'PER' in a}
    #     print(cleaned)

    # Wordclouds
    # for path in paths:
    #     out_path = f"wordclouds/{path.split('/')[-1].rstrip('.txt')}_wordcloud.png"
    #     bio_obj = Read(path).from_tsv()
    #     new_bio = Interpret(bio_obj).concatenate_bios()
    #     answer = Interpret(new_bio).count_word_rank()
    #     done = Visualize(answer, out_path).as_wordcloud()
    #     print('Files made')
