'''This script can be used to visualize dictionary data'''
class Read:
    import json
    import os
    '''Can read files from different objects (dir, single_file)'''
    def __init__(self, path):
        path = self.path
    
    def from_directory(self):
        pass

    def from_file(self):
        bio_obj = []
        with open(self.path, 'r') as json_file:
            for line in json_file:
                bio_obj.append(json.loads(line)) #Appending dict to list
        return bio_obj

class Visualize:
    import matplotlib.pyplot as plt
    def __init__(self, obj: dict, path: str):
        '''This class can take a dictionary with {str: int}
        and visualize it to your liking
        :type obj: dict
        :type path: string
        :path: path to write to'''
        self.obj = obj
        self.path = path
    
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
    path = ''
    f = Read(path)
    v = f.Visualize()
    


