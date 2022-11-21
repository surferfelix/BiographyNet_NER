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