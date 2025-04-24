'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import csv
import os


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data for stage 2...')
        final_obj = {"train": {}, "test": {}}
        files = os.listdir(self.dataset_source_folder_path)
        for file in files:

            #print(f"file: {file}")
            if ".csv" in file:
                X = []
                y = []
                file_path = os.path.join(self.dataset_source_folder_path, file)
                #print(file_path)
                
                f = open(file_path, 'r')
                reader = csv.reader(f)

                for row in reader:
                    elements= [(int)(x) for x in row]
                    y.append((int)(elements[0]))
                    X.append(elements[1:])
                f.close()
                if "train" in file:
                    final_obj["train"] = {'X':X, 'y': y}
                else:
                    final_obj["test"] = {'X':X, 'y': y}
                
                    
        return final_obj
    