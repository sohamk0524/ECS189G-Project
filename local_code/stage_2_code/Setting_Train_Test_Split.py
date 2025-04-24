'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        # print("Train X:", len(loaded_data["train"]["X"]))
        # print("Train y:", len(loaded_data["train"]["y"]))
        # print("Test X:", len(loaded_data["test"]["X"]))
        # print("Test y:", len(loaded_data["test"]["y"]))


        #X_train, X_test, y_train, y_test = train_test_split(loaded_data['X'], loaded_data['y'], test_size = 0.33)
        X_train, X_test, y_train, y_test = loaded_data["train"]["X"],loaded_data["test"]["X"],loaded_data["train"]["y"], loaded_data["test"]["y"]

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        