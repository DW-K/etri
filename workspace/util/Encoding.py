import os

import pandas as pd
import numpy as np

encoding_path = r'../util/encoding'


class user_label_encoder:
    def __init__(self):
        self.path = encoding_path
        # self.label_list = [l.split(".")[0] for l in os.listdir(self.path)]
        self.label_list = ['action', 'actionOption', 'condition', 'conditionSub1Option', 'conditionSub2Option', 'place']

    def get_label_code(self, label):
        if label in self.label_list:
            print(os.listdir(f'{self.path}'))
            code_df = pd.read_table(f'{self.path}/{label}.txt')
            return code_df
        else:
            return False

    def get_label_size(self, label):
        code_df = self.get_label_code(label)
        return code_df.shape[0]

    def get_label(self):
        return self.label_list
