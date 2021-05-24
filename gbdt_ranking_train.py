import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle


# %% 推荐- GBDT训练

class GBDTTRainer():
    def __init__(self):
        self.eval_dir='./processing_data/eval_data/'
        pass

    def load_data(self):
        data_list = os.listdir('./processing_data/eval_data/')
        data_list.sort()

        for i, date in enumerate(data_list):
            if test_date in date:
                break
        train_paths = [os.path.join(eval_dir, i) for i in data_list[i - window:i]]
        train_data = pd.concat([pd.read_csv(path, index_col=0) for path in train_paths])
                                
        train_data = train_data.fillna(0)
        train_data = train_data[train_data['sim_val'] != 0]
        x_train_data, y_train_data = train_data.iloc[:, :-1], train_data.iloc[:, -1]

    def train(test_date, window=15, eval_dir='./processing_data/eval_data/'):
        self.model = GradientBoostingClassifier(random_state=1)
        print('training model ...')
        self.model.fit(x_train_data, y_train_data)
    
    def save_model(self, model_path=''):
        # 保存模型
        with open('./model/ranking_gbdt.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def run(self):
        self.load_data()
        self.train()
        self.save_model()
