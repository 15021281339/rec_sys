import os
import pandas as pd
import numpy as np
from src.predict.gbdt_ranking_predict import *
import pickle
from src.train.cf_train import *
from src.train.gbdt_ranking_train import *


def run():
    generator = CFCandidate()
    generator.run()
    # train('2020-02-29', window=15, eval_dir='./processing_data/eval_data/')
    gbdt_trainer = GBDTTRainer()
    predict(test_date='2020-02-29')

if __name__ == '__main__':
    run()
