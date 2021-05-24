import numpy as np
import scipy
import operator
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import datetime
import pickle


# %% 基于用户的CF-生成候选集

class UserCf:

    def __init__(self, target_user_id, base_mat, base_user_id, base_item_id, item_popular_map, top_k=50):
        self.base_user_map = {k: v for v, k in enumerate(base_user_id)}
        self.inv_user_map = {k: v for k, v in enumerate(base_user_id)}
        self.inv_item_map = {k: v for k, v in enumerate(base_item_id)}
        self.target_user_id = [self.base_user_map.get(i) for i in target_user_id]
        self.base_mat = base_mat[base_user_id, :][:, base_item_id]
        self.itemPopular_map = item_popular_map
        self.top_k = top_k
        self.rec_dict = {}
        self.sim_dict = {}

    def get_similarity_by_user(self):
        sim = cosine_similarity(self.base_mat[self.target_user_id, :],
                                self.base_mat, dense_output=False)
        # inplace 1
        sim.setdiag(0)
        return sim

    def get_top_k_item(self):
        pop_list = self.itemPopular_map.index.to_list()
        sim = self.get_similarity_by_user()
        item_score = (sim @ self.base_mat)
        item_l = item_score.tolil()
        datas, rows = item_l.data, item_l.rows
        for user_id, data, row in zip(self.target_user_id, datas, rows):
            if data:
                rec_num = len(data)
                if rec_num >= self.top_k:
                    data, row = zip(*sorted(zip(data, row), reverse=True)[:self.top_k])
                    self.rec_dict[self.inv_user_map[user_id]] = [self.inv_item_map[i] for i in row]
                    self.sim_dict[self.inv_user_map[user_id]] = list(data)
                else:
                    data, row = zip(*sorted(zip(data, row), reverse=True))
                    rec_item = [self.inv_item_map[i] for i in row]
                    self.rec_dict[self.inv_user_map[user_id]] = rec_item + pop_list[:self.top_k - rec_num]
                    self.sim_dict[self.inv_user_map[user_id]] = list(data) + [0] * (self.top_k - rec_num)

            else:
                self.rec_dict[self.inv_user_map[user_id]] = pop_list[:self.top_k]
                self.sim_dict[self.inv_user_map[user_id]] = [0] * self.top_k
        return self.rec_dict, self.sim_dict


# %% 编码映射

def mapping_cf_candidate(eval_df):

    with open('./processing_data/item_data.pkl', 'rb') as f:
        item_set_map = pickle.load(f)

    with open('./processing_data/user_data.pkl', 'rb') as f:
        user_set_map = pickle.load(f)

    with open('./processing_data/price.pkl', 'rb') as f:
        price_map = pickle.load(f)

    for item_key in item_set_map.keys():
        item_map = item_set_map[item_key]
        eval_df[item_key] = eval_df['item_id'].map(item_map)

    for user_key in user_set_map.keys():
        user_map = user_set_map[user_key]
        eval_df[user_key] = eval_df['user_id'].map(user_map)

    eval_df['price'] = eval_df['item_id'].map(price_map)
    return eval_df

    # eval_df.to_csv('./processing_data/eval_df.csv', index=False)


class CFCandidate:

    def __init__(self, base_dir='./processing_data/date_data'):
        data_list = os.listdir(base_dir)
        data_list.sort()
        # load pop
        with open('./processing_data/popular.pkl', 'rb') as f:
            pop_map = pickle.load(f)
        self.cf_mat = None
        self.item_pop = None
        self.pop_map = pop_map
        self.base_dir = base_dir
        self.data_list = data_list
        self.window = 10

    def generate_train_test_path(self, test_date):
        for i, date in enumerate(self.data_list):
            if test_date in date: break
        train_paths, test_path = [os.path.join(self.base_dir, i) for i in self.data_list[i - self.window:i]], \
                                 os.path.join(self.base_dir, self.data_list[i])
        return train_paths, test_path

    def get_cf_candidate(self, test_date):
        self.cf_mat = scipy.sparse.load_npz(os.path.join('./processing_data', 'sparse_%s.npz' % test_date))
        train_paths, test_path = self.generate_train_test_path(test_date)
        train_data, test_data = pd.concat([pd.read_csv(path, index_col=0) for path in train_paths]), \
                                pd.read_csv(test_path, index_col=0)
        base_data = pd.concat([train_data, test_data])
        self.item_pop = self.pop_map[datetime.datetime.strptime(test_date, '%Y-%m-%d').date()]
        # using cf alg
        test_user_id, test_item_id = test_data['user_id'].unique().tolist(), test_data['item_id'].unique().tolist()
        base_user_id, base_item_id = base_data['user_id'].unique().tolist(), base_data['item_id'].unique().tolist()
        model = UserCf(test_user_id, self.cf_mat, base_user_id, base_item_id, self.item_pop)
        return model.get_top_k_item(), test_data

    def trans_candidate_to_dataframe(self, rec_dict, sim_dict):
        # generate eval_df
        user_ids = rec_dict.keys()
        pop_map_dict = self.item_pop.to_dict()
        users_id, items_id, sims_val, pops_val = [], [], [], []
        for user_id in user_ids:
            items = rec_dict[user_id]
            items_id.extend(items)
            users_id.extend([user_id] * len(items))
            sims_val.extend(sim_dict[user_id])
            pops_val.extend([pop_map_dict.get(i) for i in items])
        eval_df = pd.DataFrame(index=range(len(users_id)), columns=['user_id', 'item_id', 'sim_val', 'pop_val'])
        eval_df['user_id'] = users_id
        eval_df['item_id'] = items_id
        eval_df['sim_val'] = sims_val
        eval_df['pop_val'] = pops_val
        return eval_df

    def call(self, begin_date='2020-02-01'):
        for i, date in enumerate(self.data_list):
            if begin_date in date: break
        test_list = [j.split('.')[0] for j in self.data_list[i:]]
        for test_date in test_list:
            (rec_dict, sim_dict), test_data = self.get_cf_candidate(test_date)
            print('cf alg has done')
            eval_df = self.trans_candidate_to_dataframe(rec_dict=rec_dict, sim_dict=sim_dict)
            print('trans to df has done')
            eval_df = self.mapping_cf_candidate(eval_df=eval_df)
            test_data['rate'] = 1
            eval_df = pd.merge(eval_df, test_data[['user_id', 'item_id', 'rate']], how='left')
            if not os.path.exists('./processing_data/eval_data/'): os.makedirs('./processing_data/eval_data/')
            # print('推准率: %s \n覆盖率: %s' % (eval_df['rate'].sum() / eval_df.shape[0],
            #                              eval_df['rate'].sum() / test_data.shape[0]))
            eval_df.to_csv('./processing_data/eval_data/%s.csv' % test_date)
            print('%s data has generated ... ' % test_date)

    def run(self):
        self.call()

# %%
# if __name__ == '__main__':
#     Cf_processing = CFCandidate()
#     Cf_processing.call()
