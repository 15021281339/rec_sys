import os
import pandas as pd
import numpy as np
import pickle
import datetime
import scipy
from scipy.sparse import coo_matrix
import warnings

warnings.filterwarnings("ignore")


# %%推荐 - 数据预处理
class ProcessData:

    def __init__(self, path, n_days=10):
        # to save feat
        self.user_set_map = {}
        self.item_set_map = {}
        # to save id
        self.item_map = {}
        self.user_map = {}
        # to save item popular
        self.ItemPopular = {}
        self.Price_map = {}
        # read data
        self.data_user = pd.read_csv(os.path.join(path, 'users.csv'))
        self.data_item = pd.read_csv(os.path.join(path, 'items.csv'), usecols=range(3))
        self.data_buy = pd.read_csv(os.path.join(path, 'buys-63w.csv'))
        self.n_days = n_days

    def fill_na_trans(self):
        self.trans_time()
        self.data_item.fillna('None', inplace=True)
        self.data_user.fillna('None', inplace=True)
        self.data_buy.fillna('None', inplace=True)

    def trans_time(self):
        def time_func(x):
            date = datetime.datetime.utcfromtimestamp(int(x))
            return date.date()

        self.data_buy['order_date'] = self.data_buy['order_time'].apply(time_func)

    def save_map_id(self):
        self.fill_na_trans()
        items = self.data_buy['item_id'].unique().tolist()
        self.item_map = {k: v for v, k in enumerate(items)}
        users = self.data_buy['email'].unique().tolist()
        self.user_map = {k: v for v, k in enumerate(users)}
        self.data_buy['user_id'] = self.data_buy['email'].map(self.user_map)
        self.data_buy['item_id'] = self.data_buy['item_id'].map(self.item_map)

        # saving buy map
        with open('./processing_data/item.pkl', 'wb') as f:
            pickle.dump(self.item_map, f)

        with open('./processing_data/user.pkl', 'wb') as f:
            pickle.dump(self.user_map, f)

        # get items data map
        self.data_item['item_id'] = self.data_item['item_id'].map(self.item_map)

        self.data_item = self.data_item.dropna()
        self.data_item.index = self.data_item.pop('item_id')

        for i in self.data_item.columns:
            feat_map = self.data_item[i].unique().tolist()
            map_dict = {k: v for v, k in enumerate(feat_map)}
            with open('./processing_data/%s.pkl' % i, 'wb') as f:
                pickle.dump(map_dict, f)

            self.data_item[i] = self.data_item[i].map(map_dict)
            self.item_set_map[i] = self.data_item[i].to_dict()

        with open('./processing_data/item_data.pkl', 'wb') as f:
            pickle.dump(self.item_set_map, f)

        # get users data map
        self.data_user['user_id'] = self.data_user['email'].map(self.user_map)

        self.data_user = self.data_user.dropna()
        self.data_user.index = self.data_user.pop('user_id')
        for i in self.data_user.columns[1:]:
            feat_map = self.data_user[i].unique().tolist()
            map_dict = {k: v for v, k in enumerate(feat_map)}
            with open('./processing_data/%s.pkl' % i, 'wb') as f:
                pickle.dump(map_dict, f)

            self.data_user[i] = self.data_user[i].map(map_dict)
            self.user_set_map[i] = self.data_user[i].to_dict()

        with open('./processing_data/user_data.pkl', 'wb') as f:
            pickle.dump(self.user_set_map, f)

    def popular_rank(self):
        date_generate = self.data_buy['order_date'].unique().tolist()
        date_generate.sort()
        for i in date_generate[self.n_days:]:
            item_popular = {}
            sub_df = self.data_buy[(self.data_buy["order_date"] >= i - datetime.timedelta(days=self.n_days)) &
                                   (self.data_buy["order_date"] < i)]
            items = sub_df['item_id'].tolist()
            for j in items:
                item_popular[j] = item_popular.get(j, 0) + 1
            pop_ser = pd.Series(item_popular)
            sorted_popular = pop_ser.sort_values(ascending=False)
            self.ItemPopular[i] = sorted_popular

        # item price map
        self.Price_map = self.data_buy.groupby('item_id')['price'].mean().to_dict()
        with open('./processing_data/price.pkl', 'wb') as f:
            pickle.dump(self.Price_map, f)

        # popular map
        with open('./processing_data/popular.pkl', 'wb') as f:
            pickle.dump(self.ItemPopular, f)

    def saving_date_data(self):
        date_generate = self.data_buy['order_date'].unique().tolist()
        date_generate.sort()
        if not os.path.exists('./processing_data/date_data/'):
            os.mkdir('./processing_data/date_data/')
        for i in date_generate:
            test_data = self.data_buy[self.data_buy["order_date"] == i]
            test_data.to_csv('./processing_data/date_data/%s.csv' % str(i))

    def save_cf_data(self):
        date_generate = self.data_buy['order_date'].unique().tolist()
        date_generate.sort()
        for i in date_generate[self.n_days:]:
            data = self.data_buy[(self.data_buy["order_date"] >= i - datetime.timedelta(days=self.n_days)) &
                                 (self.data_buy["order_date"] < i)]
            cf_df = data[['user_id', 'item_id']]
            cf_df['rate'] = 1
            cf_group = cf_df.groupby(['item_id', 'user_id'])['rate'].sum()
            row_ = cf_group.index.get_level_values('user_id').to_numpy()
            col_ = cf_group.index.get_level_values('item_id').to_numpy()
            val_ = cf_group.values
            cf_sparse = coo_matrix((val_, (row_, col_)),
                                   shape=(len(self.user_map), len(self.item_map)),  # saving cf data
                                   dtype=int)

            scipy.sparse.save_npz('./processing_data/sparse_%s.npz' % i, cf_sparse.tocsr())

    def call(self):
        print('saving mapping ....')
        self.save_map_id()
        print('saving cf data ...')
        self.save_cf_data()
        print('get popular by date....')
        self.popular_rank()
        print('saving data by date....')
        self.saving_date_data()
        print('processing has done')


# %%
if __name__ == '__main__':
    process_data = ProcessData('./data')
    process_data.call()
