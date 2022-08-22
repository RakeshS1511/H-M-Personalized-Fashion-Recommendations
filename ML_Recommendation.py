# -*- coding: utf-8 -*-


import pandas as pd


from collections import Counter

transactions = pd.read_csv('D:/VIEH INTERNSHIP H&M/venv/transactions_train.csv')
sample_submission=pd.read_csv('D:/VIEH INTERNSHIP H&M/venv/sample_submission.csv')

transactions.head()

sample_submission.head()
 
# Keeping only transactions from 2020-05-01 since the data is too large for Kaggle cluster
transactions = transactions[transactions["t_dat"]>='2020-07-01']

# Considering each article_id as str
transactions['article_id'] = transactions['article_id'].astype(str)

# Splitting train and test as 30%
test_set = 0.3
train_set = (1-test_set)
split = int(transactions.shape[0]*train_set)

transactions_train = transactions.sort_values('t_dat').iloc[:split]
transactions_test = transactions.sort_values('t_dat').iloc[split:]
# print("Train set size :", transactions_train.shape[0])
# print("Test set size :", transactions_test.shape[0])

class BestSellersRecsys():
    """FIXME"""
    def __init__(self):
        """FIXME"""
        pass
    
    def fit(self, X):
        """FIXME"""
        self.X_occur = Counter(list(X['article_id']))
        self.best_sellers_10 = [article_id for (article_id, occ) in self.X_occur.most_common(10)]
        return self
    
    def predict(self, basket, k=10):
        """FIXME"""
        if k==10:
            return self.best_sellers_10
        else:
            best_sellers = [article_id for (article_id, occ) in self.X_occur.most_common(k)]
            return best_sellers

bs_recsys = BestSellersRecsys()
bs_recsys.fit(transactions_train)

bs_recsys.predict([])

from math import sqrt

class CosineSimCoOccurenceRecsys():
    """FIXME"""
    def __init__(self):
        """FIXME"""
        pass
    
    def _get_l_of_two_from_l(self, l):
        """FIXME"""
        # Dropping duplicates to avoid having more co-occur than occur
        l = list(set(l))
        if len(l)>1:
            l_of_two = []
            for i in range(0,len(l)):
                for j in range(i+1,len(l)):
                    if int(l[i])<int(l[j]):
                        l_of_two.append(f"{l[i]}|{l[j]}")
                    else:
                        l_of_two.append(f"{l[j]}|{l[i]}")
            return l_of_two
        return None
    
    def _get_frequency(self, article_id, counter_occur_article_id):
        """FIXME"""
        return counter_occur_article_id[article_id]
    
    def _get_co_occ_frequency(self, article_id_a, article_id_b, counter_co_occ_article_id):
        """FIXME"""
        if int (article_id_a) < int(article_id_b):
            str_co_occ = article_id_a + '|' + article_id_b
        else:
            str_co_occ = article_id_b + '|' + article_id_a
        return counter_co_occ_article_id[str_co_occ]
    
    def _get_cosine_sim(self, freq_a, freq_b, co_occ_freq):
        """FIXME"""
        return (co_occ_freq/sqrt(freq_a*freq_b))
        
    def fit(self, X):
        """FIXME"""        
        # Getting baskets
        # FIXME 
        X_grouped = X.head(10000).groupby(['t_dat', 'customer_id'])['article_id'].apply(list).reset_index()
        
        # Getting frequencies of article_ids
        l_baskets_len2 = list(X_grouped[X_grouped['article_id'].map(len) >1]['article_id'])
        l_baskets_len2_flat_list = [x for xs in l_baskets_len2 for x in xs]
        counter_occur_article_id = Counter(l_baskets_len2_flat_list)

        # Getting co-occur frequencies of article_ids
        l_co_occur_article_id = []
        for basket in l_baskets_len2:
            l_of_two = self._get_l_of_two_from_l(basket)
            if l_of_two:
                l_co_occur_article_id += l_of_two

        counter_co_occ_article_id = Counter(l_co_occur_article_id)
        
        # Creating pandas dataframe
        l_article_id_a = [i.split('|', 1)[0] for i in l_co_occur_article_id]
        l_article_id_b = [i.split('|', 1)[1] for i in l_co_occur_article_id]
        df_cosine = pd.DataFrame()
        df_cosine['article_id_a'] = l_article_id_a
        df_cosine['article_id_b'] = l_article_id_b
        df_cosine['freq_a'] = df_cosine['article_id_a'].apply(lambda x : self._get_frequency(x, counter_occur_article_id))
        df_cosine['freq_b'] = df_cosine['article_id_b'].apply(lambda x : self._get_frequency(x, counter_occur_article_id))
        df_cosine['co_occ_freq'] = df_cosine.apply(lambda x: self._get_co_occ_frequency(x.article_id_a, x.article_id_b, counter_co_occ_article_id), axis=1)
        df_cosine['cosine_sim'] = df_cosine.apply(lambda x: self._get_cosine_sim(x.freq_a, x.freq_b, x.co_occ_freq), axis=1)
        df_cosine = df_cosine.sort_values('cosine_sim', ascending=False)
        self.df_cosine = df_cosine
        return self
    
    def predict(self, basket, k=10, strat='mean'):
        """FIXME"""
        df = self.df_cosine[(self.df_cosine['article_id_a'].isin(basket)) & (~self.df_cosine['article_id_b'].isin(basket))]
        if strat=='mean':
            result = list(df[['article_id_b', 'cosine_sim']].groupby(['article_id_b']).mean().reset_index()['article_id_b'])[:k]
        elif strat=='max':
            result = list(df[['article_id_b', 'cosine_sim']].groupby(['article_id_b']).max().reset_index()['article_id_b'])[:k]
        else:
            result = []
        return result

cs_recsys = CosineSimCoOccurenceRecsys().fit(transactions_train)
res = []
res = cs_recsys.predict(['610776083'])
print(res)
