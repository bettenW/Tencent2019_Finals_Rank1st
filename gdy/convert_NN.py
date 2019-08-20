import os
import pandas as pd
import numpy as np
import random
import json
import gc
from sklearn import preprocessing
from tqdm import tqdm
import time
np.random.seed(2019)
random.seed(2019)

#convert NN data format
def norm(train_df,test_df,features):   
    df=pd.concat([train_df,test_df])[features]
    scaler = preprocessing.QuantileTransformer(random_state=0)
    scaler.fit(df[features]) 
    train_df[features]=scaler.transform(train_df[features])
    test_df[features]=scaler.transform(test_df[features])



    
for path1,path2,flag in [('preprocess_data/train_dev.pkl','preprocess_data/dev.pkl','dev'),('preprocess_data/train.pkl','preprocess_data/test.pkl','test')]:
        print(path1,path2)
        train_df=pd.read_pickle(path1)
        test_df=pd.read_pickle(path2)
        train_df=train_df.sort_values(by=['aid','day'])
        train_df=train_df.drop_duplicates(keep='last')
        train_df['bid_feature']=train_df['bid']
        test_df['bid_feature']=test_df['bid']
        train_df['request_conts']=train_df['request_cont']
        test_df['request_conts']=test_df['request_cont']

        print(train_df.shape,test_df.shape) 

        float_features=['aid_predict_imp_2', 'good_id_imp_median', 
             'good_id_imp_min', 'aid_imp_std', 'aid_history_positive_num',
             'good_id_imp_max', 'aid_imp_mean', 'good_id_1_negative_num', 'advertiser_predict_imp_2', 
             'aid_2_positive_num', 'good_id_2_positive_num', 'good_id_1_positive_num', 'aid_imp_max',
             'good_id_history_positive_num', 'good_id_predict_imp_1', 'good_id_2_negative_num',
             'aid_kfold_history_imp', 'good_id_imp_std', 'aid_1_positive_num', 'advertiser_imp_max', 
             'aid_imp_median', 'advertiser_history_positive_num', 'advertiser_imp_median', 'aid_2_negative_num',
             'advertiser_predict_imp_1', 'aid_kfold_imp', 'advertiser_2_negative_num', 'advertiser_1_negative_num', 
             'advertiser_imp_min', 'advertiser_history_negative_num', 'create_timestamp', 
             'good_id_history_negative_num', 'aid_predict_imp_1', 'aid_predict_imp', 'advertiser_2_positive_num',
             'good_id_predict_imp_2', 'good_id_predict_imp', 'aid_history_negative_num', 'advertiser_imp_mean', 
             'aid_1_negative_num', 'advertiser_1_positive_num', 'request_cont', 'advertiser_predict_imp',
             'aid_imp_min', 'advertiser_imp_std', 'aid_3_positive_num', 'aid_3_negative_num', 'aid_predict_imp_3',
             'good_id_3_positive_num', 'good_id_3_negative_num', 'good_id_predict_imp_3', 'advertiser_3_positive_num',
             'advertiser_3_negative_num', 'advertiser_predict_imp_3', 'aid_4_positive_num', 'aid_4_negative_num', 
             'aid_predict_imp_4', 'good_id_4_positive_num', 'good_id_4_negative_num', 'good_id_predict_imp_4', 
             'advertiser_4_positive_num', 'advertiser_4_negative_num', 'advertiser_predict_imp_4',
                        
                        
'aid_aid_rate_kfold_mean_imps', 'aid_aid_rate_kfold_median_imps', 'aid_goods_id_rate_kfold_mean_imps', 'aid_goods_id_rate_kfold_median_imps', 'aid_account_id_rate_kfold_mean_imps', 'aid_account_id_rate_kfold_median_imps', 'aid_aid_cnts_kfold_mean', 'aid_aid_cnts_kfold_median', 'aid_goods_id_cnts_kfold_mean', 'aid_goods_id_cnts_kfold_median', 'aid_account_id_cnts_kfold_mean', 'aid_account_id_cnts_kfold_median', 'aid_aid_sums_kfold_mean', 'aid_aid_sums_kfold_median', 'aid_aid_sums_kfold_mean_negs', 'aid_aid_sums_kfold_median_negs', 'aid_goods_id_sums_kfold_mean', 'aid_goods_id_sums_kfold_median', 'aid_goods_id_sums_kfold_mean_negs', 'aid_goods_id_sums_kfold_median_negs', 'aid_account_id_sums_kfold_mean', 'aid_account_id_sums_kfold_median', 'aid_account_id_sums_kfold_mean_negs', 'aid_account_id_sums_kfold_median_negs', 'aid_aid_negs_kfold_mean', 'aid_aid_negs_kfold_median', 'aid_aid_negs_kfold_mean_sums', 'aid_aid_negs_kfold_median_sums', 'aid_goods_id_negs_kfold_mean', 'aid_goods_id_negs_kfold_median', 'aid_goods_id_negs_kfold_mean_sums', 'aid_goods_id_negs_kfold_median_sums', 'aid_account_id_negs_kfold_mean', 'aid_account_id_negs_kfold_median', 'aid_account_id_negs_kfold_mean_sums', 'aid_account_id_negs_kfold_median_sums',
'goods_id_aid_rate_kfold_mean_imps', 'goods_id_aid_rate_kfold_median_imps', 
'goods_id_goods_id_rate_kfold_mean_imps', 'goods_id_goods_id_rate_kfold_median_imps', 
'goods_id_account_id_rate_kfold_mean_imps', 'goods_id_account_id_rate_kfold_median_imps',
'goods_id_aid_cnts_kfold_mean', 'goods_id_aid_cnts_kfold_median', 'goods_id_goods_id_cnts_kfold_mean', 
'goods_id_goods_id_cnts_kfold_median', 'goods_id_account_id_cnts_kfold_mean', 'goods_id_account_id_cnts_kfold_median',
'goods_id_aid_sums_kfold_mean', 'goods_id_aid_sums_kfold_median', 'goods_id_aid_sums_kfold_mean_negs', 
'goods_id_aid_sums_kfold_median_negs', 'goods_id_goods_id_sums_kfold_mean', 'goods_id_goods_id_sums_kfold_median', 
'goods_id_goods_id_sums_kfold_mean_negs', 'goods_id_goods_id_sums_kfold_median_negs', 
'goods_id_account_id_sums_kfold_mean', 'goods_id_account_id_sums_kfold_median', 
'goods_id_account_id_sums_kfold_mean_negs', 'goods_id_account_id_sums_kfold_median_negs',
'goods_id_aid_negs_kfold_mean', 'goods_id_aid_negs_kfold_median', 'goods_id_aid_negs_kfold_mean_sums', 
'goods_id_aid_negs_kfold_median_sums', 'goods_id_goods_id_negs_kfold_mean', 'goods_id_goods_id_negs_kfold_median',
'goods_id_goods_id_negs_kfold_mean_sums', 'goods_id_goods_id_negs_kfold_median_sums',
'goods_id_account_id_negs_kfold_mean', 'goods_id_account_id_negs_kfold_median',
'goods_id_account_id_negs_kfold_mean_sums', 'goods_id_account_id_negs_kfold_median_sums',
'account_id_aid_rate_kfold_mean_imps', 'account_id_aid_rate_kfold_median_imps',
'account_id_goods_id_rate_kfold_mean_imps', 'account_id_goods_id_rate_kfold_median_imps', 
'account_id_account_id_rate_kfold_mean_imps', 'account_id_account_id_rate_kfold_median_imps', 'account_id_aid_cnts_kfold_mean', 'account_id_aid_cnts_kfold_median', 'account_id_goods_id_cnts_kfold_mean', 'account_id_goods_id_cnts_kfold_median', 'account_id_account_id_cnts_kfold_mean', 'account_id_account_id_cnts_kfold_median', 'account_id_aid_sums_kfold_mean', 'account_id_aid_sums_kfold_median', 'account_id_aid_sums_kfold_mean_negs', 'account_id_aid_sums_kfold_median_negs', 'account_id_goods_id_sums_kfold_mean', 'account_id_goods_id_sums_kfold_median', 'account_id_goods_id_sums_kfold_mean_negs', 'account_id_goods_id_sums_kfold_median_negs', 'account_id_account_id_sums_kfold_mean', 'account_id_account_id_sums_kfold_median', 'account_id_account_id_sums_kfold_mean_negs', 'account_id_account_id_sums_kfold_median_negs', 'account_id_aid_negs_kfold_mean', 'account_id_aid_negs_kfold_median', 'account_id_aid_negs_kfold_mean_sums', 'account_id_aid_negs_kfold_median_sums', 'account_id_goods_id_negs_kfold_mean', 'account_id_goods_id_negs_kfold_median', 'account_id_goods_id_negs_kfold_mean_sums', 'account_id_goods_id_negs_kfold_median_sums', 'account_id_account_id_negs_kfold_mean', 'account_id_account_id_negs_kfold_median', 'account_id_account_id_negs_kfold_mean_sums', 'account_id_account_id_negs_kfold_median_sums', 'industry_id_aid_rate_kfold_mean_imps', 'industry_id_aid_rate_kfold_median_imps', 'industry_id_goods_id_rate_kfold_mean_imps', 'industry_id_goods_id_rate_kfold_median_imps', 'industry_id_account_id_rate_kfold_mean_imps', 'industry_id_account_id_rate_kfold_median_imps', 'industry_id_aid_cnts_kfold_mean', 'industry_id_aid_cnts_kfold_median', 'industry_id_goods_id_cnts_kfold_mean', 'industry_id_goods_id_cnts_kfold_median', 'industry_id_account_id_cnts_kfold_mean', 'industry_id_account_id_cnts_kfold_median', 'industry_id_aid_sums_kfold_mean', 'industry_id_aid_sums_kfold_median', 'industry_id_aid_sums_kfold_mean_negs', 'industry_id_aid_sums_kfold_median_negs', 'industry_id_goods_id_sums_kfold_mean', 'industry_id_goods_id_sums_kfold_median', 'industry_id_goods_id_sums_kfold_mean_negs', 'industry_id_goods_id_sums_kfold_median_negs', 'industry_id_account_id_sums_kfold_mean', 'industry_id_account_id_sums_kfold_median', 'industry_id_account_id_sums_kfold_mean_negs', 'industry_id_account_id_sums_kfold_median_negs', 'industry_id_aid_negs_kfold_mean', 'industry_id_aid_negs_kfold_median', 'industry_id_aid_negs_kfold_mean_sums', 'industry_id_aid_negs_kfold_median_sums', 'industry_id_goods_id_negs_kfold_mean', 'industry_id_goods_id_negs_kfold_median', 'industry_id_goods_id_negs_kfold_mean_sums', 'industry_id_goods_id_negs_kfold_median_sums', 'industry_id_account_id_negs_kfold_mean', 'industry_id_account_id_negs_kfold_median', 'industry_id_account_id_negs_kfold_mean_sums', 'industry_id_account_id_negs_kfold_median_sums', 'aid_size_aid_rate_kfold_mean_imps', 'aid_size_aid_rate_kfold_median_imps', 'aid_size_goods_id_rate_kfold_mean_imps', 'aid_size_goods_id_rate_kfold_median_imps', 'aid_size_account_id_rate_kfold_mean_imps', 'aid_size_account_id_rate_kfold_median_imps', 'aid_size_aid_cnts_kfold_mean', 'aid_size_aid_cnts_kfold_median', 'aid_size_goods_id_cnts_kfold_mean', 'aid_size_goods_id_cnts_kfold_median', 'aid_size_account_id_cnts_kfold_mean', 'aid_size_account_id_cnts_kfold_median', 'aid_size_aid_sums_kfold_mean', 'aid_size_aid_sums_kfold_median', 'aid_size_aid_sums_kfold_mean_negs', 'aid_size_aid_sums_kfold_median_negs', 'aid_size_goods_id_sums_kfold_mean', 'aid_size_goods_id_sums_kfold_median', 'aid_size_goods_id_sums_kfold_mean_negs', 'aid_size_goods_id_sums_kfold_median_negs', 'aid_size_account_id_sums_kfold_mean', 'aid_size_account_id_sums_kfold_median', 'aid_size_account_id_sums_kfold_mean_negs', 'aid_size_account_id_sums_kfold_median_negs', 'aid_size_aid_negs_kfold_mean', 'aid_size_aid_negs_kfold_median', 'aid_size_aid_negs_kfold_mean_sums', 'aid_size_aid_negs_kfold_median_sums', 'aid_size_goods_id_negs_kfold_mean', 'aid_size_goods_id_negs_kfold_median', 'aid_size_goods_id_negs_kfold_mean_sums', 'aid_size_goods_id_negs_kfold_median_sums', 'aid_size_account_id_negs_kfold_mean', 'aid_size_account_id_negs_kfold_median', 'aid_size_account_id_negs_kfold_mean_sums', 'aid_size_account_id_negs_kfold_median_sums',

            
                       ]


        columns = ['aid_kfold_history_rate', 'aid_1_rate',  'aid_history_rate', 
                   'aid_kfold_rate', 'aid_2_rate', 'aid_3_rate', 'aid_4_rate', 'aid_predict_imp_2', 
                   'aid_history_positive_num', 'aid_2_positive_num', 'aid_kfold_history_imp', 'aid_1_positive_num', 
                   'aid_kfold_imp', 'aid_predict_imp_1', 'aid_predict_imp', 'aid_3_positive_num', 'aid_predict_imp_3', 
                   'aid_4_positive_num', 'aid_predict_imp_4', ]
        for feat in ['advertiser','good_id','good_type','ad_type_id','ad_size']:
            for f in columns:
                if 'rate' in f:
                    pass
                else:
                    float_features.append(feat + '_' + f + '_day_mean')
                    float_features.append(feat + '_' + f + '_day_median')  
                  
        train_df[float_features]=train_df[float_features].fillna(-99999)
        test_df[float_features]=test_df[float_features].fillna(-99999)
        norm(train_df,test_df,float_features)
        
        print(train_df[float_features])
        
        k=1
        train_df=train_df.sample(frac=1)
        train=[(path2[:-4]+'_NN.pkl',test_df)]
        for i in range(k):
            train.append((path1[:-4]+'_NN_'+str(i)+'.pkl',train_df.iloc[int(i/k*len(train_df)):int((i+1)/k*len(train_df))]))
        del train_df
        gc.collect()
        for file,temp in train:
            print(file,temp.shape)               
            temp=temp.fillna(0)           
            temp.to_pickle(file)
            del temp
            gc.collect()