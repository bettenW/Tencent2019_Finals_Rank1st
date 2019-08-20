import numpy as np
import pandas as pd
import ctrNet
import tensorflow as tf
from src import misc_utils as utils
import os
import gc
from sklearn import metrics
from sklearn import preprocessing
import random
np.random.seed(2019)



single_features=['aid', 'objective', 'bid_type','advertiser',
'good_id', 'good_type', 'ad_type_id','ad_size',
'aid_predict_imp_2', 'aid_history_positive_num',
'good_id_1_negative_num', 'advertiser_predict_imp_2', 'aid_2_positive_num',
'good_id_2_positive_num',  'good_id_1_positive_num',  
'good_id_history_positive_num', 'good_id_predict_imp_1', 'good_id_2_negative_num', 
'aid_kfold_history_imp', 'aid_1_positive_num', 'advertiser_history_positive_num',
'aid_2_negative_num', 'advertiser_predict_imp_1', 'aid_kfold_imp', 'advertiser_2_negative_num',  
'advertiser_1_negative_num', 'advertiser_history_negative_num', 'create_timestamp', 
'good_id_history_negative_num','aid_predict_imp_1', 'aid_predict_imp', 'advertiser_2_positive_num', 'good_id_predict_imp_2',
'good_id_predict_imp', 'aid_history_negative_num','aid_1_negative_num',
'advertiser_1_positive_num', 'request_cont', 'advertiser_predict_imp', 
'aid_3_positive_num', 'aid_3_negative_num',  'aid_predict_imp_3', 
'good_id_3_positive_num', 'good_id_3_negative_num', 
'good_id_predict_imp_3', 'advertiser_3_positive_num', 'advertiser_3_negative_num',
 'advertiser_predict_imp_3', 'aid_4_positive_num', 'aid_4_negative_num',
 'aid_predict_imp_4', 'good_id_4_positive_num', 'good_id_4_negative_num',
 'good_id_predict_imp_4', 'advertiser_4_positive_num', 'advertiser_4_negative_num',
'advertiser_predict_imp_4'
]

cross_features=['aid', 'objective', 'bid_type','advertiser',
'good_id', 'good_type', 'ad_type_id','ad_size',          
                ]

multi_features=['ad_size']




sequence_features=None
dense_features=['aid_kfold_history_rate','good_id_history_rate','aid_1_rate','aid_2_rate','aid_history_rate',
 'aid_kfold_rate','advertiser_history_rate','aid_2_rate','good_id_2_rate','advertiser_2_rate','aid_3_rate',
 'good_id_3_rate','advertiser_3_rate','aid_4_rate','good_id_4_rate','advertiser_4_rate']
#dense_features+=['uid_embedding_aid_16_'+str(i) for i in range(16)]+['uid_embedding_good_id_16_'+str(i) for i in range(16)]+['uid_embedding_advertiser_16_'+str(i) for i in range(16)]

kv_features=['aid_predict_imp_2', 'good_id_imp_median', 
             'good_id_imp_min', 'aid_history_positive_num',
             'good_id_imp_max',  'good_id_1_negative_num', 'advertiser_predict_imp_2', 
             'aid_2_positive_num', 'good_id_2_positive_num', 'good_id_1_positive_num', 
             'good_id_history_positive_num', 'good_id_predict_imp_1', 'good_id_2_negative_num',
             'aid_kfold_history_imp', 'good_id_imp_std', 'aid_1_positive_num', 'advertiser_imp_max', 
              'advertiser_history_positive_num', 'advertiser_imp_median', 'aid_2_negative_num',
             'advertiser_predict_imp_1', 'aid_kfold_imp', 'advertiser_2_negative_num', 'advertiser_1_negative_num', 
             'advertiser_imp_min', 'advertiser_history_negative_num', 'create_timestamp', 
             'good_id_history_negative_num', 'aid_predict_imp_1', 'aid_predict_imp', 'advertiser_2_positive_num',
             'good_id_predict_imp_2', 'good_id_predict_imp', 'aid_history_negative_num', 'advertiser_imp_mean', 
             'aid_1_negative_num', 'advertiser_1_positive_num', 'request_cont', 'advertiser_predict_imp',
              'advertiser_imp_std', 'aid_3_positive_num', 'aid_3_negative_num', 'aid_predict_imp_3',
             'good_id_3_positive_num', 'good_id_3_negative_num', 'good_id_predict_imp_3', 'advertiser_3_positive_num',
             'advertiser_3_negative_num', 'advertiser_predict_imp_3', 'aid_4_positive_num', 'aid_4_negative_num', 
             'aid_predict_imp_4', 'good_id_4_positive_num', 'good_id_4_negative_num', 'good_id_predict_imp_4', 
             'advertiser_4_positive_num', 'advertiser_4_negative_num', 'advertiser_predict_imp_4',
'aid_kfold_history_rate','good_id_history_rate','aid_1_rate','aid_2_rate','aid_history_rate',
 'aid_kfold_rate','advertiser_history_rate','aid_2_rate','good_id_2_rate','advertiser_2_rate','aid_3_rate',
 'good_id_3_rate','advertiser_3_rate','aid_4_rate','good_id_4_rate','advertiser_4_rate',
             
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
            kv_features.append(feat + '_' + f + '_day_mean')
            kv_features.append(feat + '_' + f + '_day_median')  
            
            
hparam=tf.contrib.training.HParams(
            model='xdeepfm',
            norm=True,
            batch_norm_decay=0.9,
            hidden_size=[1024,512],
            dense_hidden_size=[300],
            cross_layer_sizes=[128,128],
            k=8,
            single_k=8,
            num_units=64,
            num_layer=1,
            encoder_type='uni',
            max_length=100,
            cross_hash_num=int(5e6),
            single_hash_num=int(5e6),
            multi_hash_num=int(1e6),
            sequence_hash_num=int(1e4),
            batch_size=128,
            infer_batch_size=2**10,
            optimizer="adam",
            dropout=0,
            kv_batch_num=20,
            learning_rate=0.0002,
            num_display_steps=1000,
            num_eval_steps=1000,
            epoch=1, #don't modify
            metric='score',
            activation=['relu','relu','relu'],
            init_method='tnormal',
            cross_activation='relu',
            init_value=0.001,
            single_features=single_features,
            cross_features=cross_features,
            multi_features=multi_features,
            sequence_features=sequence_features,
            dense_features=dense_features,
            kv_features=kv_features,
            label='imp',
            model_name="xdeepfm",
            bid='bid_feature',
            use_bid=False,
            bias=1)
utils.print_hparams(hparam)




test=pd.read_pickle('preprocess_data/test_NN.pkl').reset_index()
dev=pd.read_pickle('preprocess_data/dev_NN.pkl').reset_index()
train=pd.read_pickle('preprocess_data/train_NN_0.pkl').reset_index()
train_dev=pd.read_pickle('preprocess_data/train_dev_NN_0.pkl').reset_index()
train=train[train['day']!=23]
train_dev=train_dev[train_dev['day']!=21]
train['gold_imp']=train['imp']
train_dev['gold_imp']=train_dev['imp']
dev['gold_imp']=dev['imp']
for df in [test,dev,train,train_dev]:
    df['cont']=1
train['imp']=train[['imp','cont']].apply(lambda x:np.log(x[0]+hparam.bias)/x[1],axis=1)
train_dev['imp']=train_dev[['imp','cont']].apply(lambda x:np.log(x[0]+hparam.bias)/x[1],axis=1)




print(dev.shape)
print(train_dev.shape)
scaler=preprocessing.MinMaxScaler(feature_range=(0,8))
scaler.fit(train_dev[['imp']]) 
hparam.train_scaler=scaler
hparam.test_scaler=scaler
print("*"*80)
model=ctrNet.build_model(hparam)
model.train(train_dev,dev)
dev_preds=np.zeros(len(dev))
dev_preds=model.infer(dev)
dev_preds=(np.exp(dev_preds*np.array(dev['cont'].values))-hparam.bias)                  
print(np.mean(dev_preds))
print("*"*80)

print(test.shape)
print(train.shape)
scaler=preprocessing.MinMaxScaler(feature_range=(0,8))
scaler.fit(train[['imp']]) 
hparam.train_scaler=scaler
hparam.test_scaler=scaler



test_preds=np.zeros(len(test))
scores=[]

for i in range(5):
    print("Fold",i)
    model=ctrNet.build_model(hparam)
    model.train(train,None)
    test_preds+=model.infer(test)/5
    print(np.mean((np.exp(test_preds*5/(i+1)*np.array(test['cont'].values)))-hparam.bias))
    del model
    gc.collect()



test_preds=(np.exp(test_preds*np.array(test['cont'].values)) -hparam.bias)  
  
print(dev_preds.mean())
print(test_preds.mean())

dev['nn_preds'] = dev_preds
dev_fea=dev[['aid','bid_feature','gold','imp','nn_preds']]
test['nn_preds'] = test_preds
test_fea=test[['aid','nn_preds']]
test_fea.to_csv('stacking/nn_pred_{}_test.csv'.format(hparam.model_name),index=False)
dev_fea.to_csv('stacking/nn_pred_{}_dev.csv'.format(hparam.model_name),index=False)



test['preds']=pd.read_csv('stacking/nn_pred_{}_test.csv'.format(hparam.model_name))['nn_preds']
test['rank']=test[['aid', 'bid']].groupby('aid')['bid'].apply(lambda row: pd.Series(dict(zip(row.index, row.rank()))))-1
print(test['preds'].mean())
test['preds']=test['preds'].apply(lambda x: 0 if x<0  else x)
print(test['preds'].mean())
test['preds']=test[['preds','request_conts']].apply(lambda x:min(x) ,axis=1)
test['preds']=test['preds'].apply(lambda x: round(x,4))
print(test['preds'].mean())
test[['id','preds']].to_csv('../submission/gdy_NN.csv',index=False,header=False)
print(test['preds'])
            
            
            