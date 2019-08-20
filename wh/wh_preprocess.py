import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import sparse
from tqdm import tqdm
import time
import datetime
import os
import gc
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

path = '../data/'

# 静态数据
ad_static_feature = pd.read_table(path+'map_ad_static.out',
                                  names=['aid', 'create_time', 'account_id', 'goods_id', 'goods_type',
                                         'industry_id', 'aid_size'])
# 操作数据
ad_operation = pd.read_table(path+'final_map_bid_opt.out',
                                  names=['aid','update_time','op_type','objective','bid_type','bid'])


# 10-22号日志数据
logs_item=[]
for day in tqdm(range(20190410, 20190423),total=len(range(20190410, 20190423))):
    print('-'*50,day,'-'*50)
    file = path+'track_log_{}.out'.format(day)
    log_df = pd.read_table(file, names=['aid_request','request_time','uid','aid_location','aid_info'])
    print('-'*50,day,'-'*50)
    for item in tqdm(log_df[['aid_location','aid_info']].values, total=len(log_df)):
        for it in item[-1].split(';'):
            temp=list(item[:-1])
            it=it.split(',')
            temp.append(int(day))
            temp.append(int(it[0]))
            temp.append(int(it[5]))
            temp.append(int(it[6]))
            if temp[-1]==1:
                logs_item.append(temp)
            else:
                logs_item.append(temp)
    del log_df
    gc.collect()
    
logs=pd.DataFrame(logs_item)
del logs_item
gc.collect()
logs.columns=['aid_location','day','aid','fliter','isExp']
logs['day'] =  logs['day'].values % 100

# 构造训练集
data = logs[['day','aid','isExp']].copy()
data = data.groupby(['aid','day'])['isExp'].agg({'sum'}).reset_index()
data.columns = ['aid','day','label']
data['label'] = data['label'].fillna(0)
print('init data:', data.shape)

# 合并静态数据
ad_operation = ad_operation.drop_duplicates(subset=['aid'], keep='first')
ad_static_feature = ad_static_feature.merge(ad_operation[['aid','op_type']], on='aid', how='left')
data = data.merge(ad_static_feature, on='aid', how='left')
data['op_type'] = data['op_type'].fillna(-999)
data = data.loc[(data.op_type!=-999)]
data = data.reset_index()
del data['index']

columns = ['aid','goods_id','account_id','aid_size','industry_id','goods_type']
logs = logs.merge(ad_static_feature[columns], on='aid', how='left')

# 构造基本特征
for col in ['aid','goods_id','account_id']:
    print(col)
    result = logs.groupby([col,'day'], as_index=False)['isExp'].agg({
        col+'_cnts'      : 'count',
        col+'_sums'      : 'sum',
        col+'_rate'      : 'mean'
        })
    result[col+'_negs'] = result[col+'_cnts'] - result[col+'_sums']
    data = data.merge(result, how='left', on=[col,'day'])
##############
# 保存训练数据
##############
data.to_csv('preprocess_data/data.csv'   , index=False)



# 23号待预估测试集
atest_logs = pd.read_table(path+'final_select_test_request.out',names=['aid','request_set'])
logs = []
for item in tqdm(atest_logs[['aid','request_set']].values,total=len(atest_logs)):
    for it in item[-1].split('|'):
        temp=list(item[:-1])
        it=it.split(',')
        temp.append(int(it[0]))
        temp.append(float(it[1]))
        logs.append(temp)
del atest_logs
gc.collect()
logs=pd.DataFrame(logs)
logs.columns=['aid','aid1','aid_location']

# 日志数据增加类别
logs = logs.merge(ad_static_feature[['aid','industry_id','goods_type','goods_id','account_id','aid_size']], on=['aid'], how='left') 
# 测试集样本
atest = pd.read_table(path+'test_sample_bid.out',names=['sample_id', 'aid', 'objective', 'bid_type', 'bid'])
# 增加类别特征
atest = atest.merge(ad_static_feature[['aid','industry_id','goods_type','goods_id','account_id','aid_size']], on=['aid'], how='left') 

columns = ['aid','goods_id','account_id','aid_size','industry_id','goods_type']
logs = logs.reset_index()
for col in columns:
    result = logs.groupby([col], as_index=False)['index'].agg({
        col+'_cnts'      : 'count'
        })
    atest = atest.merge(result, on=[col], how='left')
#################
# 保存23号测试集
#################
atest.to_csv('preprocess_data/Atest.csv', index=False)


# 24号待预估测试集
btest_logs = pd.read_table(path+'Btest_select_request_20190424.out',names=['aid','request_set'])
logs = []
for item in tqdm(btest_logs[['aid','request_set']].values,total=len(btest_logs)):
    for it in item[-1].split('|'):
        temp=list(item[:-1])
        it=it.split(',')
        temp.append(int(it[0]))
        temp.append(float(it[1]))
        logs.append(temp)
del btest_logs
gc.collect()
logs=pd.DataFrame(logs)
logs.columns=['aid','aid1','aid_location']

# 日志数据增加类别
logs = logs.merge(ad_static_feature[['aid','industry_id','goods_type','goods_id','account_id','aid_size']], on=['aid'], how='left') 

# 测试集样本
btest = pd.read_table(path+'Btest_sample_bid.out',names=['sample_id', 'aid', 'objective', 'bid_type', 'bid'])
# 增加类别特征
btest = btest.merge(ad_static_feature[['aid','industry_id','goods_type','goods_id','account_id','aid_size']], on=['aid'], how='left') 

columns = ['aid','goods_id','account_id','aid_size','industry_id','goods_type']
logs = logs.reset_index()
for col in columns:
    result = logs.groupby([col], as_index=False)['index'].agg({
        col+'_cnts'      : 'count'
        })
    btest = btest.merge(result, on=[col], how='left')
#################
# 保存24号测试集
#################
btest.to_csv('preprocess_data/Btest.csv', index=False)


# 23号非待预估广告的请求日志和竞价队列,第一条非过滤的为曝光
log_df = pd.read_table(path+'test_tracklog_20190423.last.out', names=['aid_request','request_time','uid','aid_location','aid_info'])
items=[]
for item in tqdm(log_df['aid_info'].values,total=len(log_df)):
    temp=[]
    find=False
    for its in item.split(';'):
        it=its.split(',')
        if int(it[-1])==0 and find is False:
            temp.append(','.join(it+['1']))
            find=True
        else:
            temp.append(','.join(it+['0']))

    items.append(';'.join(temp))

log_df['aid_info']=items

logs_item = []
for item in tqdm(log_df[['aid_location','aid_info']].values, total=len(log_df)):
    for it in item[-1].split(';'):
        temp=list(item[:-1])
        it=it.split(',')
        temp.append(int(23))
        temp.append(int(it[0]))
        temp.append(int(it[5]))
        temp.append(int(it[6]))
        if temp[-1]==1:
            logs_item.append(temp)
        else:
            logs_item.append(temp)
del log_df
gc.collect()
logs=pd.DataFrame(logs_item)
del logs_item
gc.collect()
logs.columns=['aid_location','day','aid','fliter','isExp']


data = logs[['day','aid','isExp']].copy()
data = data.groupby(['aid','day'])['isExp'].agg({'sum'}).reset_index()
data.columns = ['aid','day','label']
data['label'] = data['label'].fillna(0)
print('init data:', data.shape)

data = data.merge(ad_static_feature, on='aid', how='left')

data = data.reset_index()
del data['index']
print('data:',data.shape)
print('data:',data.columns)

# logs merge ad_static_feature
columns = ['aid','goods_id','account_id','aid_size','industry_id','goods_type']
logs = logs.merge(ad_static_feature[columns], on='aid', how='left')
print('logs:',logs.shape)
print('logs:',logs.columns)

# day isExp
for col in ['aid','goods_id','account_id']:
    print(col)
    result = logs.groupby([col,'day'], as_index=False)['isExp'].agg({
        col+'_cnts'      : 'count',
        col+'_sums'      : 'sum',
        col+'_rate'      : 'mean'
        })
    result[col+'_negs'] = result[col+'_cnts'] - result[col+'_sums']
    data = data.merge(result, how='left', on=[col,'day'])

#################
# 保存23号日志数据
#################
data.to_csv('preprocess_data/atest_track.csv', index=False)