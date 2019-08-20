import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import sparse
import time
import datetime
import os
import gc
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")


path = './preprocess_data/'


# 考虑A榜测试集添加
atest = pd.read_csv(path + 'Atest.csv')
atest_label = pd.read_csv('../submission/A.csv', header=None)
atest_label.columns = ['sample_id','label']
atest = atest.merge(atest_label, on='sample_id', how='left')
atest = atest.drop_duplicates(subset=['aid'], keep='last')
atest['label'] = atest['label'].apply(round)
# 测试集
test = pd.read_csv(path + 'Btest.csv')

# 训练集
data = pd.read_csv(path + 'data.csv')
data = data.loc[(data['aid_cnts']>50)|(data['label']!=0)]
data = data.reset_index()
del data['index']
print('data:',data.shape)

# 训练集扩充,添加23号非待预估广告
atest_track = pd.read_csv(path + 'atest_track.csv')
atest_track = atest_track.loc[(atest_track['aid_cnts']>50)]
data = pd.concat([data, atest_track], axis=0, ignore_index=True)

# 训练集扩充,添加23号待预估广告
# aid_cnts
atest['aid_sums'] = atest['label'].values
atest['aid_rate'] = (atest['aid_sums'].values+1) / (atest['aid_cnts'].values+1)
atest['aid_negs'] = atest['aid_cnts'].values - atest['aid_sums'].values
temp = data.drop_duplicates(subset=['aid'], keep='last')
# goods_id_cnts
temp = data.drop_duplicates(subset=['goods_id'], keep='last')
atest = atest.merge(temp[['goods_id','goods_id_rate','goods_id_sums','goods_id_negs']], on='goods_id', how='left')
# account_id_cnts
temp = data.drop_duplicates(subset=['account_id'], keep='last')
atest = atest.merge(temp[['account_id','account_id_rate','account_id_sums','account_id_negs']], on='account_id', how='left')

atest['day'] = 23
print('data:',data.shape)
data = pd.concat([data, atest], axis=0, ignore_index=True)
print('data:',data.shape)


# label encoder
columns = ['goods_id','account_id','aid_size','industry_id','goods_type']
tmp = pd.concat([data[columns],test[columns]], axis=0, ignore_index=True)
for f in columns:
    print('-'*10,f,'-'*10)
    tmp[f].fillna(-999,inplace=True)
    data[f]    = data[f].fillna(-999)
    test[f]    = test[f].fillna(-999)
    
    data[f]    =    data[f].map(dict(zip(tmp[f].unique(), range(0, tmp[f].nunique()))))
    test[f]    =    test[f].map(dict(zip(tmp[f].unique(), range(0, tmp[f].nunique()))))
    tmp[f]     =     tmp[f].map(dict(zip(tmp[f].unique(), range(0, tmp[f].nunique()))))
print('data:',data.shape)
print('test:',test.shape)


# 五折构造统计特征
def get_kfold_features(data_df_, test_df_):
    
    data_df = data_df_.copy()
    test_df = test_df_.copy()
    folds = KFold(n_splits=5, shuffle=True, random_state=2019)

    data_df['fold'] = None
    for fold_,(trn_idx,val_idx) in enumerate(folds.split(data_df,data_df)):
        data_df.loc[val_idx, 'fold'] = fold_

    kfold_features1 = []
    kfold_features2 = []
    
    print('二阶交叉...')
    for feat in ['goods_id','account_id','industry_id','aid_size']:

        nums_columns = ['aid_rate','goods_id_rate','account_id_rate',
                        'aid_cnts','goods_id_cnts','account_id_cnts',
                        'aid_sums','goods_id_sums','account_id_sums',
                        'aid_negs','goods_id_negs','account_id_negs']

        for f in nums_columns:
            if feat not in f:
                colname1 = feat + '_' + f + '_kfold_mean'
                colname2 = feat + '_' + f + '_kfold_median'
                print(feat,f,' mean/median...')
                kfold_features1.append(colname1)
                kfold_features1.append(colname2)

                data_df[colname1] = None
                data_df[colname2] = None

                # train
                for fold_,(trn_idx,val_idx) in enumerate(folds.split(data_df,data_df)):
                    Log_trn     = data_df.iloc[trn_idx]
                    # mean
                    order_label = Log_trn.groupby([feat])[f].mean()
                    tmp         = data_df.loc[data_df.fold==fold_,[feat]]
                    data_df.loc[data_df.fold==fold_, colname1] = tmp[feat].map(order_label)
                    # median
                    order_label = Log_trn.groupby([feat])[f].median()
                    tmp         = data_df.loc[data_df.fold==fold_,[feat]]
                    data_df.loc[data_df.fold==fold_, colname2] = tmp[feat].map(order_label)
                # test
                test_df[colname1] = None
                test_df[colname2] = None
                order_label   = data_df.groupby([feat])[f].mean()
                test_df[colname1] = test_df[feat].map(order_label)
                order_label   = data_df.groupby([feat])[f].median()
                test_df[colname2] = test_df[feat].map(order_label)

                if 'rate' in colname1:
                    kfold_features1.append(colname1+'_imps')
                    kfold_features1.append(colname2+'_imps')
                    data_df[colname1+'_imps'] = data_df[colname1].values * data_df[f[:-5]+'_cnts'].values
                    data_df[colname2+'_imps'] = data_df[colname2].values * data_df[f[:-5]+'_cnts'].values  
                    test_df[colname1+'_imps'] = test_df[colname1].values * test_df[f[:-5]+'_cnts'].values
                    test_df[colname2+'_imps'] = test_df[colname2].values * test_df[f[:-5]+'_cnts'].values

                if 'sums' in colname1:
                    kfold_features1.append(colname1+'_negs')
                    kfold_features1.append(colname2+'_negs')
                    data_df[colname1+'_negs'] = data_df[f[:-5]+'_cnts'].values - data_df[colname1].values
                    data_df[colname2+'_negs'] = data_df[f[:-5]+'_cnts'].values - data_df[colname2].values
                    test_df[colname1+'_negs'] = test_df[f[:-5]+'_cnts'].values - test_df[colname1].values
                    test_df[colname2+'_negs'] = test_df[f[:-5]+'_cnts'].values - test_df[colname2].values

                if 'negs' in colname1:
                    kfold_features1.append(colname1+'_sums')
                    kfold_features1.append(colname2+'_sums')
                    data_df[colname1+'_sums'] = data_df[f[:-5]+'_cnts'].values - data_df[colname1].values
                    data_df[colname2+'_sums'] = data_df[f[:-5]+'_cnts'].values - data_df[colname2].values
                    test_df[colname1+'_sums'] = test_df[f[:-5]+'_cnts'].values - test_df[colname1].values
                    test_df[colname2+'_sums'] = test_df[f[:-5]+'_cnts'].values - test_df[colname2].values
                   
    print('三阶交叉...')
    for feat1, feat2 in [('goods_id','account_id'),('goods_id','industry_id'),('goods_id','aid_size'),
                         ('account_id','industry_id'),('account_id','aid_size'),('industry_id','aid_size')]:
        colname = feat1+'_and_'+feat2
        data_df[colname] = data_df[feat1].astype(str).values + '_and_' + data_df[feat2].astype(str).values
        test_df[colname] = test_df[feat1].astype(str).values + '_and_' + test_df[feat2].astype(str).values
        for f in nums_columns:
            if (feat1 not in f) and (feat2 not in f):
                colname1 = colname+'_'+f+'_kfold_mean'
                colname2 = colname+'_'+f+'_kfold_median'
                print(feat1,feat2,f,' mean/median...')
                kfold_features2.append(colname1)
                kfold_features2.append(colname2)

                data_df[colname1] = None
                data_df[colname2] = None

                # train
                for fold_,(trn_idx,val_idx) in enumerate(folds.split(data_df,data_df)):
                    Log_trn     = data_df.iloc[trn_idx]
                    # mean
                    order_label = Log_trn.groupby([colname])[f].mean()
                    tmp         = data_df.loc[data_df.fold==fold_,[colname]]
                    data_df.loc[data_df.fold==fold_, colname1] = tmp[colname].map(order_label)
                    # median
                    order_label = Log_trn.groupby([colname])[f].median()
                    tmp         = data_df.loc[data_df.fold==fold_,[colname]]
                    data_df.loc[data_df.fold==fold_, colname2] = tmp[colname].map(order_label)
                # test
                test_df[colname1] = None
                test_df[colname2] = None
                order_label   = data_df.groupby([colname])[f].mean()
                test_df[colname1] = test_df[colname].map(order_label)
                order_label   = data_df.groupby([colname])[f].median()
                test_df[colname2] = test_df[colname].map(order_label)

                if 'rate' in colname1:
                    kfold_features2.append(colname1+'_imps')
                    kfold_features2.append(colname2+'_imps')
                    data_df[colname1+'_imps'] = data_df[colname1].values * data_df[f[:-5]+'_cnts'].values
                    data_df[colname2+'_imps'] = data_df[colname2].values * data_df[f[:-5]+'_cnts'].values  
                    test_df[colname1+'_imps'] = test_df[colname1].values * test_df[f[:-5]+'_cnts'].values
                    test_df[colname2+'_imps'] = test_df[colname2].values * test_df[f[:-5]+'_cnts'].values

                if 'sums' in colname1:
                    kfold_features2.append(colname1+'_negs')
                    kfold_features2.append(colname2+'_negs')
                    data_df[colname1+'_negs'] = data_df[f[:-5]+'_cnts'].values - data_df[colname1].values
                    data_df[colname2+'_negs'] = data_df[f[:-5]+'_cnts'].values - data_df[colname2].values
                    test_df[colname1+'_negs'] = test_df[f[:-5]+'_cnts'].values - test_df[colname1].values
                    test_df[colname2+'_negs'] = test_df[f[:-5]+'_cnts'].values - test_df[colname2].values

                if 'negs' in colname1:
                    kfold_features2.append(colname1+'_sums')
                    kfold_features2.append(colname2+'_sums')
                    data_df[colname1+'_sums'] = data_df[f[:-5]+'_cnts'].values - data_df[colname1].values
                    data_df[colname2+'_sums'] = data_df[f[:-5]+'_cnts'].values - data_df[colname2].values
                    test_df[colname1+'_sums'] = test_df[f[:-5]+'_cnts'].values - test_df[colname1].values
                    test_df[colname2+'_sums'] = test_df[f[:-5]+'_cnts'].values - test_df[colname2].values
    
    
    del data_df['fold']
    return data_df, test_df, kfold_features1, kfold_features2

data, test, kfold_features1, kfold_features2 = get_kfold_features(data, test)


# 合并测试集
test['day'] = 24
all_data = pd.concat([data, test], axis=0, ignore_index=True)
all_data['day'] = all_data['day'].astype(int)


# day交叉特征
def get_cross_day_features(all_data_df_, data_df_):
    
    all_data_df = all_data_df_.copy()
    data_df     = data_df_.copy()
    
    cross_day_columns1 = []
    cross_day_columns2 = []
    
    print('二阶day交叉特征...')
    for feat in ['goods_id','account_id','industry_id','aid_size']:
        for f in ['aid_rate','aid_cnts','aid_sums','aid_negs']:
            colname1 = feat + '_' + f + '_day_mean'
            colname2 = feat + '_' + f + '_day_median'
            print(feat,f,'-'*10)
            cross_day_columns1.append(colname1)
            cross_day_columns1.append(colname2)

            tmp  = data_df.groupby([feat,'day'], as_index=False)[f].agg({
                    colname1  : 'mean',
                    colname2 : 'median',
                    })
            all_data_df = all_data_df.merge(tmp, on=[feat,'day'], how='left')
            
    print('三阶day交叉特征...')
    for feat1, feat2 in [('goods_id','account_id'),('goods_id','industry_id'),('goods_id','aid_size'),
                         ('account_id','industry_id'),('account_id','aid_size'),('industry_id','aid_size')]:
        for f in ['aid_rate','aid_cnts','aid_sums','aid_negs']:
            colname1 = feat1+'_and_'+feat2+'_'+f+'_day_mean'
            colname2 = feat1+'_and_'+feat2+'_'+f+'_day_median'
            print(feat1,feat2,f,'-'*10)
            cross_day_columns2.append(colname1)
            cross_day_columns2.append(colname2)
            
            tmp  = data_df.groupby([feat1,feat2,'day'], as_index=False)[f].agg({
                            feat1+'_and_'+feat2+'_'+f+'_day_mean'   : 'mean',
                            feat1+'_and_'+feat2+'_'+f+'_day_median' : 'median',
                            })
            all_data_df = all_data_df.merge(tmp, on=[feat1,feat2,'day'], how='left')
    
    return all_data_df, cross_day_columns1, cross_day_columns2

all_data, cross_day_columns1, cross_day_columns2 = get_cross_day_features(all_data, data)


# 最近一天
def get_history_features(df_, mean_data, features, feat='aid', bf=0):
    df    = df_.copy()
    dt    = pd.DataFrame()
    bf = str(bf)

    cols = []
    for f in features:
        cols.append(f+'_'+bf)
   
    for d in range(11,24):
        p = mean_data.loc[mean_data['day']<=(d-int(bf)) ,  [feat] + features] 
        p.columns  = [feat] + cols
        p = p.drop_duplicates(subset=[feat], keep='last') 
           
        tmp = df.loc[df['day']==(d+1),['index',feat]]
        tmp = tmp.merge(p, on=feat, how='left')
        
        if dt.shape[0] == 0:
            dt = tmp
        else:
            dt = pd.concat([dt, tmp], axis=0, ignore_index=True)
        
    dt = dt[['index'] + cols]

    return dt, cols

def get_history_second_features(df_, mean_data, features, item=[], bf=0):
    df    = df_.copy()
    dt    = pd.DataFrame()
    bf = str(bf)

    cols = []
    for f in features:
        cols.append(f+'_'+bf)
    
    feat1 = item[0]
    feat2 = item[1]
    for d in range(11,24):
        p = mean_data.loc[mean_data['day']<=(d-int(bf)) ,  [feat1,feat2] + features]
        p.columns  = [feat1, feat2] + cols
        p = p.drop_duplicates(subset=[feat1, feat2], keep='last') 
           
        tmp = df.loc[df['day']==(d+1),['index',feat1,feat2]]
        tmp = tmp.merge(p, on=[feat1,feat2], how='left')
        
        if dt.shape[0] == 0:
            dt = tmp
        else:
            dt = pd.concat([dt, tmp], axis=0, ignore_index=True)
        
    dt = dt[['index'] + cols]

    return dt, cols

all_data = all_data.reset_index()

data = all_data[all_data.day!=24]

history_features   = []
history_features1_1  = []
history_features1_2  = []

for i in [0,1]:

    print('最近',i+1,'天...')
    # 当天竟胜率、请求数、请求成功、请求失败
    if i == 0:
        columns = ['aid_rate','goods_id_rate','account_id_rate',
                   'aid_cnts','goods_id_cnts','account_id_cnts',
                   'aid_sums','goods_id_sums','account_id_sums',
                   'aid_negs','goods_id_negs','account_id_negs'] + cross_day_columns1 + cross_day_columns2
    elif i == 1:
        all_columns = ['aid_rate','goods_id_rate','account_id_rate',
                       'aid_cnts','goods_id_cnts','account_id_cnts',
                       'aid_sums','goods_id_sums','account_id_sums',
                       'aid_negs','goods_id_negs','account_id_negs'] + cross_day_columns1 + cross_day_columns2
        columns = []
        for f in all_columns:
            if 'aid' in f:
                if ('rate' in f)|('sums' in f):
                    columns.append(f)
    for col in columns:
        print(col,'-'*10)
        # 一阶或二阶
        if 'and' not in col:
            feat = col.split('_')[0]
            if feat != 'aid':
                feat = col.split('_')[0] + '_' + col.split('_')[1]
            mean_data = data[[feat, col, 'day']]
            mean_data = mean_data.drop_duplicates(subset=[feat, 'day'], keep='first')
            bf,cols   = get_history_features(all_data, mean_data, [col], feat, bf=i)
        # 三阶
        elif 'and' in col:
            feat1 = col.split('_')[0] + '_' + col.split('_')[1]
            feat2 = col.split('_')[3] + '_' + col.split('_')[4]
            mean_data = data[[feat1, feat2, col, 'day']]
            mean_data = mean_data.drop_duplicates(subset=[feat1, feat2, 'day'], keep='first')
            bf,cols   = get_history_second_features(all_data, mean_data, [col], [feat1, feat2], bf=i)
                
        all_data  = all_data.merge(bf, on='index', how='left')
        
        if col not in cross_day_columns2:
            history_features1_1 = history_features1_1 + cols
        elif col in cross_day_columns2:
            history_features1_2 = history_features1_2 + cols

print(all_data.shape)
del all_data['index']
gc.collect()


history_features2_1 = []
history_features2_2 = []
for col in history_features1_1 + history_features1_2:
     
    # 一阶或二阶
    if 'and' not in col:        
        # 最近一天的竞争胜率 * 当天竞争数目
        if ('rate' in col)&('aid' in col):
            colname1 = col+'_imps'
            colname2 = 'aid_cnts'
            all_data[colname1] = all_data[col].values * all_data[colname2]
            history_features2_1.append(colname1)       
        elif ('rate' in col)&('goods_id' in col):
            colname1 = col+'_imps'
            colname2 = 'goods_id_cnts'
            all_data[colname1] = all_data[col].values * all_data[colname2]
            history_features2_1.append(colname1)
        elif ('rate' in col)&('account_id' in col):
            colname1 = col+'_imps'
            colname2 = 'account_id_cnts'
            all_data[colname1] = all_data[col].values * all_data[colname2]
            history_features2_1.append(colname1)

        # 当天竞争数目 - 最近一天的正样本   
        if ('sums' in col)&('aid' in col):
            colname1 = col+'_negs'
            colname2 = 'aid_cnts'
            all_data[colname1] = all_data[colname2] - all_data[col].values
            history_features2_1.append(colname1)
        elif ('sums' in col)&('goods_id' in col):
            colname1 = col+'_negs'
            colname2 = 'goods_id_cnts'
            all_data[colname1] = all_data[colname2] - all_data[col].values
            history_features2_1.append(colname1)
        elif ('sums' in col)&('account_id' in col):
            colname1 = col+'_negs'
            colname2 = 'account_id_cnts'
            all_data[colname1] = all_data[colname2] - all_data[col].values
            history_features2_1.append(colname1)

        # 当天竞争数目 - 最近一天的负样本
        if ('negs' in col)&('aid' in col):
            colname1 = col+'_sums'
            colname2 = 'aid_cnts'
            all_data[colname1] = all_data[colname2] - all_data[col].values
            history_features2_1.append(colname1)
        elif ('negs' in col)&('goods_id' in col):
            colname1 = col+'_sums'
            colname2 = 'goods_id_cnts'
            all_data[colname1] = all_data[colname2] - all_data[col].values
            history_features2_1.append(colname1)
        elif ('negs' in col)&('account_id' in col):
            colname1 = col+'_sums'
            colname2 = 'account_id_cnts'
            all_data[colname1] = all_data[colname2] - all_data[col].values
            history_features2_1.append(colname1)
    
    # 三阶
    elif 'and' in col:        
        # 最近一天的竞争胜率 * 当天竞争数目
        if 'rate' in col:
            colname1 = col+'_imps'
            colname2 = 'aid_cnts'
            all_data[colname1] = all_data[col].values * all_data[colname2]
            history_features2_2.append(colname1)       
        # 当天竞争数目 - 最近一天的正样本   
        elif 'sums' in col:
            colname1 = col+'_negs'
            colname2 = 'aid_cnts'
            all_data[colname1] = all_data[colname2] - all_data[col].values
            history_features2_2.append(colname1)
        # 当天竞争数目 - 最近一天的负样本
        elif 'negs' in col:
            colname1 = col+'_sums'
            colname2 = 'aid_cnts'
            all_data[colname1] = all_data[colname2] - all_data[col].values
            history_features2_2.append(colname1)
        

# 其余天、历史全部、前5天(sums/cnts,rate)
add_columns  = []
add_features = []
print('其余天竟胜率（sums/cnts） 填充 当天竟胜率...')
for col in ['aid','goods_id','account_id']:
    colname = col+'_else1_rate'
    add_columns.append(colname)
    result = pd.DataFrame()
    for day in range(10,25):
        tmp  = all_data.loc[(all_data['day']!=day)&(all_data['day']!=24)].groupby([col], as_index=False)[col+'_cnts'].agg({
            col+'_cnts'      : 'sum'
            })
        tmp1 = all_data.loc[(all_data['day']!=day)&(all_data['day']!=24)].groupby([col], as_index=False)[col+'_sums'].agg({
            col+'_sums'      : 'sum'
            })
        tmp[colname] = tmp1[col+'_sums'].values / tmp[col+'_cnts'].values
        res = all_data.loc[all_data['day']==day,[col,'day']]
        res = res.drop_duplicates(subset=[col,'day'], keep='first')
        res = res.merge(tmp[[col,colname]], how='left', on=[col])
        result = result.append(res, ignore_index=True)
    all_data = all_data.merge(result, how='left', on=[col,'day'])
    # 预计曝光次数 = 竞争胜率 * 当天竞争数目
    all_data[colname+'_imp'] = all_data[colname] * all_data[col+'_cnts']
    add_features.append(colname+'_imp')

print('历史竟胜率（sums/cnts） 填充 当天竟胜率...')
for col in ['aid','goods_id','account_id']:
    colname = col+'_hist1_rate'
    add_columns.append(colname)
    result = pd.DataFrame()
    for day in range(10,25):
        tmp  = all_data.loc[(all_data['day']<day)].groupby([col], as_index=False)[col+'_cnts'].agg({
            col+'_cnts'      : 'sum'
            })
        tmp1 = all_data.loc[(all_data['day']<day)].groupby([col], as_index=False)[col+'_sums'].agg({
            col+'_sums'      : 'sum'
            })
        tmp[colname] = tmp1[col+'_sums'].values / tmp[col+'_cnts'].values
        res = all_data.loc[all_data['day']==day,[col,'day']]
        res = res.drop_duplicates(subset=[col,'day'], keep='first')
        res = res.merge(tmp[[col,colname]], how='left', on=[col])
        result = result.append(res, ignore_index=True)
    all_data = all_data.merge(result, how='left', on=[col,'day'])
    # 预计曝光次数 = 竞争胜率 * 当天竞争数目
    all_data[colname+'_imp'] = all_data[colname] * all_data[col+'_cnts']
    add_features.append(colname+'_imp')

print('最近5天竟胜率（sums/cnts） 填充 当天竟胜率...')
for col in ['aid','goods_id','account_id']:
    colname = col+'_last1_rate'
    add_columns.append(colname)
    result = pd.DataFrame()
    for day in range(10,25):
        tmp  = all_data.loc[(all_data['day']<day)&(all_data['day']>=(day-5))].groupby([col], as_index=False)[col+'_cnts'].agg({
            col+'_cnts'      : 'sum'
            })
        tmp1 = all_data.loc[(all_data['day']<day)&(all_data['day']>=(day-5))].groupby([col], as_index=False)[col+'_sums'].agg({
            col+'_sums'      : 'sum'
            })
        tmp[colname] = tmp1[col+'_sums'].values / tmp[col+'_cnts'].values
        res = all_data.loc[all_data['day']==day,[col,'day']]
        res = res.drop_duplicates(subset=[col,'day'], keep='first')
        res = res.merge(tmp[[col,colname]], how='left', on=[col])
        result = result.append(res, ignore_index=True)
    all_data = all_data.merge(result, how='left', on=[col,'day'])
    # 预计曝光次数 = 竞争胜率 * 当天竞争数目
    all_data[colname+'_imp'] = all_data[colname] * all_data[col+'_cnts']
    add_features.append(colname+'_imp')

# 直接对rate构造均值和中位数，可以考虑添加day交叉特征rate
cross_columns = ['goods_id_aid_rate_day_mean', 'goods_id_aid_rate_day_median',
                 'account_id_aid_rate_day_mean', 'account_id_aid_rate_day_median',
                 'industry_id_aid_rate_day_mean', 'industry_id_aid_rate_day_median',
                 'aid_size_aid_rate_day_mean', 'aid_size_aid_rate_day_median']
print('其余天竟胜率（rate mean/median） 填充 当天竟胜率...')
for col in ['aid','goods_id','account_id'] + cross_columns:
    colname1 = col+'_else2_mean'
    colname2 = col+'_else2_median'
    add_columns.append(colname1)
    add_columns.append(colname2)
    result = pd.DataFrame()
    if col not in cross_columns:
        for day in range(10,25):
            tmp  = all_data.loc[(all_data['day']!=day)&(all_data['day']!=24)].groupby([col], as_index=False)[col+'_rate'].agg({
                colname1  : 'mean',
                colname2  : 'median'
                })
            res = all_data.loc[all_data['day']==day,[col,'day']]
            res = res.drop_duplicates(subset=[col,'day'], keep='first')
            res = res.merge(tmp[[col,colname1,colname2]], how='left', on=[col])
            result = result.append(res, ignore_index=True)
        all_data = all_data.merge(result, how='left', on=[col,'day'])  
        # 预计曝光次数 = 竞争胜率 * 当天竞争数目
        all_data[colname1+'_imp'] = all_data[colname1] * all_data[col+'_cnts']
        add_features.append(colname1+'_imp')
        all_data[colname2+'_imp'] = all_data[colname2] * all_data[col+'_cnts']
        add_features.append(colname2+'_imp')
    elif col in cross_columns:        
        if 'goods_id' in col:
            col1 = 'goods_id'
        elif 'account_id' in col:
            col1 = 'account_id'
        elif 'industry_id' in col:
            col1 = 'industry_id'
        elif 'aid_size' in col:
            col1 = 'aid_size'  
        for day in range(10,25):
            tmp  = all_data.loc[(all_data['day']!=day)&(all_data['day']!=24)].groupby([col1], as_index=False)[col].agg({
                colname1  : 'mean',
                colname2  : 'median'
                })
            res = all_data.loc[all_data['day']==day,[col1,'day']]
            res = res.drop_duplicates(subset=[col1,'day'], keep='first')
            res = res.merge(tmp[[col1,colname1,colname2]], how='left', on=[col1])
            result = result.append(res, ignore_index=True)
        all_data = all_data.merge(result, how='left', on=[col1,'day'])
        # 预计曝光次数 = 竞争胜率 * 当天竞争数目
        all_data[colname1+'_imp'] = all_data[colname1] * all_data['aid_cnts']
        add_features.append(colname1+'_imp')
        all_data[colname2+'_imp'] = all_data[colname2] * all_data['aid_cnts']
        add_features.append(colname2+'_imp')
    
print('历史竟胜率均值（rate mean/median） 填充 当天竟胜率...')
for col in ['aid','goods_id','account_id']:
    colname1 = col+'_hist2_mean'
    colname2 = col+'_hist2_median'
    add_columns.append(colname1)
    add_columns.append(colname2)
    result = pd.DataFrame()
    if col not in cross_columns:
        for day in range(10,25):
            tmp  = all_data.loc[(all_data['day']<day)].groupby([col], as_index=False)[col+'_rate'].agg({
                colname1  : 'mean',
                colname2  : 'median'
                })
            res = all_data.loc[all_data['day']==day,[col,'day']]
            res = res.drop_duplicates(subset=[col,'day'], keep='first')
            res = res.merge(tmp[[col,colname1,colname2]], how='left', on=[col])
            result = result.append(res, ignore_index=True)
        all_data = all_data.merge(result, how='left', on=[col,'day'])
        # 预计曝光次数 = 竞争胜率 * 当天竞争数目
        all_data[colname1+'_imp'] = all_data[colname1] * all_data[col+'_cnts']
        add_features.append(colname1+'_imp')
        all_data[colname2+'_imp'] = all_data[colname2] * all_data[col+'_cnts']
        add_features.append(colname2+'_imp')       
    elif col in cross_columns:
        if 'goods_id' in col:
            col1 = 'goods_id'
        elif 'account_id' in col:
            col1 = 'account_id'
        elif 'industry_id' in col:
            col1 = 'industry_id'
        elif 'aid_size' in col:
            col1 = 'aid_size'
        for day in range(10,25):
            tmp  = all_data.loc[(all_data['day']<day)].groupby([col1], as_index=False)[col].agg({
                colname1  : 'mean',
                colname2  : 'median'
                })
            res = all_data.loc[all_data['day']==day,[col1,'day']]
            res = res.drop_duplicates(subset=[col1,'day'], keep='first')
            res = res.merge(tmp[[col1,colname1,colname2]], how='left', on=[col1])
            result = result.append(res, ignore_index=True)
        all_data = all_data.merge(result, how='left', on=[col1,'day'])      
        # 预计曝光次数 = 竞争胜率 * 当天竞争数目
        all_data[colname1+'_imp'] = all_data[colname1] * all_data['aid_cnts']
        add_features.append(colname1+'_imp')
        all_data[colname2+'_imp'] = all_data[colname2] * all_data['aid_cnts']
        add_features.append(colname2+'_imp')

print('最近5天竟胜率均值（rate mean/median） 填充 当天竟胜率...')
for col in ['aid','goods_id','account_id']:
    colname1 = col+'_last2_mean'
    colname2 = col+'_last2_median'
    add_columns.append(colname1)
    add_columns.append(colname2)
    result = pd.DataFrame()
    if col not in cross_columns:
        for day in range(10,25):
            tmp  = all_data.loc[(all_data['day']<day)&(all_data['day']>=(day-5))].groupby([col], as_index=False)[col+'_rate'].agg({
                colname1  : 'mean',
                colname2  : 'median'
                })
            res = all_data.loc[all_data['day']==day,[col,'day']]
            res = res.drop_duplicates(subset=[col,'day'], keep='first')
            res = res.merge(tmp[[col,colname1,colname2]], how='left', on=[col])
            result = result.append(res, ignore_index=True)
        all_data = all_data.merge(result, how='left', on=[col,'day'])
        # 预计曝光次数 = 竞争胜率 * 当天竞争数目
        all_data[colname1+'_imp'] = all_data[colname1] * all_data[col+'_cnts']
        add_features.append(colname1+'_imp')
        all_data[colname2+'_imp'] = all_data[colname2] * all_data[col+'_cnts']
        add_features.append(colname2+'_imp')
    elif col in cross_columns:
        if 'goods_id' in col:
            col1 = 'goods_id'
        elif 'account_id' in col:
            col1 = 'account_id'
        elif 'industry_id' in col:
            col1 = 'industry_id'
        elif 'aid_size' in col:
            col1 = 'aid_size'
        for day in range(10,25):
            tmp  = all_data.loc[(all_data['day']<day)&(all_data['day']>=(day-5))].groupby([col1], as_index=False)[col].agg({
                colname1  : 'mean',
                colname2  : 'median'
                })
            res = all_data.loc[all_data['day']==day,[col1,'day']]
            res = res.drop_duplicates(subset=[col1,'day'], keep='first')
            res = res.merge(tmp[[col1,colname1,colname2]], how='left', on=[col1])
            result = result.append(res, ignore_index=True)
        all_data = all_data.merge(result, how='left', on=[col1,'day'])
        # 预计曝光次数 = 竞争胜率 * 当天竞争数目
        all_data[colname1+'_imp'] = all_data[colname1] * all_data['aid_cnts']
        add_features.append(colname1+'_imp')
        all_data[colname2+'_imp'] = all_data[colname2] * all_data['aid_cnts']
        add_features.append(colname2+'_imp')
            
# add 当天请求数目
add_features = add_features + ['aid_cnts','goods_id_cnts','account_id_cnts']

# 删除10 11  考虑保留所有样本
all_data = all_data.loc[all_data['day']>11]
print(all_data.shape)
gc.collect()


# cnts转换
train = all_data[all_data.day<=23]
train['logcnts'] = np.log(train['aid_cnts'].values + 1)
train_logcnts = train['logcnts'].values
test  = all_data[all_data.day==24]
test['logcnts'] = np.log(test['aid_cnts'].values + 1)
test_logcnts    = test['logcnts'].values
all_data = pd.concat([train, test], axis=0, ignore_index=True)


def get_train_data(df_):
    
    df = df_.copy()
    
    categorical_features = ['aid_size','goods_type','goods_id','industry_id','account_id']
    print('categorical_features:', categorical_features)
    
    numerical_features   = ['day'] + \
                            history_features1_1 + history_features1_2 + \
                            history_features2_1 + history_features2_2 + \
                            add_features + add_columns + \
                            kfold_features1 + kfold_features2
    print('numerical_features:',numerical_features)
    
    mutil_features       = []
    print('mutil_features:',mutil_features)
    
    print('transform data type...')
    for f in numerical_features:
        df[f] = df[f].astype('float')
        
    train   = df[df.day<=23]
    train   = train.reset_index()
    del train['index']
    train_y = train.pop('label')
    test    = df[df.day==24]
    test    = test.drop('label',axis=1)
    print('train:',train.shape)
    print('test: ',test.shape)
    
    train_x = train[numerical_features + categorical_features]
    test_x  = test[ numerical_features + categorical_features]
    print('train_x:',train_x.shape)
    print('test_x: ',test_x.shape)
    
    return df, train_x, test_x, train_y, test

data, train_x, test_x, train_y, test = get_train_data(all_data)


lgb_params = {'num_leaves': 2**7-1,
              'min_data_in_leaf': 25, 
              'objective':'regression_l2',
              'max_depth': -1,
              'learning_rate': 0.1,
              'min_child_samples': 20,
              'boosting': 'gbdt',
              'feature_fraction': 0.6,
              'bagging_fraction': 0.9,
              'bagging_seed': 11,
              'metric': 'mae',
              'seed':1024,
              'lambda_l1': 0.2}


def train_model(X, X_test, y, train_logcnts, test_logcnts, params, folds, model_type='lgb', label_type='bid'):

    if label_type == 'bid':
        y = np.log(y + 1) / train_logcnts
    elif label_type == 'nobid':
        y = np.log(y + 1)
    
    oof = np.zeros(X.shape[0])
    predictions = np.zeros(X_test.shape[0])
    scores = []
    models = []
    for fold_n, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        
        if model_type == 'lgb':
            trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
            val_data = lgb.Dataset(X[val_idx], y[val_idx])
            clf = lgb.train(params,
                            trn_data,
                            num_boost_round=3000,
                            valid_sets=[trn_data,val_data],
                            valid_names=['train','valid'],
                            early_stopping_rounds=100,
                            verbose_eval=500,
                            )
            oof[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)
            tmp = clf.predict(X_test, num_iteration=clf.best_iteration)
            
            if label_type == 'bid':
                predictions += (np.e**(tmp*test_logcnts) - 1) / folds.n_splits
            elif label_type == 'nobid':
                predictions += (np.e**tmp - 1) / folds.n_splits
        
        if label_type == 'bid':
            p = np.e**(oof[val_idx]*train_logcnts[val_idx]) - 1
            t = np.e**(  y[val_idx]*train_logcnts[val_idx]) - 1
        elif label_type == 'nobid':
            p = np.e**oof[val_idx] - 1
            t = np.e**y[val_idx]   - 1
        
        s = abs(p- t) / ((p + t) * 2)
       
        scores.append(s.mean())
        models.append(clf)       
    
    if label_type == 'bid':
        oof = np.e**(oof*train_logcnts) - 1
    elif label_type == 'nobid':
        oof = np.e**oof - 1
        
    print(np.mean(scores), np.std(scores), scores)
    
    return oof, predictions, scores, models


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
oof, predictions, scores, models  = train_model(train_x.values ,test_x.values, train_y, train_logcnts, test_logcnts, \
                                                lgb_params, folds=folds, model_type='lgb', label_type='nobid')


test['pred'] = predictions
test['pred'] = test['pred'].apply(lambda x: round(x,4))
test['sample_id'] = test['sample_id'].astype(int)
test.loc[test.pred<0, 'pred'] = 0
test[['sample_id','pred']].to_csv('../submission/wh_lgb.csv', index=False, header=None)