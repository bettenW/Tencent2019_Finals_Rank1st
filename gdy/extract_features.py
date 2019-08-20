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




def kfold_static(train_df,test_df,f,label):
    print("K-fold static:",f+'_'+label)
    #K-fold positive and negative num
    avg_rate=train_df[label].mean()
    num=len(train_df)//5
    index=[0 for i in range(num)]+[1 for i in range(num)]+[2 for i in range(num)]+[3 for i in range(num)]+[4 for i in range(len(train_df)-4*num)]
    random.shuffle(index)
    train_df['index']=index

        

    dic=[{} for i in range(5)]
    dic_all={}
    
    for item in train_df[['index',f,label]].values:
        try:
            dic[int(item[0])][item[1]].append(item[2])
        except:
            dic[int(item[0])][item[1]]=[]
            dic[int(item[0])][item[1]].append(item[2])
    print("static done!")
                
    mean=[]
    median=[]
    std=[]
    Min=[]
    Max=[]
    cache={}
    for item in train_df[['index',f]].values:
        if tuple(item) not in cache:
            temp=[]
            for i in range(5):
                 if i!=item[0]:
                    try:
                        temp+=dic[i][item[1]]
                    except:
                        pass
            if len(temp)==0:
                cache[tuple(item)]=[-1]*5
            else:
                cache[tuple(item)]=[np.mean(temp),np.median(temp),np.std(temp),np.min(temp),np.max(temp)]
        temp=cache[tuple(item)]
        mean.append(temp[0])
        median.append(temp[1])
        std.append(temp[2])
        Min.append(temp[3])
        Max.append(temp[4])              

          
    del cache        
    train_df[f+'_'+label+'_mean']=mean
    train_df[f+'_'+label+'_median']=median
    train_df[f+'_'+label+'_std']=std
    train_df[f+'_'+label+'_min']=Min
    train_df[f+'_'+label+'_max']=Max   
    print("train done!")
    #for test
    mean=[]
    median=[]
    std=[]
    Min=[]
    Max=[]
    cache={}
    for uid in test_df[f].values:
        if uid not in cache:
            temp=[]
            for i in range(5):
                try:
                    temp+=dic[i][uid]
                except:
                    pass
            if len(temp)==0:
                cache[uid]=[-1]*5
            else:
                cache[uid]=[np.mean(temp),np.median(temp),np.std(temp),np.min(temp),np.max(temp)]
        temp=cache[uid]
        mean.append(temp[0])
        median.append(temp[1])
        std.append(temp[2])
        Min.append(temp[3])
        Max.append(temp[4])           
        
    test_df[f+'_'+label+'_mean']=mean
    test_df[f+'_'+label+'_median']=median
    test_df[f+'_'+label+'_std']=std
    test_df[f+'_'+label+'_min']=Min
    test_df[f+'_'+label+'_max']=Max   
    print("test done!")
    del train_df['index']
    print(f+'_'+label+'_mean')
    print(f+'_'+label+'_median')
    print(f+'_'+label+'_std')
    print(f+'_'+label+'_min')
    print(f+'_'+label+'_max')
    print('avg of mean',np.mean(train_df[f+'_'+label+'_mean']),np.mean(test_df[f+'_'+label+'_mean']))
    print('avg of median',np.mean(train_df[f+'_'+label+'_median']),np.mean(test_df[f+'_'+label+'_median']))
    print('avg of std',np.mean(train_df[f+'_'+label+'_std']),np.mean(test_df[f+'_'+label+'_std']))
    print('avg of min',np.mean(train_df[f+'_'+label+'_min']),np.mean(test_df[f+'_'+label+'_min']))
    print('avg of max',np.mean(train_df[f+'_'+label+'_max']),np.mean(test_df[f+'_'+label+'_max']))
    
    
def request_cont(train_df,test_df,flag):
    print("request_cont")
    request=pd.read_pickle('preprocess_data/aid_request_{}.pkl'.format(flag))
    dic={}
    for item in request[['aid','day','request_cont']].values:
        dic[(item[0],item[1])]=int(item[2])
    train_df['request_cont']=train_df[['aid','day']].apply(lambda x:dic[tuple(x)],axis=1)
    test_df['request_cont']=test_df[['aid','day']].apply(lambda x:dic[tuple(x)],axis=1)
    
def kfold_static_log(train_df,test_df,f,flag):
    label='label'
    log=pd.read_pickle('preprocess_data/log_label_{}.pkl'.format(flag))
    if f!='aid':
        op_df=pd.read_csv('../data/final_map_bid_opt.out', sep='\t',names=['aid','op_time','op_type','objective','bid_type','bid']).sort_values(by=['aid','op_time'])
        op_df['day']=op_df['op_time']//1000000%100
        op_dfs=[]
        for day in range(10,25): 
            op_df1=op_df[op_df['day']<=day]
            op_df1=op_df1.drop_duplicates('aid',keep='last')
            op_df1['day']=day
            op_dfs.append(op_df1)
        op_df=pd.concat(op_dfs,0)
        ad_df =pd.read_pickle('preprocess_data/ad_static_feature.pkl')
        ad_df['good_id_advertiser']=ad_df['good_id']*1000000+ad_df['advertiser']
        log=log.merge(ad_df,on=['aid'],how='left')
        log=log.merge(op_df,on=['aid','day'],how='left')
    print("kfold_static_log:",f+'_'+label)
    #K-fold positive and negative num
    log[f]=log[f].fillna(-1).astype(int)

    dic=[{} for i in range(100)]
    dic_all={}
    for item in tqdm(log[['day',f,'label_0','label_1']].values,total=len(log)):
        try:
            dic[item[0]][item[1]][0]+=item[2]
            dic[item[0]][item[1]][1]+=item[3]
        except:
            dic[item[0]][item[1]]=[0,0]
            dic[item[0]][item[1]][0]+=item[2]
            dic[item[0]][item[1]][1]+=item[3]            
        try:
            dic_all[item[1]][0]+=item[2]
            dic_all[item[1]][1]+=item[3]
        except:
            dic_all[item[1]]=[0,0]
            dic_all[item[1]][0]+=item[2]
            dic_all[item[1]][1]+=item[3]
    print("static done!")                
    positive=[]
    negative=[]
    rate=[]
    for item in train_df[['day',f]].values:
        n,p=dic_all[item[1]]
        try:
            p-=dic[item[0]][item[1]][1]
            n-=dic[item[0]][item[1]][0] 
        except:
            pass
        if p==0 and n==0:
            positive.append(-1)
            negative.append(-1)
            rate.append(0)
        else:
            positive.append(p)
            negative.append(n)
            rate.append(p/(p+n)) 
    train_df[f+'_kfold_rate']=rate
    print("train done!")
    #for test
    positive=[]
    negative=[]
    rate=[]
    for uid in test_df[f].values:
        p=0
        n=0
        try:
            p=dic_all[uid][1]
            n=dic_all[uid][0]
        except:
            pass
        if p==0 and n==0:
            positive.append(-1)
            negative.append(-1)
            rate.append(0)
        else:
            positive.append(p)
            negative.append(n)
            rate.append(p/(p+n))
    test_df[f+'_kfold_rate']=rate  
    
    train_df[f+'_kfold_imp']=train_df[f+'_kfold_rate']*train_df['request_cont']
    test_df[f+'_kfold_imp']=test_df[f+'_kfold_rate']*test_df['request_cont']
    print("test done!")
    del log
    gc.collect()
    print(f+'_kfold_rate')
    print('avg of rate',np.mean(train_df[f+'_kfold_rate']),np.mean(test_df[f+'_kfold_rate']))
    print('avg of imp',np.mean(train_df[f+'_kfold_imp']),np.mean(test_df[f+'_kfold_imp']))

def kfold_static_log_history(train_df,test_df,f,flag):
    label='label'
    log=pd.read_pickle('preprocess_data/log_label_{}.pkl'.format(flag))
    if f!='aid':
        op_df=pd.read_csv('../data/final_map_bid_opt.out', sep='\t',names=['aid','op_time','op_type','objective','bid_type','bid']).sort_values(by=['aid','op_time'])
        op_df['day']=op_df['op_time']//1000000%100
        op_dfs=[]
        for day in range(10,25): 
            op_df1=op_df[op_df['day']<=day]
            op_df1=op_df1.drop_duplicates('aid',keep='last')
            op_df1['day']=day
            op_dfs.append(op_df1)
        op_df=pd.concat(op_dfs,0)
        ad_df =pd.read_pickle('preprocess_data/ad_static_feature.pkl')
        ad_df['good_id_advertiser']=ad_df['good_id']*1000000+ad_df['advertiser']
        log=log.merge(ad_df,on=['aid'],how='left')
        log=log.merge(op_df,on=['aid','day'],how='left')

    print("kfold_static_log_history:",f+'_'+label)
    #K-fold positive and negative num
    log[f]=log[f].fillna(-1).astype(int)

    dic=[{} for i in range(100)]
    dic_all={}
    for item in tqdm(log[['day',f,'label_0','label_1']].values,total=len(log)):
        try:
            dic[item[0]][item[1]][0]+=item[2]
            dic[item[0]][item[1]][1]+=item[3]
        except:
            dic[item[0]][item[1]]=[0,0]
            dic[item[0]][item[1]][0]+=item[2]
            dic[item[0]][item[1]][1]+=item[3]            
        try:
            dic_all[item[1]][0]+=item[2]
            dic_all[item[1]][1]+=item[3]
        except:
            dic_all[item[1]]=[0,0]
            dic_all[item[1]][0]+=item[2]
            dic_all[item[1]][1]+=item[3]
    print("static done!")        
    positive=[]
    negative=[]
    rate=[]
    for item in train_df[['day',f]].values:
        n,p=dic_all[item[1]]
        for day in range(9,26):
            if day>=item[0]:
                try:
                    p-=dic[day][item[1]][1]
                    n-=dic[day][item[1]][0] 
                except:
                    pass
        if p==0 and n==0:
            positive.append(-1)
            negative.append(-1)
            rate.append(0)
        else:
            positive.append(p)
            negative.append(n)
            rate.append(p/(p+n)) 
    train_df[f+'_kfold_history_rate']=rate    
    print("train done!")
    
    #for test
    positive=[]
    negative=[]
    rate=[]
    for uid in test_df[f].values:
        p=0
        n=0
        try:
            p=dic_all[uid][1]
            n=dic_all[uid][0]
        except:
            pass
        if p==0 and n==0:
            positive.append(-1)
            negative.append(-1)
            rate.append(0)
        else:
            positive.append(p)
            negative.append(n)
            rate.append(p/(p+n))
            
    test_df[f+'_kfold_history_rate']=rate

    train_df[f+'_kfold_history_imp']=train_df[f+'_kfold_history_rate']*train_df['request_cont']
    test_df[f+'_kfold_history_imp']=test_df[f+'_kfold_history_rate']*test_df['request_cont']
    print("test done!")
    del log
    gc.collect()
    print(f+'_kfold_rate')
    print('avg of rate',np.mean(train_df[f+'_kfold_history_rate']),np.mean(test_df[f+'_kfold_history_rate']))
    print('avg of imp',np.mean(train_df[f+'_kfold_history_imp']),np.mean(test_df[f+'_kfold_history_imp']))

def history_static_log(train_df,test_df,f,flag):
    label='label'
    log=pd.read_pickle('preprocess_data/log_label_{}.pkl'.format(flag))
    if f!='aid':
        op_df=pd.read_csv('../data/final_map_bid_opt.out', sep='\t',names=['aid','op_time','op_type','objective','bid_type','bid']).sort_values(by=['aid','op_time'])
        op_df['day']=op_df['op_time']//1000000%100
        op_dfs=[]
        for day in range(10,25): 
            op_df1=op_df[op_df['day']<=day]
            op_df1=op_df1.drop_duplicates('aid',keep='last')
            op_df1['day']=day
            op_dfs.append(op_df1)
        op_df=pd.concat(op_dfs,0)
        ad_df =pd.read_pickle('preprocess_data/ad_static_feature.pkl')
        ad_df['good_id_advertiser']=ad_df['good_id']*1000000+ad_df['advertiser']
        log=log.merge(ad_df,on=['aid'],how='left')
        log=log.merge(op_df,on=['aid','day'],how='left')
    log['day']=log['day'].astype(int)
    log[f]=log[f].fillna(-1).astype(int)
    print("history_static_log:",f+'_'+label)
    #K-fold positive and negative num



    dic_all={}
    for item in tqdm(log[['day',f,'label_0','label_1']].values,total=len(log)):
        try:
            dic_all[(item[0],item[1])][0]+=item[2]
            dic_all[(item[0],item[1])][1]+=item[3]
        except:
            dic_all[(item[0],item[1])]=[0,0]
            dic_all[(item[0],item[1])][0]+=item[2]
            dic_all[(item[0],item[1])][1]+=item[3]
    print("static done!")              
    positive=[]
    negative=[]
    sequence=[]
    rate=[]
    for item in train_df[['day',f,'request_cont']].values:
        n=0
        p=0
        first=True
        temp=[]
        for day in range(int(item[0])-1,9,-1):
            if (day,item[1]) in dic_all:
                if first is True:
                    n+=dic_all[(day,item[1])][0]
                    p+=dic_all[(day,item[1])][1]
                    first=False
                k0,k1=dic_all[(day,item[1])]

        if p==0 and n==0:
            positive.append(-1)
            negative.append(-1)
            rate.append(0)
        else:
            positive.append(p)
            negative.append(n)
            rate.append(p/(p+n))
            
    train_df[f+'_'+'history'+'_positive_num']=positive
    train_df[f+'_'+'history'+'_negative_num']=negative
    train_df[f+'_'+'history'+'_rate']=rate

    
    print("train done!")
    #for test
    positive=[]
    negative=[]
    sequence=[]
    rate=[]
    log_rate=[]
    for item in test_df[['day',f,'request_cont']].values:
        n=0
        p=0
        first=True
        temp=[]
        for day in range(int(item[0])-1,9,-1):
            if (day,item[1]) in dic_all:
                if first is True:
                    n+=dic_all[(day,item[1])][0]
                    p+=dic_all[(day,item[1])][1]
                    first=False
                k0,k1=dic_all[(day,item[1])]

        sequence.append(' '.join(temp))
        if p==0 and n==0:
            positive.append(-1)
            negative.append(-1)
            rate.append(0)
            log_rate.append(0)
        else:
            positive.append(p)
            negative.append(n)
            rate.append(p/(p+n))
            log_rate.append(np.log(p+1)/np.log(p+n+1))
        
    test_df[f+'_'+'history'+'_positive_num']=positive
    test_df[f+'_'+'history'+'_negative_num']=negative  
    test_df[f+'_'+'history'+'_rate']=rate

    print("test done!")
    train_df[f+'_predict_imp']=train_df[f+'_'+'history'+'_rate']*train_df['request_cont']
    test_df[f+'_predict_imp']=test_df[f+'_'+'history'+'_rate']*test_df['request_cont']  
    del log
    gc.collect()
    print(f+'_'+label+'_positive_num')
    print(f+'_'+label+'_negative_num')
    print(f+'_'+label+'_rate')
    print('avg of positive num',np.mean(train_df[f+'_'+'history'+'_positive_num']),np.mean(test_df[f+'_'+'history'+'_positive_num']))
    print('avg of negative num',np.mean(train_df[f+'_'+'history'+'_negative_num']),np.mean(test_df[f+'_'+'history'+'_negative_num']))
    print('avg of rate',np.mean(train_df[f+'_'+'history'+'_rate']),np.mean(test_df[f+'_'+'history'+'_rate']))
    print('avg of predict imp',train_df[f+'_predict_imp'].mean(),test_df[f+'_predict_imp'].mean())



def history_static_log_diff(train_df,test_df,f,flag,diff_day):
    label='label'
    log=pd.read_pickle('preprocess_data/log_label_{}.pkl'.format(flag))
    if f!='aid':
        op_df=pd.read_csv('../data/final_map_bid_opt.out', sep='\t',names=['aid','op_time','op_type','objective','bid_type','bid']).sort_values(by=['aid','op_time'])
        op_df['day']=op_df['op_time']//1000000%100
        op_dfs=[]
        for day in range(10,26): 
            op_df1=op_df[op_df['day']<=day]
            op_df1=op_df1.drop_duplicates('aid',keep='last')
            op_df1['day']=day
            op_dfs.append(op_df1)
        op_df=pd.concat(op_dfs,0)
        ad_df =pd.read_pickle('preprocess_data/ad_static_feature.pkl')
        ad_df['good_id_advertiser']=ad_df['good_id']*1000000+ad_df['advertiser']
        log=log.merge(ad_df,on=['aid'],how='left')
        log=log.merge(op_df,on=['aid','day'],how='left')

    log['day']=log['day'].astype(int)
    log[f]=log[f].fillna(-1).astype(int)
    print("history_static_log_diff:",f+'_'+label,diff_day)
    #K-fold positive and negative num
    dic_all={}
    for item in tqdm(log[['day',f,'label_0','label_1']].values,total=len(log)):
        try:
            dic_all[(item[0],item[1])][0]+=item[2]
            dic_all[(item[0],item[1])][1]+=item[3]
        except:
            dic_all[(item[0],item[1])]=[0,0]
            dic_all[(item[0],item[1])][0]+=item[2]
            dic_all[(item[0],item[1])][1]+=item[3]
    print("static done!")
         
    positive=[]
    negative=[]
    rate=[]
    for item in train_df[['day',f,'request_cont']].values:
        n=0
        p=0
        first=True
        temp=[]
        for day in [item[0]-diff_day]:
            if (day,item[1]) in dic_all:
                if first is True:
                    n+=dic_all[(day,item[1])][0]
                    p+=dic_all[(day,item[1])][1]
                    first=False

        if p==0 and n==0:
            positive.append(-1)
            negative.append(-1)
            rate.append(0)
        else:
            positive.append(p)
            negative.append(n)
            rate.append(p/(p+n))  
            
    train_df[f+'_'+str(diff_day)+'_positive_num']=positive
    train_df[f+'_'+str(diff_day)+'_negative_num']=negative
    train_df[f+'_'+str(diff_day)+'_rate']=rate

    print("train done!")
    #for test
    positive=[]
    negative=[]
    rate=[]
    for item in test_df[['day',f,'request_cont']].values:
        n=0
        p=0
        first=True
        temp=[]
        for day in [item[0]-diff_day]:
            if (day,item[1]) in dic_all:
                if first is True:
                    n+=dic_all[(day,item[1])][0]
                    p+=dic_all[(day,item[1])][1]
                    first=False
        if p==0 and n==0:
            positive.append(-1)
            negative.append(-1)
            rate.append(0)
        else:
            positive.append(p)
            negative.append(n)
            rate.append(p/(p+n))
        
    test_df[f+'_'+str(diff_day)+'_positive_num']=positive
    test_df[f+'_'+str(diff_day)+'_negative_num']=negative  
    test_df[f+'_'+str(diff_day)+'_rate']=rate

    
    
    print("test done!")
    train_df[f+'_predict_imp_'+str(diff_day)]=train_df[f+'_'+str(diff_day)+'_rate']*train_df['request_cont']
    test_df[f+'_predict_imp_'+str(diff_day)]=test_df[f+'_'+str(diff_day)+'_rate']*test_df['request_cont']
   
    del log
    gc.collect()
    print(f+'_'+label+'_positive_num')
    print(f+'_'+label+'_negative_num')
    print(f+'_'+label+'_rate')
    print('avg of positive num',np.mean(train_df[f+'_'+str(diff_day)+'_positive_num']),np.mean(test_df[f+'_'+str(diff_day)+'_positive_num']))
    print('avg of negative num',np.mean(train_df[f+'_'+str(diff_day)+'_negative_num']),np.mean(test_df[f+'_'+str(diff_day)+'_negative_num']))
    print('avg of rate',np.mean(train_df[f+'_'+str(diff_day)+'_rate']),np.mean(test_df[f+'_'+str(diff_day)+'_rate']))
    print('avg of predict imp',train_df[f+'_predict_imp_'+str(diff_day)].mean(),test_df[f+'_predict_imp_'+str(diff_day)].mean())

    
    
    
    
from sklearn.model_selection import KFold
def get_kfold_features(data_df_, test_df_,flag):
    
    data_df = data_df_.copy()
    test_df = test_df_.copy()
    data_df['goods_id']=data_df['good_id']
    test_df['goods_id']=test_df['good_id']
    data_df['account_id']=data_df['advertiser']
    test_df['account_id']=test_df['advertiser']    
    data_df['industry_id']=data_df['ad_type_id']
    test_df['industry_id']=test_df['ad_type_id']   
    data_df['aid_size']=data_df['ad_size']
    test_df['aid_size']=test_df['ad_size']
    log=pd.read_pickle('preprocess_data/log_label_{}.pkl'.format(flag))
    ad_df =pd.read_pickle('preprocess_data/ad_static_feature.pkl')
    log=log.merge(ad_df,on=['aid'],how='left')
    log['account_id']=log['advertiser']
    log['industry_id']=log['ad_type_id']
    log['aid_size']=log['ad_size']
    log['goods_id']=log['good_id']
    ad_df['account_id']=ad_df['advertiser']
    ad_df['industry_id']=ad_df['ad_type_id']
    ad_df['aid_size']=ad_df['ad_size']
    ad_df['goods_id']=ad_df['good_id']    
    log['request_cont']=log['label_0']+log['label_1']
    for f in ['aid','goods_id','account_id']:
        
        result= log.groupby([f,'day'],as_index=False)['request_cont'].sum()
        result.columns=[f,'day',f+'_cnts']
        try:
            del data_df[f+'_cnts']
        except:
            pass
        data_df=data_df.merge(result,on=[f,'day'],how='left')
     
        result= log.groupby([f,'day'],as_index=False)['label_0'].sum()
        result.columns=[f,'day',f+'_negs']
        try:
            del data_df[f+'_negs']
        except:
            pass        
        data_df=data_df.merge(result,on=[f,'day'],how='left')
        
        result= log.groupby([f,'day'],as_index=False)['label_1'].sum()
        result.columns=[f,'day',f+'_sums']
        try:
            del data_df[f+'_sums']
        except:
            pass       
        data_df=data_df.merge(result,on=[f,'day'],how='left')
        
        data_df[f+'_rate']=data_df[f+'_sums']/(data_df[f+'_sums']+data_df[f+'_negs'])
        ######################################################################################
        if f=='aid':
            test_df[f+'_cnts']=test_df['request_cont']
        else:
            convert={}
            for item in ad_df[['aid',f]].values:
                convert[int(item[0])]=int(item[1])
            dic={}
            cont=[0,0,0,0]
            test_log=pd.read_pickle('preprocess_data/log_{}.pkl'.format(flag))
            for item in test_log['aid_info'].values:
                for it in item.split(';'):
                    aid=int(it.split(',')[0])
                    if aid not in convert:
                        cont[0]+=1

                    else:
                        if convert[aid] in dic:
                            dic[convert[aid]]+=1
                        else:
                            dic[convert[aid]]=1
                        cont[1]+=1
            request=pd.read_pickle('preprocess_data/request_{}.pkl'.format(flag))

            for item in request[['aid','request_cont']].values:
                aid=int(item[0])
                length=int(item[1])
                if aid not in convert:
                    cont[2]+=1
                else:
                    if convert[aid] in dic:
                        dic[convert[aid]]+=length
                    else:
                        dic[convert[aid]]=length
                    cont[3]+=1
            print(cont)
            test_df[f+'_cnts']=test_df[f].apply(lambda x:dic[x])
    folds = KFold(n_splits=5, shuffle=True, random_state=2019)
    data_df['fold'] = None
    for fold_,(trn_idx,val_idx) in enumerate(folds.split(data_df,data_df)):
        data_df.loc[val_idx, 'fold'] = fold_
    kfold_features = []
    for feat in ['aid','goods_id','account_id','industry_id','aid_size']:

        nums_columns = ['aid_rate','goods_id_rate','account_id_rate',
                        'aid_cnts','goods_id_cnts','account_id_cnts',
                        'aid_sums','goods_id_sums','account_id_sums',
                        'aid_negs','goods_id_negs','account_id_negs']

        for f in nums_columns:
            colname1 = feat + '_' + f + '_kfold_mean'
            colname2 = feat + '_' + f + '_kfold_median'
            print(feat,f,' mean/median...')
            kfold_features.append(colname1)
            kfold_features.append(colname2)

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
                kfold_features.append(colname1+'_imps')
                kfold_features.append(colname2+'_imps')
                data_df[colname1+'_imps'] = data_df[colname1] * data_df[f[:-5]+'_cnts']
                data_df[colname2+'_imps'] = data_df[colname2] * data_df[f[:-5]+'_cnts'] 
                test_df[colname1+'_imps'] = test_df[colname1] * test_df[f[:-5]+'_cnts']
                test_df[colname2+'_imps'] = test_df[colname2] * test_df[f[:-5]+'_cnts']

            if 'sums' in colname1:
                kfold_features.append(colname1+'_negs')
                kfold_features.append(colname2+'_negs')
                data_df[colname1+'_negs'] = data_df[f[:-5]+'_cnts'].values - data_df[colname1].values
                data_df[colname2+'_negs'] = data_df[f[:-5]+'_cnts'].values - data_df[colname2].values
                test_df[colname1+'_negs'] = test_df[f[:-5]+'_cnts'].values - test_df[colname1].values
                test_df[colname2+'_negs'] = test_df[f[:-5]+'_cnts'].values - test_df[colname2].values

            if 'negs' in colname1:
                kfold_features.append(colname1+'_sums')
                kfold_features.append(colname2+'_sums')
                data_df[colname1+'_sums'] = data_df[f[:-5]+'_cnts'].values - data_df[colname1].values
                data_df[colname2+'_sums'] = data_df[f[:-5]+'_cnts'].values - data_df[colname2].values
                test_df[colname1+'_sums'] = test_df[f[:-5]+'_cnts'].values - test_df[colname1].values
                test_df[colname2+'_sums'] = test_df[f[:-5]+'_cnts'].values - test_df[colname2].values
             
    del data_df['fold']
    return data_df, test_df, kfold_features




def get_cross_day_features(train_df, test_df):
    columns = [ 'aid_predict_imp_2', 
               'aid_history_positive_num', 'aid_2_positive_num', 'aid_kfold_history_imp', 'aid_1_positive_num', 
               'aid_kfold_imp', 'aid_predict_imp_1', 'aid_predict_imp', 'aid_3_positive_num', 'aid_predict_imp_3', 
               'aid_4_positive_num', 'aid_predict_imp_4', ]
    cross_day_columns = []
    
    for feat in ['advertiser','good_id','good_type','ad_type_id','ad_size']:
        for f in columns:
            print(feat,f,'-'*10)
            data=train_df[[feat,'day',f]].append(test_df[[feat,'day',f]])
            tmp  = data.groupby([feat,'day'], as_index=False)[f].agg({
                    feat + '_' + f + '_day_mean'   : 'mean',
                    feat + '_' + f + '_day_median' : 'median',
                    })
            try:
                del train_df[feat + '_' + f + '_day_mean']
                del train_df[feat + '_' + f + '_day_median']
                del test_df[feat + '_' + f + '_day_mean']
                del test_df[feat + '_' + f + '_day_median']  
            except:
                pass
            train_df = train_df.merge(tmp, on=[feat,'day'], how='left')
            test_df = test_df.merge(tmp, on=[feat,'day'], how='left')
            print(train_df[feat + '_' + f + '_day_mean'].mean(),test_df[feat + '_' + f + '_day_mean'].mean())
            print(train_df[feat + '_' + f + '_day_median'].mean(),test_df[feat + '_' + f + '_day_median'].mean())
            if 'rate' in f:
                train_df[feat + '_' + f + '_day_mean_imp']=train_df[feat + '_' + f + '_day_mean']*train_df['request_cont']
                test_df[feat + '_' + f + '_day_mean_imp']=test_df[feat + '_' + f + '_day_mean']*test_df['request_cont']
                train_df[feat + '_' + f + '_day_median_imp']=train_df[feat + '_' + f + '_day_median']*train_df['request_cont']
                test_df[feat + '_' + f + '_day_median_imp']=test_df[feat + '_' + f + '_day_median']*test_df['request_cont']
                print(train_df[feat + '_' + f + '_day_mean_imp'].mean(),test_df[feat + '_' + f + '_day_mean_imp'].mean())
                print(train_df[feat + '_' + f + '_day_median_imp'].mean(),test_df[feat + '_' + f + '_day_median_imp'].mean())   
    return train_df,test_df

#extract features
for path1,path2,log_path,flag in [('preprocess_data/train_dev.pkl','preprocess_data/dev.pkl','preprocess_data/user_log_dev.pkl','dev'),('preprocess_data/train.pkl','preprocess_data/test.pkl','preprocess_data/user_log_test.pkl','test')]:
        print(path1,path2,log_path,flag)
        train_df=pd.read_pickle(path1)
        test_df=pd.read_pickle(path2)
        request_cont(train_df,test_df,flag)
        
        kfold_static_log(train_df,test_df,'aid',flag)
        kfold_static_log_history(train_df,test_df,'aid',flag)
        
        history_static_log(train_df,test_df,'good_id',flag)
        history_static_log(train_df,test_df,'aid',flag)
        history_static_log(train_df,test_df,'advertiser',flag)

        kfold_static(train_df,test_df,'aid','imp')
        kfold_static(train_df,test_df,'good_id','imp')
        kfold_static(train_df,test_df,'advertiser','imp')

        for i in range(1,5):
            history_static_log_diff(train_df,test_df,'aid',flag,i)
            history_static_log_diff(train_df,test_df,'good_id',flag,i)
            history_static_log_diff(train_df,test_df,'advertiser',flag,i)

        train_df, test_df, kfold_features=get_kfold_features(train_df, test_df,flag)


        train_df, test_df=get_cross_day_features(train_df, test_df)


        print(train_df.shape,test_df.shape)
        train_df.to_pickle(path1) 
        test_df.to_pickle(path2)  
        print("*"*80)
        print("save done!")
