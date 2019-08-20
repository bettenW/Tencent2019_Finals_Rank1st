import os
import pandas as pd
import numpy as np
import random
import gc
from tqdm import tqdm
np.random.seed(2019)
random.seed(2019)


def construct_last_day_data():
    #构造21号label
    file='../data/track_log_20190421.out'
    log_df=pd.read_csv(file, sep='\t',names=['request_id','request_timestamp','uid','position','aid_info']).sort_values(by='request_timestamp')
    
    ids=set()
    for item in tqdm(log_df[['request_id','request_timestamp','uid','position','aid_info']].values,total=len(log_df)):
        for it in item[-1].split(';'):
            it=it.split(',')
            ids.add(int(it[0]))

    op_df=pd.read_csv('../data/final_map_bid_opt.out', sep='\t',names=['aid','op_time','op_type','objective','bid_type','bid']).sort_values(by=['aid','op_time'])
    op_df['day']=op_df['op_time']//1000000%100
    op_df=op_df.drop_duplicates(['aid','day'],keep='last')
    aids=set(op_df[op_df['day']<=20]['aid'])-set(op_df[op_df['day']==21]['aid'])
    aids=aids&ids
    op_df=op_df[op_df['day']<=20]
    op_df=op_df.drop_duplicates('aid',keep='last')
    op_df['is']=op_df['aid'].apply(lambda x: x in aids)
    op_df=op_df[op_df['is']==True]
    op_df=op_df[['aid','objective','bid_type','bid']]

    imp_dic={}
    for key in aids:
        imp_dic[key]=0
    for item in tqdm(log_df['aid_info'].values,total=len(log_df)):
        for its in item.split(';'):
            it=its.split(',')
            if int(it[0]) in aids:
                if int(it[-1])==1:
                    imp_dic[int(it[0])]+=1
    op_df['imp']=op_df['aid'].apply(lambda x: imp_dic[x])
    op_df=op_df[op_df['imp']!=0]
    op_df=op_df.sample(frac=0.3)
    aids=set(op_df['aid'])

    items=[]
    for item in tqdm(log_df['aid_info'].values,total=len(log_df)):
        temp=[]
        for its in item.split(';'):
            it=its.split(',')
            if int(it[0]) not in aids:
                temp.append(its)
        if len(temp)!=0:
            temp=';'.join(temp)
            items.append(temp)
        else:
            items.append('-1')
    log_df['aid_info']=items
    log_df=log_df[log_df['aid_info']!='-1']


    items=[]
    for item in tqdm(log_df['aid_info'].values,total=len(log_df)):
        temp=[]
        find=False
        for its in item.split(';'):
            it=its.split(',')
            if int(it[-2])==0 and find is False:
                it[-1]='1'
                temp.append(','.join(it))
                find=True
            else:
                it[-1]='0'
                temp.append(','.join(it))
        items.append(';'.join(temp))
    log_df['aid_info']=items   
    log_df.to_pickle('preprocess_data/Atest_log_dev.pkl')
    
    #构造23号的label
    log_df=log_df=pd.read_csv('../data/test_tracklog_20190423.last.out', sep='\t',names=['request_id','request_timestamp','uid','position','aid_info']).sort_values(by='request_timestamp')
    log_df['day']=23
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
    log_df.to_pickle('preprocess_data/Atest_log_test.pkl')

    
def construct_dev_data():
    #构造验证集
    file='../data/track_log_20190422.out'
    log_df=pd.read_csv(file, sep='\t',names=['request_id','request_timestamp','uid','position','aid_info']).sort_values(by='request_timestamp')
    ids=set()
    for item in tqdm(log_df[['request_id','request_timestamp','uid','position','aid_info']].values,total=len(log_df)):
        for it in item[-1].split(';'):
            it=it.split(',')
            ids.add(int(it[0]))

    op_df=pd.read_csv('../data/final_map_bid_opt.out', sep='\t',names=['aid','op_time','op_type','objective','bid_type','bid']).sort_values(by=['aid','op_time'])
    op_df['day']=op_df['op_time']//1000000%100
    op_df=op_df.drop_duplicates(['aid','day'],keep='last')
    aids=set(op_df[op_df['day']<=21]['aid'])-set(op_df[op_df['day']==22]['aid'])
    aids=aids&ids
    op_df=op_df[op_df['day']<=21]
    op_df=op_df.drop_duplicates('aid',keep='last')
    op_df['is']=op_df['aid'].apply(lambda x: x in aids)
    op_df=op_df[op_df['is']==True]
    op_df=op_df[['aid','objective','bid_type','bid']]

    ##################################################################################################
    imp_dic={}
    imp_request={}
    for key in aids:
        imp_dic[key]=0
        imp_request[key]=0
    items=[]
    for item in tqdm(log_df['aid_info'].values,total=len(log_df)):
        temp=[]
        for its in item.split(';'):
            it=its.split(',')
            if int(it[0]) in aids:
                if int(it[-1])==1:
                    imp_dic[int(it[0])]+=1
                imp_request[int(it[0])]+=1
            else:
                temp.append(str(it[0]))
        if len(temp)!=0:
            temp=';'.join(temp)
            items.append(temp)
        else:
            items.append('-1')
    log_df['aid_info']=items
    log_df=log_df[log_df['aid_info']!='-1']
    log_df['day']=22
    op_df['imp']=op_df['aid'].apply(lambda x: imp_dic[x])
    op_df=op_df[op_df['imp']!=0]
    op_df=op_df.sample(frac=0.3)
    requests=[]
    for key in imp_request:
        requests.append([key,imp_request[key]])
    request_df=pd.DataFrame(requests)
    request_df.columns=['aid','request_cont']

    ##################################################################################################
    keys=list(op_df)
    keys.remove('bid')
    keys+=['bid']
    items=[]
    for item in op_df[keys].values:
        item=list(item)
        items.append(item+[1])
        for i in range(10):
            while True:
                t=random.randint(0,2*item[-1])
                if t!=item[-1]:
                    items.append(item[:-1]+[t,0])
                    break
                else:
                    continue
    op_df=pd.DataFrame(items)
    op_df.columns=keys+['gold']
    del items
    gc.collect()
    ##################################################################################################
    log_df['day']=22
    request_df['day']=22
    op_df['day']=22
    log_df.to_pickle('preprocess_data/log_dev.pkl')
    request_df.to_pickle('preprocess_data/request_dev.pkl')
    op_df.to_pickle('preprocess_data/dev1.pkl')
    del log_df
    del request_df
    del op_df
    gc.collect()
    
def construct_test_data():
    #构造测试集
    #construct test dataset
    op_df=pd.read_csv('../data/Btest_sample_bid.out', sep='\t',names=['id','aid','objective','bid_type','bid'])
    request_df=pd.read_csv('../data/Btest_select_request_20190424.out', sep='\t',names=['aid','request_set'])
    request_df['request_cont']=request_df['request_set'].apply(lambda x:len(x.split('|')))
    del request_df['request_set']
    log_df=log_df=pd.read_csv('../data/BTest_tracklog_20190424.txt', sep='\t',names=['request_id','request_timestamp','uid','position','aid_info']).sort_values(by='request_timestamp')
    log_df['aid_info']=log_df['aid_info'].apply(lambda x: ';'.join([y.split(',')[0] for y in x.split(';')]))
    log_df['day']=24
    request_df['day']=24
    op_df['day']=24
    log_df.to_pickle('preprocess_data/log_test.pkl')
    request_df.to_pickle('preprocess_data/request_test.pkl')
    op_df.to_pickle('preprocess_data/test1.pkl')
    del log_df
    del request_df
    del op_df
    gc.collect()
    
def construct_other_data():
    df =pd.read_csv('../data/map_ad_static.out', sep='\t', 
                  names=['aid','create_timestamp','advertiser','good_id','good_type','ad_type_id','ad_size']).sort_values(by='create_timestamp')
    df=df.fillna(-1)
    for f in ['aid','create_timestamp','advertiser','good_id','good_type','ad_type_id']:
        items=[]
        for item in df[f].values:
            try:
                items.append(int(item))
            except:
                items.append(-1)
        df[f]=items
        df[f]=df[f].astype(int)
    for f in ['ad_size']:
        df[f]=df[f].apply(lambda x:' '.join([str(int(float(y))) for y in str(x).split(',')]))    

    df.to_pickle('preprocess_data/ad_static_feature.pkl')
    del df
    gc.collect()

def construct_train_data():
    #construct train dataset
    log_dfs=[]
    request_dfs=[]
    op_dfs=[]
    log_imp_dfs=[]
    imp_labels_all=[]
    for file,day in tqdm([('../data/track_log_201904{}.out'.format(day),day) for day in range(10,23)]+[('preprocess_data/Atest_log_dev.pkl',21),('preprocess_data/Atest_log_test.pkl',23)],total=len(range(10,23))+2):
        if file[-3:]=='out':
            log_df=pd.read_csv(file, sep='\t',names=['request_id','request_timestamp','uid','position','aid_info']).sort_values(by='request_timestamp')
        else:
            log_df=pd.read_pickle(file).sort_values(by='request_timestamp')
        ids=set()
        for item in log_df[['request_id','request_timestamp','uid','position','aid_info']].values:
            for it in item[-1].split(';'):
                it=it.split(',')
                ids.add(int(it[0]))

        op_df=pd.read_csv('../data/final_map_bid_opt.out', sep='\t',names=['aid','op_time','op_type','objective','bid_type','bid']).sort_values(by=['aid','op_time'])
        op_df['day']=op_df['op_time']//1000000%100
        op_df=op_df.drop_duplicates(['aid','day'],keep='last')
        op_df=op_df[op_df['day']<=day]
        op_df=op_df.drop_duplicates('aid',keep='last')
        op_df=op_df[['aid','objective','bid_type','bid']]


        imp_labels={}
        imp_dic={}
        imp_request={}
        for key in set(op_df['aid']):
            imp_dic[key]=0
            imp_request[key]=0



        for item in log_df[['aid_info','request_id','position','request_timestamp','uid']].values:
            vals=item[1:]
            key=','.join([str(item[1]),str(item[2])])
            item=item[0]
            temp=[]
            for its in item.split(';'):
                it=its.split(',')
                if (int(it[0]),day) not in imp_labels:
                    imp_labels[(int(it[0]),day)]=[0,0]
                imp_labels[(int(it[0]),day)][int(it[-1])]+=1
                if int(it[0]) not in imp_dic:
                    imp_dic[int(it[0])]=0
                if int(it[0]) not in imp_request:
                    imp_request[int(it[0])]=0
          
                imp_request[int(it[0])]+=1
                if int(it[-1])==1:
                    imp_dic[int(it[0])]+=1



        items=[]
        for key in imp_dic:
            items.append([key,imp_dic[key],imp_request[key]])

        temp=pd.DataFrame(items)
        temp.columns=['aid','imp','request_cont']
        op_df=temp.merge(op_df,on='aid',how='left')
        op_df=op_df[(op_df['imp']!=0)|(op_df['request_cont']>=50)]
        op_df['day']=day
        aids=set(op_df['aid'].values)
        items=[]
        for item in log_df[['aid_info','request_id','position','request_timestamp','uid']].values:
            vals=item[1:]
            key=','.join([str(item[1]),str(item[2])])
            item=item[0]
            temp=[]
            for its in item.split(';'):
                it=its.split(',')
                if int(it[0]) not in aids:
                    temp.append(str(it[0]))
            if len(temp)!=0:
                temp=';'.join(temp)
                items.append(temp)
            else:
                items.append('-1')


        log_df['aid_info']=items
        log_df=log_df[log_df['aid_info']!='-1']
        log_df['day']=day
        imp_labels_all.append(imp_labels)
        

        requests=[]
        for key in imp_request:
            if key in aids:
                requests.append([key,imp_request[key]])
        request_df=pd.DataFrame(requests)
        request_df.columns=['aid','request_cont']
        request_df['day']=day

        ##################################################################################################
        log_dfs.append(log_df)
        request_dfs.append(request_df)
        op_dfs.append(op_df)







    request_df=pd.concat(request_dfs[:-2]+[request_dfs[-1]],0)
    request_df.to_pickle('preprocess_data/request_train.pkl')
    op_df=pd.concat(op_dfs[:-2]+[op_dfs[-1]],0)
    op_df.to_pickle('preprocess_data/train1.pkl')
    print(request_df.shape,op_df.shape, op_df['imp'].mean())
    del request_df
    del op_df
    

    request_df=pd.concat(request_dfs[:-4]+[request_dfs[-2]],0)
    request_df.to_pickle('preprocess_data/request_train_dev.pkl')
    op_df=pd.concat(op_dfs[:-4]+[op_dfs[-2]],0)
    op_df.to_pickle('preprocess_data/train_dev1.pkl')
    print(request_df.shape,op_df.shape, op_df['imp'].mean())
    del request_df
    del op_df
    del log_dfs
    del op_dfs
    del request_dfs
    gc.collect()
    items=[]
    for imp_labels in imp_labels_all[:-2]+[imp_labels_all[-1]]:
        for key in imp_labels:
            items.append(list(key)+imp_labels[key])
    log_label=pd.DataFrame(items)
    log_label.columns=['aid','day','label_0','label_1']
    log_label.to_pickle('preprocess_data/log_label_test.pkl')
    items=[]
    for imp_labels in imp_labels_all[:-4]+[imp_labels_all[-2]]:
        for key in imp_labels:
            items.append(list(key)+imp_labels[key])
    log_label=pd.DataFrame(items)
    log_label.columns=['aid','day','label_0','label_1']
    log_label.to_pickle('preprocess_data/log_label_dev.pkl')
    
def combine_all():
    for flag in ['dev','test']:
        if flag=='dev':
            request=pd.read_pickle('preprocess_data/request_train_dev.pkl').append(pd.read_pickle('preprocess_data/request_dev.pkl')).reset_index()
            train=pd.read_pickle('preprocess_data/train_dev1.pkl').reset_index()
            test=pd.read_pickle('preprocess_data/dev1.pkl').reset_index()


        else:
            request=pd.read_pickle('preprocess_data/request_train.pkl').append(pd.read_pickle('preprocess_data/request_test.pkl')).reset_index()
            train=pd.read_pickle('preprocess_data/train1.pkl').reset_index()
            test=pd.read_pickle('preprocess_data/test1.pkl').reset_index()

            op_df=pd.read_csv('../data/test_sample_bid.out', sep='\t',names=['id','aid','objective','bid_type','bid'])
            request_df=pd.read_csv('../data/final_select_test_request.out', sep='\t',names=['aid','request_set'])
            request_df['day']=23
            op_df['day']=23
            
            op_df['imp']=pd.read_csv('../submission/A.csv',header=None)[1]
            op_df=op_df.drop_duplicates('aid')
            train=train.append(op_df)
            dic={}
            for item in request_df[['aid','day','request_set']].values:
                dic[(item[0],item[1])]=len(item[2].split('|'))
            op_df['request_cont']=op_df[['aid','day']].apply(lambda x:dic[tuple(x)],axis=1)   
            op_df['label_1']=op_df['imp']
            op_df['label_0']=op_df['request_cont']-op_df['label_1']
            request_df['request_cont']=request_df['request_set'].apply(lambda x:len(x.split('|')))
            del request_df['request_set']            
            request=request.append(request_df)
            log_label=pd.read_pickle("preprocess_data/log_label_test.pkl")
            log_label=log_label.append(op_df[['aid','day','label_0','label_1']])
            log_label.to_pickle('preprocess_data/log_label_test.pkl')


          
        ad_df =pd.read_pickle('preprocess_data/ad_static_feature.pkl')
        train=train.merge(ad_df,on=['aid'],how='left')
        train=train.fillna(-1)
        test=test.merge(ad_df,on=['aid'],how='left')
        test=test.fillna(-1)


        print(request.shape,train.shape,test.shape) 
        if flag=='dev':                     
            request.to_pickle('preprocess_data/aid_request_dev.pkl')
            train.to_pickle('preprocess_data/train_dev.pkl')
            test.to_pickle('preprocess_data/dev.pkl')
        else:                    
            request.to_pickle('preprocess_data/aid_request_test.pkl')
            train.to_pickle('preprocess_data/train.pkl')
            test.to_pickle('preprocess_data/test.pkl') 
        del request
        del train
        del test
        gc.collect()

    

    
    
print("construct_last_day_data")    
construct_last_day_data()   
print("construct_dev_data")
construct_dev_data()
print("construct_test_data")
construct_test_data()
print("construct_train_data")
construct_train_data()
print("construct_other_data")
construct_other_data()
print("combine all data")
combine_all()
